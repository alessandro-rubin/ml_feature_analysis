"""Backend adapters, driven by fake clients.

Both modules keep their SDK import inside ``build_backend``, so the adapter
logic — which is where the wire-format bugs live — is fully testable with
neither ``openai`` nor ``anthropic`` installed.
"""

from __future__ import annotations

import json
import types

import pytest

from tessa.ai.backend_anthropic import AnthropicBackend
from tessa.ai.backend_openai import (
    LOCAL_TIMEOUT,
    PROVIDERS,
    OpenAICompatibleBackend,
    build_backend,
    recover_tool_calls,
)
from tessa.ai.loop import ToolCall


# ── OpenAI-compatible ────────────────────────────────────────────────────────


def _openai_response(content=None, tool_calls=(), finish="stop"):
    message = types.SimpleNamespace(content=content, tool_calls=list(tool_calls))
    choice = types.SimpleNamespace(message=message, finish_reason=finish)
    usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=3)
    return types.SimpleNamespace(choices=[choice], usage=usage)


class FakeOpenAIClient:
    def __init__(self, response):
        self.response = response
        self.last_payload = None
        outer = self

        class Completions:
            def create(self, **payload):
                outer.last_payload = payload
                return outer.response

        self.chat = types.SimpleNamespace(completions=Completions())


def _fn_call(cid, name, arguments):
    return types.SimpleNamespace(
        id=cid, function=types.SimpleNamespace(name=name, arguments=arguments)
    )


def test_openai_backend_reads_text_and_usage():
    client = FakeOpenAIClient(_openai_response(content=" hello "))
    backend = OpenAICompatibleBackend(model="m", client=client)
    turn = backend.complete("sys", [{"role": "user", "content": "hi"}], backend.tool_schemas())
    assert turn.text == "hello"
    assert turn.usage == {"prompt_tokens": 11, "completion_tokens": 3}
    assert client.last_payload["messages"][0]["role"] == "system"
    assert client.last_payload["tool_choice"] == "auto"


def test_openai_backend_parses_tool_call_arguments_as_json():
    """Never string-match serialized arguments; models escape them differently."""
    client = FakeOpenAIClient(
        _openai_response(tool_calls=[_fn_call("c1", "materialize", '{"session_id": "s1"}')])
    )
    backend = OpenAICompatibleBackend(model="m", client=client)
    turn = backend.complete("sys", [], backend.tool_schemas())
    assert turn.tool_calls[0].arguments == {"session_id": "s1"}
    assert turn.tool_calls[0].parse_error is None


def test_openai_backend_flags_unparseable_arguments():
    client = FakeOpenAIClient(_openai_response(tool_calls=[_fn_call("c1", "x", "{oops")]))
    backend = OpenAICompatibleBackend(model="m", client=client)
    turn = backend.complete("sys", [], backend.tool_schemas())
    assert turn.tool_calls[0].parse_error


def test_openai_backend_omits_tools_when_none_are_offered():
    client = FakeOpenAIClient(_openai_response(content="x"))
    OpenAICompatibleBackend(model="m", client=client).complete("sys", [], [])
    assert "tools" not in client.last_payload


def test_openai_message_round_trip_shape():
    backend = OpenAICompatibleBackend(model="m", client=FakeOpenAIClient(_openai_response()))
    messages: list[dict] = []
    call = ToolCall(id="c1", name="t", arguments={}, raw_arguments='{"a":1}')
    backend.append_turn(messages, types.SimpleNamespace(text="", tool_calls=[call]))
    backend.append_results(messages, [(call, '{"ok":true}')])

    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"a":1}'
    assert messages[1] == {"role": "tool", "tool_call_id": "c1", "content": '{"ok":true}'}


def test_openai_result_slots_point_at_tool_payloads():
    backend = OpenAICompatibleBackend(model="m", client=FakeOpenAIClient(_openai_response()))
    messages = [
        {"role": "user", "content": "go"},
        {"role": "tool", "tool_call_id": "a", "content": "first"},
        {"role": "assistant", "content": "x"},
        {"role": "tool", "tool_call_id": "b", "content": "second"},
    ]
    assert [c[k] for c, k in backend.result_slots(messages)] == ["first", "second"]


# -- tool calls left in the message text --------------------------------------

NAMES = {"create_session", "build_events"}


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name": "create_session", "arguments": {"data_root": "d"}}</tool_call>',
        '<tool_call>\n{"name": "create_session", "arguments": {"data_root": "d"}}\n',
        '{"name": "create_session", "parameters": {"data_root": "d"}}',
        '```json\n{"name": "create_session", "arguments": {"data_root": "d"}}\n```',
        '[TOOL_CALLS][{"name": "create_session", "arguments": {"data_root": "d"}}]',
        '<|python_tag|>{"name": "create_session", "parameters": {"data_root": "d"}}',
        '{"name": "create_session", "arguments": "{\\"data_root\\": \\"d\\"}"}',
    ],
)
def test_recover_tool_calls_reads_the_common_text_formats(text):
    calls, remaining = recover_tool_calls(text, NAMES)
    assert [c["name"] for c in calls] == ["create_session"]
    arguments = calls[0]["arguments"]
    assert (json.loads(arguments) if isinstance(arguments, str) else arguments) == {
        "data_root": "d"
    }
    assert remaining == ""


def test_recover_tool_calls_keeps_the_prose_around_tagged_calls():
    text = (
        "Let me start.\n"
        '<tool_call>{"name": "create_session", "arguments": {"data_root": "d"}}</tool_call>\n'
        '<tool_call>{"name": "build_events", "arguments": {"session_id": "s"}}</tool_call>'
    )
    calls, remaining = recover_tool_calls(text, NAMES)
    assert [c["name"] for c in calls] == ["create_session", "build_events"]
    assert remaining == "Let me start."


@pytest.mark.parametrize(
    "text",
    [
        # A report that quotes a call is not a call.
        'I called {"name": "create_session", "arguments": {}} and it worked.',
        # Unknown tool names are not recovered.
        '{"name": "rm_rf", "arguments": {}}',
        # Ordinary JSON output.
        '{"verdict": "separable"}',
        "<tool_call>not json</tool_call>",
        "The classes are separable.",
    ],
)
def test_recover_tool_calls_leaves_everything_else_alone(text):
    assert recover_tool_calls(text, NAMES) == ([], text)


def test_local_backend_turns_text_calls_into_tool_calls():
    content = '<tool_call>{"name": "build_events", "arguments": {"session_id": "s1"}}</tool_call>'
    backend = OpenAICompatibleBackend(
        model="m",
        client=FakeOpenAIClient(_openai_response(content=content)),
        recover_text_calls=True,
    )
    turn = backend.complete("sys", [], backend.tool_schemas())

    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [
        ("text_call_1", "build_events", {"session_id": "s1"})
    ]
    assert turn.text == ""
    assert turn.warnings and "did not parse" in turn.warnings[0]

    # The call goes back to the server as a structured call, as if it had parsed it.
    messages: list[dict] = []
    backend.append_turn(messages, turn)
    sent = messages[0]["tool_calls"][0]
    assert sent["id"] == "text_call_1"
    assert json.loads(sent["function"]["arguments"]) == {"session_id": "s1"}


def test_a_reply_cut_off_before_any_call_is_flagged():
    """A reasoning model can spend its whole max_tokens thinking; say which knob to turn."""
    client = FakeOpenAIClient(_openai_response(content="", finish="length"))
    backend = OpenAICompatibleBackend(model="m", client=client, max_tokens=4096)
    turn = backend.complete("sys", [], backend.tool_schemas())
    assert turn.warnings and "max_tokens (4096)" in turn.warnings[0]


def test_text_calls_are_not_recovered_unless_enabled_or_without_tools():
    content = '{"name": "build_events", "arguments": {"session_id": "s1"}}'
    hosted = OpenAICompatibleBackend(
        model="m", client=FakeOpenAIClient(_openai_response(content=content))
    )
    assert hosted.complete("sys", [], hosted.tool_schemas()).tool_calls == []

    local = OpenAICompatibleBackend(
        model="m",
        client=FakeOpenAIClient(_openai_response(content=content)),
        recover_text_calls=True,
    )
    # The tool-free wrap-up turn must come back as text.
    turn = local.complete("sys", [], [])
    assert turn.tool_calls == [] and turn.text == content


# ── provider table ───────────────────────────────────────────────────────────


def test_every_provider_is_fully_specified():
    for name, spec in PROVIDERS.items():
        assert spec.name == name
        # A local server without a default model uses the one it serves.
        assert spec.default_model or spec.local
        assert spec.note


def test_local_presets_cover_the_common_servers():
    from tessa.ai.local import _HINTS

    local = {name for name, spec in PROVIDERS.items() if spec.local}
    assert {"ollama", "lmstudio", "llamacpp", "vllm", "local"} <= local
    for name in local:
        spec = PROVIDERS[name]
        assert spec.server in _HINTS
        assert spec.base_url.startswith("http://localhost:")
        assert spec.temperature > 0, "greedy decoding loops small models"
    # `local` predates the presets and keeps meaning Ollama.
    assert PROVIDERS["local"].server == "ollama"
    assert PROVIDERS["local"].base_url == PROVIDERS["ollama"].base_url


def test_hosted_providers_still_decode_greedily():
    for spec in PROVIDERS.values():
        if not spec.local:
            assert spec.temperature == 0.0


def test_local_backend_needs_no_key_and_does_not_retry(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    backend = build_backend(provider="ollama")
    assert backend.model == "qwen3:14b"
    assert backend.temperature == 0.6
    assert backend.recover_text_calls is True
    assert backend.client.max_retries == 0
    assert backend.client.timeout == LOCAL_TIMEOUT
    # A vLLM server started without --api-key needs none either.
    assert build_backend(provider="vllm", model="m").model == "m"


def test_local_backend_sends_a_key_the_server_was_started_with(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    assert build_backend(provider="vllm", model="m").client.api_key == "secret"


def test_local_preset_without_a_default_model_asks_for_one():
    pytest.importorskip("openai")
    with pytest.raises(RuntimeError, match="--model"):
        build_backend(provider="llamacpp")


def test_explicit_settings_override_the_preset():
    pytest.importorskip("openai")
    backend = build_backend(
        provider="ollama", model="gpt-oss:20b", temperature=1.0, max_tokens=2048, timeout=60
    )
    assert (backend.model, backend.temperature, backend.max_tokens) == ("gpt-oss:20b", 1.0, 2048)
    assert backend.client.timeout == 60


def test_groq_is_the_default_provider():
    assert PROVIDERS["groq"].base_url.startswith("https://api.groq.com")
    assert PROVIDERS["groq"].key_env == "GROQ_API_KEY"


def test_unknown_provider_lists_the_valid_ones():
    with pytest.raises(RuntimeError, match="openrouter"):
        build_backend(provider="nope")


def test_missing_credential_names_the_variable_to_set(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    pytest.importorskip("openai")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_backend(provider="groq")


# ── Anthropic ────────────────────────────────────────────────────────────────


class FakeAnthropicClient:
    def __init__(self, response):
        self.response = response
        self.last_kwargs = None
        outer = self

        class Stream:
            def __init__(self, **kwargs):
                outer.last_kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return outer.response

        self.messages = types.SimpleNamespace(stream=Stream)


def _anthropic_response(blocks, stop="end_turn"):
    return types.SimpleNamespace(
        content=blocks,
        stop_reason=stop,
        usage=types.SimpleNamespace(input_tokens=7, output_tokens=2, cache_read_input_tokens=5),
    )


def test_anthropic_backend_sends_the_current_api_shape():
    client = FakeAnthropicClient(
        _anthropic_response([types.SimpleNamespace(type="text", text="hi")])
    )
    backend = AnthropicBackend(model="claude-opus-5", client=client)
    turn = backend.complete("sys", [{"role": "user", "content": "x"}], backend.tool_schemas())

    kwargs = client.last_kwargs
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    # budget_tokens was removed on this model family and is rejected.
    assert "budget_tokens" not in str(kwargs["thinking"])
    # effort belongs inside output_config, not at the top level.
    assert kwargs["output_config"]["effort"] == "high"
    assert "effort" not in {k for k in kwargs if k != "output_config"}
    # The cache breakpoint sits on the system block.
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert turn.text == "hi"
    assert turn.usage["cache_read_input_tokens"] == 5


def test_anthropic_backend_extracts_tool_use_blocks():
    blocks = [
        types.SimpleNamespace(type="text", text="thinking out loud"),
        types.SimpleNamespace(
            type="tool_use", id="t1", name="materialize", input={"session_id": "s"}
        ),
    ]
    backend = AnthropicBackend(model="m", client=FakeAnthropicClient(_anthropic_response(blocks)))
    turn = backend.complete("sys", [], backend.tool_schemas())
    assert [c.name for c in turn.tool_calls] == ["materialize"]
    assert turn.tool_calls[0].arguments == {"session_id": "s"}


def test_anthropic_results_go_back_in_a_single_user_message():
    backend = AnthropicBackend(model="m", client=FakeAnthropicClient(_anthropic_response([])))
    messages: list[dict] = []
    a = ToolCall(id="a", name="t", arguments={})
    b = ToolCall(id="b", name="t", arguments={})
    backend.append_results(messages, [(a, '{"ok":1}'), (b, '{"error":{"type":"X"}}')])

    assert len(messages) == 1 and messages[0]["role"] == "user"
    blocks = messages[0]["content"]
    assert [x["tool_use_id"] for x in blocks] == ["a", "b"]
    assert "is_error" not in blocks[0]
    assert blocks[1]["is_error"] is True


def test_anthropic_result_slots_point_at_tool_result_blocks():
    backend = AnthropicBackend(model="m", client=FakeAnthropicClient(_anthropic_response([])))
    messages: list[dict] = [{"role": "user", "content": "go"}]
    a = ToolCall(id="a", name="t", arguments={})
    b = ToolCall(id="b", name="t", arguments={})
    backend.append_results(messages, [(a, "first"), (b, "second")])
    assert [c[k] for c, k in backend.result_slots(messages)] == ["first", "second"]

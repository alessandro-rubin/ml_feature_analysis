"""Backend adapters, driven by fake clients.

Both modules keep their SDK import inside ``build_backend``, so the adapter
logic — which is where the wire-format bugs live — is fully testable with
neither ``openai`` nor ``anthropic`` installed.
"""

from __future__ import annotations

import types

import pytest

from tessa.ai.backend_anthropic import AnthropicBackend
from tessa.ai.backend_openai import PROVIDERS, OpenAICompatibleBackend, build_backend
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


# ── provider table ───────────────────────────────────────────────────────────


def test_every_provider_is_fully_specified():
    for name, spec in PROVIDERS.items():
        assert spec.name == name
        assert spec.default_model
        assert spec.note


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

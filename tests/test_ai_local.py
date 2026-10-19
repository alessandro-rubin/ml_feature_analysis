"""Local model servers, emulated over real HTTP on 127.0.0.1.

The fake server answers the endpoints each local server really exposes, so the
probe, the CLI and the OpenAI SDK are exercised end to end with no model, no
download and no network beyond the loopback interface.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import pytest

from tessa.ai.local import (
    RECOMMENDED_CONTEXT,
    LocalServerError,
    probe,
    static_prompt_tokens,
)


class FakeServer:
    """Routes ``(method, path)`` to a JSON reply and records every request."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Any] = {}
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"null")
                outer.requests.append(
                    {"method": method, "path": self.path, "body": body, "headers": self.headers}
                )
                route = outer.routes.get((method, self.path))
                status, payload = (404, {"error": "not found"}) if route is None else route
                if callable(payload):
                    payload = payload(body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                self._reply("GET")

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                self._reply("POST")

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.root = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.base_url = f"{self.root}/v1"
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self.thread.start()

    def route(self, method: str, path: str, payload: Any, status: int = 200) -> None:
        self.routes[(method, path)] = (status, payload)

    def models(self, *entries: dict[str, Any] | str) -> None:
        data = [{"id": e, "object": "model"} if isinstance(e, str) else e for e in entries]
        self.route("GET", "/v1/models", {"object": "list", "data": data})

    def paths(self) -> list[str]:
        return [r["path"] for r in self.requests]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    fake = FakeServer()
    yield fake
    fake.close()


def _ollama(
    server: FakeServer,
    model: str = "qwen3:14b",
    *,
    context: int = 32_768,
    capabilities: tuple[str, ...] = ("completion", "tools"),
    size_vram: int = 9_000,
    **show: Any,
) -> None:
    server.models(model)
    server.route(
        "POST",
        "/api/show",
        {
            "capabilities": list(capabilities),
            "model_info": {"qwen3.context_length": 40_960},
            **show,
        },
    )
    server.route("POST", "/api/generate", {"model": model, "done": True, "done_reason": "load"})
    server.route(
        "GET",
        "/api/ps",
        {
            "models": [
                {
                    "name": model,
                    "model": model,
                    "size": 9_000,
                    "size_vram": size_vram,
                    "context_length": context,
                }
            ]
        },
    )


def _closed_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


# -- Ollama -------------------------------------------------------------------


def test_ollama_probe_resolves_the_model_and_reads_the_loaded_window(server):
    _ollama(server)
    found = probe("ollama", base_url=server.base_url)

    assert (found.server, found.model, found.context_length) == ("ollama", "qwen3:14b", 32_768)
    assert found.warnings == []
    # Only a loaded model reports its window, so the probe loads it.
    assert server.paths() == ["/v1/models", "/api/show", "/api/generate", "/api/ps"]
    assert server.requests[2]["body"] == {"model": "qwen3:14b", "stream": False}


def test_ollama_bare_model_names_mean_latest(server):
    _ollama(server, "qwen3:latest")
    assert probe("ollama", base_url=server.base_url, model="qwen3").model == "qwen3:latest"


def test_a_model_that_is_not_pulled_names_the_pull_command(server):
    _ollama(server, "llama3.2:latest")
    with pytest.raises(LocalServerError, match="ollama pull qwen3:14b") as info:
        probe("ollama", base_url=server.base_url)
    assert "llama3.2:latest" in str(info.value)


def test_a_model_without_tool_calling_is_refused(server):
    _ollama(server, capabilities=("completion",))
    with pytest.raises(LocalServerError, match="cannot call tools"):
        probe("ollama", base_url=server.base_url)


@pytest.mark.parametrize("context", [4_096, 8_192, 12_288])
def test_a_window_without_room_for_results_is_refused_with_the_fix(server, context):
    """4k is Ollama's default below 24 GiB of VRAM: not even the system prompt fits."""
    _ollama(server, context=context)
    with pytest.raises(LocalServerError, match="OLLAMA_CONTEXT_LENGTH=32768") as info:
        probe("ollama", base_url=server.base_url)
    assert f"{context}-token context window" in str(info.value)


def test_a_workable_but_small_window_only_warns(server):
    _ollama(server, context=16_384)
    found = probe("ollama", base_url=server.base_url)
    assert found.context_length == 16_384
    assert any(str(RECOMMENDED_CONTEXT) in w for w in found.warnings)


def test_cpu_offload_is_reported(server):
    _ollama(server, size_vram=4_500)
    found = probe("ollama", base_url=server.base_url)
    assert any("50% of qwen3:14b runs on the CPU" in w for w in found.warnings)


def test_an_ollama_cloud_model_is_flagged_and_not_loaded(server):
    _ollama(server, "gpt-oss:120b-cloud", remote_host="https://ollama.com:443")
    found = probe("ollama", base_url=server.base_url, model="gpt-oss:120b-cloud")
    assert any("not kept on this machine" in w for w in found.warnings)
    assert "/api/generate" not in server.paths()


def test_a_failed_load_reports_the_servers_reason(server):
    _ollama(server)
    server.route(
        "POST",
        "/api/generate",
        {"error": "model requires more system memory (12.0 GiB) than is available"},
        status=500,
    )
    with pytest.raises(LocalServerError, match="more system memory"):
        probe("ollama", base_url=server.base_url)


def test_no_server_names_the_start_command():
    with pytest.raises(LocalServerError, match="ollama serve"):
        probe("ollama", base_url=_closed_port_url(), timeout=2)


def test_a_base_url_without_v1_gets_a_hint(server):
    _ollama(server)
    with pytest.raises(LocalServerError, match="usually ends in /v1"):
        probe("ollama", base_url=server.root)


def test_another_server_behind_the_ollama_preset_is_tolerated(server):
    server.models("qwen3:14b")  # but no /api/* endpoints
    found = probe("local", base_url=server.base_url)
    assert found.context_length is None
    assert any("no Ollama API" in w for w in found.warnings)
    assert any("--context-length" in w for w in found.warnings)


# -- llama.cpp, vLLM, LM Studio -----------------------------------------------


def test_llamacpp_uses_the_only_model_and_reads_n_ctx(server):
    server.models("ggml-org/gpt-oss-20b-GGUF")
    server.route("GET", "/props", {"default_generation_settings": {"n_ctx": 65_536}})
    found = probe("llamacpp", base_url=server.base_url)
    assert (found.model, found.context_length) == ("ggml-org/gpt-oss-20b-GGUF", 65_536)


def test_llamacpp_router_mode_is_asked_per_model(server):
    server.models("a", "b")
    server.route(
        "GET", "/props?model=b", {"default_generation_settings": {"n_ctx": RECOMMENDED_CONTEXT}}
    )
    assert probe("llamacpp", base_url=server.base_url, model="b").context_length == 32_768


def test_several_models_and_no_choice_asks_for_one(server):
    server.models("a", "b")
    with pytest.raises(LocalServerError, match="choose one with --model: a, b"):
        probe("llamacpp", base_url=server.base_url)


def test_vllm_reports_max_model_len(server):
    server.models({"id": "Qwen/Qwen3-8B", "max_model_len": 40_960})
    found = probe("vllm", base_url=server.base_url)
    assert (found.model, found.context_length) == ("Qwen/Qwen3-8B", 40_960)


def test_vllm_key_is_sent_and_a_rejection_names_it(server, monkeypatch):
    server.route("GET", "/v1/models", {"error": "Unauthorized"}, status=401)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    with pytest.raises(LocalServerError, match="VLLM_API_KEY"):
        probe("vllm", base_url=server.base_url)

    monkeypatch.setenv("VLLM_API_KEY", "secret")
    server.models({"id": "m", "max_model_len": 32_768})
    probe("vllm", base_url=server.base_url)
    assert server.requests[-1]["headers"]["Authorization"] == "Bearer secret"


def test_lmstudio_reads_the_loaded_context(server):
    server.models("qwen/qwen3-14b")
    server.route(
        "GET",
        "/api/v0/models/qwen%2Fqwen3-14b",
        {"id": "qwen/qwen3-14b", "state": "loaded", "loaded_context_length": 32_768},
    )
    assert probe("lmstudio", base_url=server.base_url).context_length == 32_768


def test_lmstudio_warns_when_the_model_is_not_loaded(server):
    server.models("qwen/qwen3-14b")
    server.route(
        "GET",
        "/api/v0/models/qwen%2Fqwen3-14b",
        {"id": "qwen/qwen3-14b", "state": "not-loaded", "max_context_length": 40_960},
    )
    found = probe("lmstudio", base_url=server.base_url)
    assert found.context_length is None
    assert any("lms load qwen/qwen3-14b --context-length 32768" in w for w in found.warnings)


def test_hosted_providers_are_not_probed():
    with pytest.raises(ValueError, match="not a local provider"):
        probe("groq")


def test_the_static_prompt_does_not_fit_the_ollama_default():
    """Why the probe exists: 4k tokens cannot hold the prompt and tool definitions."""
    assert static_prompt_tokens() > 4_096


# -- the CLI, end to end through the OpenAI SDK -------------------------------


def _completion(message: dict[str, Any], finish: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen3:14b",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _scripted(*replies: dict[str, Any]) -> Callable[[Any], dict[str, Any]]:
    queue = list(replies)
    return lambda body: queue.pop(0) if len(queue) > 1 else queue[0]


def _run_cli(server: FakeServer, tmp_path, *extra: str) -> int:
    from tessa.ai.cli import main

    return main(
        [
            "--data-root",
            str(tmp_path),
            "--goal",
            "describe the separability analysis",
            "--provider",
            "ollama",
            "--base-url",
            server.base_url,
            *extra,
        ]
    )


def _chat_requests(server: FakeServer) -> list[dict[str, Any]]:
    return [r["body"] for r in server.requests if r["path"] == "/v1/chat/completions"]


def test_cli_drives_a_local_server(server, tmp_path, capsys):
    pytest.importorskip("openai")
    _ollama(server)
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "describe_analysis", "arguments": '{"name": "separability"}'},
    }
    server.route(
        "POST",
        "/v1/chat/completions",
        _scripted(
            _completion({"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"),
            _completion({"role": "assistant", "content": "Report: done."}, "stop"),
        ),
    )

    assert _run_cli(server, tmp_path) == 0
    out = capsys.readouterr().out
    assert "context window: 32768 tokens" in out
    assert "describe_analysis" in out and "Report: done." in out

    first, second = _chat_requests(server)
    assert first["model"] == "qwen3:14b"
    assert first["temperature"] == 0.6
    assert first["max_tokens"] == 8192
    assert len(first["tools"]) > 10
    assert second["messages"][-1]["role"] == "tool"
    assert second["messages"][-1]["tool_call_id"] == "call_1"


def test_cli_recovers_a_call_the_server_left_in_the_text(server, tmp_path, capsys):
    pytest.importorskip("openai")
    _ollama(server)
    text_call = '<tool_call>{"name": "describe_analysis", "arguments": {"name": "pairwise"}}'
    server.route(
        "POST",
        "/v1/chat/completions",
        _scripted(
            _completion({"role": "assistant", "content": text_call}, "stop"),
            _completion({"role": "assistant", "content": "Report: done."}, "stop"),
        ),
    )

    assert _run_cli(server, tmp_path) == 0
    assert "recovered 1 tool call" in capsys.readouterr().out
    _, second = _chat_requests(server)
    assistant, result = second["messages"][-2:]
    assert assistant["tool_calls"][0]["function"]["name"] == "describe_analysis"
    assert result["tool_call_id"] == assistant["tool_calls"][0]["id"]


def test_cli_refuses_a_window_too_small_for_the_prompt(server, tmp_path, capsys):
    _ollama(server, context=4_096)
    assert _run_cli(server, tmp_path) == 2
    assert "OLLAMA_CONTEXT_LENGTH=32768" in capsys.readouterr().err
    assert _chat_requests(server) == []


def test_cli_reports_a_server_that_fails_mid_run(server, tmp_path, capsys):
    pytest.importorskip("openai")
    _ollama(server)
    server.route(
        "POST", "/v1/chat/completions", {"error": {"message": "runner crashed"}}, status=500
    )
    assert _run_cli(server, tmp_path) == 1
    assert "backend_error" in capsys.readouterr().out

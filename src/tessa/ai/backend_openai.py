"""OpenAI-compatible backend: Groq, OpenRouter, OpenAI, or a local server.

This is the primary driver for this project. It speaks the OpenAI chat-completions
protocol, which every one of those providers implements, so switching between them
is a change of base URL, environment variable and model name — nothing more.

Local servers (Ollama, LM Studio, llama.cpp, vLLM) speak the same protocol, so
they need no backend of their own; what differs is checked before the run by
``tessa.ai.local`` and handled here by two switches a preset turns on:

* **tool calls left in the message text are recovered.** When a server's chat
  template or tool parser does not recognize a model's call format, the call
  arrives as plain content and the run would end on its first turn.
* **a moderate temperature is sent.** Greedy decoding sends small open-weight
  models into repetition loops (Qwen3's model card warns against it), and
  Ollama's OpenAI endpoint substitutes 1.0 when no temperature is sent at all.

This module deliberately contains no Anthropic SDK usage; see
``backend_anthropic`` for that path.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from tessa.ai.loop import BackendTurn, ToolCall, parse_arguments
from tessa.ai.schemas import to_openai_tools

__all__ = [
    "PROVIDERS",
    "DEFAULT_MAX_TOKENS",
    "LOCAL_TIMEOUT",
    "Provider",
    "OpenAICompatibleBackend",
    "build_backend",
    "recover_tool_calls",
]

DEFAULT_MAX_TOKENS = 8192

# Seconds one completion may take on a local server. The SDK default (600 s) is
# too short for a long turn on a CPU or a partly offloaded model.
LOCAL_TIMEOUT = 1800.0


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str | None
    key_env: str | None
    default_model: str | None
    note: str = ""
    server: str | None = None
    """The local server this preset talks to (``ollama``, ``lmstudio``,
    ``llamacpp``, ``vllm``), or None for a hosted API."""
    temperature: float = 0.0

    @property
    def local(self) -> bool:
        return self.server is not None


# Default models are the current tool-calling-capable choice for each provider and
# are overridable with --model; nothing here should be treated as pinned. A local
# preset without a default uses whichever model the server reports it serves.
PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        note="Fast and free-tier friendly; solid tool calling for an open-weight model.",
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        default_model="anthropic/claude-sonnet-4.5",
        note="One key, many models — including frontier ones. Reliability tracks "
        "whichever model you route to.",
    ),
    "openai": Provider(
        name="openai",
        base_url=None,
        key_env="OPENAI_API_KEY",
        default_model="gpt-4o",
        note="The SDK default endpoint.",
    ),
    "ollama": Provider(
        name="ollama",
        base_url="http://localhost:11434/v1",
        key_env=None,
        default_model="qwen3:14b",
        note="Ollama on this machine: no key, no egress. Start it with a 32k+ context "
        "window (OLLAMA_CONTEXT_LENGTH=32768 ollama serve).",
        server="ollama",
        temperature=0.6,
    ),
    "lmstudio": Provider(
        name="lmstudio",
        base_url="http://localhost:1234/v1",
        key_env=None,
        default_model=None,
        note="LM Studio's local server. Load the model with a 32k+ context length.",
        server="lmstudio",
        temperature=0.6,
    ),
    "llamacpp": Provider(
        name="llamacpp",
        base_url="http://localhost:8080/v1",
        key_env="LLAMA_API_KEY",
        default_model=None,
        note="llama.cpp's llama-server. Start it with --jinja (the default in recent "
        "builds) so tool calls are parsed.",
        server="llamacpp",
        temperature=0.6,
    ),
    "vllm": Provider(
        name="vllm",
        base_url="http://localhost:8000/v1",
        key_env="VLLM_API_KEY",
        default_model=None,
        note="vLLM's OpenAI-compatible server. Start it with --enable-auto-tool-choice "
        "and the model's --tool-call-parser.",
        server="vllm",
        temperature=0.6,
    ),
    "local": Provider(
        name="local",
        base_url="http://localhost:11434/v1",
        key_env=None,
        default_model="qwen3:14b",
        note="Alias of `ollama`, kept for compatibility.",
        server="ollama",
        temperature=0.6,
    ),
}


_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.DOTALL)
_FENCE = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL)
_CALL_MARKERS = ("[TOOL_CALLS]", "<|python_tag|>")


def recover_tool_calls(text: str, tool_names: set[str]) -> tuple[list[dict[str, Any]], str]:
    """Pull tool calls out of message text, for a server that did not parse them.

    Only two shapes are accepted, so a written report that merely quotes a call
    is not mistaken for one:

    * ``<tool_call>{...}</tool_call>`` blocks (the Hermes / Qwen format);
    * a message that is nothing but one JSON object or a list of them, allowing
      for a code fence or a ``[TOOL_CALLS]`` / ``<|python_tag|>`` marker.

    Each object must name a known tool, with its arguments under ``arguments``
    or ``parameters``.

    Returns
    -------
    (calls, remaining_text)
        ``calls`` holds ``{"name", "arguments"}`` dicts, where ``arguments`` is
        whatever the model wrote (a dict or a JSON string); ``remaining_text``
        is the text with the recovered calls removed. No calls means the text
        comes back unchanged.
    """

    def as_call(obj: Any) -> dict[str, Any] | None:
        if not isinstance(obj, dict) or obj.get("name") not in tool_names:
            return None
        return {"name": obj["name"], "arguments": obj.get("arguments", obj.get("parameters", {}))}

    def parse(fragment: str) -> list[dict[str, Any]] | None:
        try:
            value = json.loads(fragment)
        except ValueError:
            return None
        items = value if isinstance(value, list) else [value]
        calls = [as_call(item) for item in items]
        if not calls or any(c is None for c in calls):
            return None
        return calls

    if "<tool_call>" in text:
        calls: list[dict[str, Any]] = []

        def take(match: re.Match) -> str:
            parsed = parse(match.group(1))
            if parsed is None:
                return match.group(0)
            calls.extend(parsed)
            return ""

        remaining = _TOOL_CALL_TAG.sub(take, text)
        return (calls, remaining.strip()) if calls else ([], text)

    body = text.strip()
    for marker in _CALL_MARKERS:
        if body.startswith(marker):
            body = body[len(marker) :].strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    if not body.startswith(("{", "[")):
        return [], text
    parsed = parse(body)
    return (parsed, "") if parsed else ([], text)


@dataclass
class OpenAICompatibleBackend:
    """Adapter over the chat-completions protocol."""

    model: str
    client: Any
    name: str = "openai-compatible"
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    recover_text_calls: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    _recovered: int = field(default=0, init=False, repr=False)

    def tool_schemas(self) -> list[dict]:
        return to_openai_tools()

    def complete(self, system: str, messages: list[dict], tools: list[Any]) -> BackendTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**payload)
        choice = response.choices[0]
        message = choice.message
        text = (message.content or "").strip()

        calls: list[ToolCall] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            arguments, error = parse_arguments(raw_call.function.arguments)
            calls.append(
                ToolCall(
                    id=raw_call.id,
                    name=raw_call.function.name,
                    arguments=arguments,
                    raw_arguments=raw_call.function.arguments or "",
                    parse_error=error,
                )
            )

        warnings: list[str] = []
        if not calls and tools and self.recover_text_calls and text:
            names = {t["function"]["name"] for t in tools}
            recovered, text = recover_tool_calls(text, names)
            for item in recovered:
                arguments, error = parse_arguments(item["arguments"])
                raw = item["arguments"]
                self._recovered += 1
                calls.append(
                    ToolCall(
                        id=f"text_call_{self._recovered}",
                        name=item["name"],
                        arguments=arguments,
                        raw_arguments=raw if isinstance(raw, str) else json.dumps(raw),
                        parse_error=error,
                    )
                )
            if recovered:
                warnings.append(
                    f"recovered {len(recovered)} tool call(s) from the message text: the "
                    "server did not parse them. Check its tool-calling setup (see "
                    "AI_INTEGRATION.md, 'Running with a local model')."
                )
        if choice.finish_reason == "length" and not calls:
            # Typical of a reasoning model that spent the whole reply thinking.
            warnings.append(
                f"the reply was cut off at max_tokens ({self.max_tokens}) before it called a "
                "tool; raise --max-tokens, or the context window it is capped by."
            )

        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            }

        return BackendTurn(
            text=text,
            tool_calls=calls,
            stop_reason=choice.finish_reason or "",
            usage=usage,
            raw=response,
            warnings=warnings,
        )

    def append_turn(self, messages: list[dict], turn: BackendTurn) -> None:
        messages.append(
            {
                "role": "assistant",
                "content": turn.text or None,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.raw_arguments or "{}"},
                    }
                    for c in turn.tool_calls
                ],
            }
        )

    def append_results(self, messages: list[dict], results: list[tuple[ToolCall, str]]) -> None:
        # One `tool` message per call, all appended before the next completion —
        # the OpenAI-protocol equivalent of returning them in a single turn.
        for call, payload in results:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": payload})

    def result_slots(self, messages: list[dict]) -> list[tuple[dict, str]]:
        """Where each tool result's payload sits in ``messages``, oldest first."""
        return [(m, "content") for m in messages if m.get("role") == "tool"]


def build_backend(
    provider: str = "groq",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    **extra: Any,
) -> OpenAICompatibleBackend:
    """Construct a backend for one of the known providers.

    This makes no network call. For a local preset, run
    ``tessa.ai.local.probe`` first: it checks the server, and resolves the model
    when the preset has no default.

    Raises
    ------
    RuntimeError
        If the SDK is missing, no credential is available, or no model is
        known, with the exact command needed to fix it.
    """
    if provider not in PROVIDERS:
        raise RuntimeError(
            f"Unknown provider {provider!r}. Choose one of {sorted(PROVIDERS)}, "
            "or pass an explicit base_url."
        )
    spec = PROVIDERS[provider]
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "The `openai` package is required for OpenAI-compatible providers. "
            "Install it with:  uv sync --extra ai"
        ) from exc

    key = api_key or (os.environ.get(spec.key_env) if spec.key_env else None)
    # A local server only wants a key if it was started with one.
    if spec.key_env and not key and not spec.local:
        raise RuntimeError(
            f"No API key for provider {provider!r}. Set {spec.key_env}, "
            f"e.g.  export {spec.key_env}=..."
        )

    resolved_model = model or spec.default_model
    if not resolved_model:
        raise RuntimeError(
            f"Provider {provider!r} has no default model. Pass --model with a name "
            "the server lists under /v1/models."
        )

    client_options: dict[str, Any] = {}
    if spec.local:
        # Retrying a turn that timed out on local hardware only repeats the wait.
        client_options = {"timeout": timeout or LOCAL_TIMEOUT, "max_retries": 0}
    elif timeout:
        client_options = {"timeout": timeout}

    client = OpenAI(
        api_key=key or "not-needed", base_url=base_url or spec.base_url, **client_options
    )
    return OpenAICompatibleBackend(
        model=resolved_model,
        client=client,
        name=provider,
        temperature=spec.temperature if temperature is None else temperature,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        recover_text_calls=spec.local,
        extra=extra,
    )

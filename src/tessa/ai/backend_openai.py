"""OpenAI-compatible backend: Groq, OpenRouter, OpenAI, or a local server.

This is the primary driver for this project. It speaks the OpenAI chat-completions
protocol, which every one of those providers implements, so switching between them
is a change of base URL, environment variable and model name — nothing more.

This module deliberately contains no Anthropic SDK usage; see
``backend_anthropic`` for that path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from tessa.ai.loop import BackendTurn, ToolCall, parse_arguments
from tessa.ai.schemas import to_openai_tools

__all__ = ["PROVIDERS", "Provider", "OpenAICompatibleBackend", "build_backend"]


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str | None
    key_env: str | None
    default_model: str
    note: str = ""


# Default models are the current tool-calling-capable choice for each provider and
# are overridable with --model; nothing here should be treated as pinned.
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
    "local": Provider(
        name="local",
        base_url="http://localhost:11434/v1",
        key_env=None,
        default_model="qwen2.5:14b",
        note="Ollama or vLLM. No key or egress; tool calling on small models is "
        "the least reliable option.",
    ),
}


@dataclass
class OpenAICompatibleBackend:
    """Adapter over the chat-completions protocol."""

    model: str
    client: Any
    name: str = "openai-compatible"
    temperature: float = 0.0
    max_tokens: int = 8192
    extra: dict[str, Any] = field(default_factory=dict)

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

        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            }

        return BackendTurn(
            text=(message.content or "").strip(),
            tool_calls=calls,
            stop_reason=choice.finish_reason or "",
            usage=usage,
            raw=response,
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


def build_backend(
    provider: str = "groq",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **extra: Any,
) -> OpenAICompatibleBackend:
    """Construct a backend for one of the known providers.

    Raises
    ------
    RuntimeError
        If the SDK is missing or no credential is available, with the exact
        command needed to fix it.
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
    if spec.key_env and not key:
        raise RuntimeError(
            f"No API key for provider {provider!r}. Set {spec.key_env}, "
            f"e.g.  export {spec.key_env}=..."
        )

    client = OpenAI(api_key=key or "not-needed", base_url=base_url or spec.base_url)
    return OpenAICompatibleBackend(
        model=model or spec.default_model,
        client=client,
        name=provider,
        extra=extra,
    )

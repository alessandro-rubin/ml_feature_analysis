"""Anthropic backend.

Kept separate from the OpenAI-compatible path so neither file mixes SDKs.

Requires an API key. If you have a Claude subscription but no API key, the MCP
server (``tessa-mcp``) is the route to use instead: Claude Desktop or Claude Code
supplies the model and the agent loop, and drives exactly the same tools.

API notes, current as of writing:

* ``claude-opus-5`` with adaptive thinking. ``budget_tokens`` was removed on this
  model family and is rejected.
* ``display="summarized"`` is set explicitly — the default is ``"omitted"``,
  which makes a long analysis turn look like a hang.
* ``effort`` belongs inside ``output_config``, not at the top level.
* The cache breakpoint sits on the system block: render order is
  tools -> system -> messages, so tools and system must stay byte-stable and
  everything volatile goes in the first user message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tessa.ai.loop import BackendTurn, ToolCall
from tessa.ai.schemas import to_anthropic_tools

__all__ = ["AnthropicBackend", "build_backend", "DEFAULT_MODEL"]

DEFAULT_MODEL = "claude-opus-5"


@dataclass
class AnthropicBackend:
    """Adapter over the Anthropic Messages API."""

    model: str
    client: Any
    name: str = "anthropic"
    max_tokens: int = 64_000
    effort: str = "high"
    extra: dict[str, Any] = field(default_factory=dict)

    def tool_schemas(self) -> list[dict]:
        return to_anthropic_tools()

    def complete(self, system: str, messages: list[dict], tools: list[Any]) -> BackendTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": self.effort},
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            **self.extra,
        }
        if tools:
            kwargs["tools"] = tools

        # Streaming: max_tokens is large enough that a non-streaming request
        # would risk an HTTP timeout.
        with self.client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()

        text_parts, calls = [], []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
                "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
            }

        return BackendTurn(
            text="\n".join(t for t in text_parts if t).strip(),
            tool_calls=calls,
            stop_reason=response.stop_reason or "",
            usage=usage,
            raw=response,
        )

    def append_turn(self, messages: list[dict], turn: BackendTurn) -> None:
        messages.append({"role": "assistant", "content": turn.raw.content})

    def append_results(self, messages: list[dict], results: list[tuple[ToolCall, str]]) -> None:
        # Every tool_result from one assistant turn must go back in ONE user
        # message, or the model learns to stop issuing parallel calls.
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": payload,
                        **({"is_error": True} if '"error"' in payload[:200] else {}),
                    }
                    for call, payload in results
                ],
            }
        )


def build_backend(
    model: str = DEFAULT_MODEL, effort: str = "high", **extra: Any
) -> AnthropicBackend:
    """Construct the Anthropic backend, resolving credentials from the environment.

    Raises
    ------
    RuntimeError
        If the SDK is missing or no credential resolves, naming both remedies.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "The `anthropic` package is required for this backend. Install it with:  "
            "uv sync --extra ai-anthropic"
        ) from exc

    try:
        # Zero-arg: resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
        # `ant auth login` profile. Never hardcode a key.
        client = anthropic.Anthropic()
    except Exception as exc:
        raise RuntimeError(
            "Could not construct an Anthropic client. Run `ant auth login`, or set "
            "ANTHROPIC_API_KEY. If you have a Claude subscription but no API key, "
            "use the MCP server (`tessa-mcp`) with Claude Desktop or Claude Code "
            "instead — it drives the same tools."
        ) from exc

    return AnthropicBackend(model=model, client=client, effort=effort, extra=extra)

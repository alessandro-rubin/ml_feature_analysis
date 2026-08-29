"""The provider-neutral agent loop.

One loop, several backends. The old AI layer had the Groq variant import the
system prompt and tool dispatcher from the Anthropic module but duplicate the
loop itself; the two then drifted. Here every backend implements a three-line
protocol and the loop owns everything else — turn capping, failure capping,
argument parsing, result truncation, and the transcript.

Invariants worth naming, because they are the ones a hand-rolled loop gets wrong:

* **All tool results from one assistant turn go back together.** Splitting them
  across messages teaches the model to stop making parallel calls. The old
  prompt instructed "Call tools ONE AT A TIME" — throwing away parallelism to
  avoid a bug rather than fixing it.
* **Arguments are always parsed as JSON**, never string-matched. Models differ in
  how they escape strings inside tool arguments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from tessa.ai import tools as tool_module
from tessa.ai.render import BYTE_CAP

__all__ = ["ToolCall", "BackendTurn", "Backend", "AgentResult", "run_loop"]

MAX_TURNS = 40
MAX_CONSECUTIVE_FAILURES = 5


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class BackendTurn:
    """What one model turn produced."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


class Backend(Protocol):
    """The whole provider surface the loop depends on."""

    name: str
    model: str

    def complete(self, system: str, messages: list[dict], tools: list[Any]) -> BackendTurn: ...

    def tool_schemas(self) -> list[Any]: ...

    def append_turn(self, messages: list[dict], turn: BackendTurn) -> None:
        """Append the assistant turn in this provider's message format."""
        ...

    def append_results(self, messages: list[dict], results: list[tuple[ToolCall, str]]) -> None:
        """Append all tool results from one turn, in this provider's format."""
        ...


@dataclass
class AgentResult:
    """The outcome of a run."""

    final_text: str
    turns: int
    tool_calls: int
    transcript: list[dict[str, Any]]
    stopped_because: str
    usage: dict[str, Any] = field(default_factory=dict)


def parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Parse tool arguments, tolerating the shapes weaker models emit."""
    if isinstance(raw, dict):
        return raw, None
    if raw is None or raw == "":
        return {}, None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return {}, f"arguments were not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return {}, f"arguments must be a JSON object, got {type(parsed).__name__}"
    return parsed, None


def _truncate(payload: str, cap: int = BYTE_CAP) -> str:
    if len(payload) <= cap:
        return payload
    return payload[:cap] + f'... [truncated at {cap} bytes; use get_result for more]"'


def _execute(call: ToolCall) -> tuple[str, bool]:
    """Run one tool call, returning its serialized result and whether it failed."""
    if call.parse_error:
        return (
            json.dumps(
                {
                    "error": {
                        "type": "MalformedArguments",
                        "message": call.parse_error,
                        "hint": "Send the arguments as a JSON object.",
                    }
                }
            ),
            True,
        )
    result = tool_module.call_tool(call.name, call.arguments)
    failed = isinstance(result, dict) and "error" in result
    return _truncate(json.dumps(result, default=str)), failed


def run_loop(
    backend: Backend,
    system: str,
    opening: str,
    *,
    max_turns: int = MAX_TURNS,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    on_event: Callable[[str, dict], None] | None = None,
) -> AgentResult:
    """Drive a backend until it stops calling tools.

    Parameters
    ----------
    backend : Backend
        The provider adapter.
    system : str
        System prompt — stable, so it can sit in a cached prefix.
    opening : str
        The first user message, carrying the volatile per-run details.
    max_turns : int
        Hard cap. On exceed, one final tool-free turn is requested so a capped
        run still produces a written deliverable rather than stopping mid-thought.
    max_consecutive_failures : int
        Give up after this many consecutive failing turns — the guard against a
        weaker model looping on a call it cannot get right.
    on_event : callable, optional
        Receives ``(kind, payload)`` for progress display.
    """
    emit = on_event or (lambda kind, payload: None)
    schemas = backend.tool_schemas()
    messages: list[dict] = [{"role": "user", "content": opening}]
    transcript: list[dict[str, Any]] = []
    usage_total: dict[str, Any] = {}

    turns = 0
    n_tool_calls = 0
    consecutive_failures = 0
    final_text = ""
    stopped = "end_turn"

    while True:
        if turns >= max_turns:
            stopped = "max_turns"
            emit("cap", {"turns": turns})
            final_text = _wrap_up(backend, system, messages, emit) or final_text
            break

        turn = backend.complete(system, messages, schemas)
        turns += 1
        for key, value in (turn.usage or {}).items():
            if isinstance(value, (int, float)):
                usage_total[key] = usage_total.get(key, 0) + value

        if turn.text:
            final_text = turn.text
            emit("text", {"text": turn.text})
        transcript.append(
            {
                "role": "assistant",
                "text": turn.text,
                "tool_calls": [c.name for c in turn.tool_calls],
            }
        )

        if not turn.tool_calls:
            stopped = turn.stop_reason or "end_turn"
            break

        backend.append_turn(messages, turn)

        results: list[tuple[ToolCall, str]] = []
        turn_failed = True
        for call in turn.tool_calls:
            emit("tool", {"name": call.name, "arguments": call.arguments})
            payload, failed = _execute(call)
            n_tool_calls += 1
            if not failed:
                turn_failed = False
            emit("result", {"name": call.name, "failed": failed, "payload": payload})
            transcript.append({"role": "tool", "name": call.name, "failed": failed})
            results.append((call, payload))

        # All results from one assistant turn go back in one message.
        backend.append_results(messages, results)

        consecutive_failures = consecutive_failures + 1 if turn_failed else 0
        if consecutive_failures >= max_consecutive_failures:
            stopped = "too_many_failures"
            emit("cap", {"consecutive_failures": consecutive_failures})
            final_text = _wrap_up(backend, system, messages, emit) or final_text
            break

    return AgentResult(
        final_text=final_text,
        turns=turns,
        tool_calls=n_tool_calls,
        transcript=transcript,
        stopped_because=stopped,
        usage=usage_total,
    )


def _wrap_up(backend: Backend, system: str, messages: list[dict], emit) -> str:
    """Ask for a written summary with no tools, so a capped run still delivers."""
    closing = list(messages) + [
        {
            "role": "user",
            "content": (
                "Stop calling tools now and write your final report: what separates "
                "the classes, which features you invented and why, the measured "
                "evidence, and what you would verify next on held-out data."
            ),
        }
    ]
    try:
        turn = backend.complete(system, closing, [])
    except Exception as exc:  # a failed wrap-up must not mask the real result
        emit("error", {"message": f"wrap-up turn failed: {exc}"})
        return ""
    if turn.text:
        emit("text", {"text": turn.text})
    return turn.text

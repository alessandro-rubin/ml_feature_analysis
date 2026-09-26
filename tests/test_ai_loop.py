"""The agent loop, driven by a stub backend. No network, no credentials."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from tessa.ai.loop import BackendTurn, ToolCall, parse_arguments, run_loop


@dataclass
class StubBackend:
    """Replays a scripted list of turns and records what the loop sent back."""

    script: list[BackendTurn]
    name: str = "stub"
    model: str = "stub-1"
    calls: int = 0
    seen_messages: list[list[dict]] = field(default_factory=list)
    tools_last_call: list = field(default_factory=list)

    def tool_schemas(self):
        return [{"name": "stub"}]

    def complete(self, system, messages, tools):
        self.seen_messages.append([dict(m) for m in messages])
        self.tools_last_call = tools
        turn = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return turn

    def append_turn(self, messages, turn):
        messages.append({"role": "assistant", "content": turn.text, "_calls": turn.tool_calls})

    def append_results(self, messages, results):
        messages.append(
            {"role": "tool_batch", "content": [{"id": c.id, "payload": p} for c, p in results]}
        )


def _call(name="list_capabilities", args=None, cid="c1", raw=None, err=None):
    return ToolCall(
        id=cid, name=name, arguments=args or {}, raw_arguments=raw or "{}", parse_error=err
    )


# ── argument parsing ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"a": 1}, {"a": 1}),
        ('{"a": 1}', {"a": 1}),
        ("", {}),
        (None, {}),
    ],
)
def test_parse_arguments_accepts_the_usual_shapes(raw, expected):
    parsed, error = parse_arguments(raw)
    assert parsed == expected and error is None


@pytest.mark.parametrize("raw", ["{not json", "[1,2]", '"a string"', "42"])
def test_parse_arguments_reports_bad_shapes(raw):
    parsed, error = parse_arguments(raw)
    assert parsed == {} and error


# ── loop mechanics ───────────────────────────────────────────────────────────


def test_loop_stops_when_the_model_stops_calling_tools():
    backend = StubBackend([BackendTurn(text="done", stop_reason="end_turn")])
    result = run_loop(backend, "sys", "go")
    assert result.turns == 1 and result.tool_calls == 0
    assert result.final_text == "done" and result.stopped_because == "end_turn"


def test_loop_executes_tools_then_continues():
    backend = StubBackend(
        [
            BackendTurn(tool_calls=[_call()], stop_reason="tool_use"),
            BackendTurn(text="finished", stop_reason="end_turn"),
        ]
    )
    result = run_loop(backend, "sys", "go")
    assert result.tool_calls == 1
    assert result.final_text == "finished"


def test_all_results_from_one_turn_go_back_together():
    """Splitting them teaches the model to stop making parallel calls."""
    backend = StubBackend(
        [
            BackendTurn(
                tool_calls=[_call(cid="a"), _call(cid="b"), _call(cid="c")],
                stop_reason="tool_use",
            ),
            BackendTurn(text="done", stop_reason="end_turn"),
        ]
    )
    run_loop(backend, "sys", "go")
    batches = [m for m in backend.seen_messages[-1] if m.get("role") == "tool_batch"]
    assert len(batches) == 1
    assert [r["id"] for r in batches[0]["content"]] == ["a", "b", "c"]


def test_unknown_tool_comes_back_as_a_recoverable_error():
    backend = StubBackend(
        [
            BackendTurn(tool_calls=[_call(name="no_such_tool")], stop_reason="tool_use"),
            BackendTurn(text="ok", stop_reason="end_turn"),
        ]
    )
    events = []
    run_loop(backend, "sys", "go", on_event=lambda k, p: events.append((k, p)))
    payload = next(p for k, p in events if k == "result")
    assert payload["failed"] is True
    assert json.loads(payload["payload"])["error"]["type"] == "UnknownTool"


def test_malformed_arguments_are_reported_not_executed():
    backend = StubBackend(
        [
            BackendTurn(
                tool_calls=[_call(err="arguments were not valid JSON: boom")],
                stop_reason="tool_use",
            ),
            BackendTurn(text="ok", stop_reason="end_turn"),
        ]
    )
    events = []
    run_loop(backend, "sys", "go", on_event=lambda k, p: events.append((k, p)))
    payload = json.loads(next(p for k, p in events if k == "result")["payload"])
    assert payload["error"]["type"] == "MalformedArguments"


# ── the caps ─────────────────────────────────────────────────────────────────


def test_turn_cap_still_produces_a_written_deliverable():
    """A capped run must not just stop mid-thought."""
    backend = StubBackend([BackendTurn(tool_calls=[_call()], stop_reason="tool_use")])
    result = run_loop(backend, "sys", "go", max_turns=3)
    assert result.stopped_because == "max_turns"
    assert result.turns == 3
    # The wrap-up turn is tool-free.
    assert backend.tools_last_call == []


def test_consecutive_failures_stop_a_stuck_model():
    backend = StubBackend([BackendTurn(tool_calls=[_call(name="bogus")], stop_reason="tool_use")])
    result = run_loop(backend, "sys", "go", max_turns=50, max_consecutive_failures=3)
    assert result.stopped_because == "too_many_failures"
    assert result.turns == 3


def test_a_successful_call_resets_the_failure_counter():
    good = BackendTurn(
        tool_calls=[_call(name="describe_analysis", args={"name": "separability"})],
        stop_reason="tool_use",
    )
    bad = BackendTurn(tool_calls=[_call(name="bogus")], stop_reason="tool_use")
    backend = StubBackend([bad, bad, good, bad, bad, BackendTurn(text="ok")])
    result = run_loop(backend, "sys", "go", max_turns=20, max_consecutive_failures=3)
    assert result.stopped_because != "too_many_failures"


def test_a_failing_wrap_up_does_not_mask_the_result():
    class Exploding(StubBackend):
        def complete(self, system, messages, tools):
            if not tools:
                raise RuntimeError("wrap-up exploded")
            return super().complete(system, messages, tools)

    backend = Exploding([BackendTurn(text="partial", tool_calls=[_call()], stop_reason="tool_use")])
    result = run_loop(backend, "sys", "go", max_turns=2)
    assert result.stopped_because == "max_turns"
    assert result.final_text == "partial"


def test_usage_is_accumulated_across_turns():
    backend = StubBackend(
        [
            BackendTurn(tool_calls=[_call()], usage={"prompt_tokens": 10}, stop_reason="tool_use"),
            BackendTurn(text="done", usage={"prompt_tokens": 5}),
        ]
    )
    assert run_loop(backend, "sys", "go").usage["prompt_tokens"] == 15


def test_large_payloads_are_truncated_before_reaching_the_model():
    backend = StubBackend(
        [
            BackendTurn(
                tool_calls=[_call(name="describe_analysis", args={"name": "separability"})],
                stop_reason="tool_use",
            ),
            BackendTurn(text="done"),
        ]
    )
    events = []
    run_loop(backend, "sys", "go", on_event=lambda k, p: events.append((k, p)))
    payload = next(p for k, p in events if k == "result")["payload"]
    assert len(payload) <= 20_000 + 200

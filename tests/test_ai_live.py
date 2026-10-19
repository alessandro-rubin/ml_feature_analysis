"""Opt-in tests that call a real model API.

Deselected by default (``addopts = "-m 'not live'"``) so CI never spends money.
Run them deliberately::

    export GROQ_API_KEY=gsk_...
    uv run pytest -m live -q

    # or against Claude (with no GROQ_API_KEY / OPENROUTER_API_KEY set)
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run pytest -m live -q

    # or against a model on a local server: free, but slow
    TESSA_LIVE_LOCAL=ollama uv run pytest -m live -q
    TESSA_LIVE_LOCAL=llamacpp TESSA_LIVE_MODEL=<id> uv run pytest -m live -q
"""

from __future__ import annotations

import os

import pytest

from tessa.ai.demo_data import generate
from tessa.ai.loop import run_loop
from tessa.ai.prompts import opening_message, system_prompt

pytestmark = pytest.mark.live


def _backend():
    """The local server named in TESSA_LIVE_LOCAL, else the first provider with a
    credential; returned with the context window to stay inside, if known."""
    local = os.environ.get("TESSA_LIVE_LOCAL")
    if local:
        pytest.importorskip("openai")
        from tessa.ai.backend_openai import DEFAULT_MAX_TOKENS, build_backend
        from tessa.ai.local import probe

        found = probe(local, model=os.environ.get("TESSA_LIVE_MODEL"))
        context = found.context_length
        max_tokens = min(DEFAULT_MAX_TOKENS, context // 4) if context else None
        return build_backend(provider=local, model=found.model, max_tokens=max_tokens), context
    if os.environ.get("GROQ_API_KEY"):
        pytest.importorskip("openai")
        from tessa.ai.backend_openai import build_backend

        return build_backend(provider="groq"), None
    if os.environ.get("OPENROUTER_API_KEY"):
        pytest.importorskip("openai")
        from tessa.ai.backend_openai import build_backend

        return build_backend(provider="openrouter"), None
    if os.environ.get("ANTHROPIC_API_KEY"):
        pytest.importorskip("anthropic")
        from tessa.ai.backend_anthropic import build_backend

        return build_backend(), None
    pytest.skip("no model credential in the environment, and TESSA_LIVE_LOCAL is not set")


def test_a_real_model_can_drive_the_tools(tmp_path):
    """A few turns against a live model: does it get as far as a table?"""
    root, labels = generate(tmp_path / "data")
    backend, context = _backend()

    seen: list[str] = []
    result = run_loop(
        backend,
        system_prompt(),
        opening_message(
            "Characterize the two classes. Materialize a table and run separability.",
            str(root),
            str(labels),
        ),
        max_turns=12,
        context_tokens=context,
        on_event=lambda kind, payload: seen.append(payload["name"]) if kind == "tool" else None,
    )

    assert result.tool_calls > 0, "the model never called a tool"
    assert "create_session" in seen, seen
    assert result.stopped_because != "too_many_failures", seen
    assert result.final_text, "the run produced no written output"

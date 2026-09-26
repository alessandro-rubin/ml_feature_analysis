"""Opt-in tests that call a real model API.

Deselected by default (``addopts = "-m 'not live'"``) so CI never spends money.
Run them deliberately::

    export GROQ_API_KEY=gsk_...
    uv run pytest -m live -q

    # or against Claude
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run pytest -m live -q --provider anthropic
"""

from __future__ import annotations

import os

import pytest

from tessa.ai.demo_data import generate
from tessa.ai.loop import run_loop
from tessa.ai.prompts import opening_message, system_prompt

pytestmark = pytest.mark.live


def _backend():
    """Pick whichever provider has a credential available."""
    if os.environ.get("GROQ_API_KEY"):
        pytest.importorskip("openai")
        from tessa.ai.backend_openai import build_backend

        return build_backend(provider="groq")
    if os.environ.get("OPENROUTER_API_KEY"):
        pytest.importorskip("openai")
        from tessa.ai.backend_openai import build_backend

        return build_backend(provider="openrouter")
    if os.environ.get("ANTHROPIC_API_KEY"):
        pytest.importorskip("anthropic")
        from tessa.ai.backend_anthropic import build_backend

        return build_backend()
    pytest.skip("no model credential in the environment")


def test_a_real_model_can_drive_the_tools(tmp_path):
    """A few turns against a live model: does it get as far as a table?"""
    root, labels = generate(tmp_path / "data")
    backend = _backend()

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
        on_event=lambda kind, payload: seen.append(payload["name"]) if kind == "tool" else None,
    )

    assert result.tool_calls > 0, "the model never called a tool"
    assert "create_session" in seen, seen
    assert result.stopped_because != "too_many_failures", seen
    assert result.final_text, "the run produced no written output"

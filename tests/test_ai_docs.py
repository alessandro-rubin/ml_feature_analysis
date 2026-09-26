"""AI_INTEGRATION.md's generated sections must match the code."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from gen_ai_docs import DOC, render, render_allowlist, render_tool_table  # noqa: E402


def test_generated_sections_are_up_to_date():
    """Run `uv run python scripts/gen_ai_docs.py` if this fails."""
    current = DOC.read_text()
    assert render(current) == current, (
        "AI_INTEGRATION.md is stale; regenerate with `uv run python scripts/gen_ai_docs.py`"
    )


def test_tool_table_covers_every_tool():
    from tessa.ai.tools import TOOLS_BY_NAME

    table = render_tool_table()
    for name in TOOLS_BY_NAME:
        assert f"| `{name}` |" in table


def test_allowlist_documents_what_the_compiler_accepts():
    from tessa.ai.expressions import allowlist_reference

    block = render_allowlist()
    for method in allowlist_reference()["expression_methods"]:
        assert f"`{method}`" in block


def test_doc_states_the_threat_model_and_selection_bias():
    """Two things this doc must not quietly omit."""
    text = DOC.read_text()
    assert "AST allowlist is the security boundary" in text
    assert "defence in depth only" in text
    assert "Selection bias" in text
    assert "confirm_on_holdout" in text

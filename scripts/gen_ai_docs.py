"""Regenerate the machine-derived sections of AI_INTEGRATION.md.

The expression allowlist and the tool table are read from the code, so the
documentation cannot drift from what the compiler actually accepts or from the
tools that are actually exposed. ``tests/test_ai_docs.py`` fails if the file on
disk is stale.

Run after changing the tool surface or the allowlist::

    uv run python scripts/gen_ai_docs.py
"""

from __future__ import annotations

from pathlib import Path

from tessa.ai.expressions import allowlist_reference
from tessa.ai.schemas import tool_schemas

DOC = Path(__file__).resolve().parent.parent / "AI_INTEGRATION.md"

# What each tool needs to have happened first — the state machine the order
# guards in tools.py enforce.
_REQUIRES = {
    "create_session": "—",
    "describe_data": "session",
    "build_events": "session",
    "list_capabilities": "session",
    "describe_analysis": "—",
    "seed_builtin_features": "events",
    "preview_feature": "events",
    "create_feature": "events",
    "create_aggregator": "session",
    "drop_feature": "session",
    "materialize": "events",
    "set_active_table": "table",
    "run_analysis": "table",
    "get_result": "analysis run",
    "characterize_classes": "table",
    "score_features": "table",
    "feature_ledger": "session",
    "confirm_on_holdout": "table + holdout",
    "save_run": "analysis run",
    "write_report": "analysis run",
}


def _names_block(title: str, items: list[str], per_line: int = 6) -> str:
    rows = [
        "  " + " ".join(f"`{x}`" for x in items[i : i + per_line])
        for i in range(0, len(items), per_line)
    ]
    return f"**{title}**\n\n" + "\n".join(rows) + "\n"


def render_allowlist() -> str:
    ref = allowlist_reference()
    return "\n".join(
        [
            _names_block("Entry points (pl.…)", ref["pl_functions"]),
            _names_block("Cast targets", ref["cast_dtypes"]),
            _names_block("Expression methods", ref["expression_methods"]),
            _names_block("dt namespace", ref["namespaces"]["dt"]),
            "**Rejected by name** (beyond everything not listed above)\n\n  "
            + " ".join(f"`{x}`" for x in ref["excluded"])
            + "\n",
        ]
    )


def render_tool_table() -> str:
    rows = []
    for schema in tool_schemas():
        summary = schema["description"].split(" Returns:")[0].split(". ")[0].rstrip(".") + "."
        required = ", ".join(f"`{r}`" for r in schema["parameters"]["required"]) or "—"
        rows.append(
            f"| `{schema['name']}` | {_REQUIRES.get(schema['name'], '?')} | {required} | {summary} |"
        )
    header = "| Tool | Requires | Required args | What it does |\n|---|---|---|---|\n"
    return header + "\n".join(rows)


def render(doc: str) -> str:
    """Return ``doc`` with both generated sections refreshed."""
    for marker, content in (
        ("ALLOWLIST", render_allowlist()),
        ("TOOLTABLE", render_tool_table()),
    ):
        start, end = f"<!-- BEGIN GENERATED {marker} -->", f"<!-- END GENERATED {marker} -->"
        a, b = doc.index(start) + len(start), doc.index(end)
        doc = doc[:a] + "\n" + content + "\n" + doc[b:]
    return doc


def main() -> int:
    current = DOC.read_text()
    updated = render(current)
    if updated != current:
        DOC.write_text(updated)
        print(f"Updated {DOC.name}")
    else:
        print(f"{DOC.name} already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

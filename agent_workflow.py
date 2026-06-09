"""
Semi-autonomous ML feature analysis agent powered by Claude Opus 4.7.

The agent receives a plain-English goal, decides which analyses to run,
interprets the results, and produces a structured report — without any
hard-coded analysis pipeline.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python agent_workflow.py --labels labels.xlsx --goal "Which features best separate TP from FP?"
    python agent_workflow.py --labels labels.xlsx --data-root /data --goal "Give me a full analysis"
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import anthropic

# Tools are implemented in mcp_server.py; the agent calls them via the Claude
# tool-use API, and this module executes them locally (same process).
import mcp_server as _srv

# ── Tool schemas (Claude reads these to decide what to call) ─────────────────

TOOLS: list[dict] = [
    {
        "name": "list_available_features",
        "description": (
            "List all registered per-sample feature transformations (e.g. rolling mean, "
            "z-score, first-difference) and aggregator names (e.g. mean, std, iqr) that "
            "can be requested during materialize_period."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "load_labels",
        "description": (
            "Load event labels from an Excel file. Must be called first. "
            "Returns a session_id required by every other tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "excel_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the Excel label file.",
                },
                "sheet": {
                    "description": "Sheet name or 0-based integer index.",
                },
                "data_root": {
                    "type": "string",
                    "description": "Root directory containing asset parquet files.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory where analysis outputs will be saved.",
                },
            },
            "required": ["excel_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "build_dataset",
        "description": (
            "Discover and lazily load the raw parquet time-series for every labeled event. "
            "Must be called after load_labels and before materialize_period."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "materialize_period",
        "description": (
            "Aggregate each event's full time-series into one summary row "
            "(period-aggregate table). This is the feature matrix used for all analyses. "
            "sources and aggregators narrow what gets computed; omit both for sensible defaults."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Raw signal column names to aggregate. Omit for all numeric columns.",
                },
                "aggregators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Aggregation stat names from list_available_features, "
                        "e.g. ['mean','std','min','max']. Omit for defaults."
                    ),
                },
                "feature_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Per-sample derived feature names from list_available_features "
                        "to include alongside the aggregations."
                    ),
                },
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_feature_importance",
        "description": (
            "Rank every feature by how well it discriminates between classes, using "
            "Random Forest MDI, permutation importance, ANOVA F-test, Kruskal-Wallis H, "
            "and mutual information. Returns a composite-ranked table of top-20 features."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "label_filter": {
                    "type": "object",
                    "description": (
                        "Restrict rows before analysis, e.g. {\"class\": [\"TP\", \"FP\"]}. "
                        "Omit to include all classes."
                    ),
                },
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_classifier_evaluation",
        "description": (
            "Train classifiers on the feature table and report accuracy, "
            "per-class precision/recall/F1, and confusion matrix. "
            "Use this to quantify how separable the classes actually are."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "label_filter": {"type": "object"},
                "run_rf": {
                    "type": "boolean",
                    "description": "Run Random Forest (default true).",
                },
                "run_lgb": {
                    "type": "boolean",
                    "description": "Run LightGBM (requires lightgbm installed).",
                },
                "run_xgb": {
                    "type": "boolean",
                    "description": "Run XGBoost (requires xgboost installed).",
                },
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_pairwise_separability",
        "description": (
            "For each pair of classes, rank features by Cliff's delta (effect size), "
            "KS statistic, and AUC. Best for pinpointing which features distinguish "
            "a specific pair (e.g. TP vs FP) rather than all classes together."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "pairs": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "description": (
                        "Class pairs to compare, e.g. [[\"TP\",\"FP\"],[\"TP\",\"TN\"]]. "
                        "Omit to compute all pairs."
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": "Return top N features per pair. Default 10.",
                },
                "label_filter": {"type": "object"},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_distribution_analysis",
        "description": (
            "Compute per-class feature distributions: quantiles (p5, p25, p50, p75, p95) "
            "and Kruskal-Wallis test (H-statistic, p-value). "
            "Good for understanding what the data looks like before deeper analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "label_filter": {"type": "object"},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    },
]

# ── Tool execution ────────────────────────────────────────────────────────────

_TOOL_FNS: dict[str, Any] = {
    "list_available_features": _srv.list_available_features,
    "load_labels": _srv.load_labels,
    "build_dataset": _srv.build_dataset,
    "materialize_period": _srv.materialize_period,
    "run_feature_importance": _srv.run_feature_importance,
    "run_classifier_evaluation": _srv.run_classifier_evaluation,
    "run_pairwise_separability": _srv.run_pairwise_separability,
    "run_distribution_analysis": _srv.run_distribution_analysis,
}


def _execute_tool(name: str, inputs: dict) -> str:
    fn = _TOOL_FNS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{name}'"})
    try:
        result = fn(**inputs)
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": type(exc).__name__ + ": " + str(exc)})


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an autonomous ML feature analysis agent. Your job is to answer a user's
analytical goal by orchestrating a set of data-loading and analysis tools.

Required call order:
  1. load_labels   — always first; gives you a session_id
  2. build_dataset — always second
  3. materialize_period — always third; creates the feature matrix
  4. analysis tools — in whatever order makes sense for the goal

Call tools ONE AT A TIME. Wait for each tool result before issuing the next
call. Never invent a session_id — read it from the load_labels result and
pass that exact string to every subsequent tool.

Guidelines:
- Start with load_labels → build_dataset → materialize_period before any analysis.
- If the user mentions specific signals or aggregations, reflect them in materialize_period.
- For "which features discriminate X from Y": use run_pairwise_separability + run_feature_importance.
- For "how separable are the classes?": use run_classifier_evaluation first.
- For "understand the data": start with run_distribution_analysis.
- Interpret numbers in plain language (e.g. "Cliff's delta of 0.6 is a large effect").
- Produce a concise final report: goal, key findings, top features, and recommended next steps.
- If a tool returns an error, explain it and try a different approach.
"""

# ── Agent loop ────────────────────────────────────────────────────────────────


def run_agent(labels_path: str, goal: str, data_root: str = "data") -> None:
    client = anthropic.Anthropic()

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Labels file: {labels_path}\n"
                f"Data root: {data_root}\n\n"
                f"Goal: {goal}"
            ),
        }
    ]

    print(f"\n{'=' * 60}")
    print(f"Goal: {goal}")
    print("=" * 60)

    while True:
        with client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=_SYSTEM,
            tools=TOOLS,
            messages=messages,
            cache_control={"type": "ephemeral"},
        ) as stream:
            response = stream.get_final_message()

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n{block.text}")

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            print(f"\n[Stopped: {response.stop_reason}]")
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            short_input = json.dumps(block.input, default=str)
            if len(short_input) > 140:
                short_input = short_input[:137] + "..."
            print(f"\n[→ {block.name}({short_input})]", flush=True)

            result_str = _execute_tool(block.name, block.input)
            result_data = json.loads(result_str)

            if "error" in result_data:
                print(f"   ERROR: {result_data['error']}")
            else:
                # Print a compact summary (omit large nested tables)
                brief = {
                    k: v
                    for k, v in result_data.items()
                    if k not in ("top_features", "summary", "results", "columns")
                }
                print(f"   {json.dumps(brief, default=str)[:240]}")

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                }
            )

        messages.append({"role": "user", "content": tool_results})


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semi-autonomous ML feature analysis powered by Claude Opus 4.7"
    )
    parser.add_argument("--labels", required=True, help="Path to the Excel label file")
    parser.add_argument(
        "--data-root", default="data", help="Root directory for asset parquet files"
    )
    parser.add_argument("--goal", required=True, help="Analysis goal in plain English")
    args = parser.parse_args()

    run_agent(labels_path=args.labels, goal=args.goal, data_root=args.data_root)


if __name__ == "__main__":
    main()

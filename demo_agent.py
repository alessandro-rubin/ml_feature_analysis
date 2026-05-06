"""
Demo for the Claude-powered AI agent in agent_workflow.py.

Generates synthetic data (reusing demo.py's generator), writes a labels Excel
file the agent can load, and runs the agent against a sample analytical goal.

Run:
    pip install -e ".[ai]"
    export ANTHROPIC_API_KEY=sk-ant-...
    python demo_agent.py

    # Override the goal:
    python demo_agent.py --goal "Rank all features by importance across all classes"

    # Use the Groq variant instead:
    python demo_agent.py --backend groq --goal "Which features separate TP from FP?"
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

import numpy as np

from demo import (
    DATA_ROOT,
    EVENT_LEN_HOURS,
    N_EVENTS_PER_CLASS,
    RANDOM_SEED,
    generate_synthetic_data,
)

LABELS_XLSX = Path("demo_labels.xlsx")
DEFAULT_GOAL = (
    "Identify the features that best separate TP from FP events. "
    "Quantify the effect sizes and recommend the strongest 3-5 features."
)


def ensure_demo_data_and_labels() -> Path:
    """Generate synthetic parquet data + labels.xlsx if either is missing."""
    if DATA_ROOT.exists() and any(DATA_ROOT.iterdir()) and LABELS_XLSX.exists():
        print(f"[setup] Reusing {DATA_ROOT}/ and {LABELS_XLSX}")
        return LABELS_XLSX

    print(f"[setup] Generating synthetic data under {DATA_ROOT}/ ...")
    rng = np.random.default_rng(RANDOM_SEED)
    labels = generate_synthetic_data(
        assets=["A01", "A02", "A03"],
        classes=["TP", "FP", "TN", "FN"],
        replacement_types=["bearing", "seal"],
        n_per_class=N_EVENTS_PER_CLASS,
        event_len_h=EVENT_LEN_HOURS,
        rng=rng,
    )
    print(f"[setup] Writing {LABELS_XLSX} ({labels.shape[0]} events)")
    labels.write_excel(LABELS_XLSX, worksheet="labels")
    return LABELS_XLSX


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI analysis agent on synthetic demo data")
    parser.add_argument("--goal", default=DEFAULT_GOAL, help="Plain-English analysis goal")
    parser.add_argument(
        "--backend",
        choices=["claude", "groq"],
        default="claude",
        help="Which agent driver to use (default: claude)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model id (groq backend only)",
    )
    args = parser.parse_args()

    if args.backend == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set. export it or pass --backend groq.")
    if args.backend == "groq" and not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set.")

    labels_path = ensure_demo_data_and_labels()

    if args.backend == "claude":
        from agent_workflow import run_agent

        run_agent(
            labels_path=str(labels_path),
            goal=args.goal,
            data_root=str(DATA_ROOT),
        )
    else:
        from agent_workflow_groq import DEFAULT_MODEL, run_agent

        run_agent(
            labels_path=str(labels_path),
            goal=args.goal,
            data_root=str(DATA_ROOT),
            model=args.model or DEFAULT_MODEL,
        )


if __name__ == "__main__":
    load_dotenv()
    main()

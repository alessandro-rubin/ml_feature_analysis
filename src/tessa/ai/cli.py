"""Command-line entry point for the TESSA agent.

Three modes:

* ``--offline`` runs a fixed tool script with no model at all. It exercises every
  tool, both registries, the expression compiler, the column budget and the
  truncation layer — for free, with no credential. Use it to verify an install or
  to debug a tool before spending tokens.
* ``--provider groq|openrouter|openai`` drives the tools with a hosted
  OpenAI-compatible model, and ``--provider ollama|lmstudio|llamacpp|vllm`` with
  a model on a local server, which is checked before the first turn.
* ``--provider anthropic`` drives them with Claude, and needs an API key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tessa.ai import tools
from tessa.ai.backend_openai import DEFAULT_MAX_TOKENS, PROVIDERS
from tessa.ai.loop import run_loop
from tessa.ai.prompts import opening_message, system_prompt

__all__ = ["main", "run_offline_script"]

_RULE = "─" * 72


def _print_payload(name: str, payload: dict, verbose: bool) -> None:
    if "error" in payload:
        print(f"  ✗ {name}: {payload['error'].get('type')}: {payload['error'].get('message')}")
        if payload["error"].get("hint"):
            print(f"    hint: {payload['error']['hint']}")
        return
    if verbose:
        print(json.dumps(payload, indent=2, default=str)[:2000])
    else:
        keys = {k: v for k, v in payload.items() if not isinstance(v, (list, dict))}
        print(f"  ✓ {name}: {json.dumps(keys, default=str)[:200]}")


def run_offline_script(data_root: str, labels_path: str | None, verbose: bool = False) -> int:
    """Drive the whole tool surface with a fixed script and no model."""
    failures = 0

    def step(title: str, payload: dict) -> dict:
        nonlocal failures
        print(f"\n{title}")
        _print_payload(title, payload, verbose)
        if "error" in payload:
            failures += 1
        return payload

    print(_RULE)
    print("  Offline tool drive — no model, no API key, no tokens spent")
    print(_RULE)

    created = step(
        "create_session",
        tools.create_session(data_root=data_root, labels_path=labels_path),
    )
    if "error" in created:
        return 1
    sid = created["session_id"]

    step("describe_data", tools.describe_data(sid))
    step("build_events", tools.build_events(sid))
    step("list_capabilities", tools.list_capabilities(sid))
    step("describe_analysis(separability)", tools.describe_analysis("separability"))

    step(
        "seed_builtin_features",
        tools.seed_builtin_features(sid, kinds=["roll_std", "diff1"], windows=[10]),
    )

    print("\n-- expression sandbox --")
    step(
        "preview_feature (valid)",
        tools.preview_feature(sid, 'pl.col("vibration").diff().abs()'),
    )
    rejected = tools.preview_feature(sid, '__import__("os").system("id")')
    print("\npreview_feature (malicious) — must be rejected")
    if "error" in rejected:
        print(f"  ✓ rejected: {rejected['error']['message'][:90]}")
    else:
        print("  ✗ SANDBOX FAILURE: a malicious expression was accepted")
        failures += 1

    step(
        "create_feature",
        tools.create_feature(
            sid,
            "vibration__jerk",
            'pl.col("vibration").diff().abs()',
            rationale="offline script: sample-to-sample movement, not level",
        ),
    )
    step(
        "create_aggregator",
        tools.create_aggregator(
            sid,
            "spread",
            "pl.col(c).quantile(0.9) - pl.col(c).quantile(0.1)",
            rationale="offline script: robust spread",
        ),
    )

    print("\n-- materialization --")
    step("materialize (baseline)", tools.materialize(sid, table_name="baseline"))
    budget = tools.materialize(sid, feature_names="__all__", max_columns=2)
    print("\nmaterialize (over budget) — must be refused")
    if "error" in budget and budget["error"]["type"] == "BudgetError":
        print(f"  ✓ refused: {budget['error']['message'][:110]}")
    else:
        print("  ✗ the column budget did not fire")
        failures += 1
    step(
        "materialize (augmented)",
        tools.materialize(sid, feature_names=["vibration__jerk"], table_name="augmented"),
    )

    print("\n-- analysis --")
    step("characterize_classes", tools.characterize_classes(sid, table="augmented"))
    step("run_analysis(importance)", tools.run_analysis(sid, "importance", table="augmented"))
    step(
        "get_result(importance.table)",
        tools.get_result(sid, "importance", frame="table", top_n=5, table="augmented"),
    )
    step(
        "score_features",
        tools.score_features(sid, table="augmented", baseline_table="baseline"),
    )
    step("feature_ledger", tools.feature_ledger(sid))

    print("\n-- persistence --")
    step("save_run", tools.save_run(sid, name="offline_drive", table="augmented"))
    step("write_report", tools.write_report(sid, table="augmented"))

    print(f"\n{_RULE}")
    if failures:
        print(f"  {failures} step(s) failed")
    else:
        print("  All steps completed.")
    print(_RULE)
    return 1 if failures else 0


def _build_backend(args: argparse.Namespace) -> tuple[Any, int | None]:
    """The backend, and the context window to keep the conversation inside."""
    if args.provider == "anthropic":
        from tessa.ai.backend_anthropic import build_backend

        backend = build_backend(model=args.model or "claude-opus-5", effort=args.effort)
        return backend, args.context_length
    from tessa.ai.backend_openai import build_backend

    model, context = args.model, args.context_length
    if PROVIDERS[args.provider].local and not args.skip_check:
        from tessa.ai.local import probe

        print(f"Checking the {args.provider} server ...")
        found = probe(args.provider, base_url=args.base_url, model=model)
        model = found.model
        if context is None and found.context_length:
            context = found.context_length
            print(f"  context window: {context} tokens ({found.context_source})")
        for warning in found.warnings:
            print(f"  warning: {warning}")

    max_tokens = args.max_tokens
    if context:
        # A reply may fill at most a quarter of the window; the rest is the conversation's.
        max_tokens = min(max_tokens or DEFAULT_MAX_TOKENS, context // 4)
    backend = build_backend(
        provider=args.provider,
        model=model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=max_tokens,
        timeout=args.timeout,
        echo_reasoning=False if args.no_reasoning_echo else None,
    )
    return backend, context


def _on_event(verbose: bool):
    def emit(kind: str, payload: dict) -> None:
        if kind == "text" and payload.get("text"):
            print(f"\n{payload['text']}\n")
        elif kind == "tool":
            args = json.dumps(payload.get("arguments", {}), default=str)
            print(f"  → {payload['name']}({args[:140]})")
        elif kind == "result":
            mark = "✗" if payload.get("failed") else "✓"
            body = payload.get("payload", "")
            print(f"    {mark} {body[:200] if verbose else body[:120]}")
        elif kind == "cap":
            print(f"  [loop stopped: {payload}]")
        elif kind == "error":
            print(f"  [error: {payload.get('message')}]")
        elif kind == "warning":
            print(f"  [warning: {payload.get('message')}]")
        elif kind == "compact":
            print(
                f"  [context: elided {payload['elided']} older tool result(s) and the "
                f"reasoning of {payload['reasoning_dropped']} older turn(s); about "
                f"{payload['estimated_tokens']} of {payload['budget_tokens']} tokens used]"
            )
        elif kind == "reasoning" and verbose:
            text = " ".join(payload.get("text", "").split())
            print(f"  (reasoning: {text[:300]}{' ...' if len(text) > 300 else ''})")

    return emit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tessa-agent",
        description="Explore event classes with an AI agent that can invent features.",
    )
    parser.add_argument("--data-root", required=True, help="Directory of per-asset parquet files")
    parser.add_argument(
        "--make-demo",
        action="store_true",
        help="Generate synthetic data and labels under --data-root first.",
    )
    parser.add_argument("--labels", help="Label table (.xlsx, .parquet or .csv)")
    parser.add_argument("--goal", help="What you want answered, in plain English")
    parser.add_argument(
        "--provider",
        default="groq",
        choices=[*PROVIDERS, "anthropic"],
        help="Which model provider to drive the tools with (default: groq). "
        "ollama, lmstudio, llamacpp and vllm are local servers; local is an alias "
        "of ollama.",
    )
    parser.add_argument("--model", help="Override the provider's default model")
    parser.add_argument("--base-url", help="Override the provider's base URL")
    parser.add_argument("--effort", default="high", help="Anthropic effort level")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument(
        "--context-length",
        type=int,
        help="The model's context window in tokens. Older reasoning and tool results "
        "are dropped to stay inside it. Local servers report it themselves when they "
        "can. Leave it off for Claude, whose newer models reject edited history.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help=f"Cap on each reply (default {DEFAULT_MAX_TOKENS}, at most a quarter of "
        "the context window). OpenAI-compatible providers only.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (default 0 for hosted providers, 0.6 for local "
        "servers). OpenAI-compatible providers only.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Seconds one model call may take (default 1800 for local servers).",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Do not check the local server before the run.",
    )
    parser.add_argument(
        "--no-reasoning-echo",
        action="store_true",
        help="Do not send a reasoning model's reasoning back with its turn (local "
        "servers do by default). For a server that rejects the field.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run a fixed tool script with no model. Verifies the install for free.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.make_demo:
        from tessa.ai.demo_data import generate

        root, labels = generate(args.data_root)
        args.labels = args.labels or str(labels)
        print(f"Generated synthetic data under {root} with labels at {labels}")

    if not Path(args.data_root).exists():
        print(f"data-root not found: {args.data_root}", file=sys.stderr)
        return 2

    if args.offline:
        return run_offline_script(args.data_root, args.labels, verbose=args.verbose)

    if not args.goal:
        parser.error("--goal is required unless --offline is set")

    try:
        backend, context = _build_backend(args)
    except RuntimeError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    print(_RULE)
    print(f"  provider: {backend.name}   model: {backend.model}")
    if context:
        print(f"  context window: {context} tokens")
    print(f"  goal: {args.goal}")
    print(_RULE)

    result = run_loop(
        backend,
        system_prompt(),
        opening_message(args.goal, args.data_root, args.labels),
        max_turns=args.max_turns,
        context_tokens=context,
        on_event=_on_event(args.verbose),
    )

    print(f"\n{_RULE}")
    print(
        f"  {result.turns} turns, {result.tool_calls} tool calls, stopped: {result.stopped_because}"
    )
    if result.error:
        print(f"  error: {result.error}")
    if result.usage:
        print(f"  usage: {json.dumps(result.usage)}")
    print(_RULE)
    return 1 if result.error else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""
Groq-backed variant of agent_workflow.py.

Same tools, same system prompt — driven by an open-weight model on Groq via
its OpenAI-compatible endpoint instead of Claude.

Usage:
    pip install openai
    export GROQ_API_KEY=gsk_...
    python agent_workflow_groq.py --labels labels.xlsx --goal "Which features best separate TP from FP?"
"""
from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI

from agent_workflow import TOOLS as _ANTHROPIC_TOOLS
from agent_workflow import _SYSTEM, _execute_tool

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _to_openai_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


TOOLS = [_to_openai_tool(t) for t in _ANTHROPIC_TOOLS]


def run_agent(
    labels_path: str,
    goal: str,
    data_root: str = "data",
    model: str = DEFAULT_MODEL,
    max_turns: int = 25,
) -> None:
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_BASE_URL,
    )

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Labels file: {labels_path}\n"
                f"Data root: {data_root}\n\n"
                f"Goal: {goal}"
            ),
        },
    ]

    print(f"\n{'=' * 60}")
    print(f"Goal: {goal}  [model: {model}]")
    print("=" * 60)

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            max_tokens=8000,
        )
        msg = response.choices[0].message

        if msg.content and msg.content.strip():
            print(f"\n{msg.content}")

        # Append the assistant turn verbatim (must include any tool_calls).
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])],
            }
        )

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                inputs = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                result_str = json.dumps({"error": f"Invalid tool arguments JSON: {exc}"})
                inputs = None
            else:
                short = json.dumps(inputs, default=str)
                if len(short) > 140:
                    short = short[:137] + "..."
                print(f"\n[→ {name}({short})]", flush=True)
                result_str = _execute_tool(name, inputs)

            result_data = json.loads(result_str)
            if "error" in result_data:
                print(f"   ERROR: {result_data['error']}")
            else:
                brief = {
                    k: v
                    for k, v in result_data.items()
                    if k not in ("top_features", "summary", "results", "columns")
                }
                print(f"   {json.dumps(brief, default=str)[:240]}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                }
            )
    else:
        print(f"\n[Stopped: hit max_turns={max_turns}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML feature analysis agent driven by an open-weight model on Groq"
    )
    parser.add_argument("--labels", required=True, help="Path to the Excel label file")
    parser.add_argument("--data-root", default="data", help="Root directory for asset parquet files")
    parser.add_argument("--goal", required=True, help="Analysis goal in plain English")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Groq model id (default: {DEFAULT_MODEL}). Try openai/gpt-oss-120b for stronger tool use.",
    )
    args = parser.parse_args()

    run_agent(
        labels_path=args.labels,
        goal=args.goal,
        data_root=args.data_root,
        model=args.model,
    )


if __name__ == "__main__":
    main()

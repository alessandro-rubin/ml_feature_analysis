# AI Integration Guide

This document explains how to use the two AI integration files shipped with
this project, the design rationale behind them, and how they fit together.

---

## Overview

The project exposes its analysis pipeline in two complementary ways:

| File | Role |
|---|---|
| `mcp_server.py` | Wraps the pipeline as **MCP tools** — callable by any MCP-compatible host (Claude Desktop, Claude Code, custom MCP clients) |
| `agent_workflow.py` | A **Claude Opus 4.7 agent** that receives a plain-English goal, decides which analyses to run, and produces an interpreted report |

They share the same underlying implementation: `mcp_server.py` defines the
tool functions, `agent_workflow.py` imports and calls them directly while using
the Claude API for orchestration logic.

---

## Installation

```bash
# Core package + AI dependencies
pip install -e ".[ai]"

# With boosting models (LightGBM, XGBoost)
pip install -e ".[ai,boosting]"
```

The `[ai]` extra adds `anthropic>=0.50.0` and `mcp>=1.0.0`.

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## MCP Server

### What it is

`mcp_server.py` is a [Model Context Protocol](https://modelcontextprotocol.io/)
server. MCP is a standard protocol that lets AI models discover and call tools
hosted by an external process. Any MCP-compatible client — Claude Desktop,
Claude Code's `/mcp` command, or a custom Python client — can connect to this
server and use its tools directly, without any changes to the analysis library.

### Tools exposed

| Tool | Inputs | Returns |
|---|---|---|
| `list_available_features` | — | All registered feature and aggregator names |
| `load_labels` | `excel_path`, `sheet`, `data_root`, `output_dir` | `session_id`, event count, asset list, class list |
| `build_dataset` | `session_id` | Event count per asset |
| `materialize_period` | `session_id`, `sources`, `aggregators`, `feature_names` | Table shape, column list, class list |
| `run_feature_importance` | `session_id`, `label_filter` | Top-20 features by composite score |
| `run_classifier_evaluation` | `session_id`, `label_filter`, `run_rf/lgb/xgb` | Accuracy, classification report, confusion matrix |
| `run_pairwise_separability` | `session_id`, `pairs`, `top_n`, `label_filter` | Cliff's delta, KS stat, AUC per feature per class pair |
| `run_distribution_analysis` | `session_id`, `label_filter` | Quantiles and KW p-values per feature |

### Required call order

The tools are stateful and must be called in order within a session:

```
load_labels  →  build_dataset  →  materialize_period  →  [any analysis tool(s)]
```

Each `load_labels` call returns a unique `session_id`. Pass that ID to every
subsequent call. Multiple independent sessions can run concurrently.

### Running as a standalone MCP server

```bash
python mcp_server.py
```

This starts the server on stdio (the MCP default transport). To connect from
Claude Code:

```jsonc
// .claude/settings.json
{
  "mcpServers": {
    "ml-feature-analysis": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/ml_feature_analysis"
    }
  }
}
```

Once connected, Claude Code can call any of the tools above directly during a
conversation.

### Using the functions directly in Python

Because the tool functions are plain Python functions decorated with
`@mcp.tool()`, they can also be imported and called without the MCP protocol:

```python
import mcp_server as srv

info = srv.load_labels("labels.xlsx", data_root="data")
sid = info["session_id"]

srv.build_dataset(sid)
srv.materialize_period(sid, aggregators=["mean", "std", "min", "max"])

result = srv.run_pairwise_separability(sid, pairs=[["TP", "FP"]], top_n=15)
print(result["results"]["TP_vs_FP"][:5])
```

### Session state and temp files

Each session stores:
- Labels as a parquet file in a `tempfile` directory (`/tmp/ml_mcp_*/`)
- Event LazyFrames in-memory (they are Polars lazy query plans, not data)
- The materialized period table as a parquet file

State lives in the server process for the lifetime of that process. Restarting
the server clears all sessions.

---

## Agent Workflow

### What it is

`agent_workflow.py` implements a **semi-autonomous analysis agent** using
Claude Opus 4.7 with tool use. You give it a plain-English goal; it decides
which tools to call and in what order, runs them, interprets the numbers, and
writes a report — without a hard-coded analysis pipeline.

### How to run it

```bash
# Basic usage
python agent_workflow.py \
  --labels labels.xlsx \
  --goal "Which features best separate TP from FP events?"

# Specify data root
python agent_workflow.py \
  --labels /data/labels/my_project.xlsx \
  --data-root /data/assets \
  --goal "Give me a complete feature analysis across all classes"

# Focused analysis
python agent_workflow.py \
  --labels labels.xlsx \
  --goal "Identify the 5 most discriminative features for the FP vs TN distinction \
          using only pressure and temperature signals"
```

### What the agent does

1. Reads your goal and the label file path
2. Calls `load_labels` → `build_dataset` → `materialize_period` (always, in order)
3. Selects analysis tools based on the goal — it will reason about which
   combination of importance ranking, pairwise comparison, classification, and
   distribution analysis is most relevant
4. Calls those tools with appropriate parameters (e.g. `label_filter` to narrow
   to relevant classes, specific `pairs` for pairwise analysis)
5. Interprets the output in plain language — effect sizes, p-values, accuracy
   numbers — and writes a final report with key findings and next steps

### Example terminal output

```
============================================================
Goal: Which features best separate TP from FP events?
============================================================

I'll start by loading the labels and building the feature matrix.

[→ load_labels({"excel_path": "labels.xlsx", "data_root": "data"})]
   {"session_id": "a3f7c2b1", "n_events": 412, "classes": ["FP", "TN", "TP"]}

[→ build_dataset({"session_id": "a3f7c2b1"})]
   {"n_events": 412, "events_per_asset": {"pump_01": 138, "pump_02": 274}}

[→ materialize_period({"session_id": "a3f7c2b1"})]
   {"shape": [412, 187], "n_feature_columns": 183, "classes": ["FP", "TN", "TP"]}

Since the goal is specifically TP vs FP, I'll run pairwise separability
focused on that pair, then confirm with feature importance.

[→ run_pairwise_separability({"session_id": "a3f7c2b1", "pairs": [["TP", "FP"]], "top_n": 15})]
   {"pairs_analyzed": ["TP_vs_FP"]}

[→ run_feature_importance({"session_id": "a3f7c2b1", "label_filter": {"class": ["TP", "FP"]}})]
   {"n_features_evaluated": 183, "class_names": ["FP", "TP"]}

## Analysis Report: TP vs FP Separability

**Top discriminating features (TP vs FP)**

The three strongest features by Cliff's delta are:

1. `pressure__iqr` — δ = 0.72 (large effect). TP events show much wider
   pressure swings during the event window.
2. `flow_rate__roll_std_30` — δ = 0.61 (large effect). Rolling volatility
   of flow rate is consistently higher in true positives.
3. `temperature__zscore_60` — δ = 0.54 (medium-large). Standardized
   temperature deviation distinguishes TP from FP with good reliability.

**Recommendation**

These three features alone likely capture most of the TP/FP signal.
Next step: run `run_classifier_evaluation` with `label_filter: {class: [TP, FP]}`
to quantify how accurately a model can separate them.
```

### Embedding the agent in your own code

```python
from agent_workflow import run_agent

run_agent(
    labels_path="labels.xlsx",
    goal="Rank all features by importance across all classes",
    data_root="data",
)
```

To capture the output programmatically instead of printing it, redirect stdout
or modify `run_agent` to collect `block.text` into a list.

---

## Design Rationale

### Why MCP?

The analysis library has a clean, modular Python API but it requires knowing
the call order and the right combination of parameters. MCP makes the pipeline
discoverable by any AI model without needing to understand the internals —
the tool descriptions teach the model what each step does and when to use it.

The secondary benefit is deployment flexibility. The same `mcp_server.py` can
be plugged into Claude Desktop for interactive exploration, connected to Claude
Code for in-editor analysis, or called via a Python client in a notebook —
all without changing the analysis code.

### Why is the agent workflow separate from the MCP server?

The MCP server handles **execution**: each tool is a thin wrapper around one
pipeline step, stateless with respect to analysis strategy.

The agent workflow handles **reasoning**: which tools to call, with what
parameters, in what order, for a given goal. Keeping them separate means:

- The MCP server works without the Anthropic SDK installed
- The analysis functions can be tested and called directly without an LLM
- The agent's system prompt and tool descriptions can be tuned independently
  of the underlying analysis logic

### Why does `agent_workflow.py` import `mcp_server` directly instead of connecting via the MCP protocol?

For single-process use, going through stdio MCP adds latency and complexity
with no benefit. The functions are plain Python — importing them is the right
call. In a production deployment (e.g. a web service where the MCP server
runs separately), you'd replace `_execute_tool` with an async MCP client call.
The tool schemas in `TOOLS` are identical either way.

### Why session IDs instead of passing DataFrames?

Tool results must be JSON-serializable for the Claude API. A 400-event period
table with 180+ feature columns is ~10 MB of JSON — far too large to pass back
and forth through the LLM context. Instead, each tool serializes state to
parquet in a temp directory and returns only a lightweight summary (shape,
column names, class list). Subsequent tools reload the data from disk.

This also means the LLM context stays small and cacheable regardless of
dataset size.

### Why Cliff's delta and KS test for pairwise separability?

Both are non-parametric and distribution-free, which suits time-series
aggregate features that are rarely Gaussian. Cliff's delta gives a normalized
effect size (−1 to +1) that is easier to interpret than a raw difference in
means. The KS statistic captures full distributional differences, not just
location shifts. Together they cover both "where the distributions are
centered" and "how differently shaped they are."

### Why `label_filter` on analysis tools?

Many real datasets are imbalanced across classes, and a three-way or four-way
classification problem obscures pairwise structure. `label_filter` lets the
agent (or a direct caller) focus a specific analysis on the class subset that
matters for a given goal, without materializing a separate feature table.

### Why adaptive thinking on Claude Opus 4.7?

The agent needs to reason about: what the data looks like from the label
summary, which analyses are most relevant for the goal, how to interpret
statistical output (effect sizes, p-values, confusion matrices), and what to
recommend next. That's genuine multi-step reasoning across heterogeneous
outputs. Adaptive thinking lets the model decide when to reason deeply (e.g.
interpreting a complex confusion matrix) vs. act quickly (e.g. calling
`build_dataset` which has no real decisions to make).

### Why prompt caching on the system prompt?

The system prompt and tool schemas are large (~3 KB) and identical across
every turn of the agent loop. `cache_control: {"type": "ephemeral"}` on
`messages.create` caches this stable prefix, so only the growing conversation
history (tool calls and results) is billed at full price on each turn.

---

## Extending the integration

**Add a new analysis tool:**
1. Implement the function in `mcp_server.py` and decorate it with `@mcp.tool()`
2. Add the corresponding JSON schema entry to `TOOLS` in `agent_workflow.py`
3. Add a line to the agent's system prompt describing when to use it

**Add a new label source (e.g. CSV, database):**
Implement the `LabelSource` protocol from `ml_analysis.labels` and swap it
into `load_labels` — no other changes needed.

**Connect to Claude Desktop:**
Add the MCP server to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) using the same JSON structure shown in the standalone server section above.

**Run as an async service:**
Replace the synchronous `mcp.run()` with `mcp.run(transport="sse")` to expose
the server over HTTP/SSE instead of stdio, then connect any SSE-capable MCP
client.

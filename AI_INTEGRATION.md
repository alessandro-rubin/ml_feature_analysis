# AI Integration

TESSA's analysis pipeline can be driven by a language model that decides what to
measure, reads the evidence, and — the point of this layer — **invents new
features** when the stock aggregates cannot see what distinguishes the classes.

Everything lives in `src/tessa/ai/`. The tool functions themselves import no
model SDK, so one implementation serves every driver below, and you can call
them straight from Python with no model at all.

---

## Three ways to drive it

| Route | Model | Needs |
|---|---|---|
| **MCP client** (Claude Desktop, Claude Code, Cline, Zed, …) | whatever the client uses | no API key of your own |
| **`tessa-agent --provider groq`** (default) | open-weight, via Groq | `GROQ_API_KEY` |
| **`tessa-agent --provider openrouter`** | any frontier model, Claude included | `OPENROUTER_API_KEY` |
| **`tessa-agent --provider anthropic`** | Claude | `ANTHROPIC_API_KEY` |
| **`tessa-agent --offline`** | none | nothing |

MCP is a protocol, not a vendor. Running `tessa-mcp` and connecting Claude
Desktop means *that client's* model drives the tools — so a Claude subscription
is enough and no API key is involved.

### Install

```bash
uv sync --extra ai              # OpenAI-compatible drivers + MCP server
uv sync --extra ai-anthropic    # adds the Claude driver
uv sync --all-extras            # everything
```

> **Note on the extra names.** Before v0.2.0 the removed `[ai]` extra meant the
> Anthropic path. The extras now split along credential lines instead: `ai` is
> everything that works without an Anthropic key, `ai-anthropic` adds the rest.

---

## Quick start

Generate a synthetic dataset and drive every tool with no model and no tokens:

```bash
uv run tessa-agent --data-root ./demo_data --make-demo --offline
```

Then with a model:

```bash
export GROQ_API_KEY=gsk_...
uv run tessa-agent \
  --data-root ./demo_data --labels ./demo_data/labels.parquet \
  --goal "What separates calm from jittery events? Invent features if the stock aggregates are not enough." \
  --provider groq
```

Or from Python, with no model in the loop at all:

```python
from tessa.ai import tools

sid = tools.create_session(data_root="data", labels_path="labels.xlsx")["session_id"]
tools.build_events(sid)
tools.create_feature(sid, "vibration__jerk", 'pl.col("vibration").diff().abs()',
                     rationale="the classes differ in ordering, not level")
tools.materialize(sid, feature_names=["vibration__jerk"])
tools.run_analysis(sid, "separability")
```

### MCP client config

```jsonc
{"mcpServers": {"tessa": {"command": "tessa-mcp"}}}
```

---

## The tools

Call order is enforced in code: `create_session` → `build_events` →
`materialize` → analyses. Getting it wrong returns an error naming the next call
rather than a traceback.

<!-- BEGIN GENERATED TOOLTABLE -->
| Tool | Requires | Required args | What it does |
|---|---|---|---|
| `create_session` | — | `data_root` | Start an analysis session and load the event labels. |
| `describe_data` | session | `session_id` | Describe the raw data: assets, channel names and dtypes, and class balance. |
| `build_events` | session | `session_id` | Build the lazy per-event dataset from the labels. |
| `list_capabilities` | session | `session_id` | List what you can currently use: features, aggregators, analyses, and the DSL. |
| `describe_analysis` | — | `name` | Full parameter list and result frames for one analysis. |
| `seed_builtin_features` | events | `session_id` | Register stock features (rolling mean/std, first difference, z-score) per channel. |
| `preview_feature` | events | `session_id`, `expression` | Compile an expression and try it on one event, WITHOUT registering it. |
| `create_feature` | events | `session_id`, `name`, `expression`, `rationale` | Register a new per-sample feature from a Polars expression. |
| `create_aggregator` | session | `session_id`, `name`, `expression`, `rationale` | Register a new aggregator, which collapses a column to one value per event. |
| `drop_feature` | session | `session_id`, `name` | Remove a feature or aggregator that did not earn its place. |
| `materialize` | events | `session_id` | Build the analysis table: one row per event (or per window). |
| `set_active_table` | table | `session_id`, `table_name` | Choose which materialized table subsequent analyses run on. |
| `run_analysis` | table | `session_id`, `name` | Run one analysis on a materialized table. |
| `get_result` | analysis run | `session_id`, `name` | Page into a frame of an already-computed analysis. |
| `characterize_classes` | table | `session_id` | Run the standard first-pass battery and return one digest. |
| `score_features` | table | `session_id` | Measure whether the features you invented actually helped. |
| `feature_ledger` | session | `session_id` | The record of every feature invented this session and how it fared. |
| `confirm_on_holdout` | table + holdout | `session_id` | Re-test on the held-out assets. |
| `save_run` | analysis run | `session_id` | Save the completed analyses, with the feature ledger in the manifest. |
| `write_report` | analysis run | `session_id` | Write a self-contained static HTML report of the analyses run so far. |
<!-- END GENERATED TOOLTABLE -->

Every tool returns a JSON-serializable dict, and failures come back as
`{"error": {"type", "message", "hint", ...}}`. `run_analysis` returns a headline
only — scalars plus the shape of each result frame; use `get_result` to page into
the rows you want. This keeps tool payloads small, which matters because every
one of them becomes permanent context.

---

## Inventing features

The model writes a Polars expression as a string. It is checked against an
allowlist, compiled, and tried on a single event before anything is registered.

```python
tools.preview_feature(sid, 'pl.col("vibration").rolling_std(50)')   # try it
tools.create_feature(sid, "vibration__vol",                          # keep it
                     'pl.col("vibration").rolling_std(50)',
                     rationale="volatility, not level, separates these classes")
```

Registration targets a **session-scoped registry**, seeded as a copy of the
process-wide one. An invented feature can never leak into the global registry or
into another session. Because every materializer already accepts
`feature_registry=`, that feature then flows through the entire pipeline
unchanged.

Aggregators work the same way, referring to the column being aggregated as `c`:

```python
tools.create_aggregator(sid, "spread", "pl.col(c).quantile(0.9) - pl.col(c).quantile(0.1)",
                        rationale="robust spread, insensitive to outliers")
```

### The expression language

`list_capabilities(session_id)` returns this list at runtime; the snapshot below
is generated from the same source.

<!-- BEGIN GENERATED ALLOWLIST -->
**Entry points (pl.…)**

  `col` `corr` `cov` `first` `int_range` `last`
  `len` `lit` `max_horizontal` `mean_horizontal` `min_horizontal` `sum_horizontal`
  `when`

**Cast targets**

  `Boolean` `Float32` `Float64` `Int32` `Int64` `UInt32`

**Expression methods**

  `abs` `all` `and_` `any` `arctan` `arg_max`
  `arg_min` `backward_fill` `cast` `cbrt` `ceil` `clip`
  `cos` `count` `cum_count` `cum_max` `cum_min` `cum_prod`
  `cum_sum` `diff` `entropy` `eq` `ewm_mean` `ewm_std`
  `ewm_var` `exp` `fill_nan` `fill_null` `floor` `forward_fill`
  `ge` `gt` `interpolate` `is_between` `is_finite` `is_in`
  `is_infinite` `is_nan` `is_not_null` `is_null` `kurtosis` `le`
  `log` `log10` `log1p` `lt` `max` `mean`
  `median` `min` `mode` `n_unique` `ne` `not_`
  `null_count` `or_` `otherwise` `over` `pct_change` `pow`
  `product` `quantile` `rank` `reverse` `rolling_max` `rolling_mean`
  `rolling_median` `rolling_min` `rolling_quantile` `rolling_skew` `rolling_std` `rolling_sum`
  `rolling_var` `round` `shift` `sign` `sin` `skew`
  `sort` `sqrt` `std` `sum` `tan` `then`
  `var`

**dt namespace**

  `epoch` `hour` `minute` `ordinal_day` `second` `total_seconds`
  `weekday`

**Rejected by name** (beyond everything not listed above)

  `alias` `apply` `head` `map_batches` `map_elements` `rolling_map` `slice` `tail`

<!-- END GENERATED ALLOWLIST -->

`alias()` is rejected because the feature's name comes from the `name` argument.
`map_elements` / `map_batches` / `apply` are rejected because they take arbitrary
Python callbacks, which is the whole thing the gate exists to prevent.

### Safety: what the sandbox is and is not

**The AST allowlist is the security boundary.** Expressions are parsed with
`ast.parse(mode="eval")` — which alone rules out imports, assignments and
statements — then walked against an allowlist of node types, identifiers and
attribute names, and only then evaluated with `__builtins__` stripped.

Stripping `__builtins__` is **defence in depth only**. On its own it is trivially
bypassed through attribute traversal (`().__class__.__bases__[0].__subclasses__()`)
or subscripting; both are blocked by rejecting `ast.Subscript` outright and by
allowlisting every attribute name. Widening either check reopens those escapes.

What it does **not** defend against is expressions that are merely ruinous:
`pl.col("x").rolling_quantile(0.5, window_size=10_000_000)` is perfectly legal.
Cost is bounded by previewing on one event and by the column budget, not by the
compiler.

Finally: the process runs with your filesystem privileges and the tools accept
model-supplied paths. Do not expose `tessa-mcp` over an untrusted transport.

### The column budget

`to_period` produces `(sources + features) × aggregators` columns, so a handful
of invented features multiplies quickly. `materialize` therefore includes **no**
derived features by default — pass `feature_names` explicitly, or the string
`"__all__"` — and refuses to build a table wider than `max_columns` (default 400),
showing the arithmetic:

```
That would build about 520 feature columns ((8 sources + 122 features) x 4
aggregators), over the limit of 400.
```

---

## The exploration loop

The playbook lives in the system prompt (`tessa/ai/prompts.py`) rather than in an
`explore_classes` tool, because choosing *which* weakness to attack and *what*
would explain it is judgement — a single tool that did it all would just be the
rigid pipeline this layer exists to replace. Only the two mechanical steps are
tools, so that they are computed identically every round:

- **`characterize_classes`** — the first-pass battery (distributions, importance,
  pairwise, separability) collapsed into one digest, including the *weakest class
  pair*: where to aim next.
- **`score_features`** — the keep/drop evidence: separability on the new table
  against the baseline, plus each feature's importance rank and best pairwise AUC.

Every invented feature is recorded in a **ledger** with its expression, rationale,
preview statistics and measured contribution. `save_run` writes the ledger into
the run manifest, so a saved run records exactly which expressions produced its
columns — without that, an AI-authored column is unexplainable after the process
exits.

### Selection bias — read this before believing a result

A model that invents features, measures them on the same data, and keeps the
winners is doing feature selection on its own evaluation set. Gains reported this
way are optimistic. Three things push back:

1. `separability` is permutation-tested, so judge `perm_p_value`, not raw accuracy.
2. Cross-validation folds are **grouped by asset**, so scores describe unseen
   *assets*, not unseen events, and a feature that memorizes an asset is punished.
3. `holdout_assets` on `create_session` hides assets from the entire exploration,
   and `confirm_on_holdout` re-tests on them once at the end. **This is the only
   one that actually resolves the problem**; the first two only reduce it.

```python
tools.create_session(data_root="data", labels_path="labels.xlsx",
                     holdout_assets=["A7", "A8"])
...
tools.confirm_on_holdout(sid)   # once, at the end
```

Note that a holdout of a single asset yields an ungrouped CV estimate, and the
library will warn accordingly.

---

## Architecture

```
tools.py ── plain functions, no SDK ──┬── mcp_server.py   (any MCP client)
                                      └── schemas.py ──┬── backend_openai.py
                                                       └── backend_anthropic.py
                                                            └── loop.py
```

One schema source, two wire formats: `schemas.py` derives JSON Schema from each
tool's signature, type hints and docstring, then adapts it per provider. One loop
serves both backends; each backend only translates message formats. The previous
version of this layer hand-wrote Anthropic schemas, re-serialized them for the
OpenAI path, and duplicated the loop — and the copies drifted apart.

| Module | Role |
|---|---|
| `session.py` | Per-run state; cloned registries; the feature ledger |
| `expressions.py` | The AST-allowlist expression compiler |
| `render.py` | JSON/truncation boundary (NaN → `null`, arrays summarized) |
| `catalog.py` | Analysis catalogue by dataclass introspection + param coercion |
| `tools.py` | The 20 tool functions |
| `schemas.py` | Schema derivation + per-provider adapters |
| `prompts.py` | System prompt and exploration playbook |
| `loop.py` | Provider-neutral agent loop |
| `backend_openai.py` | Groq / OpenRouter / OpenAI / local |
| `backend_anthropic.py` | Claude (`claude-opus-5`) |
| `cli.py` | `tessa-agent`, including `--offline` |
| `mcp_server.py` | `tessa-mcp` |
| `demo_data.py` | Synthetic data where aggregates are provably blind |

### Loop behaviour

- Turn cap (default 40). On exceed, one final tool-free turn is requested so a
  capped run still produces a written report.
- Consecutive-failure cap (default 5), the guard against a weaker model looping
  on a call it cannot get right.
- All tool results from one assistant turn are returned together; splitting them
  teaches a model to stop making parallel calls.
- Arguments are always parsed as JSON, never string-matched.
- Payloads are truncated to 20 KB before entering context.

---

## Testing

Everything above is tested with no credentials and no network — the sandbox
against real escape attempts, session isolation, the tool surface end to end on
synthetic data, schema derivation, both backend adapters against fake clients,
and the loop against a stub backend.

```bash
uv run pytest -q                    # `live` tests are deselected by default
uv run tessa-agent --data-root ./demo_data --make-demo --offline
```

The offline drive is the fastest way to verify an install: it exercises every
tool, both registries, the compiler, the budget guard and the persistence layer,
and asserts that a malicious expression is rejected.

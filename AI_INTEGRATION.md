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
| **`tessa-agent --provider ollama`** (or `lmstudio`, `llamacpp`, `vllm`) | open-weight, on your own machine | a local server; see [Running with a local model](#running-with-a-local-model) |
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

## Running with a local model

A model on your own machine drives the same tools through the same loop. There
is no API key and no per-token cost, and nothing leaves the machine: the model
sees your channel names, class balance and analysis results, and with a local
server none of that is sent to a provider.

The price is reliability and speed. A run is 20-40 turns, each a tool call that
must be valid JSON against a 20-tool schema, and that is harder for an 8-30B
model than for a frontier one. Expect more malformed calls, shallower feature
ideas and slower turns, and use the largest model your memory allows.

### What you need

| Model (Ollama tag) | Memory at a 32k window | Notes |
|---|---|---|
| `qwen3:8b` | about 10 GB | the smallest that is usable; more malformed calls |
| `qwen3:14b` | about 15 GB | **the default**: good tool calling for its size |
| `gpt-oss:20b` | about 16 GB | strong tool use; a reasoning model |
| `qwen3:30b` | about 22 GB | mixture of experts with 3B active parameters: fast even partly on CPU |

Memory figures are rough: weights at Ollama's default 4-bit quantization plus the
KV cache for a 32k-token window. Newer families with tool calling, such as Qwen
3.5 and Gemma 4, are worth trying; check their sizes in the
[Ollama library's Tools filter](https://ollama.com/search?c=tools) and pass the
tag with `--model`. Any model without tool calling is refused before the run.

**The context window is the setting that matters most.** The system prompt and
tool definitions alone take about 8k tokens, and every tool result adds to them.
Ollama's default window depends on VRAM: 4k tokens below 24 GiB, 32k from 24 to
48 GiB, 256k above. At 4k even the fixed part of the prompt does not fit. Nor
does Ollama reject a prompt that is too long: it silently drops the oldest
messages, the one stating your goal first. `tessa-agent` checks the window
before the first turn and refuses one that is too small, naming the fix, and
keeps the conversation inside a window that is big enough (see below). Aim for
32k tokens; 16k works, with older tool results elided sooner.

### Ollama (recommended)

```bash
# 1. Install it: https://ollama.com/download (on Linux: curl -fsSL https://ollama.com/install.sh | sh)

# 2. Start the server with a 32k window
OLLAMA_CONTEXT_LENGTH=32768 ollama serve

# 3. In another terminal, download the model (about 9 GB)
ollama pull qwen3:14b

# 4. Run the agent
uv sync --extra ai
uv run tessa-agent --data-root ./demo_data --make-demo --provider ollama \
  --goal "What separates calm from jittery events? Invent features if the stock aggregates are not enough."
```

If Ollama already runs in the background (the desktop app, or the Linux
installer's systemd service), set the variable where that process reads it,
then restart it:

- **macOS app:** `launchctl setenv OLLAMA_CONTEXT_LENGTH 32768`, or the context
  length slider in the app's settings.
- **Linux service:** `sudo systemctl edit ollama.service`, add
  `Environment="OLLAMA_CONTEXT_LENGTH=32768"` under `[Service]`, then
  `sudo systemctl daemon-reload && sudo systemctl restart ollama`.
- **Windows:** add `OLLAMA_CONTEXT_LENGTH` to your user environment variables,
  quit Ollama from the taskbar and start it again.

If you cannot change the server, give the model its own window instead:

```bash
printf 'FROM qwen3:14b\nPARAMETER num_ctx 32768\n' > Modelfile
ollama create qwen3-14b-32k -f Modelfile
uv run tessa-agent ... --provider ollama --model qwen3-14b-32k
```

`ollama ps` shows the window a loaded model got (`CONTEXT`) and how much of it
runs on the GPU (`PROCESSOR`). For Ollama on another machine, start it there
with `OLLAMA_HOST=0.0.0.0:11434` and pass `--base-url http://<host>:11434/v1`.
Ollama has no authentication, so do that on a trusted network only.

### LM Studio

1. Install [LM Studio](https://lmstudio.ai) and download a model that supports
   tool use, in the app or with `lms get <model>`.
2. Start the server and load the model with a 32k window. Load it explicitly:
   a model the server loads on demand gets the default context length.

   ```bash
   lms server start
   lms load <model> --context-length 32768
   ```

3. Run the agent. `--model` is only needed when LM Studio offers several models;
   the names are those `lms ls` lists.

   ```bash
   uv run tessa-agent ... --provider lmstudio --model <model>
   ```

### llama.cpp

Install it with `brew install llama.cpp` (macOS, Linux), `winget install
llama.cpp` (Windows), or a [release binary](https://github.com/ggml-org/llama.cpp/releases),
then start the server on a GGUF file or a Hugging Face repository:

```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF --jinja -c 32768 -np 1 --port 8080
# or a file you already have:
llama-server -m ./model.gguf --jinja -c 32768 -np 1 --port 8080

uv run tessa-agent ... --provider llamacpp
```

`--jinja` (the default in recent builds) is what makes the server parse tool
calls; `-np 1` gives the whole window to one conversation instead of splitting
it between parallel slots. If the server was started with `--api-key`, export
the same key as `LLAMA_API_KEY`.

### vLLM

For Linux with an NVIDIA or AMD GPU. vLLM parses tool calls only when told which
format the model uses:

```bash
pip install vllm
vllm serve Qwen/Qwen3-14B --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
# gpt-oss:  vllm serve openai/gpt-oss-20b --max-model-len 32768 \
#             --enable-auto-tool-choice --tool-call-parser openai

uv run tessa-agent ... --provider vllm
```

The parser for other models is in vLLM's tool-calling documentation. If the
server was started with `--api-key`, export the same key as `VLLM_API_KEY`.

### What `tessa-agent` does differently for a local server

- **It checks the server before the first turn** (`tessa.ai.local.probe`): that
  it answers, that it has the model (with the command to get it), that the
  model can call tools (Ollama), that the context window holds the prompt, and
  whether the model spills onto the CPU (Ollama). It also warns when an Ollama
  model is a cloud model, whose prompts do leave the machine. `--skip-check`
  turns the check off.
- **It keeps the conversation inside the window.** It reads the window from the
  server (Ollama, llama.cpp, vLLM, and LM Studio for a loaded model), or takes
  it from `--context-length`. Before a request would overflow, the oldest tool
  results are replaced by a short note telling the model to call the tool again
  if it needs the result; each new result is capped at half the room
  available, and a reply at a quarter of the window. `--context-length` turns
  this on for any provider.
- **It recovers tool calls a server left in the text.** When the server does not
  parse a model's call format, the call arrives as text, `<tool_call>{...}`
  or a bare JSON object, and the run would end on its first turn. Such calls are
  run anyway, with a warning, because they mean the server's tool-calling setup
  needs fixing.
- **It samples at temperature 0.6** instead of 0: greedy decoding sends small
  models into repetition loops, and Ollama substitutes 1.0 when no temperature is
  sent. `--temperature` overrides it.
- **A model call may take 30 minutes** and is not retried, since a retry would
  repeat the wait. `--timeout` changes the limit.
- **A server that fails mid-run** (out of memory, restarted) ends the run with
  `stopped: backend_error` and exit code 1, keeping the turns already done.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No Ollama server is answering at ...` | the server is not running, or listens elsewhere | `ollama serve` (or the server's own command); `--base-url` |
| `... runs <model> with a 4096-token context window` | the server's default window | the window settings above |
| `does not serve 'qwen3:14b'` | the model is not downloaded | `ollama pull qwen3:14b`, or `--model` with one the message lists |
| `cannot call tools` | the model has no tool calling | a model with the Tools tag |
| warning `recovered N tool call(s) from the message text` | the server does not parse the model's calls | llama.cpp: `--jinja`; vLLM: `--enable-auto-tool-choice --tool-call-parser ...` |
| the run ends after one turn, with JSON in the text | as above, in a format that cannot be recovered | as above |
| `stopped: too_many_failures` | the model keeps sending invalid calls | a larger model; `-v` shows the calls and the hints it was given |
| every turn takes minutes | the model runs partly or fully on the CPU | a smaller model or window, or `qwen3:30b` |
| `stopped: backend_error` | the server ran out of memory or timed out | a smaller model or window; `--timeout` |

To check a setup end to end with a short real run:

```bash
TESSA_LIVE_LOCAL=ollama uv run pytest -m live -q
TESSA_LIVE_LOCAL=llamacpp TESSA_LIVE_MODEL=<id> uv run pytest -m live -q
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
                                      └── schemas.py ──┬── backend_openai.py ── local.py (preflight)
                                                       └── backend_anthropic.py
                                                            └── loop.py
```

One schema source, two wire formats: `schemas.py` derives JSON Schema from each
tool's signature, type hints and docstring, then adapts it per provider. One loop
serves both backends; each backend only translates message formats. The previous
version of this layer hand-wrote Anthropic schemas, re-serialized them for the
OpenAI path, and duplicated the loop — and the copies drifted apart.

Local servers need no backend of their own: they speak the OpenAI protocol, so a
local preset is a base URL plus two switches on the OpenAI-compatible backend
(text tool-call recovery, a non-zero temperature). What differs between servers
is checked before the run, by `local.py`.

| Module | Role |
|---|---|
| `session.py` | Per-run state; cloned registries; the feature ledger |
| `expressions.py` | The AST-allowlist expression compiler |
| `render.py` | JSON/truncation boundary (NaN → `null`, arrays summarized) |
| `catalog.py` | Analysis catalogue by dataclass introspection + param coercion |
| `tools.py` | The 20 tool functions |
| `schemas.py` | Schema derivation + per-provider adapters |
| `prompts.py` | System prompt and exploration playbook |
| `loop.py` | Provider-neutral agent loop, including context-window compaction |
| `backend_openai.py` | Groq / OpenRouter / OpenAI / Ollama / LM Studio / llama.cpp / vLLM |
| `local.py` | Preflight check of a local server: model, tool support, context window |
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
- Given a context window (`context_tokens`), the oldest tool results are
  replaced by a short note before a request would overflow it, down to 60% of
  the budget so that it happens rarely: each time changes the prompt prefix a
  local server caches between turns. The opening message and the newest
  results are never elided. Tokens are estimated at 3 characters each, on the
  safe side for this JSON-heavy traffic.
- A model call that raises ends the run with `stopped_because="backend_error"`
  instead of an exception, so a server crash late in a run keeps what was done.

---

## Testing

Everything above is tested with no credentials and no network — the sandbox
against real escape attempts, session isolation, the tool surface end to end on
synthetic data, schema derivation, both backend adapters against fake clients,
the loop against a stub backend, and the local-server path (preflight checks,
CLI runs through the real OpenAI SDK) against a fake server on 127.0.0.1 that
answers like Ollama, LM Studio, llama.cpp or vLLM.

```bash
uv run pytest -q                    # `live` tests are deselected by default
uv run tessa-agent --data-root ./demo_data --make-demo --offline
```

The offline drive is the fastest way to verify an install: it exercises every
tool, both registries, the compiler, the budget guard and the persistence layer,
and asserts that a malicious expression is rejected.

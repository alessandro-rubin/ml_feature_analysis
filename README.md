# TESSA — Time-series Event Statistics and Separability Analysis

A modular, polars-first toolkit for analyzing time-series anomaly-detection
labels across many assets.

Given labeled events of the form `(asset_id, start, end, class, *extras)`,
the library answers generic questions like:

- *Which feature(s) discriminate class A from class B?*
- *Do strata (e.g. replacement type) behave differently?*
- *Are the classes separable at all, and via which representation?*

Class labels are deliberately abstract — the same pipeline works for
TP/FP/TN/FN, replacement-type subclasses, or any categorical label the
caller supplies.

## Requirements

- Python 3.11+
- Raw data laid out as `data/{asset_id}/{dataname}_{start}_{end}.parquet`
  (one asset per folder, dates as `YYMMDD` or `YYYYMMDD`). An asset may hold
  several data sets: files sharing a `dataname` prefix have identical columns
  and cover different periods (concatenated across time); different prefixes
  carry different variables, possibly at different sampling rates, and are
  full-outer-joined on the timestamp at load time.
- A label table with at least `(asset_id, start, end, class)` — the default
  source is an Excel sheet, but any `LabelSource` implementation works.

## Install

```bash
# with uv
uv sync

# or with pip
pip install -e .[dev]
```

Optional extras: `boosting` (lightgbm, xgboost), `clustering` (hdbscan,
umap-learn), `dashboard` (streamlit), `jupyter`, `dev` (pytest, ruff),
`ai` (OpenAI-compatible agent drivers + MCP server), `ai-anthropic`
(Claude driver), `all`.

## Package layout

The repo is a uv workspace with two packages: **`asset_loader`**, the standalone
data loader (only dependency: polars — reusable in other projects on the
same data sources, see `packages/asset_loader/README.md`), and **`tessa`**,
the analysis pipeline that depends on it.

```
packages/asset_loader/
  src/asset_loader/
    config.py        # LoaderConfig (data root, filename pattern, timestamp col)
    loader.py        # multi-source discovery + lazy loading (load_asset/load_event)
src/tessa/
  config.py          # Config dataclass (extends asset_loader.LoaderConfig)
  labels/            # LabelSource protocol + Excel implementation
  dataset/           # per-event builder (loader re-exported from asset_loader)
  features/          # FeatureSpec / AggSpec registries + materializers
  analysis/          # supervised (importance, classifier, pairwise,
                     # distributions, stratified), separability,
                     # unsupervised (anomaly, clustering, changepoint,
                     # correlation), semi-supervised (label spreading, PU),
                     # relations (lagged, MI network), plus DAG runner
  io/                # writers for figures, tables, parquet outputs
  results/           # AnalysisResult, ResultStore, UI-independent figures,
                     # static HTML report
  dashboard/         # streamlit run browser over a ResultStore
  ai/                # optional: LLM agent that can invent features
                     # (tools, expression sandbox, MCP server, CLI)
```

One-line data access without the pipeline:

```python
from asset_loader import load_asset

df = load_asset("A1", "path/to/data")                  # entire history, all sources
df = load_asset("A1", "path/to/data", columns=["x"])   # subset, still one line
```

### Data flow

```
LabelSource -> label_table
                    |
                    v
            dataset.builder   <-- dataset.loader (lazy per-event scan)
                    |
                    v  dict[event_id -> LazyFrame]
            features.materialize
              |         |            |
              v         v            v
         per_sample  windowed   period_aggregate
                                     |
                                     v
                                 analysis.*
```

Everything stays on polars `LazyFrame` until the analysis boundary. Pandas
only appears at the sklearn handoff.

### Default analysis input

Period-aggregate (one row per event) is the default input for analyses.
Windowed and per-sample are opt-in when an analysis needs dynamics.

### Label handling

All analyses accept:

- `target_col` — which label column is the class (default `class`).
- `label_filter` — restrict rows to a subset, e.g. `{class: [TP, FP]}`.
- `stratify_by` — run the analysis per stratum value (e.g. `replacement_type`).

"TP vs FP", "TN vs FN vs nominal", and "behavior by replacement type" are
configuration, not new code paths.

## Quick start

```python
import polars as pl
from tessa import Config
from tessa.labels.excel import ExcelLabelSource
from tessa.dataset.builder import build
from tessa.features.materialize import to_period
from tessa.features import builtins  # registers stock features/aggregators

cfg = Config(data_root="data/")

labels = ExcelLabelSource("labels.xlsx").load()
events = build(labels, cfg=cfg)                  # dict[event_id -> LazyFrame]
period = to_period(events, feature_specs=[...], agg_specs=[...])

# period is one row per event, ready to feed analyses.
```

### Notebook-first API (`Run`)

`Run` is the primary interface: one object from analysis table to saved
results, with shared `prepare_xy` caching across analyses.

```python
from tessa import Config, Run

run = Run(period, target_col="class", cfg=Config(random_state=7))

run.separability().summary()       # "are the classes distinguishable at all?"
run.importance().frames["table"]   # blended feature ranking
run.anomaly().summary()            # unsupervised ensemble; target_col optional
run.figures()                      # curated matplotlib figures per analysis

run.save("outputs/runs")           # parquet + manifest, dashboard-ready
run.report("outputs/report.html")  # self-contained static HTML
```

Leave `target_col=None` for fully unsupervised data: label-requiring
analyses are skipped (with a warning) while anomaly, clustering,
changepoint, correlation, and relations still run.

Browse a saved run with the dashboard:

```bash
streamlit run src/tessa/dashboard/app.py -- outputs/runs
```

`demo.py` runs the whole pipeline end-to-end on synthetic data; see
`tests/` for runnable examples of each analysis.

## AI agent (optional)

The pipeline can be driven by a language model that decides what to measure and
— the point of the layer — **writes new features** when the stock aggregates
cannot see what distinguishes the classes. Feature expressions are ordinary
Polars, supplied as strings and checked against an AST allowlist before they run:

```python
create_feature("vibration__jerk", 'pl.col("vibration").diff().abs()',
               rationale="the classes differ in ordering, not in level")
```

Three ways to drive it, two of which need no Anthropic key:

```bash
uv run tessa-agent --data-root ./demo_data --make-demo --offline   # no model at all
uv run tessa-agent --data-root data --labels labels.xlsx \
  --goal "What separates TP from FP?" --provider groq             # Groq/OpenRouter/OpenAI/local
uv run tessa-mcp                                                   # any MCP client
```

`tessa-mcp` exposes the same tools over the Model Context Protocol, so Claude
Desktop, Claude Code, Cline or Zed can drive them using their own model — no API
key of yours involved. The tool functions are plain Python and import no SDK, so
they are equally usable directly from a notebook.

Every invented feature is recorded with its expression, rationale and measured
contribution, and the ledger is written into the run manifest so a saved run
explains its own columns. Because features are selected by measuring them on the
same data, `holdout_assets` + `confirm_on_holdout` reserve assets that the
exploration never sees.

See [AI_INTEGRATION.md](AI_INTEGRATION.md) for the tool reference, the expression
language, and the sandbox's threat model.

## Statistical tests and corroboration

The analyses ship with a layered statistical-testing toolkit: multiple-
testing correction (Bonferroni / Holm / BH-FDR), effect sizes
(Cohen's d, Hedges' g, Cliff's delta, Wasserstein, Jensen-Shannon),
bootstrap CIs, cross-validated classifier metrics
(MCC / balanced accuracy / Brier / ECE), importance stability
(bootstrap CIs + Spearman method agreement), and cluster validation
(Hopkins statistic + permutation p-values for ARI / V-measure).

See [STATISTICAL_TESTS.md](STATISTICAL_TESTS.md) for what each test
does, when to use it, and how to interpret the output.

## Status

v0.2 (101 tests passing). The toolkit covers the full mission scope:

- **Foundation** — `Config`, per-event lazy loader + builder, pluggable
  label sources (Excel), feature/aggregator registries, per-sample /
  windowed / period materializers (parallel `collect_all`), pluggable
  analysis module with a DAG runner.
- **Supervised** — importance (rank-blended), distributions, pairwise,
  classifier / cross-validated classifier (grouped CV), importance
  stability, stratified runs.
- **Separability** — permutation-tested "are the classes distinguishable
  at all?" with effect sizes and a plain-language verdict.
- **Unsupervised** — anomaly ensemble (IsolationForest + LOF + robust
  Mahalanobis) with per-feature attribution, clustering + validation,
  changepoint (CUSUM), correlation structure.
- **Semi-supervised** — label spreading, PU learning.
- **Relations** (exploratory) — lagged relations, MI network.
- **Consumption** — `Run` notebook facade, `ResultStore` with a
  reproducibility manifest, UI-independent figures shared by a static HTML
  report and a Streamlit dashboard.

See [CHANGELOG.md](CHANGELOG.md) for the release history,
[ARCHITECTURE.md](ARCHITECTURE.md) for the design, build order, and the
original phased plan (§11 appendix), and
[STATISTICAL_TESTS.md](STATISTICAL_TESTS.md) for the statistical-rigor
layer. [AUDIT.md](AUDIT.md) is kept as a historical record of the audit
that drove the evolution.

## Development

```bash
pytest            # run test suite
ruff check .      # lint
```

GitHub Actions (`.github/workflows/ci.yml`) runs `uv sync --all-extras`,
`ruff check`, `ruff format --check`, and `pytest` on every push. Tests marked
`live` call a real model API and are deselected by default, so CI never spends
money; run them with `pytest -m live`.

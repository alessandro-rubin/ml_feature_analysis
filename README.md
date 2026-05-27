# ml_analysis

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
- Raw data laid out as `data/{asset_id}/input/{asset_id}_{start}_{end}.parquet`
  (one asset per folder, ~1 Hz samples, multiple files per asset).
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
umap-learn), `all`.

## Package layout

```
src/ml_analysis/
  config.py          # Config dataclass (paths, timestamp col, filename pattern)
  labels/            # LabelSource protocol + Excel implementation
  dataset/           # per-event lazy loader + builder
  features/          # FeatureSpec / AggSpec registries + materializers
  analysis/          # importance / clustering / classifier / pairwise /
                     # stratified / distributions, plus DAG runner
  io/                # writers for figures, tables, parquet outputs
  legacy/            # kept until the Phase 5 port is fully verified
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
from ml_analysis import Config
from ml_analysis.labels.excel import ExcelLabelSource
from ml_analysis.dataset.builder import build
from ml_analysis.features.materialize import to_period
from ml_analysis.features import builtins  # registers stock features/aggregators

cfg = Config(data_root="data/")

labels = ExcelLabelSource("labels.xlsx").load()
events = build(labels, cfg=cfg)                  # dict[event_id -> LazyFrame]
period = to_period(events, feature_specs=[...], agg_specs=[...])

# period is one row per event, ready to feed analyses.
```

See `tests/` for runnable end-to-end examples on synthetic data.

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

Phases 0-6 of [PLAN.md](PLAN.md) have landed:

- scaffolding, `Config`
- per-event lazy dataset loader + builder
- pluggable label sources (Excel)
- feature and aggregator registries
- per-sample / windowed / period materializers
- pluggable analysis module with DAG runner + ported legacy logic
- pairwise / stratified / distribution analyses

See [PLAN.md](PLAN.md) for the full design and phase breakdown.

## Development

```bash
pytest            # run test suite
ruff check .      # lint
```

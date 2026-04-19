# ml_analysis — Project Plan

## Goal

A modular, polars-first toolkit for analyzing time-series anomaly-detection
labels across many assets. Given labeled events of the form
`(asset_id, start, end, class, *extras)`, the library must answer generic
questions of the form:

- *Which feature(s) discriminate class A from class B?*
- *Do strata (e.g. replacement type) behave differently?*
- *Are classes separable at all, and via which representation?*

The class labels are deliberately abstract: the same pipeline must work for
TP/FP/TN/FN, for replacement-type subclasses, or any other categorical label
a user provides.

## Core assumptions

- Raw data lives at `data/{asset_id}/input/{asset_id}_{start}_{end}.parquet`,
  multiple files per asset, ~90 features @ 1 Hz, ~1 year per asset,
  few hundred GB total.
- Labels arrive as a table `(asset_id, start, end, class, *extras)` from a
  pluggable source (default: Excel file).
- Everything runs on polars `LazyFrame` until the analysis boundary.
  Pandas only appears at the sklearn handoff.

## Architecture

```
src/ml_analysis/
  config.py                # global config, paths, timestamp col, etc.
  dataset/
    loader.py              # (asset_id, start, end) -> LazyFrame
    builder.py              # list[label row] -> dict[event_id -> LazyFrame]
  labels/
    base.py                # LabelSource protocol
    excel.py               # Excel implementation
  features/
    registry.py            # FeatureSpec, @feature decorator, DAG resolver
    aggregates.py          # AggSpec, @aggregate decorator
    materialize.py         # to_per_sample / to_windowed / to_period
    builtins.py            # stock time-series features & aggregators
  analysis/
    base.py                # Analysis protocol, registry, DAG runner
    importance.py          # ported from legacy
    clustering.py          # ported from legacy
    classifier.py          # ported from legacy
    pairwise.py            # Phase 6
    stratified.py          # Phase 6
    distributions.py       # Phase 6
  io/
    writers.py             # save figures, tables, parquet outputs
  legacy/
    ml_analysis.py         # original script, kept until Phase 5
```

### Data flow

```
LabelSource -> label_table (polars DF)
                    |
                    v
            dataset.builder   <-- dataset.loader (lazy scan per event)
                    |
                    v  dict[event_id -> LazyFrame]
            features.materialize
              |         |              |
              v         v              v
         per_sample  windowed     period_aggregate
                                       |
                                       v
                                analysis.* (default input)
```

### Default analysis consumer

Period-aggregate (one row per event) is the default input for analyses.
Windowed and per-sample are opt-in when an analysis needs dynamics.
Rationale: cross-event comparison is the main use case; windowed views
are used as a diagnostic when a period-aggregate analysis cannot separate
classes.

### Label handling in analyses

All analyses accept:

- `target_col`: which label column to treat as the class (default `class`).
- `label_filter`: optional callable/dict to restrict rows to a subset
  (e.g. `{class: [TP, FP]}` answers *"can we discriminate TP from FP?"*).
- `stratify_by`: optional column to run the analysis per-stratum
  (e.g. `replacement_type`).

This means "TP vs FP", "TN vs FN vs nominal", and "behavior by replacement
type" are all expressed as configuration, not as new code paths.

## Defaults (adjustable)

| Item | Default | Notes |
|------|---------|-------|
| Timestamp column | `timestamp` | configurable in `Config` |
| Filename date format | `YYYYMMDD` | configurable regex in loader |
| Default analysis input | period-aggregate | see above |
| Legacy code | kept in `legacy/` until Phase 5 | deleted after port |
| Python version | 3.11+ | |
| Package manager | `uv` / `pip` via `pyproject.toml` | |

## Execution phases

### Phase 0 — scaffolding
- `pyproject.toml`, `src/ml_analysis/` package skeleton.
- Move current `ml_analysis.py` into `src/ml_analysis/legacy/` unchanged.
- `Config` dataclass (paths, timestamp col, filename pattern).
- `.gitignore`, empty `tests/`.

### Phase 1 — dataset builder
- `dataset/loader.py::load_event(asset_id, start, end) -> LazyFrame`
  - glob files under `data/{asset_id}/input/`
  - parse date range from filename, keep overlapping files
  - `pl.scan_parquet(...)` + filter on timestamp
- `dataset/builder.py::build(labels: pl.DataFrame) -> dict[event_id, LazyFrame]`
  - attaches label metadata as literal columns
  - returns lazy — no materialization yet

### Phase 2 — label sources
- `labels/base.py::LabelSource` protocol:
  `load() -> pl.DataFrame[asset_id, start, end, class, *extras]`
- `labels/excel.py::ExcelLabelSource(path, sheet=..., column_map=...)`
- Schema validation (required cols present, types coerced).

### Phase 3 — feature registry
- `FeatureSpec(name, deps: list[str], expr: Callable[[], pl.Expr])`
- `@feature` decorator registers into a module-level registry.
- Dependency resolver orders specs topologically so features can reference
  previously-registered features by name.
- `features/builtins.py`: rolling mean / std / min / max / p95, first
  difference, slope, z-score, etc.

### Phase 4 — three materializers
- `to_per_sample(lf, specs) -> LazyFrame`
  adds feature columns to each row, preserves label metadata.
- `to_windowed(lf, specs, aggs, every, period) -> LazyFrame`
  `groupby_dynamic` over timestamp, applies aggregators.
- `to_period(lf, specs, aggs) -> DataFrame`
  one row per event (full-range aggregate). Implemented as a degenerate
  case of `to_windowed` with the window = event duration, so aggregator
  code is shared.
- `AggSpec(name, source, agg_fn, params)` and `@aggregate` decorator.

### Phase 5 — analysis module refactor
- `analysis/base.py::Analysis` protocol:
  `name`, `requires: list[str]`, `run(ctx) -> AnalysisResult`.
- Registry + DAG runner: given a list of analyses, resolve dependencies
  and execute in order, sharing a `ctx` dict.
- Port legacy importance / clustering / classifier code into
  `analysis/importance.py`, `analysis/clustering.py`, `analysis/classifier.py`.
- All ports accept `target_col` + `label_filter` + `stratify_by`.
- Delete `legacy/ml_analysis.py` once parity is verified.

### Phase 6 — question-specific analyses
- `analysis/pairwise.py`
  - for each pair `(A, B)` of classes: importance, 1-feature AUC,
    Cliff's delta, KS statistic.
  - directly answers *"can we discriminate TP from FP?"*.
- `analysis/stratified.py`
  - wraps any analysis and runs it per stratum value, then diffs results.
  - directly answers *"do replacement types behave differently?"*.
- `analysis/distributions.py`
  - per-feature violin / ECDF / density plots split by class.
  - often more interpretable than importance ranks.

## Deliverables per phase

Each phase ends with:

1. Working code + minimal tests on synthetic data.
2. A short note in `CHANGELOG.md` describing what landed.
3. A git commit (`feat(phaseN): ...`).

## Open items / flags

- Timestamp column name and filename date regex are assumed defaults;
  confirm against real data before Phase 1 ends.
- Label schema extras (`replacement_type` etc.) will be inferred from the
  Excel source — no fixed list.
- No streaming / Dask layer yet. Polars lazy + event-scoped scans should
  suffice for a few hundred GB when one event at a time is materialized.

# Architecture: time-series predictive-maintenance & fault-identification toolkit

Status: **target architecture**, designed from the mission scope and `AUDIT.md`
only — deliberately *without* reading the existing source. A separate section
at the end (added after a code review) records the compatibility verdict:
evolve vs. redesign.

> **Build complete (v0.1).** The "EVOLVE" verdict and the revised build
> order in §10 have landed in full — see [PROGRESS.md](PROGRESS.md) for
> per-step verification and [CHANGELOG.md](CHANGELOG.md) for the release
> notes. Illustrative names in the snippets below (e.g. `import pmkit as
> pm`) are design sketches; the shipped package is `ml_analysis` with the
> `Run` facade documented in [README.md](README.md).

---

## 1. Mission & design constraints

The tool answers three questions about multi-asset time-series data
(~100 sensor channels, tens of GB, single machine):

1. **Which features are related to failures?**
2. **Which features (and how) discriminate classes** — failure/healthy,
   failure modes, asset cohorts?
3. **Is class discrimination possible at all** — and with what confidence?

Operating regimes (both first-class):

- **Fully labeled:** an event table `(asset_id, t_start, t_end, label)` exists.
  Supervised discrimination, importance, and separability testing apply.
- **Unlabeled / sparsely labeled:** discover anomalies, similarities,
  correlations, regime changes; optionally let a handful of labels guide the
  search (semi-supervised).

Consumption: **notebook-first Python API** (result objects with `.summary()` /
`.plot()`), plus an **interactive dashboard** layered on the same results.

Hard constraints derived from the audit:

- The analysis unit is the **reduced aggregate row** (one row per
  window/event), never the raw sample stream. sklearn sees thousands of rows,
  not billions.
- All heavy reduction happens in **lazy polars, collected once** (streaming
  engine when needed). No per-event Python loops with `.collect()` inside.
- **No silent data loss:** null handling, row drops, and capped p-values are
  reported, never swallowed.
- **One seed to rule them all:** `Config.random_state` is injected into every
  estimator at run time; nothing hardcodes its own seed.

---

## 2. Layer overview

```
┌─────────────────────────────────────────────────────────────┐
│  consume    notebook API · static HTML report · dashboard   │
├─────────────────────────────────────────────────────────────┤
│  results    typed Result objects · ResultStore · run        │
│             manifest (reproducibility)                      │
├─────────────────────────────────────────────────────────────┤
│  analysis   Analysis protocol + registry + DAG runner       │
│             supervised │ unsupervised │ semi-sup │ relations│
├─────────────────────────────────────────────────────────────┤
│  features   declarative FeatureSet (polars exprs) ·         │
│             Materializer (one lazy query) · feature store   │
├─────────────────────────────────────────────────────────────┤
│  data       Dataset (lazy, multi-asset) · EventTable ·      │
│             WindowSpec · schema/metadata catalog            │
└─────────────────────────────────────────────────────────────┘
```

Each layer depends only on the one below. Registries make features and
analyses pluggable (a new analysis is one class + one decorator).

---

## 3. Data layer (`data/`)

### 3.1 `Dataset`

A thin, **lazy** wrapper over the raw store:

```python
ds = Dataset.scan_parquet("data/**/*.parquet", schema=SensorSchema(...))
ds.assets            # list of asset ids
ds.channels          # ~100 sensor columns, with dtype/unit metadata
ds.lazy(asset=...)   # -> pl.LazyFrame, never collected here
```

- Canonical long schema: `asset_id`, `timestamp`, `<channel...>`. Adapters
  (`from_wide`, `from_csv`, `from_database`) normalize into it.
- **Schema catalog**: per-channel metadata (numeric/categorical, unit,
  expected range). Explicit — no dtype-sniffing heuristics to decide what is
  a label vs. a sensor (an audit-flagged latent bug class).
- Validation pass on registration: monotone timestamps per asset, duplicate
  detection, channel coverage report. If files are chronologically written,
  `set_sorted` instead of re-sorting.

*Shipped (`dataset/facade.py`):* `Dataset(Config(data_root=...))` with
`.assets`, `.channels(asset)`, `.lazy(asset, start, end)`, `.events(labels)`.
The `scan_parquet`/`SensorSchema` constructor, the explicit schema catalog,
and the registration-time validation pass are **design sketches — not
implemented** (`Config.assume_sorted` is the only `set_sorted` lever).

### 3.2 `EventTable` (labels & windows of interest)

```python
events = EventTable.from_frame(df)   # asset_id, t_start, t_end, label
events.labels                        # may contain nulls -> semi-supervised
```

- Interval-based labels; point events get a configurable horizon
  (e.g. "48 h pre-failure" windows for early-warning features).
- **Nullable label column is legal** — unlabeled rows flow into the
  unsupervised/semi-supervised paths instead of being rejected.
- Helpers: `healthy_baseline()` (sample healthy reference windows away from
  any event), class balance report, leakage guard (no window overlap between
  train/test splits at the *asset* or *event* level).

*Not shipped:* there is no `EventTable` class. Labels enter through the
`LabelSource` protocol (`labels/`, `ExcelLabelSource`) and become per-event
frames via `Dataset.events(label_table)`. Nullable labels flow through, but
`healthy_baseline()` and the leakage guard were not built.

### 3.3 `WindowSpec`

One declarative object for "what is a row in the analysis table":

- `event` — one row per labeled event (post-hoc fault characterization),
- `tumbling(every=...)` / `sliding(every=..., period=...)` — unlabeled
  monitoring grid,
- `pre_event(horizon=...)` — early-warning windows.

The spec compiles to polars `group_by_dynamic` / interval-join expressions, so
windowing stays inside the lazy query.

*Shipped (`features/windows.py`):* `WindowSpec.event()`, `.tumbling(every)`,
`.sliding(every, period)`. `pre_event(horizon=...)` was **not** built.

---

## 4. Feature layer (`features/`)

### 4.1 Declarative features

A **feature is a named polars expression factory**, registered once:

```python
@feature("rms", tags=["energy"])
def rms(col: str) -> pl.Expr:
    return (pl.col(col) ** 2).mean().sqrt()
```

Built-in families:

- **Aggregate stats:** mean/std/min/max/quantiles, skew, kurtosis, RMS,
  peak-to-peak, crest factor, zero-crossing rate.
- **Trend/dynamics:** linear slope, EWMA level vs. raw, diff stats, lag
  autocorrelation at chosen lags.
- **Spectral (optional plugin):** band power, spectral centroid/entropy via
  rFFT on the window (numpy at the boundary, but vectorized per window).
- **Cross-channel:** rolling correlation between channel pairs, ratios.

`FeatureSet` = channels × features (with include/exclude rules), serializable
to/from config so a run is reproducible from its manifest.

*Not shipped:* there is no `FeatureSet` object and `@feature` takes no
`tags`. Its real signature is `@feature(name, deps=())`; selection happens
through `sources` / `feature_names` / `aggregators` lists passed to
`materialize` / `to_period`.

### 4.2 Materializer — *one* query, *one* collect

```python
table = materialize(ds, window_spec, feature_set, events=events)
# -> polars DataFrame: one row per (asset, window), ~100×k feature columns,
#    plus asset_id, window bounds, label (nullable)
```

- Builds a single lazy graph: `scan -> interval-join windows -> group_by
  (asset_id, window_id).agg(feature exprs)` and collects **once**, with
  `engine="streaming"` above a size threshold. Per-window isolation for
  rolling/diff features via `.over(window_id)` — never a Python loop.
- Schema is resolved once, not per event.
- **Feature store:** results cached to parquet keyed by
  `hash(data fingerprint, window_spec, feature_set)`; notebook restarts and
  the dashboard reuse the cache instead of recomputing.

*Partially shipped:* `materialize(lfs, spec, cfg, ...)` takes per-event
LazyFrames and a `Config` (not `ds` + `feature_set` + `events`). The feature
store (parquet cache of materialized features) was **not** built — only
analysis *results* are persisted, via `ResultStore` (§6.1).

### 4.3 Null & quality policy (explicit, reported)

`prepare_xy(table, policy=...)` is the single sklearn boundary:

- Policies: `drop_rows`, `drop_features(max_null_frac=...)`, `impute(median)`.
- Always returns a `PreparationReport`: rows in/out, features dropped and why,
  per-feature null fractions. Listwise deletion without a report is banned by
  construction (audit §2, `prepare_xy` row-drop issue).
- Output `PreparedXY` (X as float matrix, y optional, feature names, report)
  is **cached on the run context** so chained analyses share one preparation
  and one model fit where applicable (audit §3, redundant refits).

---

## 5. Analysis layer (`analysis/`)

### 5.1 Protocol & runner

```python
class Analysis(Protocol):
    name: str
    requires: tuple[str, ...]          # DAG deps, e.g. ("prepared_xy",)
    needs_labels: Literal["full", "partial", "none"]

    def run(self, ctx: Context) -> Result: ...
```

- `Context` carries: config (incl. the **injected** `random_state`), the
  materialized table, cached `PreparedXY`, cached fitted models, and prior
  results. The DAG runner resolves `requires`, runs each node once, and
  caches by node name.
- `needs_labels` lets the runner auto-select what is runnable for the current
  data: a fully-unlabeled dataset silently skips supervised nodes with a
  clear message instead of erroring.
- Pre-built pipelines: `pipeline.supervised_screen()`,
  `pipeline.anomaly_discovery()`, `pipeline.separability_check()` — but every
  analysis is also directly callable in a notebook.

*Not shipped:* there is no `pipeline` module. The equivalent is the `Run`
facade (`run.py`) — `Run(table).run_all([...])` or the per-analysis
shortcuts (`run.importance()`, `run.anomaly()`, …), which share one
`AnalysisContext` and its `prepare_xy` cache.

### 5.2 Supervised family — *"which features discriminate, and how?"*

| Analysis | Output |
|---|---|
| `importance` | blended feature ranking — **rank aggregation** across RF impurity, permutation importance, mutual information, ANOVA/Kruskal (never min-max-mean of raw scores — audit §3) |
| `distributions` | per-feature class-conditional distributions: test statistic, **BH/Bonferroni-corrected p**, **effect size** (Cliff's δ / Cohen's d) + bootstrap CI; capped p-values flagged as capped |
| `pairwise` | which class *pairs* a feature separates (one-vs-one effect sizes) — answers "which failure modes does this sensor distinguish?" |
| `classifier`, `cv_classifier` | baseline RF/GBM with **asset/event-grouped CV** (leakage guard), confusion matrix, per-class recall |
| `importance_stability` | rank stability across bootstrap refits; shares the prepared X/y and seeds from config |

### 5.3 Separability family — *"is discrimination possible at all?"*

- `separability`: permutation-tested CV score — observed grouped-CV accuracy
  vs. the null distribution from label shuffling ⇒ a p-value for "the classes
  are distinguishable at all".
- `embedding_overlap`: PCA/UMAP embedding + silhouette/class-overlap measures
  on labels; visual + quantitative.
- `cluster_alignment`: unsupervised clustering scored against labels
  (ARI/NMI) — *one diagnostic among several*, not the only unsupervised tool.
- Hopkins/clusterability statistics implemented with **power-1 distances**
  (the power-`d` variant overflows at ~100 features — audit §3 bug).

*Shipped as:* `SeparabilityTest` (the permutation test) plus
`ClusterAnalysis` + `ClusterValidation` (label-scored clustering with the
power-1 Hopkins fix). The named `embedding_overlap` and `cluster_alignment`
analyses were **not** built.

### 5.4 Unsupervised family — *"what's anomalous / structured in unlabeled data?"*

- `anomaly`: ensemble scoring — IsolationForest, LOF, robust-covariance
  Mahalanobis, optionally One-Class SVM / autoencoder — fit on a healthy
  baseline (or all data), producing per-window anomaly scores **and
  per-feature contributions** (e.g. z-score decomposition / SHAP on the
  isolation model) so an anomaly is always traceable to sensors.
- `clustering` + **internal** validation (silhouette, stability under
  resampling) that works with zero labels.
- `correlation_structure`: feature redundancy (hierarchical clustering on
  |Spearman|), near-duplicate channel detection — also feeds feature pruning.
- `changepoint` / `drift`: per-asset regime changes on key channels
  (e.g. ruptures/PELT or rolling-stat CUSUM) — surfaces "when did this asset
  start behaving differently".

### 5.5 Semi-supervised family — *"a few labels, lots of unlabeled"*

- `label_spreading`: propagate sparse labels over a kNN graph of windows;
  output = soft labels + confidence, feeding the supervised screens.
- `pu_learning`: positive-unlabeled wrapper (labeled failures + unlabeled
  rest) for early-warning scoring.
- `guided_anomaly`: anomaly ensemble calibrated/thresholded on the few known
  failure windows.

*Shipped as:* `LabelSpreadingAnalysis` and `PULearningAnalysis`. There is no
separate `guided_anomaly` — `AnomalyDetection(baseline_filter=...)` covers
fitting the ensemble on known-healthy windows.

### 5.6 Relations family — *exploratory, clearly flagged*

- `lagged_relations`: cross-correlation / Granger-style lagged regression
  between channels and failure indicators (reported as *predictive
  association*, not proven causation).
- `mi_network`: mutual-information graph between features for systemic-view
  plots.

### 5.7 Statistical rigor (cross-cutting)

Shared `stats/` utilities: multiple-testing correction, effect sizes with
bootstrap CIs (no bare `except Exception` — failures during resampling
surface as warnings with counts), permutation tests, all consuming the
context RNG.

*Shipped, but not as a `stats/` package:* these live under `analysis/`
(`multiple_testing.py`, `effect_sizes.py`).

---

## 6. Results & consumption

### 6.1 Typed results

Every analysis returns a frozen dataclass:

```python
@dataclass(frozen=True)
class ImportanceResult(Result):
    ranking: pl.DataFrame          # feature, per-method rank, blended rank
    def summary(self) -> str: ...  # human-readable text
    def plot(self) -> Figure: ...  # plotly, notebook-renderable
    def to_frames(self) -> dict[str, pl.DataFrame]   # for export/dashboard
```

- Tables are polars; plots are **plotly** (interactive in notebooks *and*
  embeddable in the dashboard/HTML with no second plotting stack).
- `ResultStore` persists results + the **run manifest** (data fingerprint,
  feature-set hash, config incl. seed, package versions) to a run directory —
  full reproducibility and the dashboard's data source.

*Shipped differently:* there is one generic `AnalysisResult` (not
per-analysis subclasses like `ImportanceResult`) with `.summary()` and a
`.frames` dict (not `to_frames()`). Plotting is **matplotlib**, not plotly:
the UI-independent `results/figures.py` factory (`figures_for_result` /
`Run.figures()`) feeds both the HTML report and the Streamlit dashboard.
`ResultStore` (with the run manifest) shipped as described.

### 6.2 Notebook-first API (the primary interface)

```python
import pmkit as pm

ds      = pm.Dataset.scan_parquet("data/*.parquet")
events  = pm.EventTable.from_csv("events.csv")
table   = pm.materialize(ds, pm.windows.event(), pm.features.default())

run = pm.Run(table, events, config=pm.Config(random_state=7))
run.separability().summary()      # "classes distinguishable, p<0.001 ..."
run.importance().plot()
run.anomaly(baseline="healthy").plot(asset="A07")
run.report("run_2026-06-09/")     # static HTML, all results
```

*This snippet is a design sketch.* `Run` and `run.report(...)` are real, but
the package is `ml_analysis` (not `pmkit`), `Run(table, target_col=..., cfg=
...)` takes a materialized table (no `EventTable`), results expose `.frames`
not `.plot()` (use `run.figures()`), and the anomaly baseline parameter is
`baseline_filter={...}`. The accurate version is in
[README.md](README.md#notebook-first-api-run).

### 6.3 Dashboard (thin layer, no second compute path)

- Streamlit (or Dash) app that **reads the `ResultStore`** — it never
  recomputes analyses; at most it triggers `Run` jobs through the same API.
- Pages: asset browser (raw channels + windows + anomaly score timeline),
  feature explorer (distributions by class, importance ranking),
  separability view (embedding + permutation test), run comparison.

### 6.4 Static report

`run.report()` renders all `Result.to_html()` fragments into one
self-contained HTML — shareable without the dashboard running.

---

## 7. Configuration & reproducibility

- One `Config` (pydantic): seed, null policy, CV scheme (grouped by
  asset/event), estimator defaults, multiple-testing method, thresholds.
- **Seed injection rule:** estimator factories take `random_state=None` and
  receive `ctx.cfg.random_state` at `run()` time. A test asserts that two
  runs with different seeds produce different forests and the same seed
  reproduces bit-identical rankings.
- Run manifest written before compute starts; results refuse to load against
  a mismatched manifest without `allow_mismatch=True`.

---

## 8. Packaging, quality gates

- `pyproject.toml` with `readme = "README.md"`; package layout `src/<pkg>/`.
- CI: ruff + pytest on every push; the audit's hygiene items (unused imports,
  no CHANGELOG) become CI-enforced.
- No `legacy/` second source of truth: anything ported is ported with tests,
  then the legacy file is deleted in the same change.
- Test pillars: golden-number tests for stats utilities, property tests for
  the materializer (loop-free result == reference per-event result), seed
  reproducibility test, null-policy report test.

---

## 9. Build order

1. **Data + feature layers** (Dataset, EventTable, WindowSpec, single-query
   materializer with feature store) — everything sits on this.
2. **Context/DAG runner + PreparedXY caching + seed injection** — the
   correctness backbone.
3. **Separability + supervised screens** (importance with rank aggregation,
   distributions with corrected p/effect sizes, grouped CV).
4. **Unsupervised: anomaly ensemble with per-feature contributions**, then
   clustering/internal validation, correlation structure, changepoints.
5. **Semi-supervised modes** (label spreading, PU, guided anomaly).
6. **ResultStore + static report**, then the **dashboard** on top.
7. Relations family (lagged/MI) last — exploratory value-add.

---

## 10. Compatibility verdict (written *after* reviewing the existing code)

Everything above this line was designed blind. The existing
`src/ml_analysis/` was then reviewed and compared against it.

### Verdict: **EVOLVE** — the skeletons align; the gaps are extensions and targeted rewrites, not a re-architecture

The existing code independently arrived at the same load-bearing patterns
this plan calls for, which makes redesigning from scratch wasteful:

| Target design (above) | Existing code | Alignment |
|---|---|---|
| `@feature` registry of polars expr factories, dep-sorted (§4.1) | `features/registry.py` — `FeatureSpec`, `@feature`, topo `resolve()` | **near-identical**, plus a separate `AggregatorRegistry` (a good split the plan keeps) |
| `Analysis` Protocol + `requires` + DAG runner (§5.1) | `analysis/base.py` — same Protocol, same topo-sort runner | **near-identical** |
| One `Config` with `random_state`, column names (§7) | `config.py` dataclass | **same shape** (injection not wired — see fixes) |
| `EventTable` from pluggable sources (§3.2) | `labels/` `LabelSource` Protocol + `validate()` | **same idea**, intervals + extras carried through |
| Supervised + separability analyses (§5.2–5.3) | 9 analyses incl. importance, distributions, pairwise, CV classifier, stability, cluster validation; rigor layer (effect sizes, multiple testing) | **present**, modulo the audit's correctness fixes |
| Materializer granularities (§3.3, §4.2) | `to_per_sample` / `to_windowed` / `to_period` | **same coarse-to-fine spectrum**, wrong execution strategy in `to_period` |

### Divergences that must change (rewrite-in-place, API mostly preserved)

1. **Materialization strategy** — `to_period` is the audit's per-event Python
   loop (`collect()` + `collect_schema()` per event; `builder.build` returns
   a `{event_id: LazyFrame}` dict that forces this shape). Rebuild as §4.2's
   single lazy query (interval-join + `group_by("event_id")`, one streaming
   collect, `.over(event_id)` isolation). The `{event_id: LazyFrame}` dict
   stays accepted as input; the loop goes.
2. **No unlabeled mode** — `AnalysisContext.target_col` is mandatory and
   `prepare_xy` listwise-drops nulls silently. Make `target_col` optional,
   add `needs_labels` to the Protocol, and replace the null handling with
   §4.3's policy + `PreparationReport`. This unlocks the entire
   unsupervised/semi-supervised column of the plan.
3. **`_label_cols` dtype heuristic** (`features/materialize.py`) — replace
   with the explicit schema catalog of §3.1; "non-numeric ⇒ label" is the
   audit's latent-bug class.
4. **Audit correctness fixes** as specified: Hopkins power-1, seed injection
   from `cfg.random_state` (forests currently hardcode 42), rank-based
   importance blending, `PreparedXY` caching on the context, no bare
   `except` in bootstrap.

### Pure additions (no existing code touched)

- `analysis/anomaly.py`, `semi/` (label spreading, PU, guided anomaly),
  `changepoint`, `correlation_structure`, `relations` — §5.4–5.6.
- `Dataset` / `WindowSpec` facade (§3.1, §3.3) wrapping the existing
  loader/builder so notebooks get the §6.2 API without breaking callers.
- Typed `Result` objects, `ResultStore` + run manifest, static HTML report,
  dashboard (§6). Existing analyses currently return ad-hoc dicts/frames;
  wrap first, migrate gradually.
- **Plotting stack:** existing `io/` is matplotlib/seaborn-to-disk. New
  `Result.plot()` is plotly (notebook + dashboard need interactivity);
  matplotlib writers remain for batch artefacts until ported. Add `plotly`,
  `streamlit` (extra), `pydantic` to dependencies.

### Housekeeping (from the audit, unchanged)

Delete `legacy/ml_analysis.py`, fix `pyproject readme = "PLAN.md"` →
`README.md`, add CI (ruff + pytest), CHANGELOG, remove unused imports.

### Revised build order for the evolution

1. Correctness fixes (#4 above) + `prepare_xy` policy/report — small,
   high-trust, keeps the 45-test suite green.
2. Single-query materializer rewrite (#1) with a property test: new path ==
   old per-event loop output.
3. Optional-label context (#2) → anomaly ensemble + separability
   permutation test (the two analyses that most close the mission gap).
4. `Dataset`/`WindowSpec` facade + typed Results + ResultStore.
5. Semi-supervised, changepoint, correlation structure; report + dashboard.
6. Relations family; housekeeping throughout.

---

## 11. Appendix: original phased plan (historical)

Before this target architecture existed, the project followed a phased plan
(formerly `PLAN.md`, now folded in here). It built the original supervised
label-discrimination toolkit; §1–§10 above supersede its design, but the
phase breakdown is preserved as a record of how the codebase first came
together.

The original goal was narrower — *"given labeled events
`(asset_id, start, end, class, *extras)`, which features discriminate class
A from class B, do strata differ, and are the classes separable at all?"* —
with the same polars-first, abstract-label stance the current design keeps.

### Execution phases

- **Phase 0 — scaffolding.** `pyproject.toml`, `src/ml_analysis/` skeleton,
  `Config` dataclass (paths, timestamp col, filename pattern), `.gitignore`,
  empty `tests/`. The original script was parked in `legacy/ml_analysis.py`
  (deleted in the evolution's housekeeping).
- **Phase 1 — dataset builder.** `dataset/loader.py::load_event` (glob +
  filename date-range parse + `scan_parquet` + timestamp filter) and
  `dataset/builder.py::build` returning `dict[event_id -> LazyFrame]` with
  label metadata attached as literal columns; lazy, no materialization.
- **Phase 2 — label sources.** `labels/base.py::LabelSource` protocol and
  `labels/excel.py::ExcelLabelSource(path, sheet, column_map)` with schema
  validation (required cols, type coercion).
- **Phase 3 — feature registry.** `FeatureSpec(name, deps, expr)`, the
  `@feature` decorator, a topological dependency resolver, and the
  `features/builtins.py` stock time-series features.
- **Phase 4 — three materializers.** `to_per_sample` / `to_windowed` /
  `to_period` (the last originally a degenerate single-window case of
  `to_windowed`), plus `AggSpec` + `@aggregate`. *(The evolution rewrote
  `to_period` into the parallel `collect_all` path of §4.2.)*
- **Phase 5 — analysis module refactor.** `analysis/base.py::Analysis`
  protocol (`name`, `requires`, `run(ctx)`), a registry + DAG runner sharing
  a `ctx`, and the importance / clustering / classifier ports — each
  accepting `target_col` + `label_filter` + `stratify_by`. Legacy deleted
  once parity was verified.
- **Phase 6 — question-specific analyses.** `pairwise` (per class-pair
  importance / AUC / Cliff's delta / KS), `stratified` (run any analysis
  per stratum and diff), `distributions` (per-feature class-split tests).

### Original defaults & open items

- Defaults (all configurable): timestamp column `timestamp`, filename date
  format `YYYYMMDD`, period-aggregate as the default analysis input, Python
  3.11+, `uv`/`pip` via `pyproject.toml`.
- Label handling was uniform across analyses via `target_col` /
  `label_filter` / `stratify_by`, so "TP vs FP", "TN vs FN vs nominal", and
  "behaviour by replacement type" were configuration, not new code paths.
- Open items flagged at the time: confirm the timestamp column name and
  filename date regex against real data; infer label-schema extras
  (`replacement_type`, …) from the Excel source rather than a fixed list;
  no streaming/Dask layer yet (assumed sufficient for a few hundred GB) —
  the constraint the audit and §2 later revisited.

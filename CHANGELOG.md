# Changelog

## Unreleased

### Added
- **Figures that lead with the finding.** Each analysis's plots now open with
  the answer it exists to give, and titles state that answer when the numbers
  support one ("5 of 6 class pairs separable; not FN–TN", "96% of errors are
  FN ↔ TN"):
  - `pairwise`: a class-pair separability matrix. Each cell is the best
    single-feature AUC among features surviving BH-FDR, with the count of such
    features; a pair where none survives sits at chance, because the best raw
    AUC over many noise features is inflated by selection. It is backed by a new
    `pair_summary` frame (`analysis.pairwise.summarize_pairs`), computed before
    any `top_n` cut. The volcano / AUC-CI drill-down now shows the hardest pair
    that still separates.
  - `distributions`: small multiples of the top features' per-class quantiles
    (median, IQR, 5–95%), with each source signal's best feature included,
    drawn from `per_feature_class`.
  - `classifier` / `cv_classifier`: confusion matrices as row percentages,
    titled with the accuracy and the class pair holding most errors.
    `cv_classifier` adds a pooled out-of-fold confusion matrix and, for binary
    targets, a calibration curve; it now returns `y_true`, row-aligned with
    `oof_pred` / `oof_proba`.
  - `cluster_validation`: the ARI permutation null. The analysis now returns
    `ari_null` / `v_measure_null`.
  - `headline_metrics`: a "Separable pairs" KPI.
- **One visual system for every plot** (`io/stat_plots.py`): `theme()`, a
  colour-blind-safe categorical order, `class_colors` (a class keeps its colour
  in every figure of a run), `signal_colors`, a single-hue `SEQUENTIAL` ramp and
  a red↔blue `DIVERGING` ramp. New plots `pair_separability_matrix`,
  `confusion_matrix_plot`, `class_quantile_panel` and `embedding_scatter`, and
  helpers `pair_separability_headline`, `dominant_confusion`,
  `confusion_headline` and `metric_chance_level`.
- `demo.py` writes `demo_outputs/overview.png`, the demo's conclusion on one
  page (raw events per class, the pair matrix, the out-of-fold confusion
  matrix), plus every curated figure under `demo_outputs/figures/`, from the
  same factory as the report and dashboard.
- **`asset_loader` 0.2.0 — merge strategies, duplicate columns, provenance.**
  `load_event` / `load_asset` gained three options, all defaulting to the
  previous behaviour:
  - `merge=` chooses how sources covering the same period are combined —
    `"outer"` (the old hard-coded full outer join), `"left"` and `"inner"`
    joins, `"vertical"` row stacking, or `"asof"` (with `asof_strategy` /
    `asof_tolerance`) to carry a slow source onto a fast source's grid.
    `source_order=` sets the anchor of a `left`/`asof` merge, the priority
    used for duplicates, and the output column order.
  - `on_duplicate=` handles a column name shared by several sources
    instead of always raising: `"rename"` suffixes each copy with its
    source, `"coalesce"` folds them into one column by priority, and
    `"first"` / `"last"` keep a single source's copy. `"error"` remains
    the default.
  - `with_metadata=True` returns a `LoadResult` (a `(frame, metadata)`
    NamedTuple) whose dict records the files read per source, their
    periods and columns, the merge and duplicate resolution applied, the
    resulting schema, and an ordered `operations` log of every step.
- **AI layer restored and rebuilt** (`src/tessa/ai/`, `[ai]` / `[ai-anthropic]`
  extras, `tessa-agent` and `tessa-mcp` entry points). The layer stripped in
  0.2.0 (see its *Removed* section) is back, rewritten against the current
  architecture rather than reverted: it targets `tessa` rather than
  `ml_analysis`, reaches all 16 registered analyses through `Run.run`'s generic
  dispatcher instead of the 4 the old tools exposed, and drives the `Run` /
  `AnalysisResult` / `ResultStore` layer that did not exist before.
- **The AI can author its own features.** `create_feature` / `create_aggregator`
  accept a Polars expression as a string, validated by an AST allowlist and
  compiled into an ordinary `FeatureSpec`. Registration targets a session-scoped
  registry cloned from the process-wide one, injected through the
  `feature_registry=` parameter the materializers already accepted — so an
  invented feature flows through the whole pipeline, and can never leak into the
  global registry or another session.
- `preview_feature` compiles and tries an expression on one event without
  registering it, so a broken expression never enters the registry.
- A **feature ledger** records every invented feature with its expression,
  rationale, preview statistics and measured contribution, and `save_run` writes
  it into the run manifest — a saved run explains its own columns.
- `holdout_assets` / `confirm_on_holdout` reserve assets that the exploration
  never sees, the only real answer to selecting features on the data you then
  measure on.
- A **column budget** on `materialize`, because derived features multiply by
  aggregators; and `feature_names` now defaults to *none* rather than *all*, with
  `"__all__"` as the explicit opt-in.
- `tessa-agent --offline` drives the whole tool surface with no model and no
  credential — the fastest way to verify an install.
- `tessa.ai.demo_data.generate` builds a synthetic dataset whose two classes
  share every marginal statistic by construction, so period aggregates provably
  cannot separate them and a derived feature can.

- **Label/data availability** (`tessa.dataset.availability`). Label sheets name
  assets that were never exported and periods outside the export, and `build`
  raised on the first such event. `data_availability(labels, cfg)` annotates
  every label row with the files overlapping its window, their sources, the
  fraction of the window they cover, and why an event cannot be loaded;
  `filter_available(labels, cfg, min_coverage=...)` keeps the loadable rows and
  warns with the count dropped per reason. Both decide from file names by
  default; `check="rows"` also scans the timestamp column, catching gaps inside
  files. `Dataset.availability()` / `Dataset.available()` wrap them.
- **`asset_loader.file_catalog(root, assets=None)`**: every parquet file the
  loader can see, as a frame of `asset_id, source, path, start, end` parsed from
  file names, without reading file contents.
- **Analysis-project template** (`templates/analysis-project/`): a uv project
  that pins tessa to a release tag, a `.gitignore` keeping data and outputs out
  of git, and `notebooks/01_tp_vs_fp.ipynb`, which takes an Excel label sheet
  and a parquet store to a separability verdict, feature ranking and HTML
  report. Every placeholder is in one settings cell; as shipped it runs end to
  end on generated data.
- `tessa.examples.make_example_dataset`: a two-source, multi-rate store in
  monthly files plus a label sheet with spreadsheet-style headers and rows the
  store cannot serve, used by the template.
- **`tessa-dashboard [root]`** entry point, so the dashboard runs from any
  installed environment instead of a path into a clone.

### Changed
- The `[ai]` extra now means the OpenAI-compatible drivers (Groq, OpenRouter,
  OpenAI, local) plus the MCP server — the paths needing no Anthropic key —
  while `[ai-anthropic]` adds the Claude driver. In 0.2.0 the removed `[ai]`
  extra meant the Anthropic path; the names now split along credential lines.
- Tool schemas are derived once from each function's signature and docstring and
  adapted per provider, and one provider-neutral loop serves both backends. The
  previous layer hand-wrote Anthropic schemas, re-serialized them for the Groq
  path, and duplicated the loop.
- `cv_metric_boxplot` draws each fold as a dot below 8 folds, and marks each
  metric's own chance level (pass `n_classes`) instead of one line at 0.5, which
  was wrong for multi-class accuracy and for MCC / Cohen's kappa.
- `calibration_plot` defaults to equal-count bins, about one per 20 predictions
  (3 to 10), and reports the ECE in its title.
- `volcano_plot` greys out features below the thresholds and labels the top 3
  in a column with leader lines instead of 6 overlapping labels. A p-value that
  underflowed to 0 is pinned to the smallest observed one instead of dropped.
- The clustering elbow plot uses two stacked panels instead of two y-scales on
  one axis.
- `cluster_class_heatmap_panel` names algorithms that found fewer than two
  clusters in its title instead of drawing empty panels.
- `demo.py` and `demo_notebook.ipynb` use the library plots instead of
  hand-rolled ones: the ARI null is read from `ClusterValidation` instead of
  recomputed, and the demo's `PairwiseSeparability` no longer cuts to
  `top_n=15`, which had left the volcano plot without its non-significant cloud.

### Fixed
- **`Dataset.lazy(asset)` without a window failed on any asset with more than
  one source** (it scanned every file in the folder as one schema), and
  `Dataset.channels()` listed only the first file's columns. Both now go through
  `load_event`, so a whole-asset frame is assembled exactly like an event.
  `Config.assume_sorted` is therefore not applied anywhere at the moment, and is
  documented as reserved.
- README: the quick start called `ExcelLabelSource.load()` without its `cfg` and
  `to_period` with arguments it does not take; the dashboard command passed the
  results root positionally, which the app ignored; the install section
  suggested `pip install -e .`, which cannot resolve `asset_loader`.
- The dashboard's missing-dependency message named the old `ml-analysis` package.
- **`permutation_null_plot` drew the null invisibly** when the observed value
  sat far from it: 40 narrow bars with white edges washed out to nothing, so
  the plot showed only the observed line. Bins now span the null alone, drawn as
  one filled shape.
- **Bar charts indexed by feature were labelled 0, 1, 2, … under pandas 3**,
  whose string index dtype is `str`, not `object`. This hit the importance and
  Kruskal–Wallis charts in the HTML report and the dashboard.
- `figures_for_result` left open the figures a failing builder had created
  before raising, and those beyond `max_figures`.
- **`FeatureRegistry.resolve([])` returned every registered feature instead of
  none.** The guard read `set(names) if names else set(self._specs)`, and `[]` is
  falsy. Every materializer docstring documents the opposite ("``[]`` skips
  them"), so there was no way to request no features at all — which also made
  the derived-feature explosion unavoidable. `None` now means all, `[]` means
  none.
- `FeatureRegistry` and `AggregatorRegistry` gained `copy`/`unregister`/
  `__contains__`/`specs` (and `merge` on the former). `register` deliberately
  refuses to overwrite, so revising a feature needs an explicit unregister, and
  a private registry previously meant reaching into `_specs`.
- The five `make_*` builtins hardcoded the process-wide registry, so they could
  not populate a private one and raised on a second call for the same
  `(source, window)`. They now accept an optional `registry=`.
- **Asset leakage in cross-validation.** `cv_classifier` and `separability`
  used `StratifiedKFold(shuffle=True)` with no grouping, so events of the
  same asset landed in both train and test folds. With several events per
  asset — the norm — the reported metrics described unseen *events* of
  known assets, not unseen assets, and were optimistic. Both now default to
  `StratifiedGroupKFold` grouped on the asset id via the new
  `base.make_cv` / `base.CVPlan` helpers, which cap `n_splits` at the group
  count and fall back to ungrouped folds **with a warning** when no asset
  column is present. Opt out per analysis with `group_by_asset=False`.
  Results carry `cv_scheme` / `cv_grouped` / `cv_reason` so a number can
  always be traced to how it was validated.
- `separability` no longer delegates to `permutation_test_score`, which
  couples fold grouping to the permutation null (passing `groups` also
  restricts shuffling to within a group, degenerating the p-value to 1.0
  when each asset carries one label). Folds are grouped while the null
  stays global; `permute_within_assets=True` opts into the within-asset
  null and falls back with a warning when the data can't support it.
- `prepare_xy` excluded the literal `"asset_id"` instead of
  `cfg.asset_col`. A numeric asset identifier under a custom name
  (`Config(asset_col="vin")`) entered the feature matrix. Both names are
  now excluded via `base.id_cols_for`, and `asset_groups` resolves either
  as the grouping key.
- `cv_classifier` reports `n_rows_used` / `n_rows_dropped` from the
  `prepare_xy` report, so null-driven row loss is visible in the result.

## 0.2.0 — 2026-06-28

### Added
- UI-independent figure factory (`results/figures.py`):
  `figures_for_result` / `figures_for_run` / `headline_metrics`, shared by
  the Streamlit dashboard (now a thin renderer), the static HTML report,
  and `Run.figures()`. Analyses also emit flattened top-level frames
  (clustering `embedding`/`k_values`, pairwise `pairs_long`, classifier
  `confusion_long`) so iconic plots survive a `ResultStore` round trip.
- `make_constant_counter` built-in feature (+ tests).

### Removed
- Optional Groq/Claude AI integration layer (`agent_workflow.py`,
  `agent_workflow_groq.py`, `demo_agent.py`, `mcp_server.py`,
  `AI_INTEGRATION.md`) and the `[ai]`/`[ai-groq]` extras. The core
  polars-first library, demo, and tests are unaffected.

## 0.1.0 — 2026-06-09

Evolution of the toolkit along `ARCHITECTURE.md` §10 (see `PROGRESS.md`
for per-step verification evidence).

### Added
- **Unsupervised mode**: `AnalysisContext.target_col` optional; analyses
  declare `needs_labels` and the DAG runner skips (with warnings) what the
  data can't support.
- `AnomalyDetection` — IsolationForest + LOF + robust Mahalanobis ensemble
  with healthy-baseline fitting and per-feature robust z-score attribution.
- `SeparabilityTest` — permutation-tested CV balanced accuracy ("are the
  classes distinguishable at all?") with chance level and verdict.
- Semi-supervised: `LabelSpreadingAnalysis` (kNN label propagation) and
  `PULearningAnalysis` (bagging positive-unlabeled scoring).
- `ChangepointDetection` — per-(asset, channel) two-sided CUSUM, robust
  scale from successive differences, Monte-Carlo-calibrated threshold.
- `CorrelationStructure` — |Spearman| clusters, near-duplicate channels,
  suggested keep set.
- Relations (exploratory): `LaggedRelations` (lead/lag association) and
  `MutualInfoNetwork` (nonlinear dependence graph).
- Facades: `Dataset` (lazy asset access), `WindowSpec` + `materialize`,
  notebook-first `Run` with per-analysis shortcuts and caching.
- `AnalysisResult` (typed view), `ResultStore` (parquet/npy/json runs with
  config + version + data-fingerprint manifest), static HTML report,
  Streamlit dashboard (`dashboard` extra).
- `prepare_xy`: null policies (`drop_rows`/`drop_features`/`impute_median`)
  with an always-attached `PreparationReport`; prepared matrices cached on
  the context; `ids`/`row_index` for traceability.
- `Config.assume_sorted` to skip per-event sorts on chronological files.

### Changed
- Importance composite is rank-based (mean of per-method ranks) instead of
  min-max averaging of incomparable scales.
- `to_period` collects all events in parallel (`pl.collect_all`) with one
  schema resolution instead of a sequential per-event loop (~3× faster).
- Estimator seeds come from `Config.random_state` (explicit params still
  win); `mutual_info_classif` no longer hardcodes its seed.

### Fixed
- Hopkins statistic used power-`d` distances and overflowed at ~100
  features; now power-1.
- `bootstrap_ci` no longer silently swallows resample errors.
- `ClusterAnalysis._best_k` no longer crashes with < 3 candidate k values.
- Listwise null deletion in `prepare_xy` is reported and warned about
  instead of silent.

### Removed
- `legacy/ml_analysis.py` (superseded; `pyproject` readme now points to
  `README.md`).

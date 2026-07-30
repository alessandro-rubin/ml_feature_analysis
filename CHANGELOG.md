# Changelog

## Unreleased

### Fixed
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

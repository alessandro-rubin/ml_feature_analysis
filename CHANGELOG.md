# Changelog

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

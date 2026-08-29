# Changelog

## Unreleased

### Added
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

### Changed
- The `[ai]` extra now means the OpenAI-compatible drivers (Groq, OpenRouter,
  OpenAI, local) plus the MCP server — the paths needing no Anthropic key —
  while `[ai-anthropic]` adds the Claude driver. In 0.2.0 the removed `[ai]`
  extra meant the Anthropic path; the names now split along credential lines.
- Tool schemas are derived once from each function's signature and docstring and
  adapted per provider, and one provider-neutral loop serves both backends. The
  previous layer hand-wrote Anthropic schemas, re-serialized them for the Groq
  path, and duplicated the loop.

### Fixed
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

# Build-order progress tracker

Execution log for the queue in `ARCHITECTURE.md` §10 ("Revised build order
for the evolution"). Each step lists what changed and **what was checked**
(verification evidence). Baseline before any change: **45 tests passed**.

## Queue

- [x] **Step 1** — Correctness fixes: Hopkins power-1, seed injection from
  `cfg.random_state`, rank-based importance composite, `prepare_xy` null
  policy + `PreparationReport` + context caching, no bare `except` in
  `bootstrap_ci`, `_best_k` edge guard.
- [ ] **Step 2** — Single-query `to_period` materializer (one streaming
  collect, no per-event Python loop) + property test new == old.
- [ ] **Step 3** — Optional-label context (`needs_labels`) → anomaly
  ensemble + separability permutation test.
- [ ] **Step 4** — `Dataset`/`WindowSpec` facade + typed `Result` objects +
  `ResultStore` with run manifest.
- [ ] **Step 5** — Semi-supervised (label spreading, PU), changepoint,
  correlation structure; static HTML report + dashboard.
- [ ] **Step 6** — Relations family (lagged relations, MI network);
  housekeeping (delete `legacy/`, readme pin, CI, CHANGELOG, unused imports).

## Checked

- [x] Environment: `uv sync --all-extras` clean; baseline `pytest -q` →
  **45 passed** (matches the audit's count).
- [x] Read before modifying: `analysis/base.py`, `config.py`,
  `features/registry.py`, `features/materialize.py`, `dataset/builder.py`,
  `labels/base.py`, `analysis/{importance,classifier,cv_classifier,
  stability,clustering,cluster_validation,effect_sizes}.py`.
- [x] Test-surface scan: no test pins `score_composite` semantics; tests
  passing explicit `random_state` in `rf_params` remain valid under
  fill-if-unset seed injection.

### Step 1 (commit: see git log)

Changes:
- `analysis/base.py`: `NullPolicy` (drop_rows / drop_features /
  impute_median), `PreparationReport` (always attached to `PreparedXY`,
  warning when row-drop fraction exceeds threshold), `prepare_xy` result
  cached on `AnalysisContext._xy_cache` (+ `invalidate_cache()`), `seeded()`
  helper (fills `random_state` from `Config` only when unset).
- `importance.py` / `classifier.py` / `cv_classifier.py` / `stability.py`:
  `random_state=42` hardcodes → `None` + injection via `seeded()`;
  `mutual_info_classif` now uses `cfg.random_state` (was hardcoded 0).
- `importance.py`: composite = mean of per-method ranks (`mean_rank`
  column); `score_composite` kept as a [0,1] normalized rank score, same
  sort direction as before.
- `cluster_validation.py`: Hopkins uses power-1 distances (power-`d`
  overflowed at ~100 features).
- `effect_sizes.py`: `bootstrap_ci` counts failed resamples and warns with
  the last error instead of silently swallowing.
- `clustering.py`: `_best_k` falls back to silhouette winner when < 3
  candidate k values (old code raised on `np.diff` of a short list).

Checked:
- [x] Full suite green: **55 passed** (45 pre-existing + 10 new in
  `tests/test_step1_fixes.py`), no regressions.
- [x] Seed actually reaches the forest: same `Config.random_state` →
  bit-identical MDI; different seed → different MDI; explicit
  `rf_params["random_state"]` still wins (back-compat with existing tests).
- [x] Hopkins finite and > 0.5 on clustered 200×100 data with
  RuntimeWarnings escalated to errors (overflow would have raised).
- [x] All three null policies verified on data with partial + all-null
  features: row counts, dropped/imputed feature sets, no NaNs after impute.
- [x] `prepare_xy` returns the identical cached object on the second call.

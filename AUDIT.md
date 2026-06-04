# Audit: `ml_analysis` time-series feature-analysis toolkit

Date: 2026-06-04
Scope reviewed: full source tree (`config` / `dataset` / `features` / `analysis`
/ `io` / `legacy`), `PLAN.md`, `README.md`, `demo.py`, and the test suite
(**45 passed** at time of audit).

This audit measures the code against its stated mission:

> Analyze large time-series datasets (hundreds of millions to billions of
> samples) and find relevant features for anomaly detection and fault
> identification, using **unsupervised and semi-supervised** methods.

## Overall assessment

The code is genuinely well-built: clean separation of concerns, good
docstrings, registry/DAG patterns, and a serious statistical-rigor layer
(multiple-testing correction, effect sizes, bootstrap CIs, permutation
tests). That craftsmanship is real and worth keeping.

However, two structural gaps and a set of correctness/scaling issues stand
between the current code and the stated mission. Findings are ordered by
impact.

---

## 1. Biggest gap: it's a *supervised label-discrimination* toolkit, not an *anomaly-detection* one

Every analysis keys off `target_col` and a fully-labeled event table:

- `importance`, `pairwise`, `distributions`, `classifier`, `cv_classifier`
  are all **supervised** — they require a class label on every row.
- `clustering` / `cluster_validation` are the only unsupervised pieces, but
  they are scored *against labels* (ARI / NMI / V-measure / homogeneity),
  i.e. used to **validate label separability**, not to discover anomalies.

What the mission asks for but is **absent**:

- **No anomaly-detection method at all** — no IsolationForest, LOF,
  One-Class SVM, reconstruction-error / autoencoder, density or distance
  scoring, or residual / changepoint analysis. These are the natural
  unsupervised tools for "find anomalies."
- **No semi-supervised path** — no label propagation, PU learning, or
  "use a handful of labels to guide an otherwise unsupervised model." The
  pipeline assumes *every* event already carries a class.

Net effect: the toolkit answers *"given already-labeled event classes, which
aggregate features separate them?"* — valuable for **post-hoc fault
characterization**, but it sits **downstream** of the actual
anomaly-detection problem. It does not find candidate anomalies in unlabeled
streams, which is what the headline asks for.

**Recommendation:** add an `analysis/anomaly.py` family (IsolationForest /
LOF / OC-SVM scoring + per-feature contribution, reconstruction error), and a
semi-supervised mode (e.g. `LabelSpreading`, or PU-learning wrappers) that
consumes a `label_filter` where most rows are unlabeled. This is the change
that most closes the gap to the stated goal.

---

## 2. Unit of analysis is the *pre-cut, labeled event* — the scale claim mostly bites in materialization, not analysis

The default analysis input is the **period aggregate: one row per event**.
So the sklearn layer never sees billions of samples — it sees `n_events`
rows. That is the *right* shape: push the heavy reduction into polars and
keep sklearn on the reduced table.

The billion-sample pressure therefore lands entirely in **materialization**,
and that's where the design has a throughput problem:

- **`to_period` is a sequential Python loop with a `.collect()` per event**
  (`features/materialize.py:254-268`). Polars cannot parallelize across a
  Python `for` loop, so with tens of thousands of events this is the
  wall-clock bottleneck and it is single-core-bound.
  - **Fix:** collect events in parallel with `pl.collect_all([...])`, *or*
    express the period aggregate as one query — `scan_parquet(all files) ->
    join event windows -> group_by("event_id").agg(...)` — and collect once
    under the **streaming engine** (`.collect(engine="streaming")`), which
    gives out-of-core + multicore. The per-event isolation the loop currently
    enforces (so rolling/diff features don't bleed across events) can be
    preserved with `.over("event_id")`.
- **`collect_schema()` is called inside the per-event loop**
  (`features/materialize.py:161, 256`) although the schema is identical
  across events — redundant metadata reads.
- **`load_event` does a `.sort(timestamp)` per event**
  (`dataset/loader.py:131`). Sort forces full materialization and is
  O(n log n); for ~1 yr @ 1 Hz an event can be tens of millions of rows. If
  files are already chronological, consider `set_sorted` or pushing the sort
  responsibility to `group_by_dynamic` only where needed.
- **`prepare_xy` does listwise null deletion across ~90 features**
  (`analysis/base.py:70`, `X.notna().all(axis=1)`). A single null feature
  (e.g. `std` of a one-sample window) drops the *entire event* row, silently.
  At 90 features this can delete most events with no warning.
  - **Fix:** report drop counts, and offer column-drop or imputation instead
    of row-drop.

`PLAN.md` explicitly scopes this ("no streaming/Dask layer yet… should
suffice for a few hundred GB"). That is a reasonable v1 stance, but it
directly contradicts the "billions of samples" framing, so it should be the
first thing revisited if scale is real.

---

## 3. Correctness / robustness issues

- **`hopkins_statistic` raises distances to the power of `d` = n_features**
  (`analysis/cluster_validation.py:76-77`, `u_dists ** d`). With ~90 features
  this overflows to `inf` / `0` and the statistic collapses to `nan` or a
  meaningless value — exactly in the high-dimensional regime where it is run.
  Robust implementations use power 1 (or normalize). **Real bug for this data
  shape.**
- **`cfg.random_state` is ignored by the actual estimators.** Every analysis
  hardcodes `random_state=42` in the RF `default_factory`
  (`analysis/importance.py:29`, `analysis/classifier.py:40`,
  `analysis/cv_classifier.py:79`, `analysis/stability.py:44`);
  `mutual_info_classif(..., random_state=0)` (`analysis/importance.py:65`) is
  also hardcoded. Setting `Config(random_state=7)` changes the CV split and
  permutation RNG but **not** the forests — so "reproducible with my seed" is
  misleading.
  - **Fix:** default these to `None` and inject `ctx.cfg.random_state` at
    `run()` time.
- **Composite importance score mixes incomparable scales via min-max then
  averages** (`analysis/importance.py:71-77`). `anova_f` and `kw_stat` are
  unbounded; one outlier feature with a huge F dominates min-max and drowns
  the rank signal. **Rank-based aggregation** (mean of per-method ranks) is
  far more robust and is the standard for blended importance.
- **Redundant heavy compute across the DAG.** Chaining `importance` +
  `classifier` + `cv_classifier` + `importance_stability` refits Random
  Forests 1 + 1 + 5 + 50 times on the same matrix, and each analysis re-runs
  `ctx.filtered().to_pandas()` from scratch. The DAG shares `ctx.results` but
  nothing caches the prepared `X/y` or a fitted model.
  - **Fix:** cache `PreparedXY` on the context; let stability/importance share
    one fit.
- **`_label_cols` heuristic misclassifies non-numeric signal columns as
  labels** (`features/materialize.py:44-50`) — any string/categorical sensor
  channel is carried via `.first()` instead of aggregated. Fine for
  purely-numeric sensors, latent bug otherwise.
- **`_best_k` needs >= 3 k values** (`analysis/clustering.py:59`, double
  `np.diff`); a `k_range` yielding < 3 candidates throws. Minor edge guard.
- **`bootstrap_ci` swallows bare `Exception`** (`analysis/effect_sizes.py:119`),
  masking real errors during resampling.

---

## 4. Packaging / hygiene

- **`legacy/ml_analysis.py` (786 lines) is still shipped** though `PLAN.md`
  says delete after Phase-5 parity — dead weight and a second, drifting source
  of truth (it carries its own hardcoded params).
- **`pyproject.toml: readme = "PLAN.md"`** — the user-facing README is
  `README.md`; the published long-description will be the internal plan.
- **No `CHANGELOG.md`** despite `PLAN.md` promising one per phase; **no CI
  config** for the existing test/lint commands.
- Unused `field` imports in `features/aggregates.py:17` and
  `features/registry.py:17` (ruff would catch).
- The Anderson–Darling p-value warnings in the test run are expected
  (floor/cap) but worth suppressing or surfacing as a capped flag so users
  don't misread `0.001` / `0.25` as exact.

> Note: `pyarrow>=24.0.0` looked like a typo but was verified to be a real,
> installable version — no action needed.

---

## Suggested priority order

1. **Add unsupervised anomaly detection + semi-supervised mode** (closes the
   mission gap) — *highest value*.
2. **Fix the Hopkins power-`d` bug** and the **`random_state` propagation** —
   correctness, low effort.
3. **Parallelize / stream `to_period`** (`collect_all` or a single streaming
   `group_by`) and **surface `prepare_xy` row-drops** — the real scale levers.
4. **Rank-based composite importance** + **cache `PreparedXY` / model across
   the DAG**.
5. Housekeeping: delete `legacy/`, fix the `readme` pin, add CHANGELOG / CI,
   drop unused imports.

Items 2, 4, and the housekeeping in 5 are low-risk. Items 1 and 3 are larger
design changes worth a short alignment before implementation.

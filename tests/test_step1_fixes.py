"""Step-1 correctness fixes: seeds, Hopkins, null policy, composite, guards."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from ml_analysis.analysis import (
    AnalysisContext,
    FeatureImportance,
    NullPolicy,
    prepare_xy,
)
from ml_analysis.analysis.cluster_validation import hopkins_statistic
from ml_analysis.analysis.clustering import ClusterAnalysis
from ml_analysis.analysis.effect_sizes import bootstrap_ci, cohens_d
from ml_analysis.config import Config


def _toy_df(n_per_class: int = 30, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cls, shift in (("A", 0.0), ("B", 2.0)):
        for _ in range(n_per_class):
            rows.append({
                "class": cls,
                "f_sep": float(rng.normal(shift, 1.0)),
                "f_noise": float(rng.normal(0.0, 1.0)),
                "f_also": float(rng.normal(shift * 0.5, 1.0)),
            })
    return pl.DataFrame(rows)


def _ctx(df: pl.DataFrame, seed: int = 42, **kwargs) -> AnalysisContext:
    return AnalysisContext(df=df, cfg=Config(random_state=seed), target_col="class", **kwargs)


# ── Seed injection ──────────────────────────────────────────────────────────

def test_config_seed_reaches_the_forest():
    df = _toy_df()
    imp = FeatureImportance(permutation_repeats=2, rf_params={"n_estimators": 20, "n_jobs": 1})

    t_a1 = imp.run(_ctx(df, seed=1))["table"]
    t_a2 = imp.run(_ctx(df, seed=1))["table"]
    t_b = imp.run(_ctx(df, seed=2))["table"]

    # same seed -> bit-identical MDI; different seed -> different forest
    assert (t_a1["rf_mdi"] == t_a2["rf_mdi"]).all()
    assert not (t_a1["rf_mdi"] == t_b["rf_mdi"]).all()


def test_explicit_random_state_wins_over_config():
    df = _toy_df()
    imp = FeatureImportance(
        permutation_repeats=2,
        rf_params={"n_estimators": 20, "n_jobs": 1, "random_state": 0},
    )
    t1 = imp.run(_ctx(df, seed=1))["table"]
    t2 = imp.run(_ctx(df, seed=2))["table"]
    assert (t1["rf_mdi"] == t2["rf_mdi"]).all()


# ── Rank-based composite ────────────────────────────────────────────────────

def test_composite_is_rank_based_and_bounded():
    df = _toy_df()
    out = FeatureImportance(
        permutation_repeats=2, rf_params={"n_estimators": 30, "n_jobs": 1}
    ).run(_ctx(df))
    tbl = out["table"]
    assert "mean_rank" in tbl.columns
    assert tbl["score_composite"].between(0, 1).all()
    # sorted best-first by mean rank, and the separable feature wins
    assert tbl["mean_rank"].is_monotonic_increasing
    assert tbl.index[0] == "f_sep"


# ── Hopkins in high dimension ───────────────────────────────────────────────

def test_hopkins_no_overflow_at_100_features():
    rng = np.random.default_rng(0)
    # two tight Gaussian blobs in 100-D -> strongly clustered
    X = np.vstack([
        rng.normal(0, 0.1, size=(100, 100)),
        rng.normal(5, 0.1, size=(100, 100)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # overflow would raise
        h = hopkins_statistic(X, rng=rng)
    assert np.isfinite(h)
    assert h > 0.5


# ── prepare_xy null policies + report + cache ───────────────────────────────

def _df_with_nulls() -> pl.DataFrame:
    df = _toy_df(n_per_class=20)
    # f_noise null on first 10 rows; one fully-null feature
    noise = df["f_noise"].to_list()
    for i in range(10):
        noise[i] = None
    return df.with_columns(
        pl.Series("f_noise", noise, dtype=pl.Float64),
        pl.lit(None, dtype=pl.Float64).alias("f_all_null"),
    )


def test_drop_rows_policy_reports_and_warns():
    ctx = _ctx(_df_with_nulls())
    with pytest.warns(UserWarning, match="dropped"):
        prep = prepare_xy(ctx)
    # f_all_null nulls every row under drop_rows -> everything would vanish;
    # the report must say so loudly.
    rep = prep.report
    assert rep is not None
    assert rep.n_rows_in == 40
    assert rep.n_rows_out == len(prep.X)
    assert "f_all_null" in rep.feature_null_fracs


def test_drop_features_policy_keeps_rows():
    ctx = _ctx(_df_with_nulls(), null_policy=NullPolicy(kind="drop_features"))
    prep = prepare_xy(ctx)
    assert "f_all_null" in prep.report.dropped_features
    assert "f_all_null" not in prep.feature_cols
    # only the 10 rows where f_noise is null are lost
    assert prep.report.n_rows_out == 30


def test_impute_policy_keeps_all_rows():
    ctx = _ctx(_df_with_nulls(), null_policy=NullPolicy(kind="impute_median"))
    prep = prepare_xy(ctx)
    assert prep.report.n_rows_out == 40
    assert "f_noise" in prep.report.imputed_features
    assert "f_all_null" in prep.report.dropped_features
    assert not prep.X.isna().any().any()


def test_prepare_xy_is_cached_on_context():
    ctx = _ctx(_toy_df())
    assert prepare_xy(ctx) is prepare_xy(ctx)
    ctx.invalidate_cache()
    assert ctx._xy_cache == {}


# ── bootstrap_ci surfaces resample failures ─────────────────────────────────

def test_bootstrap_ci_warns_on_failing_resamples():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    b = rng.normal(1, 1, 30)
    calls = {"n": 0}

    def flaky(x, y):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise ValueError("boom")
        return cohens_d(x, y)

    with pytest.warns(UserWarning, match="resamples raised"):
        point, lo, hi = bootstrap_ci(flaky, a, b, n_resamples=30, rng=rng)
    assert np.isfinite(point) and np.isfinite(lo) and np.isfinite(hi)


# ── _best_k edge guard ──────────────────────────────────────────────────────

def test_best_k_with_two_candidates():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(0, 0.2, size=(40, 3)),
        rng.normal(4, 0.2, size=(40, 3)),
        rng.normal(8, 0.2, size=(40, 3)),
    ])
    ca = ClusterAnalysis(k_range=(2, 4))  # only k=2,3 -> old code raised
    best_k, inertias, sils = ca._best_k(X, random_state=0)
    assert best_k in (2, 3)
    assert len(inertias) == 2

import numpy as np
import polars as pl

from tessa import Config
from tessa.analysis import (
    AnalysisContext,
    ClusterValidation,
    CrossValidatedClassifier,
    DistributionAnalysis,
    FeatureImportance,
    ImportanceStability,
    PairwiseSeparability,
    run_analyses,
)


def _separable_df(n: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    half = n // 2
    return pl.DataFrame(
        {
            "f_sep":   list(rng.normal(0, 1, half)) + list(rng.normal(3.0, 1, half)),
            "f_sep2":  list(rng.normal(0, 1, half)) + list(rng.normal(2.5, 1, half)),
            "f_weak":  list(rng.normal(0, 1, half)) + list(rng.normal(0.4, 1, half)),
            "f_noise": list(rng.normal(0, 1, n)),
            "class":   ["A"] * half + ["B"] * half,
        }
    )


def test_distributions_emits_corrected_pvalues_and_shape_stats():
    ctx = AnalysisContext(df=_separable_df(), cfg=Config(), target_col="class")
    out = DistributionAnalysis().run(ctx)
    summary = out["summary"]
    for col in ("kw_p_bonferroni", "kw_p_bh_fdr", "anova_p_bh_fdr", "ad_p", "levene_p"):
        assert col in summary.columns
    # Bonferroni should not be smaller than the raw p-value (per feature)
    pairs = summary[["kw_p", "kw_p_bonferroni"]].dropna()
    assert (pairs["kw_p_bonferroni"] + 1e-12 >= pairs["kw_p"]).all()
    # the strongly separable feature is significant after correction
    f_sep_row = summary[summary["feature"] == "f_sep"].iloc[0]
    assert f_sep_row["kw_p_bh_fdr"] < 0.05
    # shape stats per (feature, class)
    pfc = out["per_feature_class"]
    for col in ("skew", "kurtosis", "mad", "iqr"):
        assert col in pfc.columns


def test_pairwise_emits_full_battery_and_ci():
    ctx = AnalysisContext(df=_separable_df(), cfg=Config(), target_col="class")
    out = PairwiseSeparability(bootstrap_n=100).run(ctx)
    table = out["pairs"][("A", "B")]
    for col in (
        "mwu_p", "bm_p", "welch_p", "rank_biserial", "cohens_d", "hedges_g",
        "wasserstein", "js_divergence",
        "mwu_p_bh_fdr", "ks_p_bh_fdr",
        "auc_ci_low", "auc_ci_high", "cliffs_ci_low", "cliffs_ci_high",
    ):
        assert col in table.columns
    sep_row = table[table["feature"] == "f_sep"].iloc[0]
    assert sep_row["auc_ci_low"] <= sep_row["auc"] <= sep_row["auc_ci_high"]
    assert abs(sep_row["cohens_d"]) > 1.0  # ~3 SD shift
    assert sep_row["wasserstein"] > 1.0


def test_cv_classifier_runs_and_reports_metrics():
    ctx = AnalysisContext(df=_separable_df(), cfg=Config(), target_col="class")
    out = CrossValidatedClassifier(
        n_splits=3,
        rf_params={"n_estimators": 30, "n_jobs": -1, "random_state": 0},
    ).run(ctx)
    summary = out["summary"]
    assert summary.loc["accuracy", "mean"] > 0.7
    assert "mcc" in summary.index
    assert "balanced_accuracy" in summary.index
    assert "roc_auc" in summary.index  # binary case
    assert "brier" in summary.index
    # per-fold table has n_splits rows
    assert len(out["per_fold"]) == 3


def test_importance_stability_uses_cached_importance():
    df = _separable_df()
    ctx = AnalysisContext(df=df, cfg=Config(), target_col="class")
    analyses = [
        FeatureImportance(
            permutation_repeats=2,
            rf_params={"n_estimators": 50, "n_jobs": -1, "random_state": 0},
        ),
        ImportanceStability(
            n_bootstrap=15,
            top_k=2,
            rf_params={"n_estimators": 30, "n_jobs": -1, "random_state": 0},
        ),
    ]
    results = run_analyses(analyses, ctx)
    boot = results["importance_stability"]["bootstrap_table"]
    assert "stability_top2" in boot.columns
    # f_sep should be in the top-2 in the vast majority of resamples
    sep_row = boot[boot["feature"] == "f_sep"].iloc[0]
    assert sep_row["stability_top2"] > 0.8
    agree = results["importance_stability"]["method_agreement"]
    assert not agree.empty
    assert agree.shape[0] == agree.shape[1]
    # diagonal == 1
    assert np.allclose(np.diag(agree.values), 1.0)


def test_cluster_validation_detects_real_alignment():
    ctx = AnalysisContext(df=_separable_df(), cfg=Config(), target_col="class")
    out = ClusterValidation(n_permutations=200).run(ctx)
    s = out["summary"].iloc[0]
    assert 0.0 <= s["hopkins"] <= 1.0
    assert s["hopkins"] > 0.5  # there *is* structure
    assert s["ari_perm_p"] < 0.05  # real alignment, not random
    assert s["v_measure_perm_p"] < 0.05

import numpy as np
import polars as pl

from ml_analysis import Config
from ml_analysis.analysis import (
    AnalysisContext,
    DistributionAnalysis,
    FeatureImportance,
    PairwiseSeparability,
    Stratified,
)


def _separable_df(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    half = n // 2
    return pl.DataFrame(
        {
            "f_sep":   list(rng.normal(0, 1, half)) + list(rng.normal(5, 1, half)),
            "f_noise": list(rng.normal(0, 1, n)),
            "class":   ["A"] * half + ["B"] * half,
            "stratum": (["s1"] * (n // 4) + ["s2"] * (n // 4)) * 2,
        }
    )


def test_pairwise_ranks_separable_first():
    ctx = AnalysisContext(df=_separable_df(), cfg=Config(), target_col="class")
    out = PairwiseSeparability().run(ctx)
    table = out["pairs"][("A", "B")]
    assert table.iloc[0]["feature"] == "f_sep"
    assert table.iloc[0]["auc"] > 0.9
    assert table.iloc[-1]["feature"] == "f_noise"


def test_distributions_summary_orders_by_kw():
    ctx = AnalysisContext(df=_separable_df(), cfg=Config(), target_col="class")
    out = DistributionAnalysis().run(ctx)
    assert out["summary"].iloc[0]["feature"] == "f_sep"
    rows_per_feature = out["per_feature_class"].groupby("feature").size()
    assert (rows_per_feature == 2).all()


def test_stratified_runs_per_value():
    inner = FeatureImportance(permutation_repeats=2, rf_params={"n_estimators": 50, "n_jobs": -1, "random_state": 0})
    ctx = AnalysisContext(
        df=_separable_df(),
        cfg=Config(),
        target_col="class",
        stratify_by="stratum",
    )
    out = Stratified(inner=inner).run(ctx)
    assert out["by"] == "stratum"
    assert set(out["per_stratum"].keys()) == {"s1", "s2"}
    for r in out["per_stratum"].values():
        assert "table" in r

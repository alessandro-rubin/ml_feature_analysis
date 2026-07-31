"""Step-3: optional-label context, anomaly ensemble, separability test."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tessa.analysis import (
    AnalysisContext,
    AnomalyDetection,
    FeatureImportance,
    SeparabilityTest,
    prepare_xy,
    run_analyses,
)
from tessa.config import Config


def _unlabeled_df(n: int = 200, n_outliers: int = 8, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(5, 2, n)
    f3 = rng.normal(-3, 0.5, n)
    # inject outliers into f2 only, on the last rows
    f2[-n_outliers:] += 25.0
    return pl.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(n)],
            "asset_id": [f"A{i % 4}" for i in range(n)],
            "f1": f1,
            "f2": f2,
            "f3": f3,
        }
    )


def _labeled_df(n_per_class: int = 40, separable: bool = True, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for cls, shift in (("healthy", 0.0), ("fault", 3.0 if separable else 0.0)):
        for _ in range(n_per_class):
            rows.append(
                {
                    "class": cls,
                    "f_a": float(rng.normal(shift, 1.0)),
                    "f_b": float(rng.normal(0, 1.0)),
                }
            )
    return pl.DataFrame(rows)


# ── optional-label context ──────────────────────────────────────────────────


def test_prepare_xy_without_target():
    ctx = AnalysisContext(df=_unlabeled_df(), cfg=Config())
    prep = prepare_xy(ctx)
    assert prep.y is None and prep.encoder is None
    assert prep.class_names == []
    assert prep.feature_cols == ["f1", "f2", "f3"]
    assert list(prep.ids.columns) == ["event_id", "asset_id"]


def test_runner_skips_supervised_without_labels():
    ctx = AnalysisContext(df=_unlabeled_df(), cfg=Config())
    analyses = [
        FeatureImportance(rf_params={"n_estimators": 10, "n_jobs": 1}),
        AnomalyDetection(
            run_lof=False, run_mahalanobis=False, iforest_params={"n_estimators": 50, "n_jobs": 1}
        ),
    ]
    with pytest.warns(UserWarning, match="requires labels"):
        results = run_analyses(analyses, ctx)
    assert "importance" not in results
    assert "anomaly" in results


def test_runner_skips_dependents_of_skipped():
    class Downstream:
        name = "downstream"
        requires = ("importance",)
        needs_labels = "none"

        def run(self, ctx):
            return "ran"

    ctx = AnalysisContext(df=_unlabeled_df(), cfg=Config())
    with pytest.warns(UserWarning):
        results = run_analyses(
            [FeatureImportance(rf_params={"n_estimators": 10}), Downstream()], ctx
        )
    assert "downstream" not in results


# ── anomaly ensemble ────────────────────────────────────────────────────────


def test_anomaly_finds_injected_outliers():
    n, n_out = 200, 8
    ctx = AnalysisContext(df=_unlabeled_df(n, n_out), cfg=Config())
    out = AnomalyDetection(
        contamination=0.05,
        iforest_params={"n_estimators": 100, "n_jobs": 1},
    ).run(ctx)

    scores = out["scores"]
    assert {"iforest", "lof", "mahalanobis", "ensemble"} <= set(scores.columns)
    top = scores.nlargest(n_out, "ensemble")["event_id"].tolist()
    expected = {f"e{i}" for i in range(n - n_out, n)}
    # at least 6 of the 8 injected outliers in the top-8
    assert len(expected & set(top)) >= 6


def test_anomaly_attribution_points_at_perturbed_feature():
    n, n_out = 200, 8
    ctx = AnalysisContext(df=_unlabeled_df(n, n_out), cfg=Config())
    out = AnomalyDetection(iforest_params={"n_estimators": 100, "n_jobs": 1}).run(ctx)
    for row in out["top_contributors"][-n_out:]:
        assert row[0][0] == "f2"  # the feature we shifted by +25


def test_anomaly_scores_reproducible_with_seed():
    df = _unlabeled_df()
    out1 = AnomalyDetection(iforest_params={"n_estimators": 50, "n_jobs": 1}).run(
        AnalysisContext(df=df, cfg=Config(random_state=7))
    )
    out2 = AnomalyDetection(iforest_params={"n_estimators": 50, "n_jobs": 1}).run(
        AnalysisContext(df=df, cfg=Config(random_state=7))
    )
    assert np.allclose(out1["scores"]["ensemble"], out2["scores"]["ensemble"])


# ── separability ────────────────────────────────────────────────────────────


def test_separable_classes_get_small_p():
    ctx = AnalysisContext(df=_labeled_df(separable=True), cfg=Config(), target_col="class")
    out = SeparabilityTest(n_permutations=50, rf_params={"n_estimators": 50, "n_jobs": 1}).run(ctx)
    s = out["summary"].iloc[0]
    assert s["cv_balanced_accuracy"] > 0.85
    assert s["perm_p_value"] < 0.05
    assert s["verdict"] == "separable"


def test_random_labels_are_not_separable():
    ctx = AnalysisContext(df=_labeled_df(separable=False), cfg=Config(), target_col="class")
    out = SeparabilityTest(n_permutations=50, rf_params={"n_estimators": 50, "n_jobs": 1}).run(ctx)
    s = out["summary"].iloc[0]
    assert s["perm_p_value"] > 0.05
    assert s["verdict"] == "not separable"

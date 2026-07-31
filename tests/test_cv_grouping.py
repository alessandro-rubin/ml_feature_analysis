"""Cross-validation must not leak assets between train and test folds.

The regression these cover: with several events per asset, ungrouped
``StratifiedKFold`` lets a model see an asset in training and be scored on
another event of that same asset, which inflates the reported metrics.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from tessa.analysis import (
    AnalysisContext,
    CrossValidatedClassifier,
    SeparabilityTest,
    asset_groups,
    make_cv,
    prepare_xy,
)
from tessa.config import Config


def _asset_confounded_frame(n_assets: int = 12, per_asset: int = 5, seed: int = 0) -> pl.DataFrame:
    """Events whose features encode the *asset*, not the class.

    Each asset gets its own feature offset and a class assigned at the asset
    level. A model can therefore reach perfect accuracy by memorising the
    asset — which it can only do when folds leak.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_assets):
        offset = float(a) * 10.0
        cls = "TP" if a % 2 == 0 else "FP"
        for _ in range(per_asset):
            rows.append(
                {
                    "asset_id": f"asset_{a}",
                    "event_id": f"e{len(rows)}",
                    "class": cls,
                    "f1": offset + rng.normal(0, 0.01),
                    "f2": offset * 2 + rng.normal(0, 0.01),
                }
            )
    return pl.DataFrame(rows)


def _ctx(df: pl.DataFrame, **cfg_kw) -> AnalysisContext:
    return AnalysisContext(df=df, cfg=Config(**cfg_kw), target_col="class")


# ── make_cv ──────────────────────────────────────────────────────────────────


def test_make_cv_groups_by_asset_and_folds_are_disjoint():
    ctx = _ctx(_asset_confounded_frame())
    prep = prepare_xy(ctx)
    plan = make_cv(prep, ctx, n_splits=5)

    assert plan.grouped
    assert plan.scheme == "stratified_group_kfold"
    assert plan.n_groups == 12

    X, y = prep.X.to_numpy(), prep.y
    seen_test = []
    for tr, te in plan.split(X, y):
        train_assets = set(plan.groups[tr])
        test_assets = set(plan.groups[te])
        assert not (train_assets & test_assets), "asset present in train and test"
        seen_test.append(test_assets)
    # every row is tested exactly once
    assert sum(len(s) for s in seen_test) == 12


def test_make_cv_caps_splits_at_number_of_assets():
    ctx = _ctx(_asset_confounded_frame(n_assets=3, per_asset=6))
    prep = prepare_xy(ctx)
    plan = make_cv(prep, ctx, n_splits=5)
    assert plan.n_splits == 3


def test_make_cv_warns_and_falls_back_without_asset_column():
    df = _asset_confounded_frame().drop("asset_id")
    ctx = _ctx(df)
    prep = prepare_xy(ctx)
    with pytest.warns(UserWarning, match="Ungrouped cross-validation"):
        plan = make_cv(prep, ctx, n_splits=3)
    assert not plan.grouped
    assert plan.scheme == "stratified_kfold"
    assert "no asset column" in plan.reason


def test_make_cv_falls_back_for_a_single_asset():
    df = _asset_confounded_frame(n_assets=1, per_asset=10).with_columns(
        pl.Series("class", ["TP", "FP"] * 5)
    )
    ctx = _ctx(df)
    prep = prepare_xy(ctx)
    with pytest.warns(UserWarning, match="single asset"):
        plan = make_cv(prep, ctx, n_splits=3)
    assert not plan.grouped


def test_make_cv_opt_out_is_silent():
    ctx = _ctx(_asset_confounded_frame())
    prep = prepare_xy(ctx)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = make_cv(prep, ctx, n_splits=3, group_by_asset=False)
    assert not plan.grouped
    assert "disabled" in plan.reason


# ── the leak itself ──────────────────────────────────────────────────────────


def test_grouped_cv_reports_lower_accuracy_than_leaky_cv():
    """The point of the fix: grouping removes the inflated score."""
    ctx = _ctx(_asset_confounded_frame())

    grouped = CrossValidatedClassifier(n_splits=4).run(ctx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        leaky = CrossValidatedClassifier(n_splits=4, group_by_asset=False).run(ctx)

    grouped_acc = grouped["summary"].loc["accuracy", "mean"]
    leaky_acc = leaky["summary"].loc["accuracy", "mean"]

    # features encode the asset only, so nothing generalises to a held-out
    # asset — while the leaky split can memorise its way to a high score.
    assert leaky_acc > 0.9
    assert grouped_acc < leaky_acc - 0.2


def test_cv_classifier_reports_its_scheme():
    ctx = _ctx(_asset_confounded_frame())
    res = CrossValidatedClassifier(n_splits=4).run(ctx)
    assert res["cv_scheme"] == "stratified_group_kfold"
    assert res["cv_grouped"] is True
    assert res["cv_n_assets"] == 12
    assert res["n_rows_used"] == 60
    assert res["n_rows_dropped"] == 0


def test_cv_classifier_oof_covers_every_row_once():
    ctx = _ctx(_asset_confounded_frame())
    res = CrossValidatedClassifier(n_splits=4).run(ctx)
    assert len(res["oof_pred"]) == 60
    assert res["oof_proba"].shape == (60, 2)
    # every row got a prediction: probabilities sum to 1 everywhere
    assert np.allclose(res["oof_proba"].sum(axis=1), 1.0)


# ── separability ─────────────────────────────────────────────────────────────


def test_separability_uses_grouped_folds_and_a_global_null():
    ctx = _ctx(_asset_confounded_frame())
    res = SeparabilityTest(n_splits=4, n_permutations=15).run(ctx)
    s = res["summary"].iloc[0]

    assert res["cv_grouped"] is True
    assert s["cv_scheme"] == "stratified_group_kfold"
    assert s["permutation_null"] == "global"
    assert 0.0 < s["perm_p_value"] <= 1.0
    # asset-confounded features must not look separable across unseen assets
    assert s["verdict"] == "not separable"


def test_separability_within_asset_null_falls_back_when_degenerate():
    """One class per asset makes within-asset shuffling a no-op."""
    ctx = _ctx(_asset_confounded_frame())
    with pytest.warns(UserWarning, match="within-asset shuffling"):
        res = SeparabilityTest(n_splits=4, n_permutations=10, permute_within_assets=True).run(ctx)
    assert res["summary"].iloc[0]["permutation_null"] == "global"


def test_separability_detects_a_real_signal_under_grouping():
    """A genuine per-event signal survives asset-disjoint folds."""
    rng = np.random.default_rng(1)
    rows = []
    for a in range(10):
        for j in range(6):
            cls = "TP" if j % 2 == 0 else "FP"
            shift = 3.0 if cls == "TP" else -3.0
            rows.append(
                {
                    "asset_id": f"asset_{a}",
                    "event_id": f"e{len(rows)}",
                    "class": cls,
                    "f1": shift + rng.normal(0, 0.3),
                    "f2": rng.normal(0, 1.0),
                }
            )
    ctx = _ctx(pl.DataFrame(rows))
    res = SeparabilityTest(n_splits=4, n_permutations=25).run(ctx)
    s = res["summary"].iloc[0]
    assert s["verdict"] == "separable"
    assert s["cv_scheme"] == "stratified_group_kfold"


# ── custom asset column name ─────────────────────────────────────────────────


def test_numeric_custom_asset_column_is_not_a_feature():
    """Config(asset_col="vin") with a numeric vin must not leak into X."""
    df = _asset_confounded_frame().rename({"asset_id": "vin_str"})
    df = df.with_columns(
        pl.col("vin_str").str.replace("asset_", "").cast(pl.Int64).alias("vin")
    ).drop("vin_str")

    ctx = _ctx(df, asset_col="vin")
    prep = prepare_xy(ctx)

    assert "vin" not in prep.feature_cols
    assert set(prep.feature_cols) == {"f1", "f2"}
    # and it is still available as a grouping key
    groups = asset_groups(prep, ctx)
    assert groups is not None
    assert len(np.unique(groups)) == 12


def test_custom_asset_column_drives_grouping():
    df = _asset_confounded_frame().rename({"asset_id": "vin"})
    ctx = _ctx(df, asset_col="vin")
    prep = prepare_xy(ctx)
    plan = make_cv(prep, ctx, n_splits=4)
    assert plan.grouped
    assert plan.n_groups == 12

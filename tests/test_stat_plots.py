"""Shared visual system and the finding-first plots in ``tessa.io.stat_plots``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: tests must not require a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytest

from tessa import Config
from tessa.analysis import (
    AnalysisContext,
    ClusterValidation,
    CrossValidatedClassifier,
    PairwiseSeparability,
)
from tessa.analysis.pairwise import summarize_pairs
from tessa.io import stat_plots as sp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _two_class_df(n: int = 40, n_informative: int = 4) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    cols = {
        f"f{i}": np.r_[rng.normal(0, 1, n), rng.normal(2.0, 1, n)] for i in range(n_informative)
    }
    return pl.DataFrame({"class": ["A"] * n + ["B"] * n, **cols, "noise": rng.normal(0, 1, 2 * n)})


def _pair_tables() -> dict[tuple[str, str], pd.DataFrame]:
    feats = [f"f{i}" for i in range(5)]
    separable = pd.DataFrame(
        {
            "feature": feats,
            "auc": [0.95, 0.9, 0.6, 0.55, 0.5],
            "mwu_p_bh_fdr": [1e-6, 1e-4, 0.2, 0.5, 0.9],
        }
    )
    # The best raw AUC here is the selection-inflated max of noise.
    noise = pd.DataFrame(
        {
            "feature": feats,
            "auc": [0.66, 0.6, 0.55, 0.52, 0.5],
            "mwu_p_bh_fdr": [0.3, 0.5, 0.8, 0.9, 0.9],
        }
    )
    return {("A", "B"): separable, ("A", "C"): separable, ("B", "C"): noise}


# ── visual system ────────────────────────────────────────────────────────────


def test_class_colors_are_stable_whatever_the_order():
    a = sp.class_colors(["TP", "FP", "TN", "FN"])
    assert a == sp.class_colors(["FN", "TN", "FP", "TP"])
    assert len(set(a.values())) == 4


def test_class_colors_fold_past_the_palette_instead_of_cycling():
    colors = list(sp.class_colors([f"c{i}" for i in range(10)]).values())
    assert colors[:8] == list(sp.CATEGORICAL)
    assert set(colors[8:]) == {sp.OTHER}


def test_signal_colors_start_from_the_far_end():
    assert sp.signal_colors(["temperature"]) == {"temperature": sp.CATEGORICAL[-1]}
    assert sp.feature_signal("temperature__roll_mean_10__max") == "temperature"


def test_theme_does_not_leak_into_global_rcparams():
    before = plt.rcParams["axes.spines.top"]
    sp.auc_bootstrap_plot(pd.DataFrame({"feature": ["a"], "auc": [0.9]}))
    assert plt.rcParams["axes.spines.top"] == before


# ── pairwise ─────────────────────────────────────────────────────────────────


def test_summarize_pairs_counts_only_corrected_hits():
    s = summarize_pairs(_pair_tables()).set_index(["class_a", "class_b"])
    assert s.loc[("A", "B"), "n_significant"] == 2
    assert s.loc[("A", "B"), "best_feature"] == "f0"
    assert s.loc[("A", "B"), "best_auc_significant"] == pytest.approx(0.95)
    assert s.loc[("B", "C"), "n_significant"] == 0
    assert np.isnan(s.loc[("B", "C"), "best_auc_significant"])
    assert s.loc[("B", "C"), "best_auc_raw"] == pytest.approx(0.66)


def test_pair_summary_is_computed_before_the_top_n_cut():
    ctx = AnalysisContext(df=_two_class_df(), cfg=Config(random_state=0), target_col="class")
    out = PairwiseSeparability(top_n=1).run(ctx)
    assert len(out["pairs"][("A", "B")]) == 1
    row = out["pair_summary"].iloc[0]
    assert row["n_features"] == 5  # every feature, not the one kept by top_n
    assert row["n_significant"] >= 4


def test_pair_matrix_puts_an_unseparable_pair_at_chance():
    fig = sp.pair_separability_matrix(summarize_pairs(_pair_tables()))
    ax = fig.axes[0]
    arr = np.ma.filled(ax.images[0].get_array().astype(float), np.nan)
    assert arr.shape == (3, 3)
    assert np.isnan(np.diag(arr)).all()
    assert arr[1, 2] == arr[2, 1] == 0.5  # B–C: nothing survives FDR
    assert arr[0, 1] == pytest.approx(0.95)
    assert ax.get_title(loc="left") == "2 of 3 class pairs separable; not B–C"


def test_pair_headline_variants():
    s = summarize_pairs(_pair_tables())
    assert sp.pair_separability_headline(s[s["n_significant"] > 0]) == "All 2 class pairs separable"
    assert sp.pair_separability_headline(s[s["n_significant"] == 0]).startswith("No class pair")


def test_volcano_keeps_the_nonsignificant_cloud_and_zero_p_values():
    table = pd.DataFrame(
        {
            "feature": ["hit", "zero_p", "cloud1", "cloud2"],
            "cliffs_delta": [0.9, -0.95, 0.05, 0.1],
            "mwu_p_bh_fdr": [1e-8, 0.0, 0.7, 0.9],
        }
    )
    fig = sp.volcano_plot(table, pair_label="A vs B")
    ax = fig.axes[0]
    grey, accent = ax.collections[:2]
    assert len(grey.get_offsets()) == 2
    assert len(accent.get_offsets()) == 2  # p = 0 is plotted, pinned to the smallest p
    assert "2 of 4 features significant" in ax.get_title(loc="left")


# ── classifiers ──────────────────────────────────────────────────────────────


def test_confusion_plot_row_normalises_and_names_the_dominant_mixup():
    cm = np.array([[8, 0, 4], [0, 12, 0], [6, 0, 6]])
    names = ["FN", "FP", "TN"]
    fig = sp.confusion_matrix_plot(cm, names)
    ax = fig.axes[0]
    assert np.allclose(np.asarray(ax.images[0].get_array(), dtype=float).sum(axis=1), 1.0)
    assert sp.dominant_confusion(cm, names) == (1.0, "FN", "TN")
    assert ax.get_title(loc="left") == "accuracy 72%; 100% of errors are FN ↔ TN"


def test_confusion_headline_without_errors_or_dominant_pair():
    assert sp.confusion_headline(np.eye(3) * 5, ["a", "b", "c"]) == "accuracy 100%"
    spread = np.array([[5, 1, 1], [1, 5, 1], [1, 1, 5]])
    assert "errors" not in sp.confusion_headline(spread, ["a", "b", "c"])


def test_cv_plot_uses_dots_for_few_folds_and_per_metric_chance():
    per_fold = pd.DataFrame({"accuracy": [0.7, 0.72, 0.75], "mcc": [0.6, 0.62, 0.65]})
    ax = sp.cv_metric_boxplot(per_fold, n_classes=4).axes[0]
    assert not ax.patches  # three folds: dots, not boxes
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["fold", "mean", "chance"]

    many = pd.DataFrame({"accuracy": np.linspace(0.6, 0.8, 10)})
    assert sp.cv_metric_boxplot(many).axes[0].patches  # ten folds: a real box


def test_metric_chance_levels():
    assert sp.metric_chance_level("balanced_accuracy", 4) == 0.25
    assert sp.metric_chance_level("accuracy") is None  # needs n_classes
    assert sp.metric_chance_level("mcc") == 0.0
    assert sp.metric_chance_level("roc_auc") == 0.5
    assert sp.metric_chance_level("pr_auc", 2) is None


def test_calibration_uses_equal_count_bins_by_default():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200)
    y = (rng.uniform(0, 1, 200) < p).astype(int)
    ax = sp.calibration_plot(y, p).axes[0]
    points = ax.collections[0]
    assert len(points.get_offsets()) == 10  # one bin per ~20 predictions
    sizes = points.get_sizes()
    assert sizes.max() / sizes.min() < 1.2  # equal-count bins → near-equal markers
    assert "ECE" in ax.get_title(loc="left")


def test_calibration_handles_constant_predictions():
    ax = sp.calibration_plot(np.array([0, 1, 1]), np.array([0.5, 0.5, 0.5])).axes[0]
    assert len(ax.collections[0].get_offsets()) == 1


def test_permutation_null_stays_visible_far_from_the_observed_value():
    null = np.random.default_rng(0).normal(0, 0.005, 400)
    ax = sp.permutation_null_plot(null, observed=0.33, p_value=0.0025).axes[0]
    (silhouette,) = ax.patches  # one filled shape, no per-bar edges to wash it out
    xs = silhouette.get_path().vertices[:, 0]
    assert xs.min() >= null.min() - 1e-12 and xs.max() <= null.max() + 1e-12
    assert "0 of 400 permutations reach the observed ARI" in ax.get_title(loc="left")


# ── distributions & clustering ───────────────────────────────────────────────


def test_class_quantile_panel_draws_one_panel_per_known_feature():
    pfc = pd.DataFrame(
        [
            {"feature": f, "class": c, "q05": 0, "q25": 1, "q50": 2, "q75": 3, "q95": 4}
            for f in ("x", "y")
            for c in ("A", "B", "C")
        ]
    )
    fig = sp.class_quantile_panel(pfc, ["x", "y", "missing"])
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title(loc="left") for ax in visible] == ["x", "y"]
    assert {t.get_text() for t in visible[0].get_yticklabels()} == {"A", "B", "C"}


def test_embedding_scatter_labels_classes_and_noise():
    coords = np.random.default_rng(0).normal(size=(8, 2))
    fig = sp.embedding_scatter(
        coords, np.array(["A", "B"] * 4), np.array([0, 1, -1, 0, 1, -1, 0, 1])
    )
    left, right = fig.axes[:2]
    assert {t.get_text() for t in left.get_legend().get_texts()} == {"A", "B"}
    assert {t.get_text() for t in right.get_legend().get_texts()} == {"0", "1", "noise"}
    markers = {c.get_paths()[0].vertices.tobytes() for c in left.collections}
    assert len(markers) == 2  # each class has its own marker shape


# ── analysis outputs the plots rely on ───────────────────────────────────────


def test_cluster_validation_stores_permutation_nulls():
    ctx = AnalysisContext(df=_two_class_df(), cfg=Config(random_state=0), target_col="class")
    out = ClusterValidation(n_permutations=30).run(ctx)
    assert out["ari_null"].shape == (30,)
    assert out["v_measure_null"].shape == (30,)


def test_cv_classifier_returns_labels_aligned_with_oof_predictions():
    ctx = AnalysisContext(df=_two_class_df(), cfg=Config(random_state=0), target_col="class")
    rf = {"n_estimators": 20, "n_jobs": 1, "random_state": 0}
    with pytest.warns(UserWarning, match="Ungrouped"):
        out = CrossValidatedClassifier(n_splits=3, rf_params=rf).run(ctx)
    assert out["y_true"].shape == out["oof_pred"].shape
    assert set(np.unique(out["y_true"])) == {0, 1}

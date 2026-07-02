import matplotlib

matplotlib.use("Agg")  # headless: tests must not require a display

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ml_analysis import Config
from ml_analysis.analysis import AnalysisContext, ClusterAnalysis
from ml_analysis.io.stat_plots import (
    cluster_class_heatmap,
    cluster_class_heatmap_panel,
)


def _separable_df(n: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    half = n // 2
    return pl.DataFrame(
        {
            "f_sep": list(rng.normal(0, 1, half)) + list(rng.normal(6.0, 1, half)),
            "f_noise": list(rng.normal(0, 1, n)),
            "class": ["A"] * half + ["B"] * half,
        }
    )


def test_cluster_analysis_exposes_aligned_y_true():
    ctx = AnalysisContext(df=_separable_df(), cfg=Config(), target_col="class")
    out = ClusterAnalysis().run(ctx)
    assert "y_true" in out
    y = out["y_true"]
    # y_true is row-aligned with every clustering label array
    for lab in out["labels"].values():
        assert len(lab) == len(y)
    # encoded labels match the reported class_names
    assert set(np.unique(y).tolist()) <= set(range(len(out["class_names"])))


def test_heatmap_row_normalizes_and_annotates_counts():
    # Cluster 0 -> 2×A, 1×B ; Cluster 1 -> 2×B
    labels = np.array([0, 0, 0, 1, 1])
    y_true = np.array([0, 0, 1, 1, 1])
    fig = cluster_class_heatmap(labels, y_true, ["A", "B"], normalize="cluster")

    ax = fig.axes[0]
    arr = np.asarray(ax.images[0].get_array(), dtype=float)
    assert np.allclose(arr, [[2 / 3, 1 / 3], [0.0, 1.0]])
    # cell text is the raw count, not the fraction
    assert sorted(t.get_text() for t in ax.texts) == ["0", "1", "2", "2"]
    plt.close(fig)


def test_heatmap_excludes_noise_points():
    # First two rows are DBSCAN/HDBSCAN noise and must be dropped.
    labels = np.array([-1, -1, 0, 0, 1])
    y_true = np.array([0, 1, 0, 0, 1])
    fig = cluster_class_heatmap(labels, y_true, ["A", "B"], normalize="cluster")
    ax = fig.axes[0]
    arr = np.asarray(ax.images[0].get_array(), dtype=float)
    # only clusters 0 and 1 survive: 0 -> all A, 1 -> all B
    assert arr.shape == (2, 2)
    assert np.allclose(arr, [[1.0, 0.0], [0.0, 1.0]])
    plt.close(fig)


def test_heatmap_class_normalization_sums_down_columns():
    labels = np.array([0, 0, 1, 1])
    y_true = np.array([0, 1, 0, 1])
    fig = cluster_class_heatmap(labels, y_true, ["A", "B"], normalize="class")
    arr = np.asarray(fig.axes[0].images[0].get_array(), dtype=float)
    # each class is split 50/50 across the two clusters
    assert np.allclose(arr.sum(axis=0), [1.0, 1.0])
    plt.close(fig)


def test_heatmap_handles_all_noise_gracefully():
    labels = np.array([-1, -1, -1])
    y_true = np.array([0, 1, 0])
    fig = cluster_class_heatmap(labels, y_true, ["A", "B"])
    # placeholder text, no image drawn
    assert len(fig.axes[0].images) == 0
    plt.close(fig)


def test_panel_draws_one_heatmap_per_algorithm():
    ctx = AnalysisContext(df=_separable_df(), cfg=Config(), target_col="class")
    out = ClusterAnalysis().run(ctx)
    fig = cluster_class_heatmap_panel(out["labels"], out["y_true"], out["class_names"])
    assert isinstance(fig, plt.Figure)
    # one heatmap Axes per algorithm (colorbars add extra axes, hence >=)
    assert len(fig.axes) >= len(out["labels"])
    plt.close(fig)

"""Plot helpers for the corroborating statistical analyses.

Every function returns the matplotlib Figure so the caller decides
whether to ``save_fig`` it, show it, or composite it into a larger
panel. Plots are intentionally minimal — no styling beyond labels,
titles, and reference lines — so they slot into Jupyter notebooks,
reports, and headless demo scripts alike.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def volcano_plot(
    pair_table: pd.DataFrame,
    pair_label: str = "",
    p_col: str = "mwu_p_bh_fdr",
    effect_col: str = "cliffs_delta",
    sig_threshold: float = 0.05,
    effect_threshold: float = 0.33,
    annotate_top: int = 6,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Volcano plot: |effect size| (x) vs -log10(corrected p) (y).

    Each point is a feature; top-right corner = large effect AND
    significant after multiple-testing correction. Quadrant lines mark
    the conventional thresholds. The top ``annotate_top`` features by
    |effect| × significance get a label.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    if p_col not in pair_table.columns or effect_col not in pair_table.columns:
        ax.text(0.5, 0.5, f"missing {p_col} or {effect_col}",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    eff = pair_table[effect_col].abs().values
    p = pair_table[p_col].values.astype(float)
    p_safe = np.where(np.isfinite(p) & (p > 0), p, np.nan)
    neglog = -np.log10(p_safe)

    ax.scatter(eff, neglog, s=22, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axhline(-np.log10(sig_threshold), color="grey", lw=0.8, ls="--",
               label=f"p={sig_threshold:g}")
    ax.axvline(effect_threshold, color="grey", lw=0.8, ls=":",
               label=f"|effect|={effect_threshold:g}")

    score = np.where(np.isfinite(neglog), eff * neglog, -np.inf)
    top_idx = np.argsort(-score)[:annotate_top]
    for i in top_idx:
        if not np.isfinite(neglog[i]):
            continue
        ax.annotate(
            pair_table["feature"].iloc[i],
            (eff[i], neglog[i]),
            xytext=(4, 2), textcoords="offset points", fontsize=8,
        )

    ax.set_xlabel(f"|{effect_col}|")
    ax.set_ylabel(f"-log10({p_col})")
    title = f"Volcano — {pair_label}" if pair_label else "Volcano"
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    return fig


def auc_bootstrap_plot(
    pair_table: pd.DataFrame,
    top_n: int = 12,
    pair_label: str = "",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Per-feature AUC with bootstrap CI error bars (sorted descending).

    Requires ``auc_ci_low`` / ``auc_ci_high`` columns produced by
    ``PairwiseSeparability(bootstrap_n=...)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * top_n)))
    else:
        fig = ax.figure
    needed = {"auc", "auc_ci_low", "auc_ci_high", "feature"}
    if not needed.issubset(pair_table.columns):
        ax.text(0.5, 0.5, "no bootstrap CI columns —\nset bootstrap_n>0",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    sub = pair_table.head(top_n).iloc[::-1]
    y = np.arange(len(sub))
    auc = sub["auc"].values
    lo = sub["auc_ci_low"].values
    hi = sub["auc_ci_high"].values
    err = np.vstack([auc - lo, hi - auc])

    ax.errorbar(auc, y, xerr=err, fmt="o", color="steelblue",
                ecolor="lightsteelblue", capsize=3)
    ax.axvline(0.5, color="grey", lw=0.8, ls="--", label="chance")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"].values, fontsize=8)
    ax.set_xlabel("AUC (95% bootstrap CI)")
    title = f"AUC ± CI — {pair_label}" if pair_label else "AUC ± CI"
    ax.set_title(title)
    ax.set_xlim(0.45, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    return fig


def importance_stability_plot(
    stability_table: pd.DataFrame,
    top_n: int = 12,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bootstrap MDI median + CI bars for the top-N features.

    A wide bar that touches zero is unstable. The companion column
    ``stability_top<k>`` (fraction of resamples where the feature is in
    the top-k) is drawn as a colour shade behind each bar.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * top_n)))
    else:
        fig = ax.figure
    if stability_table.empty:
        ax.text(0.5, 0.5, "empty stability table",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    sub = stability_table.head(top_n).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(sub))
    med = sub["mdi_median"].values
    lo = sub["mdi_ci_low"].values
    hi = sub["mdi_ci_high"].values
    err = np.vstack([med - lo, hi - med])

    stab_cols = [c for c in sub.columns if c.startswith("stability_top")]
    stab = sub[stab_cols[0]].values if stab_cols else np.ones(len(sub))
    bars = ax.barh(y, med, color=plt.cm.viridis(stab), alpha=0.85,
                   edgecolor="black", linewidth=0.5)
    ax.errorbar(med, y, xerr=err, fmt="none", ecolor="black", capsize=3, lw=1)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"].values, fontsize=8)
    ax.set_xlabel("RF MDI (median + 95% bootstrap CI)")
    ax.set_title("Importance stability — colour ∝ top-k stability")
    ax.axvline(0, color="grey", lw=0.6)

    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(stab_cols[0] if stab_cols else "stability", fontsize=8)
    return fig


def method_agreement_heatmap(
    matrix: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Spearman rank-correlation heatmap across importance methods."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure
    if matrix.empty:
        ax.text(0.5, 0.5, "empty — run FeatureImportance first",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    data = matrix.values
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Importance methods agreement")
    return fig


def cv_metric_boxplot(
    per_fold: pd.DataFrame,
    metrics: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Per-fold distribution of the selected metrics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    if metrics is None:
        metrics = [
            c for c in (
                "accuracy", "balanced_accuracy", "f1_macro",
                "mcc", "cohen_kappa", "roc_auc", "pr_auc",
            )
            if c in per_fold.columns
        ]
    data = [per_fold[m].dropna().values for m in metrics]
    bp = ax.boxplot(data, tick_labels=metrics, showmeans=True,
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.set_ylabel("score")
    ax.set_title(f"Cross-validated metrics  (k={len(per_fold)} folds)")
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(0.5, color="grey", lw=0.6, ls="--")
    return fig


def calibration_plot(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Reliability diagram for binary probabilistic predictions."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_proba_pos, bins[1:-1])
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        xs.append(float(y_proba_pos[m].mean()))
        ys.append(float(np.asarray(y_true)[m].mean()))
        ns.append(int(m.sum()))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    sizes = np.array(ns) / max(ns) * 200 if ns else 50
    ax.scatter(xs, ys, s=sizes, color="firebrick", alpha=0.8, edgecolor="white")
    ax.plot(xs, ys, color="firebrick", lw=1, alpha=0.6)
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration  (marker ∝ bin size)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    return fig


def permutation_null_plot(
    null_distribution: np.ndarray,
    observed: float,
    p_value: float,
    statistic_name: str = "ARI",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Histogram of the permutation null with the observed value marked."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    ax.hist(null_distribution, bins=40, color="lightgrey", edgecolor="white")
    ax.axvline(observed, color="firebrick", lw=2,
               label=f"observed = {observed:.3f}")
    ax.set_xlabel(statistic_name)
    ax.set_ylabel("count")
    ax.set_title(f"Permutation null  (p = {p_value:.4g})")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def cluster_class_heatmap(
    labels: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    normalize: str = "cluster",
    title: str = "",
    cmap: str = "Blues",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Contingency heatmap of cluster IDs (rows) vs true classes (columns).

    Colour encodes the normalised fraction; each cell's text is the raw
    count. ``normalize`` selects the denominator:

    - ``"cluster"`` (row): fraction of each cluster falling in each class —
      answers *"what is this cluster made of?"* (the default).
    - ``"class"`` (column): fraction of each class captured by each cluster —
      answers *"where did this class end up?"*.
    - ``"none"``: colour encodes the raw counts directly.

    Noise points (label ``-1`` from DBSCAN / HDBSCAN) are dropped, matching
    the alignment metrics in ``ClusterAnalysis``. ``labels`` and ``y_true``
    are the same-length arrays carried in the ``clustering`` result
    (``labels[algo]`` and ``y_true``).
    """
    labels = np.asarray(labels)
    y_true = np.asarray(y_true)
    keep = labels != -1
    lab = labels[keep]
    yt = y_true[keep]
    cluster_ids = sorted(set(lab.tolist()))

    if ax is None:
        w = max(3.5, 1.1 * len(class_names) + 2.0)
        h = max(2.5, 0.55 * max(len(cluster_ids), 1) + 1.5)
        fig, ax = plt.subplots(figsize=(w, h))
    else:
        fig = ax.figure

    if not cluster_ids:
        ax.text(0.5, 0.5, "no non-noise clusters",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title or "Cluster vs true class")
        return fig

    n_classes = len(class_names)
    counts = np.zeros((len(cluster_ids), n_classes), dtype=int)
    for r, cl in enumerate(cluster_ids):
        row = lab == cl
        for ci in range(n_classes):
            counts[r, ci] = int(np.sum(row & (yt == ci)))

    if normalize == "cluster":
        denom = counts.sum(axis=1, keepdims=True)
        color = np.divide(counts, denom, out=np.zeros(counts.shape), where=denom > 0)
        vmin, vmax, cbar_label = 0.0, 1.0, "Fraction of cluster"
    elif normalize == "class":
        denom = counts.sum(axis=0, keepdims=True)
        color = np.divide(counts, denom, out=np.zeros(counts.shape), where=denom > 0)
        vmin, vmax, cbar_label = 0.0, 1.0, "Fraction of class"
    else:
        color = counts.astype(float)
        vmin, vmax, cbar_label = 0.0, float(color.max() or 1.0), "Count"

    im = ax.imshow(color, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f"Cluster {c}" for c in cluster_ids], fontsize=8)
    ax.set_xlabel("True class")
    ax.set_ylabel("Cluster")

    shade = color / vmax if vmax else color
    for r in range(len(cluster_ids)):
        for c in range(n_classes):
            ax.text(c, r, str(counts[r, c]), ha="center", va="center",
                    fontsize=8, color="white" if shade[r, c] > 0.5 else "black")

    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title or "Cluster vs true class")
    return fig


def cluster_class_heatmap_panel(
    all_labels: dict[str, np.ndarray],
    y_true: np.ndarray,
    class_names: list[str],
    normalize: str = "cluster",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """One :func:`cluster_class_heatmap` per clustering algorithm.

    ``all_labels`` is the ``labels`` dict from ``ClusterAnalysis`` (algorithm
    name → label array); ``y_true`` and ``class_names`` are the companion
    entries in the same result. Shows at a glance whether each algorithm's
    clusters line up with the supervised classes.
    """
    n = max(len(all_labels), 1)
    if figsize is None:
        per = max(4.5, 1.1 * len(class_names) + 2.5)
        figsize = (per * n, 5.0)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)

    if not all_labels:
        axes[0, 0].text(0.5, 0.5, "no clusterings to plot",
                        ha="center", va="center", transform=axes[0, 0].transAxes)
        return fig

    for ax, (name, lab) in zip(axes[0], all_labels.items()):
        cluster_class_heatmap(
            lab, y_true, class_names, normalize=normalize, title=name, ax=ax,
        )

    fig.suptitle(
        "Cluster composition by true class  (colour = fraction, text = count)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def diagnostics_panel(
    pair_table: pd.DataFrame | None = None,
    pair_label: str = "",
    stability_table: pd.DataFrame | None = None,
    method_agreement: pd.DataFrame | None = None,
    cv_per_fold: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (16, 11),
) -> plt.Figure:
    """Composite 2x2 figure: volcano, AUC-CI, stability bars, CV box."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    if pair_table is not None:
        volcano_plot(pair_table, pair_label=pair_label, ax=axes[0, 0])
        auc_bootstrap_plot(pair_table, top_n=10, pair_label=pair_label,
                           ax=axes[0, 1])
    else:
        for a in (axes[0, 0], axes[0, 1]):
            a.text(0.5, 0.5, "pair_table not provided",
                   ha="center", va="center", transform=a.transAxes)
    if stability_table is not None:
        importance_stability_plot(stability_table, top_n=10, ax=axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, "no stability_table",
                        ha="center", va="center",
                        transform=axes[1, 0].transAxes)
    if cv_per_fold is not None:
        cv_metric_boxplot(cv_per_fold, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, "no cv_per_fold",
                        ha="center", va="center",
                        transform=axes[1, 1].transAxes)
    fig.tight_layout()
    return fig


__all__: list[str] = [
    "volcano_plot",
    "auc_bootstrap_plot",
    "importance_stability_plot",
    "method_agreement_heatmap",
    "cv_metric_boxplot",
    "calibration_plot",
    "permutation_null_plot",
    "cluster_class_heatmap",
    "cluster_class_heatmap_panel",
    "diagnostics_panel",
]

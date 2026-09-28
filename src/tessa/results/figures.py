"""UI-independent figure factory for analysis results.

The dashboard and the static HTML report were both *table-only*: a reader
opening a saved run saw rankings and scalars but no visual read on what
the numbers mean. This module closes that gap **without** coupling the
plots to any UI — every function returns plain matplotlib ``Figure``
objects, so the same code feeds the Streamlit dashboard (``st.pyplot``),
the self-contained HTML report (base64 PNG), and notebooks/scripts
(``Run.figures()``) alike.

The entry points consume an :class:`AnalysisResult`, so they work
identically on a *live* result (nested dicts and fitted models still in
``.objects``) and on one *reloaded from the store* (only the serialized
frames/arrays/scalars survive). Each analysis gets curated plots when its
shape is recognized; anything unrecognized falls back to histograms of
its 1-D numeric arrays. Builders never raise — a plot that can't be drawn
is simply skipped, so a partial run still renders everything it can.

Plots lead with the answer the analysis exists to give: which class pairs
separate (``pairwise``), where the classifier's errors go
(``classifier``, ``cv_classifier``), what the discriminating features look
like per class (``distributions``). Titles state the finding when the
numbers support one, and every figure shares :mod:`tessa.io.stat_plots`'s
visual system, so a class keeps its colour across the whole run.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

from tessa.io import stat_plots
from tessa.results.result import AnalysisResult

Figure = plt.Figure
TitledFigure = tuple[str, Figure]

_ACCENT = stat_plots.CATEGORICAL[0]


# ── small shared helpers ─────────────────────────────────────────────────────


def _pdf(frame: Any) -> pd.DataFrame:
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


def _is_text(values: pd.Index | pd.Series) -> bool:
    # pandas >= 3 gives string labels the ``str`` dtype, not ``object``.
    return pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)


def _resolve_labels(
    pdf: pd.DataFrame, candidates: tuple[str, ...]
) -> tuple[np.ndarray, pd.DataFrame]:
    """Row labels for a frame, robust to the store's index→column round-trip.

    A frame indexed by feature/metric names keeps that index when live, but
    ``reset_index`` turns it into an ``index`` column on the way through
    parquet. Try the (string) index first, then the named candidates.
    """
    idx = pdf.index
    if not isinstance(idx, pd.RangeIndex) and _is_text(idx):
        return np.array([str(i) for i in idx]), pdf
    for col in candidates:
        if col in pdf.columns:
            return pdf[col].astype(str).to_numpy(), pdf
    return np.array([str(i) for i in range(len(pdf))]), pdf


def _square_matrix(frame: Any) -> tuple[np.ndarray, list[str]] | None:
    """Extract an (N×N matrix, labels) pair from a square correlation-like frame."""
    pdf = _pdf(frame)
    if pdf.empty:
        return None
    first = pdf.columns[0]
    if first in ("index", "level_0", "") and _is_text(pdf[first]):
        pdf = pdf.set_index(first)
    num = pdf.select_dtypes(include="number")
    if num.shape[0] != num.shape[1] or num.shape[0] < 2:
        return None
    return num.to_numpy(dtype=float), [str(c) for c in num.columns]


def _heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    *,
    cmap=stat_plots.SEQUENTIAL,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    annotate_max: int = 16,
) -> Figure:
    n = len(labels)
    size = float(np.clip(0.45 * n + 1.5, 4.0, 11.0))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    finite = matrix[np.isfinite(matrix)]
    norm = mcolors.Normalize(
        vmin=vmin if vmin is not None else (finite.min() if finite.size else 0.0),
        vmax=vmax if vmax is not None else (finite.max() if finite.size else 1.0),
    )
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    if n <= 30:
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)
        stat_plots._cell_gaps(ax, n, n)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if n <= annotate_max:
        for i in range(n):
            for j in range(n):
                v = matrix[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=stat_plots._ink_on(cmap(norm(v))),
                )
    cbar = fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _barh_top(
    labels: np.ndarray,
    values: np.ndarray,
    title: str,
    xlabel: str,
    *,
    top_n: int = 15,
    color=_ACCENT,
    colors: np.ndarray | None = None,
) -> Figure | None:
    finite = np.isfinite(values)
    labels, values = labels[finite], values[finite]
    if colors is not None:
        colors = colors[finite]
    if len(values) == 0:
        return None
    order = np.argsort(values)[::-1][:top_n][::-1]  # ascending for barh top-down
    fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.34 * len(order) + 1)))
    ax.barh(
        np.array(labels)[order],
        values[order],
        height=0.7,
        color=(colors[order] if colors is not None else color),
        edgecolor="white",
        linewidth=1,
    )
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _top_per_signal(ranked_features: Sequence[str], k: int = 6) -> list[str]:
    """The best feature of each source signal, then the next best overall, up to ``k``.

    The raw top-k is often k variants of one signal; taking each signal's best
    first shows which signals matter at all. Returned in ranking order.
    """
    ranked = list(ranked_features)
    picks: list[str] = []
    seen: set[str] = set()
    for feat in ranked:
        signal = stat_plots.feature_signal(feat)
        if signal not in seen:
            seen.add(signal)
            picks.append(feat)
    picks = picks[:k]
    for feat in ranked:
        if len(picks) >= k:
            break
        if feat not in picks:
            picks.append(feat)
    return sorted(picks, key=ranked.index)


def _confusion_from_labels(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    keep = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    np.add.at(cm, (y_true[keep].astype(int), y_pred[keep].astype(int)), 1)
    return cm


# ── per-analysis builders ────────────────────────────────────────────────────


def _b_distributions(res: AnalysisResult) -> list[Figure]:
    if "summary" not in res.frames:
        return []
    pdf = _pdf(res.frames["summary"])
    labels, pdf = _resolve_labels(pdf, ("feature", "index"))
    if "kw_stat" not in pdf.columns:
        return []
    kw = pdf["kw_stat"].to_numpy(dtype=float)
    figs: list[Figure] = []

    colors = None
    title = "Class separation per feature — Kruskal–Wallis H"
    xlabel = "Kruskal–Wallis H"
    if "kw_p_bh_fdr" in pdf.columns:
        sig = pdf["kw_p_bh_fdr"].to_numpy(dtype=float) < 0.05
        colors = np.where(sig, _ACCENT, stat_plots.OTHER)
        title = (
            f"Kruskal–Wallis — {int(sig.sum())} of {len(sig)} features differ "
            "across classes (BH-FDR < 0.05)"
        )
        xlabel = "Kruskal–Wallis H  (grey = not significant)"
    fig = _barh_top(labels, kw, title, xlabel, colors=colors)
    if fig is not None:
        figs.append(fig)

    if "per_feature_class" in res.frames:
        ranked = labels[np.argsort(-np.nan_to_num(kw, nan=-np.inf))]
        figs.append(
            stat_plots.class_quantile_panel(
                _pdf(res.frames["per_feature_class"]),
                _top_per_signal(ranked),
                title="Top features by class — each signal's best included (median, IQR, 5–95%)",
            )
        )
    return figs


def _b_importance(res: AnalysisResult) -> list[Figure]:
    if "table" not in res.frames:
        return []
    pdf = _pdf(res.frames["table"])
    labels, pdf = _resolve_labels(pdf, ("feature", "index"))
    if "score_composite" not in pdf.columns:
        return []
    values = pdf["score_composite"].to_numpy(dtype=float)
    signals = np.array([stat_plots.feature_signal(f) for f in labels])
    shown = np.argsort(-np.nan_to_num(values, nan=-np.inf))[:15]
    palette = stat_plots.signal_colors(signals[shown])

    title = "Top features by composite importance"
    top10 = Counter(signals[shown[:10]])
    if len(palette) > 1 and top10:
        signal, count = top10.most_common(1)[0]
        if count * 2 >= sum(top10.values()):
            title = (
                f"Composite importance — {signal} holds {count} of the top {sum(top10.values())}"
            )
    colors = (
        np.array([palette.get(s, stat_plots.OTHER) for s in signals]) if len(palette) > 1 else None
    )
    fig = _barh_top(labels, values, title, "Composite importance (rank-blended)", colors=colors)
    if fig is None:
        return []
    if colors is not None:
        handles = [Patch(color=c, label=s) for s, c in palette.items()]
        fig.axes[0].legend(handles=handles, title="signal", loc="lower right")
    return [fig]


def _b_importance_stability(res: AnalysisResult) -> list[Figure]:
    figs: list[Figure] = []
    if "bootstrap_table" in res.frames:
        bt = _pdf(res.frames["bootstrap_table"])
        if not bt.empty:
            figs.append(stat_plots.importance_stability_plot(bt, top_n=12))
    if "method_agreement" in res.frames:
        mat = _square_matrix(res.frames["method_agreement"])
        if mat is not None:
            m, labels = mat
            figs.append(
                _heatmap(
                    m,
                    labels,
                    "Agreement between importance methods",
                    cmap=stat_plots.DIVERGING,
                    vmin=-1,
                    vmax=1,
                    cbar_label="Spearman ρ",
                )
            )
    return figs


def _cv_headline(per_fold: pd.DataFrame, n_classes: int | None) -> str | None:
    metric = next((m for m in ("balanced_accuracy", "accuracy") if m in per_fold.columns), None)
    if metric is None:
        return None
    vals = per_fold[metric].dropna()
    if vals.empty:
        return None
    spread = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    text = f"{metric.replace('_', ' ').capitalize()} {vals.mean():.2f} ± {spread:.2f} over {len(vals)} folds"
    chance = stat_plots.metric_chance_level(metric, n_classes)
    return text + (f" (chance {chance:.2f})" if chance is not None else "")


def _b_cv_classifier(res: AnalysisResult) -> list[Figure]:
    figs: list[Figure] = []
    class_names = [str(c) for c in (res.scalars.get("class_names") or [])]
    n_classes = len(class_names) or None
    if "per_fold" in res.frames:
        per_fold = _pdf(res.frames["per_fold"])
        if not per_fold.empty:
            fig = stat_plots.cv_metric_boxplot(per_fold, n_classes=n_classes)
            headline = _cv_headline(per_fold, n_classes)
            if headline:
                fig.axes[0].set_title(headline)
            figs.append(fig)

    y = res.arrays.get("y_true")
    pred = res.arrays.get("oof_pred")
    if y is not None and pred is not None and n_classes and len(y) == len(pred):
        cm = _confusion_from_labels(np.asarray(y), np.asarray(pred), n_classes)
        fig = stat_plots.confusion_matrix_plot(cm, class_names)
        ax = fig.axes[0]
        ax.set_title(
            "Out-of-fold confusion, all folds pooled\n"
            + stat_plots.confusion_headline(cm, class_names)
        )
        figs.append(fig)

    proba = res.arrays.get("oof_proba")
    if y is not None and proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
        if len(proba) == len(y):
            figs.append(stat_plots.calibration_plot(np.asarray(y), proba[:, 1]))
    return figs


def _b_separability(res: AnalysisResult) -> list[Figure]:
    if "perm_scores" not in res.arrays or "summary" not in res.frames:
        return []
    s = _pdf(res.frames["summary"]).iloc[0]
    observed = float(s["cv_balanced_accuracy"])
    perm = np.asarray(res.arrays["perm_scores"], dtype=float)
    fig = stat_plots.permutation_null_plot(
        perm,
        observed=observed,
        p_value=float(s["perm_p_value"]),
        statistic_name="balanced accuracy",
    )
    ax = fig.axes[0]
    title = _axes_title(ax)
    if "chance_level" in s:
        chance = float(s["chance_level"])
        ax.axvline(chance, color=stat_plots.MUTED, ls=":", lw=1.2, label=f"chance = {chance:.2f}")
        ax.legend(loc="upper right")
        title = f"balanced accuracy {observed:.2f} vs chance {chance:.2f} (p = {float(s['perm_p_value']):.3g})"
    verdict = str(s.get("verdict", ""))
    if verdict:
        ax.set_title(f"Separability: {verdict} — {title}")
    return [fig]


def _b_cluster_validation(res: AnalysisResult) -> list[Figure]:
    if "summary" not in res.frames:
        return []
    s = _pdf(res.frames["summary"]).iloc[0]
    figs: list[Figure] = []
    rows = [
        ("Hopkins\n(>0.6 clusterable)", s.get("hopkins"), None),
        ("ARI vs labels", s.get("ari"), s.get("ari_perm_p")),
        ("V-measure vs labels", s.get("v_measure"), s.get("v_measure_perm_p")),
    ]
    rows = [(lbl, float(v), p) for lbl, v, p in rows if v is not None and np.isfinite(v)]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        y = np.arange(len(rows))
        vals = [v for _, v, _ in rows]
        ax.barh(y, vals, height=0.6, color=_ACCENT, edgecolor="white", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([lbl for lbl, _, _ in rows])
        ax.tick_params(axis="y", length=0)
        ax.axvline(0.0, color=stat_plots.AXIS, lw=0.8)
        for yi, (_, v, p) in zip(y, rows):
            txt = f"{v:.3f}" + (f"  (perm p={p:.3g})" if p is not None and np.isfinite(p) else "")
            ax.text(v, yi, "  " + txt, va="center", fontsize=8, color=stat_plots.INK_SECONDARY)
        ax.set_xlim(min(0, min(vals)) - 0.05, max(1.0, max(vals)) + 0.25)
        ax.set_title("Cluster validation scorecard")
        fig.tight_layout()
        figs.append(fig)

    null = res.arrays.get("ari_null")
    ari = float(s.get("ari", np.nan))
    if null is not None and np.asarray(null).size and np.isfinite(ari):
        figs.append(
            stat_plots.permutation_null_plot(
                null,
                observed=ari,
                p_value=float(s.get("ari_perm_p", np.nan)),
                statistic_name="ARI",
            )
        )
    return figs


def _b_clustering(res: AnalysisResult) -> list[Figure]:
    figs: list[Figure] = []
    figs.extend(_clustering_elbow(res))
    figs.extend(_clustering_metrics(res))
    figs.extend(_clustering_embedding(res))
    figs.extend(_clustering_class_heatmap(res))
    return figs


def _clustering_elbow(res: AnalysisResult) -> list[Figure]:
    inertias = res.scalars.get("k_inertias")
    sils = res.scalars.get("k_silhouettes")
    if not inertias or not sils:
        return []
    inertias = list(inertias)
    sils = list(sils)
    ks = res.scalars.get("k_values") or list(range(2, 2 + len(inertias)))
    ks = list(ks)[: len(inertias)]
    # Two panels on a shared k axis, never two y-scales on one plot.
    fig, (ax_in, ax_sil) = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True)
    ax_in.plot(ks, inertias, "o-", color=_ACCENT, lw=1.5, markersize=5)
    ax_in.set_ylabel("KMeans inertia")
    ax_sil.plot(ks[: len(sils)], sils, "o-", color=_ACCENT, lw=1.5, markersize=5)
    ax_sil.set_ylabel("silhouette")
    ax_sil.set_xlabel("k (number of clusters)")
    best_k = res.scalars.get("best_k")
    for ax in (ax_in, ax_sil):
        ax.yaxis.grid(True)
        if best_k is not None:
            ax.axvline(int(best_k), color=stat_plots.MUTED, ls=":", lw=1.2)
    title = "Choosing k — elbow & silhouette"
    if best_k is not None:
        title += f" pick k = {int(best_k)}"
    ax_in.set_title(title)
    fig.tight_layout()
    return [fig]


def _clustering_metrics(res: AnalysisResult) -> list[Figure]:
    if "metrics" not in res.frames:
        return []
    pdf = _pdf(res.frames["metrics"])
    if pdf.empty:
        return []
    algos, pdf = _resolve_labels(pdf, ("Algorithm", "index"))
    wanted = [c for c in ("ARI", "NMI", "V-measure", "Silhouette") if c in pdf.columns]
    if not wanted:
        return []
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(algos))
    width = 0.8 / len(wanted)
    for i, col in enumerate(wanted):
        ax.bar(
            x + i * width,
            pdf[col].to_numpy(dtype=float),
            width,
            label=col,
            edgecolor="white",
            linewidth=1,
        )
    ax.set_xticks(x + width * (len(wanted) - 1) / 2)
    ax.set_xticklabels(algos, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.yaxis.grid(True)
    ax.axhline(0, color=stat_plots.AXIS, lw=0.8)
    ax.set_title("Clustering vs. true classes — agreement & quality")
    ax.legend(ncol=len(wanted))
    fig.tight_layout()
    return [fig]


def _clustering_embedding_data(
    res: AnalysisResult,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
    """(reduction_name, coords[N,2], true_class[N], {algo: labels[N]}) — live or stored."""
    # Live: reductions/labels survive in objects, y_true in arrays.
    reductions = res.objects.get("reductions")
    labels = res.objects.get("labels")
    class_names = res.scalars.get("class_names")
    if reductions and labels and "y_true" in res.arrays and class_names:
        name = "UMAP" if "UMAP" in reductions else next(iter(reductions))
        coords = np.asarray(reductions[name])
        true_cls = np.asarray(class_names)[res.arrays["y_true"]]
        return name, coords, true_cls, {k: np.asarray(v) for k, v in labels.items()}
    # Stored: the long-form ``embedding`` frame carries it all.
    if "embedding" in res.frames:
        pdf = _pdf(res.frames["embedding"])
        if pdf.empty or "reduction" not in pdf.columns:
            return None
        red = "UMAP" if (pdf["reduction"] == "UMAP").any() else pdf["reduction"].iloc[0]
        sub = pdf[pdf["reduction"] == red]
        coords = sub[["dim1", "dim2"]].to_numpy(dtype=float)
        true_cls = sub["true_class"].to_numpy()
        algo_cols = [c for c in sub.columns if c.startswith("cluster::")]
        labels = {c.split("::", 1)[1]: sub[c].to_numpy() for c in algo_cols}
        return str(red), coords, true_cls, labels
    return None


def _clustering_embedding(res: AnalysisResult) -> list[Figure]:
    data = _clustering_embedding_data(res)
    if data is None:
        return []
    name, coords, true_cls, labels = data
    primary = next((k for k in labels if k.startswith("KMeans")), next(iter(labels), None))
    if primary is None:
        return []
    return [
        stat_plots.embedding_scatter(
            coords, true_cls, labels[primary], reduction_name=name, cluster_name=primary
        )
    ]


def _clustering_class_heatmap(res: AnalysisResult) -> list[Figure]:
    class_names = res.scalars.get("class_names")
    if not class_names:
        return []
    labels = res.objects.get("labels")
    if labels and "y_true" in res.arrays:
        y_true = res.arrays["y_true"]
        labels = {k: np.asarray(v) for k, v in labels.items()}
    else:
        data = _clustering_embedding_data(res)
        if data is None:
            return []
        _, _, true_cls, labels = data
        index = {c: i for i, c in enumerate(class_names)}
        y_true = np.array([index.get(str(c), -1) for c in true_cls])
        if (y_true < 0).any():
            return []
    if not labels:
        return []
    return [stat_plots.cluster_class_heatmap_panel(labels, y_true, list(class_names))]


def _b_anomaly(res: AnalysisResult) -> list[Figure]:
    if "scores" not in res.frames:
        return []
    pdf = _pdf(res.frames["scores"])
    if "ensemble" not in pdf.columns or pdf.empty:
        return []
    figs: list[Figure] = []
    ens = pdf["ensemble"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(ens[np.isfinite(ens)], bins=30, color=_ACCENT, edgecolor="white")
    ax.set_xlabel("ensemble anomaly score (1 = most anomalous)")
    ax.set_ylabel("rows")
    ax.set_title("Distribution of anomaly scores")
    fig.tight_layout()
    figs.append(fig)

    id_col = next((c for c in ("event_id", "asset_id") if c in pdf.columns), None)
    if id_col is not None:
        top = pdf.nlargest(12, "ensemble")
        fig2 = _barh_top(
            top[id_col].astype(str).to_numpy(),
            top["ensemble"].to_numpy(dtype=float),
            "Most anomalous rows",
            "ensemble score",
            top_n=12,
        )
        if fig2 is not None:
            figs.append(fig2)
    return figs


def _b_correlation_structure(res: AnalysisResult) -> list[Figure]:
    if "correlation" not in res.frames:
        return []
    mat = _square_matrix(res.frames["correlation"])
    if mat is None:
        return []
    m, labels = mat
    return [
        _heatmap(
            m,
            labels,
            "Feature correlation structure (|Spearman|)",
            vmin=0,
            vmax=1,
            cbar_label="|ρ|",
        )
    ]


def _b_mi_network(res: AnalysisResult) -> list[Figure]:
    figs: list[Figure] = []
    if "matrix" in res.frames:
        mat = _square_matrix(res.frames["matrix"])
        if mat is not None:
            m, labels = mat
            figs.append(
                _heatmap(
                    m,
                    labels,
                    "Mutual-information network",
                    vmin=0,
                    cbar_label="MI (bits)",
                )
            )
    if "edges" in res.frames:
        e = _pdf(res.frames["edges"])
        if not e.empty and {"feature_a", "feature_b", "mutual_info"} <= set(e.columns):
            lbl = (e["feature_a"].astype(str) + " – " + e["feature_b"].astype(str)).to_numpy()
            fig = _barh_top(
                lbl,
                e["mutual_info"].to_numpy(dtype=float),
                "Strongest dependences",
                "mutual information (bits)",
                top_n=12,
            )
            if fig is not None:
                figs.append(fig)
    return figs


def _b_changepoint(res: AnalysisResult) -> list[Figure]:
    if "per_channel" not in res.frames:
        return []
    pdf = _pdf(res.frames["per_channel"])
    if pdf.empty or "n_changepoints" not in pdf.columns:
        return []
    fig = _barh_top(
        pdf["channel"].astype(str).to_numpy(),
        pdf["n_changepoints"].to_numpy(dtype=float),
        "Detected regime changes per channel",
        "changepoints",
        top_n=20,
    )
    return [fig] if fig is not None else []


def _b_lagged_relations(res: AnalysisResult) -> list[Figure]:
    if "table" not in res.frames:
        return []
    pdf = _pdf(res.frames["table"])
    if pdf.empty or "correlation_at_best_lag" not in pdf.columns:
        return []
    lbl = (pdf["leading"].astype(str) + " → " + pdf["following"].astype(str)).to_numpy()
    vals = pdf["correlation_at_best_lag"].to_numpy(dtype=float)
    order = np.argsort(np.abs(vals))[::-1][:12][::-1]
    fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.34 * len(order) + 1)))
    # Sign is polarity: the diverging pair's two poles.
    colors = np.where(vals[order] >= 0, stat_plots.CATEGORICAL[0], stat_plots.CATEGORICAL[7])
    ax.barh(lbl[order], vals[order], height=0.7, color=colors, edgecolor="white", linewidth=1)
    ax.axvline(0, color=stat_plots.AXIS, lw=0.8)
    ax.xaxis.grid(True)
    ax.set_xlabel("correlation at best lag  (blue = positive, red = negative)")
    ax.set_title("Strongest lagged associations (exploratory)")
    fig.tight_layout()
    return [fig]


def _b_label_spreading(res: AnalysisResult) -> list[Figure]:
    if "table" not in res.frames:
        return []
    pdf = _pdf(res.frames["table"])
    if "confidence" not in pdf.columns:
        return []
    conf = pdf["confidence"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(conf[np.isfinite(conf)], bins=25, color=_ACCENT, edgecolor="white")
    ax.set_xlabel("propagated-label confidence")
    ax.set_ylabel("rows")
    ax.set_title("Label-spreading confidence")
    fig.tight_layout()
    return [fig]


def _b_pu_learning(res: AnalysisResult) -> list[Figure]:
    frame = res.frames.get("ranked_unlabeled", res.frames.get("table"))
    if frame is None:
        return []
    pdf = _pdf(frame)
    if "pu_score" not in pdf.columns:
        return []
    score = pdf["pu_score"].to_numpy(dtype=float)
    score = score[np.isfinite(score)]
    if score.size == 0:
        return []
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(score, bins=25, color=_ACCENT, edgecolor="white")
    ax.set_xlabel("PU score (probability of being positive)")
    ax.set_ylabel("unlabeled rows")
    ax.set_title("PU learning — who looks positive?")
    fig.tight_layout()
    return [fig]


def _b_classifier(res: AnalysisResult) -> list[Figure]:
    class_names = list(res.scalars.get("class_names") or [])
    matrices: dict[str, np.ndarray] = {}
    # Stored path: the flattened confusion frame.
    if "confusion_long" in res.frames:
        cl = _pdf(res.frames["confusion_long"])
        if not cl.empty and {"model", "true_class", "pred_class", "count"} <= set(cl.columns):
            order = class_names or sorted(set(cl["true_class"]) | set(cl["pred_class"]))
            for model, g in cl.groupby("model"):
                piv = g.pivot_table(
                    index="true_class",
                    columns="pred_class",
                    values="count",
                    aggfunc="sum",
                    fill_value=0,
                ).reindex(index=order, columns=order, fill_value=0)
                matrices[str(model)] = piv.to_numpy(dtype=float)
                class_names = order
    # Live path: confusion matrices still attached to the fitted models.
    elif res.objects.get("models"):
        for model, info in res.objects["models"].items():
            cm = info.get("confusion_matrix") if isinstance(info, dict) else None
            if cm is not None:
                matrices[str(model)] = np.asarray(cm, dtype=float)
    if not matrices:
        return []
    names = list(matrices)[:3]
    k = next(iter(matrices.values())).shape[0]
    ticks = [str(c) for c in class_names] or [str(i) for i in range(k)]
    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=((0.9 * k + 1.8) * len(names), 0.9 * k + 2.1),
        squeeze=False,
    )
    for i, (ax, model) in enumerate(zip(axes[0], names)):
        cm = matrices[model]
        stat_plots.confusion_matrix_plot(
            cm, ticks, title=f"{model}\n{stat_plots.confusion_headline(cm, ticks)}", ax=ax
        )
        if i:
            ax.set_ylabel("")
    fig.suptitle("Confusion matrices, held-out test split (row % of each true class)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))  # keep the suptitle clear of the two-line titles
    return [fig]


def _pairwise_tables(res: AnalysisResult) -> dict[tuple[str, str], pd.DataFrame]:
    pairs = res.objects.get("pairs")
    if pairs:
        return {(str(a), str(b)): _pdf(v) for (a, b), v in pairs.items()}
    if "pairs_long" in res.frames:
        pl_df = _pdf(res.frames["pairs_long"])
        if not pl_df.empty and {"class_a", "class_b"} <= set(pl_df.columns):
            return {
                (str(a), str(b)): g.drop(columns=["class_a", "class_b"])
                for (a, b), g in pl_df.groupby(["class_a", "class_b"])
            }
    return {}


def _pair_summary(
    res: AnalysisResult, pairs: dict[tuple[str, str], pd.DataFrame]
) -> pd.DataFrame | None:
    """The analysis's ``pair_summary`` frame, or one rebuilt from the tables for older runs."""
    if "pair_summary" in res.frames:
        return _pdf(res.frames["pair_summary"])
    if not pairs:
        return None
    from tessa.analysis.pairwise import summarize_pairs

    return summarize_pairs(pairs)


def _focus_pair(
    summary: pd.DataFrame | None, pairs: dict[tuple[str, str], pd.DataFrame]
) -> tuple[str, str]:
    """The hardest pair that still separates: where the choice of feature matters most.

    Falls back to the pair with the highest raw AUC when no pair separates.
    """
    if summary is not None and not summary.empty:
        s = summary.assign(
            class_a=summary["class_a"].astype(str), class_b=summary["class_b"].astype(str)
        )
        s = s[[(a, b) in pairs for a, b in zip(s["class_a"], s["class_b"])]]
        separable = s[s["n_significant"] > 0]
        if not separable.empty:
            row = separable.sort_values(["best_auc_significant", "n_significant"]).iloc[0]
            return row["class_a"], row["class_b"]
        if not s.empty:
            row = s.sort_values("best_auc_raw", ascending=False).iloc[0]
            return row["class_a"], row["class_b"]

    def _top_auc(df: pd.DataFrame) -> float:
        return float(df["auc"].max()) if "auc" in df.columns and len(df) else -np.inf

    return max(pairs, key=lambda k: _top_auc(pairs[k]))


def _b_pairwise(res: AnalysisResult) -> list[Figure]:
    pairs = _pairwise_tables(res)
    summary = _pair_summary(res, pairs)
    figs: list[Figure] = []
    if summary is not None and not summary.empty:
        figs.append(stat_plots.pair_separability_matrix(summary))
    if not pairs:
        return figs

    a, b = _focus_pair(summary, pairs)
    tbl = pairs[(a, b)]
    label = f"{a} vs {b}"
    figs.append(stat_plots.volcano_plot(tbl, pair_label=label))
    if {"auc", "auc_ci_low", "auc_ci_high"} <= set(tbl.columns):
        figs.append(stat_plots.auc_bootstrap_plot(tbl, top_n=10, pair_label=label))
    return figs


def _generic_array_histograms(res: AnalysisResult) -> list[Figure]:
    figs: list[Figure] = []
    for name, arr in res.arrays.items():
        if arr.ndim != 1 or arr.size == 0 or arr.size > 200_000:
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(arr[np.isfinite(arr)], bins=40, color=_ACCENT, edgecolor="white")
        ax.set_title(name)
        fig.tight_layout()
        figs.append(fig)
    return figs


_DISPATCH: dict[str, Callable[[AnalysisResult], list[Figure]]] = {
    "distributions": _b_distributions,
    "importance": _b_importance,
    "importance_stability": _b_importance_stability,
    "cv_classifier": _b_cv_classifier,
    "separability": _b_separability,
    "cluster_validation": _b_cluster_validation,
    "clustering": _b_clustering,
    "anomaly": _b_anomaly,
    "correlation_structure": _b_correlation_structure,
    "mi_network": _b_mi_network,
    "changepoint": _b_changepoint,
    "lagged_relations": _b_lagged_relations,
    "label_spreading": _b_label_spreading,
    "pu_learning": _b_pu_learning,
    "classifier": _b_classifier,
    "pairwise": _b_pairwise,
}


# ── public API ───────────────────────────────────────────────────────────────


def figures_for_result(result: AnalysisResult | Any, *, max_figures: int = 8) -> list[TitledFigure]:
    """Return ``[(title, Figure), ...]`` for one analysis result.

    Accepts an :class:`AnalysisResult` or a raw result mapping. Curated
    plots are chosen by analysis name; results without a dedicated builder
    fall back to histograms of their 1-D numeric arrays. Never raises.
    """
    res = (
        result
        if isinstance(result, AnalysisResult)
        else AnalysisResult.from_raw(getattr(result, "name", "result"), result)
    )
    builder = _DISPATCH.get(res.name) or _DISPATCH.get(res.name.split("__", 1)[0])
    figs: list[Figure] = []
    with stat_plots.theme():
        if builder is not None:
            before = set(plt.get_fignums())
            try:
                figs = builder(res)
            except Exception:
                # Don't leak the figures a failing builder opened before it raised.
                for num in set(plt.get_fignums()) - before:
                    plt.close(num)
                figs = []
        if not figs:
            figs = _generic_array_histograms(res)
    for fig in figs[max_figures:]:
        plt.close(fig)
    return [(_figure_title(fig) or res.name, fig) for fig in figs[:max_figures]]


def _axes_title(ax: plt.Axes) -> str:
    # The shared theme left-aligns titles; ``get_title()`` alone reads the centre slot.
    return next((t for t in (ax.get_title(loc) for loc in ("left", "center", "right")) if t), "")


def _figure_title(fig: Figure) -> str:
    sup = fig.get_suptitle() if hasattr(fig, "get_suptitle") else ""
    if sup:
        return sup
    return _axes_title(fig.axes[0]) if fig.axes else ""


def figures_for_run(
    results: dict[str, AnalysisResult | Any], *, max_figures: int = 8
) -> dict[str, list[TitledFigure]]:
    """Map ``{analysis_name: [(title, Figure), ...]}`` for a whole run."""
    return {name: figures_for_result(res, max_figures=max_figures) for name, res in results.items()}


def headline_metrics(results: dict[str, AnalysisResult | Any]) -> list[dict[str, Any]]:
    """UI-independent KPIs across a run, for an at-a-glance overview.

    Returns a list of ``{"label", "value", "help"}`` dicts pulled from the
    analyses that carry a single headline number (separability verdict,
    separable class pairs, CV accuracy, clustering k, Hopkins, …). Safe on
    partial runs.
    """

    def _res(name: str) -> AnalysisResult | None:
        r = results.get(name)
        if r is None:
            return None
        return r if isinstance(r, AnalysisResult) else AnalysisResult.from_raw(name, r)

    out: list[dict[str, Any]] = []

    sep = _res("separability")
    if sep is not None and "summary" in sep.frames:
        s = _pdf(sep.frames["summary"]).iloc[0]
        out.append(
            {
                "label": "Separability",
                "value": str(s.get("verdict", "—")),
                "help": (
                    f"CV balanced acc {float(s['cv_balanced_accuracy']):.3f} "
                    f"vs chance {float(s['chance_level']):.2f} "
                    f"(perm p={float(s['perm_p_value']):.3g})"
                ),
            }
        )

    pw = _res("pairwise")
    if pw is not None:
        summary = _pair_summary(pw, _pairwise_tables(pw))
        if summary is not None and not summary.empty:
            blocked = summary[summary["n_significant"] == 0]
            names = [f"{a}–{b}" for a, b in zip(blocked["class_a"], blocked["class_b"])]
            out.append(
                {
                    "label": "Separable pairs",
                    "value": f"{len(summary) - len(blocked)} / {len(summary)}",
                    "help": "class pairs with at least one feature at BH-FDR < 0.05"
                    + (f"; not separable: {', '.join(names)}" if names else ""),
                }
            )

    cv = _res("cv_classifier")
    if cv is not None and "summary" in cv.frames:
        sm = _pdf(cv.frames["summary"])
        labels, sm = _resolve_labels(sm, ("index",))
        if "mean" in sm.columns and "accuracy" in labels:
            i = list(labels).index("accuracy")
            out.append(
                {
                    "label": "CV accuracy",
                    "value": f"{float(sm['mean'].iloc[i]):.3f}",
                    "help": f"mean over {len(_pdf(cv.frames['per_fold']))} folds"
                    if "per_fold" in cv.frames
                    else "cross-validated",
                }
            )

    clu = _res("clustering")
    if clu is not None and clu.scalars.get("best_k") is not None:
        out.append(
            {
                "label": "Best k",
                "value": str(int(clu.scalars["best_k"])),
                "help": "KMeans elbow/silhouette winner",
            }
        )

    cval = _res("cluster_validation")
    if cval is not None and "summary" in cval.frames:
        s = _pdf(cval.frames["summary"]).iloc[0]
        if np.isfinite(float(s.get("hopkins", np.nan))):
            out.append(
                {
                    "label": "Hopkins",
                    "value": f"{float(s['hopkins']):.2f}",
                    "help": "cluster tendency (>0.6 clusterable, ~0.5 none)",
                }
            )

    return out


__all__ = ["figures_for_result", "figures_for_run", "headline_metrics"]

"""Plot helpers for the corroborating statistical analyses.

Every function returns the matplotlib Figure so the caller decides
whether to ``save_fig`` it, show it, or composite it into a larger
panel (pass ``ax=``). Plots slot into Jupyter notebooks, reports, and
headless demo scripts alike.

All plots share one visual system, applied through :func:`theme`:

- a fixed categorical palette in a colour-blind-safe order, where each
  class keeps its colour in every figure of a run (:func:`class_colors`);
- a single-hue sequential ramp (``SEQUENTIAL``) for magnitudes and a
  red↔blue ramp with a grey midpoint (``DIVERGING``) for signed values;
- recessive grey chrome, so the data carries the contrast.

Titles state the finding where the data allows it ("5 of 6 class pairs
separable") instead of only naming the chart.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# ── visual system ────────────────────────────────────────────────────────────

# Categorical slots in a validated order: adjacent pairs stay distinct under
# colour-vision deficiency. Assigned in order and never cycled — a ninth
# category folds to ``OTHER``.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OTHER = "#b4b2a9"  # folded categories, non-highlighted marks

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "tessa_sequential",
    ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)
DIVERGING = LinearSegmentedColormap.from_list("tessa_diverging", ("#e34948", "#f0efec", "#2a78d6"))

_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

_RC = {
    "axes.prop_cycle": cycler(color=CATEGORICAL),
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.labelcolor": INK_SECONDARY,
    "axes.labelsize": 9,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.titlelocation": "left",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "figure.titleweight": "semibold",
    "font.size": 9,
}


@contextmanager
def theme() -> Iterator[None]:
    """Apply the shared visual system to every figure created inside the block."""
    with plt.rc_context(_RC):
        yield


def _themed(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with theme():
            return fn(*args, **kwargs)

    return wrapper


def _fold(names: Iterable, palette: Sequence[str]) -> dict[str, str]:
    ordered = sorted({str(n) for n in names})
    return {n: palette[i] if i < len(palette) else OTHER for i, n in enumerate(ordered)}


def class_colors(class_names: Iterable) -> dict[str, str]:
    """Stable colour per class: sorted names take the categorical slots in order.

    The same set of classes always maps to the same colours whatever order an
    analysis lists them in, so a class is one colour in every figure of a run.
    """
    return _fold(class_names, CATEGORICAL)


def signal_colors(signals: Iterable) -> dict[str, str]:
    """Stable colour per source signal, taken from the palette's far end.

    Signals and classes rarely share a figure, but starting from the other end
    keeps "temperature" from borrowing the colour a reader learned for a class.
    """
    return _fold(signals, CATEGORICAL[::-1])


def feature_signal(feature: str) -> str:
    """Source signal of a feature name: ``temperature__roll_mean_10__max`` → ``temperature``."""
    return str(feature).split("__", 1)[0]


def _ink_on(color) -> str:
    """Dark or white text, whichever reads on ``color``."""
    rgb = np.asarray(mcolors.to_rgb(color))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(lin @ [0.2126, 0.7152, 0.0722])
    return INK if luminance > 0.35 else "white"


def _fig_ax(ax: plt.Axes | None, figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def _legend(ax: plt.Axes, **kwargs) -> None:
    """A legend only when something is labelled (avoids matplotlib's empty-legend warning)."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(**kwargs)


def _placeholder(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, transform=ax.transAxes)


def _cell_gaps(ax: plt.Axes, n_rows: int, n_cols: int) -> None:
    """Separate heatmap cells with a thin surface-coloured gap instead of a border."""
    # Inner boundaries only: a gap drawn on the outer edge gets half-clipped
    # and leaves a hairline of cell colour outside the heatmap.
    ax.set_xticks(np.arange(0.5, n_cols - 1, 1), minor=True)
    ax.set_yticks(np.arange(0.5, n_rows - 1, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── pairwise separability ────────────────────────────────────────────────────


def pair_separability_headline(pair_summary: pd.DataFrame) -> str:
    """One line stating which class pairs separate, e.g. "5 of 6 class pairs separable; not FN–TN"."""
    if pair_summary.empty:
        return "No class pairs scored"
    n = len(pair_summary)
    blocked = pair_summary[pair_summary["n_significant"] == 0]
    if blocked.empty:
        return f"All {n} class pairs separable"
    if len(blocked) == n:
        return f"No class pair separable ({n} tested)"
    names = [f"{a}–{b}" for a, b in zip(blocked["class_a"], blocked["class_b"])]
    shown = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
    return f"{n - len(blocked)} of {n} class pairs separable; not {shown}"


@_themed
def pair_separability_matrix(
    pair_summary: pd.DataFrame,
    class_names: Sequence[str] | None = None,
    alpha: float = 0.05,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Which class pairs can be told apart, one cell per pair.

    ``pair_summary`` is the ``pair_summary`` frame from ``PairwiseSeparability``
    (``class_a, class_b, n_features, n_significant, best_auc_significant``). A
    cell's colour is the best single-feature AUC among features that survive
    multiple-testing correction; a pair where none survives sits at chance
    (0.5) and reads "none". Colouring by the raw best AUC instead would
    flatter noise: the best of ~90 unrelated features routinely scores 0.65.
    ``alpha`` only labels the colour bar; pass the threshold the summary was
    built with.
    """
    classes = [str(c) for c in class_names] if class_names else None
    if classes is None and not pair_summary.empty:
        classes = sorted(
            set(map(str, pair_summary["class_a"])) | set(map(str, pair_summary["class_b"]))
        )
    k = len(classes or [])
    fig, ax = _fig_ax(ax, (1.1 * k + 2.4, 1.0 * k + 1.4))
    if pair_summary.empty or k < 2:
        _placeholder(ax, "no class pairs scored")
        return fig

    pos = {c: i for i, c in enumerate(classes)}
    matrix = np.full((k, k), np.nan)
    labels: dict[tuple[int, int], str] = {}
    for row in pair_summary.itertuples(index=False):
        a, b = str(row.class_a), str(row.class_b)
        if a not in pos or b not in pos:
            continue
        n_sig, n_feat = int(row.n_significant), int(row.n_features)
        auc = float(row.best_auc_significant)
        value = auc if n_sig > 0 and np.isfinite(auc) else 0.5
        text = f"{auc:.2f}\n{n_sig}/{n_feat}" if n_sig > 0 else f"none\n0/{n_feat}"
        for i, j in ((pos[a], pos[b]), (pos[b], pos[a])):
            matrix[i, j] = value
            labels[(i, j)] = text

    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=SEQUENTIAL, norm=norm)
    for (i, j), text in labels.items():
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=8,
            color=_ink_on(SEQUENTIAL(norm(matrix[i, j]))),
        )
    ax.set_xticks(range(k))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(k))
    ax.set_yticklabels(classes)
    _cell_gaps(ax, k, k)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"best AUC among features with BH-FDR < {alpha:g}", fontsize=8)
    cbar.outline.set_visible(False)
    ax.set_xlabel("cell: best AUC · significant / tested features", color=MUTED, fontsize=7.5)
    ax.set_title(pair_separability_headline(pair_summary))
    return fig


@_themed
def volcano_plot(
    pair_table: pd.DataFrame,
    pair_label: str = "",
    p_col: str = "mwu_p_bh_fdr",
    effect_col: str = "cliffs_delta",
    sig_threshold: float = 0.05,
    effect_threshold: float = 0.33,
    annotate_top: int = 3,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Volcano plot: |effect size| (x) vs -log10(corrected p) (y).

    Each point is a feature. Features past both thresholds (large effect AND
    significant after multiple-testing correction) are drawn in the accent
    colour and counted in the title; the rest stay grey. The top
    ``annotate_top`` features by |effect| × significance are labelled in a
    column with leader lines, so labels never pile up when the winners crowd
    one corner. Pass the full per-pair table, not a ``top_n`` cut: without the
    cloud of unremarkable features there is nothing to compare against.
    """
    fig, ax = _fig_ax(ax, (7, 5))
    if p_col not in pair_table.columns or effect_col not in pair_table.columns:
        _placeholder(ax, f"missing {p_col} or {effect_col}")
        return fig

    eff = pair_table[effect_col].abs().to_numpy(dtype=float)
    p = pair_table[p_col].to_numpy(dtype=float)
    positive = np.isfinite(p) & (p > 0)
    # A p-value that underflowed to 0 is the *most* significant: pin it to the
    # smallest observed instead of dropping it as log(0).
    floor = float(p[positive].min()) if positive.any() else 1e-300
    neglog = -np.log10(np.where(np.isfinite(p), np.clip(p, floor, 1.0), np.nan))
    hit = (neglog > -np.log10(sig_threshold)) & (eff >= effect_threshold)

    ax.scatter(
        eff[~hit],
        neglog[~hit],
        s=20,
        color=OTHER,
        edgecolor="white",
        linewidth=0.5,
        label="other features",
    )
    ax.scatter(
        eff[hit],
        neglog[hit],
        s=24,
        color=CATEGORICAL[0],
        edgecolor="white",
        linewidth=0.5,
        label="significant, large effect",
    )
    ax.axhline(
        -np.log10(sig_threshold), color=MUTED, lw=0.8, ls="--", label=f"p = {sig_threshold:g}"
    )
    ax.axvline(
        effect_threshold, color=MUTED, lw=0.8, ls=":", label=f"|effect| = {effect_threshold:g}"
    )
    if effect_col in ("cliffs_delta", "rank_biserial"):
        ax.set_xlim(0, 1.02)  # bounded effect: show the whole scale

    score = np.where(np.isfinite(neglog), eff * neglog, -np.inf)
    top = [i for i in np.argsort(-score)[:annotate_top] if np.isfinite(score[i])]
    top.sort(key=lambda i: -neglog[i])  # label order follows height: leaders don't cross
    names = pair_table["feature"].astype(str).to_numpy() if "feature" in pair_table else None
    for rank, i in enumerate(top if names is not None else []):
        ax.annotate(
            names[i],
            xy=(eff[i], neglog[i]),
            xytext=(0.03, 0.96 - 0.07 * rank),
            textcoords="axes fraction",
            fontsize=7.5,
            color=INK_SECONDARY,
            va="top",
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6, "shrinkB": 3},
        )

    ax.set_xlabel(f"|{effect_col}|")
    ax.set_ylabel(f"-log10({p_col})")
    n = int(np.isfinite(neglog).sum())
    finding = f"{int(hit.sum())} of {n} features significant with |effect| ≥ {effect_threshold:g}"
    ax.set_title(f"Volcano — {pair_label}: {finding}" if pair_label else f"Volcano — {finding}")
    _legend(ax, loc="lower right")
    return fig


@_themed
def auc_bootstrap_plot(
    pair_table: pd.DataFrame,
    top_n: int = 12,
    pair_label: str = "",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Per-feature AUC with bootstrap CI error bars (sorted descending).

    Requires ``auc_ci_low`` / ``auc_ci_high`` columns produced by
    ``PairwiseSeparability(bootstrap_n=...)``. Features whose interval
    excludes chance (0.5) are drawn in the accent colour.
    """
    fig, ax = _fig_ax(ax, (7, max(3, 0.35 * top_n)))
    needed = {"auc", "auc_ci_low", "auc_ci_high", "feature"}
    if not needed.issubset(pair_table.columns):
        _placeholder(ax, "no bootstrap CI columns —\nset bootstrap_n>0")
        return fig

    sub = pair_table.head(top_n).iloc[::-1]
    y = np.arange(len(sub))
    auc = sub["auc"].to_numpy(dtype=float)
    lo = sub["auc_ci_low"].to_numpy(dtype=float)
    hi = sub["auc_ci_high"].to_numpy(dtype=float)
    clear = lo > 0.5
    for mask, color in ((clear, CATEGORICAL[0]), (~clear, OTHER)):
        if mask.any():
            ax.errorbar(
                auc[mask],
                y[mask],
                xerr=np.vstack([auc - lo, hi - auc])[:, mask],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=0,
                markersize=5,
            )
    ax.axvline(0.5, color=MUTED, lw=0.8, ls="--", label="chance")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"].values)
    ax.set_xlabel("AUC (95% bootstrap CI)")
    ax.xaxis.grid(True)
    ax.set_xlim(0.45, 1.02)
    title = f"Top features by AUC — {pair_label}" if pair_label else "Top features by AUC"
    ax.set_title(f"{title}: {int(clear.sum())} of {len(sub)} CIs exclude chance")
    _legend(ax, loc="lower left")
    return fig


# ── importance ───────────────────────────────────────────────────────────────


@_themed
def importance_stability_plot(
    stability_table: pd.DataFrame,
    top_n: int = 12,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bootstrap MDI median + CI bars for the top-N features.

    A wide bar that touches zero is unstable. The companion column
    ``stability_top<k>`` (fraction of resamples where the feature is in
    the top-k) shades each bar, darker = more stable.
    """
    fig, ax = _fig_ax(ax, (7, max(3, 0.35 * top_n)))
    if stability_table.empty:
        _placeholder(ax, "empty stability table")
        return fig

    sub = stability_table.head(top_n).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(sub))
    med = sub["mdi_median"].to_numpy(dtype=float)
    lo = sub["mdi_ci_low"].to_numpy(dtype=float)
    hi = sub["mdi_ci_high"].to_numpy(dtype=float)

    stab_cols = [c for c in sub.columns if c.startswith("stability_top")]
    stab = sub[stab_cols[0]].to_numpy(dtype=float) if stab_cols else np.ones(len(sub))
    ax.barh(y, med, height=0.7, color=SEQUENTIAL(stab), edgecolor="white", linewidth=1)
    ax.errorbar(med, y, xerr=np.vstack([med - lo, hi - med]), fmt="none", ecolor=INK, lw=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"].values)
    ax.set_xlabel("RF MDI (median + 95% bootstrap CI)")
    ax.xaxis.grid(True)
    ax.set_title("Importance stability — shade = how often in the top-k")

    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(stab_cols[0] if stab_cols else "stability", fontsize=8)
    cbar.outline.set_visible(False)
    return fig


@_themed
def method_agreement_heatmap(
    matrix: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Spearman rank-correlation heatmap across importance methods."""
    fig, ax = _fig_ax(ax, (5.5, 4.5))
    if matrix.empty:
        _placeholder(ax, "empty — run FeatureImportance first")
        return fig

    data = matrix.to_numpy(dtype=float)
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(data, cmap=DIVERGING, norm=norm)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_ink_on(DIVERGING(norm(v))),
                )
    _cell_gaps(ax, *data.shape)
    cbar = fig.colorbar(im, ax=ax, label="Spearman ρ")
    cbar.outline.set_visible(False)
    ax.set_title("Agreement between importance methods")
    return fig


# ── classifiers ──────────────────────────────────────────────────────────────

_CHANCE_FIXED = {"mcc": 0.0, "cohen_kappa": 0.0, "roc_auc": 0.5, "roc_auc_ovr": 0.5}
_CHANCE_UNIFORM = ("accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro")


def metric_chance_level(metric: str, n_classes: int | None = None) -> float | None:
    """What a metric scores for an uninformed classifier, or ``None`` if unknown.

    Accuracy-type scores assume uniform guessing (1/K), which needs
    ``n_classes``; MCC and Cohen's kappa sit at 0 and ROC-AUC at 0.5.
    """
    if metric in _CHANCE_FIXED:
        return _CHANCE_FIXED[metric]
    if metric in _CHANCE_UNIFORM and n_classes:
        return 1.0 / n_classes
    return None


@_themed
def cv_metric_boxplot(
    per_fold: pd.DataFrame,
    metrics: list[str] | None = None,
    n_classes: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Per-fold spread of the selected CV metrics, with each metric's chance level.

    With fewer than 8 folds every fold is drawn as a dot with a mean tick: a box
    summarising three numbers suggests a distribution that is not there. From 8
    folds on, box plots. Chance levels are per metric (:func:`metric_chance_level`);
    pass ``n_classes`` to get them for the accuracy-type scores too.
    """
    fig, ax = _fig_ax(ax, (7, 4.5))
    if metrics is None:
        metrics = [
            c
            for c in (
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "mcc",
                "cohen_kappa",
                "roc_auc",
                "pr_auc",
            )
            if c in per_fold.columns
        ]
    data = [per_fold[m].dropna().to_numpy(dtype=float) for m in metrics]
    x = np.arange(len(metrics))
    if len(per_fold) < 8:
        for xi, vals in zip(x, data):
            ax.scatter(
                np.full(len(vals), xi),
                vals,
                s=30,
                color=CATEGORICAL[0],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label="fold" if xi == 0 else None,
            )
            if len(vals):
                ax.hlines(
                    vals.mean(),
                    xi - 0.22,
                    xi + 0.22,
                    color=INK,
                    lw=1.5,
                    zorder=4,
                    label="mean" if xi == 0 else None,
                )
    else:
        ax.boxplot(
            data,
            positions=x,
            widths=0.5,
            showmeans=True,
            patch_artist=True,
            boxprops={"facecolor": "#cde2fb", "edgecolor": CATEGORICAL[0]},
            medianprops={"color": INK},
            whiskerprops={"color": AXIS},
            capprops={"color": AXIS},
            meanprops={
                "marker": "D",
                "markerfacecolor": INK,
                "markeredgecolor": "white",
                "markersize": 5,
            },
            flierprops={"markeredgecolor": MUTED, "markersize": 4},
        )
    first = True
    for xi, m in zip(x, metrics):
        chance = metric_chance_level(m, n_classes)
        if chance is not None:
            ax.hlines(
                chance,
                xi - 0.32,
                xi + 0.32,
                color=MUTED,
                lw=1.2,
                ls=(0, (2, 2)),
                label="chance" if first else None,
            )
            first = False
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.yaxis.grid(True)
    ax.set_title(f"Cross-validated metrics ({len(per_fold)} folds)")
    _legend(ax, loc="best")
    return fig


def dominant_confusion(cm: np.ndarray, class_names: Sequence[str]) -> tuple[float, str, str] | None:
    """Share of all errors between the most-confused pair of classes (both directions).

    Returns ``(share, class_a, class_b)``, or ``None`` when there are no errors.
    """
    cm = np.asarray(cm, dtype=float)
    off = cm - np.diag(np.diag(cm))
    total = off.sum()
    if total <= 0 or cm.shape[0] < 2:
        return None
    sym = off + off.T
    iu = np.triu_indices_from(sym, k=1)
    best = int(np.argmax(sym[iu]))
    i, j = iu[0][best], iu[1][best]
    return float(sym[i, j] / total), str(class_names[i]), str(class_names[j])


def confusion_headline(cm: np.ndarray, class_names: Sequence[str]) -> str:
    """Accuracy, plus the class pair holding most errors when one pair holds at least half."""
    cm = np.asarray(cm, dtype=float)
    total = cm.sum()
    if total <= 0:
        return "no predictions"
    text = f"accuracy {np.trace(cm) / total:.0%}"
    dom = dominant_confusion(cm, class_names)
    if dom is not None and dom[0] >= 0.5:
        share, a, b = dom
        text += f"; {share:.0%} of errors are {a} ↔ {b}"
    return text


@_themed
def confusion_matrix_plot(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Confusion matrix as row percentages (share of each true class), with counts.

    Row-normalising makes classes of different size comparable and turns a
    systematic mix-up into a visible block. The default title is
    :func:`confusion_headline`.
    """
    cm = np.asarray(cm, dtype=float)
    k = cm.shape[0]
    fig, ax = _fig_ax(ax, (0.9 * k + 1.8, 0.9 * k + 1.4))
    rows = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    ax.imshow(frac, cmap=SEQUENTIAL, vmin=0, vmax=1)
    if k <= 10:
        for i in range(k):
            for j in range(k):
                empty = cm[i, j] == 0  # a quiet "0" lets the populated cells stand out
                ax.text(
                    j,
                    i,
                    "0" if empty else f"{frac[i, j]:.0%}\n{int(cm[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=MUTED if empty else _ink_on(SEQUENTIAL(frac[i, j])),
                )
    names = [str(c) for c in class_names]
    ax.set_xticks(range(k))
    ax.set_xticklabels(names, rotation=45 if k > 4 else 0, ha="right" if k > 4 else "center")
    ax.set_yticks(range(k))
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true  (row % · count)")
    _cell_gaps(ax, k, k)
    ax.set_title(title if title is not None else confusion_headline(cm, names))
    return fig


@_themed
def calibration_plot(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    n_bins: int | None = None,
    strategy: str = "quantile",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Reliability diagram for binary probabilistic predictions.

    ``strategy="quantile"`` (default) uses equal-count bins, so every point
    rests on the same number of predictions; ``"uniform"`` uses equal-width
    bins. ``n_bins`` defaults to one bin per ~20 predictions, between 3 and 10:
    ten bins over a hundred predictions is mostly noise. The title reports the
    expected calibration error over the same bins.
    """
    fig, ax = _fig_ax(ax, (5, 5))
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_proba_pos, dtype=float)
    n = len(p)
    if n == 0:
        _placeholder(ax, "no predictions")
        return fig
    if n_bins is None:
        n_bins = int(np.clip(n // 20, 3, 10))
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    if len(edges) < 2:  # every prediction identical: one bin
        edges = np.array([p.min(), p.max()])
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)

    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            xs.append(float(p[m].mean()))
            ys.append(float(y[m].mean()))
            ns.append(int(m.sum()))
    ece = float(sum(nb / n * abs(yb - xb) for xb, yb, nb in zip(xs, ys, ns)))

    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="perfectly calibrated")
    ax.plot(xs, ys, color=CATEGORICAL[0], lw=1.5)
    sizes = np.array(ns) / max(ns) * 160 + 20
    ax.scatter(
        xs,
        ys,
        s=sizes,
        color=CATEGORICAL[0],
        edgecolor="white",
        linewidth=1,
        zorder=3,
        label="bin (size ∝ predictions)",
    )
    ax.set_xlabel("predicted probability (bin mean)")
    ax.set_ylabel("observed positive rate")
    kind = "equal-count" if strategy == "quantile" else "equal-width"
    ax.set_title(f"Calibration — ECE {ece:.3f}  ({len(xs)} {kind} bins, n={n})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True)
    _legend(ax, loc="upper left", markerscale=0.5)
    return fig


@_themed
def permutation_null_plot(
    null_distribution: np.ndarray,
    observed: float,
    p_value: float,
    statistic_name: str = "ARI",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Histogram of the permutation null with the observed value marked.

    The bins span the null alone and are drawn as one filled silhouette, so
    the null stays visible even when the observed value sits far outside it,
    which is the usual case for a real effect. The title says how many
    permutations reached the observed value.
    """
    fig, ax = _fig_ax(ax, (6.5, 4))
    null = np.asarray(null_distribution, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        _placeholder(ax, "empty null distribution")
        return fig
    lo, hi = float(null.min()), float(null.max())
    bins = np.linspace(lo, hi if hi > lo else lo + 1e-9, 31)
    ax.hist(
        null,
        bins=bins,
        histtype="stepfilled",
        color=OTHER,
        edgecolor=MUTED,
        linewidth=0.8,
        label=f"null ({null.size} permutations)",
    )
    ax.axvline(observed, color=CATEGORICAL[1], lw=2, label=f"observed = {observed:.3g}")
    reached = int((null >= observed).sum())
    ax.set_xlabel(statistic_name)
    ax.set_ylabel("permutations")
    ax.set_title(
        f"Permutation null — {reached} of {null.size} permutations reach the observed "
        f"{statistic_name} (p = {p_value:.3g})"
    )
    _legend(ax, loc="upper right")
    return fig


# ── clustering ───────────────────────────────────────────────────────────────


@_themed
def cluster_class_heatmap(
    labels: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    normalize: str = "cluster",
    title: str = "",
    cmap=None,
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
    cmap = SEQUENTIAL if cmap is None else plt.get_cmap(cmap)
    labels = np.asarray(labels)
    y_true = np.asarray(y_true)
    keep = labels != -1
    lab = labels[keep]
    yt = y_true[keep]
    cluster_ids = sorted(set(lab.tolist()))

    w = max(3.5, 1.1 * len(class_names) + 2.0)
    h = max(2.5, 0.55 * max(len(cluster_ids), 1) + 1.5)
    fig, ax = _fig_ax(ax, (w, h))

    if not cluster_ids:
        _placeholder(ax, "no non-noise clusters")
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
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f"Cluster {c}" for c in cluster_ids])
    ax.set_xlabel("True class")
    ax.set_ylabel("Cluster")

    for r in range(len(cluster_ids)):
        for c in range(n_classes):
            ax.text(
                c,
                r,
                str(counts[r, c]),
                ha="center",
                va="center",
                fontsize=8,
                color=_ink_on(cmap((color[r, c] - vmin) / (vmax - vmin or 1.0))),
            )
    _cell_gaps(ax, len(cluster_ids), n_classes)

    cbar = fig.colorbar(im, ax=ax, label=cbar_label)
    cbar.outline.set_visible(False)
    ax.set_title(title or "Cluster vs true class")
    return fig


@_themed
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
    clusters line up with the supervised classes. Algorithms that found fewer
    than two clusters have nothing to compare, so they are listed in the
    title instead of drawn as an empty panel.
    """
    usable: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for name, lab in all_labels.items():
        lab = np.asarray(lab)
        n_clusters = len(set(lab[lab != -1].tolist()))
        if n_clusters >= 2:
            usable[name] = lab
        else:
            skipped.append(f"{name} ({'all noise' if n_clusters == 0 else '1 cluster'})")

    n = max(len(usable), 1)
    if figsize is None:
        per = max(4.5, 1.1 * len(class_names) + 2.5)
        figsize = (per * n, 5.0)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)

    if not usable:
        _placeholder(axes[0, 0], "no clustering found two or more clusters")
    for ax, (name, lab) in zip(axes[0], usable.items()):
        cluster_class_heatmap(lab, y_true, class_names, normalize=normalize, title=name, ax=ax)

    title = "Cluster composition by true class  (colour = fraction, text = count)"
    if skipped:
        title += "\nnot shown: " + ", ".join(skipped)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


@_themed
def embedding_scatter(
    coords: np.ndarray,
    true_classes: np.ndarray,
    cluster_labels: np.ndarray | None = None,
    reduction_name: str = "PCA",
    cluster_name: str = "clusters",
) -> plt.Figure:
    """2-D embedding coloured by true class, and by cluster when labels are given.

    Classes take :func:`class_colors`; clusters take the palette from its far
    end, so a cluster never wears a class's colour. Every category also gets
    its own marker shape, so identity does not rest on colour alone when
    several categories overlap. Noise (``-1``) is grey.
    """
    coords = np.asarray(coords, dtype=float)
    panels = [(np.asarray(true_classes).astype(str), "true class", class_colors)]
    if cluster_labels is not None:
        panels.append((np.asarray(cluster_labels), cluster_name, _cluster_colors))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    for ax, (values, what, palette) in zip(axes[0], panels):
        cats = sorted(pd.unique(values), key=str)
        colors = palette(cats)
        for i, cat in enumerate(cats):
            m = values == cat
            noise = palette is _cluster_colors and _is_noise(cat)
            ax.scatter(
                coords[m, 0],
                coords[m, 1],
                s=18,
                alpha=0.85,
                color=OTHER if noise else colors[str(cat)],
                marker="o" if noise else _MARKERS[i % len(_MARKERS)],
                edgecolor="white",
                linewidth=0.4,
                label="noise" if noise else str(cat),
            )
        ax.set_xlabel(f"{reduction_name} 1")
        ax.set_ylabel(f"{reduction_name} 2")
        ax.set_title(f"{reduction_name} — coloured by {what}")
        _legend(ax, markerscale=1.2, loc="best")
    fig.suptitle("2-D embedding of the feature space")
    fig.tight_layout()
    return fig


def _is_noise(cluster_id) -> bool:
    return _is_number(cluster_id) and float(cluster_id) == -1


def _cluster_colors(cluster_ids: Iterable) -> dict[str, str]:
    ids = [c for c in cluster_ids if not _is_noise(c)]
    ordered = sorted(ids, key=lambda c: (float(c) if _is_number(c) else np.inf, str(c)))
    palette = CATEGORICAL[::-1]
    return {str(c): palette[i] if i < len(palette) else OTHER for i, c in enumerate(ordered)}


def _is_number(value) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


# ── distributions ────────────────────────────────────────────────────────────


@_themed
def class_quantile_panel(
    per_feature_class: pd.DataFrame,
    features: Sequence[str],
    ncols: int = 3,
    title: str | None = None,
) -> plt.Figure:
    """Small multiples of each feature's distribution per class, one row per class.

    Drawn from ``DistributionAnalysis``'s ``per_feature_class`` quantiles: a
    thin line spans the 5th–95th percentile, a thick bar the interquartile
    range, and a ringed dot marks the median. Each panel keeps its own x-axis,
    since features live on different scales, and class colours match every
    other figure of the run.
    """
    needed = {"feature", "class", "q05", "q25", "q50", "q75", "q95"}
    pfc = per_feature_class
    have = set(pfc["feature"].astype(str)) if "feature" in pfc.columns else set()
    features = [f for f in features if f in have]
    if not needed.issubset(pfc.columns) or not features:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        _placeholder(ax, "no per-class quantiles to draw")
        return fig

    classes = sorted(pfc["class"].astype(str).unique())
    colors = class_colors(classes)
    ypos = {c: len(classes) - 1 - i for i, c in enumerate(classes)}  # first class on top
    ncols = min(ncols, len(features))
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.7 * ncols, (0.38 * len(classes) + 1.0) * nrows),
        squeeze=False,
        sharey=True,
    )
    for ax, feat in zip(axes.flat, features):
        sub = pfc[pfc["feature"].astype(str) == feat]
        for r in sub.itertuples(index=False):
            cls = str(r[sub.columns.get_loc("class")])
            y, c = ypos[cls], colors[cls]
            ax.hlines(y, r.q05, r.q95, color=c, lw=1.2)
            ax.hlines(y, r.q25, r.q75, color=c, lw=7)
            ax.scatter([r.q50], [y], s=26, color="white", edgecolor=c, linewidth=1.5, zorder=3)
        ax.set_title(feat, fontsize=8.5)
        ax.xaxis.grid(True)
        ax.tick_params(axis="y", length=0)
    for ax in axes[:, 0]:
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels(list(ypos.keys()))
        ax.set_ylim(-0.6, len(classes) - 0.4)
    for ax in axes.flat[len(features) :]:
        ax.set_visible(False)
    fig.suptitle(title or "Feature distributions by class — median, IQR and 5–95% range")
    fig.tight_layout()
    return fig


# ── composite ────────────────────────────────────────────────────────────────


@_themed
def diagnostics_panel(
    pair_table: pd.DataFrame | None = None,
    pair_label: str = "",
    stability_table: pd.DataFrame | None = None,
    method_agreement: pd.DataFrame | None = None,
    cv_per_fold: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (16, 11),
) -> plt.Figure:
    """Composite 2x2 figure: volcano, AUC-CI, stability bars, CV metrics."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    if pair_table is not None:
        volcano_plot(pair_table, pair_label=pair_label, ax=axes[0, 0])
        auc_bootstrap_plot(pair_table, top_n=10, pair_label=pair_label, ax=axes[0, 1])
    else:
        for a in (axes[0, 0], axes[0, 1]):
            _placeholder(a, "pair_table not provided")
    if stability_table is not None:
        importance_stability_plot(stability_table, top_n=10, ax=axes[1, 0])
    else:
        _placeholder(axes[1, 0], "no stability_table")
    if cv_per_fold is not None:
        cv_metric_boxplot(cv_per_fold, ax=axes[1, 1])
    else:
        _placeholder(axes[1, 1], "no cv_per_fold")
    fig.tight_layout()
    return fig


__all__: list[str] = [
    "CATEGORICAL",
    "SEQUENTIAL",
    "DIVERGING",
    "theme",
    "class_colors",
    "signal_colors",
    "feature_signal",
    "pair_separability_headline",
    "pair_separability_matrix",
    "volcano_plot",
    "auc_bootstrap_plot",
    "importance_stability_plot",
    "method_agreement_heatmap",
    "metric_chance_level",
    "cv_metric_boxplot",
    "dominant_confusion",
    "confusion_headline",
    "confusion_matrix_plot",
    "calibration_plot",
    "permutation_null_plot",
    "cluster_class_heatmap",
    "cluster_class_heatmap_panel",
    "embedding_scatter",
    "class_quantile_panel",
    "diagnostics_panel",
]

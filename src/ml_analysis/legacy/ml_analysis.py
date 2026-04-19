"""
ml_analysis.py
==============
Modular pipeline for:
  - Feature importance analysis (RF, LightGBM, XGBoost + statistical tests)
  - Cluster analysis (KMeans, DBSCAN, HDBSCAN + PCA / UMAP projections)
  - Multi-classifier evaluation with confusion matrices

Usage
-----
    from ml_analysis import Config, run_all

    cfg = Config(df=your_df, target_col="your_class_col")
    run_all(cfg)

Or call individual stages:
    from ml_analysis import (
        prepare_data,
        run_feature_importance,
        run_cluster_analysis,
        run_classifier_evaluation,
    )
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from dataclasses import dataclass, field
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import kruskal

# Optional dependencies
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    try:
        import hdbscan as _hdbscan
        HDBSCAN = _hdbscan.HDBSCAN
    except ImportError:
        HDBSCAN = None

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


# ── CONFIG ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Required
    df: object                      # Polars or Pandas DataFrame
    target_col: str                 # name of the classification column

    # Model switches
    run_rf:  bool = True
    run_lgb: bool = True
    run_xgb: bool = True

    # Data scale
    train_sample_size: Optional[int] = 200_000   # None = use all rows
    test_size:         float         = 0.20
    shap_sample_size:  Optional[int] = None       # None = skip SHAP

    # Cluster settings
    k_range:              tuple = (2, 15)
    dbscan_neighbors:     int   = 5
    hdbscan_min_samples:  int   = 5
    umap_neighbors:       int   = 15

    # Importance sweep
    rf_permutation_repeats: int = 20

    # Model hyperparameters
    rf_params: dict = field(default_factory=lambda: dict(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42,
    ))
    lgb_params: dict = field(default_factory=lambda: dict(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        n_jobs=-1, random_state=42, verbose=-1,
    ))
    xgb_params: dict = field(default_factory=lambda: dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        tree_method="hist", device="cpu", random_state=42,
        verbosity=0, eval_metric="mlogloss",
    ))

    random_state: int = 42


# ── DATA PREPARATION ──────────────────────────────────────────────────────────

def prepare_data(cfg: Config) -> dict:
    """
    Convert to pandas, drop nulls, encode target, split train/test.

    Returns a dict with keys:
        X_full, y_full, X_train, X_test, y_train, y_test,
        feature_cols, le, class_names, n_classes
    """
    df = (
        cfg.df.to_pandas()
        if HAS_POLARS and isinstance(cfg.df, pl.DataFrame)
        else cfg.df.copy()
    )

    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != cfg.target_col
    ]

    X_full    = df[feature_cols].copy()
    y_raw_full = df[cfg.target_col].copy()

    mask      = X_full.notna().all(axis=1) & y_raw_full.notna()
    X_full    = X_full[mask].reset_index(drop=True)
    y_raw_full = y_raw_full[mask].reset_index(drop=True)

    le           = LabelEncoder()
    y_full       = le.fit_transform(y_raw_full)
    class_names  = [str(c) for c in le.classes_]
    n_classes    = len(class_names)

    print(f"Rows after cleaning : {len(y_full):,}")
    print(f"Features            : {len(feature_cols)}")
    print(f"Classes ({n_classes})          : {class_names}\n")

    # Optional stratified subsample for training
    if cfg.train_sample_size and len(X_full) > cfg.train_sample_size:
        print(f"Subsampling to {cfg.train_sample_size:,} rows (stratified) ...")
        idx = np.arange(len(X_full))
        _, idx_s = train_test_split(
            idx,
            test_size=cfg.train_sample_size / len(X_full),
            stratify=y_full,
            random_state=cfg.random_state,
        )
        X_use = X_full.iloc[idx_s].reset_index(drop=True)
        y_use = y_full[idx_s]
    else:
        X_use, y_use = X_full, y_full

    X_train, X_test, y_train, y_test = train_test_split(
        X_use, y_use,
        test_size=cfg.test_size,
        stratify=y_use,
        random_state=cfg.random_state,
    )
    print(f"Train : {len(y_train):,}  |  Test : {len(y_test):,}\n")

    return dict(
        X_full=X_full, y_full=y_full,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_cols=feature_cols,
        le=le, class_names=class_names, n_classes=n_classes,
    )


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0


def _scatter_labels(ax, coords, labels, title, noise_label=-1):
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    for i, lbl in enumerate(unique_labels):
        m     = labels == lbl
        color = "lightgray" if lbl == noise_label else cmap(i)
        name  = "noise"     if lbl == noise_label else str(lbl)
        ax.scatter(coords[m, 0], coords[m, 1], c=[color],
                   s=18, alpha=0.65, linewidths=0, label=name)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    n_uniq = len(unique_labels)
    if n_uniq <= 15:
        ax.legend(markerscale=1.5, fontsize=7, loc="best",
                  framealpha=0.6, ncol=2 if n_uniq > 8 else 1)


def _save_show(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved -> {path}")


# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────

def _compute_rf_importance(X_train, y_train, feature_cols, cfg) -> tuple:
    """Train a RF and return (mdi series, perm_mean series, perm_std series, model)."""
    print("  Fitting Random Forest for importance ...")
    rf = RandomForestClassifier(**cfg.rf_params)
    rf.fit(X_train, y_train)

    mdi = pd.Series(rf.feature_importances_, index=feature_cols, name="rf_mdi")

    print("  Computing permutation importance ...")
    perm = permutation_importance(
        rf, X_train, y_train,
        n_repeats=cfg.rf_permutation_repeats,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    perm_mean = pd.Series(perm.importances_mean, index=feature_cols, name="perm_mean")
    perm_std  = pd.Series(perm.importances_std,  index=feature_cols, name="perm_std")
    return mdi, perm_mean, perm_std, rf


def _compute_statistical_importance(X, y, feature_cols) -> pd.DataFrame:
    """ANOVA, Kruskal-Wallis, mutual information."""
    print("  Computing statistical tests ...")
    f_scores, f_pvals = f_classif(X, y)
    anova = pd.DataFrame(
        {"anova_f": f_scores, "anova_p": f_pvals},
        index=feature_cols,
    )

    kw_rows = []
    for col in feature_cols:
        groups = [X.loc[y == c, col].values for c in np.unique(y)]
        stat, pval = kruskal(*groups)
        kw_rows.append({"feature": col, "kw_stat": stat, "kw_p": pval})
    kw = pd.DataFrame(kw_rows).set_index("feature")

    mi = pd.Series(
        mutual_info_classif(X, y, random_state=0),
        index=feature_cols,
        name="mutual_info",
    )
    return pd.concat([anova, kw, mi], axis=1)


def _build_importance_table(mdi, perm_mean, perm_std, stat_df) -> pd.DataFrame:
    results = pd.concat([mdi, perm_mean, perm_std, stat_df], axis=1)
    results["score_composite"] = (
        _minmax(results["rf_mdi"])
        + _minmax(results["perm_mean"])
        + _minmax(results["anova_f"])
        + _minmax(results["kw_stat"])
        + _minmax(results["mutual_info"])
    ) / 5
    results = results.sort_values("score_composite", ascending=False)
    results.insert(0, "rank", range(1, len(results) + 1))
    return results


def _plot_importance(results: pd.DataFrame, target_col: str) -> plt.Figure:
    feature_cols = results.index.tolist()
    n     = len(feature_cols)
    y_pos = np.arange(n)
    pal   = plt.cm.RdYlGn(np.linspace(0.85, 0.2, n))

    fig = plt.figure(figsize=(16, max(6, n * 0.45 + 2)))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_left  = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    bars = ax_left.barh(y_pos, results["score_composite"],
                        color=pal, edgecolor="white", linewidth=0.5,
                        label="Composite score")
    ax_left.errorbar(results["perm_mean"], y_pos,
                     xerr=results["perm_std"],
                     fmt="D", color="#333", markersize=4, linewidth=1.2,
                     label="Permutation importance (+/- std)")
    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(feature_cols, fontsize=9)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("Normalized score")
    ax_left.set_title("Feature ranking\n(composite + permutation)",
                      fontsize=11, fontweight="bold")
    ax_left.legend(fontsize=8, loc="lower right")
    ax_left.axvline(0, color="black", linewidth=0.6)
    for i, (_, val) in enumerate(zip(bars, results["score_composite"])):
        ax_left.text(val + 0.005, i, f"#{i+1}", va="center", fontsize=7.5)

    scores = {
        "ANOVA F":       _minmax(results["anova_f"]),
        "Kruskal-Wallis":_minmax(results["kw_stat"]),
        "Mutual Info":   _minmax(results["mutual_info"]),
    }
    colors_stat = ["#4C78A8", "#F58518", "#72B7B2"]
    width = 0.25
    for mi, (label, vals) in enumerate(scores.items()):
        ax_right.barh(y_pos + (mi - 1) * width, vals.values,
                      height=width, label=label,
                      color=colors_stat[mi], edgecolor="white", linewidth=0.4)

    for i, feat in enumerate(feature_cols):
        tags = []
        if results.loc[feat, "anova_p"] > 0.05: tags.append("ANOVA ns")
        if results.loc[feat, "kw_p"]    > 0.05: tags.append("KW ns")
        if tags:
            ax_right.text(1.02, i, " | ".join(tags), va="center",
                          fontsize=7, color="crimson",
                          transform=ax_right.get_yaxis_transform())

    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(feature_cols, fontsize=9)
    ax_right.invert_yaxis()
    ax_right.set_xlabel("Normalized score")
    ax_right.set_title("Statistical tests\n(normalized per method)",
                       fontsize=11, fontweight="bold")
    ax_right.legend(fontsize=8, loc="lower right")
    ax_right.axvline(0, color="black", linewidth=0.6)

    fig.suptitle(f"Feature importance for '{target_col}'",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def run_feature_importance(cfg: Config, data: dict) -> pd.DataFrame:
    """
    Full feature importance pipeline.
    Returns the importance results DataFrame.
    """
    print("=== Feature Importance ===")
    X_train      = data["X_train"]
    y_train      = data["y_train"]
    feature_cols = data["feature_cols"]

    mdi, perm_mean, perm_std, _ = _compute_rf_importance(
        X_train, y_train, feature_cols, cfg
    )
    stat_df  = _compute_statistical_importance(X_train, y_train, feature_cols)
    results  = _build_importance_table(mdi, perm_mean, perm_std, stat_df)

    print("\nTop features:")
    print(results[["rank", "score_composite", "rf_mdi", "perm_mean",
                   "anova_f", "anova_p", "kw_stat", "kw_p", "mutual_info"]]
          .head(15).to_string())

    fig = _plot_importance(results, cfg.target_col)
    _save_show(fig, "feature_importance.png")
    plt.close(fig)

    return results


# ── CLUSTER ANALYSIS ──────────────────────────────────────────────────────────

def _scale(X: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def _find_best_k(X_scaled: np.ndarray, k_range, random_state: int) -> tuple:
    """Elbow + silhouette sweep. Returns (best_k, inertias, sil_scores)."""
    print("  Sweeping k for KMeans ...")
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    diffs2  = np.diff(np.diff(np.array(inertias)))
    elbow_k = list(k_range)[np.argmax(diffs2) + 1]
    sil_k   = list(k_range)[np.argmax(sil_scores)]
    best_k  = (
        elbow_k
        if sil_scores[list(k_range).index(elbow_k)] >= sil_scores[list(k_range).index(sil_k)]
        else sil_k
    )
    print(f"  Elbow k={elbow_k}  |  Best silhouette k={sil_k}  |  Chosen k={best_k}")
    return best_k, inertias, sil_scores


def _plot_k_selection(k_range, inertias, sil_scores, best_k) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, "o-", color="#4C78A8")
    axes[0].axvline(best_k, color="crimson", linestyle="--", label=f"chosen k={best_k}")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow curve"); axes[0].legend()

    axes[1].plot(list(k_range), sil_scores, "o-", color="#F58518")
    axes[1].axvline(best_k, color="crimson", linestyle="--", label=f"chosen k={best_k}")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette sweep"); axes[1].legend()

    plt.suptitle("KMeans k selection", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def _fit_clustering_algorithms(X_scaled: np.ndarray, best_k: int, cfg: Config) -> dict:
    """Fit KMeans, DBSCAN, HDBSCAN. Returns {algo_name: labels array}."""
    labels = {}

    # KMeans
    km = KMeans(n_clusters=best_k, random_state=cfg.random_state, n_init="auto")
    labels[f"KMeans (k={best_k})"] = km.fit_predict(X_scaled)

    # DBSCAN with auto eps
    nbrs = NearestNeighbors(n_neighbors=cfg.dbscan_neighbors).fit(X_scaled)
    dists, _ = nbrs.kneighbors(X_scaled)
    knn_dists = np.sort(dists[:, -1])
    diffs2    = np.diff(np.diff(knn_dists))
    eps_auto  = knn_dists[np.argmax(diffs2) + 1]
    print(f"  DBSCAN auto eps={eps_auto:.4f}")
    db = DBSCAN(eps=eps_auto, min_samples=cfg.dbscan_neighbors, n_jobs=-1)
    db_labels = db.fit_predict(X_scaled)
    n_db  = len(set(db_labels) - {-1})
    print(f"  DBSCAN -> {n_db} clusters, {(db_labels==-1).sum()} noise points")
    labels[f"DBSCAN (eps={eps_auto:.3f})"] = db_labels

    # HDBSCAN
    if HDBSCAN is not None:
        hdb = HDBSCAN(min_cluster_size=cfg.hdbscan_min_samples)
        hdb_labels = hdb.fit_predict(X_scaled)
        n_hdb = len(set(hdb_labels) - {-1})
        print(f"  HDBSCAN -> {n_hdb} clusters, {(hdb_labels==-1).sum()} noise points")
        labels["HDBSCAN"] = hdb_labels
    else:
        print("  HDBSCAN not available, skipping.")

    return labels


def _compute_reductions(X_scaled: np.ndarray, cfg: Config) -> dict:
    """PCA (always) + UMAP (if available). Returns {name: 2D array}."""
    reductions = {}

    print("  Running PCA ...")
    pca = PCA(n_components=2, random_state=cfg.random_state)
    reductions["PCA"] = pca.fit_transform(X_scaled), pca.explained_variance_ratio_

    if HAS_UMAP:
        print("  Running UMAP ...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=cfg.umap_neighbors,
            random_state=cfg.random_state,
        )
        reductions["UMAP"] = reducer.fit_transform(X_scaled), None

    return reductions


def _plot_cluster_scatter(reductions: dict, all_labels: dict, y: np.ndarray) -> plt.Figure:
    n_reductions = len(reductions)
    n_algos      = len(all_labels)
    n_cols       = n_algos + 1
    fig, axes = plt.subplots(
        n_reductions, n_cols,
        figsize=(5 * n_cols, 4.5 * n_reductions),
        squeeze=False,
    )
    for row_i, (red_name, (coords, var)) in enumerate(reductions.items()):
        _scatter_labels(axes[row_i, n_cols - 1], coords, y,
                        f"{red_name} | True classes")
        for col_i, (algo_name, lab) in enumerate(all_labels.items()):
            _scatter_labels(axes[row_i, col_i], coords, lab,
                            f"{red_name} | {algo_name}")
        if var is not None:
            axes[row_i, 0].set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=8)
            axes[row_i, 0].set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=8)

    plt.suptitle("Cluster structure vs true class labels",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def _compute_alignment_metrics(all_labels: dict, y: np.ndarray,
                                X_scaled: np.ndarray) -> pd.DataFrame:
    def safe_sil(labels):
        m = labels != -1
        return silhouette_score(X_scaled[m], labels[m]) if len(set(labels[m])) >= 2 else np.nan

    def safe_db(labels):
        m = labels != -1
        return davies_bouldin_score(X_scaled[m], labels[m]) if len(set(labels[m])) >= 2 else np.nan

    rows = []
    for name, lab in all_labels.items():
        m  = lab != -1
        h, c, v = homogeneity_completeness_v_measure(y[m], lab[m])
        rows.append({
            "Algorithm"    : name,
            "Clusters"     : len(set(lab) - {-1}),
            "Noise pts"    : (lab == -1).sum(),
            "Silhouette"   : safe_sil(lab),
            "Davies-Bouldin": safe_db(lab),
            "ARI"          : adjusted_rand_score(y[m], lab[m]),
            "NMI"          : normalized_mutual_info_score(y[m], lab[m]),
            "Homogeneity"  : h,
            "Completeness" : c,
            "V-measure"    : v,
        })
    return pd.DataFrame(rows).set_index("Algorithm")


def _plot_cluster_heatmap(all_labels: dict, y: np.ndarray,
                          class_names: list) -> plt.Figure:
    n_algos = len(all_labels)
    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 5), squeeze=False)
    for col_i, (algo_name, lab) in enumerate(all_labels.items()):
        m          = lab != -1
        l_valid    = lab[m]
        y_valid    = y[m]
        cluster_ids = sorted(set(l_valid))
        hm = pd.DataFrame(
            index=[f"Cluster {c}" for c in cluster_ids],
            columns=class_names, data=0,
        )
        for cl in cluster_ids:
            for ci, cn in enumerate(class_names):
                hm.loc[f"Cluster {cl}", cn] = ((l_valid == cl) & (y_valid == ci)).sum()
        hm_norm = hm.div(hm.sum(axis=1), axis=0)
        sns.heatmap(hm_norm.astype(float), ax=axes[0, col_i],
                    annot=hm.values, fmt="d", cmap="Blues",
                    linewidths=0.4, vmin=0, vmax=1,
                    cbar_kws={"label": "Fraction of cluster"})
        axes[0, col_i].set_title(algo_name, fontsize=10, fontweight="bold")
        axes[0, col_i].set_xlabel("True class")
        axes[0, col_i].set_ylabel("Cluster")

    plt.suptitle("Cluster vs true class (color = row fraction, numbers = counts)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def run_cluster_analysis(cfg: Config, data: dict) -> dict:
    """
    Full cluster analysis pipeline.
    Returns dict with keys: labels, metrics, reductions.
    """
    print("\n=== Cluster Analysis ===")
    X_scaled  = _scale(data["X_full"])
    y         = data["y_full"]
    class_names = data["class_names"]
    k_range   = range(*cfg.k_range) if isinstance(cfg.k_range, tuple) else cfg.k_range

    best_k, inertias, sil_scores = _find_best_k(X_scaled, k_range, cfg.random_state)

    fig_k = _plot_k_selection(k_range, inertias, sil_scores, best_k)
    _save_show(fig_k, "cluster_k_selection.png")
    plt.close(fig_k)

    all_labels = _fit_clustering_algorithms(X_scaled, best_k, cfg)
    reductions = _compute_reductions(X_scaled, cfg)

    fig_scatter = _plot_cluster_scatter(reductions, all_labels, y)
    _save_show(fig_scatter, "cluster_scatter.png")
    plt.close(fig_scatter)

    metrics = _compute_alignment_metrics(all_labels, y, X_scaled)
    print("\nAlignment metrics:")
    print(metrics.round(4).to_string())

    fig_hm = _plot_cluster_heatmap(all_labels, y, class_names)
    _save_show(fig_hm, "cluster_heatmap.png")
    plt.close(fig_hm)

    return dict(labels=all_labels, metrics=metrics, reductions=reductions)


# ── CLASSIFIER EVALUATION ─────────────────────────────────────────────────────

def _build_model_registry(cfg: Config) -> dict:
    models = {}
    if cfg.run_rf:
        models["Random Forest"] = RandomForestClassifier(**cfg.rf_params)
    if cfg.run_lgb and HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(**cfg.lgb_params)
    elif cfg.run_lgb:
        print("LightGBM not installed, skipping.")
    if cfg.run_xgb and HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(**cfg.xgb_params)
    elif cfg.run_xgb:
        print("XGBoost not installed, skipping.")
    if not models:
        raise RuntimeError("No models available. Check your installs.")
    return models


def _train_and_evaluate(models: dict, X_train, y_train,
                        X_test, y_test, class_names: list) -> dict:
    results = {}
    for name, model in models.items():
        print(f"  Training {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds,
                                    target_names=class_names, zero_division=0))
        imp = (
            pd.Series(model.feature_importances_,
                      index=X_train.columns, name=name)
            if hasattr(model, "feature_importances_")
            else None
        )
        results[name] = dict(model=model, preds=preds, accuracy=acc, importances=imp)
    return results


def _plot_confusion_matrices(results: dict, y_test,
                             class_names: list) -> plt.Figure:
    n = len(results)
    fig, axes = plt.subplots(
        1, n, figsize=(6 * n, max(5, len(class_names) * 0.55 + 2)),
        squeeze=False,
    )
    for col_i, (name, res) in enumerate(results.items()):
        cm     = confusion_matrix(y_test, res["preds"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ax     = axes[0, col_i]
        sns.heatmap(cm_norm, ax=ax, annot=cm, fmt="d",
                    cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.3, vmin=0, vmax=1,
                    cbar_kws={"label": "Row fraction"})
        ax.set_title(f"{name}\nAccuracy: {res['accuracy']:.4f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(
        "Confusion matrices (color = row-normalized, text = counts)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    return fig


def _plot_combined_importance(results: dict, feature_cols: list) -> plt.Figure:
    imp_series = [r["importances"] for r in results.values()
                  if r["importances"] is not None]
    if not imp_series:
        print("No feature importances available.")
        return None

    imp_df = pd.concat(imp_series, axis=1)
    imp_df["composite"] = imp_df.apply(_minmax, axis=0).mean(axis=1)
    imp_df = imp_df.sort_values("composite", ascending=False)

    n     = len(feature_cols)
    y_pos = np.arange(n)
    pal   = plt.cm.RdYlGn(np.linspace(0.85, 0.2, n))
    model_colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.4 + 2)))
    ax.barh(y_pos, imp_df["composite"], color=pal, edgecolor="white")

    for mi, mname in enumerate(results.keys()):
        if mname in imp_df.columns:
            ax.scatter(
                _minmax(imp_df[mname]), y_pos,
                marker="D", s=25, zorder=3,
                color=model_colors[mi % len(model_colors)],
                label=mname, alpha=0.85,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_df.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized importance")
    ax.set_title("Combined feature importance across models",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def run_classifier_evaluation(cfg: Config, data: dict) -> dict:
    """
    Train all configured classifiers, produce confusion matrices
    and combined feature importance plot.
    Returns dict with keys: results (per-model metrics and model objects).
    """
    print("\n=== Classifier Evaluation ===")
    models  = _build_model_registry(cfg)
    results = _train_and_evaluate(
        models,
        data["X_train"], data["y_train"],
        data["X_test"],  data["y_test"],
        data["class_names"],
    )

    fig_cm = _plot_confusion_matrices(results, data["y_test"], data["class_names"])
    _save_show(fig_cm, "confusion_matrices.png")
    plt.close(fig_cm)

    fig_imp = _plot_combined_importance(results, data["feature_cols"])
    if fig_imp is not None:
        _save_show(fig_imp, "feature_importance_combined.png")
        plt.close(fig_imp)

    return dict(results=results)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def run_all(cfg: Config) -> dict:
    """
    Run the full pipeline:
      1. prepare_data
      2. run_feature_importance
      3. run_cluster_analysis
      4. run_classifier_evaluation

    Returns a dict with all outputs from each stage.
    """
    data       = prepare_data(cfg)
    importance = run_feature_importance(cfg, data)
    clusters   = run_cluster_analysis(cfg, data)
    classifiers = run_classifier_evaluation(cfg, data)

    print("\nAll done.")
    return dict(
        data=data,
        importance=importance,
        clusters=clusters,
        classifiers=classifiers,
    )


# ── EXAMPLE USAGE ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Replace with your actual DataFrame and target column
    # import polars as pl
    # df = pl.read_csv("your_data.csv")

    # cfg = Config(
    #     df=df,
    #     target_col="your_class_col",
    #     train_sample_size=200_000,   # None to use all rows
    #     run_rf=True,
    #     run_lgb=True,
    #     run_xgb=True,
    # )
    # output = run_all(cfg)

    # Or run individual stages:
    # data = prepare_data(cfg)
    # run_feature_importance(cfg, data)
    # run_cluster_analysis(cfg, data)
    # run_classifier_evaluation(cfg, data)
    pass

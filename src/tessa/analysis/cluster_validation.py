"""Clusterability and cluster-label association significance.

``ClusterAnalysis`` already reports silhouette / Davies-Bouldin (cluster
quality) and ARI / NMI / V-measure (alignment with class labels). Those
are scores without significance, so we complement them with:

- *Hopkins statistic* — tests cluster tendency. Values near 0.5 indicate
  the data is uniform (not clusterable); values near 1.0 indicate strong
  clustering structure exists. Run *before* trusting any clustering
  result.
- *Calinski-Harabasz index* — within/between-cluster dispersion ratio,
  a third internal validity score that complements silhouette and
  Davies-Bouldin.
- *Permutation test on ARI/V-measure* — by reshuffling class labels we
  derive a null distribution and a p-value for the observed alignment
  between cluster IDs and class labels.

If the upstream ``clustering`` analysis has been run, this analysis
reads its KMeans labels via ``ctx.results['clustering']``; otherwise
it fits a fresh KMeans with ``k = n_classes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    homogeneity_completeness_v_measure,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from tessa.analysis.base import AnalysisContext, prepare_xy


def hopkins_statistic(
    X: np.ndarray, n_samples: int | None = None, rng: np.random.Generator | None = None
) -> float:
    """Hopkins statistic for cluster tendency.

    Returns a value in [0, 1]. ~0.5 = uniform (no structure),
    ~1.0 = highly clustered, ~0.0 = regularly spaced.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n, d = X.shape
    if n_samples is None:
        n_samples = min(int(0.1 * n), 100)
    n_samples = max(n_samples, 2)
    if n_samples >= n:
        n_samples = n // 2
    if n_samples < 2:
        return float("nan")

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    # uniformly sampled points in the bounding box
    u_pts = rng.uniform(mins, maxs, size=(n_samples, d))
    sample_idx = rng.choice(n, size=n_samples, replace=False)
    w_pts = X[sample_idx]

    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    # nearest-neighbour distance for sampled real points (skip self)
    d_w, _ = nbrs.kneighbors(w_pts)
    w_dists = d_w[:, 1]
    # nearest-neighbour distance for uniform points
    d_u, _ = nbrs.kneighbors(u_pts)
    u_dists = d_u[:, 0]

    # Power-1 distances: the classical `** d` variant overflows/underflows
    # for high-dimensional data (~100 features collapses the statistic to
    # nan or 0/1), exactly the regime this toolkit targets.
    num = np.sum(u_dists)
    den = num + np.sum(w_dists)
    if den == 0:
        return float("nan")
    return float(num / den)


def _ari_perm_test(
    y_true: np.ndarray, labels: np.ndarray, n_perm: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Permutation p-value for ARI between class labels and cluster IDs."""
    mask = labels != -1
    y_true = y_true[mask]
    labels = labels[mask]
    if len(y_true) < 2 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    observed = adjusted_rand_score(y_true, labels)
    null = np.empty(n_perm, dtype=float)
    perm = y_true.copy()
    for i in range(n_perm):
        rng.shuffle(perm)
        null[i] = adjusted_rand_score(perm, labels)
    # two-sided p: fraction of null at least as extreme as observed
    p = float((np.abs(null) >= abs(observed)).sum() + 1) / (n_perm + 1)
    return float(observed), p


def _vmeasure_perm_test(
    y_true: np.ndarray, labels: np.ndarray, n_perm: int, rng: np.random.Generator
) -> tuple[float, float]:
    mask = labels != -1
    y_true = y_true[mask]
    labels = labels[mask]
    if len(y_true) < 2 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    observed = homogeneity_completeness_v_measure(y_true, labels)[2]
    null = np.empty(n_perm, dtype=float)
    perm = y_true.copy()
    for i in range(n_perm):
        rng.shuffle(perm)
        null[i] = homogeneity_completeness_v_measure(perm, labels)[2]
    p = float((null >= observed).sum() + 1) / (n_perm + 1)
    return float(observed), p


@dataclass
class ClusterValidation:
    name: str = "cluster_validation"
    requires: tuple[str, ...] = ()
    n_permutations: int = 500
    hopkins_n_samples: int | None = None

    def _kmeans_labels(self, X: np.ndarray, ctx: AnalysisContext, n_classes: int) -> np.ndarray:
        cached = ctx.results.get("clustering") if hasattr(ctx, "results") else None
        if cached and "labels" in cached:
            for name, lab in cached["labels"].items():
                if name.startswith("KMeans"):
                    return np.asarray(lab)
        k = max(n_classes, 2)
        return KMeans(
            n_clusters=k, random_state=ctx.cfg.random_state, n_init="auto"
        ).fit_predict(X)

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx)
        X = StandardScaler().fit_transform(prep.X)
        y = prep.y
        rng = np.random.default_rng(ctx.cfg.random_state)
        n_classes = len(prep.class_names)

        hopkins = hopkins_statistic(X, self.hopkins_n_samples, rng=rng)

        labels = self._kmeans_labels(X, ctx, n_classes)

        ch = (
            float(calinski_harabasz_score(X, labels))
            if len(np.unique(labels)) > 1
            else float("nan")
        )

        ari_obs, ari_p = _ari_perm_test(y, labels, self.n_permutations, rng)
        v_obs, v_p = _vmeasure_perm_test(y, labels, self.n_permutations, rng)

        summary = pd.DataFrame([{
            "hopkins": hopkins,
            "calinski_harabasz": ch,
            "ari": ari_obs,
            "ari_perm_p": ari_p,
            "v_measure": v_obs,
            "v_measure_perm_p": v_p,
            "n_permutations": self.n_permutations,
        }])

        return {"summary": summary, "labels_used": labels}

"""Clustering analysis: KMeans, DBSCAN, HDBSCAN + PCA / UMAP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from tessa.analysis.base import AnalysisContext, prepare_xy

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    try:
        import hdbscan as _hdbscan

        HDBSCAN = _hdbscan.HDBSCAN
    except ImportError:
        HDBSCAN = None

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


@dataclass
class ClusterAnalysis:
    name: str = "clustering"
    requires: tuple[str, ...] = ()
    k_range: tuple[int, int] = (2, 10)
    dbscan_neighbors: int = 5
    hdbscan_min_samples: int = 5
    umap_neighbors: int = 15

    def _best_k(self, X: np.ndarray, random_state: int) -> tuple[int, list, list]:
        ks = list(range(*self.k_range))
        inertias, sils = [], []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            lbl = km.fit_predict(X)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X, lbl))
        diffs2 = np.diff(np.diff(inertias))
        elbow_k = ks[np.argmax(diffs2) + 1]
        sil_k = ks[int(np.argmax(sils))]
        best_k = elbow_k if sils[ks.index(elbow_k)] >= sils[ks.index(sil_k)] else sil_k
        return best_k, inertias, sils

    def _fit_all(self, X: np.ndarray, best_k: int, random_state: int) -> dict:
        labels = {}
        km = KMeans(n_clusters=best_k, random_state=random_state, n_init="auto")
        labels[f"KMeans (k={best_k})"] = km.fit_predict(X)

        nn = NearestNeighbors(n_neighbors=self.dbscan_neighbors).fit(X)
        d, _ = nn.kneighbors(X)
        knn = np.sort(d[:, -1])
        eps_auto = float(knn[np.argmax(np.diff(np.diff(knn))) + 1])
        labels[f"DBSCAN (eps={eps_auto:.3f})"] = DBSCAN(
            eps=eps_auto, min_samples=self.dbscan_neighbors, n_jobs=-1
        ).fit_predict(X)

        if HDBSCAN is not None:
            labels["HDBSCAN"] = HDBSCAN(
                min_cluster_size=self.hdbscan_min_samples
            ).fit_predict(X)
        return labels

    def _reduce(self, X: np.ndarray, random_state: int) -> dict:
        red = {"PCA": PCA(n_components=2, random_state=random_state).fit_transform(X)}
        if HAS_UMAP:
            red["UMAP"] = umap.UMAP(
                n_components=2,
                n_neighbors=self.umap_neighbors,
                random_state=random_state,
            ).fit_transform(X)
        return red

    def _alignment(self, all_labels: dict, y: np.ndarray, X: np.ndarray) -> pd.DataFrame:
        rows = []
        for name, lab in all_labels.items():
            m = lab != -1
            if m.sum() < 2 or len(set(lab[m])) < 2:
                continue
            h, c, v = homogeneity_completeness_v_measure(y[m], lab[m])
            rows.append({
                "Algorithm": name,
                "Clusters": len(set(lab) - {-1}),
                "Noise pts": int((lab == -1).sum()),
                "Silhouette": silhouette_score(X[m], lab[m]),
                "Davies-Bouldin": davies_bouldin_score(X[m], lab[m]),
                "ARI": adjusted_rand_score(y[m], lab[m]),
                "NMI": normalized_mutual_info_score(y[m], lab[m]),
                "Homogeneity": h,
                "Completeness": c,
                "V-measure": v,
            })
        return pd.DataFrame(rows).set_index("Algorithm") if rows else pd.DataFrame()

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx)
        X = StandardScaler().fit_transform(prep.X)
        y = prep.y

        best_k, inertias, sils = self._best_k(X, ctx.cfg.random_state)
        all_labels = self._fit_all(X, best_k, ctx.cfg.random_state)
        reductions = self._reduce(X, ctx.cfg.random_state)
        metrics = self._alignment(all_labels, y, X)

        return {
            "best_k": best_k,
            "k_inertias": inertias,
            "k_silhouettes": sils,
            "labels": all_labels,
            "reductions": reductions,
            "metrics": metrics,
            "class_names": prep.class_names,
            # encoded true labels aligned row-for-row with every array in
            # `labels` / `reductions` (post null-drop, same order as prepare_xy).
            "y_true": y,
        }

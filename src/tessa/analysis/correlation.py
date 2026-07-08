"""Feature redundancy structure: correlation clusters and near-duplicates.

With ~100 sensors many channels move together; redundancy inflates
importance ties and wastes model capacity. This analysis:

- computes the |Spearman| correlation matrix of the feature matrix,
- hierarchically clusters features with distance = 1 - |rho|
  (average linkage), cutting at ``1 - cluster_threshold``,
- lists near-duplicate pairs (|rho| >= ``duplicate_threshold``) and a
  suggested keep/drop set (keep the first feature of each cluster).

Unsupervised (``needs_labels="none"``) and cheap, so it can run first and
feed feature pruning for everything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from tessa.analysis.base import AnalysisContext, prepare_xy


@dataclass
class CorrelationStructure:
    name: str = "correlation_structure"
    requires: tuple[str, ...] = ()
    needs_labels: str = "none"
    cluster_threshold: float = 0.8
    duplicate_threshold: float = 0.95

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx, ignore_target=True)
        if len(prep.feature_cols) < 2:
            raise ValueError("Need >= 2 features for correlation structure.")
        # to_numpy(copy=...) — pandas copy-on-write makes .values read-only
        vals = prep.X.corr(method="spearman").abs().fillna(0.0).to_numpy(copy=True)
        np.fill_diagonal(vals, 1.0)
        corr = pd.DataFrame(vals, index=prep.feature_cols, columns=prep.feature_cols)

        dist = 1.0 - vals
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="average")
        cluster_ids = fcluster(link, t=1.0 - self.cluster_threshold,
                               criterion="distance")

        clusters = pd.DataFrame({
            "feature": prep.feature_cols,
            "cluster": cluster_ids,
        }).sort_values(["cluster", "feature"]).reset_index(drop=True)
        keep = clusters.groupby("cluster")["feature"].first()
        clusters["suggested_keep"] = clusters["feature"].isin(set(keep))

        iu = np.triu_indices(len(corr), k=1)
        rho = vals[iu]
        dup_mask = rho >= self.duplicate_threshold
        feat = np.asarray(prep.feature_cols)
        duplicates = pd.DataFrame({
            "feature_a": feat[iu[0][dup_mask]],
            "feature_b": feat[iu[1][dup_mask]],
            "abs_spearman": rho[dup_mask],
        }).sort_values("abs_spearman", ascending=False).reset_index(drop=True)

        return {
            "correlation": corr,
            "clusters": clusters,
            "duplicates": duplicates,
            "n_clusters": int(clusters["cluster"].nunique()),
            "n_features": len(prep.feature_cols),
        }

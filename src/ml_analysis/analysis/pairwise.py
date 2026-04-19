"""Pairwise class separability: for each (A, B), how distinguishable are they?"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from ml_analysis.analysis.base import AnalysisContext, prepare_xy


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size in [-1, 1]. 0 = no separation, +/-1 = perfect."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    x = np.asarray(a)
    y = np.asarray(b)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def _single_feature_auc(values: np.ndarray, y_binary: np.ndarray) -> float:
    """AUC for the rule 'larger value -> class 1'. Use max(auc, 1-auc) for direction-free."""
    try:
        auc = roc_auc_score(y_binary, values)
        return float(max(auc, 1 - auc))
    except ValueError:
        return float("nan")


@dataclass
class PairwiseSeparability:
    """For each pair of classes, score every feature for discriminative power.

    Output: dict[(class_a, class_b)] -> DataFrame with columns
        feature, n_a, n_b, mean_a, mean_b, cliffs_delta, ks_stat, ks_p, auc.
    """

    name: str = "pairwise"
    requires: tuple[str, ...] = ()
    pairs: list[tuple[str, str]] | None = None  # default: all pairs
    top_n: int | None = None

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx)
        X = prep.X.values
        y_str = np.array(prep.class_names)[prep.y]

        classes = list(prep.class_names)
        pairs = self.pairs or list(combinations(classes, 2))

        results: dict[tuple[str, str], pd.DataFrame] = {}
        for a, b in pairs:
            mask_a = y_str == a
            mask_b = y_str == b
            if mask_a.sum() == 0 or mask_b.sum() == 0:
                continue

            rows = []
            for j, feat in enumerate(prep.feature_cols):
                va = X[mask_a, j]
                vb = X[mask_b, j]
                ks_stat, ks_p = ks_2samp(va, vb)
                y_bin = np.concatenate([np.zeros(len(va)), np.ones(len(vb))])
                vals = np.concatenate([va, vb])
                rows.append({
                    "feature": feat,
                    "n_a": int(len(va)),
                    "n_b": int(len(vb)),
                    "mean_a": float(np.mean(va)),
                    "mean_b": float(np.mean(vb)),
                    "cliffs_delta": _cliffs_delta(va, vb),
                    "ks_stat": float(ks_stat),
                    "ks_p": float(ks_p),
                    "auc": _single_feature_auc(vals, y_bin),
                })
            df = (
                pd.DataFrame(rows)
                .assign(abs_delta=lambda d: d["cliffs_delta"].abs())
                .sort_values(["auc", "abs_delta"], ascending=False)
                .drop(columns="abs_delta")
                .reset_index(drop=True)
            )
            if self.top_n:
                df = df.head(self.top_n)
            results[(a, b)] = df

        return {"pairs": results}

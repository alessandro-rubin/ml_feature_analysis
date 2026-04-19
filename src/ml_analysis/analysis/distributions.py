"""Per-feature distribution comparisons across classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from ml_analysis.analysis.base import AnalysisContext, prepare_xy


@dataclass
class DistributionAnalysis:
    """Per-feature group statistics + Kruskal-Wallis significance.

    Output is a long-form DataFrame with one row per (feature, class)
    plus a summary DataFrame with one row per feature ranking
    overall separability.
    """

    name: str = "distributions"
    requires: tuple[str, ...] = ()
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx)
        X = prep.X
        y_str = np.array(prep.class_names)[prep.y]

        rows = []
        summary_rows = []
        for feat in prep.feature_cols:
            groups = []
            for cls in prep.class_names:
                v = X.loc[y_str == cls, feat].values
                if len(v) == 0:
                    continue
                groups.append(v)
                row = {
                    "feature": feat,
                    "class": cls,
                    "n": int(len(v)),
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                }
                for q in self.quantiles:
                    row[f"q{int(q * 100):02d}"] = float(np.quantile(v, q))
                rows.append(row)

            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                try:
                    stat, p = kruskal(*groups)
                except ValueError:
                    stat, p = float("nan"), float("nan")
            else:
                stat, p = float("nan"), float("nan")
            summary_rows.append({"feature": feat, "kw_stat": float(stat), "kw_p": float(p)})

        per_feature_class = pd.DataFrame(rows)
        summary = (
            pd.DataFrame(summary_rows)
            .sort_values("kw_stat", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        return {"per_feature_class": per_feature_class, "summary": summary}

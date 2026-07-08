"""Feature importance analysis (RF MDI + permutation + statistical tests)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance

from tessa.analysis.base import AnalysisContext, prepare_xy


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0


@dataclass
class FeatureImportance:
    name: str = "importance"
    requires: tuple[str, ...] = ()
    rf_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
        )
    )
    permutation_repeats: int = 10

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx)
        X, y = prep.X, prep.y

        rf = RandomForestClassifier(**self.rf_params)
        rf.fit(X, y)
        mdi = pd.Series(rf.feature_importances_, index=prep.feature_cols, name="rf_mdi")

        perm = permutation_importance(
            rf, X, y,
            n_repeats=self.permutation_repeats,
            random_state=ctx.cfg.random_state,
            n_jobs=-1,
        )
        perm_mean = pd.Series(perm.importances_mean, index=prep.feature_cols, name="perm_mean")
        perm_std = pd.Series(perm.importances_std, index=prep.feature_cols, name="perm_std")

        f_scores, f_pvals = f_classif(X, y)
        anova = pd.DataFrame(
            {"anova_f": f_scores, "anova_p": f_pvals},
            index=prep.feature_cols,
        )

        kw_rows = []
        for col in prep.feature_cols:
            groups = [X.loc[y == c, col].values for c in np.unique(y)]
            stat, pval = kruskal(*groups)
            kw_rows.append({"feature": col, "kw_stat": stat, "kw_p": pval})
        kw = pd.DataFrame(kw_rows).set_index("feature")

        mi = pd.Series(
            mutual_info_classif(X, y, random_state=0),
            index=prep.feature_cols,
            name="mutual_info",
        )

        results = pd.concat([mdi, perm_mean, perm_std, anova, kw, mi], axis=1)
        composite = (
            _minmax(results["rf_mdi"])
            + _minmax(results["perm_mean"])
            + _minmax(results["anova_f"])
            + _minmax(results["kw_stat"])
            + _minmax(results["mutual_info"])
        ) / 5
        results["score_composite"] = composite
        results = results.sort_values("score_composite", ascending=False)
        results.insert(0, "rank", range(1, len(results) + 1))

        return {"table": results, "model": rf, "class_names": prep.class_names}

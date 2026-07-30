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

from tessa.analysis.base import AnalysisContext, prepare_xy, seeded


@dataclass
class FeatureImportance:
    name: str = "importance"
    requires: tuple[str, ...] = ()
    rf_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=None
        )
    )
    permutation_repeats: int = 10

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx)
        X, y = prep.X, prep.y

        rf = RandomForestClassifier(**seeded(self.rf_params, ctx.cfg))
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
            mutual_info_classif(X, y, random_state=ctx.cfg.random_state),
            index=prep.feature_cols,
            name="mutual_info",
        )

        results = pd.concat([mdi, perm_mean, perm_std, anova, kw, mi], axis=1)
        # Rank-based aggregation: per-method ranks are scale-free, so one
        # feature with a huge unbounded F statistic cannot dominate the blend
        # the way min-max normalization lets it.
        method_cols = ["rf_mdi", "perm_mean", "anova_f", "kw_stat", "mutual_info"]
        ranks = results[method_cols].rank(ascending=False, na_option="bottom")
        results["mean_rank"] = ranks.mean(axis=1)
        n = len(results)
        results["score_composite"] = (
            1.0 - (results["mean_rank"] - 1.0) / max(n - 1, 1)
        )
        results = results.sort_values("mean_rank", ascending=True)
        results.insert(0, "rank", range(1, len(results) + 1))

        return {"table": results, "model": rf, "class_names": prep.class_names}

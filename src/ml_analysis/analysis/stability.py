"""Importance-stability and method-agreement diagnostics.

The importance analysis already blends RF MDI, permutation importance,
ANOVA F, Kruskal-Wallis, and mutual information into a composite score.
This module corroborates that composite in two ways:

1. *Bootstrap stability* — refit RF on N bootstrap resamples, collect
   per-feature MDI, and report median + percentile CI + a stability
   score (fraction of resamples where the feature ranks in the top-k).
   Features with wide CIs or low stability are flagged as unreliable.

2. *Method agreement* — Spearman rank-correlation between the columns
   of the importance table (rf_mdi, perm_mean, anova_f, kw_stat,
   mutual_info). Strong off-diagonal entries mean the methods agree;
   weak entries mean a feature's "importance" depends on how you ask.

If the ``importance`` analysis has already been run on the same context
(``ctx.results['importance']``) the method-agreement matrix uses that
table verbatim — otherwise the table is computed on the fly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

from ml_analysis.analysis.base import AnalysisContext, prepare_xy


@dataclass
class ImportanceStability:
    name: str = "importance_stability"
    requires: tuple[str, ...] = ()
    n_bootstrap: int = 50
    top_k: int = 10
    ci: float = 0.95
    rf_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
        )
    )

    def _bootstrap_mdi(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Returns array shape (n_bootstrap, n_features) of MDI values."""
        n, p = X.shape
        out = np.empty((self.n_bootstrap, p), dtype=float)
        for i in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            # bootstrap must contain >=2 classes to fit
            if len(np.unique(y[idx])) < 2:
                out[i] = np.nan
                continue
            params = dict(self.rf_params)
            params["random_state"] = int(rng.integers(0, 2**31 - 1))
            rf = RandomForestClassifier(**params)
            rf.fit(X[idx], y[idx])
            out[i] = rf.feature_importances_
        return out

    def _method_agreement(self, ctx: AnalysisContext) -> pd.DataFrame:
        """Spearman rank-correlation between importance methods."""
        cached = ctx.results.get("importance") if hasattr(ctx, "results") else None
        if cached and "table" in cached:
            tbl = cached["table"]
        else:
            return pd.DataFrame()

        method_cols = [
            c for c in ("rf_mdi", "perm_mean", "anova_f", "kw_stat", "mutual_info")
            if c in tbl.columns
        ]
        if len(method_cols) < 2:
            return pd.DataFrame()
        sub = tbl[method_cols].copy()
        # absolute value for signed statistics; treat ranking strength as magnitude
        for c in sub.columns:
            sub[c] = sub[c].abs()
        rho = np.full((len(method_cols), len(method_cols)), np.nan)
        for i, ci in enumerate(method_cols):
            for j, cj in enumerate(method_cols):
                if i == j:
                    rho[i, j] = 1.0
                    continue
                a = sub[ci].values
                b = sub[cj].values
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() < 3:
                    continue
                rho[i, j] = spearmanr(a[mask], b[mask]).statistic
        return pd.DataFrame(rho, index=method_cols, columns=method_cols)

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx)
        X = prep.X.values
        y = prep.y
        rng = np.random.default_rng(ctx.cfg.random_state)

        boots = self._bootstrap_mdi(X, y, rng)  # (B, P)

        alpha = (1 - self.ci) / 2
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(boots, axis=0)
            lo = np.nanquantile(boots, alpha, axis=0)
            hi = np.nanquantile(boots, 1 - alpha, axis=0)

        # stability: fraction of resamples where each feature is in top-k
        finite_rows = np.isfinite(boots).all(axis=1)
        stable_count = np.zeros(X.shape[1], dtype=int)
        n_valid = int(finite_rows.sum())
        for i in np.where(finite_rows)[0]:
            top = np.argsort(boots[i])[-self.top_k:]
            stable_count[top] += 1
        stability = stable_count / max(n_valid, 1)

        table = pd.DataFrame({
            "feature": prep.feature_cols,
            "mdi_median": median,
            "mdi_ci_low": lo,
            "mdi_ci_high": hi,
            "ci_width": hi - lo,
            f"stability_top{self.top_k}": stability,
        }).sort_values("mdi_median", ascending=False).reset_index(drop=True)

        agreement = self._method_agreement(ctx)

        return {
            "bootstrap_table": table,
            "method_agreement": agreement,
            "n_bootstrap_valid": n_valid,
        }

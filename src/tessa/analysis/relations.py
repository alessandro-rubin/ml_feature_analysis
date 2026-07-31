"""Exploratory relations between channels: lagged association + MI network.

Both analyses are **exploratory** — they surface *predictive association*,
not proven causation, and their outputs say so explicitly.

- `LaggedRelations` — cross-correlation over a lag grid between a
  ``reference`` channel and every other channel (or all pairs when no
  reference is given). A channel that correlates with the reference best
  at a *positive* lag leads it — a candidate early-warning signal.
- `MutualInfoNetwork` — pairwise mutual information between features,
  catching nonlinear/monotone dependences Spearman misses. O(p²) MI
  estimates, so features are capped (top-variance) by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from tessa.analysis.base import AnalysisContext, prepare_xy


def lagged_correlations(x: np.ndarray, y: np.ndarray, max_lag: int) -> pd.DataFrame:
    """Pearson correlation of ``x[t]`` vs ``y[t+lag]`` for each lag.

    Positive lag = x leads y. Rows with non-finite values are dropped
    pairwise per lag.
    """
    rows = []
    n = len(x)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = x[: n - lag or None], y[lag:]
        else:
            a, b = x[-lag:], y[: n + lag]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(a[m], b[m])[0, 1])
        rows.append({"lag": lag, "correlation": corr})
    return pd.DataFrame(rows)


@dataclass
class LaggedRelations:
    name: str = "lagged_relations"
    requires: tuple[str, ...] = ()
    needs_labels: str = "none"
    reference: str | None = None  # channel to relate everything to
    max_lag: int = 24
    max_channels: int = 30
    channels: list[str] | None = None

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        df = ctx.filtered().to_pandas()
        ts_col = ctx.cfg.timestamp_col
        if ts_col in df.columns:
            sort_cols = (["asset_id"] if "asset_id" in df.columns else []) + [ts_col]
            df = df.sort_values(sort_cols).reset_index(drop=True)

        skip = {ctx.target_col, "event_id", "asset_id", ts_col}
        channels = self.channels or [
            c for c in df.select_dtypes(include="number").columns if c not in skip
        ]
        if self.reference is not None and self.reference not in channels:
            raise ValueError(f"reference {self.reference!r} not a numeric channel")
        if len(channels) > self.max_channels and self.reference is None:
            raise ValueError(
                f"{len(channels)} channels would make "
                f"{len(channels) * (len(channels) - 1) // 2} pairs; pass "
                "`reference=` or `channels=[...]` (or raise max_channels)."
            )

        pairs = (
            [(self.reference, c) for c in channels if c != self.reference]
            if self.reference is not None
            else list(combinations(channels, 2))
        )

        rows = []
        for a, b in pairs:
            lc = lagged_correlations(
                df[a].to_numpy(dtype=float),
                df[b].to_numpy(dtype=float),
                self.max_lag,
            )
            if lc["correlation"].notna().any():
                best = lc.iloc[lc["correlation"].abs().idxmax()]
                rows.append(
                    {
                        "leading": a,
                        "following": b,
                        "best_lag": int(best["lag"]),
                        "correlation_at_best_lag": float(best["correlation"]),
                        "correlation_at_zero": float(lc.loc[lc["lag"] == 0, "correlation"].iloc[0]),
                    }
                )
        table = (
            pd.DataFrame(rows)
            .sort_values("correlation_at_best_lag", key=abs, ascending=False)
            .reset_index(drop=True)
            if rows
            else pd.DataFrame(
                columns=[
                    "leading",
                    "following",
                    "best_lag",
                    "correlation_at_best_lag",
                    "correlation_at_zero",
                ]
            )
        )
        return {
            "table": table,
            "max_lag": self.max_lag,
            "note": "exploratory: lagged association, not causation",
        }


@dataclass
class MutualInfoNetwork:
    name: str = "mi_network"
    requires: tuple[str, ...] = ()
    needs_labels: str = "none"
    max_features: int = 30  # MI is O(p^2); cap by variance
    edge_threshold: float = 0.2  # bits; edges below are dropped
    mi_neighbors: int = 3

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx, ignore_target=True)
        X = prep.X
        if X.shape[1] > self.max_features:
            keep = X.var().sort_values(ascending=False).head(self.max_features).index.tolist()
            X = X[keep]
        cols = list(X.columns)
        Xv = X.to_numpy(dtype=float)

        p = len(cols)
        mi = np.zeros((p, p))
        for j in range(p):
            others = [i for i in range(p) if i != j]
            vals = mutual_info_regression(
                Xv[:, others],
                Xv[:, j],
                n_neighbors=self.mi_neighbors,
                random_state=ctx.cfg.random_state,
            )
            for i, v in zip(others, vals):
                mi[i, j] = v
        mi = (mi + mi.T) / 2  # symmetrize the kNN estimate

        matrix = pd.DataFrame(mi, index=cols, columns=cols)
        iu = np.triu_indices(p, k=1)
        edges = pd.DataFrame(
            {
                "feature_a": np.asarray(cols)[iu[0]],
                "feature_b": np.asarray(cols)[iu[1]],
                "mutual_info": mi[iu],
            }
        )
        edges = (
            edges[edges["mutual_info"] >= self.edge_threshold]
            .sort_values("mutual_info", ascending=False)
            .reset_index(drop=True)
        )
        return {
            "matrix": matrix,
            "edges": edges,
            "n_features_used": p,
            "note": "exploratory: dependence structure, not causation",
        }

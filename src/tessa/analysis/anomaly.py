"""Unsupervised anomaly detection with per-feature attribution.

This is the unsupervised entry point the toolkit was missing: it scores
every analysed row (window/event aggregate) for "how anomalous", with no
labels required, and traces every anomaly back to the sensors driving it.

Methods (each opt-out-able):

- *IsolationForest* — tree-isolation depth; robust default for ~100 features.
- *Local Outlier Factor* — density ratio vs. k nearest neighbours.
- *Robust Mahalanobis* (EllipticEnvelope / MinCovDet) — distance from the
  robust covariance ellipsoid; assumes a roughly elliptical healthy core.

Per-method scores are rank-normalized to [0, 1] (1 = most anomalous) so
they are comparable, and blended into ``ensemble`` by mean — the same
scale-free aggregation rationale as the importance composite.

A *baseline* can be supplied (e.g. ``baseline_filter={"class": ["healthy"]}``):
models are then fit on baseline rows only and score everything, which turns
"outlier among everything" into "deviates from healthy".

Per-feature attribution is a robust z-score against the baseline (median /
MAD): for each row, features with the largest |z| are the sensors that make
it anomalous. Model-agnostic, fast at 100 features, and directly plottable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from tessa.analysis.base import AnalysisContext, prepare_xy, seeded


def _rank_normalize(raw: np.ndarray) -> np.ndarray:
    """Map raw anomaly scores (higher = more anomalous) to [0, 1] ranks."""
    n = len(raw)
    if n <= 1:
        return np.zeros(n)
    return (sstats.rankdata(raw) - 1) / (n - 1)


@dataclass
class AnomalyDetection:
    """Ensemble anomaly scoring + per-feature contributions.

    Parameters
    ----------
    baseline_filter : dict, optional
        ``{column: [allowed values]}`` selecting the rows to fit on
        (e.g. healthy windows). ``None`` fits on all rows.
    contamination : float
        Expected anomaly fraction, forwarded to the detectors.
    top_k_contributors : int
        How many top features to report per row in ``top_contributors``.
    """

    name: str = "anomaly"
    requires: tuple[str, ...] = ()
    needs_labels: str = "none"
    baseline_filter: dict[str, list] | None = None
    contamination: float = 0.05
    n_neighbors: int = 20
    run_iforest: bool = True
    run_lof: bool = True
    run_mahalanobis: bool = True
    top_k_contributors: int = 5
    iforest_params: dict = field(
        default_factory=lambda: dict(n_estimators=300, n_jobs=-1, random_state=None)
    )

    def _baseline_mask(self, ctx: AnalysisContext, prep) -> np.ndarray:
        if self.baseline_filter is None:
            return np.ones(len(prep.X), dtype=bool)
        # the filter columns live on the full frame; rebuild them aligned to X
        df = ctx.filtered().to_pandas()
        aligned = df.loc[prep.X.index] if len(df) == len(prep.X) else None
        if aligned is None:
            # prepare_xy reset the index after row drops; refilter via ids
            raise ValueError(
                "baseline_filter requires the filter columns to survive "
                "preparation; ensure no rows are dropped or filter via "
                "ctx.label_filter instead."
            )
        mask = np.ones(len(aligned), dtype=bool)
        for col, allowed in self.baseline_filter.items():
            mask &= aligned[col].isin(allowed).to_numpy()
        if mask.sum() < 5:
            raise ValueError(
                f"baseline_filter matched only {int(mask.sum())} rows; need at least 5 to fit."
            )
        return mask

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx, ignore_target=True)
        X = prep.X.to_numpy(dtype=float)
        base_mask = self._baseline_mask(ctx, prep)

        scaler = StandardScaler().fit(X[base_mask])
        Z = scaler.transform(X)
        Z_base = Z[base_mask]

        raw: dict[str, np.ndarray] = {}
        models: dict[str, Any] = {}

        if self.run_iforest:
            iso = IsolationForest(
                contamination=self.contamination,
                **seeded(self.iforest_params, ctx.cfg),
            ).fit(Z_base)
            raw["iforest"] = -iso.score_samples(Z)  # higher = more anomalous
            models["iforest"] = iso

        if self.run_lof:
            k = min(self.n_neighbors, max(2, len(Z_base) - 1))
            lof = LocalOutlierFactor(
                n_neighbors=k, novelty=True, contamination=self.contamination
            ).fit(Z_base)
            raw["lof"] = -lof.score_samples(Z)
            models["lof"] = lof

        if self.run_mahalanobis and len(Z_base) > Z.shape[1]:
            try:
                mcd = MinCovDet(random_state=ctx.cfg.random_state).fit(Z_base)
                raw["mahalanobis"] = mcd.mahalanobis(Z)
                models["mahalanobis"] = mcd
            except Exception as err:  # singular covariance etc.
                models["mahalanobis"] = f"skipped: {type(err).__name__}: {err}"

        if not raw:
            raise RuntimeError("No anomaly detector produced scores.")

        scores = pd.DataFrame({name: _rank_normalize(r) for name, r in raw.items()})
        scores["ensemble"] = scores.mean(axis=1)
        if prep.ids is not None:
            scores = pd.concat([prep.ids.reset_index(drop=True), scores], axis=1)

        # Robust z-score attribution vs the baseline distribution.
        med = np.median(X[base_mask], axis=0)
        mad = sstats.median_abs_deviation(X[base_mask], axis=0, scale="normal")
        mad = np.where(mad > 0, mad, np.nan)  # constant features contribute 0
        with np.errstate(invalid="ignore"):
            z = (X - med) / mad
        z = np.nan_to_num(z, nan=0.0)
        contributions = pd.DataFrame(z, columns=prep.feature_cols)

        order = np.argsort(-np.abs(z), axis=1)[:, : self.top_k_contributors]
        feat = np.asarray(prep.feature_cols)
        top_contributors = [[(feat[j], float(z[i, j])) for j in order[i]] for i in range(len(z))]

        return {
            "scores": scores,
            "feature_contributions": contributions,
            "top_contributors": top_contributors,
            "baseline_mask": base_mask,
            "models": models,
            "preparation": prep.report,
        }

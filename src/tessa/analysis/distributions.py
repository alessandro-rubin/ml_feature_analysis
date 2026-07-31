"""Per-feature distribution comparisons across classes.

The summary table reports the Kruskal-Wallis H-test (rank-based,
non-parametric) as the primary "are these groups different?" statistic.
It is corroborated by:

- Anderson-Darling k-sample test (more sensitive to tail differences),
- One-way ANOVA F-test (parametric counterpart),
- Levene's test (variance equality — diagnoses why a t-test would fail),
- multiple-testing-corrected p-values (Bonferroni, BH-FDR) so that the
  per-feature p-values can be interpreted across the ~90-feature family.

The per-feature-class table is extended with skewness, excess kurtosis,
median absolute deviation, and IQR so distribution shape — not just
location — is visible.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import (
    anderson_ksamp,
    f_oneway,
    iqr,
    kruskal,
    kurtosis,
    levene,
    skew,
)

from tessa.analysis.base import AnalysisContext, prepare_xy
from tessa.analysis.multiple_testing import benjamini_hochberg, bonferroni


def _safe_kruskal(groups: list[np.ndarray]) -> tuple[float, float]:
    if len(groups) < 2 or any(len(g) == 0 for g in groups):
        return float("nan"), float("nan")
    try:
        s, p = kruskal(*groups)
        return float(s), float(p)
    except ValueError:
        return float("nan"), float("nan")


def _safe_anova(groups: list[np.ndarray]) -> tuple[float, float]:
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return float("nan"), float("nan")
    try:
        s, p = f_oneway(*groups)
        return float(s), float(p)
    except (ValueError, ZeroDivisionError):
        return float("nan"), float("nan")


def _safe_anderson(groups: list[np.ndarray]) -> tuple[float, float, bool]:
    """Returns (statistic, p, capped). SciPy floors/caps the A-D p-value to
    [0.001, 0.25]; ``capped`` flags those so 0.001/0.25 aren't read as exact."""
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return float("nan"), float("nan"), False
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*p-value (capped|floored).*")
            res = anderson_ksamp(groups, variant="midrank")
        # SciPy >= ~1.13 returns a SignificanceResult with a ``pvalue`` field;
        # older versions exposed ``significance_level`` as a percentage.
        p = getattr(res, "pvalue", None)
        if p is None:
            p = float(getattr(res, "significance_level", float("nan"))) / 100.0
        capped = bool(np.isclose(p, 0.001) or np.isclose(p, 0.25))
        return float(res.statistic), float(p), capped
    except (ValueError, OverflowError):
        return float("nan"), float("nan"), False


def _safe_levene(groups: list[np.ndarray]) -> tuple[float, float]:
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return float("nan"), float("nan")
    try:
        s, p = levene(*groups, center="median")
        return float(s), float(p)
    except ValueError:
        return float("nan"), float("nan")


@dataclass
class DistributionAnalysis:
    """Per-feature group statistics + multi-test significance.

    Output:
        per_feature_class: long-form DataFrame, one row per (feature, class)
            with descriptive stats including shape (skew, kurtosis, MAD, IQR).
        summary: one row per feature with k-group test statistics and
            multiple-testing-corrected p-values.
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
            groups: list[np.ndarray] = []
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
                    "skew": float(skew(v)) if len(v) > 2 else float("nan"),
                    "kurtosis": float(kurtosis(v)) if len(v) > 3 else float("nan"),
                    "mad": float(np.median(np.abs(v - np.median(v)))),
                    "iqr": float(iqr(v)) if len(v) > 1 else float("nan"),
                }
                for q in self.quantiles:
                    row[f"q{int(q * 100):02d}"] = float(np.quantile(v, q))
                rows.append(row)

            kw_stat, kw_p = _safe_kruskal(groups)
            anova_f, anova_p = _safe_anova(groups)
            ad_stat, ad_p, ad_capped = _safe_anderson(groups)
            lev_stat, lev_p = _safe_levene(groups)
            summary_rows.append(
                {
                    "feature": feat,
                    "kw_stat": kw_stat,
                    "kw_p": kw_p,
                    "anova_f": anova_f,
                    "anova_p": anova_p,
                    "ad_stat": ad_stat,
                    "ad_p": ad_p,
                    "ad_p_capped": ad_capped,
                    "levene_stat": lev_stat,
                    "levene_p": lev_p,
                }
            )

        per_feature_class = pd.DataFrame(rows)
        summary = pd.DataFrame(summary_rows)

        # Multiple-testing correction over the family of features.
        if len(summary):
            for col in ("kw_p", "anova_p", "ad_p", "levene_p"):
                summary[f"{col}_bonferroni"] = bonferroni(summary[col].values)
                summary[f"{col}_bh_fdr"] = benjamini_hochberg(summary[col].values)

        summary = summary.sort_values("kw_stat", ascending=False, na_position="last").reset_index(
            drop=True
        )
        return {"per_feature_class": per_feature_class, "summary": summary}

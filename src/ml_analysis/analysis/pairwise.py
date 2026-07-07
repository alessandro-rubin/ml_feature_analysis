"""Pairwise class separability with corroborating tests and CIs.

For each (A, B) class pair, every feature is scored with a battery of
complementary statistics so a single test's quirks (KS sensitive to any
distribution difference, Mann-Whitney sensitive to stochastic dominance,
t-test parametric) don't dominate the ranking:

- Kolmogorov-Smirnov (any difference in CDF)
- Mann-Whitney U (rank-sum, stochastic dominance)
- Brunner-Munzel (Mann-Whitney variant robust to unequal variance)
- Welch's t (parametric, unequal variance)

Effect sizes (Cliff's delta, rank-biserial, Cohen's d, Hedges' g) and
distribution distances (Wasserstein, Jensen-Shannon) corroborate the
p-values. Bootstrap percentile CIs surface uncertainty around AUC and
Cliff's delta. Per-pair BH-FDR adjusts the p-values across the
feature family.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import brunnermunzel, ks_2samp, mannwhitneyu, ttest_ind, wasserstein_distance
from sklearn.metrics import roc_auc_score

from ml_analysis.analysis.base import AnalysisContext, prepare_xy
from ml_analysis.analysis.effect_sizes import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    hedges_g,
    js_divergence,
    rank_biserial_from_u,
)
from ml_analysis.analysis.multiple_testing import benjamini_hochberg, bonferroni


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Backward-compatible re-export — preserved for external imports."""
    return cliffs_delta(a, b)


def _single_feature_auc(values: np.ndarray, y_binary: np.ndarray) -> float:
    """AUC for 'larger value -> class 1', direction-free via max(auc, 1-auc)."""
    try:
        auc = roc_auc_score(y_binary, values)
        return float(max(auc, 1 - auc))
    except ValueError:
        return float("nan")


def _safe_test(fn, va: np.ndarray, vb: np.ndarray) -> tuple[float, float]:
    if len(va) < 2 or len(vb) < 2:
        return float("nan"), float("nan")
    try:
        # Degenerate inputs (e.g. a group constant in value/rank) make some
        # scipy tests divide by zero and emit RuntimeWarnings while still
        # returning NaN. Silence the noise; the NaN result is handled downstream.
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            res = fn(va, vb)
        return float(res[0]), float(res[1])
    except (ValueError, ZeroDivisionError):
        return float("nan"), float("nan")


@dataclass
class PairwiseSeparability:
    """For each pair of classes, score every feature for discriminative power.

    Output:
        ``pairs``: dict[(class_a, class_b)] -> DataFrame with columns
            ``feature, n_a, n_b, mean_a, mean_b, cliffs_delta,
            rank_biserial, cohens_d, hedges_g, wasserstein, js_divergence,
            ks_stat, ks_p, mwu_stat, mwu_p, bm_p, welch_t, welch_p, auc``
            plus ``*_bh_fdr`` / ``*_bonferroni`` columns and
            ``auc_ci_low``, ``auc_ci_high``, ``cliffs_ci_low``,
            ``cliffs_ci_high`` when ``bootstrap_n > 0``.
    """

    name: str = "pairwise"
    requires: tuple[str, ...] = ()
    pairs: list[tuple[str, str]] | None = None
    top_n: int | None = None
    bootstrap_n: int = 0
    bootstrap_ci: float = 0.95

    _MTC_PVAL_COLS = ("ks_p", "mwu_p", "bm_p", "welch_p")

    def _row_for_feature(
        self, feat: str, va: np.ndarray, vb: np.ndarray, rng: np.random.Generator
    ) -> dict[str, Any]:
        ks_stat, ks_p = _safe_test(ks_2samp, va, vb)
        mwu_stat, mwu_p = _safe_test(
            lambda x, y: mannwhitneyu(x, y, alternative="two-sided"), va, vb
        )
        _, bm_p = _safe_test(
            lambda x, y: brunnermunzel(x, y, alternative="two-sided"), va, vb
        )
        welch_t, welch_p = _safe_test(
            lambda x, y: ttest_ind(x, y, equal_var=False), va, vb
        )

        y_bin = np.concatenate([np.zeros(len(va)), np.ones(len(vb))])
        vals = np.concatenate([va, vb])
        auc = _single_feature_auc(vals, y_bin)

        delta = cliffs_delta(va, vb)
        rb = (
            rank_biserial_from_u(mwu_stat, len(va), len(vb))
            if np.isfinite(mwu_stat)
            else float("nan")
        )

        try:
            wd = float(wasserstein_distance(va, vb))
        except ValueError:
            wd = float("nan")

        row: dict[str, Any] = {
            "feature": feat,
            "n_a": int(len(va)),
            "n_b": int(len(vb)),
            "mean_a": float(np.mean(va)) if len(va) else float("nan"),
            "mean_b": float(np.mean(vb)) if len(vb) else float("nan"),
            "cliffs_delta": delta,
            "rank_biserial": rb,
            "cohens_d": cohens_d(va, vb),
            "hedges_g": hedges_g(va, vb),
            "wasserstein": wd,
            "js_divergence": js_divergence(va, vb),
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "mwu_stat": mwu_stat,
            "mwu_p": mwu_p,
            "bm_p": bm_p,
            "welch_t": welch_t,
            "welch_p": welch_p,
            "auc": auc,
        }

        if self.bootstrap_n > 0 and len(va) > 1 and len(vb) > 1:
            def _auc_stat(x, y):
                yb = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
                return _single_feature_auc(np.concatenate([x, y]), yb)

            _, auc_lo, auc_hi = bootstrap_ci(
                _auc_stat, va, vb,
                n_resamples=self.bootstrap_n, ci=self.bootstrap_ci, rng=rng,
            )
            _, dl_lo, dl_hi = bootstrap_ci(
                cliffs_delta, va, vb,
                n_resamples=self.bootstrap_n, ci=self.bootstrap_ci, rng=rng,
            )
            row.update({
                "auc_ci_low": auc_lo,
                "auc_ci_high": auc_hi,
                "cliffs_ci_low": dl_lo,
                "cliffs_ci_high": dl_hi,
            })
        return row

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx)
        X = prep.X.values
        y_str = np.array(prep.class_names)[prep.y]

        classes = list(prep.class_names)
        pairs = self.pairs or list(combinations(classes, 2))
        rng = np.random.default_rng(ctx.cfg.random_state)

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
                rows.append(self._row_for_feature(feat, va, vb, rng))

            df = pd.DataFrame(rows)
            for col in self._MTC_PVAL_COLS:
                if col in df.columns:
                    df[f"{col}_bh_fdr"] = benjamini_hochberg(df[col].values)
                    df[f"{col}_bonferroni"] = bonferroni(df[col].values)

            df = (
                df.assign(abs_delta=lambda d: d["cliffs_delta"].abs())
                .sort_values(["auc", "abs_delta"], ascending=False)
                .drop(columns="abs_delta")
                .reset_index(drop=True)
            )
            if self.top_n:
                df = df.head(self.top_n)
            results[(a, b)] = df

        # `pairs` is a dict keyed by class tuples — the store drops it, so the
        # dashboard/report would lose the volcano + AUC-CI plots. A flattened
        # copy (one frame, class pair as columns) survives serialization and
        # is queryable too.
        if results:
            pairs_long = pd.concat(
                [df.assign(class_a=a, class_b=b) for (a, b), df in results.items()],
                ignore_index=True,
            )
        else:
            pairs_long = pd.DataFrame()

        return {"pairs": results, "pairs_long": pairs_long}

"""Effect-size and distribution-distance utilities.

Effect sizes complement p-values: a tiny p-value with a negligible
effect size is not a useful discovery. We expose both parametric
(Cohen's d, Hedges' g) and non-parametric (Cliff's delta, rank-biserial
from Mann-Whitney U) families, plus distribution-level distances
(Wasserstein, Jensen-Shannon) that don't assume a location shift.

A small bootstrap helper produces percentile CIs for any scalar
statistic of two samples.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]. 0 = no separation."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    x = np.asarray(a)
    y = np.asarray(b)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD standardized mean difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Small-sample-corrected Cohen's d."""
    d = cohens_d(a, b)
    n1, n2 = len(a), len(b)
    df = n1 + n2 - 2
    if df < 1 or not np.isfinite(d):
        return float("nan")
    # Hedges' correction factor (approximate; close to 1 for df > ~20)
    j = 1.0 - 3.0 / (4 * df - 1)
    return float(j * d)


def rank_biserial_from_u(u: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation derived from the Mann-Whitney U statistic."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    return float(1.0 - (2.0 * u) / (n1 * n2))


def js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    """Jensen-Shannon divergence (bits) between two 1-D samples.

    Histograms are built on the shared support; small additive smoothing
    avoids log(0). JS in bits is bounded by 1.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(a, bins=edges, density=False)
    q, _ = np.histogram(b, bins=edges, density=False)
    p = p.astype(float) + 1e-12
    q = q.astype(float) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def bootstrap_ci(
    stat_fn: Callable[..., float],
    *samples: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for any scalar two-sample statistic.

    Returns (point_estimate, lower, upper). Each resample draws with
    replacement from each input sample at its original size.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    point = stat_fn(*samples)
    if not np.isfinite(point):
        return float(point), float("nan"), float("nan")
    sizes = [len(s) for s in samples]
    if any(n == 0 for n in sizes):
        return float(point), float("nan"), float("nan")
    arrs = [np.asarray(s) for s in samples]
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resampled = tuple(
            a[rng.integers(0, n, size=n)] for a, n in zip(arrs, sizes)
        )
        try:
            stats[i] = stat_fn(*resampled)
        except Exception:
            stats[i] = np.nan
    finite = stats[np.isfinite(stats)]
    if len(finite) == 0:
        return float(point), float("nan"), float("nan")
    alpha = (1 - ci) / 2
    lo = float(np.quantile(finite, alpha))
    hi = float(np.quantile(finite, 1 - alpha))
    return float(point), lo, hi


__all__ = [
    "cliffs_delta",
    "cohens_d",
    "hedges_g",
    "rank_biserial_from_u",
    "js_divergence",
    "bootstrap_ci",
]

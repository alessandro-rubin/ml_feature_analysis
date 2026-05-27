"""Multiple-testing correction utilities.

With ~90 features per analysis, raw p-values from per-feature tests
(Kruskal-Wallis, Mann-Whitney, KS, etc.) over-report significance.
Use Bonferroni for strict family-wise error control, Holm for a
slightly less conservative step-down variant, and Benjamini-Hochberg
for false-discovery-rate control (typical default in ML feature
selection).
"""

from __future__ import annotations

import numpy as np


def _as_array(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError("p-values must be a 1-D array")
    return p


def bonferroni(pvals) -> np.ndarray:
    p = _as_array(pvals)
    finite = np.isfinite(p)
    m = int(finite.sum())
    out = np.full_like(p, np.nan)
    if m == 0:
        return out
    out[finite] = np.minimum(p[finite] * m, 1.0)
    return out


def holm(pvals) -> np.ndarray:
    """Holm step-down. Returns FWER-adjusted p-values in the original order."""
    p = _as_array(pvals)
    finite = np.isfinite(p)
    m = int(finite.sum())
    out = np.full_like(p, np.nan)
    if m == 0:
        return out
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for k, i in enumerate(order):
        val = min((m - k) * p[i], 1.0)
        running_max = max(running_max, val)
        adj[k] = running_max
    for k, i in enumerate(order):
        out[i] = adj[k]
    return out


def benjamini_hochberg(pvals) -> np.ndarray:
    """BH-FDR adjusted p-values (a.k.a. q-values) in the original order."""
    p = _as_array(pvals)
    finite = np.isfinite(p)
    m = int(finite.sum())
    out = np.full_like(p, np.nan)
    if m == 0:
        return out
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    sorted_p = p[order]
    ranks = np.arange(1, m + 1)
    adj_sorted = sorted_p * m / ranks
    # enforce monotone non-decreasing in original p order by taking
    # the running min from the largest p downward
    for k in range(m - 2, -1, -1):
        adj_sorted[k] = min(adj_sorted[k], adj_sorted[k + 1])
    adj_sorted = np.minimum(adj_sorted, 1.0)
    for k, i in enumerate(order):
        out[i] = adj_sorted[k]
    return out


__all__ = ["bonferroni", "holm", "benjamini_hochberg"]

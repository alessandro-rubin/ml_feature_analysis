"""Stock per-sample features and aggregators.

These register into the default registries on import. Add your own using
the same @feature / @aggregate decorators.
"""

from __future__ import annotations

import polars as pl

from ml_analysis.features.aggregates import aggregate
from ml_analysis.features.registry import feature


# ── Per-sample feature factories ─────────────────────────────────────────────
# These are templates: they need to be parameterized per source column at
# registration time. Users typically call `make_*` to register one per signal.


def make_rolling_mean(source: str, window: int) -> None:
    name = f"{source}__roll_mean_{window}"

    @feature(name, deps=(source,))
    def _():
        return pl.col(source).rolling_mean(window)


def make_rolling_std(source: str, window: int) -> None:
    name = f"{source}__roll_std_{window}"

    @feature(name, deps=(source,))
    def _():
        return pl.col(source).rolling_std(window)


def make_first_difference(source: str) -> None:
    name = f"{source}__diff1"

    @feature(name, deps=(source,))
    def _():
        return pl.col(source).diff()


def make_zscore(source: str, window: int) -> None:
    """Rolling z-score using rolling mean/std of the same window."""
    name = f"{source}__zscore_{window}"

    @feature(name, deps=(source,))
    def _():
        m = pl.col(source).rolling_mean(window)
        s = pl.col(source).rolling_std(window)
        return (pl.col(source) - m) / s


# ── Aggregators ──────────────────────────────────────────────────────────────


@aggregate("mean")
def _(c: str) -> pl.Expr:
    return pl.col(c).mean()


@aggregate("std")
def _(c: str) -> pl.Expr:
    return pl.col(c).std()


@aggregate("min")
def _(c: str) -> pl.Expr:
    return pl.col(c).min()


@aggregate("max")
def _(c: str) -> pl.Expr:
    return pl.col(c).max()


@aggregate("median")
def _(c: str) -> pl.Expr:
    return pl.col(c).median()


@aggregate("p05")
def _(c: str) -> pl.Expr:
    return pl.col(c).quantile(0.05)


@aggregate("p95")
def _(c: str) -> pl.Expr:
    return pl.col(c).quantile(0.95)


@aggregate("range")
def _(c: str) -> pl.Expr:
    return pl.col(c).max() - pl.col(c).min()


@aggregate("iqr")
def _(c: str) -> pl.Expr:
    return pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25)

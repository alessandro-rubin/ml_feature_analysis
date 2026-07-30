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
    """Register a rolling-mean feature for ``source``.

    Parameters
    ----------
    source : str
        Name of the input column.
    window : int
        Rolling-window size in samples.

    Notes
    -----
    The new feature is registered as ``"<source>__roll_mean_<window>"`` in
    the default feature registry.
    """
    name = f"{source}__roll_mean_{window}"

    @feature(name, deps=(source,))
    def _():
        return pl.col(source).rolling_mean(window)


def make_rolling_std(source: str, window: int) -> None:
    """Register a rolling-std feature for ``source``.

    Parameters
    ----------
    source : str
        Name of the input column.
    window : int
        Rolling-window size in samples.

    Notes
    -----
    The new feature is registered as ``"<source>__roll_std_<window>"`` in
    the default feature registry.
    """
    name = f"{source}__roll_std_{window}"

    @feature(name, deps=(source,))
    def _():
        return pl.col(source).rolling_std(window)


def make_first_difference(source: str) -> None:
    """Register a first-difference feature for ``source``.

    Parameters
    ----------
    source : str
        Name of the input column.

    Notes
    -----
    The new feature is registered as ``"<source>__diff1"`` in the default
    feature registry. The first sample of each event is null because there
    is no prior value to subtract.
    """
    name = f"{source}__diff1"

    @feature(name, deps=(source,))
    def _():
        return pl.col(source).diff()


def make_zscore(source: str, window: int) -> None:
    """Register a rolling z-score feature for ``source``.

    The z-score is computed as ``(x - rolling_mean) / rolling_std``, both
    over the same window.

    Parameters
    ----------
    source : str
        Name of the input column.
    window : int
        Rolling-window size in samples used for both mean and std.

    Notes
    -----
    The new feature is registered as ``"<source>__zscore_<window>"`` in the
    default feature registry.
    """
    name = f"{source}__zscore_{window}"

    @feature(name, deps=(source,))
    def _():
        m = pl.col(source).rolling_mean(window)
        s = pl.col(source).rolling_std(window)
        return (pl.col(source) - m) / s


def make_constant_counter(source: str) -> None:
    """Register a constant-sample counter for ``source``.

    The counter starts at 0 and increases by one for each sample whose value
    equals the previous sample's, resetting to 0 whenever the value changes.
    In other words it is the 0-based position within the current run of
    identical consecutive values::

        [0, 1, 6, 2, 3, 3, 3, 7] -> [0, 0, 0, 0, 0, 1, 2, 0]

    Parameters
    ----------
    source : str
        Name of the input column.

    Notes
    -----
    The new feature is registered as ``"<source>__const_count"`` in the
    default feature registry. Because features are materialised one event at
    a time, the counter never carries a run across event boundaries; the first
    sample of each event is always 0.
    """
    name = f"{source}__const_count"

    @feature(name, deps=(source,))
    def _():
        # A new run starts wherever the value differs from the prior sample;
        # the first sample (null shift) also begins a run. Numbering rows
        # from 0 within each run gives the per-run counter.
        run_id = (pl.col(source) != pl.col(source).shift(1)).fill_null(True).cum_sum()
        return pl.int_range(pl.len()).over(run_id)


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

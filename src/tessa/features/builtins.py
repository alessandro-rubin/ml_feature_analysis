"""Stock per-sample features and aggregators.

These register into the default registries on import. Add your own using
the same @feature / @aggregate decorators.

Per-sample features run on each signal's own samples
------------------------------------------------------
:func:`asset_loader.load_event` combines an asset's sources with a full outer
join on the timestamp by default, so a 5-minute source joined onto a 1-minute
source holds a null on 4 of every 5 rows. Taken row by row, ``diff()`` and
rolling windows on such a column would be null everywhere.

The ``make_*`` factories below therefore evaluate their expression over the
non-null rows of the source column only (a window partitioned by
``is_null()``, see :func:`on_own_samples`) and leave the null rows null. For a
slow source this means:

- ``diff1`` is the step between consecutive *samples of that source*;
- ``window`` counts samples of that source, not rows of the joined frame
  (``make_rolling_std("pressure", 10)`` spans 10 pressure samples, i.e. 50
  minutes at a 5-minute rate);
- the feature is non-null only on rows where the source was sampled, which
  the aggregators (null-skipping) handle transparently.

The same applies to genuine gaps in a single source: a missing sample is
skipped, not propagated through the whole window. To align slow sources onto
the fast grid instead (and difference the held values), load the events with
``merge="asof"`` (see :func:`tessa.dataset.builder.build`).
"""

from __future__ import annotations

import polars as pl

from tessa.features.aggregates import aggregate
from tessa.features.registry import FeatureRegistry, feature


# ── Per-sample feature factories ─────────────────────────────────────────────
# These are templates: they need to be parameterized per source column at
# registration time. Users typically call `make_*` to register one per signal.


def on_own_samples(expr: pl.Expr, source: str) -> pl.Expr:
    """Evaluate ``expr`` on the non-null rows of ``source`` only.

    The frame is partitioned by ``pl.col(source).is_null()``, so order-aware
    operations (``diff``, ``shift``, rolling windows, cumulative sums) see the
    source's samples back to back, and results are scattered back to the
    original rows. Rows where ``source`` is null get whatever ``expr`` yields
    on an all-null partition, which for the stock features is null.

    Parameters
    ----------
    expr : pl.Expr
        Expression over ``pl.col(source)``.
    source : str
        Column whose nulls mark rows the source was not sampled at.

    Returns
    -------
    pl.Expr
        ``expr`` windowed over the null mask of ``source``.
    """
    return expr.over(pl.col(source).is_null().alias(f"__{source}__is_null"))


def make_rolling_mean(source: str, window: int, registry: FeatureRegistry | None = None) -> None:
    """Register a rolling-mean feature for ``source``.

    Parameters
    ----------
    source : str
        Name of the input column.
    window : int
        Rolling-window size in samples of ``source`` (null rows are skipped,
        see the module docstring).
    registry : FeatureRegistry, optional
        Target registry. Defaults to the process-wide default registry.

    Notes
    -----
    The new feature is registered as ``"<source>__roll_mean_<window>"`` in
    the target registry.
    """
    name = f"{source}__roll_mean_{window}"

    @feature(name, deps=(source,), registry=registry)
    def _():
        return on_own_samples(pl.col(source).rolling_mean(window), source)


def make_rolling_std(source: str, window: int, registry: FeatureRegistry | None = None) -> None:
    """Register a rolling-std feature for ``source``.

    Parameters
    ----------
    source : str
        Name of the input column.
    window : int
        Rolling-window size in samples of ``source`` (null rows are skipped,
        see the module docstring).
    registry : FeatureRegistry, optional
        Target registry. Defaults to the process-wide default registry.

    Notes
    -----
    The new feature is registered as ``"<source>__roll_std_<window>"`` in
    the target registry.
    """
    name = f"{source}__roll_std_{window}"

    @feature(name, deps=(source,), registry=registry)
    def _():
        return on_own_samples(pl.col(source).rolling_std(window), source)


def make_first_difference(source: str, registry: FeatureRegistry | None = None) -> None:
    """Register a first-difference feature for ``source``.

    Parameters
    ----------
    source : str
        Name of the input column.
    registry : FeatureRegistry, optional
        Target registry. Defaults to the process-wide default registry.

    Notes
    -----
    The new feature is registered as ``"<source>__diff1"`` in the target
    registry. It is the step between consecutive non-null samples of
    ``source``, so a slower source joined onto a faster grid still gets one
    value per own sample. The first sample of each event is null because
    there is no prior value to subtract, and so is every row where
    ``source`` is null.
    """
    name = f"{source}__diff1"

    @feature(name, deps=(source,), registry=registry)
    def _():
        return on_own_samples(pl.col(source).diff(), source)


def make_zscore(source: str, window: int, registry: FeatureRegistry | None = None) -> None:
    """Register a rolling z-score feature for ``source``.

    The z-score is computed as ``(x - rolling_mean) / rolling_std``, both
    over the same window.

    Parameters
    ----------
    source : str
        Name of the input column.
    window : int
        Rolling-window size in samples of ``source``, used for both mean and
        std (null rows are skipped, see the module docstring).
    registry : FeatureRegistry, optional
        Target registry. Defaults to the process-wide default registry.

    Notes
    -----
    The new feature is registered as ``"<source>__zscore_<window>"`` in the
    target registry.
    """
    name = f"{source}__zscore_{window}"

    @feature(name, deps=(source,), registry=registry)
    def _():
        m = pl.col(source).rolling_mean(window)
        s = pl.col(source).rolling_std(window)
        return on_own_samples((pl.col(source) - m) / s, source)


def make_constant_counter(source: str, registry: FeatureRegistry | None = None) -> None:
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
    registry : FeatureRegistry, optional
        Target registry. Defaults to the process-wide default registry.

    Notes
    -----
    The new feature is registered as ``"<source>__const_count"`` in the
    target registry. Because features are materialised one event at
    a time, the counter never carries a run across event boundaries; the first
    sample of each event is always 0. Rows where ``source`` is null are
    skipped: they get a null counter and neither extend nor break a run.
    """
    name = f"{source}__const_count"

    @feature(name, deps=(source,), registry=registry)
    def _():
        # A new run starts wherever the value differs from the prior sample;
        # the first sample (null shift) also begins a run. Numbering rows
        # from 0 within each run gives the per-run counter. Both steps are
        # partitioned by the null mask so only the source's own samples count.
        x = pl.col(source)
        run_id = on_own_samples((x != x.shift(1)).fill_null(True).cum_sum(), source)
        counter = pl.int_range(pl.len()).over(
            x.is_null().alias(f"__{source}__is_null"), run_id.alias(f"__{source}__run")
        )
        return pl.when(x.is_not_null()).then(counter)


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

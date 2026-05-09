"""Materialise per-event LazyFrames into per-sample, windowed, or period outputs.

Three entry points form a coarse-to-fine spectrum:

- :func:`to_per_sample` — adds registered features as new columns, one row
  per input sample.
- :func:`to_windowed` — groups each event into fixed time windows and
  aggregates within each, returning N rows per event.
- :func:`to_period` — collapses each event to a single row of summary
  statistics over the full event duration.

All three accept a feature registry and an aggregator registry; each falls
back to the process-wide default registries when not specified.
"""

from __future__ import annotations

from typing import Iterable

import polars as pl

from ml_analysis.config import Config
from ml_analysis.features.aggregates import (
    AggregatorRegistry,
    default_registry as default_aggs,
)
from ml_analysis.features.registry import (
    FeatureRegistry,
    default_registry as default_features,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _label_cols(lf: pl.LazyFrame, cfg: Config) -> list[str]:
    """Return the columns that came from label metadata + ids.

    Heuristic: known label/id columns (``event_id``, ``asset_id``,
    ``cfg.class_col``) plus any other non-numeric, non-timestamp columns.
    Used by the materialisers to know what to carry through aggregation
    and what to exclude from automatic source selection.
    """
    schema = lf.collect_schema()
    candidates = ["event_id", "asset_id", cfg.class_col]
    return [c for c in candidates if c in schema.names()] + [
        c for c in schema.names()
        if c not in candidates
        and c != cfg.timestamp_col
        and not schema[c].is_numeric()
    ]


def _resolve_features(
    feature_names: list[str] | None,
    registry: FeatureRegistry,
) -> list:
    """Return registered specs for ``feature_names`` (or all if ``None``)."""
    if feature_names is None:
        return registry.resolve()
    return registry.resolve(feature_names)


# ── Per-sample ───────────────────────────────────────────────────────────────

def to_per_sample(
    lf: pl.LazyFrame,
    cfg: Config,
    feature_names: list[str] | None = None,
    registry: FeatureRegistry | None = None,
) -> pl.LazyFrame:
    """Add registered features as columns, leaving row count unchanged.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input frame, typically one event from :func:`ml_analysis.dataset.builder.build`.
    cfg : Config
        Project configuration. Currently unused inside this function but
        kept for signature symmetry with the other materialisers.
    feature_names : list of str, optional
        Subset of features to apply. If ``None`` (default), every feature
        in ``registry`` is applied. Pass ``[]`` to skip features entirely.
    registry : FeatureRegistry, optional
        Registry to resolve features from. Defaults to the process-wide
        registry.

    Returns
    -------
    pl.LazyFrame
        The input frame with one new column per applied feature, in
        dependency order.
    """
    reg = registry or default_features()
    specs = _resolve_features(feature_names, reg)
    out = lf
    for spec in specs:
        out = out.with_columns(spec.expr())
    return out


# ── Windowed (groupby_dynamic) ───────────────────────────────────────────────

def to_windowed(
    lf: pl.LazyFrame,
    cfg: Config,
    every: str,
    period: str | None = None,
    sources: list[str] | None = None,
    aggregators: list[str] | None = None,
    feature_names: list[str] | None = None,
    feature_registry: FeatureRegistry | None = None,
    aggregator_registry: AggregatorRegistry | None = None,
) -> pl.LazyFrame:
    """Group each event into fixed time windows and aggregate within each.

    Internally calls :func:`to_per_sample` to materialise features, then
    Polars' ``group_by_dynamic`` over the timestamp column. When ``event_id``
    is present, windows are computed *per event* so different events do not
    bleed into each other.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input frame, typically one event from :func:`ml_analysis.dataset.builder.build`.
    cfg : Config
        Project configuration. ``cfg.timestamp_col`` is used as the
        time axis.
    every : str
        Step between window starts, in Polars duration syntax (``"1m"``,
        ``"1h"``, ``"500ms"``...).
    period : str, optional
        Window length. Defaults to ``every``, producing non-overlapping
        tumbling windows. Set ``period > every`` for sliding/overlapping
        windows.
    sources : list of str, optional
        Columns to aggregate. Defaults to all numeric columns that aren't
        the timestamp or a label column, after feature materialisation.
    aggregators : list of str, optional
        Names of aggregators to apply (looked up in ``aggregator_registry``).
        Defaults to ``["mean", "std"]``.
    feature_names : list of str, optional
        Subset of features to materialise before aggregating. ``None``
        applies all registered features; ``[]`` skips them.
    feature_registry : FeatureRegistry, optional
        Source of feature specs. Defaults to the process-wide registry.
    aggregator_registry : AggregatorRegistry, optional
        Source of aggregator specs. Defaults to the process-wide registry.

    Returns
    -------
    pl.LazyFrame
        One row per ``(event_id, window)``. Numeric output columns are
        named ``"<source>__<aggregator>"``; label columns are carried
        through unchanged via ``.first()``.
    """
    fr = feature_registry or default_features()
    ar = aggregator_registry or default_aggs()

    base = to_per_sample(lf, cfg, feature_names, fr)

    schema = base.collect_schema()
    label_cols = _label_cols(base, cfg)
    if sources is None:
        sources = [
            c for c in schema.names()
            if c != cfg.timestamp_col
            and c not in label_cols
            and schema[c].is_numeric()
        ]
    aggregators = aggregators or ["mean", "std"]

    agg_exprs: list[pl.Expr] = []
    for src in sources:
        for agg_name in aggregators:
            agg_exprs.append(ar.get(agg_name).apply(src))

    group_by = [c for c in ("event_id",) if c in schema.names()] or None
    grouped = set(group_by or ())

    # carry remaining label columns through (constant within an event); use .first()
    label_exprs = [pl.col(c).first().alias(c) for c in label_cols if c not in grouped]

    return (
        base.sort(cfg.timestamp_col)
        .group_by_dynamic(
            cfg.timestamp_col,
            every=every,
            period=period or every,
            group_by=group_by,
        )
        .agg(label_exprs + agg_exprs)
    )


# ── Period (one row per event) ───────────────────────────────────────────────

def to_period(
    lfs: dict[str, pl.LazyFrame] | Iterable[pl.LazyFrame] | pl.LazyFrame,
    cfg: Config,
    sources: list[str] | None = None,
    aggregators: list[str] | None = None,
    feature_names: list[str] | None = None,
    feature_registry: FeatureRegistry | None = None,
    aggregator_registry: AggregatorRegistry | None = None,
) -> pl.DataFrame:
    """Collapse each event to a single row of summary statistics.

    Equivalent to :func:`to_windowed` with ``period`` equal to the full
    event duration and no time grouping — i.e. one row per input event.
    Unlike :func:`to_windowed`, this function eagerly collects.

    Parameters
    ----------
    lfs : dict, iterable, or pl.LazyFrame
        Either a single event LazyFrame, an iterable of them, or the
        ``{event_id: LazyFrame}`` dict returned by
        :func:`ml_analysis.dataset.builder.build`.
    cfg : Config
        Project configuration; ``cfg.timestamp_col`` is excluded from
        automatic source selection.
    sources : list of str, optional
        Columns to aggregate. Defaults to all numeric columns that aren't
        the timestamp or a label column, after feature materialisation.
    aggregators : list of str, optional
        Names of aggregators to apply. Defaults to
        ``["mean", "std", "min", "max"]``.
    feature_names : list of str, optional
        Subset of features to materialise before aggregating. ``None``
        applies all registered features; ``[]`` skips them.
    feature_registry : FeatureRegistry, optional
        Source of feature specs. Defaults to the process-wide registry.
    aggregator_registry : AggregatorRegistry, optional
        Source of aggregator specs. Defaults to the process-wide registry.

    Returns
    -------
    pl.DataFrame
        One row per input event, with columns
        ``"<source>__<aggregator>"`` plus any label columns.
        Returns an empty :class:`pl.DataFrame` if ``lfs`` is empty.
    """
    fr = feature_registry or default_features()
    ar = aggregator_registry or default_aggs()

    if isinstance(lfs, pl.LazyFrame):
        items = [lfs]
    elif isinstance(lfs, dict):
        items = list(lfs.values())
    else:
        items = list(lfs)

    aggregators = aggregators or ["mean", "std", "min", "max"]
    rows: list[pl.DataFrame] = []
    for lf in items:
        base = to_per_sample(lf, cfg, feature_names, fr)
        schema = base.collect_schema()
        label_cols = _label_cols(base, cfg)
        srcs = sources
        if srcs is None:
            srcs = [
                c for c in schema.names()
                if c != cfg.timestamp_col
                and c not in label_cols
                and schema[c].is_numeric()
            ]
        agg_exprs = [ar.get(a).apply(s) for s in srcs for a in aggregators]
        label_exprs = [pl.col(c).first().alias(c) for c in label_cols]
        rows.append(base.select(label_exprs + agg_exprs).collect())

    return pl.concat(rows, how="vertical_relaxed") if rows else pl.DataFrame()

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

from tessa.config import Config
from tessa.features.aggregates import (
    AggregatorRegistry,
    default_registry as default_aggs,
)
from tessa.features.registry import (
    FeatureRegistry,
    default_registry as default_features,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _label_cols_from_schema(
    schema: pl.Schema, cfg: Config, declared: Iterable[str] = ()
) -> list[str]:
    """Label/id columns given an already-resolved schema (see `_label_cols`)."""
    names = schema.names()
    candidates = list(dict.fromkeys(["event_id", cfg.asset_col, cfg.class_col, *declared]))
    known = [c for c in candidates if c in names and c != cfg.timestamp_col]
    return known + [
        c for c in names if c not in known and c != cfg.timestamp_col and not schema[c].is_numeric()
    ]


def _label_cols(lf: pl.LazyFrame, cfg: Config, declared: Iterable[str] = ()) -> list[str]:
    """Return the columns that came from label metadata + ids.

    A column is a label column when it is one of the known label/id columns
    (``event_id``, ``cfg.asset_col``, ``cfg.class_col``), when it is
    *declared* as one (``declared``, normally the ``label_cols`` recorded by
    :func:`tessa.dataset.builder.build`), or, as a fallback for frames built
    by hand, when it is non-numeric and not the timestamp. Declared columns
    are labels whatever their dtype, so a numeric label column is never
    aggregated as a signal. Used by the materialisers to know what to carry
    through aggregation and what to exclude from automatic source selection.
    """
    return _label_cols_from_schema(lf.collect_schema(), cfg, declared)


def _declared_label_cols(lfs: object, label_cols: Iterable[str] | None) -> tuple[str, ...]:
    """Union of ``label_cols`` and the ``label_cols`` recorded on ``lfs``.

    ``lfs`` carries them when it is the :class:`~tessa.dataset.EventFrames`
    returned by :func:`tessa.dataset.builder.build`; a plain dict, list or
    LazyFrame does not.
    """
    recorded = getattr(lfs, "label_cols", None) or ()
    return tuple(dict.fromkeys([*recorded, *(label_cols or ())]))


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
    feature_registry: FeatureRegistry | None = None,
) -> pl.LazyFrame:
    """Add registered features as columns, leaving row count unchanged.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input frame, typically one event from :func:`tessa.dataset.builder.build`.
    cfg : Config
        Project configuration. Currently unused inside this function but
        kept for signature symmetry with the other materialisers.
    feature_names : list of str, optional
        Subset of features to apply. If ``None`` (default), every feature
        in ``feature_registry`` is applied. Pass ``[]`` to skip features entirely.
    feature_registry : FeatureRegistry, optional
        Registry to resolve features from. Defaults to the process-wide
        registry.

    Returns
    -------
    pl.LazyFrame
        The input frame with one new column per applied feature, in
        dependency order.
    """
    reg = feature_registry or default_features()
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
    label_cols: Iterable[str] | None = None,
) -> pl.LazyFrame:
    """Group each event into fixed time windows and aggregate within each.

    Internally calls :func:`to_per_sample` to materialise features, then
    Polars' ``group_by_dynamic`` over the timestamp column. When ``event_id``
    is present, windows are computed *per event* so different events do not
    bleed into each other.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input frame, typically one event from :func:`tessa.dataset.builder.build`.
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
    label_cols : iterable of str, optional
        Columns that are label metadata, whatever their dtype: they are
        carried through via ``.first()`` and never aggregated. Added to
        ``event_id``, ``cfg.asset_col``, ``cfg.class_col``, any non-numeric
        column, and the ``label_cols`` recorded on an
        :class:`~tessa.dataset.EventFrames` input. A single event frame
        carries no such record, so pass ``label_cols=events.label_cols``
        when calling this per event.

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
    label_cols = _label_cols_from_schema(schema, cfg, _declared_label_cols(lf, label_cols))
    if sources is None:
        sources = [
            c
            for c in schema.names()
            if c != cfg.timestamp_col and c not in label_cols and schema[c].is_numeric()
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
    label_cols: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Collapse each event to a single row of summary statistics.

    Equivalent to :func:`to_windowed` with ``period`` equal to the full
    event duration and no time grouping — i.e. one row per input event.
    Unlike :func:`to_windowed`, this function eagerly collects.

    All per-event aggregations are built lazily (feature expressions are
    applied inside each event, so rolling/diff features never bleed across
    events), the schema is resolved **once** for the whole batch, and the
    events are collected together via ``pl.collect_all`` — Polars runs
    them in parallel on its thread pool instead of the old sequential
    per-event ``collect()`` loop, which was single-core-bound.
    (A single fused ``concat -> group_by`` query was benchmarked too: its
    plan-optimization cost grows superlinearly with the number of events
    and loses badly past ~1k events, so ``collect_all`` is the lever used.)

    Parameters
    ----------
    lfs : dict, iterable, or pl.LazyFrame
        Either a single event LazyFrame, an iterable of them, or the
        ``{event_id: LazyFrame}`` mapping returned by
        :func:`tessa.dataset.builder.build` (an
        :class:`~tessa.dataset.EventFrames`, whose recorded label columns
        are honoured).
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
    label_cols : iterable of str, optional
        Columns that are label metadata, whatever their dtype: they are
        carried through via ``.first()`` and never aggregated. Added to
        ``event_id``, ``cfg.asset_col``, ``cfg.class_col``, any non-numeric
        column, and the ``label_cols`` recorded on an
        :class:`~tessa.dataset.EventFrames` input. Needed only for
        frames built by hand or a subset taken with a plain dict
        comprehension.

    Returns
    -------
    pl.DataFrame
        One row per input event (input order preserved), with columns
        ``"<source>__<aggregator>"`` plus the label columns, each holding
        its first value within the event.
        Returns an empty :class:`pl.DataFrame` if ``lfs`` is empty.
    """
    fr = feature_registry or default_features()
    ar = aggregator_registry or default_aggs()
    declared = _declared_label_cols(lfs, label_cols)

    if isinstance(lfs, pl.LazyFrame):
        items = [lfs]
    elif isinstance(lfs, dict):
        items = list(lfs.values())
    else:
        items = list(lfs)
    if not items:
        return pl.DataFrame()

    aggregators = aggregators or ["mean", "std", "min", "max"]

    bases = [to_per_sample(lf, cfg, feature_names, fr) for lf in items]
    # Events share one schema (same files, same features): resolve once.
    schema = bases[0].collect_schema()
    label_names = _label_cols_from_schema(schema, cfg, declared)
    srcs = sources
    if srcs is None:
        srcs = [
            c
            for c in schema.names()
            if c != cfg.timestamp_col and c not in label_names and schema[c].is_numeric()
        ]
    agg_exprs = [ar.get(a).apply(s) for s in srcs for a in aggregators]
    label_exprs = [pl.col(c).first().alias(c) for c in label_names]

    lazy_rows = [base.select(label_exprs + agg_exprs) for base in bases]
    return pl.concat(pl.collect_all(lazy_rows), how="vertical_relaxed")

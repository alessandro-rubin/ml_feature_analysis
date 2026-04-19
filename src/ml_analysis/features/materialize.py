"""Materialize a per-event LazyFrame into per-sample / windowed / period outputs."""

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
    """Columns that came from label metadata + ids (everything non-temporal/non-numeric typical)."""
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
    """Add registered features as columns. Lazy."""
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
    """Group by fixed time windows; emit one row per window per event.

    Parameters
    ----------
    every : str       step between window starts (e.g. "1m", "1h").
    period : str      window length; defaults to `every` (non-overlapping).
    sources : list    columns to aggregate. Defaults to all numeric non-label cols
                      after feature materialization.
    aggregators : list   names of aggregators to apply (default: ["mean", "std"]).
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
    """One row per event: the aggregate of every source over the full event.

    Accepts a single LazyFrame (one event), an iterable of them, or the dict
    returned by `dataset.build`.
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

"""Declarative window specs + a single `materialize` entry point.

A `WindowSpec` answers "what is one row of the analysis table":

- ``WindowSpec.event()``            -> one row per event (`to_period`)
- ``WindowSpec.tumbling("1h")``     -> fixed non-overlapping windows
- ``WindowSpec.sliding("1h", "6h")``-> overlapping windows (step, length)

`materialize(lfs, spec, cfg, ...)` dispatches to the right materializer
and always returns an eager DataFrame — the boundary where laziness ends.
Events are collected in memory-bounded batches (see
:mod:`tessa.features.batching`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import polars as pl

from tessa.config import Config
from tessa.features.aggregates import AggregatorRegistry
from tessa.features.batching import collect_batched, memory_budget_bytes
from tessa.features.materialize import (
    _declared_label_cols,
    _event_sizes,
    to_per_sample,
    to_period,
    to_windowed,
)
from tessa.features.registry import FeatureRegistry


@dataclass(frozen=True)
class WindowSpec:
    kind: Literal["event", "tumbling", "sliding"]
    every: str | None = None
    period: str | None = None

    @classmethod
    def event(cls) -> "WindowSpec":
        """One row per event — post-hoc fault characterization."""
        return cls(kind="event")

    @classmethod
    def tumbling(cls, every: str) -> "WindowSpec":
        """Non-overlapping windows of length ``every`` — monitoring grid."""
        return cls(kind="tumbling", every=every)

    @classmethod
    def sliding(cls, every: str, period: str) -> "WindowSpec":
        """Windows of length ``period`` starting every ``every``."""
        return cls(kind="sliding", every=every, period=period)


def materialize(
    lfs: dict[str, pl.LazyFrame] | Iterable[pl.LazyFrame] | pl.LazyFrame,
    spec: WindowSpec,
    cfg: Config,
    sources: list[str] | None = None,
    aggregators: list[str] | None = None,
    feature_names: list[str] | None = None,
    feature_registry: FeatureRegistry | None = None,
    aggregator_registry: AggregatorRegistry | None = None,
    label_cols: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Materialize event frames into the analysis table described by ``spec``.

    ``label_cols`` (plus any recorded on an :class:`~tessa.dataset.EventFrames`
    input) are carried through as label metadata and never aggregated; see
    :func:`~tessa.features.materialize.to_period`.
    """
    label_cols = _declared_label_cols(lfs, label_cols)
    if spec.kind == "event":
        return to_period(
            lfs,
            cfg,
            sources=sources,
            aggregators=aggregators,
            feature_names=feature_names,
            feature_registry=feature_registry,
            aggregator_registry=aggregator_registry,
            label_cols=label_cols,
        )

    if spec.every is None:
        raise ValueError(f"WindowSpec({spec.kind}) requires `every`.")
    if isinstance(lfs, pl.LazyFrame):
        items = [lfs]
    elif isinstance(lfs, dict):
        items = list(lfs.values())
    else:
        items = list(lfs)
    if not items:
        return pl.DataFrame()

    sizes = None
    if memory_budget_bytes(cfg.memory_budget) is not None:
        if sources is not None:
            n_columns = len(sources) + 1
        else:  # numeric columns of the per-sample frame
            schema = to_per_sample(items[0], cfg, feature_names, feature_registry).collect_schema()
            n_columns = sum(1 for dtype in schema.dtypes() if dtype.is_numeric()) + 1
        sizes = _event_sizes(lfs, cfg, n_columns)
    lazy_parts = [
        to_windowed(
            lf,
            cfg,
            every=spec.every,
            period=spec.period or spec.every,
            sources=sources,
            aggregators=aggregators,
            feature_names=feature_names,
            feature_registry=feature_registry,
            aggregator_registry=aggregator_registry,
            label_cols=label_cols,
        )
        for lf in items
    ]
    parts = collect_batched(
        lazy_parts,
        cfg,
        sizes=sizes,
        names=list(lfs) if isinstance(lfs, dict) else None,
        stacklevel=2,
    )
    return pl.concat(parts, how="vertical_relaxed")

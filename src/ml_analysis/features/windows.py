"""Declarative window specs + a single `materialize` entry point.

A `WindowSpec` answers "what is one row of the analysis table":

- ``WindowSpec.event()``            -> one row per event (`to_period`)
- ``WindowSpec.tumbling("1h")``     -> fixed non-overlapping windows
- ``WindowSpec.sliding("1h", "6h")``-> overlapping windows (step, length)

`materialize(lfs, spec, cfg, ...)` dispatches to the right materializer
and always returns an eager DataFrame — the boundary where laziness ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import polars as pl

from ml_analysis.config import Config
from ml_analysis.features.aggregates import AggregatorRegistry
from ml_analysis.features.materialize import to_period, to_windowed
from ml_analysis.features.registry import FeatureRegistry


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
) -> pl.DataFrame:
    """Materialize event frames into the analysis table described by ``spec``."""
    if spec.kind == "event":
        return to_period(
            lfs, cfg,
            sources=sources, aggregators=aggregators,
            feature_names=feature_names,
            feature_registry=feature_registry,
            aggregator_registry=aggregator_registry,
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

    lazy_parts = [
        to_windowed(
            lf, cfg,
            every=spec.every,
            period=spec.period or spec.every,
            sources=sources, aggregators=aggregators,
            feature_names=feature_names,
            feature_registry=feature_registry,
            aggregator_registry=aggregator_registry,
        )
        for lf in items
    ]
    return pl.concat(pl.collect_all(lazy_parts), how="vertical_relaxed")

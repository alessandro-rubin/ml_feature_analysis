"""Aggregator registry. Used by windowed and period materializers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import polars as pl

AggFactory = Callable[[str], pl.Expr]  # source column name -> expression


@dataclass(frozen=True)
class AggSpec:
    name: str           # output column suffix, e.g. "mean"
    factory: AggFactory  # given source col, produce the aggregation expr

    def apply(self, source: str) -> pl.Expr:
        return self.factory(source).alias(f"{source}__{self.name}")


class AggregatorRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, AggSpec] = {}

    def register(self, spec: AggSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Aggregator already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> AggSpec:
        return self._specs[name]

    def names(self) -> list[str]:
        return list(self._specs)


_default_registry = AggregatorRegistry()


def aggregate(
    name: str, registry: AggregatorRegistry | None = None
) -> Callable[[AggFactory], AggFactory]:
    """Decorator: register an aggregator factory by name."""
    reg = registry or _default_registry

    def wrap(fn: AggFactory) -> AggFactory:
        reg.register(AggSpec(name=name, factory=fn))
        return fn

    return wrap


def default_registry() -> AggregatorRegistry:
    return _default_registry

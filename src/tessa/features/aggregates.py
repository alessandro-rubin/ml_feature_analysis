"""Aggregator registry.

Aggregators turn a numeric column into a single Polars expression — e.g.
``pl.col("temperature").mean()``. They are the building block consumed by
:func:`tessa.features.materialize.to_windowed` and
:func:`tessa.features.materialize.to_period` to produce one column per
``(source, aggregator)`` pair.

Each aggregator is registered under a short name (``"mean"``, ``"p95"``,
``"iqr"``) via the :func:`aggregate` decorator. Stock aggregators are
defined in :mod:`tessa.features.builtins` and populate the default
registry on import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import polars as pl

AggFactory = Callable[[str], pl.Expr]  # source column name -> expression


@dataclass(frozen=True)
class AggSpec:
    """A registered aggregator.

    Parameters
    ----------
    name : str
        Suffix used to label the resulting column (e.g. ``"mean"`` produces
        columns named ``"<source>__mean"``).
    factory : Callable[[str], pl.Expr]
        Function that, given a source column name, returns the Polars
        aggregation expression.
    """

    name: str           # output column suffix, e.g. "mean"
    factory: AggFactory  # given source col, produce the aggregation expr

    def apply(self, source: str) -> pl.Expr:
        """Build the aggregation expression for ``source``.

        Returns
        -------
        pl.Expr
            Expression aliased to ``"<source>__<name>"``.
        """
        return self.factory(source).alias(f"{source}__{self.name}")


class AggregatorRegistry:
    """Registry of :class:`AggSpec` objects, keyed by name."""

    def __init__(self) -> None:
        self._specs: dict[str, AggSpec] = {}

    def register(self, spec: AggSpec) -> None:
        """Add an aggregator to the registry.

        Raises
        ------
        ValueError
            If an aggregator with the same name is already registered.
        """
        if spec.name in self._specs:
            raise ValueError(f"Aggregator already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> AggSpec:
        """Return the aggregator registered under ``name``.

        Raises
        ------
        KeyError
            If no aggregator with that name is registered.
        """
        return self._specs[name]

    def names(self) -> list[str]:
        """Return the names of all registered aggregators in insertion order."""
        return list(self._specs)


_default_registry = AggregatorRegistry()


def aggregate(
    name: str, registry: AggregatorRegistry | None = None
) -> Callable[[AggFactory], AggFactory]:
    """Register an aggregator factory by name.

    Parameters
    ----------
    name : str
        Suffix used in the output column name (``"<source>__<name>"``).
    registry : AggregatorRegistry, optional
        Target registry. Defaults to the process-wide registry returned
        by :func:`default_registry`.

    Returns
    -------
    Callable
        Decorator that registers the wrapped factory and returns it
        unchanged.

    Examples
    --------
    >>> @aggregate("p95")
    ... def _(c: str) -> pl.Expr:
    ...     return pl.col(c).quantile(0.95)
    """
    reg = registry or _default_registry

    def wrap(fn: AggFactory) -> AggFactory:
        reg.register(AggSpec(name=name, factory=fn))
        return fn

    return wrap


def default_registry() -> AggregatorRegistry:
    """Return the process-wide default aggregator registry."""
    return _default_registry

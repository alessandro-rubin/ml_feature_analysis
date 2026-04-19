"""Per-sample feature registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import polars as pl

ExprFactory = Callable[[], pl.Expr]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    deps: tuple[str, ...]
    factory: ExprFactory

    def expr(self) -> pl.Expr:
        return self.factory().alias(self.name)


class FeatureRegistry:
    """Module-level registry. Use the @feature decorator to populate."""

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Feature already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def names(self) -> list[str]:
        return list(self._specs)

    def resolve(self, names: list[str] | None = None) -> list[FeatureSpec]:
        """Topologically sort the requested specs (default: all). Detects cycles."""
        wanted = set(names) if names else set(self._specs)
        ordered: list[FeatureSpec] = []
        seen: set[str] = set()
        in_progress: set[str] = set()

        def visit(n: str) -> None:
            if n in seen:
                return
            if n in in_progress:
                raise ValueError(f"Cyclic feature dependency at: {n}")
            if n not in self._specs:
                # External (raw) column dependency, fine.
                return
            in_progress.add(n)
            for d in self._specs[n].deps:
                visit(d)
            in_progress.discard(n)
            seen.add(n)
            ordered.append(self._specs[n])

        for n in wanted:
            visit(n)
        return ordered


_default_registry = FeatureRegistry()


def feature(
    name: str,
    deps: tuple[str, ...] = (),
    registry: FeatureRegistry | None = None,
) -> Callable[[ExprFactory], ExprFactory]:
    """Decorator: register a polars expression factory as a named feature."""
    reg = registry or _default_registry

    def wrap(fn: ExprFactory) -> ExprFactory:
        reg.register(FeatureSpec(name=name, deps=tuple(deps), factory=fn))
        return fn

    return wrap


def default_registry() -> FeatureRegistry:
    return _default_registry

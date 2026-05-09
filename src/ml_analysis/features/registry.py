"""Per-sample feature registry.

Features are named Polars expressions that produce one output column per sample.
Each one is registered with the :func:`feature` decorator (or directly via
:meth:`FeatureRegistry.register`) and can declare dependencies on raw input
columns or on other features. The registry topologically sorts those
dependencies so that downstream materialisers can apply them in a valid order.

The module exposes a single process-wide ``_default_registry`` accessed through
:func:`default_registry`. Stock features in :mod:`ml_analysis.features.builtins`
populate it on import; user code typically does the same via the ``make_*``
helpers in that module or by writing its own ``@feature`` decorators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import polars as pl

ExprFactory = Callable[[], pl.Expr]


@dataclass(frozen=True)
class FeatureSpec:
    """A registered per-sample feature.

    Parameters
    ----------
    name : str
        Output column name produced by this feature.
    deps : tuple of str
        Names of columns or other features this expression reads. Used by
        :meth:`FeatureRegistry.resolve` to compute a topological order.
        Names that don't refer to a registered feature are treated as raw
        input columns and ignored by the sort.
    factory : Callable[[], pl.Expr]
        Zero-argument callable returning the Polars expression that computes
        the feature. The expression is built lazily on each call to
        :meth:`expr`, which means it is safe to use across multiple frames.
    """

    name: str
    deps: tuple[str, ...]
    factory: ExprFactory

    def expr(self) -> pl.Expr:
        """Build the feature expression, aliased to :attr:`name`.

        Returns
        -------
        pl.Expr
            Polars expression suitable for ``LazyFrame.with_columns``.
        """
        return self.factory().alias(self.name)


class FeatureRegistry:
    """Module-level registry of :class:`FeatureSpec` objects.

    Use the :func:`feature` decorator (or :meth:`register`) to populate it.
    Materialisers call :meth:`resolve` to obtain specs in dependency order.
    """

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        """Add a spec to the registry.

        Parameters
        ----------
        spec : FeatureSpec
            Feature to register. Its ``name`` must be unique within the
            registry.

        Raises
        ------
        ValueError
            If a feature with the same name is already registered.
        """
        if spec.name in self._specs:
            raise ValueError(f"Feature already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        """Return the spec registered under ``name``.

        Raises
        ------
        KeyError
            If no feature with that name is registered.
        """
        return self._specs[name]

    def names(self) -> list[str]:
        """Return the names of all registered features in insertion order."""
        return list(self._specs)

    def resolve(self, names: list[str] | None = None) -> list[FeatureSpec]:
        """Topologically sort the requested specs.

        Parameters
        ----------
        names : list of str, optional
            Subset of features to resolve. If ``None``, resolves every
            registered feature. Names that are not registered are treated
            as raw input columns and silently skipped — this lets features
            declare ``deps=("temperature",)`` without having to register
            the raw columns themselves.

        Returns
        -------
        list of FeatureSpec
            Specs ordered such that every spec appears after each of its
            registered dependencies.

        Raises
        ------
        ValueError
            If a cycle is detected in the dependency graph.
        """
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
    """Register a Polars expression factory as a named feature.

    Parameters
    ----------
    name : str
        Output column name.
    deps : tuple of str, optional
        Columns or other features this expression reads. Used to order
        materialisation; see :meth:`FeatureRegistry.resolve`.
    registry : FeatureRegistry, optional
        Target registry. Defaults to the process-wide registry returned
        by :func:`default_registry`.

    Returns
    -------
    Callable
        Decorator that registers the wrapped factory and returns it
        unchanged, so the function can still be called directly.

    Examples
    --------
    >>> @feature("temperature__diff1", deps=("temperature",))
    ... def _():
    ...     return pl.col("temperature").diff()
    """
    reg = registry or _default_registry

    def wrap(fn: ExprFactory) -> ExprFactory:
        reg.register(FeatureSpec(name=name, deps=tuple(deps), factory=fn))
        return fn

    return wrap


def default_registry() -> FeatureRegistry:
    """Return the process-wide default feature registry."""
    return _default_registry

"""Sandboxed compilation of model-authored Polars expressions.

The AI layer lets a model invent features at runtime by writing a Polars
expression as a string::

    pl.col("temperature").rolling_std(50) / pl.col("pressure").abs()

This module turns such a string into a :data:`~tessa.features.registry.ExprFactory`
that the ordinary feature registry can hold, after checking it against an
allowlist of AST node types, identifiers, and method names.

Threat model
------------
**The AST allowlist is the security boundary.** Passing ``{"__builtins__": {}}``
to :func:`eval` is defence in depth only — on its own it is trivially bypassed
via attribute traversal (``().__class__.__bases__[0].__subclasses__()``) or
subscripting. Both are blocked here by rejecting :class:`ast.Subscript` outright
and by allowlisting every attribute name, which is why relaxing either check
reopens the classic escapes.

What this does *not* defend against: expressions that are merely ruinous rather
than malicious. ``pl.col("x").rolling_quantile(0.5, window_size=10_000_000)``
is perfectly legal here. Cost is bounded by previewing on a single event and by
the column budget in :mod:`tessa.ai.tools`, not by this compiler.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import polars as pl

from tessa.features.aggregates import AggFactory
from tessa.features.registry import ExprFactory

__all__ = [
    "ExpressionError",
    "CompiledExpr",
    "compile_feature_expression",
    "compile_aggregator_expression",
    "allowlist_reference",
]

MAX_SOURCE_CHARS = 2000
MAX_AST_NODES = 400
MAX_CALL_ARGS = 8

# Entry points reachable as ``pl.<name>``.
_PL_TOP = frozenset(
    {
        "col", "lit", "when", "len", "first", "last", "int_range",
        "min_horizontal", "max_horizontal", "sum_horizontal", "mean_horizontal",
        "corr", "cov",
    }
)  # fmt: skip

# Dtypes permitted as ``cast`` targets.
_DTYPES = frozenset({"Float64", "Float32", "Int64", "Int32", "UInt32", "Boolean"})

# Methods callable on an expression.
_EXPR_METHODS = frozenset(
    # arithmetic / elementwise maths
    {
        "abs", "sign", "exp", "log", "log1p", "log10", "sqrt", "cbrt",
        "sin", "cos", "tan", "arctan", "floor", "ceil", "round", "clip", "pow",
    }
    # reductions
    | {
        "mean", "median", "std", "var", "min", "max", "sum", "product", "count",
        "n_unique", "null_count", "quantile", "mode", "skew", "kurtosis",
        "entropy", "arg_min", "arg_max",
    }
    # rolling / exponentially weighted
    | {
        "rolling_mean", "rolling_std", "rolling_var", "rolling_min", "rolling_max",
        "rolling_sum", "rolling_median", "rolling_quantile", "rolling_skew",
        "ewm_mean", "ewm_std", "ewm_var",
    }
    # sequence-shaped
    | {
        "diff", "shift", "pct_change", "cum_sum", "cum_max", "cum_min", "cum_prod",
        "cum_count", "fill_null", "fill_nan", "forward_fill", "backward_fill",
        "interpolate", "rank", "reverse", "sort",
    }
    # predicates / null handling
    | {
        "is_null", "is_not_null", "is_nan", "is_finite", "is_infinite", "is_in",
        "is_between", "not_", "any", "all", "eq", "ne", "lt", "le", "gt", "ge",
        "and_", "or_",
    }
    # when/then chains, windowing, casting
    | {"then", "otherwise", "over", "cast"}
)  # fmt: skip

_NS_METHODS: dict[str, frozenset[str]] = {
    "dt": frozenset(
        {"hour", "minute", "second", "weekday", "ordinal_day", "epoch", "total_seconds"}
    )
}
_NAMESPACES = frozenset(_NS_METHODS)
_NON_PL_ATTRS = _EXPR_METHODS | _NAMESPACES | frozenset().union(*_NS_METHODS.values())

# Node types the grammar accepts. Everything else is rejected by type.
_ALLOWED_NODES: frozenset[type[ast.AST]] = frozenset(
    {
        ast.Expression, ast.Call, ast.Attribute, ast.Name, ast.Load, ast.Constant,
        ast.keyword, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.List, ast.Tuple, ast.IfExp,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Not, ast.Invert,
        ast.And, ast.Or,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.BitAnd, ast.BitOr, ast.BitXor,
    }
)  # fmt: skip

# Named purely so a test can assert they stay rejected. Relaxing any of these
# reopens a documented sandbox escape; see the module docstring.
_MUST_REJECT_NODES: frozenset[type[ast.AST]] = frozenset(
    {
        ast.Subscript, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
        ast.GeneratorExp, ast.Starred, ast.JoinedStr, ast.FormattedValue,
        ast.Dict, ast.Set, ast.Slice, ast.NamedExpr, ast.Await, ast.Yield,
        ast.YieldFrom, ast.Store, ast.Del,
    }
)  # fmt: skip

# Rejected by name as well as by construction, so the intent is auditable.
_BANNED_METHODS = frozenset(
    {"map_elements", "map_batches", "rolling_map", "apply", "alias", "head", "tail", "slice"}
)


class ExpressionError(ValueError):
    """An expression was rejected by the allowlist, or failed to build."""


@dataclass(frozen=True)
class CompiledExpr:
    """A validated expression, ready to register as a feature."""

    source: str
    deps: tuple[str, ...]
    factory: ExprFactory


def _fail(msg: str, hint: str = "") -> ExpressionError:
    return ExpressionError(f"{msg}{(' ' + hint) if hint else ''}")


def _validate(tree: ast.Expression, allowed_names: frozenset[str]) -> None:
    """Walk the tree, rejecting anything outside the allowlist."""
    n_nodes = 0
    for node in ast.walk(tree):
        n_nodes += 1
        if n_nodes > MAX_AST_NODES:
            raise _fail(f"Expression is too complex (over {MAX_AST_NODES} AST nodes).")

        kind = type(node)
        if kind not in _ALLOWED_NODES:
            raise _fail(
                f"{kind.__name__} is not allowed in a feature expression.",
                "Only Polars expression syntax is accepted — no comprehensions, "
                "lambdas, subscripts, f-strings or literal containers.",
            )

        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise _fail(
                    f"Unknown name {node.id!r}.",
                    f"Only {sorted(allowed_names)} may be referenced.",
                )

        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_"):
                raise _fail(f"Attribute {attr!r} is not allowed.")
            if attr in _BANNED_METHODS:
                hint = (
                    "the feature name comes from the `name` argument"
                    if attr == "alias"
                    else "it is excluded from the expression allowlist"
                )
                raise _fail(f"{attr}() is not allowed:", hint + ".")
            on_pl = isinstance(node.value, ast.Name) and node.value.id == "pl"
            permitted = (_PL_TOP | _DTYPES) if on_pl else _NON_PL_ATTRS
            if attr not in permitted:
                where = "pl" if on_pl else "an expression"
                raise _fail(
                    f"{attr!r} is not an allowed attribute on {where}.",
                    "Call list_capabilities() to see the permitted names.",
                )

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute):
                raise _fail(
                    "Only attribute calls are allowed (e.g. pl.col(...), .rolling_mean(...)).",
                )
            if len(node.args) > MAX_CALL_ARGS:
                raise _fail(f"Too many arguments in one call (max {MAX_CALL_ARGS}).")
            if any(kw.arg is None for kw in node.keywords):
                raise _fail("`**kwargs` unpacking is not allowed.")

        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (str, int, float, bool, type(None))):
                raise _fail(f"Constants of type {type(node.value).__name__} are not allowed.")


def _extract_deps(tree: ast.Expression) -> tuple[str, ...]:
    """Collect the column names referenced via ``pl.col("...")``.

    Non-literal ``pl.col(x)`` is unreachable — ``x`` would be a rejected
    :class:`ast.Name` — so every reference is a string constant by construction.
    """
    deps: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        func = node.func
        if func.attr != "col":
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "pl"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value not in deps:
                    deps.append(arg.value)
    return tuple(deps)


def _parse(src: str, allowed_names: frozenset[str]) -> ast.Expression:
    if not src or not src.strip():
        raise _fail("Expression is empty.")
    if len(src) > MAX_SOURCE_CHARS:
        raise _fail(f"Expression is too long ({len(src)} chars, max {MAX_SOURCE_CHARS}).")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:  # includes statements, which `mode="eval"` forbids
        raise _fail(f"Could not parse expression: {exc.msg}.") from exc
    except (ValueError, MemoryError, RecursionError) as exc:
        raise _fail(f"Could not parse expression: {exc}.") from exc
    _validate(tree, allowed_names)
    return tree


def compile_feature_expression(src: str, *, extra_names: tuple[str, ...] = ()) -> CompiledExpr:
    """Validate and compile a per-sample feature expression.

    Parameters
    ----------
    src : str
        Polars expression source, e.g. ``'pl.col("x").diff().abs()'``.
    extra_names : tuple of str, optional
        Additional bare identifiers to permit (used by
        :func:`compile_aggregator_expression` to bind its column argument).

    Returns
    -------
    CompiledExpr
        With ``deps`` derived from the ``pl.col`` references, and a zero-argument
        ``factory`` matching the :class:`~tessa.features.registry.FeatureSpec`
        contract — the expression is rebuilt on each call, so it is safe to reuse
        across frames.

    Raises
    ------
    ExpressionError
        If the expression is rejected by the allowlist, or does not build into a
        :class:`polars.Expr`.
    """
    allowed = frozenset({"pl", *extra_names})
    tree = _parse(src, allowed)
    deps = _extract_deps(tree)
    code = compile(tree, "<tessa-feature>", "eval")

    def factory() -> pl.Expr:
        return eval(code, {"__builtins__": {}, "pl": pl})  # noqa: S307 — gated above

    # Build once now, so a bad call signature surfaces here rather than during a
    # materialize run minutes later.
    try:
        probe = factory()
    except ExpressionError:
        raise
    except Exception as exc:
        raise _fail(f"Expression failed to build: {type(exc).__name__}: {exc}") from exc
    if not isinstance(probe, pl.Expr):
        raise _fail(
            f"Expression produced {type(probe).__name__}, not a Polars expression.",
            "It must evaluate to something like pl.col(...)...",
        )
    return CompiledExpr(source=src, deps=deps, factory=factory)


def compile_aggregator_expression(src: str, *, arg: str = "c") -> AggFactory:
    """Validate and compile an aggregator expression.

    Aggregators are parameterized by their source column, so the expression
    refers to it through a free name (``c`` by default)::

        pl.col(c).quantile(0.9) - pl.col(c).quantile(0.1)

    Returns
    -------
    AggFactory
        Callable taking the source column name and returning the expression,
        matching the :class:`~tessa.features.aggregates.AggSpec` contract.
    """
    tree = _parse(src, frozenset({"pl", arg}))
    if arg not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}:
        raise _fail(
            f"Aggregator expression never references {arg!r}.",
            f"Use pl.col({arg}) so it applies to whichever column it aggregates.",
        )
    code = compile(tree, "<tessa-aggregator>", "eval")

    def factory(source: str) -> pl.Expr:
        return eval(code, {"__builtins__": {}, "pl": pl, arg: source})  # noqa: S307

    try:
        probe = factory("__probe__")
    except Exception as exc:
        raise _fail(f"Aggregator failed to build: {type(exc).__name__}: {exc}") from exc
    if not isinstance(probe, pl.Expr):
        raise _fail(f"Aggregator produced {type(probe).__name__}, not a Polars expression.")
    return factory


def allowlist_reference() -> dict[str, object]:
    """Return the allowlist, for tool payloads and generated documentation.

    Kept as the single source so docs cannot drift from what the compiler
    actually accepts.
    """
    return {
        "pl_functions": sorted(_PL_TOP),
        "cast_dtypes": sorted(_DTYPES),
        "expression_methods": sorted(_EXPR_METHODS),
        "namespaces": {ns: sorted(m) for ns, m in _NS_METHODS.items()},
        "excluded": sorted(_BANNED_METHODS),
    }

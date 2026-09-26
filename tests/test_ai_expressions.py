"""The expression sandbox is the security boundary of the AI layer.

Every rejection case below is a documented escape route or a footgun; if one of
these starts passing, the allowlist has been widened too far.
"""

from __future__ import annotations

import ast

import polars as pl
import pytest

from tessa.ai.expressions import (
    _ALLOWED_NODES,
    _MUST_REJECT_NODES,
    ExpressionError,
    allowlist_reference,
    compile_aggregator_expression,
    compile_feature_expression,
)

ACCEPTED = [
    'pl.col("t").rolling_std(50)',
    'pl.col("a") / pl.col("b").abs()',
    '(pl.col("a") - pl.col("a").rolling_mean(10)) / pl.col("a").rolling_std(10)',
    'pl.when(pl.col("a") > 0).then(pl.col("a")).otherwise(0)',
    'pl.col("a").diff().abs().cum_sum()',
    'pl.col("a").ewm_mean(half_life=5.0)',
    'pl.col("a").fill_null(0).cast(pl.Float64)',
    'pl.col("a").rolling_quantile(0.9, window_size=20)',
    'pl.max_horizontal(pl.col("a"), pl.col("b"))',
    'pl.col("a").is_between(0, 1)',
]

REJECTED = [
    # sandbox escapes
    ('__import__("os").system("id")', "import"),
    ('open("/etc/passwd").read()', "open"),
    ("(1).__class__.__bases__[0].__subclasses__()", "subclass traversal"),
    ("pl.col.__class__", "dunder attribute"),
    ('pl.col("a").__reduce__()', "dunder method"),
    ("{}.__class__", "dict literal + dunder"),
    ("globals()", "globals"),
    ('getattr(pl, "col")', "getattr"),
    ('eval("1")', "eval"),
    ('f"{__import__}"', "f-string"),
    # forbidden syntax
    ("lambda: 1", "lambda"),
    ("[x for x in ().__class__.__mro__]", "comprehension"),
    ('pl.col("a")[0]', "subscript"),
    ('{"a": 1}', "dict literal"),
    ('pl.col(*["a"])', "starred"),
    # forbidden polars surface
    ('pl.read_parquet("x")', "io"),
    ('pl.scan_parquet("x")', "io"),
    ('pl.col("a").map_elements(pl.col)', "map_elements"),
    ('pl.col("a").alias("b")', "alias"),
    ('pl.col("a").head(3)', "head"),
    ('pl.col("a").nonexistent_method()', "unknown method"),
    ('pl.nonexistent("a")', "unknown pl function"),
    # malformed / abusive
    ("", "empty"),
    ("   ", "blank"),
    ('pl.col("a") +', "syntax error"),
    ('import os; pl.col("a")', "statement"),
    ("42", "not an expression object"),
    ('"just a string"', "not an expression object"),
]


@pytest.mark.parametrize("src", ACCEPTED)
def test_accepts_legitimate_expressions(src):
    compiled = compile_feature_expression(src)
    assert isinstance(compiled.factory(), pl.Expr)


@pytest.mark.parametrize("src,label", REJECTED, ids=[label for _, label in REJECTED])
def test_rejects(src, label):
    with pytest.raises(ExpressionError):
        compile_feature_expression(src)


def test_rejection_happens_before_execution(tmp_path):
    """A rejected expression must never run — check a real side effect."""
    marker = tmp_path / "pwned.txt"
    src = f'open({str(marker)!r}, "w").write("x")'
    with pytest.raises(ExpressionError):
        compile_feature_expression(src)
    assert not marker.exists()


def test_oversized_and_overdeep_inputs_are_bounded():
    with pytest.raises(ExpressionError, match="too long"):
        compile_feature_expression('pl.col("a")' + " + 1" * 5000)
    with pytest.raises(ExpressionError, match="too complex"):
        compile_feature_expression('pl.col("a")' + ".abs()" * 300)


def test_forbidden_node_set_stays_forbidden():
    """Guard against a future relaxation quietly re-admitting an escape."""
    assert _MUST_REJECT_NODES.isdisjoint(_ALLOWED_NODES)
    for node_type in _MUST_REJECT_NODES:
        assert node_type not in _ALLOWED_NODES, node_type


# ── dependency derivation ────────────────────────────────────────────────────


def test_deps_are_derived_in_order_and_deduped():
    c = compile_feature_expression('pl.col("b") + pl.col("a") - pl.col("b").mean()')
    assert c.deps == ("b", "a")


def test_deps_may_name_another_feature():
    """Chained features must topo-sort; unregistered names are raw columns."""
    c = compile_feature_expression('pl.col("temperature__diff1").abs()')
    assert c.deps == ("temperature__diff1",)


def test_expression_with_no_columns_has_no_deps():
    assert compile_feature_expression("pl.lit(1.0)").deps == ()


# ── round trip against real data ─────────────────────────────────────────────


def test_compiled_factory_evaluates_on_a_frame():
    c = compile_feature_expression('pl.col("x").diff().abs()')
    df = pl.DataFrame({"x": [1.0, 3.0, 2.0]})
    out = df.with_columns(c.factory().alias("f"))
    assert out["f"].to_list() == [None, 2.0, 1.0]


def test_factory_is_reusable_across_frames():
    """FeatureSpec rebuilds the expression per call; it must not be consumed."""
    c = compile_feature_expression('pl.col("x") * 2')
    for _ in range(3):
        df = pl.DataFrame({"x": [1.0]})
        assert df.with_columns(c.factory().alias("f"))["f"].to_list() == [2.0]


# ── aggregators ──────────────────────────────────────────────────────────────


def test_aggregator_binds_the_source_column():
    fac = compile_aggregator_expression("pl.col(c).quantile(0.9) - pl.col(c).quantile(0.1)")
    df = pl.DataFrame({"v": [0.0, 1.0, 2.0, 3.0, 4.0]})
    got = df.select(fac("v").alias("spread"))["spread"][0]
    # Polars' default quantile interpolation is "nearest": q90 = 4.0, q10 = 0.0.
    assert got == pytest.approx(4.0, abs=1e-6)

    # The same factory applies to whichever column it is handed.
    df2 = pl.DataFrame({"w": [10.0, 20.0]})
    assert df2.select(fac("w").alias("spread"))["spread"][0] == pytest.approx(10.0)


def test_aggregator_must_reference_its_column():
    with pytest.raises(ExpressionError, match="never references"):
        compile_aggregator_expression('pl.col("hardcoded").mean()')


def test_aggregator_rejects_escapes_too():
    with pytest.raises(ExpressionError):
        compile_aggregator_expression("__import__('os').getcwd()")


# ── documentation source ─────────────────────────────────────────────────────


def test_allowlist_reference_is_serializable_and_populated():
    import json

    ref = allowlist_reference()
    json.dumps(ref)
    assert "rolling_std" in ref["expression_methods"]
    assert "col" in ref["pl_functions"]
    assert "alias" in ref["excluded"]


def test_every_allowed_node_is_an_ast_type():
    assert all(isinstance(n, type) and issubclass(n, ast.AST) for n in _ALLOWED_NODES)

import polars as pl
import pytest

from ml_analysis.features import FeatureRegistry, feature


def test_register_and_resolve_topological():
    reg = FeatureRegistry()

    @feature("a", deps=("x",), registry=reg)
    def _():
        return pl.col("x") * 2

    @feature("b", deps=("a",), registry=reg)
    def _():
        return pl.col("a") + 1

    @feature("c", deps=("b", "a"), registry=reg)
    def _():
        return pl.col("b") + pl.col("a")

    ordered = reg.resolve(["c"])
    names = [s.name for s in ordered]
    assert names.index("a") < names.index("b") < names.index("c")


def test_cycle_detected():
    reg = FeatureRegistry()

    @feature("a", deps=("b",), registry=reg)
    def _():
        return pl.col("b")

    @feature("b", deps=("a",), registry=reg)
    def _():
        return pl.col("a")

    with pytest.raises(ValueError, match="Cyclic"):
        reg.resolve(["a"])


def test_duplicate_registration():
    reg = FeatureRegistry()

    @feature("a", registry=reg)
    def _():
        return pl.col("x")

    with pytest.raises(ValueError, match="already registered"):

        @feature("a", registry=reg)
        def _():
            return pl.col("y")


def test_external_dep_is_ignored():
    reg = FeatureRegistry()

    @feature("a", deps=("raw_x",), registry=reg)
    def _():
        return pl.col("raw_x") * 2

    ordered = reg.resolve(["a"])
    assert [s.name for s in ordered] == ["a"]

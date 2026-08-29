import polars as pl
import pytest

from tessa.features import FeatureRegistry, feature


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


# ── resolve() empty-list contract ────────────────────────────────────────────


def test_resolve_empty_list_selects_nothing():
    """``[]`` must mean "no features"; ``None`` means "all of them".

    These are distinct requests and used to collapse together, because
    ``[]`` is falsy — so there was no way to ask for no features at all.
    """
    reg = FeatureRegistry()

    @feature("a", deps=("x",), registry=reg)
    def _():
        return pl.col("x") * 2

    assert reg.resolve([]) == []
    assert [s.name for s in reg.resolve()] == ["a"]
    assert [s.name for s in reg.resolve(None)] == ["a"]


# ── copy / unregister / merge ────────────────────────────────────────────────


def _reg_with(name: str) -> FeatureRegistry:
    reg = FeatureRegistry()

    @feature(name, deps=("x",), registry=reg)
    def _():
        return pl.col("x")

    return reg


def test_copy_is_independent_in_both_directions():
    src = _reg_with("a")
    clone = src.copy()
    assert clone.names() == ["a"]

    @feature("b", deps=("x",), registry=clone)
    def _():
        return pl.col("x") + 1

    assert "b" not in src
    clone.unregister("a")
    assert "a" in src and "a" not in clone


def test_unregister_allows_redefinition():
    reg = _reg_with("a")
    with pytest.raises(ValueError, match="already registered"):

        @feature("a", deps=("x",), registry=reg)
        def _():
            return pl.col("x") * 3

    reg.unregister("a")

    @feature("a", deps=("x",), registry=reg)
    def _():
        return pl.col("x") * 3

    assert reg.get("a").expr().meta.output_name() == "a"


def test_unregister_unknown_raises():
    with pytest.raises(KeyError):
        FeatureRegistry().unregister("nope")


def test_merge_refuses_collision_unless_overwrite():
    a, b = _reg_with("a"), _reg_with("a")
    with pytest.raises(ValueError, match="already registered"):
        a.merge(b)
    a.merge(b, overwrite=True)
    assert a.names() == ["a"]

    c = _reg_with("c")
    a.merge(c)
    assert sorted(a.names()) == ["a", "c"]


def test_specs_returns_a_shallow_copy():
    reg = _reg_with("a")
    specs = reg.specs()
    del specs["a"]
    assert "a" in reg

"""Unit tests for stock per-sample feature factories in ``features.builtins``."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from tessa import Config
from tessa.features import to_per_sample
from tessa.features.builtins import (
    make_constant_counter,
    make_first_difference,
    make_rolling_mean,
    make_rolling_std,
    make_zscore,
)


def _lf(values: list, col: str, event_id: str = "e1") -> pl.LazyFrame:
    t0 = datetime(2024, 1, 1)
    n = len(values)
    return pl.LazyFrame(
        {
            "timestamp": [t0 + timedelta(seconds=i) for i in range(n)],
            col: values,
            "event_id": [event_id] * n,
        }
    )


def _counter(values: list, col: str) -> list:
    # `make_constant_counter` registers into the process-wide default registry,
    # so each call uses a distinct source column to avoid name collisions.
    make_constant_counter(col)
    out = to_per_sample(_lf(values, col), Config(), [f"{col}__const_count"]).collect()
    return out[f"{col}__const_count"].to_list()


def test_constant_counter_matches_example():
    assert _counter([0, 1, 6, 2, 3, 3, 3, 7], "ex") == [0, 0, 0, 0, 0, 1, 2, 0]


def test_constant_counter_leading_run():
    assert _counter([5, 5, 5, 1, 2, 2], "lead") == [0, 1, 2, 0, 0, 1]


def test_constant_counter_single_row():
    assert _counter([9], "one") == [0]


def test_constant_counter_all_equal():
    assert _counter([4, 4, 4, 4], "flat") == [0, 1, 2, 3]


def test_make_helpers_target_an_explicit_registry():
    """The ``make_*`` factories must be able to avoid the global registry.

    Without this a caller cannot build a private feature set, and calling a
    factory twice in one process is a hard error.
    """
    from tessa.features import default_feature_registry
    from tessa.features.registry import FeatureRegistry

    reg = FeatureRegistry()
    make_rolling_mean("temperature", 5, registry=reg)
    make_rolling_std("temperature", 5, registry=reg)
    make_first_difference("temperature", registry=reg)
    make_zscore("temperature", 5, registry=reg)
    make_constant_counter("temperature", registry=reg)

    assert sorted(reg.names()) == [
        "temperature__const_count",
        "temperature__diff1",
        "temperature__roll_mean_5",
        "temperature__roll_std_5",
        "temperature__zscore_5",
    ]
    # The process-wide registry is untouched.
    for name in reg.names():
        assert name not in default_feature_registry()

    # A second, isolated registry can reuse the same names.
    other = FeatureRegistry()
    make_rolling_mean("temperature", 5, registry=other)
    assert other.names() == ["temperature__roll_mean_5"]


def test_registered_builtin_expression_evaluates():
    from tessa.features.registry import FeatureRegistry

    reg = FeatureRegistry()
    make_first_difference("x", registry=reg)
    df = pl.DataFrame({"x": [1.0, 3.0, 6.0]})
    out = df.with_columns(reg.get("x__diff1").expr())
    assert out["x__diff1"].to_list() == [None, 2.0, 3.0]

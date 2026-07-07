"""Unit tests for stock per-sample feature factories in ``features.builtins``."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from ml_analysis import Config
from ml_analysis.features import to_per_sample
from ml_analysis.features.builtins import make_constant_counter


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

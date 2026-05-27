from datetime import datetime, timedelta

import polars as pl
import pytest

from ml_analysis import Config
from ml_analysis.features import (
    FeatureRegistry,
    feature,
    to_per_sample,
    to_period,
    to_windowed,
)
from ml_analysis.features.aggregates import AggregatorRegistry, aggregate


def _toy_lf(n: int = 60) -> pl.LazyFrame:
    t0 = datetime(2024, 1, 1)
    ts = [t0 + timedelta(seconds=i) for i in range(n)]
    return pl.LazyFrame(
        {
            "timestamp": ts,
            "x": list(range(n)),
            "y": [float(i) * 0.5 for i in range(n)],
            "event_id": ["e1"] * n,
            "asset_id": ["A1"] * n,
            "class": ["TP"] * n,
        }
    )


def _registries() -> tuple[FeatureRegistry, AggregatorRegistry]:
    fr = FeatureRegistry()
    ar = AggregatorRegistry()

    @feature("x_double", deps=("x",), registry=fr)
    def _():
        return pl.col("x") * 2

    @aggregate("mean", registry=ar)
    def _(c):
        return pl.col(c).mean()

    @aggregate("max", registry=ar)
    def _(c):
        return pl.col(c).max()

    return fr, ar


def test_per_sample_adds_column():
    cfg = Config()
    fr, _ = _registries()
    out = to_per_sample(_toy_lf(), cfg, ["x_double"], feature_registry=fr).collect()
    assert "x_double" in out.columns
    assert out["x_double"].to_list() == [i * 2 for i in range(60)]


def test_period_one_row_per_event():
    cfg = Config()
    fr, ar = _registries()
    df = to_period(
        _toy_lf(),
        cfg,
        sources=["x", "y", "x_double"],
        aggregators=["mean", "max"],
        feature_names=["x_double"],
        feature_registry=fr,
        aggregator_registry=ar,
    )
    assert df.height == 1
    assert df["x__mean"][0] == pytest.approx(29.5)
    assert df["x_double__max"][0] == 118
    assert df["class"][0] == "TP"


def test_windowed_groups_by_dynamic():
    cfg = Config()
    fr, ar = _registries()
    out = to_windowed(
        _toy_lf(60),
        cfg,
        every="10s",
        period="10s",
        sources=["x"],
        aggregators=["mean"],
        feature_registry=fr,
        aggregator_registry=ar,
    ).collect()
    assert out.height == 6
    assert "x__mean" in out.columns
    assert "class" in out.columns

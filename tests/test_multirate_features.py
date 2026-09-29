"""Per-sample features on sources sampled at different rates.

``load_event`` outer-joins sources on the timestamp by default, so a slow
source sits on the fast grid with nulls between its samples. The stock
features must be computed on each signal's own samples, not row by row.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import polars as pl
import pytest

from tessa import Config, Dataset, Run
from tessa.analysis import prepare_xy
from tessa.examples import make_example_dataset
from tessa.features import FeatureRegistry, to_per_sample, to_period
from tessa.features.builtins import (
    make_constant_counter,
    make_first_difference,
    make_rolling_mean,
    make_rolling_std,
    make_zscore,
)
from tessa.labels import ExcelLabelSource


def _interleaved() -> pl.LazyFrame:
    """A slow signal ``x`` (every 2nd row) next to a fast signal ``y``."""
    t0 = datetime(2024, 1, 1)
    x = [1.0, None, 4.0, None, 9.0, None, 16.0, None, 16.0, None]
    return pl.LazyFrame(
        {
            "timestamp": [t0 + timedelta(minutes=i) for i in range(len(x))],
            "x": x,
            "y": [float(i) for i in range(len(x))],
            "event_id": ["e1"] * len(x),
        }
    )


def test_features_skip_other_sources_rows():
    reg = FeatureRegistry()
    make_first_difference("x", registry=reg)
    make_rolling_mean("x", 2, registry=reg)
    make_rolling_std("x", 2, registry=reg)
    make_zscore("x", 3, registry=reg)
    make_constant_counter("x", registry=reg)
    out = to_per_sample(_interleaved(), Config(), feature_registry=reg).collect()

    own = out.filter(pl.col("x").is_not_null())
    assert own["x__diff1"].to_list() == [None, 3.0, 5.0, 7.0, 0.0]
    assert own["x__roll_mean_2"].to_list() == [None, 2.5, 6.5, 12.5, 16.0]
    assert own["x__roll_std_2"].null_count() == 1
    assert own["x__zscore_3"].null_count() == 2
    assert own["x__const_count"].to_list() == [0, 0, 0, 0, 1]

    # Rows where the source was not sampled stay null for every feature.
    gaps = out.filter(pl.col("x").is_null())
    for name in reg.names():
        assert gaps[name].null_count() == gaps.height, name


def test_features_unchanged_without_nulls():
    t0 = datetime(2024, 1, 1)
    values = [1.0, 4.0, 9.0, 16.0]
    lf = pl.LazyFrame({"timestamp": [t0 + timedelta(minutes=i) for i in range(4)], "v": values})
    reg = FeatureRegistry()
    make_first_difference("v", registry=reg)
    make_rolling_mean("v", 2, registry=reg)
    out = to_per_sample(lf, Config(), feature_registry=reg).collect()
    assert out["v__diff1"].to_list() == [None, 3.0, 5.0, 7.0]
    assert out["v__roll_mean_2"].to_list() == [None, 2.5, 6.5, 12.5]


@pytest.fixture(scope="module")
def example(tmp_path_factory) -> tuple[Config, pl.DataFrame]:
    ex = make_example_dataset(tmp_path_factory.mktemp("store"), events_per_class=3)
    cfg = Config(data_root=ex.data_root)
    labels = ExcelLabelSource(ex.labels_path, sheet="labels", column_map=ex.column_map).load(cfg)
    with pytest.warns(UserWarning, match="dropped"):
        labels = Dataset(cfg).available(labels, check="rows")
    return cfg, labels.with_columns(pl.col("replacement_type").cast(pl.String))


def _dynamics(signals: list[str]) -> FeatureRegistry:
    reg = FeatureRegistry()
    for s in signals:
        make_first_difference(s, registry=reg)
        make_rolling_std(s, window=10, registry=reg)
    return reg


def test_slow_source_dynamics_are_not_null(example):
    cfg, labels = example
    events = Dataset(cfg).events(labels)
    reg = _dynamics(["temperature", "pressure", "flow"])
    period = to_period(events, cfg, aggregators=["mean", "std"], feature_registry=reg)

    dynamics = [c for c in period.columns if "__diff1__" in c or "__roll_std_10__" in c]
    assert {"pressure__diff1__mean", "flow__roll_std_10__std"} <= set(dynamics)
    for col in dynamics:
        assert period[col].null_count() == 0, col

    # Same numbers as computing the features on the 5-minute source alone.
    alone = Dataset(cfg).events(labels, columns=["pressure", "flow"])
    first = next(iter(alone.values())).collect()
    assert first["pressure"].null_count() == 0  # only the `process` source loaded
    ref = to_period(
        alone, cfg, aggregators=["mean", "std"], feature_registry=_dynamics(["pressure", "flow"])
    )
    for col in ("pressure__diff1__std", "flow__roll_std_10__mean"):
        assert period[col].to_list() == pytest.approx(ref[col].to_list())


def test_default_run_keeps_every_event(example):
    cfg, labels = example
    events = Dataset(cfg).events(labels)
    period = to_period(events, cfg, feature_registry=_dynamics(["pressure", "flow"]))
    run = Run(period, target_col=cfg.class_col, cfg=cfg)  # default NullPolicy("drop_rows")
    prep = prepare_xy(run.ctx)
    assert prep.report.n_rows_out == period.height


def test_events_forward_asof_merge(example):
    cfg, labels = example
    events = Dataset(cfg).events(labels, merge="asof", source_order=["sensor"], asof_tolerance="5m")
    frame = next(iter(events.values())).collect()
    assert frame.height == 4 * 60  # the 1-minute sensor grid
    assert frame["pressure"].null_count() == 0
    assert frame["flow"].null_count() == 0


def test_events_reject_unknown_merge(example):
    cfg, labels = example
    with pytest.raises(ValueError):
        Dataset(cfg).events(labels.head(1), merge="sideways")

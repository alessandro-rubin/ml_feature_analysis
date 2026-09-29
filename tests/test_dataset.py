from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from tessa import Config, Run
from tessa.analysis.base import prepare_xy
from tessa.dataset import EventFrames, build, load_asset, load_event
from tessa.features import to_period
from tessa.features.windows import WindowSpec, materialize


@pytest.fixture
def fake_data(tmp_path: Path) -> Config:
    """Asset A1 with two sources: hourly sensor x/y and 2-hourly flow z."""
    cfg = Config(data_root=tmp_path)
    folder = cfg.asset_dir("A1")
    folder.mkdir(parents=True)

    n = 24 * 31
    ts = pl.datetime_range(
        datetime(2024, 1, 1), datetime(2024, 1, 1) + timedelta(hours=n - 1), "1h", eager=True
    )
    pl.DataFrame(
        {cfg.timestamp_col: ts, "x": list(range(n)), "y": [i * 0.5 for i in range(n)]}
    ).write_parquet(folder / "sensor_20240101_20240131.parquet")

    ts = pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 31, 22), "2h", eager=True)
    pl.DataFrame({cfg.timestamp_col: ts, "z": [float(i) for i in range(len(ts))]}).write_parquet(
        folder / "flow_rate_240101_240131.parquet"
    )
    return cfg


def test_config_works_as_loader_config(fake_data: Config):
    """tessa.Config is a asset_loader.LoaderConfig and drives the loader."""
    from asset_loader import LoaderConfig

    assert isinstance(fake_data, LoaderConfig)
    df = load_event("A1", datetime(2024, 1, 10), datetime(2024, 1, 12), fake_data).collect()
    assert set(df.columns) == {"timestamp", "x", "y", "z"}
    assert df.height == 49


def test_load_asset_reexported(fake_data: Config, tmp_path: Path):
    df = load_asset("A1", tmp_path)
    assert df.height == 24 * 31


def test_legacy_loader_module_shim():
    from tessa.dataset.loader import (  # noqa: F401
        discover_files,
        discover_sources,
        load_asset,
        load_event,
    )


def test_build_attaches_labels(fake_data: Config):
    labels = pl.DataFrame(
        {
            "asset_id": ["A1"],
            "start": [datetime(2024, 1, 10)],
            "end": [datetime(2024, 1, 12)],
            "class": ["TP"],
            "replacement_type": ["bearing"],
        }
    )
    out = build(labels, fake_data)
    assert len(out) == 1
    df = next(iter(out.values())).collect()
    assert df["class"][0] == "TP"
    assert df["replacement_type"][0] == "bearing"
    assert df["asset_id"][0] == "A1"
    assert {"x", "y", "z"} <= set(df.columns)


def _labels_with_numeric_extra() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "asset_id": ["A1", "A1"],
            "start": [datetime(2024, 1, 10), datetime(2024, 1, 20)],
            "end": [datetime(2024, 1, 12), datetime(2024, 1, 22)],
            "class": ["TP", "FP"],
            "severity": [3, 1],
            "replacement_type": ["bearing", "seal"],
        }
    )


def test_build_records_label_columns(fake_data: Config):
    events = build(_labels_with_numeric_extra(), fake_data)
    assert isinstance(events, EventFrames)
    assert events.label_cols == ("event_id", "asset_id", "class", "severity", "replacement_type")
    first = next(iter(events))
    assert events.subset([first]).label_cols == events.label_cols


def test_numeric_label_column_is_not_a_feature(fake_data: Config):
    events = build(_labels_with_numeric_extra(), fake_data)
    df = to_period(events, fake_data, aggregators=["mean", "max"], feature_names=[])

    assert not [c for c in df.columns if c.startswith("severity__")]
    assert df["severity"].to_list() == [3, 1]
    assert df["severity"].dtype.is_integer()
    assert df["replacement_type"].to_list() == ["bearing", "seal"]
    assert {"x__mean", "y__mean", "z__mean"} <= set(df.columns)


def test_numeric_label_column_is_not_a_feature_windowed(fake_data: Config):
    events = build(_labels_with_numeric_extra(), fake_data)
    df = materialize(
        events, WindowSpec.tumbling("1d"), fake_data, aggregators=["mean"], feature_names=[]
    )
    assert not [c for c in df.columns if c.startswith("severity__")]
    assert set(df["severity"].to_list()) == {3, 1}
    assert "x__mean" in df.columns


def test_plain_dict_needs_explicit_label_cols(fake_data: Config):
    """Hand-built {event_id: LazyFrame} dicts keep working; declare labels explicitly."""
    events = build(_labels_with_numeric_extra(), fake_data)
    plain = dict(events)

    leaky = to_period(plain, fake_data, aggregators=["mean"], feature_names=[])
    assert "severity__mean" in leaky.columns  # dtype heuristic alone cannot tell

    df = to_period(
        plain, fake_data, aggregators=["mean"], feature_names=[], label_cols=["severity"]
    )
    assert "severity__mean" not in df.columns
    assert df["severity"].to_list() == [3, 1]


def test_run_never_uses_label_columns_as_features(fake_data: Config):
    events = build(_labels_with_numeric_extra(), fake_data)
    df = to_period(events, fake_data, aggregators=["mean"], feature_names=[])

    prep = prepare_xy(Run(df, target_col="class", cfg=fake_data, label_cols=events.label_cols).ctx)
    assert "severity" not in prep.feature_cols
    assert {"x__mean", "y__mean", "z__mean"} <= set(prep.feature_cols)

    # Without the declaration a numeric label column would be read as a feature.
    assert "severity" in prepare_xy(Run(df, target_col="class", cfg=fake_data).ctx).feature_cols

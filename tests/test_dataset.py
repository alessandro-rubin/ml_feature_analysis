from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from tessa import Config
from tessa.dataset import build, load_asset, load_event


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

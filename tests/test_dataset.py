from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ml_analysis import Config
from ml_analysis.dataset import build, discover_files, load_event


@pytest.fixture
def fake_data(tmp_path: Path) -> Config:
    cfg = Config(data_root=tmp_path)
    asset = "A1"
    folder = cfg.asset_dir(asset)
    folder.mkdir(parents=True)

    chunks = [
        ("20240101", "20240131", datetime(2024, 1, 1), 24 * 31),
        ("20240201", "20240228", datetime(2024, 2, 1), 24 * 28),
    ]
    for s, e, t0, n in chunks:
        ts = pl.datetime_range(t0, t0 + timedelta(hours=n - 1), "1h", eager=True)
        df = pl.DataFrame(
            {cfg.timestamp_col: ts, "x": list(range(n)), "y": [i * 0.5 for i in range(n)]}
        )
        df.write_parquet(folder / f"{asset}_{s}_{e}.parquet")
    return cfg


def test_discover_files_overlap(fake_data: Config):
    files = discover_files("A1", datetime(2024, 1, 15), datetime(2024, 2, 5), fake_data)
    assert len(files) == 2


def test_discover_files_outside(fake_data: Config):
    files = discover_files("A1", datetime(2025, 1, 1), datetime(2025, 1, 31), fake_data)
    assert files == []


def test_load_event_filters(fake_data: Config):
    lf = load_event("A1", datetime(2024, 1, 10), datetime(2024, 1, 12), fake_data)
    df = lf.collect()
    assert df["timestamp"].min() >= datetime(2024, 1, 10)
    assert df["timestamp"].max() <= datetime(2024, 1, 12)
    assert set(df.columns) == {"timestamp", "x", "y"}


def test_load_event_column_subset(fake_data: Config):
    lf = load_event("A1", datetime(2024, 1, 10), datetime(2024, 1, 12), fake_data, columns=["x"])
    df = lf.collect()
    assert set(df.columns) == {"timestamp", "x"}


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

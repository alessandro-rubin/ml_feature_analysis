from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ml_analysis import Config
from ml_analysis.dataset import build, discover_files, discover_sources, load_event


@pytest.fixture
def fake_data(tmp_path: Path) -> Config:
    """Asset A1 with two sources of different columns and sampling rates.

    - ``sensor``: hourly ``x``/``y``, split into two monthly files
      (8-digit dates in the filename).
    - ``flow_rate``: 2-hourly ``z``, one file covering both months
      (6-digit dates, underscore in the source name).
    """
    cfg = Config(data_root=tmp_path)
    folder = cfg.asset_dir("A1")
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
        df.write_parquet(folder / f"sensor_{s}_{e}.parquet")

    ts = pl.datetime_range(
        datetime(2024, 1, 1), datetime(2024, 2, 28, 22), "2h", eager=True
    )
    df = pl.DataFrame({cfg.timestamp_col: ts, "z": [float(i) for i in range(len(ts))]})
    df.write_parquet(folder / "flow_rate_240101_240228.parquet")
    return cfg


def test_discover_files_overlap(fake_data: Config):
    files = discover_files("A1", datetime(2024, 1, 15), datetime(2024, 2, 5), fake_data)
    assert len(files) == 3  # both sensor chunks + the flow_rate file


def test_discover_sources_grouping(fake_data: Config):
    groups = discover_sources(
        "A1", datetime(2024, 1, 15), datetime(2024, 2, 5), fake_data
    )
    assert set(groups) == {"sensor", "flow_rate"}
    assert len(groups["sensor"]) == 2
    assert len(groups["flow_rate"]) == 1


def test_discover_files_outside(fake_data: Config):
    files = discover_files("A1", datetime(2025, 1, 1), datetime(2025, 1, 31), fake_data)
    assert files == []


def test_load_event_joins_sources(fake_data: Config):
    lf = load_event("A1", datetime(2024, 1, 10), datetime(2024, 1, 12), fake_data)
    df = lf.collect()
    assert set(df.columns) == {"timestamp", "x", "y", "z"}
    assert df["timestamp"].min() >= datetime(2024, 1, 10)
    assert df["timestamp"].max() <= datetime(2024, 1, 12)
    # hourly union grid: Jan 10 00:00 .. Jan 12 00:00 inclusive
    assert df.height == 49
    assert df["timestamp"].is_sorted()
    # z sampled every 2h -> null on odd hours
    assert df["z"].null_count() == 24
    assert df["x"].null_count() == 0


def test_load_event_spans_time_chunks(fake_data: Config):
    lf = load_event("A1", datetime(2024, 1, 31, 20), datetime(2024, 2, 1, 4), fake_data)
    df = lf.collect()
    assert df.height == 9  # hourly across the Jan/Feb file boundary
    assert df["x"].null_count() == 0


def test_load_event_column_subset_prunes_sources(fake_data: Config):
    lf = load_event(
        "A1", datetime(2024, 1, 10), datetime(2024, 1, 12), fake_data, columns=["x"]
    )
    df = lf.collect()
    assert set(df.columns) == {"timestamp", "x"}
    # flow_rate source skipped entirely -> pure hourly grid, no extra rows
    assert df.height == 49


def test_load_event_column_subset_across_sources(fake_data: Config):
    lf = load_event(
        "A1",
        datetime(2024, 1, 10),
        datetime(2024, 1, 12),
        fake_data,
        columns=["x", "z"],
    )
    df = lf.collect()
    assert set(df.columns) == {"timestamp", "x", "z"}


def test_load_event_missing_column(fake_data: Config):
    with pytest.raises(ValueError, match="not found in any source"):
        load_event(
            "A1",
            datetime(2024, 1, 10),
            datetime(2024, 1, 12),
            fake_data,
            columns=["nope"],
        )


def test_load_event_duplicate_column_across_sources(fake_data: Config):
    folder = fake_data.asset_dir("A1")
    ts = pl.datetime_range(
        datetime(2024, 1, 1), datetime(2024, 1, 31), "1d", eager=True
    )
    pl.DataFrame({fake_data.timestamp_col: ts, "x": [0.0] * len(ts)}).write_parquet(
        folder / "dup_240101_240131.parquet"
    )
    with pytest.raises(ValueError, match="unique across sources"):
        load_event("A1", datetime(2024, 1, 10), datetime(2024, 1, 12), fake_data)


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

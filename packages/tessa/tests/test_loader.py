from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from tessa import LoaderConfig, discover_files, discover_sources, load_asset, load_event

N_JAN = 24 * 31
N_FEB = 24 * 28
N_FLOW = (N_JAN + N_FEB) // 2  # 2-hourly over the same span


@pytest.fixture
def fake_data(tmp_path: Path) -> LoaderConfig:
    """Asset A1 with two sources of different columns and sampling rates.

    - ``sensor``: hourly ``x``/``y``, split into two monthly files
      (8-digit dates in the filename).
    - ``flow_rate``: 2-hourly ``z``, one file covering both months
      (6-digit dates, underscore in the source name).
    """
    cfg = LoaderConfig(data_root=tmp_path)
    folder = cfg.asset_dir("A1")
    folder.mkdir(parents=True)

    chunks = [
        ("20240101", "20240131", datetime(2024, 1, 1), N_JAN),
        ("20240201", "20240228", datetime(2024, 2, 1), N_FEB),
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


def test_discover_files_overlap(fake_data: LoaderConfig):
    files = discover_files("A1", datetime(2024, 1, 15), datetime(2024, 2, 5), fake_data)
    assert len(files) == 3  # both sensor chunks + the flow_rate file


def test_discover_sources_grouping(fake_data: LoaderConfig):
    groups = discover_sources(
        "A1", datetime(2024, 1, 15), datetime(2024, 2, 5), fake_data
    )
    assert set(groups) == {"sensor", "flow_rate"}
    assert len(groups["sensor"]) == 2
    assert len(groups["flow_rate"]) == 1


def test_discover_files_outside(fake_data: LoaderConfig):
    files = discover_files("A1", datetime(2025, 1, 1), datetime(2025, 1, 31), fake_data)
    assert files == []


def test_discover_unbounded(fake_data: LoaderConfig):
    assert len(discover_files("A1", None, None, fake_data)) == 3
    assert len(discover_files("A1", datetime(2024, 2, 1), None, fake_data)) == 2
    assert len(discover_files("A1", None, datetime(2024, 1, 31), fake_data)) == 2


def test_load_event_joins_sources(fake_data: LoaderConfig):
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


def test_load_event_full_history(fake_data: LoaderConfig):
    df = load_event("A1", None, None, fake_data).collect()
    assert df.height == N_JAN + N_FEB
    assert df["z"].null_count() == N_JAN + N_FEB - N_FLOW
    assert df["timestamp"].is_sorted()


def test_load_event_spans_time_chunks(fake_data: LoaderConfig):
    lf = load_event("A1", datetime(2024, 1, 31, 20), datetime(2024, 2, 1, 4), fake_data)
    df = lf.collect()
    assert df.height == 9  # hourly across the Jan/Feb file boundary
    assert df["x"].null_count() == 0


def test_load_event_column_subset_prunes_sources(fake_data: LoaderConfig):
    lf = load_event(
        "A1", datetime(2024, 1, 10), datetime(2024, 1, 12), fake_data, columns=["x"]
    )
    df = lf.collect()
    assert set(df.columns) == {"timestamp", "x"}
    # flow_rate source skipped entirely -> pure hourly grid, no extra rows
    assert df.height == 49


def test_load_event_column_subset_across_sources(fake_data: LoaderConfig):
    lf = load_event(
        "A1",
        datetime(2024, 1, 10),
        datetime(2024, 1, 12),
        fake_data,
        columns=["x", "z"],
    )
    df = lf.collect()
    assert set(df.columns) == {"timestamp", "x", "z"}


def test_load_event_missing_column(fake_data: LoaderConfig):
    with pytest.raises(ValueError, match="not found in any source"):
        load_event(
            "A1",
            datetime(2024, 1, 10),
            datetime(2024, 1, 12),
            fake_data,
            columns=["nope"],
        )


def test_load_event_duplicate_column_across_sources(fake_data: LoaderConfig):
    folder = fake_data.asset_dir("A1")
    ts = pl.datetime_range(
        datetime(2024, 1, 1), datetime(2024, 1, 31), "1d", eager=True
    )
    pl.DataFrame({fake_data.timestamp_col: ts, "x": [0.0] * len(ts)}).write_parquet(
        folder / "dup_240101_240131.parquet"
    )
    with pytest.raises(ValueError, match="unique across sources"):
        load_event("A1", datetime(2024, 1, 10), datetime(2024, 1, 12), fake_data)


def test_load_asset_one_liner(fake_data: LoaderConfig, tmp_path: Path):
    df = load_asset("A1", tmp_path)
    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {"timestamp", "x", "y", "z"}
    assert df.height == N_JAN + N_FEB  # entire history
    assert df["timestamp"].is_sorted()


def test_load_asset_accepts_str_root_and_config(fake_data: LoaderConfig, tmp_path: Path):
    assert load_asset("A1", str(tmp_path)).height == N_JAN + N_FEB
    assert load_asset("A1", fake_data).height == N_JAN + N_FEB


def test_load_asset_window_and_columns(fake_data: LoaderConfig, tmp_path: Path):
    df = load_asset(
        "A1",
        tmp_path,
        start=datetime(2024, 1, 10),
        end=datetime(2024, 1, 12),
        columns=["x"],
    )
    assert set(df.columns) == {"timestamp", "x"}
    assert df.height == 49


def test_load_asset_lazy(fake_data: LoaderConfig, tmp_path: Path):
    lf = load_asset("A1", tmp_path, lazy=True)
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().height == N_JAN + N_FEB


def test_load_asset_missing_asset(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_asset("nope", tmp_path)

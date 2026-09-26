from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from asset_loader import (
    LoaderConfig,
    LoadResult,
    discover_files,
    discover_sources,
    load_asset,
    load_event,
)

N_JAN = 24 * 31
N_FEB = 24 * 28
N_FLOW = (N_JAN + N_FEB) // 2  # 2-hourly over the same span

WIN = (datetime(2024, 1, 10), datetime(2024, 1, 12))  # 49 hourly / 25 2-hourly stamps
N_WIN_H, N_WIN_2H = 49, 25


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

    ts = pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 2, 28, 22), "2h", eager=True)
    df = pl.DataFrame({cfg.timestamp_col: ts, "z": [float(i) for i in range(len(ts))]})
    df.write_parquet(folder / "flow_rate_240101_240228.parquet")
    return cfg


def test_discover_files_overlap(fake_data: LoaderConfig):
    files = discover_files("A1", datetime(2024, 1, 15), datetime(2024, 2, 5), fake_data)
    assert len(files) == 3  # both sensor chunks + the flow_rate file


def test_discover_sources_grouping(fake_data: LoaderConfig):
    groups = discover_sources("A1", datetime(2024, 1, 15), datetime(2024, 2, 5), fake_data)
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
    lf = load_event("A1", datetime(2024, 1, 10), datetime(2024, 1, 12), fake_data, columns=["x"])
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
    ts = pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 31), "1d", eager=True)
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


# --- merge strategies --------------------------------------------------


def test_merge_inner_keeps_common_timestamps(fake_data: LoaderConfig):
    df = load_event("A1", *WIN, fake_data, merge="inner").collect()
    assert df.height == N_WIN_2H  # only the 2-hourly stamps flow_rate has
    assert df["z"].null_count() == 0
    assert df["x"].null_count() == 0


def test_merge_left_anchors_on_first_source(fake_data: LoaderConfig):
    df = load_event("A1", *WIN, fake_data, merge="left", source_order=["sensor"]).collect()
    assert df.height == N_WIN_H  # sensor's hourly grid
    assert df["z"].null_count() == N_WIN_H - N_WIN_2H
    # ...and the other way round: flow_rate as the anchor drops the odd hours.
    df = load_event("A1", *WIN, fake_data, merge="left", source_order=["flow_rate"]).collect()
    assert df.height == N_WIN_2H
    assert df["x"].null_count() == 0


def test_merge_asof_fills_the_slow_source_forward(fake_data: LoaderConfig):
    df = load_event("A1", *WIN, fake_data, merge="asof", source_order=["sensor"]).collect()
    assert df.height == N_WIN_H
    assert df["z"].null_count() == 0  # odd hours carry the previous 2-hourly value
    assert df["z"][0] == df["z"][1]


def test_merge_asof_tolerance_limits_the_match(fake_data: LoaderConfig):
    df = load_event(
        "A1", *WIN, fake_data, merge="asof", source_order=["sensor"], asof_tolerance="30m"
    ).collect()
    assert df.height == N_WIN_H
    assert df["z"].null_count() == N_WIN_H - N_WIN_2H  # odd hours out of reach


def test_merge_vertical_stacks_rows(fake_data: LoaderConfig):
    df = load_event("A1", *WIN, fake_data, merge="vertical").collect()
    assert set(df.columns) == {"timestamp", "x", "y", "z"}
    assert df.height == N_WIN_H + N_WIN_2H
    assert df["x"].null_count() == N_WIN_2H  # flow_rate rows carry no x
    assert df["z"].null_count() == N_WIN_H
    assert df["timestamp"].is_sorted()


def test_merge_unknown_strategy(fake_data: LoaderConfig):
    with pytest.raises(ValueError, match="Unknown merge strategy"):
        load_event("A1", *WIN, fake_data, merge="sideways")


def test_source_order_ignores_absent_sources(fake_data: LoaderConfig):
    df = load_event(
        "A1", *WIN, fake_data, merge="left", source_order=["nope", "sensor", "gone"]
    ).collect()
    assert df.height == N_WIN_H  # sensor still anchors


# --- duplicate column names across sources -----------------------------


@pytest.fixture
def dup_data(fake_data: LoaderConfig) -> LoaderConfig:
    """Add a ``backup`` source re-recording ``x`` over the Jan 10-12 window.

    Its ``x`` is null on odd offsets, so a coalesce falls back to
    ``sensor`` for half the rows.
    """
    folder = fake_data.asset_dir("A1")
    ts = pl.datetime_range(*WIN, "1h", eager=True)
    x = [1000 + i if i % 2 == 0 else None for i in range(len(ts))]
    pl.DataFrame({fake_data.timestamp_col: ts, "x": x}).write_parquet(
        folder / "backup_240110_240112.parquet"
    )
    return fake_data


def test_duplicate_default_still_errors(dup_data: LoaderConfig):
    with pytest.raises(ValueError, match="unique across sources"):
        load_event("A1", *WIN, dup_data)


def test_duplicate_rename_keeps_both_copies(dup_data: LoaderConfig):
    df = load_event("A1", *WIN, dup_data, on_duplicate="rename").collect()
    assert set(df.columns) == {"timestamp", "x__backup", "x__sensor", "y", "z"}
    assert df["x__sensor"].null_count() == 0
    assert df["x__backup"].null_count() == N_WIN_H // 2


def test_duplicate_coalesce_follows_source_order(dup_data: LoaderConfig):
    # Alphabetical order puts backup first, so its values win where present.
    df = load_event("A1", *WIN, dup_data, on_duplicate="coalesce").collect()
    assert set(df.columns) == {"timestamp", "x", "y", "z"}
    assert df["x"].null_count() == 0
    assert df["x"][0] == 1000  # backup
    assert df["x"][1] == 24 * 9 + 1  # sensor's hourly counter, backup being null here

    df = load_event(
        "A1", *WIN, dup_data, on_duplicate="coalesce", source_order=["sensor"]
    ).collect()
    assert df["x"][0] == 24 * 9  # sensor wins outright


def test_duplicate_first_and_last(dup_data: LoaderConfig):
    order = ["sensor", "backup"]
    first = load_event("A1", *WIN, dup_data, on_duplicate="first", source_order=order).collect()
    assert set(first.columns) == {"timestamp", "x", "y", "z"}
    assert first["x"].null_count() == 0  # sensor's copy

    last = load_event("A1", *WIN, dup_data, on_duplicate="last", source_order=order).collect()
    assert last["x"].null_count() == N_WIN_H // 2  # backup's copy


def test_duplicate_column_subset_selects_every_copy(dup_data: LoaderConfig):
    df = load_event("A1", *WIN, dup_data, columns=["x"], on_duplicate="rename").collect()
    assert set(df.columns) == {"timestamp", "x__backup", "x__sensor"}


def test_duplicate_vertical_merge_coalesces_per_row(dup_data: LoaderConfig):
    df = load_event("A1", *WIN, dup_data, merge="vertical", on_duplicate="coalesce").collect()
    assert set(df.columns) == {"timestamp", "x", "y", "z"}
    assert df.height == 2 * N_WIN_H + N_WIN_2H  # sensor + backup + flow_rate rows
    assert df["x"].null_count() == N_WIN_2H + N_WIN_H // 2


def test_duplicate_unknown_policy(dup_data: LoaderConfig):
    with pytest.raises(ValueError, match="Unknown on_duplicate policy"):
        load_event("A1", *WIN, dup_data, on_duplicate="ignore")


# --- metadata ----------------------------------------------------------


def test_metadata_shape_and_unpacking(fake_data: LoaderConfig):
    result = load_event("A1", *WIN, fake_data, with_metadata=True)
    assert isinstance(result, LoadResult)
    frame, meta = result  # a plain 2-tuple
    assert isinstance(frame, pl.LazyFrame)
    assert meta["asset_id"] == "A1"
    assert meta["requested"] == {"start": WIN[0], "end": WIN[1], "columns": None}
    assert meta["config"]["timestamp_col"] == "timestamp"
    assert meta["source_order"] == ["flow_rate", "sensor"]
    assert meta["columns"] == ["timestamp", "z", "x", "y"]
    assert set(meta["schema"]) == {"timestamp", "x", "y", "z"}
    assert meta["merge"]["strategy"] == "outer"
    assert meta["merge"]["on"] == "timestamp"


def test_metadata_records_sources_and_files(fake_data: LoaderConfig):
    _, meta = load_event("A1", None, None, fake_data, with_metadata=True)
    sensor = meta["sources"]["sensor"]
    assert sensor["n_files"] == 2
    assert all(f.endswith(".parquet") for f in sensor["files"])
    assert sensor["columns_available"] == ["x", "y"]
    assert sensor["period"] == {"start": datetime(2024, 1, 1), "end": datetime(2024, 2, 28)}
    assert meta["sources"]["flow_rate"]["n_files"] == 1


def test_metadata_operations_log(fake_data: LoaderConfig):
    _, meta = load_event("A1", *WIN, fake_data, with_metadata=True)
    ops = [op["op"] for op in meta["operations"]]
    assert ops[0] == "discover"
    assert ops.count("scan") == 2
    assert "concat" not in ops  # the window touches one sensor file only
    assert ops[-2:] == ["filter", "sort"]
    merge_op = next(op for op in meta["operations"] if op["op"] == "merge")
    assert merge_op["strategy"] == "outer"
    assert merge_op["sources"] == ["flow_rate", "sensor"]

    # Widening the window pulls in sensor's second monthly file.
    _, meta = load_event("A1", None, None, fake_data, with_metadata=True)
    concat = next(op for op in meta["operations"] if op["op"] == "concat")
    assert concat == {"op": "concat", "source": "sensor", "how": "vertical", "n_files": 2}


def test_metadata_records_duplicate_resolution(dup_data: LoaderConfig):
    _, meta = load_event("A1", *WIN, dup_data, on_duplicate="coalesce", with_metadata=True)
    dupes = meta["duplicates"]
    assert dupes["policy"] == "coalesce"
    assert dupes["columns"] == {"x": ["backup", "sensor"]}
    assert dupes["coalesced"] == {"x": ["x__backup", "x__sensor"]}
    assert meta["sources"]["sensor"]["renamed"] == {"x": "x__sensor"}
    assert [op["op"] for op in meta["operations"]].count("rename") == 2
    assert any(op["op"] == "coalesce" and op["column"] == "x" for op in meta["operations"])


def test_metadata_records_dropped_columns(dup_data: LoaderConfig):
    _, meta = load_event(
        "A1", *WIN, dup_data, on_duplicate="first", source_order=["sensor"], with_metadata=True
    )
    assert meta["duplicates"]["dropped"] == {"backup": ["x"]}
    assert meta["sources"]["backup"]["dropped"] == ["x"]
    # backup had nothing but x, so it drops out of the merge entirely.
    assert meta["merge"]["sources"] == ["sensor", "flow_rate"]
    skipped = next(op for op in meta["operations"] if op["op"] == "skip_source")
    assert skipped["source"] == "backup"
    assert "duplicates" in skipped["reason"]


def test_metadata_records_skipped_sources(fake_data: LoaderConfig):
    _, meta = load_event("A1", *WIN, fake_data, columns=["x"], with_metadata=True)
    skipped = [op for op in meta["operations"] if op["op"] == "skip_source"]
    assert [op["source"] for op in skipped] == ["flow_rate"]
    assert meta["merge"]["sources"] == ["sensor"]


def test_metadata_records_asof_settings(fake_data: LoaderConfig):
    _, meta = load_event(
        "A1",
        *WIN,
        fake_data,
        merge="asof",
        source_order=["sensor"],
        asof_tolerance="30m",
        with_metadata=True,
    )
    assert meta["merge"]["anchor"] == "sensor"
    assert meta["merge"]["asof_strategy"] == "backward"
    assert meta["merge"]["asof_tolerance"] == "30m"


def test_load_asset_with_metadata_is_eager(fake_data: LoaderConfig, tmp_path: Path):
    df, meta = load_asset("A1", tmp_path, with_metadata=True)
    assert isinstance(df, pl.DataFrame)
    assert df.height == N_JAN + N_FEB
    assert meta["operations"][-1] == {"op": "collect"}


def test_load_asset_with_metadata_lazy(fake_data: LoaderConfig, tmp_path: Path):
    lf, meta = load_asset("A1", tmp_path, lazy=True, with_metadata=True)
    assert isinstance(lf, pl.LazyFrame)
    assert "collect" not in [op["op"] for op in meta["operations"]]
    assert lf.collect().height == N_JAN + N_FEB


def test_load_asset_forwards_merge_options(dup_data: LoaderConfig, tmp_path: Path):
    df = load_asset(
        "A1",
        tmp_path,
        start=WIN[0],
        end=WIN[1],
        merge="inner",
        on_duplicate="rename",
    )
    assert set(df.columns) == {"timestamp", "x__backup", "x__sensor", "y", "z"}
    assert df.height == N_WIN_2H

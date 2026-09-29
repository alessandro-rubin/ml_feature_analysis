"""Matching a label table against the data store before building events."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from tessa import Config, Dataset
from tessa.dataset import build, data_availability, file_catalog, filter_available
from tessa.dataset.availability import (
    REASON_LOW_COVERAGE,
    REASON_NO_FILES,
    REASON_NO_FOLDER,
    REASON_NO_OVERLAP,
    REASON_NO_ROWS,
)


def _write(folder: Path, name: str, start: datetime, end: datetime) -> None:
    """Hourly file whose one value column is named after its source."""
    ts = pl.datetime_range(start, end, "1h", eager=True)
    column = name.split("_")[0]
    pl.DataFrame({"timestamp": ts, column: [float(i) for i in range(len(ts))]}).write_parquet(
        folder / name
    )


@pytest.fixture
def store(tmp_path: Path) -> Config:
    """A1: sensor over Jan (file stops on the 20th despite its name), flow over
    Jan 1-15. A2: data in March only. A3: folder with no matching parquet."""
    cfg = Config(data_root=tmp_path)
    a1 = cfg.asset_dir("A1")
    a1.mkdir()
    _write(a1, "sensor_20240101_20240131.parquet", datetime(2024, 1, 1), datetime(2024, 1, 20))
    _write(a1, "flow_240101_240115.parquet", datetime(2024, 1, 1), datetime(2024, 1, 15, 23))
    a2 = cfg.asset_dir("A2")
    a2.mkdir()
    _write(a2, "sensor_20240301_20240331.parquet", datetime(2024, 3, 1), datetime(2024, 3, 31))
    a3 = cfg.asset_dir("A3")
    a3.mkdir()
    (a3 / "notes.txt").write_text("not data")
    return cfg


def _labels() -> pl.DataFrame:
    rows = [
        ("A1", datetime(2024, 1, 5), datetime(2024, 1, 6), "TP"),  # both sources
        ("A1", datetime(2024, 1, 25), datetime(2024, 1, 26), "FP"),  # gap inside file
        ("A1", datetime(2024, 1, 31), datetime(2024, 2, 2), "TP"),  # half covered
        ("A2", datetime(2024, 1, 5), datetime(2024, 1, 6), "FP"),  # before A2's data
        ("A3", datetime(2024, 1, 5), datetime(2024, 1, 6), "TP"),  # no parquet files
        ("A9", datetime(2024, 1, 5), datetime(2024, 1, 6), "FP"),  # unknown asset
    ]
    return pl.DataFrame(
        rows, schema=["asset_id", "start", "end", "class"], orient="row"
    ).with_columns(note=pl.lit("keep me"))


def test_file_catalog_lists_parsed_periods(store: Config):
    cat = file_catalog(store)
    assert cat["asset_id"].to_list() == ["A1", "A1", "A2"]
    assert cat["source"].to_list() == ["flow", "sensor", "sensor"]
    assert cat.row(0)[3:] == (datetime(2024, 1, 1), datetime(2024, 1, 15))
    # Restricting to assets skips unknown ones instead of raising.
    assert file_catalog(store, ["A2", "A9"])["asset_id"].to_list() == ["A2"]
    assert file_catalog(store.data_root / "missing").is_empty()


def test_availability_from_file_names(store: Config):
    out = data_availability(_labels(), store)
    assert out.columns[:5] == ["asset_id", "start", "end", "class", "note"]
    assert out["data_available"].to_list() == [True, True, True, False, False, False]
    assert out["data_reason"].to_list()[3:] == [
        REASON_NO_OVERLAP,
        REASON_NO_FILES,
        REASON_NO_FOLDER,
    ]
    assert out["data_n_files"].to_list() == [2, 1, 1, 0, 0, 0]
    assert out["data_sources"].to_list()[0] == ["flow", "sensor"]
    # The sensor file's name claims all of Jan 31 (whole end day): 1 of 2 days.
    assert out["data_coverage"].to_list()[:3] == pytest.approx([1.0, 1.0, 0.5])


def test_rows_check_catches_gaps_inside_files(store: Config):
    out = data_availability(_labels(), store, check="rows")
    assert out["data_available"].to_list() == [True, False, False, False, False, False]
    assert out["data_reason"][1] == REASON_NO_ROWS
    # Jan 5 00:00 .. Jan 6 00:00 inclusive: 25 hourly stamps per source.
    assert out["data_n_rows"][0] == 50
    assert out["data_first_ts"][0] == datetime(2024, 1, 5)
    assert out["data_last_ts"][0] == datetime(2024, 1, 6)


def test_filter_available_keeps_loadable_rows_and_warns(store: Config):
    labels = _labels()
    with pytest.warns(UserWarning, match="dropped 3 of 6"):
        kept = filter_available(labels, store)
    assert kept.columns == labels.columns
    assert kept["class"].to_list() == ["TP", "FP", "TP"]
    # Every kept event builds; the unfiltered table does not.
    assert len(build(kept, store)) == 3
    with pytest.raises(FileNotFoundError):
        build(labels, store)


def test_filter_available_min_coverage(store: Config):
    with pytest.warns(UserWarning, match=REASON_LOW_COVERAGE):
        kept = filter_available(_labels(), store, min_coverage=1.0)
    assert kept.height == 2


def test_overlapping_sources_count_once_toward_coverage(tmp_path: Path):
    cfg = Config(data_root=tmp_path)
    folder = cfg.asset_dir("A1")
    folder.mkdir()
    _write(folder, "a_240101_240110.parquet", datetime(2024, 1, 1), datetime(2024, 1, 10))
    _write(folder, "b_240105_240110.parquet", datetime(2024, 1, 5), datetime(2024, 1, 10))
    labels = pl.DataFrame(
        {"asset_id": ["A1"], "start": [datetime(2024, 1, 6)], "end": [datetime(2024, 1, 16)]}
    )
    out = data_availability(labels, cfg)
    assert out["data_coverage"][0] == pytest.approx(0.5)  # Jan 6..11 of Jan 6..16


def test_dataset_facade_and_quiet_mode(store: Config):
    ds = Dataset(store)
    assert ds.availability(_labels())["data_available"].sum() == 3
    with pytest.warns(UserWarning):
        assert ds.available(_labels()).height == 3
    labels = _labels().filter(pl.col("asset_id") == "A1")
    assert filter_available(labels, store, warn=False).height == 3


def test_dataset_lazy_and_channels_span_sources(store: Config):
    """Regression: both used to read the folder as one schema, which broke
    (lazy) or under-reported (channels) as soon as an asset had two sources."""
    ds = Dataset(store)
    assert set(ds.channels("A1")) == {"timestamp", "sensor", "flow"}
    df = ds.lazy("A1").collect()
    assert df.columns == ["timestamp", "flow", "sensor"]
    assert df["timestamp"].is_sorted()


def test_input_validation(store: Config):
    with pytest.raises(ValueError, match="not a datetime"):
        data_availability(_labels().with_columns(pl.col("start").cast(pl.String)), store)
    with pytest.raises(ValueError, match="missing columns"):
        data_availability(_labels().drop("end"), store)
    with pytest.raises(ValueError, match="check must be"):
        data_availability(_labels(), store, check="bytes")
    with pytest.raises(ValueError, match="already has"):
        data_availability(_labels().with_columns(data_n_files=pl.lit(1)), store)


def test_zero_length_window_and_empty_table(store: Config):
    t = datetime(2024, 1, 5, 12)
    point = pl.DataFrame({"asset_id": ["A1"], "start": [t], "end": [t]})
    out = data_availability(point, store, check="rows")
    assert out["data_coverage"][0] == 1.0
    assert out["data_n_rows"][0] == 2  # one stamp per source
    empty = point.clear()
    assert data_availability(empty, store, check="rows").height == 0
    assert filter_available(empty, store).height == 0


def test_date_typed_labels(store: Config):
    labels = pl.DataFrame(
        {
            "asset_id": ["A1"],
            "start": [datetime(2024, 1, 5).date()],
            "end": [(datetime(2024, 1, 5) + timedelta(days=1)).date()],
        }
    )
    assert data_availability(labels, store)["data_available"][0]

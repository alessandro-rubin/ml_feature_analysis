"""The example store behaves like the real thing the template expects."""

from pathlib import Path

import polars as pl
import pytest

from tessa import Config, Dataset
from tessa.examples import make_example_dataset
from tessa.labels import ExcelLabelSource


def test_example_dataset_round_trip(tmp_path: Path):
    ex = make_example_dataset(tmp_path, events_per_class=3)
    cfg = Config(data_root=ex.data_root)
    ds = Dataset(cfg)
    assert ds.assets == ["A01", "A02", "A03", "A04"]
    assert set(ds.channels("A01")) == {"timestamp", "temperature", "vibration", "pressure", "flow"}

    labels = ExcelLabelSource(ex.labels_path, sheet="labels", column_map=ex.column_map).load(cfg)
    assert {"asset_id", "start", "end", "class", "replacement_type"} <= set(labels.columns)
    assert labels.height == 4 * 2 * 3 + 2

    with pytest.warns(UserWarning, match="dropped 2 of 26"):
        kept = ds.available(labels, check="rows")
    assert kept.height == 24
    assert kept.group_by("class").len().sort("class")["len"].to_list() == [12, 12]
    event = next(iter(ds.events(kept).values())).collect()
    assert event.height == 4 * 60  # 1-minute grid; 5-minute source joins onto it
    assert event["pressure"].null_count() == 4 * 60 - 48


def test_example_dataset_overwrites(tmp_path: Path):
    make_example_dataset(tmp_path, seed=1)
    ex = make_example_dataset(tmp_path, seed=2)
    files = sorted(p.name for p in (ex.data_root / "A01").iterdir())
    assert len(files) == 4 and all(f.endswith(".parquet") for f in files)
    assert pl.read_excel(ex.labels_path).height == 4 * 2 * 12 + 2

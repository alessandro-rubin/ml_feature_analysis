"""Step-4: Dataset/WindowSpec facade, AnalysisResult, ResultStore, Run."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tessa import (
    AnalysisResult,
    Config,
    Dataset,
    ResultStore,
    Run,
    WindowSpec,
    materialize,
)
from tessa.features import FeatureRegistry
from tessa.features.aggregates import AggregatorRegistry, aggregate


@pytest.fixture
def fake_data(tmp_path: Path) -> Config:
    cfg = Config(data_root=tmp_path / "data", output_dir=tmp_path / "out")
    for asset in ("A1", "A2"):
        folder = cfg.asset_dir(asset)
        folder.mkdir(parents=True)
        t0 = datetime(2024, 1, 1)
        n = 24 * 31
        ts = pl.datetime_range(t0, t0 + timedelta(hours=n - 1), "1h", eager=True)
        pl.DataFrame(
            {
                cfg.timestamp_col: ts,
                "x": list(range(n)),
                "y": [i * 0.5 for i in range(n)],
            }
        ).write_parquet(folder / f"{asset}_20240101_20240131.parquet")
    return cfg


def _registries():
    fr = FeatureRegistry()
    ar = AggregatorRegistry()
    aggregate("mean", registry=ar)(lambda c: pl.col(c).mean())
    aggregate("max", registry=ar)(lambda c: pl.col(c).max())
    return fr, ar


# ── Dataset facade ──────────────────────────────────────────────────────────


def test_dataset_assets_and_channels(fake_data: Config):
    ds = Dataset(fake_data)
    assert ds.assets == ["A1", "A2"]
    assert set(ds.channels("A1")) == {"timestamp", "x", "y"}


def test_dataset_lazy_slice(fake_data: Config):
    ds = Dataset(fake_data)
    df = ds.lazy("A1", datetime(2024, 1, 10), datetime(2024, 1, 12)).collect()
    assert df["timestamp"].min() >= datetime(2024, 1, 10)
    df_all = ds.lazy("A1").collect()
    assert df_all.height == 24 * 31


def test_dataset_events(fake_data: Config):
    ds = Dataset(fake_data)
    labels = pl.DataFrame(
        {
            "asset_id": ["A1", "A2"],
            "start": [datetime(2024, 1, 5), datetime(2024, 1, 8)],
            "end": [datetime(2024, 1, 7), datetime(2024, 1, 10)],
            "class": ["TP", "FP"],
        }
    )
    events = ds.events(labels)
    assert len(events) == 2


# ── WindowSpec + materialize ────────────────────────────────────────────────


def test_materialize_event_spec(fake_data: Config):
    ds = Dataset(fake_data)
    labels = pl.DataFrame(
        {
            "asset_id": ["A1", "A2"],
            "start": [datetime(2024, 1, 5), datetime(2024, 1, 8)],
            "end": [datetime(2024, 1, 7), datetime(2024, 1, 10)],
            "class": ["TP", "FP"],
        }
    )
    fr, ar = _registries()
    table = materialize(
        ds.events(labels),
        WindowSpec.event(),
        fake_data,
        aggregators=["mean", "max"],
        feature_names=[],
        feature_registry=fr,
        aggregator_registry=ar,
    )
    assert table.height == 2
    assert "x__mean" in table.columns and "class" in table.columns


def test_materialize_tumbling_spec(fake_data: Config):
    ds = Dataset(fake_data)
    labels = pl.DataFrame(
        {
            "asset_id": ["A1"],
            "start": [datetime(2024, 1, 5, 0)],
            "end": [datetime(2024, 1, 5, 23)],
            "class": ["TP"],
        }
    )
    fr, ar = _registries()
    table = materialize(
        ds.events(labels),
        WindowSpec.tumbling("6h"),
        fake_data,
        sources=["x"],
        aggregators=["mean"],
        feature_names=[],
        feature_registry=fr,
        aggregator_registry=ar,
    )
    assert table.height == 4  # 24h / 6h
    assert "x__mean" in table.columns


def test_windowspec_requires_every():
    with pytest.raises(ValueError, match="every"):
        materialize(pl.LazyFrame({"a": [1]}), WindowSpec(kind="tumbling"), Config())


# ── AnalysisResult ──────────────────────────────────────────────────────────


def test_from_raw_buckets():
    import pandas as pd

    raw = {
        "table": pd.DataFrame({"a": [1, 2]}),
        "scores": np.array([1.0, 2.0]),
        "accuracy": 0.9,
        "class_names": ["x", "y"],
        "model": object(),
    }
    res = AnalysisResult.from_raw("demo", raw)
    assert set(res.frames) == {"table"}
    assert set(res.arrays) == {"scores"}
    assert res.scalars["accuracy"] == 0.9
    assert res.scalars["class_names"] == ["x", "y"]
    assert set(res.objects) == {"model"}
    assert "demo" in res.summary()


# ── Run + ResultStore round trip ────────────────────────────────────────────


def _labeled_table(n_per_class: int = 40, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cls, shift in (("healthy", 0.0), ("fault", 3.0)):
        for i in range(n_per_class):
            rows.append(
                {
                    "event_id": f"{cls}_{i}",
                    "class": cls,
                    "f_a": float(rng.normal(shift, 1.0)),
                    "f_b": float(rng.normal(0, 1.0)),
                }
            )
    return pl.DataFrame(rows)


def test_run_facade_and_store_roundtrip(tmp_path: Path):
    run = Run(_labeled_table(), target_col="class", cfg=Config(random_state=7))

    imp = run.importance(permutation_repeats=2, rf_params={"n_estimators": 20, "n_jobs": 1})
    assert "table" in imp.frames
    # cached: second call without kwargs returns the same underlying result
    assert run.importance().frames["table"] is imp.frames["table"]

    sep = run.separability(n_permutations=30, rf_params={"n_estimators": 30, "n_jobs": 1})
    assert sep.frames["summary"]["verdict"][0] == "separable"

    ano = run.anomaly(iforest_params={"n_estimators": 30, "n_jobs": 1})
    assert "ensemble" in ano.frames["scores"].columns

    run_dir = run.save(tmp_path / "runs", name="test_run")
    store = ResultStore(tmp_path / "runs")
    assert store.runs() == ["test_run"]

    manifest = store.load_manifest("test_run")
    assert manifest["config"]["random_state"] == 7
    assert manifest["target_col"] == "class"
    assert manifest["data_fingerprint"]["n_rows"] == 80
    assert "scikit-learn" in manifest["versions"]

    loaded = store.load_run("test_run")
    assert set(loaded) == {"importance", "separability", "anomaly"}
    # numbers survive the round trip
    orig = sep.frames["summary"]["perm_p_value"][0]
    assert loaded["separability"].frames["summary"]["perm_p_value"][0] == orig
    # models are explicitly recorded as not serialized
    assert "models" in manifest["analyses"]["anomaly"]["not_serialized"]
    assert (run_dir / "manifest.json").exists()


def test_run_unsupervised_raises_on_supervised_call():
    df = _labeled_table().drop("class")
    run = Run(df)  # no target
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="skipped"):
        run.importance(rf_params={"n_estimators": 10})

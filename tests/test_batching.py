"""Materialisers collect events in memory-bounded batches, with unchanged results."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import tessa.features.batching as batching
import tessa.features.builtins  # noqa: F401  (registers the stock aggregators)
from tessa import Config, Dataset, WindowSpec, materialize
from tessa.features import FeatureRegistry, to_period
from tessa.features.batching import (
    BYTES_PER_VALUE,
    collect_batched,
    estimate_event_bytes,
    estimate_event_rows,
    events_per_batch,
    memory_budget_bytes,
    plan_batches,
)
from tessa.features.builtins import make_first_difference, make_rolling_std

N_ASSETS = 3
EVENTS_PER_ASSET = 4


@pytest.fixture
def store(tmp_path: Path) -> tuple[Config, pl.DataFrame]:
    """3 assets, two daily 1-minute files each, 4 two-hour events per asset.

    The last event of each asset runs over midnight, across both files.
    """
    cfg = Config(data_root=tmp_path)
    rows = []
    for a in range(N_ASSETS):
        folder = tmp_path / f"A{a}"
        folder.mkdir()
        for day in (1, 2):
            t0 = datetime(2024, 1, day)
            ts = pl.datetime_range(t0, t0 + timedelta(days=1, minutes=-1), "1m", eager=True)
            pl.DataFrame(
                {"timestamp": ts, "x": [float(i % 97) for i in range(len(ts))], "y": a}
            ).write_parquet(folder / f"sensor_{t0:%Y%m%d}_{t0:%Y%m%d}.parquet")
        for k in range(EVENTS_PER_ASSET):
            start = datetime(2024, 1, 1, 5) + timedelta(hours=6 * k)
            rows.append(
                {
                    "asset_id": f"A{a}",
                    "start": start,
                    "end": start + timedelta(hours=2),
                    "class": "TP" if k % 2 else "FP",
                }
            )
    return cfg, pl.DataFrame(rows)


def _features() -> FeatureRegistry:
    reg = FeatureRegistry()
    make_first_difference("x", registry=reg)
    make_rolling_std("x", 10, registry=reg)
    return reg


def _record_batches(monkeypatch) -> list[int]:
    """Make pl.collect_all record the size of every batch it is given."""
    sizes: list[int] = []
    real = pl.collect_all

    def spy(queries, *args, **kwargs):
        sizes.append(len(queries))
        return real(queries, *args, **kwargs)

    monkeypatch.setattr(batching.pl, "collect_all", spy)
    return sizes


# ── Settings ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (1000, 1000),
        ("8GB", 8 * 2**30),
        ("512 MB", 512 * 2**20),
        ("1.5GiB", int(1.5 * 2**30)),
        ("64k", 64 * 2**10),
    ],
)
def test_memory_budget_parses_sizes(value, expected):
    assert memory_budget_bytes(value) == expected


@pytest.mark.parametrize("value", ["lots", "8 parsecs", "-1GB", 0])
def test_memory_budget_rejects_nonsense(value):
    with pytest.raises(ValueError):
        memory_budget_bytes(value)


def test_memory_budget_auto_is_half_the_machine(monkeypatch):
    monkeypatch.setattr(batching, "_physical_memory", lambda: 16 * 2**30)
    assert memory_budget_bytes("auto") == 8 * 2**30
    monkeypatch.setattr(batching, "_physical_memory", lambda: None)
    assert memory_budget_bytes("auto") is None  # unknown machine: no cap


def test_events_per_batch_defaults_to_the_thread_pool():
    assert events_per_batch(None) == pl.thread_pool_size()
    assert events_per_batch(3) == 3
    with pytest.raises(ValueError):
        events_per_batch(0)


# ── Planning ─────────────────────────────────────────────────────────────────


def test_plan_batches_respects_count_and_budget():
    assert plan_batches([0] * 5, max_events=2, budget=None) == [[0, 1], [2, 3], [4]]
    # The 5 and the 9 exceed the budget of 4 alone: each gets its own batch.
    assert plan_batches([1, 5, 1, 1, 9, 1], max_events=10, budget=4) == [
        [0],
        [1],
        [2, 3],
        [4],
        [5],
    ]
    assert plan_batches([], max_events=4, budget=None) == []


def test_collect_batched_keeps_order(monkeypatch):
    sizes = _record_batches(monkeypatch)
    queries = [pl.LazyFrame({"i": [i]}) for i in range(7)]
    out = collect_batched(queries, Config(events_per_batch=3))
    assert [df["i"][0] for df in out] == list(range(7))
    assert sizes == [3, 3, 1]


def test_collect_batched_warns_about_events_over_budget(monkeypatch):
    sizes = _record_batches(monkeypatch)
    queries = [pl.LazyFrame({"i": [i]}) for i in range(3)]
    cfg = Config(memory_budget="1MB")
    with pytest.warns(UserWarning, match=r"1 event\(s\) are estimated above.*largest: big"):
        collect_batched(queries, cfg, sizes=[10, 5 * 2**20, 10], names=["a", "big", "c"])
    assert sizes == [1, 1, 1]


# ── Sizing ───────────────────────────────────────────────────────────────────


def test_events_record_their_windows(store):
    cfg, labels = store
    events = Dataset(cfg).events(labels)
    first = next(iter(events))
    assert events.windows[first] == ("A0", labels["start"][0], labels["end"][0])
    sub = events.subset([first])
    assert list(sub.windows) == [first] and sub.label_cols == events.label_cols


def test_estimate_event_rows_matches_the_data(store):
    cfg, labels = store
    events = Dataset(cfg).events(labels)
    estimated = estimate_event_rows(events.windows, cfg)
    actual = {k: lf.select(pl.len()).collect().item() for k, lf in events.items()}
    for k in events:
        assert estimated[k] == pytest.approx(actual[k], rel=0.02)
    # A window spanning the midnight file boundary draws on both files.
    spanning = next(k for k, (_, s, e) in events.windows.items() if s.day != e.day)
    assert estimated[spanning] == pytest.approx(actual[spanning], rel=0.02)


def test_estimate_event_bytes_needs_windows(store):
    cfg, labels = store
    events = Dataset(cfg).events(labels)
    sizes = estimate_event_bytes(events, cfg, n_columns=4)
    rows = estimate_event_rows(events.windows, cfg)
    assert sizes == [rows[k] * 4 * BYTES_PER_VALUE for k in events]
    assert estimate_event_bytes(dict(events), cfg, n_columns=4) is None  # plain dict


# ── Materialisers ────────────────────────────────────────────────────────────


def test_to_period_is_independent_of_batching(store, monkeypatch):
    cfg, labels = store
    events = Dataset(cfg).events(labels)
    reference = to_period(events, cfg, aggregators=["mean", "std"], feature_registry=_features())

    sizes = _record_batches(monkeypatch)
    small = Config(data_root=cfg.data_root, events_per_batch=5, memory_budget=None)
    batched = to_period(events, small, aggregators=["mean", "std"], feature_registry=_features())
    assert batched.equals(reference)
    assert sizes == [5, 5, 2]


def test_to_period_batches_by_estimated_size(store, monkeypatch):
    cfg, labels = store
    events = Dataset(cfg).events(labels)
    reference = to_period(events, cfg, aggregators=["mean"], feature_registry=_features())

    # Each event: ~120 rows x 5 columns (x, y, diff1, roll_std, timestamp).
    per_event = 120 * 5 * BYTES_PER_VALUE
    sizes = _record_batches(monkeypatch)
    tight = Config(
        data_root=cfg.data_root, events_per_batch=100, memory_budget=int(per_event * 3.5)
    )
    out = to_period(events, tight, aggregators=["mean"], feature_registry=_features())
    assert out.equals(reference)
    assert sizes == [3, 3, 3, 3]


def test_to_period_warns_when_one_event_exceeds_the_budget(store):
    cfg, labels = store
    events = Dataset(cfg).events(labels.head(2))
    tiny = Config(data_root=cfg.data_root, memory_budget=1024)
    with pytest.warns(UserWarning, match="memory_budget=0 MB"):
        out = to_period(events, tiny, aggregators=["mean"], feature_registry=FeatureRegistry())
    assert out.height == 2


def test_windowed_materialize_is_independent_of_batching(store, monkeypatch):
    cfg, labels = store
    events = Dataset(cfg).events(labels)
    spec = WindowSpec.tumbling("30m")
    reference = materialize(events, spec, cfg, aggregators=["mean"], feature_registry=_features())

    sizes = _record_batches(monkeypatch)
    small = Config(data_root=cfg.data_root, events_per_batch=4)
    out = materialize(events, spec, small, aggregators=["mean"], feature_registry=_features())
    assert out.equals(reference)
    assert sizes == [4, 4, 4]


def test_availability_rows_check_is_batched(store, monkeypatch):
    cfg, labels = store
    reference = Dataset(cfg).availability(labels, check="rows")
    sizes = _record_batches(monkeypatch)
    small = Config(data_root=cfg.data_root, events_per_batch=5)
    assert Dataset(small).availability(labels, check="rows").equals(reference)
    assert sizes == [5, 5, 2]

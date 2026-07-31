"""Step-2: single-query `to_period` must match the old per-event loop."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from tessa import Config
from tessa.features import FeatureRegistry, feature, to_per_sample, to_period
from tessa.features.aggregates import AggregatorRegistry, aggregate
from tessa.features.materialize import _label_cols


def _reference_to_period_loop(items, cfg, sources, aggregators, feature_names, fr, ar):
    """The pre-step-2 implementation: one collect per event."""
    rows = []
    for lf in items:
        base = to_per_sample(lf, cfg, feature_names, fr)
        schema = base.collect_schema()
        label_cols = _label_cols(base, cfg)
        srcs = sources
        if srcs is None:
            srcs = [
                c
                for c in schema.names()
                if c != cfg.timestamp_col and c not in label_cols and schema[c].is_numeric()
            ]
        agg_exprs = [ar.get(a).apply(s) for s in srcs for a in aggregators]
        label_exprs = [pl.col(c).first().alias(c) for c in label_cols]
        rows.append(base.select(label_exprs + agg_exprs).collect())
    return pl.concat(rows, how="vertical_relaxed") if rows else pl.DataFrame()


def _registries():
    fr = FeatureRegistry()
    ar = AggregatorRegistry()

    @feature("x_diff", deps=("x",), registry=fr)
    def _():
        return pl.col("x").diff()

    @feature("x_roll", deps=("x",), registry=fr)
    def _():
        return pl.col("x").rolling_mean(window_size=5)

    for name, fn in [
        ("mean", lambda c: pl.col(c).mean()),
        ("std", lambda c: pl.col(c).std()),
        ("min", lambda c: pl.col(c).min()),
        ("max", lambda c: pl.col(c).max()),
    ]:
        aggregate(name, registry=ar)(fn)
    return fr, ar


def _events(n_events: int = 7, n_rows: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = {}
    t0 = datetime(2024, 1, 1)
    for i in range(n_events):
        ts = [t0 + timedelta(days=i, seconds=s) for s in range(n_rows)]
        out[f"e{i}"] = pl.LazyFrame(
            {
                "timestamp": ts,
                "x": rng.normal(i, 1.0, n_rows),
                "y": rng.normal(0.0, 2.0, n_rows),
                "event_id": [f"e{i}"] * n_rows,
                "asset_id": [f"A{i % 3}"] * n_rows,
                "class": ["TP" if i % 2 else "FP"] * n_rows,
            }
        )
    return out


def test_single_query_matches_per_event_loop():
    cfg = Config()
    fr, ar = _registries()
    events = _events()
    aggs = ["mean", "std", "min", "max"]

    new = to_period(events, cfg, aggregators=aggs, feature_registry=fr, aggregator_registry=ar)
    old = _reference_to_period_loop(list(events.values()), cfg, None, aggs, None, fr, ar)
    assert_frame_equal(
        new.select(sorted(new.columns)),
        old.select(sorted(old.columns)),
        check_dtypes=False,
    )


def test_rolling_features_do_not_bleed_across_events():
    cfg = Config()
    fr, ar = _registries()
    events = _events(n_events=3, n_rows=50)
    out = to_period(
        events,
        cfg,
        sources=["x_diff"],
        aggregators=["mean"],
        feature_registry=fr,
        aggregator_registry=ar,
    )
    # diff within each event: mean of diffs == (last - first) / (n - 1).
    # If events bled together, event boundaries would inject huge diffs.
    frames = {k: v.collect() for k, v in events.items()}
    for row, (eid, df) in zip(out.iter_rows(named=True), frames.items()):
        x = df["x"].to_numpy()
        expected = (x[-1] - x[0]) / (len(x) - 1)
        assert row["event_id"] == eid
        assert abs(row["x_diff__mean"] - expected) < 1e-9


def test_event_order_is_preserved():
    cfg = Config()
    fr, ar = _registries()
    events = _events(n_events=5)
    out = to_period(events, cfg, aggregators=["mean"], feature_registry=fr, aggregator_registry=ar)
    assert out["event_id"].to_list() == list(events.keys())


def test_empty_input_returns_empty_frame():
    cfg = Config()
    fr, ar = _registries()
    out = to_period({}, cfg, feature_registry=fr, aggregator_registry=ar)
    assert out.is_empty()

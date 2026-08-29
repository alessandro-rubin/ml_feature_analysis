"""Shared synthetic dataset for the AI-layer tests.

Built here rather than by importing ``demo.py``, which calls
``matplotlib.use("Agg")`` and ``sys.stdout.reconfigure`` at import time.

The signal is shaped so *aggregate* statistics are provably useless and a
*derived* feature is decisive. Each pair of events shares one array of vibration
values: the "calm" event keeps it in order (a slow drift) and the "jittery" event
gets a permutation of the very same numbers. Mean, min, max, std and every
quantile are therefore identical between the two by construction, and only the
temporal ordering differs — so a period-aggregate table cannot separate the
classes at all, while a rolling-std or diff feature separates them cleanly.

That is exactly the capability the AI layer exists to provide: noticing that the
stock aggregates are blind, and inventing a feature that is not.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ASSETS = ["A1", "A2", "A3"]
CLASSES = ["calm", "jittery"]
EVENTS_PER_CLASS = 4
EVENT_MINUTES = 120
SEED = 7


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """Write per-asset parquet files and return the data root."""
    root = tmp_path_factory.mktemp("synthetic_data")
    rng = np.random.default_rng(SEED)
    start = datetime(2024, 1, 1)

    for asset in ASSETS:
        folder = root / asset
        folder.mkdir(parents=True, exist_ok=True)
        n = len(CLASSES) * EVENTS_PER_CLASS * EVENT_MINUTES
        ts = pl.datetime_range(start, start + timedelta(minutes=n - 1), "1m", eager=True)

        temperature, vibration, pressure = [], [], []
        for idx in range(len(CLASSES) * EVENTS_PER_CLASS):
            label = CLASSES[idx % len(CLASSES)]
            if label == "calm":
                # A slow drift: cumulative noise, smooth from sample to sample.
                walk = 1.0 + np.cumsum(rng.normal(0, 0.12, EVENT_MINUTES))
                pending = walk
                carried = walk
            else:
                # The SAME values, reordered. Identical marginal statistics,
                # completely different temporal structure.
                pending = rng.permutation(carried)
            vibration += list(pending)

            # Distractor channels carry no class information.
            temperature += list(20.0 + rng.normal(0, 0.3, EVENT_MINUTES))
            pressure += list(5.0 + rng.normal(0, 0.5, EVENT_MINUTES))

        pl.DataFrame(
            {
                "timestamp": ts,
                "temperature": temperature,
                "vibration": vibration,
                "pressure": pressure,
            }
        ).write_parquet(folder / "sensor_20240101_20240131.parquet")
    return root


@pytest.fixture(scope="session")
def synthetic_labels(synthetic_root: Path) -> Path:
    """Write the label table next to the data and return its path."""
    start = datetime(2024, 1, 1)
    rows = []
    for asset in ASSETS:
        for idx in range(len(CLASSES) * EVENTS_PER_CLASS):
            s = start + timedelta(minutes=idx * EVENT_MINUTES)
            rows.append(
                {
                    "asset_id": asset,
                    "start": s,
                    "end": s + timedelta(minutes=EVENT_MINUTES - 1),
                    "class": CLASSES[idx % len(CLASSES)],
                    "site": "north" if asset == "A1" else "south",
                }
            )
    path = synthetic_root / "labels.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path

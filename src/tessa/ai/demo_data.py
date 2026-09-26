"""A synthetic dataset for demonstrating and testing the AI layer.

The two classes are constructed so that **aggregate statistics cannot tell them
apart**: each "jittery" event holds a permutation of the very same vibration
values as its paired "calm" event, so their mean, std, min, max and every
quantile are identical. Only the temporal ordering differs.

That makes it a fair test of the thing this layer is for. A stock
period-aggregate table genuinely cannot separate the classes; a feature that
looks at sample-to-sample movement separates them cleanly. The model has to
notice the aggregates are blind and invent something that is not.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

__all__ = ["ASSETS", "CLASSES", "generate"]

ASSETS = ["A1", "A2", "A3"]
CLASSES = ["calm", "jittery"]
EVENTS_PER_CLASS = 4
EVENT_MINUTES = 120
SEED = 7


def generate(
    root: str | Path,
    *,
    assets: list[str] | None = None,
    events_per_class: int = EVENTS_PER_CLASS,
    event_minutes: int = EVENT_MINUTES,
    seed: int = SEED,
    labels_name: str = "labels.parquet",
) -> tuple[Path, Path]:
    """Write per-asset parquet files and a label table.

    Returns
    -------
    tuple of Path
        ``(data_root, labels_path)``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    assets = assets or ASSETS
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    n_events = len(CLASSES) * events_per_class

    for asset in assets:
        folder = root / asset
        folder.mkdir(parents=True, exist_ok=True)
        n = n_events * event_minutes
        ts = pl.datetime_range(start, start + timedelta(minutes=n - 1), "1m", eager=True)

        temperature, vibration, pressure = [], [], []
        carried = np.zeros(event_minutes)
        for idx in range(n_events):
            if CLASSES[idx % len(CLASSES)] == "calm":
                # A slow drift: cumulative noise, smooth sample to sample.
                carried = 1.0 + np.cumsum(rng.normal(0, 0.12, event_minutes))
                values = carried
            else:
                # The SAME values reordered: identical marginals, different order.
                values = rng.permutation(carried)
            vibration += list(values)
            # Distractor channels carry no class information.
            temperature += list(20.0 + rng.normal(0, 0.3, event_minutes))
            pressure += list(5.0 + rng.normal(0, 0.5, event_minutes))

        pl.DataFrame(
            {
                "timestamp": ts,
                "temperature": temperature,
                "vibration": vibration,
                "pressure": pressure,
            }
        ).write_parquet(folder / "sensor_20240101_20240131.parquet")

    rows = []
    for asset in assets:
        for idx in range(n_events):
            s = start + timedelta(minutes=idx * event_minutes)
            rows.append(
                {
                    "asset_id": asset,
                    "start": s,
                    "end": s + timedelta(minutes=event_minutes - 1),
                    "class": CLASSES[idx % len(CLASSES)],
                    "site": "north" if asset == assets[0] else "south",
                }
            )
    labels_path = root / labels_name
    pl.DataFrame(rows).write_parquet(labels_path)
    return root, labels_path

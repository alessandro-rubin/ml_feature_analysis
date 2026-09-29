"""A small, realistic-looking data store for tutorials and the analysis template.

:func:`make_example_dataset` writes what a real project hands you:

- ``<root>/data/<asset>/`` holding two sources at different sampling rates and
  split into monthly files, in the layout :mod:`asset_loader` expects:
  ``sensor_<YYYYMMDD>_<YYYYMMDD>.parquet`` (1-minute temperature, vibration)
  and ``process_<YYMMDD>_<YYMMDD>.parquet`` (5-minute pressure, flow);
- ``<root>/labels.xlsx``, a hand-maintained label sheet with spreadsheet-style
  headers ("Asset", "Start time", ...) that need a ``column_map``, and a few
  rows the store cannot serve (an asset that was never exported, an event
  after the export ends), as real label sheets always have.

TP events carry a slow temperature ramp, often with a small vibration increase;
FP events a short temperature spike, some with a weak ramp too. The classes
separate well but not perfectly, mostly on the temperature dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

__all__ = ["COLUMN_MAP", "ExampleDataset", "make_example_dataset"]

COLUMN_MAP = {
    "Asset": "asset_id",
    "Start time": "start",
    "End time": "end",
    "Label": "class",
    "Replacement type": "replacement_type",
}
"""Excel header -> tessa column name, for :class:`tessa.labels.ExcelLabelSource`."""

_ASSETS = ["A01", "A02", "A03", "A04"]
_START = datetime(2024, 1, 1)
_MONTHS = [
    (datetime(2024, 1, 1), datetime(2024, 1, 31)),
    (datetime(2024, 2, 1), datetime(2024, 2, 29)),
]
_EVENT = timedelta(hours=4)


@dataclass(frozen=True)
class ExampleDataset:
    """Paths written by :func:`make_example_dataset`."""

    data_root: Path
    labels_path: Path
    column_map: dict[str, str]


def make_example_dataset(
    root: str | Path,
    *,
    events_per_class: int = 12,
    seed: int = 0,
) -> ExampleDataset:
    """Write the example store under ``root`` (overwriting earlier output).

    Parameters
    ----------
    root : str or Path
        Target directory; ``data/`` and ``labels.xlsx`` are created inside.
    events_per_class : int, default 12
        TP and FP events per asset.
    seed : int, default 0
        Seed for the noise and the event placement.

    Returns
    -------
    ExampleDataset
        ``data_root``, ``labels_path`` and the ``column_map`` the sheet needs.
    """
    root = Path(root)
    data_root = root / "data"
    rng = np.random.default_rng(seed)
    label_rows: list[dict] = []

    for asset in _ASSETS:
        folder = data_root / asset
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.parquet"):
            old.unlink()

        end = _MONTHS[-1][1] + timedelta(days=1) - timedelta(minutes=1)
        ts = pl.datetime_range(_START, end, "1m", eager=True)
        n = len(ts)
        temperature = 60 + 2 * np.sin(np.arange(n) * 2 * np.pi / 1440) + rng.normal(0, 0.8, n)
        vibration = 1.0 + rng.normal(0, 0.15, n)
        pressure = 5 + rng.normal(0, 0.1, n)
        flow = 120 + rng.normal(0, 3, n)

        # Non-overlapping event slots, drawn without replacement.
        n_slots = n // int(_EVENT.total_seconds() // 60)
        slots = rng.choice(np.arange(1, n_slots - 1), size=2 * events_per_class, replace=False)
        classes = ["TP"] * events_per_class + ["FP"] * events_per_class
        width = int(_EVENT.total_seconds() // 60)
        for slot, cls in zip(slots, classes):
            i0 = int(slot) * width
            t = np.linspace(0, 1, width)
            # Amplitudes vary per event and some FPs carry a weak ramp too, so
            # the classes overlap: a perfect score would teach nothing.
            if cls == "TP":
                temperature[i0 : i0 + width] += rng.uniform(1.0, 4.0) * t
                vibration[i0 : i0 + width] += rng.uniform(0.0, 0.12) * t
            else:
                spike = slice(i0 + width // 3, i0 + width // 3 + 20)
                temperature[spike] += rng.uniform(1.5, 4.0)
                if rng.random() < 0.3:
                    temperature[i0 : i0 + width] += rng.uniform(0.5, 1.5) * t
            label_rows.append(
                {
                    "Asset": asset,
                    "Start time": ts[i0],
                    "End time": ts[i0 + width - 1],
                    "Label": cls,
                    "Replacement type": str(rng.choice(["bearing", "seal"])),
                    "Comment": "",
                }
            )

        frame = pl.DataFrame(
            {
                "timestamp": ts,
                "temperature": temperature,
                "vibration": vibration,
                "pressure": pressure,
                "flow": flow,
            }
        )
        for m_start, m_end in _MONTHS:
            month = frame.filter(
                pl.col("timestamp").dt.date().is_between(m_start.date(), m_end.date())
            )
            month.select("timestamp", "temperature", "vibration").write_parquet(
                folder / f"sensor_{m_start:%Y%m%d}_{m_end:%Y%m%d}.parquet"
            )
            month.gather_every(5).select("timestamp", "pressure", "flow").write_parquet(
                folder / f"process_{m_start:%y%m%d}_{m_end:%y%m%d}.parquet"
            )

    # What real label sheets contain besides loadable events.
    for asset, start, comment in [
        ("A07", datetime(2024, 1, 10, 8), "asset never exported"),
        ("A02", datetime(2024, 3, 12, 14), "after the data export ends"),
    ]:
        label_rows.append(
            {
                "Asset": asset,
                "Start time": start,
                "End time": start + _EVENT,
                "Label": "TP",
                "Replacement type": "bearing",
                "Comment": comment,
            }
        )

    labels = pl.DataFrame(label_rows).sort("Asset", "Start time")
    labels_path = root / "labels.xlsx"
    labels.write_excel(labels_path, worksheet="labels", autofit=True)
    return ExampleDataset(data_root=data_root, labels_path=labels_path, column_map=dict(COLUMN_MAP))

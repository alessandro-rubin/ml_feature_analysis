"""Build per-event LazyFrames from a label table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import polars as pl

from ml_analysis.config import Config
from ml_analysis.dataset.loader import load_event


@dataclass(frozen=True)
class Event:
    event_id: str
    asset_id: str
    start: datetime
    end: datetime
    label: dict  # class + any extras

    def attach_to(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        cols = [
            pl.lit(self.event_id).alias("event_id"),
            pl.lit(self.asset_id).alias("asset_id"),
        ]
        for k, v in self.label.items():
            cols.append(pl.lit(v).alias(k))
        return lf.with_columns(cols)


def _event_id(asset_id: str, start: datetime, end: datetime, idx: int) -> str:
    return f"{asset_id}_{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}_{idx}"


def iter_events(
    labels: pl.DataFrame,
    cfg: Config,
) -> Iterable[Event]:
    """Yield Event rows from a label table.

    Required columns: cfg.asset_col, "start", "end", cfg.class_col.
    Any other columns are carried in `label` as extras.
    """
    required = {cfg.asset_col, "start", "end", cfg.class_col}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Label table missing columns: {missing}")

    extras_cols = [c for c in labels.columns if c not in {cfg.asset_col, "start", "end"}]
    for i, row in enumerate(labels.iter_rows(named=True)):
        yield Event(
            event_id=_event_id(row[cfg.asset_col], row["start"], row["end"], i),
            asset_id=row[cfg.asset_col],
            start=row["start"],
            end=row["end"],
            label={c: row[c] for c in extras_cols},
        )


def build(
    labels: pl.DataFrame,
    cfg: Config,
    columns: list[str] | None = None,
) -> dict[str, pl.LazyFrame]:
    """Return {event_id: LazyFrame} with label metadata attached as columns."""
    out: dict[str, pl.LazyFrame] = {}
    for ev in iter_events(labels, cfg):
        lf = load_event(ev.asset_id, ev.start, ev.end, cfg, columns=columns)
        out[ev.event_id] = ev.attach_to(lf)
    return out

"""High-level, notebook-first entry point over the raw store.

`Dataset` wraps a :class:`~tessa.config.Config` and exposes the
loader/builder machinery as a small object API:

>>> ds = Dataset(Config(data_root="data"))
>>> ds.assets
['A1', 'A2', ...]
>>> lf = ds.lazy("A1")                       # whole asset, lazy
>>> lf = ds.lazy("A1", start, end)           # time slice, lazy
>>> labels = ds.available(label_table)       # drop events without data
>>> events = ds.events(labels)               # {event_id: LazyFrame}

Everything stays lazy until a materializer collects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from asset_loader import discover_files, load_event

from tessa.config import Config
from tessa.dataset.availability import (
    AvailabilityCheck,
    _filter_available,
    data_availability,
)
from tessa.dataset.builder import EventFrames, build


@dataclass
class Dataset:
    cfg: Config

    @property
    def assets(self) -> list[str]:
        """Asset ids = folders under ``data_root`` that contain parquet data."""
        root = self.cfg.data_root
        if not root.exists():
            return []
        out = []
        for p in sorted(root.iterdir()):
            if p.is_dir() and (p / self.cfg.asset_subdir).is_dir():
                out.append(p.name)
        return out

    def channels(self, asset_id: str) -> list[str]:
        """Column names available for an asset, across all its sources.

        Reads parquet schemas only, not data. Includes the timestamp column.
        """
        return self.lazy(asset_id).collect_schema().names()

    def lazy(
        self,
        asset_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Lazy frame for one asset, optionally sliced to ``[start, end]``.

        Either bound may be omitted. Sources are combined exactly as
        :func:`asset_loader.load_event` does for an event (full outer join
        on the timestamp), so a whole-asset frame and an event frame always
        agree.
        """
        return load_event(asset_id, start, end, self.cfg, columns=columns)

    def events(self, labels: pl.DataFrame, columns: list[str] | None = None) -> EventFrames:
        """Per-event LazyFrames with label metadata attached (see builder)."""
        return build(labels, self.cfg, columns=columns)

    def availability(
        self, labels: pl.DataFrame, check: AvailabilityCheck = "files"
    ) -> pl.DataFrame:
        """Label table annotated with the data found per event (see
        :func:`~tessa.dataset.availability.data_availability`)."""
        return data_availability(labels, self.cfg, check=check)

    def available(
        self,
        labels: pl.DataFrame,
        check: AvailabilityCheck = "files",
        min_coverage: float = 0.0,
    ) -> pl.DataFrame:
        """Label rows whose events can be loaded (see
        :func:`~tessa.dataset.availability.filter_available`)."""
        return _filter_available(
            labels, self.cfg, check, min_coverage, catalog=None, warn=True, stacklevel=3
        )

    def files(self, asset_id: str, start: datetime, end: datetime) -> list:
        """Parquet files overlapping a window (thin `discover_files` wrapper)."""
        return discover_files(asset_id, start, end, self.cfg)

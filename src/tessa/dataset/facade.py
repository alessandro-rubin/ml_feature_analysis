"""High-level, notebook-first entry point over the raw store.

`Dataset` wraps a :class:`~tessa.config.Config` and exposes the
loader/builder machinery as a small object API:

>>> ds = Dataset(Config(data_root="data"))
>>> ds.assets
['A1', 'A2', ...]
>>> lf = ds.lazy("A1")                       # whole asset, lazy
>>> lf = ds.lazy("A1", start, end)           # time slice, lazy
>>> events = ds.events(label_table)          # {event_id: LazyFrame}

Everything stays lazy until a materializer collects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from asset_loader import discover_files, load_event

from tessa.config import Config
from tessa.dataset.builder import build


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
        """Column names available for an asset (from the first parquet file)."""
        folder = self.cfg.asset_dir(asset_id)
        files = sorted(folder.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files under {folder}")
        return pl.scan_parquet(str(files[0])).collect_schema().names()

    def lazy(
        self,
        asset_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Lazy frame for one asset, optionally sliced to ``[start, end]``."""
        if start is not None and end is not None:
            return load_event(asset_id, start, end, self.cfg, columns=columns)
        folder = self.cfg.asset_dir(asset_id)
        files = [str(f) for f in sorted(folder.glob("*.parquet"))]
        if not files:
            raise FileNotFoundError(f"No parquet files under {folder}")
        lf = pl.scan_parquet(files)
        if columns is not None:
            lf = lf.select(list({self.cfg.timestamp_col, *columns}))
        if self.cfg.assume_sorted:
            return lf.set_sorted(self.cfg.timestamp_col)
        return lf.sort(self.cfg.timestamp_col)

    def events(
        self, labels: pl.DataFrame, columns: list[str] | None = None
    ) -> dict[str, pl.LazyFrame]:
        """Per-event LazyFrames with label metadata attached (see builder)."""
        return build(labels, self.cfg, columns=columns)

    def files(self, asset_id: str, start: datetime, end: datetime) -> list:
        """Parquet files overlapping a window (thin `discover_files` wrapper)."""
        return discover_files(asset_id, start, end, self.cfg)

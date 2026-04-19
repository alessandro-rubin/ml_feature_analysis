"""Lazy parquet loader for one (asset, time-window) event."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import polars as pl

from ml_analysis.config import Config


def _parse_filename(path: Path, pattern: re.Pattern) -> tuple[datetime, datetime] | None:
    m = pattern.match(path.name)
    if not m:
        return None
    g = m.groupdict()
    fmt = "%Y%m%d"
    return datetime.strptime(g["start"], fmt), datetime.strptime(g["end"], fmt)


def discover_files(
    asset_id: str,
    start: datetime,
    end: datetime,
    cfg: Config,
) -> list[Path]:
    """Return parquet files for `asset_id` whose filename range overlaps [start, end]."""
    folder = cfg.asset_dir(asset_id)
    if not folder.exists():
        raise FileNotFoundError(f"Asset folder not found: {folder}")

    pattern = re.compile(cfg.filename_pattern)
    out = []
    for f in sorted(folder.glob("*.parquet")):
        rng = _parse_filename(f, pattern)
        if rng is None:
            continue
        f_start, f_end = rng
        if f_end >= start and f_start <= end:
            out.append(f)
    return out


def load_event(
    asset_id: str,
    start: datetime,
    end: datetime,
    cfg: Config,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    """Lazy-load one event's raw time-series data, sliced to [start, end]."""
    files = discover_files(asset_id, start, end, cfg)
    if not files:
        raise FileNotFoundError(
            f"No parquet files for asset={asset_id} in [{start}, {end}]"
        )

    lf = pl.scan_parquet([str(f) for f in files])
    if columns is not None:
        keep = list({cfg.timestamp_col, *columns})
        lf = lf.select(keep)

    ts = pl.col(cfg.timestamp_col)
    return lf.filter((ts >= start) & (ts <= end)).sort(cfg.timestamp_col)

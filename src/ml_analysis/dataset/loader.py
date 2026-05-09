"""Lazy parquet loader for one ``(asset, time-window)`` event.

Parquet files are organised on disk as ``<data_root>/<asset_id>/<asset_subdir>/``
with each filename encoding the asset and a ``[start, end]`` date range
(see :attr:`ml_analysis.config.Config.filename_pattern`). This module
discovers the files that overlap a requested time range and returns a
:class:`polars.LazyFrame` filtered to that range — the actual scan is
deferred until the caller collects.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import polars as pl

from ml_analysis.config import Config


def _parse_filename(path: Path, pattern: re.Pattern) -> tuple[datetime, datetime] | None:
    """Extract the ``(start, end)`` range encoded in a parquet filename.

    Returns ``None`` if the filename does not match ``pattern``. The
    pattern must expose named groups ``start`` and ``end`` formatted as
    ``%Y%m%d``.
    """
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
    """Return parquet files whose filename range overlaps ``[start, end]``.

    Parameters
    ----------
    asset_id : str
        Asset folder under :attr:`Config.data_root` to search.
    start, end : datetime
        Inclusive event window.
    cfg : Config
        Project configuration. ``cfg.filename_pattern`` is compiled to
        parse filenames and ``cfg.asset_dir(asset_id)`` locates the
        parquet directory.

    Returns
    -------
    list of Path
        Matching files, sorted lexicographically (which corresponds to
        chronological order given the standard ``%Y%m%d`` naming).
        Files whose names don't match the pattern are silently skipped.

    Raises
    ------
    FileNotFoundError
        If the asset folder does not exist.
    """
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
    """Lazy-load one event's raw time-series, sliced to ``[start, end]``.

    Parameters
    ----------
    asset_id : str
        Asset folder under :attr:`Config.data_root`.
    start, end : datetime
        Inclusive event window. Rows outside this range are filtered out.
    cfg : Config
        Project configuration.
    columns : list of str, optional
        Subset of columns to load. The timestamp column is always
        included. If ``None``, all columns are loaded.

    Returns
    -------
    pl.LazyFrame
        Lazy frame sorted by timestamp, restricted to ``[start, end]``,
        and projected to ``columns`` (plus the timestamp). Note no
        ``.collect()`` is performed.

    Raises
    ------
    FileNotFoundError
        If the asset folder does not exist (raised by
        :func:`discover_files`) or if no files in the folder overlap the
        requested range.
    """
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

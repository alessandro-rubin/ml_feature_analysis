"""Lazy multi-source parquet loader for one ``(asset, time-window)`` event.

Parquet files are organised on disk as ``<data_root>/<asset_id>/`` (plus an
optional :attr:`ml_analysis.config.Config.asset_subdir`) with each filename
encoding a *source* name and a ``[start, end]`` date range, e.g.
``flow_rate_240101_240131.parquet`` (see
:attr:`ml_analysis.config.Config.filename_pattern`).

An asset's data may be split along two axes:

- **time** — files with the same source prefix hold consecutive periods of
  the *same* columns and are vertically concatenated;
- **variables** — files with different source prefixes hold *different*
  columns over the same timeline and are aligned with a full outer join on
  the timestamp column, so sources may have different sampling rates (rows
  missing from a source come out as nulls).

This module discovers the files that overlap a requested time range,
assembles them per the rules above, and returns a
:class:`polars.LazyFrame` filtered to that range — the actual scan is
deferred until the caller collects.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ml_analysis.config import Config


def _parse_filename(
    path: Path, pattern: re.Pattern
) -> tuple[str, datetime, datetime] | None:
    """Extract the ``(source, start, end)`` triple encoded in a filename.

    Returns ``None`` if the filename does not match ``pattern``. The
    pattern must expose named groups ``start`` and ``end`` (``%y%m%d``
    for 6-digit values, ``%Y%m%d`` for 8-digit). The source name is taken
    from a ``source`` group, falling back to a legacy ``asset`` group,
    falling back to ``""`` (i.e. every file in one source).
    """
    m = pattern.match(path.name)
    if not m:
        return None
    g = m.groupdict()

    def _to_dt(s: str) -> datetime:
        return datetime.strptime(s, "%y%m%d" if len(s) == 6 else "%Y%m%d")

    source = g.get("source") or g.get("asset") or ""
    return source, _to_dt(g["start"]), _to_dt(g["end"])


def discover_sources(
    asset_id: str,
    start: datetime,
    end: datetime,
    cfg: Config,
) -> dict[str, list[Path]]:
    """Group an asset's parquet files by source, keeping range overlaps.

    Parameters
    ----------
    asset_id : str
        Asset folder under :attr:`Config.data_root` to search.
    start, end : datetime
        Inclusive event window; only files whose filename range overlaps
        it are returned.
    cfg : Config
        Project configuration. ``cfg.filename_pattern`` is compiled to
        parse filenames and ``cfg.asset_dir(asset_id)`` locates the
        parquet directory.

    Returns
    -------
    dict[str, list[Path]]
        Mapping from source name (the filename prefix) to its matching
        files, each list sorted lexicographically (chronological order
        given the standard date naming). Files whose names don't match
        the pattern are silently skipped.

    Raises
    ------
    FileNotFoundError
        If the asset folder does not exist.
    """
    folder = cfg.asset_dir(asset_id)
    if not folder.exists():
        raise FileNotFoundError(f"Asset folder not found: {folder}")

    pattern = re.compile(cfg.filename_pattern)
    out: dict[str, list[Path]] = {}
    for f in sorted(folder.glob("*.parquet")):
        parsed = _parse_filename(f, pattern)
        if parsed is None:
            continue
        source, f_start, f_end = parsed
        # Filename dates have day resolution; the end date covers the
        # whole day, so a file ending 0131 still overlaps a window
        # starting at 0131 20:00.
        if f_end + timedelta(days=1) > start and f_start <= end:
            out.setdefault(source, []).append(f)
    return out


def discover_files(
    asset_id: str,
    start: datetime,
    end: datetime,
    cfg: Config,
) -> list[Path]:
    """Return all parquet files (any source) overlapping ``[start, end]``.

    Flat, sorted view of :func:`discover_sources`; kept for callers that
    only need the file list.
    """
    groups = discover_sources(asset_id, start, end, cfg)
    return sorted(p for files in groups.values() for p in files)


def load_event(
    asset_id: str,
    start: datetime,
    end: datetime,
    cfg: Config,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    """Lazy-load one event's raw time-series, sliced to ``[start, end]``.

    Files are grouped by source (filename prefix): each source is
    vertically concatenated across time, then the sources are combined
    with a full outer join on ``cfg.timestamp_col``. Sources may have
    different sampling rates; timestamps absent from a source yield
    nulls in that source's columns. Column names (other than the
    timestamp) must be unique across sources. Timestamps should be
    unique within each source — duplicated stamps would fan out through
    the join.

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
        included, and sources contributing none of the requested columns
        are skipped entirely. If ``None``, all columns of all sources
        are loaded.

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
        :func:`discover_sources`) or if no files in the folder overlap
        the requested range.
    ValueError
        If a source lacks the timestamp column, if a non-timestamp
        column appears in more than one source, or if a requested column
        exists in no source.
    """
    ts = cfg.timestamp_col
    groups = discover_sources(asset_id, start, end, cfg)
    if not groups:
        raise FileNotFoundError(
            f"No parquet files for asset={asset_id} in [{start}, {end}]"
        )

    frames: list[pl.LazyFrame] = []
    seen: dict[str, str] = {}  # column name -> source providing it
    for source, files in sorted(groups.items()):
        lf = pl.scan_parquet([str(f) for f in files])
        names = lf.collect_schema().names()
        if ts not in names:
            raise ValueError(
                f"Source {source!r} of asset {asset_id!r} has no "
                f"timestamp column {ts!r}"
            )
        value_cols = [c for c in names if c != ts]
        for c in value_cols:
            if c in seen:
                raise ValueError(
                    f"Column {c!r} appears in both source {seen[c]!r} and "
                    f"{source!r} for asset {asset_id!r}; column names must "
                    "be unique across sources"
                )
            seen[c] = source
        if columns is not None:
            value_cols = [c for c in value_cols if c in columns]
            if not value_cols:
                continue
        frames.append(lf.select([ts, *value_cols]))

    if columns is not None:
        missing = [c for c in columns if c != ts and c not in seen]
        if missing:
            raise ValueError(
                f"Columns not found in any source of asset {asset_id!r}: "
                f"{missing}"
            )
    if not frames:
        # Request reduced to the timestamp column alone.
        _, files = min(groups.items())
        frames.append(pl.scan_parquet([str(f) for f in files]).select([ts]))

    combined = frames[0]
    for lf in frames[1:]:
        combined = combined.join(lf, on=ts, how="full", coalesce=True)

    tcol = pl.col(ts)
    return combined.filter((tcol >= start) & (tcol <= end)).sort(ts)

"""Lazy multi-source parquet loader for per-asset time-series.

Parquet files are organised on disk as ``<data_root>/<asset_id>/`` (plus an
optional :attr:`asset_loader.config.LoaderConfig.asset_subdir`) with each filename
encoding a *source* name and a ``[start, end]`` date range, e.g.
``flow_rate_240101_240131.parquet`` (see
:attr:`asset_loader.config.LoaderConfig.filename_pattern`).

An asset's data may be split along two axes:

- **time** — files with the same source prefix hold consecutive periods of
  the *same* columns and are vertically concatenated;
- **variables** — files with different source prefixes hold *different*
  columns over the same timeline and are aligned with a full outer join on
  the timestamp column, so sources may have different sampling rates (rows
  missing from a source come out as nulls).

:func:`load_event` discovers the files that overlap a requested time range
(either bound may be ``None`` for "unbounded"), assembles them per the rules
above, and returns a :class:`polars.LazyFrame` filtered to that range — the
actual scan is deferred until the caller collects. :func:`load_asset` is the
one-line convenience wrapper: point it at a data root (or a
:class:`LoaderConfig`) and get the asset's history back as a collected
:class:`polars.DataFrame`.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from asset_loader.config import LoaderConfig


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
    start: datetime | None,
    end: datetime | None,
    cfg: LoaderConfig,
) -> dict[str, list[Path]]:
    """Group an asset's parquet files by source, keeping range overlaps.

    Parameters
    ----------
    asset_id : str
        Asset folder under :attr:`LoaderConfig.data_root` to search.
    start, end : datetime or None
        Inclusive window; only files whose filename range overlaps it are
        returned. Either bound may be ``None``, meaning unbounded on that
        side (``start=None, end=None`` selects the asset's full history).
    cfg : LoaderConfig
        Layout configuration. ``cfg.filename_pattern`` is compiled to
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
        if (start is None or f_end + timedelta(days=1) > start) and (
            end is None or f_start <= end
        ):
            out.setdefault(source, []).append(f)
    return out


def discover_files(
    asset_id: str,
    start: datetime | None,
    end: datetime | None,
    cfg: LoaderConfig,
) -> list[Path]:
    """Return all parquet files (any source) overlapping ``[start, end]``.

    Flat, sorted view of :func:`discover_sources`; kept for callers that
    only need the file list. ``None`` bounds mean unbounded.
    """
    groups = discover_sources(asset_id, start, end, cfg)
    return sorted(p for files in groups.values() for p in files)


def load_event(
    asset_id: str,
    start: datetime | None,
    end: datetime | None,
    cfg: LoaderConfig,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    """Lazy-load an asset's raw time-series, sliced to ``[start, end]``.

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
        Asset folder under :attr:`LoaderConfig.data_root`.
    start, end : datetime or None
        Inclusive window. Rows outside this range are filtered out.
        Either bound may be ``None``, meaning unbounded on that side;
        pass ``None, None`` for the asset's full history.
    cfg : LoaderConfig
        Layout configuration.
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
        window = "" if start is None and end is None else f" in [{start}, {end}]"
        raise FileNotFoundError(f"No parquet files for asset={asset_id}{window}")

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
    if start is not None:
        combined = combined.filter(tcol >= start)
    if end is not None:
        combined = combined.filter(tcol <= end)
    return combined.sort(ts)


def load_asset(
    asset_id: str,
    root: str | Path | LoaderConfig = "data",
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    columns: list[str] | None = None,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """One-line eager load of an asset's time-series.

    ``load_asset("A1", "path/to/data")`` returns asset A1's entire
    history — every source outer-joined on the timestamp — as a
    collected DataFrame, using the default filename convention.

    Parameters
    ----------
    asset_id : str
        Asset folder under the data root.
    root : str, Path, or LoaderConfig, default ``"data"``
        Either the data-root directory (a :class:`LoaderConfig` with
        default settings is built around it) or a full
        :class:`LoaderConfig` for non-default layouts.
    start, end : datetime, optional
        Inclusive window bounds. Omit both (the default) to load the
        asset's full history; pass just one for a half-open slice.
    columns : list of str, optional
        Subset of columns to load; the timestamp column is always
        included. If ``None``, all columns are loaded.
    lazy : bool, default False
        If True, skip the final ``.collect()`` and return the
        :class:`polars.LazyFrame` instead.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        The asset's data sorted by timestamp (lazy when ``lazy=True``).
    """
    cfg = root if isinstance(root, LoaderConfig) else LoaderConfig(data_root=Path(root))
    lf = load_event(asset_id, start, end, cfg, columns=columns)
    return lf if lazy else lf.collect()

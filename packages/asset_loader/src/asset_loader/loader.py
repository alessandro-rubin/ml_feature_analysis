"""Lazy multi-source parquet loader for per-asset time-series.

Parquet files are organised on disk as ``<data_root>/<asset_id>/`` (plus an
optional :attr:`asset_loader.config.LoaderConfig.asset_subdir`) with each filename
encoding a *source* name and a ``[start, end]`` date range, e.g.
``flow_rate_240101_240131.parquet`` (see
:attr:`asset_loader.config.LoaderConfig.filename_pattern`).

An asset's data may be split along two axes:

- **time** — files with the same source prefix hold consecutive periods of
  the *same* columns and are vertically concatenated;
- **variables** — files with different source prefixes hold columns over the
  same timeline and are combined by :func:`load_event` according to a
  ``merge`` strategy (full outer join by default, so sources may have
  different sampling rates and rows missing from a source come out as nulls).

The same column name may legitimately appear in more than one source (the
same feature recorded by two systems, a backup export, a re-processed
file). ``on_duplicate`` decides what happens then: raise, suffix each copy
with its source, coalesce them into one column, or keep a single source's
copy.

:func:`load_event` discovers the files that overlap a requested time range
(either bound may be ``None`` for "unbounded"), assembles them per the rules
above, and returns a :class:`polars.LazyFrame` filtered to that range — the
actual scan is deferred until the caller collects. :func:`load_asset` is the
one-line convenience wrapper: point it at a data root (or a
:class:`LoaderConfig`) and get the asset's history back as a collected
:class:`polars.DataFrame`.

Both loaders accept ``with_metadata=True`` to return a :class:`LoadResult`
— the frame plus a provenance dictionary recording every file read and
every operation applied to build it.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, NamedTuple, overload

import polars as pl

from asset_loader.config import LoaderConfig

MergeStrategy = Literal["outer", "left", "inner", "vertical", "asof"]
"""How sources covering the same period are combined (see :func:`load_event`)."""

DuplicatePolicy = Literal["error", "rename", "coalesce", "first", "last"]
"""What to do with a column name shared by several sources (see :func:`load_event`)."""

AsofStrategy = Literal["backward", "forward", "nearest"]
"""Direction used to match timestamps when ``merge="asof"``."""

MERGE_STRATEGIES: tuple[str, ...] = ("outer", "left", "inner", "vertical", "asof")
DUPLICATE_POLICIES: tuple[str, ...] = ("error", "rename", "coalesce", "first", "last")


class LoadResult(NamedTuple):
    """Frame plus provenance, returned when ``with_metadata=True``.

    A plain 2-tuple, so ``frame, meta = load_asset(..., with_metadata=True)``
    works as well as attribute access.

    Attributes
    ----------
    frame : pl.LazyFrame or pl.DataFrame
        The loaded data (lazy or collected, per the caller's request).
    metadata : dict
        Provenance record; see :func:`load_event` for its layout.
    """

    frame: pl.LazyFrame | pl.DataFrame
    metadata: dict[str, Any]


def _parse_filename(path: Path, pattern: re.Pattern) -> tuple[str, datetime, datetime] | None:
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
        if (start is None or f_end + timedelta(days=1) > start) and (end is None or f_start <= end):
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


def _order_sources(available: list[str], source_order: list[str] | None) -> list[str]:
    """Rank sources: ``source_order`` first, then the rest alphabetically.

    Names in ``source_order`` that the asset does not have are ignored, so
    one project-wide priority list can be reused across assets. The
    resulting order drives duplicate-column priority, the anchor of a
    ``left``/``asof`` merge, and output column order.
    """
    if not source_order:
        return sorted(available)
    listed = [s for s in source_order if s in available]
    rest = sorted(set(available) - set(listed))
    return listed + rest


def _unique_name(name: str, taken: set[str]) -> str:
    """Return ``name``, suffixed with ``_2``, ``_3``… until it is free."""
    out, i = name, 2
    while out in taken:
        out, i = f"{name}_{i}", i + 1
    return out


def _resolve_duplicates(
    per_source: list[tuple[str, list[str]]],
    policy: DuplicatePolicy,
    ts: str,
    asset_id: str,
) -> tuple[
    dict[str, dict[str, str]], dict[str, set[str]], dict[str, list[str]], dict[str, list[str]]
]:
    """Plan how to make column names unique across sources.

    Parameters
    ----------
    per_source : list of (str, list of str)
        ``(source, value columns)`` in priority order.
    policy : DuplicatePolicy
        See :func:`load_event`.
    ts, asset_id : str
        Timestamp column (reserved, never renamed) and asset name (for
        error messages).

    Returns
    -------
    renames : dict[str, dict[str, str]]
        Per source, ``{old name: new name}`` to apply before merging.
    drops : dict[str, set[str]]
        Per source, columns to discard before merging.
    coalesce_plan : dict[str, list[str]]
        ``{final name: [renamed copies in priority order]}``, merged with
        :func:`polars.coalesce` after the frames are combined.
    duplicates : dict[str, list[str]]
        ``{column: [sources providing it]}`` — the duplicates found, for
        the metadata record.

    Raises
    ------
    ValueError
        If ``policy="error"`` and a column appears in more than one source.
    """
    owners: dict[str, list[str]] = {}
    for source, cols in per_source:
        for c in cols:
            owners.setdefault(c, []).append(source)
    duplicates = {c: srcs for c, srcs in owners.items() if len(srcs) > 1}

    renames: dict[str, dict[str, str]] = {}
    drops: dict[str, set[str]] = {}
    coalesce_plan: dict[str, list[str]] = {}
    if not duplicates:
        return renames, drops, coalesce_plan, duplicates

    if policy == "error":
        col, srcs = next(iter(duplicates.items()))
        raise ValueError(
            f"Column {col!r} appears in sources {srcs!r} for asset {asset_id!r}; "
            "column names must be unique across sources — pass on_duplicate="
            "'rename'/'coalesce'/'first'/'last' to combine them instead"
        )

    if policy in ("first", "last"):
        for col, srcs in duplicates.items():
            keep = srcs[0] if policy == "first" else srcs[-1]
            for s in srcs:
                if s != keep:
                    drops.setdefault(s, set()).add(col)
        return renames, drops, coalesce_plan, duplicates

    # "rename" and "coalesce" both need every copy under its own name;
    # "coalesce" then folds those copies back into one column.
    taken = {ts} | set(owners)
    for col, srcs in duplicates.items():
        copies = []
        for s in srcs:
            new = _unique_name(f"{col}__{s}", taken)
            taken.add(new)
            renames.setdefault(s, {})[col] = new
            copies.append(new)
        if policy == "coalesce":
            coalesce_plan[col] = copies
    return renames, drops, coalesce_plan, duplicates


def _merge_frames(
    frames: list[pl.LazyFrame],
    ts: str,
    merge: MergeStrategy,
    asof_strategy: AsofStrategy,
    asof_tolerance: str | timedelta | None,
) -> pl.LazyFrame:
    """Combine per-source frames into one, per the ``merge`` strategy."""
    if len(frames) == 1:
        return frames[0]
    if merge == "vertical":
        return pl.concat(frames, how="diagonal_relaxed")
    if merge == "asof":
        frames = [f.sort(ts) for f in frames]

    combined = frames[0]
    for lf in frames[1:]:
        if merge == "outer":
            combined = combined.join(lf, on=ts, how="full", coalesce=True)
        elif merge == "asof":
            combined = combined.join_asof(
                lf, on=ts, strategy=asof_strategy, tolerance=asof_tolerance
            )
        else:  # "left" / "inner"
            combined = combined.join(lf, on=ts, how=merge)
    return combined


@overload
def load_event(
    asset_id: str,
    start: datetime | None,
    end: datetime | None,
    cfg: LoaderConfig,
    columns: list[str] | None = ...,
    *,
    merge: MergeStrategy = ...,
    on_duplicate: DuplicatePolicy = ...,
    source_order: list[str] | None = ...,
    asof_strategy: AsofStrategy = ...,
    asof_tolerance: str | timedelta | None = ...,
    with_metadata: Literal[False] = ...,
) -> pl.LazyFrame: ...


@overload
def load_event(
    asset_id: str,
    start: datetime | None,
    end: datetime | None,
    cfg: LoaderConfig,
    columns: list[str] | None = ...,
    *,
    merge: MergeStrategy = ...,
    on_duplicate: DuplicatePolicy = ...,
    source_order: list[str] | None = ...,
    asof_strategy: AsofStrategy = ...,
    asof_tolerance: str | timedelta | None = ...,
    with_metadata: Literal[True],
) -> LoadResult: ...


def load_event(
    asset_id: str,
    start: datetime | None,
    end: datetime | None,
    cfg: LoaderConfig,
    columns: list[str] | None = None,
    *,
    merge: MergeStrategy = "outer",
    on_duplicate: DuplicatePolicy = "error",
    source_order: list[str] | None = None,
    asof_strategy: AsofStrategy = "backward",
    asof_tolerance: str | timedelta | None = None,
    with_metadata: bool = False,
) -> pl.LazyFrame | LoadResult:
    """Lazy-load an asset's raw time-series, sliced to ``[start, end]``.

    Files are grouped by source (filename prefix): each source is
    vertically concatenated across time, then the sources are combined on
    ``cfg.timestamp_col`` following ``merge``. Timestamps should be unique
    within each source — duplicated stamps fan out through a join.

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
        Subset of columns to load, named as they appear *on disk* (i.e.
        before any duplicate renaming). The timestamp column is always
        included, and sources contributing none of the requested columns
        are skipped entirely. If ``None``, all columns of all sources
        are loaded.
    merge : {"outer", "left", "inner", "vertical", "asof"}, default "outer"
        How sources covering the same period are combined:

        - ``"outer"`` — full outer join on the timestamp: the union of all
          timestamps, nulls where a source has no row. Loses nothing;
          appropriate for sources at different sampling rates.
        - ``"left"`` — keep the anchor source's timestamps only (the first
          source in ``source_order``); other sources contribute where their
          stamps match exactly.
        - ``"inner"`` — keep only timestamps present in *every* source.
        - ``"vertical"`` — no join: stack the sources' rows
          (:func:`polars.concat` with ``how="diagonal_relaxed"``, so
          disjoint columns are null-filled). Use when files hold the same
          kind of record for overlapping periods rather than different
          variables.
        - ``"asof"`` — join each source onto the anchor's timestamps with
          :meth:`polars.LazyFrame.join_asof`, matching the nearest stamp in
          the ``asof_strategy`` direction within ``asof_tolerance``.
          Aligns a slow source onto a fast grid without null gaps.
    on_duplicate : {"error", "rename", "coalesce", "first", "last"}, default "error"
        What to do when a non-timestamp column name appears in more than
        one source:

        - ``"error"`` — raise :class:`ValueError` (the historical behaviour).
        - ``"rename"`` — keep every copy, suffixed with its source
          (``x`` → ``x__sensor``, ``x__backup``).
        - ``"coalesce"`` — one output column ``x``, taking the first
          non-null value across sources in ``source_order`` priority.
        - ``"first"`` / ``"last"`` — keep the copy from the highest /
          lowest priority source and drop the others.
    source_order : list of str, optional
        Source priority, highest first. Sources not listed follow in
        alphabetical order; listed names the asset lacks are ignored. Sets
        the ``"left"``/``"asof"`` anchor, duplicate-column priority, and
        output column order. Defaults to alphabetical.
    asof_strategy : {"backward", "forward", "nearest"}, default "backward"
        Match direction for ``merge="asof"``.
    asof_tolerance : str or timedelta, optional
        Maximum distance for an ``asof`` match (e.g. ``"2h"``); beyond it
        the row is null. ``None`` means unlimited.
    with_metadata : bool, default False
        If True, return a :class:`LoadResult` — ``(frame, metadata)`` —
        instead of the bare frame.

    Returns
    -------
    pl.LazyFrame or LoadResult
        Lazy frame sorted by timestamp, restricted to ``[start, end]``,
        and projected to ``columns`` (plus the timestamp). No
        ``.collect()`` is performed. With ``with_metadata=True``, a
        :class:`LoadResult` whose ``metadata`` dict holds:

        ``asset_id``, ``requested`` (window and columns), ``config``
        (data root, resolved asset dir, timestamp column, filename
        pattern), ``sources`` (per source: files, file count, covered
        period, available/loaded columns, renames, drops), ``source_order``,
        ``merge`` (strategy, join key, anchor, asof settings),
        ``duplicates`` (policy, the duplicated columns and their sources,
        how each was resolved), ``operations`` (ordered log of every
        step applied), ``columns`` and ``schema`` of the result.

    Raises
    ------
    FileNotFoundError
        If the asset folder does not exist (raised by
        :func:`discover_sources`) or if no files in the folder overlap
        the requested range.
    ValueError
        If ``merge``/``on_duplicate`` is not a known value, if a source
        lacks the timestamp column, if a requested column exists in no
        source, or if a non-timestamp column appears in more than one
        source while ``on_duplicate="error"``.
    """
    lf, metadata = _load_event(
        asset_id,
        start,
        end,
        cfg,
        columns,
        merge,
        on_duplicate,
        source_order,
        asof_strategy,
        asof_tolerance,
        with_metadata,
    )
    return LoadResult(frame=lf, metadata=metadata) if with_metadata else lf


def _load_event(
    asset_id: str,
    start: datetime | None,
    end: datetime | None,
    cfg: LoaderConfig,
    columns: list[str] | None,
    merge: MergeStrategy,
    on_duplicate: DuplicatePolicy,
    source_order: list[str] | None,
    asof_strategy: AsofStrategy,
    asof_tolerance: str | timedelta | None,
    with_metadata: bool,
) -> tuple[pl.LazyFrame, dict[str, Any]]:
    """Assemble the lazy frame, with its provenance when ``with_metadata``.

    Implements :func:`load_event`; kept separate so both public loaders
    can consume the frame and the metadata without unwrapping a union.
    """
    if merge not in MERGE_STRATEGIES:
        raise ValueError(f"Unknown merge strategy {merge!r}; expected one of {MERGE_STRATEGIES}")
    if on_duplicate not in DUPLICATE_POLICIES:
        raise ValueError(
            f"Unknown on_duplicate policy {on_duplicate!r}; expected one of {DUPLICATE_POLICIES}"
        )

    ts = cfg.timestamp_col
    groups = discover_sources(asset_id, start, end, cfg)
    if not groups:
        window = "" if start is None and end is None else f" in [{start}, {end}]"
        raise FileNotFoundError(f"No parquet files for asset={asset_id}{window}")

    ordered = _order_sources(list(groups), source_order)
    ops: list[dict[str, Any]] = [
        {
            "op": "discover",
            "asset_dir": str(cfg.asset_dir(asset_id)),
            "window": {"start": start, "end": end},
            "sources": ordered,
            "n_files": sum(len(f) for f in groups.values()),
        }
    ]
    pattern = re.compile(cfg.filename_pattern)
    src_meta: dict[str, dict[str, Any]] = {}

    # Scan each source (its files concatenated across time) and record the
    # columns it can contribute, before any duplicate resolution.
    scanned: dict[str, pl.LazyFrame] = {}
    per_source: list[tuple[str, list[str]]] = []
    for source in ordered:
        files = groups[source]
        lf = pl.scan_parquet([str(f) for f in files])
        names = lf.collect_schema().names()
        if ts not in names:
            raise ValueError(
                f"Source {source!r} of asset {asset_id!r} has no timestamp column {ts!r}"
            )
        available = [c for c in names if c != ts]
        spans = [p for p in (_parse_filename(f, pattern) for f in files) if p is not None]
        src_meta[source] = {
            "files": [str(f) for f in files],
            "n_files": len(files),
            "period": {
                "start": min(s for _, s, _ in spans) if spans else None,
                "end": max(e for _, _, e in spans) if spans else None,
            },
            "columns_available": available,
            "columns_loaded": [],
            "renamed": {},
            "dropped": [],
        }
        ops.append(
            {
                "op": "scan",
                "source": source,
                "files": [str(f) for f in files],
                "columns": available,
            }
        )
        if len(files) > 1:
            ops.append({"op": "concat", "source": source, "how": "vertical", "n_files": len(files)})
        scanned[source] = lf
        wanted = available if columns is None else [c for c in available if c in columns]
        per_source.append((source, wanted))

    if columns is not None:
        found = {c for _, cols in per_source for c in cols}
        missing = [c for c in columns if c != ts and c not in found]
        if missing:
            raise ValueError(f"Columns not found in any source of asset {asset_id!r}: {missing}")

    renames, drops, coalesce_plan, duplicates = _resolve_duplicates(
        per_source, on_duplicate, ts, asset_id
    )

    frames: list[pl.LazyFrame] = []
    used: list[str] = []
    final_cols: list[str] = [ts]
    for source, wanted in per_source:
        dropped = sorted(drops.get(source, set()) & set(wanted))
        keep = [c for c in wanted if c not in drops.get(source, set())]
        if dropped:
            src_meta[source]["dropped"] = dropped
            ops.append(
                {
                    "op": "drop",
                    "source": source,
                    "columns": dropped,
                    "reason": f"duplicate column, on_duplicate={on_duplicate!r}",
                }
            )
        if src_meta[source]["columns_available"] and not keep:
            # Nothing left to contribute but timestamps. (A source that
            # only ever had a timestamp column still joins its grid in.)
            ops.append(
                {
                    "op": "skip_source",
                    "source": source,
                    "reason": (
                        "all its columns were dropped as duplicates"
                        if dropped
                        else "contributes none of the requested columns"
                    ),
                }
            )
            continue
        lf = scanned[source].select([ts, *keep])
        mapping = {c: n for c, n in renames.get(source, {}).items() if c in keep}
        if mapping:
            lf = lf.rename(mapping)
            src_meta[source]["renamed"] = mapping
            ops.append(
                {
                    "op": "rename",
                    "source": source,
                    "mapping": mapping,
                    "reason": f"duplicate column, on_duplicate={on_duplicate!r}",
                }
            )
        loaded = [mapping.get(c, c) for c in keep]
        src_meta[source]["columns_loaded"] = loaded
        for c in loaded:
            # Coalesced copies collapse back onto the original name, held
            # at the position of the highest-priority source that has it.
            name = next((final for final, copies in coalesce_plan.items() if c in copies), c)
            if name not in final_cols:
                final_cols.append(name)
        frames.append(lf)
        used.append(source)

    if not frames:
        # Request reduced to the timestamp column alone.
        anchor = ordered[0]
        frames.append(pl.scan_parquet([str(f) for f in groups[anchor]]).select([ts]))
        used.append(anchor)
        ops.append({"op": "timestamp_only", "source": anchor})

    combined = _merge_frames(frames, ts, merge, asof_strategy, asof_tolerance)
    if len(frames) > 1:
        entry: dict[str, Any] = {"op": "merge", "strategy": merge, "sources": used}
        if merge != "vertical":
            entry["on"] = ts
        if merge in ("left", "asof"):
            entry["anchor"] = used[0]
        if merge == "asof":
            entry["asof_strategy"] = asof_strategy
            entry["asof_tolerance"] = asof_tolerance
        ops.append(entry)

    # Every copy in the plan is present: a source holding a duplicated
    # column always contributes it, and "coalesce" drops nothing.
    if coalesce_plan:
        combined = combined.with_columns(
            [
                pl.coalesce([pl.col(c) for c in copies]).alias(final)
                for final, copies in coalesce_plan.items()
            ]
        )
        ops.extend(
            {"op": "coalesce", "column": final, "from": copies}
            for final, copies in coalesce_plan.items()
        )
    combined = combined.select(final_cols)

    tcol = pl.col(ts)
    if start is not None:
        combined = combined.filter(tcol >= start)
    if end is not None:
        combined = combined.filter(tcol <= end)
    if start is not None or end is not None:
        ops.append({"op": "filter", "column": ts, "start": start, "end": end})
    combined = combined.sort(ts)
    ops.append({"op": "sort", "by": ts})

    if not with_metadata:
        return combined, {}

    schema = combined.collect_schema()
    metadata: dict[str, Any] = {
        "asset_id": asset_id,
        "requested": {
            "start": start,
            "end": end,
            "columns": list(columns) if columns is not None else None,
        },
        "config": {
            "data_root": str(cfg.data_root),
            "asset_dir": str(cfg.asset_dir(asset_id)),
            "timestamp_col": ts,
            "filename_pattern": cfg.filename_pattern,
        },
        "source_order": ordered,
        "sources": src_meta,
        "merge": {
            "strategy": merge,
            "on": None if merge == "vertical" else ts,
            "anchor": used[0] if merge in ("left", "asof") else None,
            "sources": used,
            "asof_strategy": asof_strategy if merge == "asof" else None,
            "asof_tolerance": asof_tolerance if merge == "asof" else None,
        },
        "duplicates": {
            "policy": on_duplicate,
            "columns": duplicates,
            "renamed": {s: dict(m) for s, m in renames.items()},
            "dropped": {s: sorted(c) for s, c in drops.items()},
            "coalesced": coalesce_plan,
        },
        "operations": ops,
        "columns": schema.names(),
        "schema": {name: str(dtype) for name, dtype in schema.items()},
    }
    return combined, metadata


@overload
def load_asset(
    asset_id: str,
    root: str | Path | LoaderConfig = ...,
    *,
    start: datetime | None = ...,
    end: datetime | None = ...,
    columns: list[str] | None = ...,
    lazy: bool = ...,
    merge: MergeStrategy = ...,
    on_duplicate: DuplicatePolicy = ...,
    source_order: list[str] | None = ...,
    asof_strategy: AsofStrategy = ...,
    asof_tolerance: str | timedelta | None = ...,
    with_metadata: Literal[False] = ...,
) -> pl.DataFrame | pl.LazyFrame: ...


@overload
def load_asset(
    asset_id: str,
    root: str | Path | LoaderConfig = ...,
    *,
    start: datetime | None = ...,
    end: datetime | None = ...,
    columns: list[str] | None = ...,
    lazy: bool = ...,
    merge: MergeStrategy = ...,
    on_duplicate: DuplicatePolicy = ...,
    source_order: list[str] | None = ...,
    asof_strategy: AsofStrategy = ...,
    asof_tolerance: str | timedelta | None = ...,
    with_metadata: Literal[True],
) -> LoadResult: ...


def load_asset(
    asset_id: str,
    root: str | Path | LoaderConfig = "data",
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    columns: list[str] | None = None,
    lazy: bool = False,
    merge: MergeStrategy = "outer",
    on_duplicate: DuplicatePolicy = "error",
    source_order: list[str] | None = None,
    asof_strategy: AsofStrategy = "backward",
    asof_tolerance: str | timedelta | None = None,
    with_metadata: bool = False,
) -> pl.DataFrame | pl.LazyFrame | LoadResult:
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
    merge, on_duplicate, source_order, asof_strategy, asof_tolerance
        Multi-source assembly options, forwarded verbatim to
        :func:`load_event` — how sources sharing a period are combined,
        and how a column name shared by several sources is resolved.
    with_metadata : bool, default False
        If True, return a :class:`LoadResult` — ``(frame, metadata)`` —
        instead of the bare frame. See :func:`load_event` for the
        metadata layout; an eager load appends a ``collect`` entry to
        its ``operations`` log.

    Returns
    -------
    pl.DataFrame, pl.LazyFrame, or LoadResult
        The asset's data sorted by timestamp (lazy when ``lazy=True``),
        paired with its provenance when ``with_metadata=True``.
    """
    cfg = root if isinstance(root, LoaderConfig) else LoaderConfig(data_root=Path(root))
    lf, metadata = _load_event(
        asset_id,
        start,
        end,
        cfg,
        columns,
        merge,
        on_duplicate,
        source_order,
        asof_strategy,
        asof_tolerance,
        with_metadata,
    )
    if lazy:
        return LoadResult(frame=lf, metadata=metadata) if with_metadata else lf
    df = lf.collect()
    if not with_metadata:
        return df
    metadata["operations"] = [*metadata["operations"], {"op": "collect"}]
    return LoadResult(frame=df, metadata=metadata)

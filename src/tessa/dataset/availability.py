"""Which labelled events have raw data on disk?

Label tables are usually maintained by hand, independently of the data
store: they name assets that were never exported, or periods before an asset
was instrumented. :func:`~tessa.dataset.builder.build` raises on the first
such event, so the label table should be matched against the store first.

:func:`data_availability` annotates every label row with what the loader can
find for its window, and :func:`filter_available` keeps only the rows it can
load. Both work from the file catalogue (:func:`asset_loader.file_catalog`),
which reads file *names* only; ``check="rows"`` additionally scans the
timestamp column to confirm the window holds at least one row.

>>> avail = data_availability(labels, cfg)          # diagnostics per event
>>> avail.group_by("data_reason").len()             # why events are missing
>>> labels = filter_available(labels, cfg)          # keep what can be loaded
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from typing import Literal

import polars as pl

from asset_loader import CATALOG_SCHEMA, file_catalog

from tessa.config import Config

AvailabilityCheck = Literal["files", "rows"]
"""``"files"``: a file's name-encoded period overlaps the window.
``"rows"``: additionally, the files hold at least one timestamp in it."""

REASON_NO_FOLDER = "asset folder not found"
REASON_NO_FILES = "asset has no parquet files matching the filename pattern"
REASON_NO_OVERLAP = "no file covers the event window"
REASON_NO_ROWS = "files cover the window but hold no rows in it"
REASON_LOW_COVERAGE = "coverage below min_coverage"

_ROW = "__row"
_ONE_DAY = timedelta(days=1)


def data_availability(
    labels: pl.DataFrame,
    cfg: Config,
    *,
    check: AvailabilityCheck = "files",
    catalog: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Annotate each label row with the data the loader can find for it.

    Parameters
    ----------
    labels : pl.DataFrame
        Label table with ``cfg.asset_col`` and datetime ``start`` / ``end``
        columns (e.g. the output of a :class:`~tessa.labels.LabelSource`).
        Other columns pass through untouched.
    cfg : Config
        Data layout (data root, filename pattern, timestamp column).
    check : {"files", "rows"}, default "files"
        ``"files"`` decides from file names alone and touches no file
        contents. ``"rows"`` also scans the timestamp column of the
        overlapping files, which catches gaps inside a file (a file named
        ``_240101_240131`` that actually stops on the 20th) at the cost of
        reading that one column.
    catalog : pl.DataFrame, optional
        A precomputed :func:`asset_loader.file_catalog`, to reuse one
        inventory across calls. Built from ``cfg`` when omitted.

    Returns
    -------
    pl.DataFrame
        ``labels`` in its original row order with these columns appended:

        - ``data_n_files`` (u32) -- files whose period overlaps the window.
        - ``data_sources`` (list[str]) -- sources those files belong to.
        - ``data_coverage`` (f64) -- fraction of ``[start, end]`` covered by
          the files' name-encoded periods (day resolution, so a file ending
          on a day covers all of it). A partially covered event still loads;
          it just holds less data than its window suggests.
        - ``data_n_rows``, ``data_first_ts``, ``data_last_ts`` -- only with
          ``check="rows"``: rows inside the window across all sources, and
          the first and last timestamp found.
        - ``data_available`` (bool) -- :func:`~tessa.dataset.builder.build`
          can load the event.
        - ``data_reason`` (str) -- why not, or null when available.

    Raises
    ------
    ValueError
        If a required column is missing or ``start`` / ``end`` are not
        datetimes, or if ``check`` is not a known value.
    """
    if check not in ("files", "rows"):
        raise ValueError(f"check must be 'files' or 'rows', got {check!r}")
    asset_col = cfg.asset_col
    missing = {asset_col, "start", "end"} - set(labels.columns)
    if missing:
        raise ValueError(f"Label table missing columns: {sorted(missing)}")
    for col in ("start", "end"):
        if not labels.schema[col].is_temporal():
            raise ValueError(
                f"Label column {col!r} is {labels.schema[col]}, not a datetime; "
                "parse it first (tessa.labels.validate does)"
            )
    clashes = [c for c in labels.columns if c.startswith("data_") or c == _ROW]
    if clashes:
        raise ValueError(f"Label table already has availability-style columns: {clashes}")

    assets = labels[asset_col].cast(pl.String).unique().drop_nulls().to_list()
    if catalog is None:
        catalog = file_catalog(cfg, assets)
    catalog = catalog.select(CATALOG_SCHEMA.names()).cast(dict(CATALOG_SCHEMA))

    events = labels.select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias(_ROW),
        pl.col(asset_col).cast(pl.String).alias("asset_id"),
        pl.col("start").cast(pl.Datetime("us")).alias("ev_start"),
        pl.col("end").cast(pl.Datetime("us")).alias("ev_end"),
    )

    # Files overlapping each window, with the same rule discover_sources
    # applies: a file named ..._<end> covers the whole of its end day.
    overlap = (
        events.join(catalog, on="asset_id", how="inner")
        .with_columns(file_end=pl.col("end") + _ONE_DAY)
        .filter((pl.col("file_end") > pl.col("ev_start")) & (pl.col("start") <= pl.col("ev_end")))
        .with_columns(
            clip_start=pl.max_horizontal("start", "ev_start"),
            clip_end=pl.min_horizontal("file_end", "ev_end"),
        )
    )
    per_event = _summarise_overlap(overlap)
    if check == "rows":
        per_event = per_event.join(_scan_rows(overlap, events, cfg), on=_ROW, how="left")

    with_folder = {a for a in assets if cfg.asset_dir(a).is_dir()}
    with_files = set(catalog["asset_id"].to_list())
    out = events.join(per_event, on=_ROW, how="left").with_columns(
        pl.col("data_n_files").fill_null(0),
        pl.col("data_sources").fill_null(pl.lit([], dtype=pl.List(pl.String))),
        pl.col("data_coverage").fill_null(0.0),
    )
    if check == "rows":
        out = out.with_columns(pl.col("data_n_rows").fill_null(0))
    has_data = pl.col("data_n_rows") > 0 if check == "rows" else pl.col("data_n_files") > 0

    out = out.with_columns(
        data_available=has_data,
        data_reason=(
            pl.when(has_data)
            .then(None)
            .when(~pl.col("asset_id").is_in(list(with_folder)))
            .then(pl.lit(REASON_NO_FOLDER))
            .when(~pl.col("asset_id").is_in(list(with_files)))
            .then(pl.lit(REASON_NO_FILES))
            .when(pl.col("data_n_files") == 0)
            .then(pl.lit(REASON_NO_OVERLAP))
            .otherwise(pl.lit(REASON_NO_ROWS))
        ),
    )
    extra = ["data_n_files", "data_sources", "data_coverage"]
    if check == "rows":
        extra += ["data_n_rows", "data_first_ts", "data_last_ts"]
    extra += ["data_available", "data_reason"]
    return pl.concat([labels, out.sort(_ROW).select(extra)], how="horizontal")


def filter_available(
    labels: pl.DataFrame,
    cfg: Config,
    *,
    check: AvailabilityCheck = "files",
    min_coverage: float = 0.0,
    catalog: pl.DataFrame | None = None,
    warn: bool = True,
) -> pl.DataFrame:
    """Keep only the label rows whose events the loader can find data for.

    Parameters
    ----------
    labels, cfg, check, catalog
        As in :func:`data_availability`.
    min_coverage : float, default 0.0
        Also drop events whose window is covered by less than this fraction
        (see ``data_coverage``). ``0`` keeps any event that loads at all;
        ``1.0`` keeps only fully covered windows.
    warn : bool, default True
        Emit a :class:`UserWarning` counting the dropped rows per reason, so
        a label table never shrinks silently.

    Returns
    -------
    pl.DataFrame
        The kept rows of ``labels``, original columns and order. Call
        :func:`data_availability` to see the dropped rows and why.
    """
    return _filter_available(labels, cfg, check, min_coverage, catalog, warn, stacklevel=3)


def _filter_available(
    labels: pl.DataFrame,
    cfg: Config,
    check: AvailabilityCheck,
    min_coverage: float,
    catalog: pl.DataFrame | None,
    warn: bool,
    stacklevel: int,
) -> pl.DataFrame:
    """Body of :func:`filter_available`; ``stacklevel`` points the warning
    at the user's call whichever public wrapper they went through."""
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage}")
    avail = data_availability(labels, cfg, check=check, catalog=catalog)
    reason = pl.coalesce(
        pl.col("data_reason"),
        pl.when(pl.col("data_coverage") < min_coverage).then(pl.lit(REASON_LOW_COVERAGE)),
    )
    avail = avail.with_columns(data_reason=reason)
    dropped = avail.filter(pl.col("data_reason").is_not_null())
    if warn and dropped.height:
        counts = dropped.group_by("data_reason").len().sort("len", descending=True)
        detail = ", ".join(f"{n} {r}" for r, n in counts.iter_rows())
        warnings.warn(
            f"filter_available dropped {dropped.height} of {labels.height} label rows "
            f"({detail}); call data_availability() for details",
            stacklevel=stacklevel,
        )
    return avail.filter(pl.col("data_reason").is_null()).select(labels.columns)


def _summarise_overlap(overlap: pl.DataFrame) -> pl.DataFrame:
    """Per event: file count, sources, and covered fraction of the window.

    The covered length is the length of the *union* of the clipped file
    intervals, so two sources spanning the same days count once. With the
    intervals sorted by start, each one adds only the part beyond the
    furthest end seen so far.
    """
    reach = pl.col("clip_end").cum_max().shift(1).over(_ROW)
    added = pl.col("clip_end") - pl.max_horizontal("clip_start", reach)
    overlap = overlap.sort(_ROW, "clip_start").with_columns(
        added=pl.when(added > timedelta(0)).then(added).otherwise(timedelta(0))
    )
    duration = (pl.col("ev_end") - pl.col("ev_start")).first()
    covered = pl.col("added").sum()
    return overlap.group_by(_ROW).agg(
        data_n_files=pl.len(),
        data_sources=pl.col("source").unique().sort(),
        # A zero-length window (start == end) is covered if any file overlaps.
        data_coverage=pl.when(duration > timedelta(0))
        .then(covered.dt.total_microseconds() / duration.dt.total_microseconds())
        .otherwise(1.0)
        .clip(0.0, 1.0),
    )


def _scan_rows(overlap: pl.DataFrame, events: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Count the rows inside each window by scanning the timestamp column.

    Sources are scanned separately (their files share a schema, sources do
    not) and the per-event queries run together via ``pl.collect_all``.
    Parquet statistics let polars skip row groups outside the window.
    """
    ts = cfg.timestamp_col
    files = overlap.group_by(_ROW, "source").agg(pl.col("path").sort())
    windows = {row: (s, e) for row, _, s, e in events.iter_rows()}

    rows: list[int] = []
    queries: list[pl.LazyFrame] = []
    for row, group in files.group_by(_ROW, maintain_order=True):
        start, end = windows[row[0]]
        parts = [
            pl.scan_parquet(paths).select(pl.col(ts).cast(pl.Datetime("us")))
            for paths in group["path"].to_list()
        ]
        queries.append(
            pl.concat(parts, how="vertical")
            .filter(pl.col(ts).is_between(start, end))
            .select(
                data_n_rows=pl.len(),
                data_first_ts=pl.col(ts).min(),
                data_last_ts=pl.col(ts).max(),
            )
        )
        rows.append(row[0])

    schema = {
        _ROW: pl.UInt32,
        "data_n_rows": pl.UInt32,
        "data_first_ts": pl.Datetime("us"),
        "data_last_ts": pl.Datetime("us"),
    }
    if not queries:
        return pl.DataFrame(schema=schema)
    found = pl.concat(pl.collect_all(queries), how="vertical_relaxed")
    return found.with_columns(pl.Series(_ROW, rows, dtype=pl.UInt32)).cast(schema)

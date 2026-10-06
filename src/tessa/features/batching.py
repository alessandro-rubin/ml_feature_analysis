"""Collect per-event queries in batches, so memory does not grow with the event count.

The materialisers build one lazy query per event (load, derive per-sample
features, reduce). Handing all of them to a single ``pl.collect_all`` lets
polars hold many events' per-sample frames at the same time, and peak memory
then grows with the number of events: on a benchmark of 600 events (12 h of
1 Hz data, 16 signals, 48 derived features each), one ``collect_all`` peaked
at 3.6 GB, against 1.1 GB when collecting 16 events at a time, which was also
faster.

:func:`collect_batched` therefore collects in batches, in input order:

- at most ``Config.events_per_batch`` events per batch (default: the size of
  the polars thread pool, so every core stays busy);
- when event sizes are known, at most ``Config.memory_budget`` bytes of
  estimated working set per batch. An event above the budget is collected on
  its own, with a warning.

Sizing an event
---------------
:func:`estimate_event_bytes` reads no data, only file names and parquet
footers. It takes each event's window
(recorded by :func:`tessa.dataset.builder.build` on
:attr:`~tessa.dataset.EventFrames.windows`), finds the files that overlap it,
and scales each file's row count (from the parquet footer) by the share of
the file's period the window covers. The working set is then

    rows x columns x :data:`BYTES_PER_VALUE`

where *columns* are the numeric columns the event is reduced over (raw
signals plus derived features). :data:`BYTES_PER_VALUE` is 32: 8 bytes per
float64 value times 4, because loading, joining, sorting and window features
were measured to need 3 to 5 times the raw size of the frame (polars 1.40).
Rows from different sources are added up, which over-counts sources sharing
timestamps: the estimate errs on the safe side.
"""

from __future__ import annotations

import ctypes
import functools
import os
import re
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from asset_loader import LoaderConfig, file_catalog

from tessa.config import Config

BYTES_PER_VALUE = 32
"""Estimated working-set bytes per value of an event's per-sample frame."""

_UNITS = {"": 1, "k": 2**10, "m": 2**20, "g": 2**30, "t": 2**40}


# ── Settings ─────────────────────────────────────────────────────────────────


def _physical_memory() -> int | None:
    """Physical memory of the machine (or the container's limit), if knowable."""
    total: int | None = None
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
    except ImportError:
        if sys.platform == "win32":

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total = int(status.ullTotalPhys)
        else:
            try:
                total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            except (ValueError, OSError, AttributeError):
                total = None
    # A container may be allowed far less than the host has (cgroup v2, v1).
    for limit in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            cap = int(Path(limit).read_text().strip())
        except (OSError, ValueError):  # absent, or "max" for no limit
            continue
        total = cap if total is None else min(total, cap)
    return total


def memory_budget_bytes(budget: int | str | None) -> int | None:
    """Resolve a :attr:`Config.memory_budget` value to bytes.

    Parameters
    ----------
    budget : int, str, or None
        Bytes as an int; a size string such as ``"8GB"``, ``"512 MB"`` or
        ``"1.5GiB"`` (binary units: 1 GB = 2**30 bytes); ``"auto"`` for half
        the machine's physical memory; or ``None`` for no cap.

    Returns
    -------
    int or None
        The budget in bytes, or ``None`` when there is no cap (``"auto"`` on
        a machine whose memory cannot be determined).

    Raises
    ------
    ValueError
        If the string is not a size.
    """
    if budget is None:
        return None
    if isinstance(budget, (int, float)) and not isinstance(budget, bool):
        if budget <= 0:
            raise ValueError(f"memory_budget must be positive, got {budget}")
        return int(budget)
    text = str(budget).strip().lower()
    if text == "auto":
        total = _physical_memory()
        return total // 2 if total else None
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([kmgt]?)(i?b)?", text)
    if not m or float(m.group(1)) <= 0:
        raise ValueError(f"memory_budget {budget!r} is not a size like '8GB' or '512MB'")
    return int(float(m.group(1)) * _UNITS[m.group(2)])


def events_per_batch(value: int | None) -> int:
    """Resolve :attr:`Config.events_per_batch` (``None``: the polars thread pool size)."""
    if value is None:
        return max(1, pl.thread_pool_size())
    if value < 1:
        raise ValueError(f"events_per_batch must be at least 1, got {value}")
    return int(value)


# ── Sizing ───────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=65536)
def _parquet_rows(path: str, mtime_ns: int, size: int) -> int:
    """Row count from the parquet footer; cached per file version."""
    return pl.scan_parquet(path).select(pl.len()).collect().item()


def _file_rows(path: str) -> int:
    stat = os.stat(path)
    return _parquet_rows(path, stat.st_mtime_ns, stat.st_size)


def estimate_event_rows(
    windows: Mapping[str, tuple[str, datetime | None, datetime | None]],
    cfg: LoaderConfig,
) -> dict[str, float]:
    """Rows each event's frame is expected to hold, from file names and footers.

    Parameters
    ----------
    windows : mapping
        ``{event_id: (asset_id, start, end)}``, e.g.
        :attr:`tessa.dataset.EventFrames.windows`. ``None`` bounds mean
        unbounded.
    cfg : LoaderConfig
        Layout of the data store.

    Returns
    -------
    dict
        ``{event_id: estimated rows}``, summed over the sources (an
        over-count when sources share timestamps). Events whose asset has no
        files estimate to 0.
    """
    catalog = file_catalog(cfg, assets={asset for asset, _, _ in windows.values()})
    by_asset: dict[str, list[tuple[datetime, datetime, str]]] = {}
    for asset, path, start, end in catalog.select("asset_id", "path", "start", "end").iter_rows():
        # A file's end date names its last day, which it covers in full.
        by_asset.setdefault(asset, []).append((start, end + timedelta(days=1), path))
    out: dict[str, float] = {}
    for event_id, (asset, start, end) in windows.items():
        rows = 0.0
        for f_start, f_end, path in by_asset.get(asset, []):
            lo = max(f_start, start) if start is not None else f_start
            hi = min(f_end, end) if end is not None else f_end
            if hi <= lo:
                continue
            share = (hi - lo) / (f_end - f_start)
            rows += _file_rows(path) * share
        out[event_id] = rows
    return out


def estimate_event_bytes(
    events: Mapping[str, pl.LazyFrame], cfg: LoaderConfig, n_columns: int
) -> list[float] | None:
    """Estimated working set of each event, in the order of ``events``.

    Parameters
    ----------
    events : mapping
        Usually the :class:`~tessa.dataset.EventFrames` from
        :meth:`tessa.Dataset.events`; its :attr:`windows` locate the data.
    cfg : LoaderConfig
        Layout of the data store.
    n_columns : int
        Numeric columns each event is reduced over (signals plus derived
        features, plus the timestamp).

    Returns
    -------
    list of float or None
        Bytes per event, ``rows x n_columns x BYTES_PER_VALUE``. ``None``
        when ``events`` records no windows (a plain dict or list of frames),
        in which case events are batched by count only. Events missing from
        the windows count as 0.
    """
    windows = getattr(events, "windows", None)
    if not windows or not isinstance(events, Mapping):
        return None
    rows = estimate_event_rows({k: windows[k] for k in events if k in windows}, cfg)
    return [rows.get(k, 0.0) * n_columns * BYTES_PER_VALUE for k in events]


# ── Batching ─────────────────────────────────────────────────────────────────


def plan_batches(sizes: Sequence[float], max_events: int, budget: float | None) -> list[list[int]]:
    """Split item indices into consecutive batches under both limits.

    Parameters
    ----------
    sizes : sequence of float
        Estimated bytes per item (zeros when unknown).
    max_events : int
        Most items per batch.
    budget : float or None
        Most estimated bytes per batch; ``None`` for no cap. An item larger
        than the budget gets a batch of its own.

    Returns
    -------
    list of list of int
        Indices ``0..len(sizes)-1`` in order, grouped into batches.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    load = 0.0
    for i, size in enumerate(sizes):
        full = len(current) >= max_events or (
            budget is not None and current and load + size > budget
        )
        if full:
            batches.append(current)
            current, load = [], 0.0
        current.append(i)
        load += size
    if current:
        batches.append(current)
    return batches


def _size(n: float) -> str:
    return f"{n / 2**30:.1f} GB" if n >= 2**30 else f"{n / 2**20:.0f} MB"


def collect_batched(
    queries: Sequence[pl.LazyFrame],
    cfg: Config,
    *,
    sizes: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
    stacklevel: int = 2,
) -> list[pl.DataFrame]:
    """Collect ``queries`` batch by batch, returning their results in order.

    Parameters
    ----------
    queries : sequence of pl.LazyFrame
        One query per event.
    cfg : tessa.Config
        Supplies :attr:`~tessa.Config.events_per_batch` and
        :attr:`~tessa.Config.memory_budget`.
    sizes : sequence of float, optional
        Estimated bytes per query (see :func:`estimate_event_bytes`).
        Without them, batches are limited by count only.
    names : sequence of str, optional
        Event names, used in the warning about oversized events.
    stacklevel : int, default 2
        Passed to :func:`warnings.warn`, so the warning points at the
        caller's caller.

    Returns
    -------
    list of pl.DataFrame
        ``queries[i].collect()`` for every ``i``, in order.

    Warns
    -----
    UserWarning
        If some events are estimated above the memory budget on their own.
    """
    max_events = events_per_batch(cfg.events_per_batch)
    budget = memory_budget_bytes(cfg.memory_budget) if sizes else None
    sizes = list(sizes) if sizes else [0.0] * len(queries)
    if budget is not None:
        over = [i for i, s in enumerate(sizes) if s > budget]
        if over:
            big = max(over, key=lambda i: sizes[i])
            label = names[big] if names is not None else f"#{big}"
            warnings.warn(
                f"{len(over)} event(s) are estimated above memory_budget={_size(budget)} "
                f"on their own (largest: {label}, about {_size(sizes[big])}); each is "
                "collected alone. If memory still runs out, reduce the signals, "
                "derived features or aggregators, or shorten the event windows.",
                stacklevel=stacklevel + 1,
            )
    out: list[pl.DataFrame] = []
    for batch in plan_batches(sizes, max_events, budget):
        out.extend(pl.collect_all([queries[i] for i in batch]))
    return out

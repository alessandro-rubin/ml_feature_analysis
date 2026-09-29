"""Build per-event LazyFrames from a label table.

A *label table* is a :class:`polars.DataFrame` with one row per event and at
least the columns ``asset_id``, ``start``, ``end``, ``class``. This module
turns each row into an :class:`Event`, loads the corresponding raw
time-series via :func:`asset_loader.load_event`, and attaches
the label metadata as constant columns. The result is an :class:`EventFrames`
mapping (a ``{event_id: LazyFrame}`` dict that also records which columns are
label metadata) consumed by the materialisers in
:mod:`tessa.features.materialize`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

import polars as pl

from asset_loader import load_event

from tessa.config import Config


@dataclass(frozen=True)
class Event:
    """One labelled event.

    Parameters
    ----------
    event_id : str
        Stable identifier built from ``asset_id`` and the time range
        (see :func:`_event_id`).
    asset_id : str
        Asset this event belongs to.
    start, end : datetime
        Inclusive event window.
    label : dict
        Class label plus any extra label-table columns (e.g. stratification
        metadata). Each key/value pair becomes a constant column on the
        attached frame.
    """

    event_id: str
    asset_id: str
    start: datetime
    end: datetime
    label: dict  # class + any extras

    @property
    def label_cols(self) -> tuple[str, ...]:
        """Columns :meth:`attach_to` adds: ``event_id``, ``asset_id`` and every label key."""
        return ("event_id", "asset_id", *self.label)

    def attach_to(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Add ``event_id``, ``asset_id`` and label columns to ``lf``.

        Returns
        -------
        pl.LazyFrame
            Input frame with constant columns appended via
            ``with_columns``. The label values are broadcast over every
            row of the event.
        """
        cols = [
            pl.lit(self.event_id).alias("event_id"),
            pl.lit(self.asset_id).alias("asset_id"),
        ]
        for k, v in self.label.items():
            cols.append(pl.lit(v).alias(k))
        return lf.with_columns(cols)


class EventFrames(dict[str, pl.LazyFrame]):
    """``{event_id: LazyFrame}`` mapping that remembers its label columns.

    A plain dict in every respect, plus :attr:`label_cols`: the columns that
    came from the label table rather than from the sensor data. The
    materialisers read it so that a *numeric* label column (a severity score,
    an integer work-order ID) is carried through as label metadata instead of
    being aggregated as a signal, which would leak the label into the features.

    Parameters
    ----------
    frames : mapping, optional
        Initial ``{event_id: LazyFrame}`` items.
    label_cols : iterable of str, optional
        Label/id column names attached to every frame.

    Notes
    -----
    Only this class carries :attr:`label_cols`: a dict comprehension or
    ``dict(events)`` returns a plain dict. To keep the information when
    subsetting, use :meth:`subset`, or pass ``label_cols=events.label_cols``
    to the materialiser explicitly.
    """

    def __init__(
        self,
        frames: Mapping[str, pl.LazyFrame] | None = None,
        label_cols: Iterable[str] = (),
    ) -> None:
        super().__init__(frames or {})
        self.label_cols: tuple[str, ...] = tuple(dict.fromkeys(label_cols))

    def subset(self, event_ids: Iterable[str]) -> "EventFrames":
        """Return the events in ``event_ids`` (in that order), keeping :attr:`label_cols`."""
        return EventFrames({k: self[k] for k in event_ids}, self.label_cols)

    def __repr__(self) -> str:
        return f"EventFrames({len(self)} events, label_cols={list(self.label_cols)})"


def _label_extras(labels: pl.DataFrame, cfg: Config) -> list[str]:
    """Label-table columns that become :attr:`Event.label` keys."""
    return [c for c in labels.columns if c not in {cfg.asset_col, "start", "end"}]


def _event_id(asset_id: str, start: datetime, end: datetime, idx: int) -> str:
    """Return a deterministic identifier for an event row.

    The trailing ``idx`` disambiguates multiple events on the same asset
    that share a start/end pair.
    """
    return f"{asset_id}_{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}_{idx}"


def iter_events(
    labels: pl.DataFrame,
    cfg: Config,
) -> Iterable[Event]:
    """Yield :class:`Event` objects from a label table.

    Parameters
    ----------
    labels : pl.DataFrame
        Label table. Must contain ``cfg.asset_col``, ``"start"``, ``"end"``,
        and ``cfg.class_col``. Any other columns become entries in
        :attr:`Event.label` and are carried through as constant columns
        when the event is attached to a frame.
    cfg : Config
        Project configuration; used for column-name lookups only.

    Yields
    ------
    Event
        One per row of ``labels``, in row order.

    Raises
    ------
    ValueError
        If any of the required columns are missing.
    """
    required = {cfg.asset_col, "start", "end", cfg.class_col}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Label table missing columns: {missing}")

    extras_cols = _label_extras(labels, cfg)
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
) -> EventFrames:
    """Build per-event LazyFrames with label metadata attached.

    Parameters
    ----------
    labels : pl.DataFrame
        Label table; see :func:`iter_events` for required columns.
    cfg : Config
        Project configuration. Used to resolve filesystem paths and
        column names.
    columns : list of str, optional
        Subset of raw columns to load from each event's parquet files.
        The timestamp column is always included. If ``None``, every
        column is loaded.

    Returns
    -------
    EventFrames
        Mapping from ``event_id`` to a LazyFrame for that event, sorted
        by timestamp and decorated with constant ``event_id``,
        ``asset_id``, and label columns. Its ``label_cols`` attribute
        lists those columns, so the materialisers never aggregate them
        as signals, whatever their dtype.
    """
    out = EventFrames(label_cols=["event_id", "asset_id", *_label_extras(labels, cfg)])
    for ev in iter_events(labels, cfg):
        lf = load_event(ev.asset_id, ev.start, ev.end, cfg, columns=columns)
        out[ev.event_id] = ev.attach_to(lf)
    return out

"""Pluggable label-source protocol."""

from __future__ import annotations

from typing import Protocol

import polars as pl

from ml_analysis.config import Config


class LabelSource(Protocol):
    """Anything that can produce a label table.

    The returned DataFrame must contain at least:
      - cfg.asset_col   (str)
      - "start"          (datetime)
      - "end"            (datetime)
      - cfg.class_col   (any hashable)

    Any additional columns are treated as extras and propagate as metadata.
    """

    def load(self, cfg: Config) -> pl.DataFrame: ...


def validate(labels: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Enforce required columns + cast start/end to datetime. Returns a clean copy."""
    required = {cfg.asset_col, "start", "end", cfg.class_col}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Label table missing columns: {sorted(missing)}")

    out = labels
    for col in ("start", "end"):
        if not out.schema[col].is_temporal():
            out = out.with_columns(pl.col(col).str.to_datetime())
    return out

"""Excel-backed label source."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ml_analysis.config import Config
from ml_analysis.labels.base import validate


@dataclass
class ExcelLabelSource:
    """Reads a label table from an Excel sheet.

    `column_map` translates source column names to the canonical
    {asset_col, "start", "end", class_col, *extras}. Unmapped columns are
    kept as-is (so extras pass through automatically).
    """

    path: Path
    sheet: str | int = 0
    column_map: dict[str, str] = field(default_factory=dict)

    def load(self, cfg: Config) -> pl.DataFrame:
        # Polars splits sheet selection across two kwargs: `sheet_name` (str)
        # and `sheet_id` (1-based int). Translate a 0-based int to keep the
        # pandas-style contract advertised by callers.
        if isinstance(self.sheet, int):
            df = pl.read_excel(self.path, sheet_id=self.sheet + 1)
        else:
            df = pl.read_excel(self.path, sheet_name=self.sheet)
        if self.column_map:
            df = df.rename({k: v for k, v in self.column_map.items() if k in df.columns})
        return validate(df, cfg)

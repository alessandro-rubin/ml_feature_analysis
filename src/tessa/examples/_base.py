"""What an example data store hands back: where its pieces were written."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class ExampleDataset:
    """Paths written by an example-data function.

    Parameters
    ----------
    data_root : Path
        Folder with one sub-folder per asset, in the layout
        :mod:`asset_loader` expects.
    labels_path : Path
        The label sheet (an Excel file).
    column_map : dict
        Excel header -> tessa column name, for
        :class:`tessa.labels.ExcelLabelSource`.
    signals : pl.DataFrame, optional
        One row per data column with what it measures (``column``,
        ``description``, ``unit``, ...), when the source documents its
        columns. ``None`` for the synthetic store.
    """

    data_root: Path
    labels_path: Path
    column_map: dict[str, str]
    signals: pl.DataFrame | None = None

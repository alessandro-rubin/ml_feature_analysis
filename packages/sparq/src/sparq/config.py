"""Configuration for the on-disk parquet layout.

:class:`LoaderConfig` carries everything the loader needs to find and parse
parquet files: where they live, how filenames encode their source and time
range, and the name of the timestamp column. Downstream projects can use it
directly or subclass it to add their own settings (``ml_analysis.Config``
does exactly that).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoaderConfig:
    """On-disk layout settings consumed by :mod:`sparq.loader`.

    Parameters
    ----------
    data_root : Path or str, default ``Path("data")``
        Root directory containing one folder per asset. Plain strings are
        coerced to :class:`~pathlib.Path`.
    asset_subdir : str, default ``""``
        Optional sub-directory inside each asset folder that holds parquet
        files. The default (empty string) means files live directly in
        ``<data_root>/<asset_id>/``; set e.g. ``"input"`` for a nested
        layout ``<data_root>/<asset_id>/input/``.
    filename_pattern : str
        Regular expression used to parse parquet filenames into
        ``(source, start, end)``. Must contain named groups ``start`` and
        ``end`` (parsed as ``%y%m%d`` when 6 digits, ``%Y%m%d`` when 8)
        and should contain a ``source`` group naming the data set the file
        belongs to. Files sharing a ``source`` value must have identical
        columns and are concatenated across time; files with different
        ``source`` values carry different variables and are outer-joined
        on :attr:`timestamp_col` (see :mod:`sparq.loader`). A pattern
        without a ``source`` group treats every file as one source (a
        legacy ``asset`` group is used as the source name if present).
        Default matches
        ``"<dataname>_<YYMMDD|YYYYMMDD>_<YYMMDD|YYYYMMDD>.parquet"``
        where ``<dataname>`` may itself contain underscores.
    timestamp_col : str, default ``"timestamp"``
        Name of the time column in the raw data.
    """

    data_root: Path = Path("data")
    asset_subdir: str = ""
    filename_pattern: str = (
        r"^(?P<source>.+)_(?P<start>\d{6}(?:\d{2})?)_(?P<end>\d{6}(?:\d{2})?)\.parquet$"
    )
    timestamp_col: str = "timestamp"

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)

    def asset_dir(self, asset_id: str) -> Path:
        """Return the parquet directory for a given asset.

        Parameters
        ----------
        asset_id : str
            Asset identifier matching the folder name under
            :attr:`data_root`.

        Returns
        -------
        Path
            ``<data_root>/<asset_id>/<asset_subdir>`` (or just
            ``<data_root>/<asset_id>`` when :attr:`asset_subdir` is
            empty). The directory is not required to exist.
        """
        folder = self.data_root / asset_id
        return folder / self.asset_subdir if self.asset_subdir else folder

"""Global configuration shared across the pipeline.

A single :class:`Config` instance carries the on-disk layout (where parquet
files live, how their filenames encode time ranges), the standard column
names used by the rest of the package, and miscellaneous settings such as
the random seed. Every public function that touches the filesystem or
produces analysis output takes a ``cfg: Config`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Project-wide settings.

    Parameters
    ----------
    data_root : Path, default ``Path("data")``
        Root directory containing one folder per asset.
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
        on :attr:`timestamp_col` (see
        :mod:`ml_analysis.dataset.loader`). A pattern without a
        ``source`` group treats every file as one source (a legacy
        ``asset`` group is used as the source name if present). Default
        matches ``"<dataname>_<YYMMDD|YYYYMMDD>_<YYMMDD|YYYYMMDD>.parquet"``
        where ``<dataname>`` may itself contain underscores.
    timestamp_col : str, default ``"timestamp"``
        Name of the time column in the raw data.
    asset_col : str, default ``"asset_id"``
        Column name used in label tables to identify the asset.
    class_col : str, default ``"class"``
        Column name in label tables holding the categorical target.
    output_dir : Path, default ``Path("outputs")``
        Where analyses persist their artefacts.
    random_state : int, default ``42``
        Seed propagated to scikit-learn / numpy where applicable.
    extras : dict
        Free-form bag for project-specific settings consumed by user code.
    """

    data_root: Path = Path("data")
    asset_subdir: str = ""
    filename_pattern: str = (
        r"^(?P<source>.+)_(?P<start>\d{6}(?:\d{2})?)_(?P<end>\d{6}(?:\d{2})?)\.parquet$"
    )
    timestamp_col: str = "timestamp"
    asset_col: str = "asset_id"
    class_col: str = "class"
    output_dir: Path = Path("outputs")
    random_state: int = 42
    extras: dict = field(default_factory=dict)

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

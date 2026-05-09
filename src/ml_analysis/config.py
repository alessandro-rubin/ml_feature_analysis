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
    asset_subdir : str, default ``"input"``
        Sub-directory inside each asset folder that holds parquet files,
        i.e. files are read from ``<data_root>/<asset_id>/<asset_subdir>/``.
    filename_pattern : str
        Regular expression used to parse parquet filenames into
        ``(asset, start, end)``. Must contain named groups ``asset``,
        ``start``, and ``end``; ``start`` and ``end`` are parsed as
        ``%Y%m%d``. Default matches ``"<asset>_<YYYYMMDD>_<YYYYMMDD>.parquet"``.
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
    asset_subdir: str = "input"
    filename_pattern: str = r"^(?P<asset>[^_]+)_(?P<start>\d{8})_(?P<end>\d{8})\.parquet$"
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
            ``<data_root>/<asset_id>/<asset_subdir>``. The directory is
            not required to exist.
        """
        return self.data_root / asset_id / self.asset_subdir

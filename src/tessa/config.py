"""Global configuration shared across the pipeline.

A single :class:`Config` instance carries the on-disk layout (inherited from
:class:`asset_loader.LoaderConfig` — where parquet files live and how their
filenames encode sources and time ranges), the standard column names used by
the rest of the package, and miscellaneous settings such as the random seed.
Every public function that touches the filesystem or produces analysis
output takes a ``cfg: Config`` argument; the data-loading functions in
:mod:`asset_loader` accept it directly since it *is* a ``LoaderConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from asset_loader import LoaderConfig


@dataclass
class Config(LoaderConfig):
    """Project-wide settings.

    Extends :class:`asset_loader.LoaderConfig` (``data_root``, ``asset_subdir``,
    ``filename_pattern``, ``timestamp_col``, ``asset_dir()``) with the
    pipeline-level settings below.

    Parameters
    ----------
    asset_col : str, default ``"asset_id"``
        Column name used in label tables to identify the asset.
    class_col : str, default ``"class"``
        Column name in label tables holding the categorical target.
    output_dir : Path, default ``Path("outputs")``
        Where analyses persist their artefacts.
    random_state : int, default ``42``
        Seed propagated to scikit-learn / numpy where applicable.
    assume_sorted : bool, default ``False``
        If True, raw parquet files are trusted to be chronologically
        ordered and readers mark the timestamp column with
        ``set_sorted`` instead of paying an O(n log n) sort per event.
    extras : dict
        Free-form bag for project-specific settings consumed by user code.
    """

    asset_col: str = "asset_id"
    class_col: str = "class"
    output_dir: Path = Path("outputs")
    random_state: int = 42
    assume_sorted: bool = False
    extras: dict = field(default_factory=dict)

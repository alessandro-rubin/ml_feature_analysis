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
        Reserved: meant to let readers trust chronologically ordered files
        and skip the sort. Currently not applied, since every reader goes
        through :func:`asset_loader.load_event`, which always sorts.
    events_per_batch : int, optional
        Most events the materialisers read and reduce at the same time.
        Events are collected batch by batch, so peak memory follows the
        batch, not the number of events. ``None`` (default) uses the size
        of the polars thread pool, which keeps every core busy.
    memory_budget : int, str, or None, default ``"auto"``
        Cap on the estimated working set of one batch, in bytes or as a
        string such as ``"8GB"`` or ``"512MB"``. Large events (long windows,
        high sampling rates, many derived features) then run fewer at a
        time, and an event estimated above the budget runs on its own, with
        a warning. ``"auto"`` is half the machine's physical memory;
        ``None`` disables the cap. See :mod:`tessa.features.batching` for
        how events are sized.
    extras : dict
        Free-form bag for project-specific settings consumed by user code.
    """

    asset_col: str = "asset_id"
    class_col: str = "class"
    output_dir: Path = Path("outputs")
    random_state: int = 42
    assume_sorted: bool = False
    events_per_batch: int | None = None
    memory_budget: int | str | None = "auto"
    extras: dict = field(default_factory=dict)

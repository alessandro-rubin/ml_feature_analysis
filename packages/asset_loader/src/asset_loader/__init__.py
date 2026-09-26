"""Multi-source parquet time-series loader for per-asset data.

Lazy loading of per-asset time-series split across parquet files by time
period and by source (variable group), aligned on a common timestamp
column. Sources sharing a period can be combined several ways (``merge``)
and a column name shared by several sources can be renamed, coalesced or
dropped (``on_duplicate``); ``with_metadata=True`` returns a
:class:`LoadResult` carrying a provenance dict alongside the frame. See
:mod:`asset_loader.loader` for the on-disk conventions.
"""

from asset_loader.config import LoaderConfig
from asset_loader.loader import (
    DUPLICATE_POLICIES,
    MERGE_STRATEGIES,
    AsofStrategy,
    DuplicatePolicy,
    LoadResult,
    MergeStrategy,
    discover_files,
    discover_sources,
    load_asset,
    load_event,
)

__version__ = "0.2.0"

__all__ = [
    "DUPLICATE_POLICIES",
    "MERGE_STRATEGIES",
    "AsofStrategy",
    "DuplicatePolicy",
    "LoadResult",
    "LoaderConfig",
    "MergeStrategy",
    "discover_files",
    "discover_sources",
    "load_asset",
    "load_event",
]

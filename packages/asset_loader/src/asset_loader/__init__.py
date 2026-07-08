"""Multi-source parquet time-series loader for per-asset data.

Lazy loading of per-asset time-series split across parquet files by time
period and by source (variable group), aligned on a common timestamp
column. See :mod:`asset_loader.loader` for the on-disk conventions.
"""

from asset_loader.config import LoaderConfig
from asset_loader.loader import discover_files, discover_sources, load_asset, load_event

__version__ = "0.1.0"

__all__ = [
    "LoaderConfig",
    "discover_files",
    "discover_sources",
    "load_asset",
    "load_event",
]

"""SPARQ — Source-Partitioned Asset paRQuet loader.

Lazy loading of per-asset time-series split across parquet files by time
period and by source (variable group), aligned on a common timestamp
column. See :mod:`sparq.loader` for the on-disk conventions.
"""

from sparq.config import LoaderConfig
from sparq.loader import discover_files, discover_sources, load_asset, load_event

__version__ = "0.1.0"

__all__ = [
    "LoaderConfig",
    "discover_files",
    "discover_sources",
    "load_asset",
    "load_event",
]

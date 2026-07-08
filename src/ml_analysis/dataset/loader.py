"""Compatibility shim — the loader now lives in the standalone ``tessa``
package (``packages/tessa`` in this workspace). Import from :mod:`tessa`
directly in new code; this module re-exports the public API so existing
``ml_analysis.dataset.loader`` imports keep working.
"""

from tessa import discover_files, discover_sources, load_asset, load_event

__all__ = ["discover_files", "discover_sources", "load_asset", "load_event"]

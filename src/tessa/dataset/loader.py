"""Compatibility shim — the loader now lives in the standalone ``asset_loader``
package (``packages/asset_loader`` in this workspace). Import from :mod:`asset_loader`
directly in new code; this module re-exports the public API so existing
``tessa.dataset.loader`` imports keep working.
"""

from asset_loader import discover_files, discover_sources, load_asset, load_event

__all__ = ["discover_files", "discover_sources", "load_asset", "load_event"]

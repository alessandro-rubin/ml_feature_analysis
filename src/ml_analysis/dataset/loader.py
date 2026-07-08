"""Compatibility shim — the loader now lives in the standalone ``sparq``
package (``packages/sparq`` in this workspace). Import from :mod:`sparq`
directly in new code; this module re-exports the public API so existing
``ml_analysis.dataset.loader`` imports keep working.
"""

from sparq import discover_files, discover_sources, load_asset, load_event

__all__ = ["discover_files", "discover_sources", "load_asset", "load_event"]

"""Typed result wrapper, persistent store, static report, figure factory.

The figure factory lives in the ``figures`` submodule and is imported
lazily (it pulls in matplotlib) — use ``Run.figures()`` or
``from ml_analysis.results.figures import figures_for_result``.
"""

from ml_analysis.results.report import render_html, write_report
from ml_analysis.results.result import AnalysisResult
from ml_analysis.results.store import ResultStore

__all__ = ["AnalysisResult", "ResultStore", "render_html", "write_report"]

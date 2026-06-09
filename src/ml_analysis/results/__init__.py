"""Typed result wrapper, persistent store, static report."""

from ml_analysis.results.report import render_html, write_report
from ml_analysis.results.result import AnalysisResult
from ml_analysis.results.store import ResultStore

__all__ = ["AnalysisResult", "ResultStore", "render_html", "write_report"]

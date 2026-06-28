"""ml_analysis: modular polars-first toolkit for time-series predictive
maintenance and fault identification."""

from ml_analysis.config import Config
from ml_analysis.dataset import Dataset
from ml_analysis.features import WindowSpec, materialize
from ml_analysis.results import AnalysisResult, ResultStore
from ml_analysis.run import Run

__all__ = [
    "AnalysisResult",
    "Config",
    "Dataset",
    "ResultStore",
    "Run",
    "WindowSpec",
    "materialize",
]
__version__ = "0.2.0"

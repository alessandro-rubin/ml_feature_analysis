"""TESSA — Time-series Event Statistics and Separability Analysis.

Modular polars-first toolkit for time-series predictive maintenance
and fault identification.
"""

from tessa.config import Config
from tessa.dataset import Dataset
from tessa.features import WindowSpec, materialize
from tessa.results import AnalysisResult, ResultStore
from tessa.run import Run

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

"""Shared synthetic dataset for the AI-layer tests.

The generator lives in :mod:`tessa.ai.demo_data` so the tests, the ``--make-demo``
CLI flag and the documentation all describe the same data. It is not imported
from ``demo.py``, which calls ``matplotlib.use("Agg")`` and
``sys.stdout.reconfigure`` at import time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessa.ai.demo_data import generate


@pytest.fixture(scope="session")
def _synthetic(tmp_path_factory) -> tuple[Path, Path]:
    return generate(tmp_path_factory.mktemp("synthetic_data"))


@pytest.fixture(scope="session")
def synthetic_root(_synthetic) -> Path:
    """Data root holding per-asset parquet files."""
    return _synthetic[0]


@pytest.fixture(scope="session")
def synthetic_labels(_synthetic) -> Path:
    """Label table for the synthetic data."""
    return _synthetic[1]

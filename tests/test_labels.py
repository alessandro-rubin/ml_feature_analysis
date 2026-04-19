from datetime import datetime

import polars as pl
import pytest

from ml_analysis import Config
from ml_analysis.labels import validate


def test_validate_passes_clean():
    cfg = Config()
    df = pl.DataFrame(
        {
            "asset_id": ["A1"],
            "start": [datetime(2024, 1, 1)],
            "end": [datetime(2024, 1, 2)],
            "class": ["TP"],
        }
    )
    out = validate(df, cfg)
    assert out.schema["start"].is_temporal()


def test_validate_missing_col():
    cfg = Config()
    df = pl.DataFrame({"asset_id": ["A1"], "start": ["2024-01-01"], "end": ["2024-01-02"]})
    with pytest.raises(ValueError, match="missing columns"):
        validate(df, cfg)


def test_validate_coerces_strings():
    cfg = Config()
    df = pl.DataFrame(
        {
            "asset_id": ["A1"],
            "start": ["2024-01-01"],
            "end": ["2024-01-02"],
            "class": ["TP"],
        }
    )
    out = validate(df, cfg)
    assert out["start"][0] == datetime(2024, 1, 1)

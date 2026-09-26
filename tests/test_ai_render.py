"""Tool payloads must be small, uniformly shaped and JSON-safe."""

from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tessa.ai.render import array_payload, frame_payload, jsonable, result_payload
from tessa.results import AnalysisResult


@pytest.mark.parametrize(
    "value,expected",
    [
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        (np.float64("nan"), None),
        (np.int64(3), 3),
        (np.bool_(True), True),
        (True, True),
        (None, None),
        ("s", "s"),
    ],
)
def test_jsonable_scalars(value, expected):
    assert jsonable(value) is expected or jsonable(value) == expected


def test_non_finite_never_becomes_a_string():
    """The old layer used fillna("NaN"), which made numeric columns stringly."""
    out = frame_payload(pd.DataFrame({"a": [1.0, float("nan")]}))
    assert out["rows"][1]["a"] is None
    assert "NaN" not in json.dumps(out)


def test_floats_are_rounded_to_six_significant_figures():
    assert jsonable(1.234567891234) == 1.23457
    assert jsonable(0.000123456789) == 0.000123457
    assert jsonable(0.0) == 0.0


def test_datetimes_become_iso_strings():
    out = frame_payload(pd.DataFrame({"t": [dt.datetime(2024, 1, 2, 3, 4, 5)]}))
    assert out["rows"][0]["t"] == "2024-01-02T03:04:05"


def test_row_cap_and_truncation_flag():
    df = pd.DataFrame({"a": range(100)})
    out = frame_payload(df, max_rows=10)
    assert out["n_rows"] == 100 and out["n_rows_returned"] == 10
    assert out["truncated"] is True and "get_result" in out["note"]


def test_no_truncation_flag_when_everything_fits():
    out = frame_payload(pd.DataFrame({"a": [1, 2]}), max_rows=10)
    assert "truncated" not in out


def test_column_cap_reports_what_it_dropped():
    df = pd.DataFrame({f"c{i}": [i] for i in range(40)})
    out = frame_payload(df, max_cols=5)
    assert len(out["columns"]) == 5
    assert len(out["columns_omitted"]) == 35


def test_byte_cap_shrinks_oversized_payloads():
    df = pd.DataFrame({"text": ["x" * 500 for _ in range(200)]})
    out = frame_payload(df, max_rows=200, byte_cap=4000)
    assert len(json.dumps(out)) <= 8000
    assert out["truncated"] is True


def test_sort_offset_and_column_selection():
    df = pd.DataFrame({"a": [3, 1, 2], "b": [10, 20, 30], "c": [0, 0, 0]})
    out = frame_payload(df, columns=["a", "b"], sort_by="a", ascending=True)
    assert out["columns"] == ["a", "b"]
    assert [r["a"] for r in out["rows"]] == [1, 2, 3]
    assert frame_payload(df, sort_by="a", ascending=True, offset=1)["rows"][0]["a"] == 2


def test_unknown_column_requests_are_reported_not_raised():
    out = frame_payload(pd.DataFrame({"a": [1]}), columns=["a", "ghost"], sort_by="alsoghost")
    assert set(out["unknown_columns_requested"]) == {"ghost", "alsoghost"}


def test_polars_frames_and_series_are_accepted():
    assert frame_payload(pl.DataFrame({"a": [1, 2]}))["n_rows"] == 2
    assert frame_payload(pd.Series([1, 2], name="s"))["columns"] == ["s"]


# ── arrays ───────────────────────────────────────────────────────────────────


def test_arrays_are_summarized_not_inlined():
    out = array_payload(np.arange(1000, dtype=float))
    assert out["shape"] == [1000]
    assert "rows" not in out and "values" not in out
    assert out["min"] == 0.0 and out["max"] == 999.0


def test_array_summary_ignores_non_finite():
    out = array_payload(np.array([1.0, np.nan, 3.0]))
    assert out["n_finite"] == 2 and out["max"] == 3.0


def test_empty_array_has_no_stats():
    out = array_payload(np.array([]))
    assert out["shape"] == [0] and "mean" not in out


# ── results ──────────────────────────────────────────────────────────────────


def _result() -> AnalysisResult:
    return AnalysisResult.from_raw(
        "demo",
        {
            "summary": pd.DataFrame({"metric": [1.0, float("nan")]}),
            "perm_scores": np.arange(200, dtype=float),
            "verdict": "separable",
            "model": object(),
        },
    )


def test_result_headline_omits_frame_contents_by_default():
    out = result_payload(_result(), frames=False)
    assert out["frames_available"] == {"summary": [2, 1]}
    assert "frames" not in out
    assert out["scalars"]["verdict"] == "separable"
    # Fitted estimators are not serializable; report the key, not the object.
    assert out["objects_not_serialized"] == ["model"]
    json.dumps(out)


def test_result_with_frames_is_still_json_safe():
    out = result_payload(_result(), frames=True)
    assert out["frames"]["summary"]["rows"][1]["metric"] is None
    assert not math.isnan(json.loads(json.dumps(out))["frames"]["summary"]["n_rows"])

"""The JSON boundary for tool payloads.

Every tool result becomes permanent context for the model, so nothing reaches it
un-truncated. This module is the single place where frames, arrays and scalars
are turned into small, uniformly-shaped, JSON-safe dicts.

Two rules worth stating, because getting them wrong is subtle:

* **NaN and Inf become ``None``**, never the string ``"NaN"``. The old AI layer
  used ``df.fillna("NaN")``, which turned numeric columns into mixed str/float
  and quietly invited the model into string comparisons on numbers.
* **Arrays are never inlined.** ``perm_scores`` is 200 floats and out-of-fold
  predictions are far larger; a shape-and-percentiles summary carries the same
  information for a fraction of the context.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

__all__ = [
    "MAX_ROWS",
    "BYTE_CAP",
    "jsonable",
    "frame_payload",
    "array_payload",
    "result_payload",
    "fit_bytes",
]

MAX_ROWS = 30
MAX_COLS = 25
BYTE_CAP = 20_000
SIGFIGS = 6


def _round_sig(x: float, sig: int = SIGFIGS) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def jsonable(value: Any) -> Any:
    """Convert a value into something ``json.dumps`` accepts.

    Non-finite floats collapse to ``None`` so the model never sees ``NaN`` as a
    string or as invalid JSON.
    """
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return _round_sig(f) if math.isfinite(f) else None
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, _dt.timedelta):
        return value.total_seconds()
    if isinstance(value, (np.ndarray, list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def fit_bytes(payload: Any, cap: int = BYTE_CAP) -> tuple[Any, bool]:
    """Shrink ``payload["rows"]`` until it serializes under ``cap`` bytes."""
    if not isinstance(payload, dict) or "rows" not in payload:
        return payload, False
    trimmed = False
    rows = payload["rows"]
    while rows and len(json.dumps(payload, default=str)) > cap:
        rows = rows[: max(1, len(rows) // 2)]
        payload = {**payload, "rows": rows, "n_rows_returned": len(rows)}
        trimmed = True
        if len(rows) == 1:
            break
    if trimmed:
        payload["truncated"] = True
        payload["note"] = "Payload was trimmed to fit; call get_result for more rows."
    return payload, trimmed


def _to_pandas(df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


def frame_payload(
    df: pd.DataFrame | pl.DataFrame | pd.Series,
    *,
    max_rows: int = MAX_ROWS,
    max_cols: int = MAX_COLS,
    columns: list[str] | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
    offset: int = 0,
    byte_cap: int = BYTE_CAP,
) -> dict[str, Any]:
    """Render a dataframe as a bounded, JSON-safe dict."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    pdf = _to_pandas(df)
    total_rows, total_cols = pdf.shape

    if columns:
        keep = [c for c in columns if c in pdf.columns]
        missing = [c for c in columns if c not in pdf.columns]
        if keep:
            pdf = pdf[keep]
    else:
        missing = []

    if sort_by and sort_by in pdf.columns:
        pdf = pdf.sort_values(sort_by, ascending=ascending, kind="stable")
    elif sort_by:
        missing = [*missing, sort_by]

    dropped_cols = list(pdf.columns[max_cols:])
    pdf = pdf.iloc[:, :max_cols]
    window = pdf.iloc[offset : offset + max_rows]

    payload: dict[str, Any] = {
        "columns": [str(c) for c in pdf.columns],
        "n_rows": int(total_rows),
        "n_cols": int(total_cols),
        "n_rows_returned": int(len(window)),
        "offset": int(offset),
        "rows": [
            {str(k): jsonable(v) for k, v in rec.items()} for rec in window.to_dict("records")
        ],
    }
    truncated = len(window) < total_rows or bool(dropped_cols)
    if dropped_cols:
        payload["columns_omitted"] = [str(c) for c in dropped_cols]
    if missing:
        payload["unknown_columns_requested"] = missing
    if truncated:
        payload["truncated"] = True
        payload["note"] = "Use get_result(..., top_n=, offset=, columns=, sort_by=) for more."
    payload, _ = fit_bytes(payload, byte_cap)
    return payload


def array_payload(arr: np.ndarray) -> dict[str, Any]:
    """Summarize an array. Values are never inlined."""
    a = np.asarray(arr)
    out: dict[str, Any] = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if a.size and np.issubdtype(a.dtype, np.number):
        finite = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a
        if finite.size:
            out.update(
                n_finite=int(finite.size),
                min=jsonable(np.min(finite)),
                p05=jsonable(np.percentile(finite, 5)),
                p50=jsonable(np.percentile(finite, 50)),
                p95=jsonable(np.percentile(finite, 95)),
                max=jsonable(np.max(finite)),
                mean=jsonable(np.mean(finite)),
            )
    return out


def result_payload(result: Any, *, max_rows: int = MAX_ROWS, frames: bool = True) -> dict[str, Any]:
    """Render an :class:`~tessa.results.AnalysisResult` as a headline dict.

    With ``frames=False`` only the *shapes* of the frames are reported — the
    default for ``run_analysis``, so a completed analysis costs a few hundred
    tokens and the model pulls the rows it actually wants via ``get_result``.
    """
    payload: dict[str, Any] = {
        "analysis": result.name,
        "scalars": {k: jsonable(v) for k, v in result.scalars.items()},
        "frames_available": {name: list(_to_pandas(f).shape) for name, f in result.frames.items()},
        "arrays_available": {name: array_payload(a) for name, a in result.arrays.items()},
    }
    if result.objects:
        # Matches AnalysisResult's contract: these are dropped by ResultStore.
        payload["objects_not_serialized"] = sorted(result.objects)
    if frames:
        payload["frames"] = {
            name: frame_payload(f, max_rows=max_rows) for name, f in result.frames.items()
        }
    return payload

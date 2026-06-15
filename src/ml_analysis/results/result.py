"""A uniform, typed view over heterogeneous analysis outputs.

Existing analyses return plain dicts mixing DataFrames, numpy arrays,
scalars, and fitted models. `AnalysisResult.from_raw` sorts that mix into
typed buckets so notebooks, the store, and the dashboard can consume any
analysis the same way without caring which one produced it:

- ``frames``  — pandas/polars DataFrames (rankings, summaries, fold tables)
- ``arrays``  — numpy arrays (scores, OOF predictions, permutation nulls)
- ``scalars`` — str / int / float / bool values
- ``objects`` — everything else (fitted models, reports); not serialized
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


@dataclass(frozen=True)
class AnalysisResult:
    name: str
    frames: dict[str, pd.DataFrame | pl.DataFrame] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)
    objects: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, name: str, raw: Any) -> "AnalysisResult":
        if isinstance(raw, AnalysisResult):
            return raw
        if not isinstance(raw, dict):
            return cls(name=name, objects={"value": raw})
        frames: dict = {}
        arrays: dict = {}
        scalars: dict = {}
        objects: dict = {}
        for key, val in raw.items():
            if isinstance(val, (pd.DataFrame, pl.DataFrame)):
                frames[key] = val
            elif isinstance(val, pd.Series):
                frames[key] = val.to_frame()
            elif isinstance(val, np.ndarray):
                arrays[key] = val
            elif isinstance(val, (str, int, float, bool)) or val is None:
                scalars[key] = val
            elif (
                isinstance(val, (list, tuple))
                and val
                and all(isinstance(x, (str, int, float, bool)) for x in val)
            ):
                scalars[key] = list(val)
            else:
                objects[key] = val
        return cls(
            name=name, frames=frames, arrays=arrays, scalars=scalars, objects=objects
        )

    def summary(self, max_rows: int = 10) -> str:
        """Human-readable digest: scalars + the head of each frame."""
        lines = [f"== {self.name} =="]
        for k, v in self.scalars.items():
            lines.append(f"{k}: {v}")
        for k, df in self.frames.items():
            pdf = df.to_pandas() if isinstance(df, pl.DataFrame) else df
            lines.append(f"\n[{k}] ({len(pdf)} rows)")
            lines.append(pdf.head(max_rows).to_string())
        if self.arrays:
            lines.append(
                "\narrays: "
                + ", ".join(f"{k}{list(v.shape)}" for k, v in self.arrays.items())
            )
        if self.objects:
            lines.append("objects (not serialized): " + ", ".join(self.objects))
        return "\n".join(lines)

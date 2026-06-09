"""Persist analysis runs to disk with a reproducibility manifest.

Layout of one run directory::

    <root>/<run_name>/
        manifest.json                 # config, versions, data fingerprint
        <analysis>/scalars.json
        <analysis>/<frame>.parquet    # every DataFrame in the result
        <analysis>/<array>.npy        # every numpy array in the result

Fitted models and other live objects are *not* serialized; their keys are
recorded in the manifest so a loaded run is explicit about what's missing.
The dashboard and the static report read these directories — they never
recompute analyses.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ml_analysis.config import Config
from ml_analysis.results.result import AnalysisResult


def _versions() -> dict[str, str]:
    import sklearn

    import ml_analysis

    return {
        "ml_analysis": ml_analysis.__version__,
        "polars": pl.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }


def data_fingerprint(df: pl.DataFrame) -> dict[str, Any]:
    """Cheap, stable identity for the analysis table (not cryptographic)."""
    return {
        "n_rows": df.height,
        "n_cols": df.width,
        "columns": df.columns,
        "row_hash_sum": str(int(df.hash_rows().cast(pl.UInt64).sum() or 0)),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


class ResultStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def runs(self) -> list[str]:
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and (p / "manifest.json").exists()
        )

    def save_run(
        self,
        results: dict[str, Any],
        cfg: Config,
        df: pl.DataFrame | None = None,
        name: str | None = None,
        extra_manifest: dict | None = None,
    ) -> Path:
        """Serialize a `{analysis_name: raw result}` mapping to one run dir."""
        run_name = name or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        run_dir = self.root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": _jsonable(dataclasses.asdict(cfg)),
            "versions": _versions(),
            "analyses": {},
        }
        if df is not None:
            manifest["data_fingerprint"] = data_fingerprint(df)
        if extra_manifest:
            manifest.update(_jsonable(extra_manifest))

        for analysis_name, raw in results.items():
            res = AnalysisResult.from_raw(analysis_name, raw)
            a_dir = run_dir / analysis_name
            a_dir.mkdir(exist_ok=True)
            for key, frame in res.frames.items():
                pdf = frame if isinstance(frame, pl.DataFrame) else pl.from_pandas(
                    frame.reset_index()
                    if frame.index.name or not frame.index.equals(
                        pd.RangeIndex(len(frame))
                    )
                    else frame
                )
                pdf.write_parquet(a_dir / f"{key}.parquet")
            for key, arr in res.arrays.items():
                np.save(a_dir / f"{key}.npy", arr)
            (a_dir / "scalars.json").write_text(
                json.dumps(_jsonable(res.scalars), indent=2)
            )
            manifest["analyses"][analysis_name] = {
                "frames": sorted(res.frames),
                "arrays": sorted(res.arrays),
                "scalars": sorted(res.scalars),
                "not_serialized": sorted(res.objects),
            }

        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return run_dir

    def load_manifest(self, run_name: str) -> dict:
        return json.loads((self.root / run_name / "manifest.json").read_text())

    def load_run(self, run_name: str) -> dict[str, AnalysisResult]:
        """Reload a run as `{analysis_name: AnalysisResult}` (objects absent)."""
        run_dir = self.root / run_name
        manifest = self.load_manifest(run_name)
        out: dict[str, AnalysisResult] = {}
        for analysis_name, entry in manifest["analyses"].items():
            a_dir = run_dir / analysis_name
            frames = {
                key: pl.read_parquet(a_dir / f"{key}.parquet")
                for key in entry["frames"]
            }
            arrays = {
                key: np.load(a_dir / f"{key}.npy", allow_pickle=False)
                for key in entry["arrays"]
            }
            scalars = json.loads((a_dir / "scalars.json").read_text())
            out[analysis_name] = AnalysisResult(
                name=analysis_name, frames=frames, arrays=arrays, scalars=scalars
            )
        return out

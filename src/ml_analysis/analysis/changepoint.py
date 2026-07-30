"""Per-asset regime-change detection via two-sided CUSUM.

Answers "when did this asset start behaving differently?" without labels.
Operates on a time-indexed table (per-sample or windowed aggregates):
for every (asset, channel), values are robustly standardized against the
channel's initial baseline (median / MAD of the first ``baseline_frac``
of samples) and run through a classic two-sided CUSUM with drift ``k``
and threshold ``h`` (both in robust-sigma units). Each threshold crossing
is reported as a changepoint and the statistic resets, so multiple regime
changes per series are caught.

No external changepoint dependency (ruptures etc.); CUSUM is a few lines,
fast at 100 channels, and its two knobs are interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sstats

from ml_analysis.analysis.base import AnalysisContext


def cusum_changepoints(
    x: np.ndarray,
    k: float = 0.5,
    h: float = 12.0,
    baseline_frac: float = 0.25,
    min_separation: int = 50,
) -> list[dict]:
    """Two-sided CUSUM on a robustly standardized series.

    ``h`` defaults to 12 robust sigmas — Monte-Carlo calibrated: on 400
    pure-noise samples the classic h=5 false-alarms constantly and h=8
    still ~15% of the time, unacceptable for 100-channel sweeps; h=12 is
    ~2% while still catching 4-sigma shifts >99% of the time with a
    ~3-sample delay. After
    a detection the statistic resets and accumulation pauses for
    ``min_separation`` samples, otherwise a sustained level shift would
    re-alarm every couple of samples.

    Returns one dict per detection: position, direction, statistic value.
    """
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if valid.sum() < 10:
        return []
    n_base = max(5, int(len(x) * baseline_frac))
    base = x[valid][:n_base]
    med = np.median(base)
    # Scale from successive differences over the *whole* series: robust to
    # level shifts (a jump is one outlier diff) and far less noisy than the
    # baseline-window MAD, whose estimation error otherwise inflates the
    # standardized series and causes false alarms.
    diffs = np.diff(x[valid])
    scale = sstats.median_abs_deviation(diffs, scale="normal") / np.sqrt(2)
    if not np.isfinite(scale) or scale == 0:
        return []
    z = (x - med) / scale

    out: list[dict] = []
    s_hi = 0.0
    s_lo = 0.0
    cooldown = 0
    for i, zi in enumerate(z):
        if not np.isfinite(zi):
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        s_hi = max(0.0, s_hi + zi - k)
        s_lo = max(0.0, s_lo - zi - k)
        if s_hi > h:
            out.append({"position": i, "direction": "up", "statistic": s_hi})
            s_hi = s_lo = 0.0
            cooldown = min_separation
        elif s_lo > h:
            out.append({"position": i, "direction": "down", "statistic": s_lo})
            s_hi = s_lo = 0.0
            cooldown = min_separation
    return out


@dataclass
class ChangepointDetection:
    name: str = "changepoint"
    requires: tuple[str, ...] = ()
    needs_labels: str = "none"
    drift: float = 0.5
    threshold: float = 12.0
    baseline_frac: float = 0.25
    min_separation: int = 50
    channels: list[str] | None = None  # default: all numeric non-id columns

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        df = ctx.filtered().to_pandas()
        ts_col = ctx.cfg.timestamp_col
        if ts_col in df.columns:
            sort_cols = (["asset_id"] if "asset_id" in df.columns else []) + [ts_col]
            df = df.sort_values(sort_cols).reset_index(drop=True)

        skip = {ctx.target_col, "event_id", "asset_id", ts_col}
        channels = self.channels or [
            c for c in df.select_dtypes(include="number").columns if c not in skip
        ]

        groups = (
            [(a, g) for a, g in df.groupby("asset_id", sort=False)]
            if "asset_id" in df.columns
            else [(None, df)]
        )

        rows = []
        for asset, g in groups:
            for ch in channels:
                for cp in cusum_changepoints(
                    g[ch].to_numpy(),
                    k=self.drift, h=self.threshold,
                    baseline_frac=self.baseline_frac,
                    min_separation=self.min_separation,
                ):
                    pos = cp["position"]
                    rows.append({
                        "asset_id": asset,
                        "channel": ch,
                        "position": pos,
                        "timestamp": (
                            g[ts_col].iloc[pos] if ts_col in g.columns else None
                        ),
                        "direction": cp["direction"],
                        "statistic": float(cp["statistic"]),
                    })

        table = pd.DataFrame(
            rows,
            columns=["asset_id", "channel", "position", "timestamp",
                     "direction", "statistic"],
        )
        per_channel = (
            table.groupby("channel").size().rename("n_changepoints").reset_index()
            if len(table)
            else pd.DataFrame(columns=["channel", "n_changepoints"])
        )
        return {
            "table": table,
            "per_channel": per_channel,
            "params": {
                "drift": self.drift,
                "threshold": self.threshold,
                "baseline_frac": self.baseline_frac,
            },
        }

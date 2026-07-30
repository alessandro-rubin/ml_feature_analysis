"""Run any analysis once per stratum value, then return per-stratum results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from tessa.analysis.base import Analysis, AnalysisContext


@dataclass
class Stratified:
    """Wraps a single inner analysis and runs it per stratum value.

    The stratum column is read from `ctx.stratify_by` (set on the context)
    or from the explicit `by` field here. Each stratum value gets its own
    sub-context with `df` filtered to that stratum and `stratify_by=None`
    (so the inner analysis doesn't recurse).
    """

    inner: Analysis
    name: str = "stratified"
    requires: tuple[str, ...] = ()
    by: str | None = None
    min_samples: int = 5

    def __post_init__(self) -> None:
        # name the wrapper after its inner so dependent analyses can reference it
        if self.name == "stratified":
            self.name = f"stratified__{self.inner.name}"

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        col = self.by or ctx.stratify_by
        if not col:
            raise ValueError("Stratified requires `by` or ctx.stratify_by to be set.")
        if col not in ctx.df.columns:
            raise ValueError(f"Stratify column not found: {col}")

        df = ctx.filtered()
        per_stratum: dict[str, Any] = {}
        skipped: dict[str, int] = {}
        for value in df[col].unique().to_list():
            sub = df.filter(pl.col(col) == value)
            if sub.height < self.min_samples:
                skipped[str(value)] = sub.height
                continue
            sub_ctx = AnalysisContext(
                df=sub,
                cfg=ctx.cfg,
                target_col=ctx.target_col,
                label_filter=None,
                stratify_by=None,
                output_dir=ctx.output_dir,
            )
            per_stratum[str(value)] = self.inner.run(sub_ctx)

        return {"by": col, "per_stratum": per_stratum, "skipped": skipped}

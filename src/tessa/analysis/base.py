"""Analysis protocol, context, and DAG runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder

from tessa.config import Config

LabelFilter = dict[str, list] | Callable[[pl.DataFrame], pl.DataFrame] | None


@dataclass
class AnalysisContext:
    """Shared state for a chain of analyses.

    `df` is typically the period-aggregate (one row per event).
    `target_col` is the label column being explained / predicted.
    `label_filter` restricts rows (e.g. {"class": ["TP", "FP"]}).
    `stratify_by` is consumed by the Stratified wrapper, not by base analyses.
    `results` accumulates outputs keyed by analysis name.
    """

    df: pl.DataFrame
    cfg: Config
    target_col: str
    label_filter: LabelFilter = None
    stratify_by: str | None = None
    output_dir: str | None = None
    results: dict[str, Any] = field(default_factory=dict)

    def filtered(self) -> pl.DataFrame:
        if self.label_filter is None:
            return self.df
        if callable(self.label_filter):
            return self.label_filter(self.df)
        out = self.df
        for col, allowed in self.label_filter.items():
            out = out.filter(pl.col(col).is_in(allowed))
        return out


@dataclass
class PreparedXY:
    X: pd.DataFrame
    y: np.ndarray
    feature_cols: list[str]
    class_names: list[str]
    encoder: LabelEncoder


def prepare_xy(ctx: AnalysisContext, drop_cols: tuple[str, ...] = ()) -> PreparedXY:
    """Filter, drop nulls, split into numeric X and encoded y."""
    df = ctx.filtered().to_pandas()

    target = ctx.target_col
    drop = set(drop_cols) | {target, "event_id", "asset_id", ctx.cfg.timestamp_col}

    feature_cols = [
        c for c in df.select_dtypes(include="number").columns if c not in drop
    ]
    X = df[feature_cols].copy()
    y_raw = df[target]

    mask = X.notna().all(axis=1) & y_raw.notna()
    X = X[mask].reset_index(drop=True)
    y_raw = y_raw[mask].reset_index(drop=True)

    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    return PreparedXY(
        X=X,
        y=y,
        feature_cols=feature_cols,
        class_names=[str(c) for c in enc.classes_],
        encoder=enc,
    )


class Analysis(Protocol):
    name: str
    requires: tuple[str, ...]

    def run(self, ctx: AnalysisContext) -> Any: ...


def run_analyses(
    analyses: list[Analysis], ctx: AnalysisContext
) -> dict[str, Any]:
    """Topo-sort by `requires` and execute. Stores each result on ctx.results."""
    by_name = {a.name: a for a in analyses}
    ordered: list[Analysis] = []
    seen: set[str] = set()
    in_progress: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        if name in in_progress:
            raise ValueError(f"Cyclic analysis dependency at: {name}")
        if name not in by_name:
            raise ValueError(f"Missing required analysis: {name}")
        in_progress.add(name)
        for dep in by_name[name].requires:
            visit(dep)
        in_progress.discard(name)
        seen.add(name)
        ordered.append(by_name[name])

    for a in analyses:
        visit(a.name)

    for a in ordered:
        ctx.results[a.name] = a.run(ctx)
    return ctx.results

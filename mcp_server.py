"""MCP server exposing the ml_analysis pipeline as AI-callable tools.

Launch as a standalone MCP server:
    python mcp_server.py

Or import the functions directly (used by agent_workflow.py).
"""
from __future__ import annotations

import uuid
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
import pandas as pd
from mcp.server.fastmcp import FastMCP

from ml_analysis import Config
from ml_analysis.labels.excel import ExcelLabelSource
from ml_analysis.dataset.builder import build
from ml_analysis.features import builtins  # noqa: F401 – registers default features/aggs
from ml_analysis.features.registry import default_registry as feat_registry
from ml_analysis.features.aggregates import default_registry as agg_registry
from ml_analysis.features.materialize import to_period
from ml_analysis.analysis import (
    FeatureImportance,
    ClassifierEvaluation,
    DistributionAnalysis,
    PairwiseSeparability,
    AnalysisContext,
)

mcp = FastMCP("ml-feature-analysis")

_SESSIONS: dict[str, dict[str, Any]] = {}
_TMPDIR = Path(tempfile.mkdtemp(prefix="ml_mcp_"))


def _new_session(cfg: Config) -> str:
    sid = str(uuid.uuid4())[:8]
    _SESSIONS[sid] = {"cfg": cfg}
    return sid


def _session(sid: str) -> dict[str, Any]:
    if sid not in _SESSIONS:
        raise ValueError(f"Unknown session '{sid}'. Call load_labels first.")
    return _SESSIONS[sid]


def _records(df: pd.DataFrame, max_rows: int = 50) -> list[dict]:
    return df.head(max_rows).fillna("NaN").to_dict(orient="records")


def _build_ctx(session_id: str, label_filter: dict | None = None) -> AnalysisContext:
    s = _session(session_id)
    if "period_path" not in s:
        raise ValueError("Call materialize_period first.")
    period_df = pl.read_parquet(s["period_path"])
    cfg: Config = s["cfg"]
    return AnalysisContext(
        df=period_df,
        cfg=cfg,
        target_col=cfg.class_col,
        label_filter=label_filter,
    )


@mcp.tool()
def list_available_features() -> dict:
    """Return all registered feature transformation and aggregator names."""
    return {
        "features": feat_registry().names(),
        "aggregators": agg_registry().names(),
    }


@mcp.tool()
def load_labels(
    excel_path: str,
    sheet: str | int = 0,
    data_root: str = "data",
    output_dir: str = "outputs",
) -> dict:
    """
    Load and validate event labels from an Excel file.
    Returns a session_id required by all subsequent tool calls.
    """
    cfg = Config(data_root=Path(data_root), output_dir=Path(output_dir))
    labels = ExcelLabelSource(path=Path(excel_path), sheet=sheet).load(cfg)
    sid = _new_session(cfg)
    labels_path = _TMPDIR / f"{sid}_labels.parquet"
    labels.write_parquet(labels_path)
    _SESSIONS[sid]["labels_path"] = labels_path

    classes = (
        sorted(labels[cfg.class_col].unique().to_list())
        if cfg.class_col in labels.columns
        else []
    )
    assets = (
        sorted(labels[cfg.asset_col].unique().to_list())
        if cfg.asset_col in labels.columns
        else []
    )
    return {
        "session_id": sid,
        "n_events": len(labels),
        "assets": assets,
        "classes": classes,
        "columns": labels.columns,
    }


@mcp.tool()
def build_dataset(session_id: str) -> dict:
    """
    Build the event LazyFrame dataset from loaded labels.
    Must be called before materialize_period.
    """
    s = _session(session_id)
    labels = pl.read_parquet(s["labels_path"])
    cfg: Config = s["cfg"]
    events = build(labels, cfg)
    _SESSIONS[session_id]["events"] = events

    asset_counts: dict[str, int] = {}
    for eid in events:
        asset = eid.split("_")[0]
        asset_counts[asset] = asset_counts.get(asset, 0) + 1

    return {
        "session_id": session_id,
        "n_events": len(events),
        "events_per_asset": asset_counts,
    }


@mcp.tool()
def materialize_period(
    session_id: str,
    sources: list[str] | None = None,
    aggregators: list[str] | None = None,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Aggregate each event's time-series into one row per event (period aggregate).
    sources: raw signal columns to aggregate (None = all numeric).
    aggregators: stat names from list_available_features, e.g. ['mean','std'].
    feature_names: per-sample derived features to include, e.g. ['x__diff1'].
    Returns table shape and column list.
    """
    s = _session(session_id)
    if "events" not in s:
        raise ValueError("Call build_dataset first.")
    cfg: Config = s["cfg"]
    period_df = to_period(
        s["events"],
        cfg,
        sources=sources,
        aggregators=aggregators,
        feature_names=feature_names,
    )
    period_path = _TMPDIR / f"{session_id}_period.parquet"
    period_df.write_parquet(period_path)
    _SESSIONS[session_id]["period_path"] = period_path

    classes = (
        sorted(period_df[cfg.class_col].unique().to_list())
        if cfg.class_col in period_df.columns
        else []
    )
    return {
        "session_id": session_id,
        "shape": list(period_df.shape),
        "n_feature_columns": sum(
            1 for c in period_df.columns
            if c not in (cfg.class_col, cfg.asset_col, "event_id", "start", "end")
        ),
        "columns": period_df.columns,
        "classes": classes,
    }


@mcp.tool()
def run_feature_importance(
    session_id: str,
    label_filter: dict | None = None,
) -> dict:
    """
    Rank features by discriminative power using Random Forest MDI, permutation
    importance, ANOVA F-test, Kruskal-Wallis H, and mutual information.
    label_filter: e.g. {"class": ["TP", "FP"]} to restrict to a class subset.
    Returns top-20 features by composite score.
    """
    ctx = _build_ctx(session_id, label_filter)
    result = FeatureImportance().run(ctx)
    table: pd.DataFrame = result["table"]
    return {
        "session_id": session_id,
        "n_features_evaluated": len(table),
        "class_names": result["class_names"],
        "top_features": _records(table, max_rows=20),
    }


@mcp.tool()
def run_classifier_evaluation(
    session_id: str,
    label_filter: dict | None = None,
    run_rf: bool = True,
    run_lgb: bool = False,
    run_xgb: bool = False,
) -> dict:
    """
    Train and evaluate classifiers on the period-aggregate feature table.
    Returns accuracy, classification report, and confusion matrix per model.
    """
    ctx = _build_ctx(session_id, label_filter)
    result = ClassifierEvaluation(run_rf=run_rf, run_lgb=run_lgb, run_xgb=run_xgb).run(ctx)
    models_out = {}
    for name, info in result["models"].items():
        models_out[name] = {
            "accuracy": round(float(info["accuracy"]), 4),
            "classification_report": info["report"],
            "confusion_matrix": info["confusion_matrix"].tolist(),
        }
    return {
        "session_id": session_id,
        "class_names": result["class_names"],
        "models": models_out,
    }


@mcp.tool()
def run_pairwise_separability(
    session_id: str,
    pairs: list[list[str]] | None = None,
    top_n: int | None = 10,
    label_filter: dict | None = None,
) -> dict:
    """
    Compute Cliff's delta, KS statistic, and AUC for each feature between class pairs.
    pairs: e.g. [["TP","FP"],["TP","TN"]] — None computes all pairs.
    top_n: return top N features per pair ranked by |Cliff's delta| magnitude.
    """
    ctx = _build_ctx(session_id, label_filter)
    typed_pairs = [tuple(p) for p in pairs] if pairs else None  # type: ignore[arg-type]
    result = PairwiseSeparability(pairs=typed_pairs, top_n=top_n).run(ctx)
    summary = {
        f"{a}_vs_{b}": _records(df)
        for (a, b), df in result["pairs"].items()
    }
    return {
        "session_id": session_id,
        "pairs_analyzed": list(summary.keys()),
        "results": summary,
    }


@mcp.tool()
def run_distribution_analysis(
    session_id: str,
    label_filter: dict | None = None,
) -> dict:
    """
    Compute per-class feature distributions (quantiles) and Kruskal-Wallis test.
    Returns a summary with KW H-statistic and p-value per feature.
    """
    ctx = _build_ctx(session_id, label_filter)
    result = DistributionAnalysis().run(ctx)
    summary_df: pd.DataFrame = result["summary"]
    return {
        "session_id": session_id,
        "n_features": len(summary_df),
        "summary": _records(summary_df),
    }


if __name__ == "__main__":
    mcp.run()

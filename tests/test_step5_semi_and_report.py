"""Step-5: semi-supervised, changepoint, correlation structure, report."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tessa import Config, Run
from tessa.analysis import (
    AnalysisContext,
    ChangepointDetection,
    CorrelationStructure,
    LabelSpreadingAnalysis,
    PULearningAnalysis,
)
from tessa.analysis.changepoint import cusum_changepoints
from tessa.results.report import render_html


def _sparse_label_df(n_per_class=60, n_labeled=8, seed=0) -> pl.DataFrame:
    """Two well-separated classes, but labels only on a few rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for cls, shift in (("healthy", 0.0), ("fault", 4.0)):
        for i in range(n_per_class):
            rows.append({
                "event_id": f"{cls}_{i}",
                "true_class": cls,
                "class": cls if i < n_labeled else None,
                "f_a": float(rng.normal(shift, 1.0)),
                "f_b": float(rng.normal(shift * 0.5, 1.0)),
            })
    return pl.DataFrame(rows)


# ── label spreading ─────────────────────────────────────────────────────────

def test_label_spreading_recovers_masked_labels():
    df = _sparse_label_df()
    ctx = AnalysisContext(df=df.drop("true_class"), cfg=Config(),
                          target_col="class")
    out = LabelSpreadingAnalysis().run(ctx)
    table = out["table"]
    truth = df["true_class"].to_list()
    acc = float(np.mean(table["predicted_label"].to_numpy() == np.array(truth)))
    assert acc > 0.9
    assert out["n_labeled"] == 16 and out["n_unlabeled"] == 104


def test_label_spreading_skipped_without_target_col():
    from tessa.analysis import run_analyses

    df = _sparse_label_df().drop("true_class", "class")
    ctx = AnalysisContext(df=df, cfg=Config())  # no target at all
    with pytest.warns(UserWarning, match="requires labels"):
        results = run_analyses([LabelSpreadingAnalysis()], ctx)
    assert "label_spreading" not in results


# ── PU learning ─────────────────────────────────────────────────────────────

def test_pu_learning_ranks_hidden_positives_first():
    df = _sparse_label_df(n_per_class=50, n_labeled=10)
    # only positives carry labels: healthy labels removed
    df = df.with_columns(
        pl.when(pl.col("class") == "fault")
        .then(pl.lit("fault")).otherwise(None).alias("class")
    )
    ctx = AnalysisContext(df=df.drop("true_class"), cfg=Config(),
                          target_col="class")
    out = PULearningAnalysis(
        positive_label="fault", n_iterations=15,
        rf_params={"n_estimators": 30, "n_jobs": 1},
    ).run(ctx)
    ranked = out["ranked_unlabeled"]
    # hidden faults should dominate the top of the ranking
    top = ranked.head(20)["event_id"].str.startswith("fault").mean()
    bottom = ranked.tail(20)["event_id"].str.startswith("fault").mean()
    assert top > 0.8 and bottom < 0.2


def test_pu_learning_requires_positive_label():
    ctx = AnalysisContext(df=_sparse_label_df().drop("true_class"),
                          cfg=Config(), target_col="class")
    with pytest.raises(ValueError, match="positive_label"):
        PULearningAnalysis().run(ctx)


# ── changepoint ─────────────────────────────────────────────────────────────

def test_cusum_finds_injected_shift():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(4, 1, 200)])
    cps = cusum_changepoints(x)
    assert cps, "no changepoint found"
    first = cps[0]
    assert first["direction"] == "up"
    assert 295 <= first["position"] <= 320  # shortly after the true shift at 300


def test_changepoint_analysis_per_asset():
    rng = np.random.default_rng(1)
    t0 = datetime(2024, 1, 1)
    frames = []
    for asset, shift_at in (("A1", 200), ("A2", None)):
        n = 400
        sig = rng.normal(0, 1, n)
        if shift_at:
            sig[shift_at:] += 5.0
        frames.append(pl.DataFrame({
            "timestamp": [t0 + timedelta(minutes=i) for i in range(n)],
            "asset_id": [asset] * n,
            "temp": sig,
        }))
    ctx = AnalysisContext(df=pl.concat(frames), cfg=Config())
    out = ChangepointDetection().run(ctx)
    table = out["table"]
    assert (table["asset_id"] == "A1").any()
    assert not (table["asset_id"] == "A2").any()  # no false alarm on A2


# ── correlation structure ───────────────────────────────────────────────────

def test_correlation_clusters_duplicates_together():
    rng = np.random.default_rng(0)
    n = 300
    base = rng.normal(0, 1, n)
    df = pl.DataFrame({
        "f_orig": base,
        "f_dup": base * 2.0 + 0.01 * rng.normal(0, 1, n),   # near-duplicate
        "f_indep": rng.normal(0, 1, n),
    })
    ctx = AnalysisContext(df=df, cfg=Config())
    out = CorrelationStructure().run(ctx)
    clusters = out["clusters"].set_index("feature")["cluster"]
    assert clusters["f_orig"] == clusters["f_dup"]
    assert clusters["f_indep"] != clusters["f_orig"]
    dups = out["duplicates"]
    assert {"f_orig", "f_dup"} == set(dups.iloc[0][["feature_a", "feature_b"]])


# ── report ──────────────────────────────────────────────────────────────────

def test_html_report_contains_everything(tmp_path: Path):
    df = _sparse_label_df(n_per_class=30, n_labeled=30).drop("true_class")
    run = Run(df, target_col="class", cfg=Config(random_state=3))
    run.separability(n_permutations=20, rf_params={"n_estimators": 20, "n_jobs": 1})
    run.correlation_structure()

    path = run.report(tmp_path / "report.html")
    html_text = path.read_text()
    assert "separability" in html_text
    assert "correlation_structure" in html_text
    assert "perm_p_value" in html_text
    assert "data:image/png;base64," in html_text  # perm_scores histogram

    # render_html also works straight from a loaded store
    run_dir = run.save(tmp_path / "runs", name="r1")
    from tessa.results import ResultStore
    loaded = ResultStore(tmp_path / "runs").load_run("r1")
    html2 = render_html(loaded, ResultStore(tmp_path / "runs").load_manifest("r1"))
    assert "separability" in html2
    assert run_dir.exists()


def test_dashboard_app_compiles():
    import py_compile

    py_compile.compile("src/tessa/dashboard/app.py", doraise=True)

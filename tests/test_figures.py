"""UI-independent figure factory: live results, stored results, report, KPIs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: tests must not require a display

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from tessa import Config, Run
from tessa.results import AnalysisResult, ResultStore
from tessa.results.figures import (
    figures_for_result,
    figures_for_run,
    headline_metrics,
)

_FAST_RF = {"n_estimators": 25, "n_jobs": 1, "random_state": 0}


def _multiclass_df(n_per_class: int = 30) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for cls, shift in (("A", 0.0), ("B", 3.0), ("C", 6.0)):
        for _ in range(n_per_class):
            rows.append(
                {
                    "event_id": f"{cls}_{rng.integers(1_000_000)}",
                    "class": cls,
                    "f_sep": float(rng.normal(shift, 1.0)),
                    "f_sep2": float(rng.normal(shift * 0.6, 1.0)),
                    "f_weak": float(rng.normal(shift * 0.1, 1.0)),
                    "f_noise": float(rng.normal(0.0, 1.0)),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def computed_run() -> Run:
    run = Run(_multiclass_df(), target_col="class", cfg=Config(random_state=0))
    run.distributions()
    run.pairwise(bootstrap_n=40)
    run.importance(permutation_repeats=2, rf_params=_FAST_RF)
    run.importance_stability(n_bootstrap=10, top_k=3, rf_params=_FAST_RF)
    run.clustering()
    run.cluster_validation(n_permutations=40)
    run.classifier(run_lgb=False, run_xgb=False, rf_params=_FAST_RF)
    run.cv_classifier(n_splits=3, rf_params=_FAST_RF)
    run.separability(n_permutations=20, rf_params=_FAST_RF)
    return run


_EXPECTED = [
    "distributions",
    "pairwise",
    "importance",
    "importance_stability",
    "clustering",
    "cluster_validation",
    "classifier",
    "cv_classifier",
    "separability",
]


def _close_all(figs_by_name) -> None:
    for figs in figs_by_name.values():
        for _, fig in figs:
            plt.close(fig)


def test_live_results_produce_titled_figures(computed_run: Run):
    figs = computed_run.figures()
    try:
        for name in _EXPECTED:
            assert figs.get(name), f"no figures for {name}"
            for title, fig in figs[name]:
                assert isinstance(title, str) and title
                assert isinstance(fig, plt.Figure)
    finally:
        _close_all(figs)


def test_clustering_live_has_scatter_and_heatmap(computed_run: Run):
    figs = computed_run.figures()
    try:
        titles = " ".join(t for t, _ in figs["clustering"]).lower()
        assert "embedding" in titles  # 2-D scatter
        assert "elbow" in titles or "k" in titles
        assert "composition" in titles or "cluster" in titles
    finally:
        _close_all(figs)


def test_figures_survive_store_round_trip(tmp_path, computed_run: Run):
    computed_run.save(tmp_path / "runs", name="r1")
    loaded = ResultStore(tmp_path / "runs").load_run("r1")
    figs = figures_for_run(loaded)
    try:
        # The plot-critical data we added to serialization must reconstruct
        # the iconic plots even though the nested dicts/models were dropped.
        for name in _EXPECTED:
            assert figs.get(name), f"no figures for {name} after reload"
        clustering_titles = " ".join(t for t, _ in figs["clustering"]).lower()
        assert "embedding" in clustering_titles  # from the stored embedding frame
    finally:
        _close_all(figs)


def test_pairwise_volcano_from_stored_pairs_long(tmp_path, computed_run: Run):
    computed_run.save(tmp_path / "runs", name="r2")
    loaded = ResultStore(tmp_path / "runs").load_run("r2")
    titled = figures_for_result(loaded["pairwise"])
    try:
        assert titled
        assert any("volcano" in t.lower() for t, _ in titled)
    finally:
        for _, fig in titled:
            plt.close(fig)


def test_unknown_result_falls_back_to_array_histograms():
    res = AnalysisResult(
        name="mystery", arrays={"scores": np.random.default_rng(0).normal(size=200)}
    )
    titled = figures_for_result(res)
    try:
        assert len(titled) == 1
        assert titled[0][0] == "scores"
    finally:
        plt.close(titled[0][1])


def test_headline_metrics_extracted(computed_run: Run):
    results = {n: AnalysisResult.from_raw(n, r) for n, r in computed_run.ctx.results.items()}
    metrics = headline_metrics(results)
    labels = {m["label"] for m in metrics}
    assert "Separability" in labels
    assert "Best k" in labels
    assert all(isinstance(m["value"], str) for m in metrics)


def test_report_embeds_curated_figures(tmp_path, computed_run: Run):
    path = computed_run.report(tmp_path / "report.html")
    html_text = path.read_text(encoding="utf-8")
    # multiple curated charts, not just one bare histogram
    assert html_text.count("data:image/png;base64,") >= 5
    assert "clustering" in html_text and "separability" in html_text

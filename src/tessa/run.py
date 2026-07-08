"""Notebook-first facade: one object from analysis table to saved results.

>>> import polars as pl
>>> from tessa import Config, Run
>>> run = Run(table, target_col="class", cfg=Config(random_state=7))
>>> run.separability().summary()
>>> run.importance().frames["table"]
>>> run.anomaly(baseline_filter={"class": ["healthy"]})
>>> run.save("outputs/runs")          # parquet + manifest, dashboard-ready

Each method instantiates the analysis (kwargs forwarded), executes it via
the shared context — so `prepare_xy` caching and `ctx.results` reuse apply
— and returns an :class:`AnalysisResult`. Calling a method again without
kwargs returns the cached result instead of refitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from tessa.analysis import (
    AnalysisContext,
    AnomalyDetection,
    ChangepointDetection,
    CorrelationStructure,
    LabelSpreadingAnalysis,
    LaggedRelations,
    PULearningAnalysis,
    ClassifierEvaluation,
    ClusterAnalysis,
    ClusterValidation,
    CrossValidatedClassifier,
    DistributionAnalysis,
    FeatureImportance,
    ImportanceStability,
    NullPolicy,
    MutualInfoNetwork,
    PairwiseSeparability,
    SeparabilityTest,
    run_analyses,
)
from tessa.config import Config
from tessa.results import AnalysisResult, ResultStore

_ANALYSES = {
    "importance": FeatureImportance,
    "classifier": ClassifierEvaluation,
    "cv_classifier": CrossValidatedClassifier,
    "distributions": DistributionAnalysis,
    "pairwise": PairwiseSeparability,
    "clustering": ClusterAnalysis,
    "cluster_validation": ClusterValidation,
    "importance_stability": ImportanceStability,
    "separability": SeparabilityTest,
    "anomaly": AnomalyDetection,
    "label_spreading": LabelSpreadingAnalysis,
    "pu_learning": PULearningAnalysis,
    "changepoint": ChangepointDetection,
    "correlation_structure": CorrelationStructure,
    "lagged_relations": LaggedRelations,
    "mi_network": MutualInfoNetwork,
}


class Run:
    def __init__(
        self,
        df: pl.DataFrame,
        target_col: str | None = None,
        cfg: Config | None = None,
        label_filter=None,
        null_policy: NullPolicy | None = None,
    ):
        self.ctx = AnalysisContext(
            df=df,
            cfg=cfg or Config(),
            target_col=target_col,
            label_filter=label_filter,
            null_policy=null_policy or NullPolicy(),
        )

    # ── generic execution ───────────────────────────────────────────────────

    def run(self, name: str, **kwargs: Any) -> AnalysisResult:
        """Run one analysis by name; cached unless kwargs force a re-run."""
        if name not in _ANALYSES:
            raise KeyError(f"Unknown analysis {name!r}; one of {sorted(_ANALYSES)}")
        if kwargs or name not in self.ctx.results:
            analysis = _ANALYSES[name](**kwargs)
            run_analyses([analysis], self.ctx)
        if name not in self.ctx.results:  # skipped (e.g. labels missing)
            raise RuntimeError(
                f"Analysis {name!r} was skipped — does it need labels the "
                "context doesn't have?"
            )
        return AnalysisResult.from_raw(name, self.ctx.results[name])

    def run_all(self, names: list[str] | None = None, **per_analysis_kwargs) -> dict:
        """Run several analyses (default: every label-compatible one)."""
        names = names or list(_ANALYSES)
        analyses = [
            _ANALYSES[n](**per_analysis_kwargs.get(n, {})) for n in names
        ]
        run_analyses(analyses, self.ctx)
        return {
            n: AnalysisResult.from_raw(n, self.ctx.results[n])
            for n in names
            if n in self.ctx.results
        }

    # ── named shortcuts ─────────────────────────────────────────────────────

    def importance(self, **kw) -> AnalysisResult:
        return self.run("importance", **kw)

    def classifier(self, **kw) -> AnalysisResult:
        return self.run("classifier", **kw)

    def cv_classifier(self, **kw) -> AnalysisResult:
        return self.run("cv_classifier", **kw)

    def distributions(self, **kw) -> AnalysisResult:
        return self.run("distributions", **kw)

    def pairwise(self, **kw) -> AnalysisResult:
        return self.run("pairwise", **kw)

    def clustering(self, **kw) -> AnalysisResult:
        return self.run("clustering", **kw)

    def cluster_validation(self, **kw) -> AnalysisResult:
        return self.run("cluster_validation", **kw)

    def importance_stability(self, **kw) -> AnalysisResult:
        return self.run("importance_stability", **kw)

    def separability(self, **kw) -> AnalysisResult:
        return self.run("separability", **kw)

    def anomaly(self, **kw) -> AnalysisResult:
        return self.run("anomaly", **kw)

    def label_spreading(self, **kw) -> AnalysisResult:
        return self.run("label_spreading", **kw)

    def pu_learning(self, **kw) -> AnalysisResult:
        return self.run("pu_learning", **kw)

    def changepoint(self, **kw) -> AnalysisResult:
        return self.run("changepoint", **kw)

    def correlation_structure(self, **kw) -> AnalysisResult:
        return self.run("correlation_structure", **kw)

    def lagged_relations(self, **kw) -> AnalysisResult:
        return self.run("lagged_relations", **kw)

    def mi_network(self, **kw) -> AnalysisResult:
        return self.run("mi_network", **kw)

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, store: ResultStore | str | Path, name: str | None = None) -> Path:
        """Persist everything computed so far (+ manifest) to a run dir."""
        if not isinstance(store, ResultStore):
            store = ResultStore(store)
        return store.save_run(
            self.ctx.results,
            cfg=self.ctx.cfg,
            df=self.ctx.df,
            name=name,
            extra_manifest={"target_col": self.ctx.target_col},
        )

    def report(self, path: str | Path, title: str = "tessa run report") -> Path:
        """Write a self-contained static HTML report of everything computed."""
        from tessa.results.report import write_report

        manifest = {"target_col": self.ctx.target_col}
        return write_report(path, self.ctx.results, manifest, title)

    def figures(self, max_figures: int = 8) -> dict:
        """Curated matplotlib figures per analysis (UI-independent).

        Returns ``{analysis_name: [(title, Figure), ...]}`` for everything
        computed so far — the same plots the dashboard and HTML report draw,
        ready to ``show`` in a notebook or ``save_fig`` to disk.
        """
        from tessa.results.figures import figures_for_run

        results = {
            n: AnalysisResult.from_raw(n, r) for n, r in self.ctx.results.items()
        }
        return figures_for_run(results, max_figures=max_figures)

    def summary(self) -> str:
        """Digest of every analysis computed so far."""
        if not self.ctx.results:
            return "No analyses run yet."
        return "\n\n".join(
            AnalysisResult.from_raw(n, r).summary()
            for n, r in self.ctx.results.items()
        )

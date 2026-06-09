"""Notebook-first facade: one object from analysis table to saved results.

>>> import polars as pl
>>> from ml_analysis import Config, Run
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

from ml_analysis.analysis import (
    AnalysisContext,
    AnomalyDetection,
    ClassifierEvaluation,
    ClusterAnalysis,
    ClusterValidation,
    CrossValidatedClassifier,
    DistributionAnalysis,
    FeatureImportance,
    ImportanceStability,
    NullPolicy,
    PairwiseSeparability,
    SeparabilityTest,
    run_analyses,
)
from ml_analysis.config import Config
from ml_analysis.results import AnalysisResult, ResultStore

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

    def summary(self) -> str:
        """Digest of every analysis computed so far."""
        if not self.ctx.results:
            return "No analyses run yet."
        return "\n\n".join(
            AnalysisResult.from_raw(n, r).summary()
            for n, r in self.ctx.results.items()
        )

"""Is class discrimination possible at all? — permutation-tested CV.

Answers the mission's third question with a significance level instead of
a bare accuracy number: a grouped/stratified cross-validated balanced
accuracy is compared against its null distribution obtained by refitting
on label-shuffled data (`sklearn.model_selection.permutation_test_score`).

Outputs:
- observed CV balanced accuracy and the permutation p-value
  ("probability of scoring this well with meaningless labels"),
- chance level (balanced accuracy of guessing = 1 / n_classes),
- label-conditioned silhouette on the standardized feature matrix —
  a geometry-based second opinion that doesn't involve a classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.preprocessing import StandardScaler

from ml_analysis.analysis.base import AnalysisContext, prepare_xy, seeded


@dataclass
class SeparabilityTest:
    name: str = "separability"
    requires: tuple[str, ...] = ()
    needs_labels: str = "full"
    n_splits: int = 5
    n_permutations: int = 200
    rf_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=None
        )
    )

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx)
        X, y = prep.X.to_numpy(dtype=float), prep.y
        n_classes = len(prep.class_names)
        if n_classes < 2:
            raise ValueError("Separability needs at least 2 classes.")

        class_counts = np.bincount(y)
        n_splits = int(min(self.n_splits, class_counts.min()))
        if n_splits < 2:
            raise ValueError(
                f"Smallest class has {class_counts.min()} rows; need >= 2."
            )
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=ctx.cfg.random_state
        )

        model = RandomForestClassifier(**seeded(self.rf_params, ctx.cfg))
        observed, perm_scores, p_value = permutation_test_score(
            model, X, y,
            cv=cv,
            scoring="balanced_accuracy",
            n_permutations=self.n_permutations,
            random_state=ctx.cfg.random_state,
            n_jobs=-1,
        )

        Z = StandardScaler().fit_transform(X)
        sil = (
            float(silhouette_score(Z, y))
            if len(np.unique(y)) > 1 and len(Z) > n_classes
            else float("nan")
        )

        chance = 1.0 / n_classes  # balanced accuracy of uninformed guessing
        verdict = (
            "separable" if p_value <= 0.05 and observed > chance else "not separable"
        )
        summary = pd.DataFrame([{
            "cv_balanced_accuracy": float(observed),
            "chance_level": chance,
            "perm_p_value": float(p_value),
            "n_permutations": self.n_permutations,
            "n_splits": n_splits,
            "silhouette_labels": sil,
            "verdict": verdict,
        }])

        return {
            "summary": summary,
            "perm_scores": np.asarray(perm_scores),
            "class_names": prep.class_names,
        }

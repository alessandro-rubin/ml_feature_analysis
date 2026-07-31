"""Is class discrimination possible at all? — permutation-tested CV.

Answers the mission's third question with a significance level instead of
a bare accuracy number: a grouped/stratified cross-validated balanced
accuracy is compared against its null distribution obtained by refitting
on label-shuffled data.

Folds are grouped by asset by default, so the observed score describes
discrimination on *unseen assets* (see :func:`tessa.analysis.base.make_cv`).

The permutation loop is written out rather than delegated to
``sklearn.model_selection.permutation_test_score`` because that function
couples two independent choices: passing ``groups`` to it both groups the
folds *and* restricts label shuffling to within each group. With one event
per asset — common here — within-group shuffling is a no-op, every
permutation reproduces the observed score and the p-value degenerates to
1.0. Keeping the two apart lets the folds be asset-disjoint while the null
stays the intended one; ``permute_within_assets`` opts into the stricter
within-asset null when the data can support it.

Outputs:
- observed CV balanced accuracy and the permutation p-value
  ("probability of scoring this well with meaningless labels"),
- chance level (balanced accuracy of guessing = 1 / n_classes),
- label-conditioned silhouette on the standardized feature matrix —
  a geometry-based second opinion that doesn't involve a classifier.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from tessa.analysis.base import AnalysisContext, CVPlan, make_cv, prepare_xy, seeded


def _cv_balanced_accuracy(model: Any, X: np.ndarray, y: np.ndarray, plan: CVPlan) -> float:
    """Mean balanced accuracy over ``plan``'s folds, refitting each time."""
    scores = []
    for tr, te in plan.split(X, y):
        est = clone(model)
        est.fit(X[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], est.predict(X[te])))
    return float(np.mean(scores)) if scores else float("nan")


def _shuffle(y: np.ndarray, groups: np.ndarray | None, rng) -> np.ndarray:
    """Permute ``y`` globally, or within each group when ``groups`` is given."""
    if groups is None:
        return y[rng.permutation(len(y))]
    idx = np.arange(len(y))
    for g in np.unique(groups):
        m = groups == g
        idx[m] = rng.permutation(idx[m])
    return y[idx]


@dataclass
class SeparabilityTest:
    name: str = "separability"
    requires: tuple[str, ...] = ()
    needs_labels: str = "full"
    n_splits: int = 5
    n_permutations: int = 200
    group_by_asset: bool = True
    permute_within_assets: bool = False
    rf_params: dict = field(
        default_factory=lambda: dict(n_estimators=200, max_depth=None, n_jobs=-1, random_state=None)
    )

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx)
        X, y = prep.X.to_numpy(dtype=float), prep.y
        n_classes = len(prep.class_names)
        if n_classes < 2:
            raise ValueError("Separability needs at least 2 classes.")

        plan = make_cv(prep, ctx, self.n_splits, group_by_asset=self.group_by_asset)
        n_splits = plan.n_splits

        # Which null to test against: global label exchangeability (default),
        # or exchangeability within an asset, which additionally controls for
        # asset identity as a confounder but needs assets that carry >1 label.
        perm_groups = plan.groups if self.permute_within_assets else None
        if perm_groups is not None:
            informative = sum(
                len(np.unique(y[perm_groups == g])) > 1 for g in np.unique(perm_groups)
            )
            if informative < 2:
                warnings.warn(
                    f"permute_within_assets=True but only {informative} asset(s) "
                    "carry more than one class; within-asset shuffling would be "
                    "near-degenerate. Falling back to global permutation.",
                    stacklevel=2,
                )
                perm_groups = None

        model = RandomForestClassifier(**seeded(self.rf_params, ctx.cfg))
        observed = _cv_balanced_accuracy(model, X, y, plan)
        rng = np.random.RandomState(ctx.cfg.random_state)
        perm_scores = np.asarray(
            Parallel(n_jobs=-1)(
                delayed(_cv_balanced_accuracy)(model, X, _shuffle(y, perm_groups, rng), plan)
                for _ in range(self.n_permutations)
            ),
            dtype=float,
        )
        # sklearn's convention: the observed score counts as one draw, so the
        # p-value can never be exactly 0.
        p_value = float((np.sum(perm_scores >= observed) + 1.0) / (self.n_permutations + 1))

        Z = StandardScaler().fit_transform(X)
        sil = (
            float(silhouette_score(Z, y))
            if len(np.unique(y)) > 1 and len(Z) > n_classes
            else float("nan")
        )

        chance = 1.0 / n_classes  # balanced accuracy of uninformed guessing
        verdict = "separable" if p_value <= 0.05 and observed > chance else "not separable"
        summary = pd.DataFrame(
            [
                {
                    "cv_balanced_accuracy": float(observed),
                    "chance_level": chance,
                    "perm_p_value": float(p_value),
                    "n_permutations": self.n_permutations,
                    "n_splits": n_splits,
                    "silhouette_labels": sil,
                    "verdict": verdict,
                    "cv_scheme": plan.scheme,
                    "n_assets": plan.n_groups,
                    "permutation_null": ("within_asset" if perm_groups is not None else "global"),
                }
            ]
        )

        return {
            "summary": summary,
            "perm_scores": perm_scores,
            "class_names": prep.class_names,
            "cv_scheme": plan.scheme,
            "cv_grouped": plan.grouped,
            "cv_reason": plan.reason,
        }

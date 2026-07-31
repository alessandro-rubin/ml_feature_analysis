"""Semi-supervised analyses: a few labels guide an unlabeled majority.

Two complementary entry points, both with ``needs_labels="partial"`` —
the target column must exist, but most rows may be null:

- `LabelSpreadingAnalysis` — graph-based label propagation
  (`sklearn.semi_supervised.LabelSpreading` on a kNN graph of the
  standardized feature matrix). Output: a predicted label and confidence
  for every row, including the originally unlabeled ones.
- `PULearningAnalysis` — positive-unlabeled bagging (Mordelet & Vert):
  only *positive* examples are trusted (e.g. confirmed failures); each
  bagging round trains positives vs a random unlabeled subsample, and an
  unlabeled row's score is its mean out-of-bag probability of being
  positive. Output: a ranking of unlabeled rows by "looks like a failure".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

from tessa.analysis.base import AnalysisContext, prepare_xy, seeded


def _sparse_labels(ctx: AnalysisContext, prep) -> pd.Series:
    """Target values aligned to prep.X rows (nulls = unlabeled)."""
    if ctx.target_col is None:
        raise ValueError(
            "Semi-supervised analyses need target_col set (its values may be mostly null)."
        )
    df = ctx.filtered().to_pandas()
    return df[ctx.target_col].iloc[prep.row_index].reset_index(drop=True)


@dataclass
class LabelSpreadingAnalysis:
    name: str = "label_spreading"
    requires: tuple[str, ...] = ()
    needs_labels: str = "partial"
    n_neighbors: int = 7
    alpha: float = 0.2
    max_iter: int = 100

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx, ignore_target=True)
        y_sparse = _sparse_labels(ctx, prep)
        labeled = y_sparse.notna()
        if labeled.sum() < 2:
            raise ValueError(f"Need >= 2 labeled rows; have {int(labeled.sum())}.")
        classes = sorted(y_sparse[labeled].unique())
        class_to_int = {c: i for i, c in enumerate(classes)}
        y_enc = np.full(len(y_sparse), -1, dtype=int)  # -1 = unlabeled
        y_enc[labeled.to_numpy()] = [class_to_int[v] for v in y_sparse[labeled]]

        Z = StandardScaler().fit_transform(prep.X.to_numpy(dtype=float))
        k = min(self.n_neighbors, max(2, len(Z) - 1))
        model = LabelSpreading(
            kernel="knn", n_neighbors=k, alpha=self.alpha, max_iter=self.max_iter
        ).fit(Z, y_enc)

        proba = model.label_distributions_
        pred = np.asarray(classes)[np.argmax(proba, axis=1)]
        confidence = proba.max(axis=1)

        table = pd.DataFrame(
            {
                "given_label": y_sparse,
                "predicted_label": pred,
                "confidence": confidence,
            }
        )
        if prep.ids is not None:
            table = pd.concat([prep.ids, table], axis=1)

        return {
            "table": table,
            "proba": proba,
            "classes": [str(c) for c in classes],
            "n_labeled": int(labeled.sum()),
            "n_unlabeled": int((~labeled).sum()),
            "model": model,
        }


@dataclass
class PULearningAnalysis:
    name: str = "pu_learning"
    requires: tuple[str, ...] = ()
    needs_labels: str = "partial"
    positive_label: Any = None  # required: which target value means "positive"
    n_iterations: int = 30
    rf_params: dict = field(
        default_factory=lambda: dict(n_estimators=100, n_jobs=-1, random_state=None)
    )

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        if self.positive_label is None:
            raise ValueError("PULearningAnalysis requires positive_label=...")
        prep = prepare_xy(ctx, ignore_target=True)
        y_sparse = _sparse_labels(ctx, prep)
        pos_mask = (y_sparse == self.positive_label).to_numpy()
        unl_mask = ~pos_mask
        n_pos = int(pos_mask.sum())
        if n_pos < 2:
            raise ValueError(f"Need >= 2 positive rows; have {n_pos}.")
        if unl_mask.sum() < 2:
            raise ValueError("Need >= 2 unlabeled rows.")

        X = prep.X.to_numpy(dtype=float)
        pos_idx = np.where(pos_mask)[0]
        unl_idx = np.where(unl_mask)[0]
        rng = np.random.default_rng(ctx.cfg.random_state)

        score_sum = np.zeros(len(X))
        score_cnt = np.zeros(len(X))
        sample_size = min(n_pos, len(unl_idx))
        for _ in range(self.n_iterations):
            neg_sample = rng.choice(unl_idx, size=sample_size, replace=False)
            oob = np.setdiff1d(unl_idx, neg_sample, assume_unique=True)
            if len(oob) == 0:
                oob = unl_idx
            params = seeded(self.rf_params, ctx.cfg)
            params["random_state"] = int(rng.integers(0, 2**31 - 1))
            rf = RandomForestClassifier(**params)
            rf.fit(
                np.vstack([X[pos_idx], X[neg_sample]]),
                np.concatenate([np.ones(n_pos), np.zeros(sample_size)]),
            )
            p = rf.predict_proba(X[oob])[:, 1]
            score_sum[oob] += p
            score_cnt[oob] += 1

        with np.errstate(invalid="ignore"):
            score = score_sum / score_cnt
        score[pos_idx] = np.nan  # known positives are not scored

        table = pd.DataFrame(
            {
                "given_label": y_sparse,
                "is_known_positive": pos_mask,
                "pu_score": score,
            }
        )
        if prep.ids is not None:
            table = pd.concat([prep.ids, table], axis=1)
        ranked = (
            table[~table["is_known_positive"]]
            .sort_values("pu_score", ascending=False)
            .reset_index(drop=True)
        )

        return {
            "table": table,
            "ranked_unlabeled": ranked,
            "n_positive": n_pos,
            "n_unlabeled": int(unl_mask.sum()),
            "n_iterations": self.n_iterations,
        }

"""Cross-validated classifier evaluation.

A single train/test split — what ``ClassifierEvaluation`` does — is
fragile on small / imbalanced data. ``CrossValidatedClassifier`` runs
stratified k-fold CV and reports a battery of metrics that together
corroborate a single accuracy number:

- Accuracy, balanced accuracy (immune to class imbalance),
- F1 (macro / weighted), precision, recall,
- Matthews Correlation Coefficient, Cohen's kappa (agreement),
- Log loss, Brier score (binary) — calibration / probabilistic quality,
- ROC-AUC (OVR for multi-class), PR-AUC (binary).

Each metric is summarized as mean +/- std across folds with a per-fold
table for inspection. For binary problems an ECE (expected calibration
error) is also reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from tessa.analysis.base import AnalysisContext, prepare_xy


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """ECE for binary probabilistic predictions."""
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def _safe(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except (ValueError, IndexError):
        return float("nan")


@dataclass
class CrossValidatedClassifier:
    name: str = "cv_classifier"
    requires: tuple[str, ...] = ()
    n_splits: int = 5
    rf_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
        )
    )

    def _per_fold_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
        n_classes: int,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {
            "accuracy": _safe(accuracy_score, y_true, y_pred),
            "balanced_accuracy": _safe(balanced_accuracy_score, y_true, y_pred),
            "f1_macro": _safe(f1_score, y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": _safe(
                f1_score, y_true, y_pred, average="weighted", zero_division=0
            ),
            "precision_macro": _safe(
                precision_score, y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": _safe(
                recall_score, y_true, y_pred, average="macro", zero_division=0
            ),
            "mcc": _safe(matthews_corrcoef, y_true, y_pred),
            "cohen_kappa": _safe(cohen_kappa_score, y_true, y_pred),
        }
        if y_proba is not None:
            metrics["log_loss"] = _safe(
                log_loss, y_true, y_proba, labels=list(range(n_classes))
            )
            if n_classes == 2:
                pos = y_proba[:, 1]
                metrics["roc_auc"] = _safe(roc_auc_score, y_true, pos)
                metrics["pr_auc"] = _safe(average_precision_score, y_true, pos)
                metrics["brier"] = _safe(brier_score_loss, y_true, pos)
                metrics["ece"] = _expected_calibration_error(y_true, pos)
            else:
                metrics["roc_auc_ovr"] = _safe(
                    roc_auc_score, y_true, y_proba, multi_class="ovr", average="macro",
                    labels=list(range(n_classes)),
                )
        return metrics

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx)
        X = prep.X.values
        y = prep.y
        n_classes = len(prep.class_names)
        if len(X) < self.n_splits * 2:
            raise ValueError(
                f"Need at least {self.n_splits * 2} samples for CV; have {len(X)}"
            )

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=ctx.cfg.random_state
        )

        fold_rows = []
        oof_pred = np.empty(len(y), dtype=int)
        oof_proba: np.ndarray | None = (
            np.zeros((len(y), n_classes)) if n_classes >= 2 else None
        )

        for k, (tr, te) in enumerate(skf.split(X, y)):
            model = RandomForestClassifier(**self.rf_params)
            model.fit(X[tr], y[tr])
            preds = model.predict(X[te])
            proba: np.ndarray | None = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X[te])
                # align columns to all classes (rare missing class in train fold)
                aligned = np.zeros((len(te), n_classes))
                for j, cls in enumerate(model.classes_):
                    aligned[:, int(cls)] = proba[:, j]
                proba = aligned
                oof_proba[te] = aligned  # type: ignore[index]
            oof_pred[te] = preds
            row = {"fold": k}
            row.update(self._per_fold_metrics(y[te], preds, proba, n_classes))
            fold_rows.append(row)

        per_fold = pd.DataFrame(fold_rows).set_index("fold")
        summary = pd.DataFrame({
            "mean": per_fold.mean(axis=0),
            "std": per_fold.std(axis=0, ddof=1) if len(per_fold) > 1 else 0.0,
            "min": per_fold.min(axis=0),
            "max": per_fold.max(axis=0),
        })

        return {
            "per_fold": per_fold,
            "summary": summary,
            "oof_pred": oof_pred,
            "oof_proba": oof_proba,
            "class_names": prep.class_names,
        }

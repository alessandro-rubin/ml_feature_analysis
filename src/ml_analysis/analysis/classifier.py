"""Multi-classifier evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from ml_analysis.analysis.base import AnalysisContext, prepare_xy

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


@dataclass
class ClassifierEvaluation:
    name: str = "classifier"
    requires: tuple[str, ...] = ()
    test_size: float = 0.2
    run_rf: bool = True
    run_lgb: bool = True
    run_xgb: bool = True
    rf_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
        )
    )
    lgb_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            n_jobs=-1, random_state=42, verbose=-1,
        )
    )
    xgb_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            tree_method="hist", random_state=42, verbosity=0, eval_metric="mlogloss",
        )
    )

    def _models(self) -> dict:
        m = {}
        if self.run_rf:
            m["Random Forest"] = RandomForestClassifier(**self.rf_params)
        if self.run_lgb and HAS_LGB:
            m["LightGBM"] = lgb.LGBMClassifier(**self.lgb_params)
        if self.run_xgb and HAS_XGB:
            m["XGBoost"] = xgb.XGBClassifier(**self.xgb_params)
        if not m:
            raise RuntimeError("No classifiers available — check installs / flags.")
        return m

    def run(self, ctx: AnalysisContext) -> dict[str, Any]:
        prep = prepare_xy(ctx)
        if len(prep.X) < 4:
            raise ValueError(f"Not enough samples for classifier eval: {len(prep.X)}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            prep.X, prep.y,
            test_size=self.test_size,
            stratify=prep.y if len(set(prep.y)) > 1 else None,
            random_state=ctx.cfg.random_state,
        )

        per_model = {}
        for name, model in self._models().items():
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            cm = confusion_matrix(y_te, preds)
            report = classification_report(
                y_te, preds, target_names=prep.class_names,
                zero_division=0, output_dict=True,
            )
            imp = (
                pd.Series(model.feature_importances_, index=prep.feature_cols, name=name)
                if hasattr(model, "feature_importances_")
                else None
            )
            per_model[name] = {
                "model": model,
                "preds": preds,
                "accuracy": accuracy_score(y_te, preds),
                "confusion_matrix": cm,
                "report": report,
                "importances": imp,
            }
        return {
            "models": per_model,
            "y_test": y_te,
            "class_names": prep.class_names,
        }

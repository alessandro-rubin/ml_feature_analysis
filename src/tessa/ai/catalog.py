"""What analyses exist, what they accept, and what they return.

Derived from ``tessa.run._ANALYSES`` by dataclass introspection rather than a
hand-maintained table, so the catalogue cannot drift from the code the way the
old hand-written 374-line tool-schema list did. Only the prose descriptions are
written by hand, and a test fails if an analysis ever lacks one.

The other job here is **parameter coercion**: JSON has no tuples, so a model
sending ``{"k_range": [2, 8]}`` or ``{"pairs": [["TP", "FP"]]}`` would otherwise
hand a list to a field annotated ``tuple``.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from typing import Any

from tessa.run import _ANALYSES

__all__ = [
    "ParamSpec",
    "analysis_names",
    "analysis_params",
    "analysis_entry",
    "analysis_catalogue",
    "compact_catalogue",
    "coerce_params",
    "validate_params",
    "UnknownAnalysis",
]

# Protocol fields — real dataclass fields on some analyses, but mutating them
# breaks the DAG runner, so they are never exposed as parameters.
_PROTOCOL_FIELDS = frozenset({"name", "requires", "needs_labels"})


class UnknownAnalysis(KeyError):
    """Raised for an analysis name that is not registered."""


_DESCRIPTIONS: dict[str, str] = {
    "importance": "Rank every feature by discriminative power, blending RF impurity, "
    "permutation importance, ANOVA F, Kruskal-Wallis H and mutual information into a "
    "rank-based composite.",
    "classifier": "Train/test split classifiers (RF, and LightGBM/XGBoost when installed) "
    "and report accuracy, per-class precision/recall/F1 and a confusion matrix.",
    "cv_classifier": "Cross-validated classifier metrics (MCC, balanced accuracy, Brier, ECE) "
    "with folds grouped by asset so scores describe unseen assets, not unseen events.",
    "distributions": "Per-class quantiles per feature plus Kruskal-Wallis, ANOVA, "
    "Anderson-Darling and Levene tests, with multiple-testing correction.",
    "pairwise": "For each pair of classes, rank features by Cliff's delta, KS statistic and "
    "AUC. Use this to find what separates one specific pair, not all classes at once.",
    "clustering": "KMeans/DBSCAN/HDBSCAN over the feature matrix with PCA/UMAP embeddings; "
    "reports the best k and how clusters line up with the labels.",
    "cluster_validation": "Is the clustering real? Hopkins statistic for cluster tendency plus "
    "permutation p-values for ARI and V-measure against the labels.",
    "importance_stability": "Bootstrap the importance ranking to get confidence intervals and "
    "Spearman agreement between methods — shows whether a top feature is stably top.",
    "separability": "The headline question: are the classes distinguishable at all? "
    "Permutation-tested cross-validated balanced accuracy against chance, with a verdict.",
    "anomaly": "Unsupervised ensemble (IsolationForest + LOF + robust Mahalanobis) with "
    "per-feature attribution. Works without labels.",
    "label_spreading": "Semi-supervised kNN label propagation from a few labelled rows to "
    "unlabelled ones.",
    "pu_learning": "Positive-unlabelled learning: bagged scoring of unlabelled rows given one "
    "known positive class. Requires positive_label.",
    "changepoint": "Two-sided CUSUM per (asset, channel) with a Monte-Carlo-calibrated "
    "threshold. Runs on raw per-sample data, not the period table.",
    "correlation_structure": "Cluster features by |Spearman|, flag near-duplicate channels and "
    "suggest a non-redundant keep set.",
    "lagged_relations": "Exploratory lead/lag association between channels relative to a "
    "reference channel. Association, not causation.",
    "mi_network": "Nonlinear dependence graph between features via mutual information.",
}

# Result keys worth naming up front, so the model knows what to ask get_result for.
_RESULT_KEYS: dict[str, list[str]] = {
    "importance": ["table"],
    "classifier": ["confusion_long"],
    "cv_classifier": ["per_fold", "summary"],
    "distributions": ["summary", "per_feature_class"],
    # `pairs` is keyed by class tuples, so AnalysisResult files it under
    # `objects` and ResultStore drops it. `pairs_long` is the usable frame.
    "pairwise": ["pairs_long"],
    "clustering": ["embedding", "k_values", "metrics"],
    "cluster_validation": ["summary"],
    "importance_stability": ["bootstrap_table", "method_agreement"],
    "separability": ["summary"],
    "anomaly": ["scores", "top_contributors", "feature_contributions"],
    "label_spreading": ["table"],
    "pu_learning": ["table", "ranked_unlabeled"],
    "changepoint": ["table", "per_channel"],
    "correlation_structure": ["correlation", "clusters", "duplicates"],
    "lagged_relations": ["table"],
    "mi_network": ["matrix", "edges"],
}

_CAVEATS: dict[str, str] = {
    "classifier": "run_lgb/run_xgb default to True but silently fall back to "
    "random-forest-only when lightgbm/xgboost are not installed.",
    "clustering": "HDBSCAN and UMAP need the 'clustering' extra; those parts are skipped "
    "when it is absent.",
    "pu_learning": "positive_label is required.",
    "changepoint": "Expects per-sample data — materialize with kind='per_sample', not the "
    "period-aggregate table.",
    "lagged_relations": "Expects per-sample data.",
}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    json_type: str
    default: Any
    annotation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.json_type,
            "default": self.default,
            "annotation": self.annotation,
        }


def _json_type(annotation: Any) -> str:
    text = str(annotation)
    if "bool" in text and "list" not in text:
        return "boolean"
    if "int" in text and "float" not in text and "list" not in text and "tuple" not in text:
        return "integer"
    if "float" in text and "list" not in text and "tuple" not in text:
        return "number"
    if "dict" in text:
        return "object"
    if "list" in text or "tuple" in text:
        return "array"
    if "str" in text:
        return "string"
    return "any"


def analysis_names() -> list[str]:
    return sorted(_ANALYSES)


def _cls(name: str):
    if name not in _ANALYSES:
        raise UnknownAnalysis(f"Unknown analysis {name!r}. Valid names: {analysis_names()}.")
    return _ANALYSES[name]


def analysis_params(name: str) -> list[ParamSpec]:
    """Settable constructor parameters for one analysis."""
    cls = _cls(name)
    hints = typing.get_type_hints(cls)
    out: list[ParamSpec] = []
    for f in dataclasses.fields(cls):
        if f.name in _PROTOCOL_FIELDS:
            continue
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            default = f.default_factory()  # type: ignore[misc]
        else:
            default = None
        ann = hints.get(f.name, Any)
        out.append(
            ParamSpec(
                name=f.name,
                json_type=_json_type(ann),
                default=list(default) if isinstance(default, tuple) else default,
                annotation=str(ann).replace("typing.", ""),
            )
        )
    return out


def analysis_entry(name: str) -> dict[str, Any]:
    """Catalogue entry for one analysis."""
    entry: dict[str, Any] = {
        "name": name,
        "needs_labels": getattr(_cls(name), "needs_labels", "full"),
        "description": _DESCRIPTIONS.get(name, ""),
        "parameters": [p.to_dict() for p in analysis_params(name)],
        "result_frames": _RESULT_KEYS.get(name, []),
    }
    if name in _CAVEATS:
        entry["caveat"] = _CAVEATS[name]
    return entry


def analysis_catalogue() -> list[dict[str, Any]]:
    """Full catalogue: one entry per registered analysis."""
    return [analysis_entry(name) for name in analysis_names()]


def compact_catalogue() -> str:
    """A deterministic one-line-per-analysis digest for the system prompt.

    Deterministic matters: this text sits in the cached prompt prefix, so any
    run-to-run variation would silently destroy the prompt cache.
    """
    lines = []
    for name in analysis_names():
        cls = _ANALYSES[name]
        labels = getattr(cls, "needs_labels", "full")
        params = ", ".join(p.name for p in analysis_params(name)) or "none"
        lines.append(
            f"- {name} (labels: {labels}) — {_DESCRIPTIONS.get(name, '')} Params: {params}"
        )
    return "\n".join(lines)


def coerce_params(name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Validate parameter names and convert JSON lists into the tuples fields want.

    Raises
    ------
    UnknownAnalysis
        If ``name`` is not registered.
    ValueError
        If a parameter name is not a field of that analysis — with the valid
        names listed, so the caller can correct itself.
    """
    cls = _cls(name)
    if not params:
        return {}
    hints = typing.get_type_hints(cls)
    valid = {p.name for p in analysis_params(name)}
    unknown = sorted(set(params) - valid)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {unknown} for analysis {name!r}. "
            f"Valid parameters: {sorted(valid)}."
        )

    out: dict[str, Any] = {}
    for key, value in params.items():
        ann = str(hints.get(key, ""))
        if isinstance(value, list):
            if ann.startswith("tuple") or ann.startswith("typing.Tuple"):
                value = tuple(value)
            elif "tuple" in ann:
                # e.g. list[tuple[str, str]] | None  -> list of tuples
                value = [tuple(v) if isinstance(v, list) else v for v in value]
        out[key] = value
    return out


def validate_params(name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Alias kept for readability at call sites that only want validation."""
    return coerce_params(name, params)

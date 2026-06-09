"""Analysis protocol, context, and DAG runner."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder

from ml_analysis.config import Config

LabelFilter = dict[str, list] | Callable[[pl.DataFrame], pl.DataFrame] | None


def seeded(params: dict, cfg: Config) -> dict:
    """Return ``params`` with ``random_state`` filled from ``cfg`` if unset.

    Estimator param dicts default to ``random_state=None`` so that
    ``Config.random_state`` is the single source of reproducibility; an
    explicit value in ``params`` always wins.
    """
    out = dict(params)
    if out.get("random_state") is None:
        out["random_state"] = cfg.random_state
    return out


@dataclass(frozen=True)
class NullPolicy:
    """How :func:`prepare_xy` handles nulls in the feature matrix.

    kind:
      - ``"drop_rows"`` — drop any row with a null feature (legacy
        behaviour, now reported and warned about).
      - ``"drop_features"`` — drop features whose null fraction exceeds
        ``max_feature_null_frac``, then drop remaining null rows.
      - ``"impute_median"`` — drop all-null features, fill remaining
        nulls with the per-feature median.

    Rows with a null *target* are always dropped (and counted).
    """

    kind: Literal["drop_rows", "drop_features", "impute_median"] = "drop_rows"
    max_feature_null_frac: float = 0.5
    warn_row_drop_frac: float = 0.2


@dataclass(frozen=True)
class PreparationReport:
    """What :func:`prepare_xy` did to the data — never silent."""

    policy: NullPolicy
    n_rows_in: int
    n_rows_out: int
    n_rows_dropped_null_target: int
    n_rows_dropped_null_features: int
    dropped_features: dict[str, float]  # name -> null fraction
    imputed_features: dict[str, float]  # name -> null fraction imputed
    feature_null_fracs: dict[str, float]  # offenders only (frac > 0)

    def summary(self) -> str:
        lines = [
            f"prepare_xy [{self.policy.kind}]: "
            f"{self.n_rows_in} rows in -> {self.n_rows_out} out "
            f"({self.n_rows_dropped_null_target} null-target, "
            f"{self.n_rows_dropped_null_features} null-feature rows dropped)"
        ]
        if self.dropped_features:
            lines.append(
                "dropped features: "
                + ", ".join(f"{k} ({v:.0%} null)" for k, v in self.dropped_features.items())
            )
        if self.imputed_features:
            lines.append(
                "imputed features: "
                + ", ".join(f"{k} ({v:.0%} null)" for k, v in self.imputed_features.items())
            )
        return "\n".join(lines)


@dataclass
class AnalysisContext:
    """Shared state for a chain of analyses.

    `df` is typically the period-aggregate (one row per event).
    `target_col` is the label column being explained / predicted; ``None``
    puts the context in unsupervised mode (label-requiring analyses are
    skipped by :func:`run_analyses`).
    `label_filter` restricts rows (e.g. {"class": ["TP", "FP"]}).
    `stratify_by` is consumed by the Stratified wrapper, not by base analyses.
    `results` accumulates outputs keyed by analysis name.
    `null_policy` governs null handling at the sklearn boundary.
    """

    df: pl.DataFrame
    cfg: Config
    target_col: str | None = None
    label_filter: LabelFilter = None
    stratify_by: str | None = None
    output_dir: str | None = None
    results: dict[str, Any] = field(default_factory=dict)
    null_policy: NullPolicy = field(default_factory=NullPolicy)
    _xy_cache: dict[tuple, "PreparedXY"] = field(default_factory=dict, repr=False)

    def filtered(self) -> pl.DataFrame:
        if self.label_filter is None:
            return self.df
        if callable(self.label_filter):
            return self.label_filter(self.df)
        out = self.df
        for col, allowed in self.label_filter.items():
            out = out.filter(pl.col(col).is_in(allowed))
        return out

    def invalidate_cache(self) -> None:
        """Clear cached prepared matrices (call after mutating `df`)."""
        self._xy_cache.clear()


@dataclass
class PreparedXY:
    X: pd.DataFrame
    y: np.ndarray | None
    feature_cols: list[str]
    class_names: list[str]
    encoder: LabelEncoder | None
    report: PreparationReport | None = None
    ids: pd.DataFrame | None = None  # event_id / asset_id rows aligned to X


def prepare_xy(
    ctx: AnalysisContext,
    drop_cols: tuple[str, ...] = (),
    policy: NullPolicy | None = None,
    ignore_target: bool = False,
) -> PreparedXY:
    """Filter, handle nulls per policy, split into numeric X and encoded y.

    With ``ctx.target_col is None`` (or ``ignore_target=True``) no label is
    required: ``y``/``encoder`` come back ``None``, no rows are dropped for
    null targets, and the target column (if present) is still excluded
    from the features.

    Results are cached on the context (keyed by ``drop_cols``, policy and
    target handling) so chained analyses share one preparation instead of
    redoing the polars -> pandas conversion and null handling each time.
    """
    pol = policy or ctx.null_policy
    use_target = ctx.target_col is not None and not ignore_target
    cache_key = (tuple(sorted(drop_cols)), pol, use_target)
    cached = ctx._xy_cache.get(cache_key)
    if cached is not None:
        return cached

    df = ctx.filtered().to_pandas()
    n_in = len(df)

    target = ctx.target_col
    drop = set(drop_cols) | {"event_id", "asset_id", ctx.cfg.timestamp_col}
    if target is not None:
        drop.add(target)

    id_cols = [c for c in ("event_id", "asset_id") if c in df.columns]
    feature_cols = [
        c for c in df.select_dtypes(include="number").columns if c not in drop
    ]
    X = df[feature_cols].copy()
    ids = df[id_cols].copy() if id_cols else None

    n_null_target = 0
    if use_target:
        y_raw = df[target]
        target_mask = y_raw.notna()
        n_null_target = int((~target_mask).sum())
        X = X[target_mask]
        y_raw = y_raw[target_mask]
        if ids is not None:
            ids = ids[target_mask]
    else:
        y_raw = None

    null_fracs = X.isna().mean()
    offenders = {c: float(f) for c, f in null_fracs.items() if f > 0}
    dropped_features: dict[str, float] = {}
    imputed_features: dict[str, float] = {}

    if pol.kind == "drop_features":
        dropped_features = {
            c: f for c, f in offenders.items() if f > pol.max_feature_null_frac
        }
        X = X.drop(columns=list(dropped_features))
    elif pol.kind == "impute_median":
        dropped_features = {c: f for c, f in offenders.items() if f >= 1.0}
        X = X.drop(columns=list(dropped_features))
        to_impute = {c: f for c, f in offenders.items() if c not in dropped_features}
        if to_impute:
            X = X.fillna(X.median(numeric_only=True))
            imputed_features = to_impute

    row_mask = X.notna().all(axis=1)
    n_null_rows = int((~row_mask).sum())
    X = X[row_mask].reset_index(drop=True)
    if y_raw is not None:
        y_raw = y_raw[row_mask].reset_index(drop=True)
    if ids is not None:
        ids = ids[row_mask].reset_index(drop=True)

    report = PreparationReport(
        policy=pol,
        n_rows_in=n_in,
        n_rows_out=len(X),
        n_rows_dropped_null_target=n_null_target,
        n_rows_dropped_null_features=n_null_rows,
        dropped_features=dropped_features,
        imputed_features=imputed_features,
        feature_null_fracs=offenders,
    )
    if n_in > 0 and (n_in - len(X)) / n_in > pol.warn_row_drop_frac:
        warnings.warn(
            f"prepare_xy dropped {n_in - len(X)}/{n_in} rows "
            f"({(n_in - len(X)) / n_in:.0%}). {report.summary()}",
            stacklevel=2,
        )

    if y_raw is not None:
        enc = LabelEncoder()
        y = enc.fit_transform(y_raw)
        class_names = [str(c) for c in enc.classes_]
    else:
        enc = None
        y = None
        class_names = []
    prep = PreparedXY(
        X=X,
        y=y,
        feature_cols=list(X.columns),
        class_names=class_names,
        encoder=enc,
        report=report,
        ids=ids,
    )
    ctx._xy_cache[cache_key] = prep
    return prep


class Analysis(Protocol):
    name: str
    requires: tuple[str, ...]
    # "full"  — needs a class label on every analysed row (default),
    # "partial" — works with sparse labels (semi-supervised),
    # "none" — fully unsupervised.
    needs_labels: str

    def run(self, ctx: AnalysisContext) -> Any: ...


def _labels_available(a: Analysis, ctx: AnalysisContext) -> bool:
    needs = getattr(a, "needs_labels", "full")
    if needs == "none":
        return True
    return ctx.target_col is not None


def run_analyses(
    analyses: list[Analysis], ctx: AnalysisContext
) -> dict[str, Any]:
    """Topo-sort by `requires` and execute. Stores each result on ctx.results.

    Analyses whose label requirement isn't met by the context (e.g. a
    supervised analysis on a context with ``target_col=None``) are skipped
    with a warning, along with anything that depends on them.
    """
    by_name = {a.name: a for a in analyses}
    ordered: list[Analysis] = []
    seen: set[str] = set()
    in_progress: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        if name in in_progress:
            raise ValueError(f"Cyclic analysis dependency at: {name}")
        if name not in by_name:
            raise ValueError(f"Missing required analysis: {name}")
        in_progress.add(name)
        for dep in by_name[name].requires:
            visit(dep)
        in_progress.discard(name)
        seen.add(name)
        ordered.append(by_name[name])

    for a in analyses:
        visit(a.name)

    skipped: set[str] = set()
    for a in ordered:
        dep_skipped = [d for d in a.requires if d in skipped]
        if dep_skipped:
            skipped.add(a.name)
            warnings.warn(
                f"Skipping analysis '{a.name}': depends on skipped {dep_skipped}.",
                stacklevel=2,
            )
            continue
        if not _labels_available(a, ctx):
            skipped.add(a.name)
            warnings.warn(
                f"Skipping analysis '{a.name}': it requires labels "
                f"(needs_labels={getattr(a, 'needs_labels', 'full')!r}) but the "
                "context has target_col=None.",
                stacklevel=2,
            )
            continue
        ctx.results[a.name] = a.run(ctx)
    return ctx.results

"""The tool surface exposed to a model (or to a human, or to MCP).

These are plain Python functions. They import no SDK — not ``anthropic``, not
``openai``, not ``mcp`` — so the same implementation backs every driver, and the
whole surface is testable with no network and no credentials.

Conventions, all of which exist because the primary driver is a mid-sized
open-weight model rather than a frontier one:

* Every function returns a JSON-serializable ``dict`` and never raises. Failures
  come back as ``{"error": {"type", "message", "hint", ...}}`` so the model can
  correct itself instead of seeing a traceback.
* Call order is enforced in code, and the error names the next call to make.
* Payloads are truncated and uniformly shaped by :mod:`tessa.ai.render`;
  ``run_analysis`` returns a headline and ``get_result`` pages into the detail.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import Any, Callable

import polars as pl

from tessa.ai import render
from tessa.ai.catalog import (
    UnknownAnalysis,
    analysis_entry,
    analysis_names,
    coerce_params,
    compact_catalogue,
)
from tessa.ai.expressions import ExpressionError, allowlist_reference
from tessa.ai.expressions import compile_aggregator_expression, compile_feature_expression
from tessa.ai.session import AgentSession, FeatureLedgerEntry
from tessa.ai.session import create_session as _new_session
from tessa.ai.session import get_session
from tessa.dataset.builder import build as build_events_dict
from tessa.features.aggregates import AggSpec
from tessa.features.builtins import (
    make_constant_counter,
    make_first_difference,
    make_rolling_mean,
    make_rolling_std,
    make_zscore,
)
from tessa.features.registry import FeatureSpec
from tessa.features.windows import WindowSpec, materialize as materialize_frames
from tessa.results.store import ResultStore

MAX_COLUMNS = 400
MAX_PREVIEW_ROWS = 5000

_BUILTIN_KINDS = {
    "roll_mean": make_rolling_mean,
    "roll_std": make_rolling_std,
    "diff1": make_first_difference,
    "zscore": make_zscore,
    "const_count": make_constant_counter,
}
_WINDOWED_KINDS = {"roll_mean", "roll_std", "zscore"}


# ── error plumbing ───────────────────────────────────────────────────────────


class ToolError(Exception):
    """A user-correctable failure, rendered into the error envelope."""

    def __init__(self, kind: str, message: str, hint: str = "", **extra: Any):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.hint = hint
        self.extra = extra

    def payload(self) -> dict[str, Any]:
        err: dict[str, Any] = {"type": self.kind, "message": self.message}
        if self.hint:
            err["hint"] = self.hint
        err.update(self.extra)
        return {"error": err}


def tool(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Wrap a tool so it always returns a payload, never an exception."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return fn(*args, **kwargs)
        except ToolError as exc:
            return exc.payload()
        except UnknownAnalysis as exc:
            return ToolError(
                "UnknownAnalysis", str(exc).strip("'"), valid=analysis_names()
            ).payload()
        except ExpressionError as exc:
            return ToolError(
                "ExpressionError",
                str(exc),
                "Call list_capabilities to see the allowed functions and methods.",
            ).payload()
        except KeyError as exc:
            return ToolError("NotFound", str(exc).strip("'")).payload()
        except ValueError as exc:
            return ToolError("InvalidArgument", str(exc)).payload()
        except Exception as exc:  # last resort — never leak a traceback
            return ToolError(type(exc).__name__, str(exc)).payload()

    return wrapper


def _session(session_id: str) -> AgentSession:
    return get_session(session_id)


def _need_labels(s: AgentSession) -> pl.DataFrame:
    if s.labels is None:
        raise ToolError("OrderError", "No labels loaded.", "Call create_session with labels_path.")
    return s.labels


def _need_events(s: AgentSession) -> dict[str, pl.LazyFrame]:
    if not s.events:
        raise ToolError(
            "OrderError",
            "The event dataset has not been built.",
            f"Call build_events('{s.session_id}') first.",
        )
    return s.events


def _need_table(s: AgentSession, table: str | None) -> str:
    key = table or s.active_table
    if key is None:
        raise ToolError(
            "OrderError",
            "No analysis table has been materialized.",
            f"Call materialize('{s.session_id}') first.",
        )
    if key not in s.tables:
        raise ToolError(
            "NotFound", f"Unknown table {key!r}.", f"Available tables: {sorted(s.tables)}."
        )
    return key


def _load_label_table(path: Path, sheet: str | int, cfg) -> pl.DataFrame:
    """Read labels from Excel, or from parquet/CSV when that is what is on disk."""
    from tessa.labels.base import validate
    from tessa.labels.excel import ExcelLabelSource

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return ExcelLabelSource(path=path, sheet=sheet).load(cfg)
    if suffix == ".parquet":
        return validate(pl.read_parquet(path), cfg)
    if suffix == ".csv":
        return validate(pl.read_csv(path, try_parse_dates=True), cfg)
    raise ToolError(
        "InvalidArgument",
        f"Unsupported label file type {suffix!r}.",
        "Use .xlsx, .parquet or .csv.",
    )


# ── 1. session / data ────────────────────────────────────────────────────────


@tool
def create_session(
    data_root: str,
    labels_path: str | None = None,
    labels_sheet: str | int = 0,
    class_col: str = "class",
    asset_col: str = "asset_id",
    timestamp_col: str = "timestamp",
    output_dir: str = "outputs",
    filename_pattern: str | None = None,
    random_state: int = 42,
    holdout_assets: list[str] | None = None,
) -> dict:
    """Start an analysis session and load the event labels. Call this first.

    Args:
        data_root: Directory holding per-asset parquet files.
        labels_path: Label table (.xlsx, .parquet or .csv) with asset/start/end/class.
        labels_sheet: Worksheet name, or 0-based index, for Excel labels.
        class_col: Column holding the class label.
        asset_col: Column holding the asset id.
        timestamp_col: Timestamp column in the raw parquet files.
        output_dir: Directory for saved runs, reports and figures.
        filename_pattern: Override the raw-file naming regex if it is non-standard.
        random_state: Seed threaded through every estimator.
        holdout_assets: Assets hidden from the whole exploration, so
            confirm_on_holdout has genuinely unseen data at the end.

    Returns:
        The session_id required by every other tool, plus the classes and assets found.
    """
    root = Path(data_root)
    if not root.exists():
        raise ToolError("NotFound", f"data_root {str(root)!r} does not exist.")
    s = _new_session(
        root,
        output_dir=output_dir,
        class_col=class_col,
        asset_col=asset_col,
        timestamp_col=timestamp_col,
        filename_pattern=filename_pattern,
        random_state=random_state,
        holdout_assets=tuple(holdout_assets or ()),
    )
    out: dict[str, Any] = {"session_id": s.session_id, "data_root": str(root)}

    if labels_path:
        path = Path(labels_path)
        if not path.exists():
            raise ToolError("NotFound", f"labels_path {str(path)!r} does not exist.")
        s.labels = _load_label_table(path, labels_sheet, s.cfg)
        out.update(_label_summary(s))
    else:
        out["note"] = "No labels loaded; only unsupervised analyses will be available."
    out["next"] = "describe_data, then build_events"
    return out


def _label_summary(s: AgentSession) -> dict[str, Any]:
    labels = s.labels
    assert labels is not None
    cc, ac = s.cfg.class_col, s.cfg.asset_col
    summary: dict[str, Any] = {"n_events": labels.height, "columns": labels.columns}
    if cc in labels.columns:
        counts = labels.group_by(cc).len().sort(cc)
        summary["class_counts"] = {str(r[cc]): int(r["len"]) for r in counts.iter_rows(named=True)}
    if ac in labels.columns:
        summary["assets"] = sorted(str(a) for a in labels[ac].unique().to_list())
    if s.holdout_assets:
        summary["holdout_assets"] = list(s.holdout_assets)
    return summary


@tool
def describe_data(session_id: str, asset_id: str | None = None) -> dict:
    """Describe the raw data: assets, channel names and dtypes, and class balance.

    Call this before inventing features — expressions reference channels by name,
    so you need to know what the channels are actually called.

    Args:
        session_id: From create_session.
        asset_id: Inspect this asset's channels; defaults to the first asset.

    Returns:
        Assets, channels with dtypes, class counts and per-asset event counts.
    """
    from tessa.dataset import Dataset

    s = _session(session_id)
    ds = Dataset(s.cfg)
    out: dict[str, Any] = {"session_id": session_id}
    try:
        assets = ds.assets
    except Exception as exc:
        raise ToolError("NotFound", f"Could not list assets under {s.cfg.data_root}: {exc}")
    out["assets"] = assets

    target = asset_id or (assets[0] if assets else None)
    if target:
        lf = ds.lazy(target)
        schema = lf.collect_schema()
        out["inspected_asset"] = target
        out["channels"] = [
            {"name": n, "dtype": str(schema[n]), "numeric": bool(schema[n].is_numeric())}
            for n in schema.names()
        ]
        out["numeric_channels"] = [
            n for n in schema.names() if schema[n].is_numeric() and n != s.cfg.timestamp_col
        ]
    if s.labels is not None:
        out.update(_label_summary(s))
        ac, cc = s.cfg.asset_col, s.cfg.class_col
        if ac in s.labels.columns and cc in s.labels.columns:
            per = s.labels.group_by([ac, cc]).len().sort([ac, cc])
            out["events_per_asset_class"] = [
                {"asset": str(r[ac]), "class": str(r[cc]), "n": int(r["len"])}
                for r in per.iter_rows(named=True)
            ]
    return out


@tool
def build_events(session_id: str, columns: list[str] | None = None) -> dict:
    """Build the lazy per-event dataset from the labels. Call after create_session.

    Args:
        session_id: From create_session.
        columns: Restrict loading to these raw channels (plus the timestamp).

    Returns:
        Event counts overall and per asset.
    """
    s = _session(session_id)
    labels = _need_labels(s)
    if s.holdout_assets:
        ac = s.cfg.asset_col
        labels = labels.filter(~pl.col(ac).is_in(list(s.holdout_assets)))
    s.events = build_events_dict(labels, s.cfg, columns=columns)
    per_asset: dict[str, int] = {}
    for eid in s.events:
        asset = eid.split("_")[0]
        per_asset[asset] = per_asset.get(asset, 0) + 1
    return {
        "session_id": session_id,
        "n_events": len(s.events),
        "events_per_asset": per_asset,
        "holdout_excluded": list(s.holdout_assets),
        "next": "seed_builtin_features and/or create_feature, then materialize",
    }


# ── 2. features ──────────────────────────────────────────────────────────────


@tool
def list_capabilities(session_id: str) -> dict:
    """List what you can currently use: features, aggregators, analyses, and the DSL.

    Args:
        session_id: From create_session.

    Returns:
        Session features and aggregators, the analysis catalogue in compact form,
        and the allowlist of Polars functions and methods usable in expressions.
    """
    s = _session(session_id)
    return {
        "session_id": session_id,
        "features": s.features.names(),
        "aggregators": s.aggregators.names(),
        "analyses": analysis_names(),
        "analysis_catalogue": compact_catalogue(),
        "expression_language": allowlist_reference(),
        "tables": sorted(s.tables),
        "active_table": s.active_table,
    }


@tool
def describe_analysis(name: str) -> dict:
    """Full parameter list and result frames for one analysis.

    Args:
        name: Analysis name, e.g. "separability".

    Returns:
        Parameters with types and defaults, the result frames it produces, and any caveat.
    """
    return analysis_entry(name)


@tool
def seed_builtin_features(
    session_id: str,
    sources: list[str] | None = None,
    kinds: list[str] | None = None,
    windows: list[int] | None = None,
) -> dict:
    """Register stock features (rolling mean/std, first difference, z-score) per channel.

    A cheap strong baseline, so you only hand-write expressions where the stock
    set is not enough.

    Args:
        session_id: From create_session.
        sources: Channels to build features for. Defaults to the numeric channels.
        kinds: Any of roll_mean, roll_std, diff1, zscore, const_count. Defaults to
            roll_mean, roll_std, diff1.
        windows: Window sizes for the windowed kinds. Defaults to [10, 50].

    Returns:
        The feature names registered, and any that were skipped as duplicates.
    """
    s = _session(session_id)
    events = _need_events(s)
    kinds = kinds or ["roll_mean", "roll_std", "diff1"]
    windows = windows or [10, 50]
    unknown = [k for k in kinds if k not in _BUILTIN_KINDS]
    if unknown:
        raise ToolError(
            "InvalidArgument",
            f"Unknown feature kinds {unknown}.",
            f"Valid kinds: {sorted(_BUILTIN_KINDS)}.",
        )

    if sources is None:
        schema = next(iter(events.values())).collect_schema()
        label_like = {"event_id", s.cfg.asset_col, s.cfg.class_col, s.cfg.timestamp_col}
        label_like.update(getattr(events, "label_cols", ()))
        sources = [n for n in schema.names() if schema[n].is_numeric() and n not in label_like]

    added, skipped = [], []
    for source in sources:
        for kind in kinds:
            fn = _BUILTIN_KINDS[kind]
            targets = windows if kind in _WINDOWED_KINDS else [None]
            for w in targets:
                try:
                    if w is None:
                        fn(source, registry=s.features)
                    else:
                        fn(source, w, registry=s.features)
                except ValueError:
                    skipped.append(f"{source}:{kind}{'' if w is None else f':{w}'}")
                    continue
                name = s.features.names()[-1]
                added.append(name)
                s.ledger.append(
                    FeatureLedgerEntry(
                        name=name,
                        kind="builtin",
                        expression=f"{kind}({source}" + (f", {w})" if w else ")"),
                        rationale="stock baseline feature",
                        deps=(source,),
                    )
                )
    return {
        "session_id": session_id,
        "registered": added,
        "skipped_duplicates": skipped,
        "n_features_total": len(s.features.names()),
    }


def _preview(s: AgentSession, factory, name: str, event_id: str | None, n_rows: int) -> dict:
    events = _need_events(s)
    key = event_id or next(iter(events))
    if key not in events:
        raise ToolError(
            "NotFound",
            f"Unknown event_id {key!r}.",
            f"There are {len(events)} events; omit event_id to use the first.",
        )
    lf = events[key].head(min(n_rows, MAX_PREVIEW_ROWS))
    try:
        col = lf.select(factory().alias(name)).collect()[name]
    except Exception as exc:
        raise ToolError(
            "PolarsError",
            f"Expression failed on real data: {type(exc).__name__}: {exc}",
            "Check the channel names with describe_data.",
        )
    n = col.len()
    stats: dict[str, Any] = {
        "dtype": str(col.dtype),
        "n_rows": int(n),
        "null_fraction": round(float(col.null_count()) / n, 4) if n else 1.0,
        "event_id": key,
    }
    if col.dtype.is_numeric() and n:
        clean = col.drop_nulls().drop_nans() if col.dtype.is_float() else col.drop_nulls()
        if clean.len():
            stats.update(
                n_unique=int(clean.n_unique()),
                min=render.jsonable(clean.min()),
                mean=render.jsonable(clean.mean()),
                max=render.jsonable(clean.max()),
                std=render.jsonable(clean.std()),
            )
    stats["head"] = [render.jsonable(v) for v in col.head(10).to_list()]
    return stats


@tool
def preview_feature(
    session_id: str,
    expression: str,
    name: str = "__preview__",
    event_id: str | None = None,
    n_rows: int = 2000,
) -> dict:
    """Compile an expression and try it on one event, WITHOUT registering it.

    Use this to iterate cheaply. Reject a candidate whose null_fraction is high
    or whose n_unique is tiny — it will not discriminate anything.

    Args:
        session_id: From create_session.
        expression: A Polars expression, e.g. 'pl.col("temperature").rolling_std(50)'.
        name: Name to give the previewed column.
        event_id: Which event to try it on; defaults to the first.
        n_rows: How many rows of that event to use.

    Returns:
        dtype, null fraction, n_unique, summary statistics and the first values.
    """
    s = _session(session_id)
    compiled = compile_feature_expression(expression)
    s.n_previewed += 1
    stats = _preview(s, compiled.factory, name, event_id, n_rows)
    return {
        "session_id": session_id,
        "ok": True,
        "expression": expression,
        "deps": list(compiled.deps),
        **stats,
    }


@tool
def create_feature(
    session_id: str,
    name: str,
    expression: str,
    rationale: str,
    replace: bool = False,
) -> dict:
    """Register a new per-sample feature from a Polars expression.

    The expression is validated and tried on one event before anything is
    registered, so a broken expression never enters the registry.

    Args:
        session_id: From create_session.
        name: Output column name, e.g. "temperature__volatility".
        expression: A Polars expression, e.g. 'pl.col("temperature").rolling_std(50)'.
        rationale: Why this feature should separate the classes. Recorded in the ledger.
        replace: Overwrite an existing feature of the same name.

    Returns:
        The registered name, derived dependencies and the preview statistics.
    """
    s = _session(session_id)
    _need_events(s)
    if not rationale or not rationale.strip():
        raise ToolError(
            "InvalidArgument",
            "A rationale is required.",
            "Say what you expect this feature to capture.",
        )
    reserved = {"event_id", s.cfg.asset_col, s.cfg.class_col, s.cfg.timestamp_col}
    if name in reserved:
        raise ToolError(
            "InvalidArgument",
            f"{name!r} is a reserved column name.",
            f"Reserved: {sorted(reserved)}.",
        )
    if name in s.features:
        if not replace:
            raise ToolError(
                "DuplicateName",
                f"Feature {name!r} already exists.",
                "Pass replace=true, or pick another name.",
                existing=s.features.get(name).deps,
            )
        s.features.unregister(name)

    compiled = compile_feature_expression(expression)
    stats = _preview(s, compiled.factory, name, None, 2000)
    s.features.register(FeatureSpec(name=name, deps=compiled.deps, factory=compiled.factory))

    s.ledger = [e for e in s.ledger if e.name != name]
    s.ledger.append(
        FeatureLedgerEntry(
            name=name,
            kind="feature",
            expression=expression,
            rationale=rationale.strip(),
            deps=compiled.deps,
            preview=stats,
        )
    )
    return {
        "session_id": session_id,
        "registered": name,
        "deps": list(compiled.deps),
        "preview": stats,
        "n_features_total": len(s.features.names()),
        "next": "materialize to include it in an analysis table",
    }


@tool
def create_aggregator(
    session_id: str, name: str, expression: str, rationale: str, replace: bool = False
) -> dict:
    """Register a new aggregator, which collapses a column to one value per event.

    Refer to the column being aggregated as `c`, e.g.
    'pl.col(c).quantile(0.9) - pl.col(c).quantile(0.1)'.

    Args:
        session_id: From create_session.
        name: Suffix for the output columns, which become "<source>__<name>".
        expression: A Polars expression over `c`.
        rationale: Why this summary statistic should matter. Recorded in the ledger.
        replace: Overwrite an existing aggregator of the same name.

    Returns:
        The registered name and the full aggregator list.
    """
    s = _session(session_id)
    if name in s.aggregators:
        if not replace:
            raise ToolError(
                "DuplicateName",
                f"Aggregator {name!r} already exists.",
                "Pass replace=true, or pick another name.",
            )
        s.aggregators.unregister(name)
    factory = compile_aggregator_expression(expression)
    s.aggregators.register(AggSpec(name=name, factory=factory))
    s.ledger.append(
        FeatureLedgerEntry(
            name=name,
            kind="aggregator",
            expression=expression,
            rationale=(rationale or "").strip(),
        )
    )
    return {"session_id": session_id, "registered": name, "aggregators": s.aggregators.names()}


@tool
def drop_feature(session_id: str, name: str, kind: str = "feature") -> dict:
    """Remove a feature or aggregator that did not earn its place.

    Args:
        session_id: From create_session.
        name: The registered name to remove.
        kind: Either "feature" or "aggregator".

    Returns:
        What remains registered.
    """
    s = _session(session_id)
    registry = s.features if kind == "feature" else s.aggregators
    if kind not in {"feature", "aggregator"}:
        raise ToolError("InvalidArgument", f"kind must be 'feature' or 'aggregator', got {kind!r}.")
    if name not in registry:
        raise ToolError(
            "NotFound", f"No {kind} named {name!r}.", f"Registered: {registry.names()}."
        )
    registry.unregister(name)
    entry = s.ledger_entry(name)
    if entry is not None:
        entry.verdict = "dropped"
    return {"session_id": session_id, "dropped": name, "remaining": registry.names()}


# ── 3. materialization ───────────────────────────────────────────────────────


@tool
def materialize(
    session_id: str,
    kind: str = "event",
    sources: list[str] | None = None,
    aggregators: list[str] | None = None,
    feature_names: list[str] | str | None = None,
    every: str | None = None,
    period: str | None = None,
    table_name: str = "baseline",
    max_columns: int = MAX_COLUMNS,
) -> dict:
    """Build the analysis table: one row per event (or per window).

    By default NO derived features are included — pass feature_names explicitly,
    or the string "__all__" for every registered feature. This is deliberate:
    features multiply by aggregators, so "everything" grows very fast.

    Args:
        session_id: From create_session.
        kind: "event" (one row per event), "tumbling" or "sliding".
        sources: Channels to aggregate. Defaults to all numeric columns.
        aggregators: e.g. ["mean", "std", "min", "max"]. Defaults to those four.
        feature_names: Derived features to include, or "__all__" for all of them.
        every: Window stride for tumbling/sliding windows, e.g. "1h".
        period: Window length for sliding windows, e.g. "6h".
        table_name: Name to store this table under, e.g. "baseline" or "augmented_1".
        max_columns: Refuse to build a table wider than this.

    Returns:
        Table shape, column count and class counts.
    """
    s = _session(session_id)
    events = _need_events(s)

    if feature_names in ("__all__", ["__all__"]):
        resolved_features: list[str] | None = None
        feature_count = len(s.features.names())
    elif feature_names is None:
        resolved_features = []
        feature_count = 0
    else:
        unknown = [f for f in feature_names if f not in s.features]
        if unknown:
            raise ToolError(
                "NotFound",
                f"Unknown feature(s) {unknown}.",
                f"Registered features: {s.features.names()}.",
            )
        resolved_features = list(feature_names)
        feature_count = len(resolved_features)

    aggs = aggregators or ["mean", "std", "min", "max"]
    unknown_aggs = [a for a in aggs if a not in s.aggregators]
    if unknown_aggs:
        raise ToolError(
            "NotFound",
            f"Unknown aggregator(s) {unknown_aggs}.",
            f"Registered aggregators: {s.aggregators.names()}.",
        )

    schema = next(iter(events.values())).collect_schema()
    label_like = {"event_id", s.cfg.asset_col, s.cfg.class_col}
    label_like.update(getattr(events, "label_cols", ()))
    n_sources = (
        len(sources)
        if sources
        else sum(
            1
            for n in schema.names()
            if schema[n].is_numeric() and n != s.cfg.timestamp_col and n not in label_like
        )
    )
    projected = (n_sources + feature_count) * len(aggs)
    if projected > max_columns:
        raise ToolError(
            "BudgetError",
            f"That would build about {projected} feature columns "
            f"(({n_sources} sources + {feature_count} features) x {len(aggs)} aggregators), "
            f"over the limit of {max_columns}.",
            "Narrow `sources`, `feature_names`, or `aggregators` — or raise max_columns "
            "deliberately.",
            projected_columns=projected,
        )

    if kind == "event":
        spec = WindowSpec.event()
    elif kind == "tumbling":
        if not every:
            raise ToolError("InvalidArgument", "kind='tumbling' requires `every`, e.g. '1h'.")
        spec = WindowSpec.tumbling(every)
    elif kind == "sliding":
        if not (every and period):
            raise ToolError("InvalidArgument", "kind='sliding' requires `every` and `period`.")
        spec = WindowSpec.sliding(every, period)
    else:
        raise ToolError(
            "InvalidArgument", f"Unknown kind {kind!r}.", "Use 'event', 'tumbling' or 'sliding'."
        )

    df = materialize_frames(
        events,
        spec,
        s.cfg,
        sources=sources,
        aggregators=aggs,
        feature_names=resolved_features,
        feature_registry=s.features,
        aggregator_registry=s.aggregators,
    )
    s.tables[table_name] = df
    s.active_table = table_name
    s.invalidate_runs(table_name)
    for entry in s.ledger:
        if entry.name in (
            resolved_features or (s.features.names() if resolved_features is None else [])
        ):
            if table_name not in entry.tables:
                entry.tables.append(table_name)

    id_like = {"event_id", s.cfg.asset_col, s.cfg.class_col, "start", "end"}
    out: dict[str, Any] = {
        "session_id": session_id,
        "table": table_name,
        "shape": list(df.shape),
        "n_feature_columns": sum(1 for c in df.columns if c not in id_like),
        "columns_sample": df.columns[:40],
        "n_columns": len(df.columns),
    }
    if s.cfg.class_col in df.columns:
        counts = df.group_by(s.cfg.class_col).len().sort(s.cfg.class_col)
        out["class_counts"] = {
            str(r[s.cfg.class_col]): int(r["len"]) for r in counts.iter_rows(named=True)
        }
    return out


@tool
def set_active_table(session_id: str, table_name: str) -> dict:
    """Choose which materialized table subsequent analyses run on.

    Args:
        session_id: From create_session.
        table_name: A table created by materialize.

    Returns:
        The active table and its shape.
    """
    s = _session(session_id)
    key = _need_table(s, table_name)
    s.active_table = key
    return {"session_id": session_id, "active_table": key, "shape": list(s.tables[key].shape)}


# ── 4. analysis ──────────────────────────────────────────────────────────────


@tool
def run_analysis(
    session_id: str,
    name: str,
    params: dict | None = None,
    table: str | None = None,
    include_frames: bool = False,
    max_rows: int = 15,
) -> dict:
    """Run one analysis on a materialized table.

    Returns a headline only by default — scalars plus the shape of each result
    frame. Use get_result to page into the rows you actually want.

    Args:
        session_id: From create_session.
        name: Analysis name; see list_capabilities.
        params: Constructor parameters; see describe_analysis.
        table: Which materialized table to use. Defaults to the active one.
        include_frames: Also inline the (truncated) result frames.
        max_rows: Row cap when include_frames is true.

    Returns:
        Scalars, available frames and arrays, and whether a cached result was replaced.
    """
    s = _session(session_id)
    key = _need_table(s, table)
    kwargs = coerce_params(name, params)
    run = s.run_for(key)
    replaced = bool(kwargs) and name in run.ctx.results
    try:
        result = run.run(name, **kwargs)
    except RuntimeError as exc:
        raise ToolError(
            "AnalysisSkipped",
            str(exc),
            "This analysis needs labels the table does not carry; "
            "check class_col and the analysis's needs_labels.",
        )
    except TypeError as exc:
        raise ToolError(
            "InvalidArgument",
            f"{name}: {exc}",
            f"Call describe_analysis('{name}') for the parameter list.",
        )
    payload = render.result_payload(result, max_rows=max_rows, frames=include_frames)
    payload.update(session_id=session_id, table=key)
    if replaced:
        payload["replaced_previous"] = True
        payload["note"] = (
            "Passing params re-runs the analysis and overwrites the cached result, "
            "which is what a later save_run will persist."
        )
    return payload


@tool
def get_result(
    session_id: str,
    name: str,
    frame: str | None = None,
    columns: list[str] | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
    top_n: int = 30,
    offset: int = 0,
    table: str | None = None,
) -> dict:
    """Page into a frame of an already-computed analysis. Does not recompute.

    Args:
        session_id: From create_session.
        name: The analysis whose result you want.
        frame: Which frame, e.g. "table", "summary", "pairs_long". Omit to list them.
        columns: Restrict to these columns.
        sort_by: Sort by this column before slicing.
        ascending: Sort direction.
        top_n: How many rows to return.
        offset: Skip this many rows first, for paging.
        table: Which table's run to read. Defaults to the active one.

    Returns:
        A bounded slice of the requested frame.
    """
    s = _session(session_id)
    key = _need_table(s, table)
    run = s.run_for(key)
    if name not in run.ctx.results:
        raise ToolError(
            "OrderError",
            f"Analysis {name!r} has not been run on table {key!r}.",
            f"Call run_analysis('{session_id}', '{name}') first.",
        )
    from tessa.results import AnalysisResult

    result = AnalysisResult.from_raw(name, run.ctx.results[name])
    if frame is None:
        return {
            "session_id": session_id,
            "analysis": name,
            "frames_available": sorted(result.frames),
            "arrays_available": sorted(result.arrays),
            "hint": "Re-call with frame=<one of frames_available>.",
        }
    if frame not in result.frames:
        raise ToolError(
            "NotFound",
            f"{name!r} has no frame {frame!r}.",
            f"Available frames: {sorted(result.frames)}.",
        )
    payload = render.frame_payload(
        result.frames[frame],
        max_rows=top_n,
        columns=columns,
        sort_by=sort_by,
        ascending=ascending,
        offset=offset,
    )
    payload.update(session_id=session_id, analysis=name, frame=frame, table=key)
    return payload


@tool
def characterize_classes(session_id: str, table: str | None = None) -> dict:
    """Run the standard first-pass battery and return one digest.

    Runs distributions, importance, pairwise and separability together. Use this
    to find where the classes are weakly separated, then target that weakness
    with new features.

    Args:
        session_id: From create_session.
        table: Which table to characterize. Defaults to the active one.

    Returns:
        The separability verdict, top features, and the weakest class pair.
    """
    s = _session(session_id)
    key = _need_table(s, table)
    run = s.run_for(key)
    digest: dict[str, Any] = {"session_id": session_id, "table": key}
    skipped: list[str] = []

    for name in ("distributions", "importance", "pairwise", "separability"):
        try:
            run.run(name)
        except Exception as exc:
            skipped.append(f"{name}: {type(exc).__name__}: {exc}")
    if skipped:
        digest["skipped"] = skipped

    digest["separability"] = _separability_summary(run)

    if "importance" in run.ctx.results:
        tbl = run.ctx.results["importance"].get("table")
        if tbl is not None:
            top = tbl.sort_values("rank").head(10) if "rank" in tbl else tbl.head(10)
            keep = [c for c in ("rank", "score_composite", "mean_rank") if c in top.columns]
            digest["top_features"] = render.frame_payload(
                top.reset_index().rename(columns={"index": "feature"})
                if top.index.name or "feature" not in top.columns
                else top,
                max_rows=10,
                columns=["feature", *keep] if keep else None,
            )["rows"]

    digest["weakest_pair"] = _weakest_pair(run)
    digest["next"] = (
        "Target the weakest pair: preview_feature candidates, create_feature the "
        "survivors, materialize a new table, then score_features against this one."
    )
    return digest


def _separability_summary(run) -> dict[str, Any]:
    raw = run.ctx.results.get("separability")
    if not raw or "summary" not in raw:
        return {}
    row = raw["summary"].iloc[0].to_dict()
    return {k: render.jsonable(v) for k, v in row.items()}


def _weakest_pair(run) -> dict[str, Any]:
    """The class pair whose best feature separates it least — where to aim next."""
    raw = run.ctx.results.get("pairwise")
    if not raw:
        return {}
    long = raw.get("pairs_long")
    if long is None or len(long) == 0 or "auc" not in long.columns:
        return {}
    best = long.groupby(["class_a", "class_b"])["auc"].max().reset_index()
    if best.empty:
        return {}
    worst = best.loc[best["auc"].idxmin()]
    return {
        "class_a": str(worst["class_a"]),
        "class_b": str(worst["class_b"]),
        "best_feature_auc": render.jsonable(worst["auc"]),
        "interpretation": "No single feature separates this pair well; it is the "
        "most promising target for a new feature.",
    }


# ── 5. scoring the invented features ─────────────────────────────────────────


@tool
def score_features(
    session_id: str,
    table: str | None = None,
    baseline_table: str | None = None,
    feature_names: list[str] | None = None,
) -> dict:
    """Measure whether the features you invented actually helped.

    Compares separability on `table` against `baseline_table`, and reports each
    feature's importance rank and best pairwise AUC. Results are written into the
    feature ledger.

    Args:
        session_id: From create_session.
        table: The augmented table. Defaults to the active one.
        baseline_table: The table to compare against, e.g. "baseline".
        feature_names: Restrict scoring to these features. Defaults to all invented ones.

    Returns:
        The separability delta, per-feature contributions, and a caution about
        selection bias.
    """
    s = _session(session_id)
    key = _need_table(s, table)
    run = s.run_for(key)
    for name in ("separability", "importance", "pairwise"):
        try:
            run.run(name)
        except Exception:
            pass

    current = _separability_summary(run)
    out: dict[str, Any] = {
        "session_id": session_id,
        "table": key,
        "separability": current,
    }

    if baseline_table:
        base_key = _need_table(s, baseline_table)
        base_run = s.run_for(base_key)
        try:
            base_run.run("separability")
        except Exception:
            pass
        base = _separability_summary(base_run)
        out["baseline_table"] = base_key
        out["baseline_separability"] = base
        if base and current:
            delta = float(current.get("cv_balanced_accuracy") or 0) - float(
                base.get("cv_balanced_accuracy") or 0
            )
            out["delta_cv_balanced_accuracy"] = render.jsonable(delta)
            out["verdict_changed"] = base.get("verdict") != current.get("verdict")

    invented = [e.name for e in s.ledger if e.kind in ("feature", "aggregator")]
    wanted = feature_names or invented
    out["scored"] = _per_feature_metrics(s, run, wanted, key)
    out["selection_bias_caution"] = (
        f"{s.n_previewed} feature(s) were previewed this session. Gains measured on "
        "the same data you selected on are optimistic; p-values here are "
        "permutation-based and folds are asset-grouped, but only confirm_on_holdout "
        "tests genuinely unseen assets."
    )
    return out


def _per_feature_metrics(s: AgentSession, run, wanted: list[str], table: str) -> list[dict]:
    imp = run.ctx.results.get("importance", {}).get("table")
    pw = run.ctx.results.get("pairwise", {}).get("pairs_long")
    df = s.tables[table]
    rows = []
    for feat in wanted:
        # A per-sample feature becomes several aggregate columns: <feature>__<agg>.
        cols = [c for c in df.columns if c == feat or c.startswith(f"{feat}__")]
        metrics: dict[str, Any] = {"feature": feat, "columns_in_table": cols}
        if imp is not None and len(cols):
            present = [c for c in cols if c in imp.index]
            if present:
                sub = imp.loc[present]
                if "rank" in sub.columns:
                    metrics["best_importance_rank"] = int(sub["rank"].min())
                if "score_composite" in sub.columns:
                    metrics["max_score_composite"] = render.jsonable(sub["score_composite"].max())
        if pw is not None and len(pw) and "auc" in pw.columns and cols:
            sub = pw[pw["feature"].isin(cols)]
            if len(sub):
                best = sub.loc[sub["auc"].idxmax()]
                metrics["max_pair_auc"] = render.jsonable(best["auc"])
                metrics["best_pair"] = f"{best['class_a']} vs {best['class_b']}"
        entry = s.ledger_entry(feat)
        if entry is not None:
            entry.metrics[table] = {k: v for k, v in metrics.items() if k != "feature"}
        rows.append(metrics)
    return rows


@tool
def feature_ledger(session_id: str) -> dict:
    """The record of every feature invented this session and how it fared.

    Args:
        session_id: From create_session.

    Returns:
        Ledger entries with expression, rationale, measured metrics and verdict.
    """
    s = _session(session_id)
    return {
        "session_id": session_id,
        "n_previewed": s.n_previewed,
        "entries": s.ledger_as_dicts(),
    }


@tool
def confirm_on_holdout(session_id: str, table: str | None = None) -> dict:
    """Re-test on the held-out assets. Use once, at the end.

    Everything before this measured performance on data you selected features
    against. This is the only check against assets the exploration never saw.

    Args:
        session_id: From create_session (which must have been given holdout_assets).
        table: Which table's feature set to re-test. Defaults to the active one.

    Returns:
        Separability measured on the held-out assets alone.
    """
    s = _session(session_id)
    if not s.holdout_assets:
        raise ToolError(
            "InvalidArgument",
            "This session reserved no holdout assets.",
            "Pass holdout_assets to create_session to enable this check.",
        )
    key = _need_table(s, table)
    labels = _need_labels(s)
    ac = s.cfg.asset_col
    held = labels.filter(pl.col(ac).is_in(list(s.holdout_assets)))
    if held.is_empty():
        raise ToolError("NotFound", "No labelled events belong to the held-out assets.")

    events = build_events_dict(held, s.cfg)
    spec_df = s.tables[key]
    feature_cols = [c for c in s.features.names()]
    aggs = sorted(
        {c.split("__")[-1] for c in spec_df.columns if "__" in c} & set(s.aggregators.names())
    )
    df = materialize_frames(
        events,
        WindowSpec.event(),
        s.cfg,
        aggregators=aggs or ["mean", "std", "min", "max"],
        feature_names=feature_cols or [],
        feature_registry=s.features,
        aggregator_registry=s.aggregators,
    )
    from tessa.run import Run

    holdout_run = Run(df, target_col=s.cfg.class_col, cfg=s.cfg, label_cols=events.label_cols)
    try:
        holdout_run.run("separability")
    except Exception as exc:
        raise ToolError("AnalysisSkipped", f"Could not evaluate the holdout: {exc}")
    return {
        "session_id": session_id,
        "holdout_assets": list(s.holdout_assets),
        "n_events": df.height,
        "separability": _separability_summary(holdout_run),
        "note": "Measured on assets excluded from every earlier step.",
    }


# ── 6. persistence ───────────────────────────────────────────────────────────


@tool
def save_run(session_id: str, name: str | None = None, table: str | None = None) -> dict:
    """Save the completed analyses, with the feature ledger in the manifest.

    Args:
        session_id: From create_session.
        name: Run directory name. Defaults to a timestamp.
        table: Which table's results to save. Defaults to the active one.

    Returns:
        The path written, and which analyses it contains.
    """
    s = _session(session_id)
    key = _need_table(s, table)
    run = s.run_for(key)
    if not run.ctx.results:
        raise ToolError(
            "OrderError",
            "No analyses have been run on this table.",
            "Call run_analysis or characterize_classes first.",
        )
    store = ResultStore(Path(s.cfg.output_dir) / "runs")
    path = store.save_run(
        run.ctx.results,
        s.cfg,
        df=s.tables[key],
        name=name,
        extra_manifest={
            "target_col": run.ctx.target_col,
            "table": key,
            "session_id": session_id,
            "feature_ledger": s.ledger_as_dicts(),
            "holdout_assets": list(s.holdout_assets),
        },
    )
    return {
        "session_id": session_id,
        "path": str(path),
        "analyses": sorted(run.ctx.results),
        "note": "The manifest records the expressions behind every invented column.",
    }


@tool
def write_report(session_id: str, path: str | None = None, table: str | None = None) -> dict:
    """Write a self-contained static HTML report of the analyses run so far.

    Args:
        session_id: From create_session.
        path: Destination file. Defaults to <output_dir>/report.html.
        table: Which table's results to report. Defaults to the active one.

    Returns:
        The path written.
    """
    s = _session(session_id)
    key = _need_table(s, table)
    run = s.run_for(key)
    if not run.ctx.results:
        raise ToolError(
            "OrderError",
            "No analyses have been run on this table.",
            "Call run_analysis or characterize_classes first.",
        )
    out = Path(path) if path else Path(s.cfg.output_dir) / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    written = run.report(out)
    return {"session_id": session_id, "path": str(written)}


# ── the surface ──────────────────────────────────────────────────────────────

TOOL_FUNCTIONS: list[Callable[..., dict]] = [
    create_session,
    describe_data,
    build_events,
    list_capabilities,
    describe_analysis,
    seed_builtin_features,
    preview_feature,
    create_feature,
    create_aggregator,
    drop_feature,
    materialize,
    set_active_table,
    run_analysis,
    get_result,
    characterize_classes,
    score_features,
    feature_ledger,
    confirm_on_holdout,
    save_run,
    write_report,
]

TOOLS_BY_NAME: dict[str, Callable[..., dict]] = {fn.__name__: fn for fn in TOOL_FUNCTIONS}


def call_tool(name: str, arguments: dict[str, Any]) -> dict:
    """Dispatch by name, validating the arguments against the signature."""
    fn = TOOLS_BY_NAME.get(name)
    if fn is None:
        return ToolError(
            "UnknownTool", f"No tool named {name!r}.", f"Available tools: {sorted(TOOLS_BY_NAME)}."
        ).payload()
    try:
        inspect.signature(fn).bind(**arguments)
    except TypeError as exc:
        return ToolError(
            "InvalidArgument", f"{name}: {exc}", f"Signature: {name}{inspect.signature(fn)}"
        ).payload()
    return fn(**arguments)

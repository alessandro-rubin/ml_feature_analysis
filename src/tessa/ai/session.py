"""Per-exploration state for the AI layer.

A session owns everything one exploration run mutates: the config, the loaded
labels and events, **its own feature and aggregator registries**, the analysis
tables it has materialized, and the ledger of features the model invented.

Registries are cloned rather than shared, so a model inventing
``temperature__volatility`` can never leak it into the process-wide registry or
into a concurrent session. That is what makes the ``feature_registry=`` seam on
the materializers safe to hand to a language model.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import polars as pl

# Registers the nine stock aggregators (mean/std/min/max/median/p05/p95/range/iqr).
# Importing `tessa.features` alone does NOT do this — the default aggregator
# registry is empty until `builtins` is imported.
import tessa.features.builtins  # noqa: F401
from tessa.config import Config
from tessa.features import default_aggregator_registry, default_feature_registry
from tessa.features.aggregates import AggregatorRegistry
from tessa.features.registry import FeatureRegistry
from tessa.run import Run

__all__ = [
    "FeatureLedgerEntry",
    "AgentSession",
    "create_session",
    "get_session",
    "list_sessions",
    "drop_session",
]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class FeatureLedgerEntry:
    """One feature the model invented, and how it fared.

    The ledger is the reproducibility record: without it, an AI-authored column
    in a saved run has no explanation attached and cannot be rebuilt.
    """

    name: str
    kind: Literal["feature", "aggregator", "builtin"]
    expression: str
    rationale: str
    deps: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utcnow)
    preview: dict[str, Any] = field(default_factory=dict)
    tables: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    verdict: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["deps"] = list(self.deps)
        return d


@dataclass
class AgentSession:
    """Mutable state for one exploration run."""

    session_id: str
    cfg: Config
    labels: pl.DataFrame | None = None
    events: dict[str, pl.LazyFrame] | None = None
    features: FeatureRegistry = field(default_factory=FeatureRegistry)
    aggregators: AggregatorRegistry = field(default_factory=AggregatorRegistry)
    tables: dict[str, pl.DataFrame] = field(default_factory=dict)
    active_table: str | None = None
    runs: dict[str, Run] = field(default_factory=dict)
    ledger: list[FeatureLedgerEntry] = field(default_factory=list)
    holdout_assets: tuple[str, ...] = ()
    n_previewed: int = 0
    created_at: str = field(default_factory=_utcnow)

    # ── table access ────────────────────────────────────────────────────────

    def table(self, name: str | None = None) -> pl.DataFrame:
        """Return a materialized table by name, defaulting to the active one."""
        key = name or self.active_table
        if key is None:
            raise KeyError("No table has been materialized yet.")
        if key not in self.tables:
            raise KeyError(f"Unknown table {key!r}; have {sorted(self.tables)}.")
        return self.tables[key]

    def run_for(self, name: str | None = None) -> Run:
        """Return the :class:`~tessa.run.Run` bound to a table, creating it once.

        One ``Run`` per table means ``prepare_xy`` caching and ``ctx.results``
        reuse span every analysis the model asks for on that table.
        """
        key = name or self.active_table
        if key is None:
            raise KeyError("No table has been materialized yet.")
        if key not in self.runs:
            df = self.table(key)
            target = self.cfg.class_col if self.cfg.class_col in df.columns else None
            self.runs[key] = Run(
                df,
                target_col=target,
                cfg=self.cfg,
                label_cols=getattr(self.events, "label_cols", ()),
            )
        return self.runs[key]

    def invalidate_runs(self, name: str) -> None:
        """Drop the cached Run for a table (used when the table is rebuilt)."""
        self.runs.pop(name, None)

    # ── ledger ──────────────────────────────────────────────────────────────

    def ledger_entry(self, name: str) -> FeatureLedgerEntry | None:
        return next((e for e in self.ledger if e.name == name), None)

    def ledger_as_dicts(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.ledger]

    # ── holdout ─────────────────────────────────────────────────────────────

    def visible_filter(self) -> dict[str, list] | None:
        """A ``label_filter`` excluding the held-out assets, or None.

        Held-out assets stay invisible to every iteration so a final
        ``confirm_on_holdout`` has genuinely unseen data to check against.
        """
        if not self.holdout_assets or self.labels is None:
            return None
        col = self.cfg.asset_col
        if col not in self.labels.columns:
            return None
        # Sorted: `unique()` does not guarantee an order, and an unstable
        # label_filter would make runs non-reproducible.
        keep = sorted(
            a for a in self.labels[col].unique().to_list() if a not in self.holdout_assets
        )
        return {col: keep}


_SESSIONS: dict[str, AgentSession] = {}


def create_session(
    data_root: str | Path,
    *,
    output_dir: str | Path = "outputs",
    class_col: str = "class",
    asset_col: str = "asset_id",
    timestamp_col: str = "timestamp",
    filename_pattern: str | None = None,
    random_state: int = 42,
    assume_sorted: bool = False,
    holdout_assets: tuple[str, ...] = (),
) -> AgentSession:
    """Create a session with cloned registries and register it in the store."""
    kwargs: dict[str, Any] = dict(
        data_root=Path(data_root),
        output_dir=Path(output_dir),
        class_col=class_col,
        asset_col=asset_col,
        timestamp_col=timestamp_col,
        random_state=random_state,
        assume_sorted=assume_sorted,
    )
    if filename_pattern:
        kwargs["filename_pattern"] = filename_pattern
    cfg = Config(**kwargs)

    session = AgentSession(
        session_id=uuid.uuid4().hex[:8],
        cfg=cfg,
        # The default feature registry is normally empty; the aggregator registry
        # carries the nine built-ins imported above.
        features=default_feature_registry().copy(),
        aggregators=default_aggregator_registry().copy(),
        holdout_assets=tuple(holdout_assets),
    )
    _SESSIONS[session.session_id] = session
    return session


def get_session(session_id: str) -> AgentSession:
    """Look up a live session.

    Raises
    ------
    KeyError
        With the live session ids in the message, so a caller that invented an
        id can recover.
    """
    if session_id not in _SESSIONS:
        raise KeyError(
            f"Unknown session {session_id!r}. Live sessions: {sorted(_SESSIONS) or 'none'}. "
            "Call create_session first."
        )
    return _SESSIONS[session_id]


def list_sessions() -> list[str]:
    return sorted(_SESSIONS)


def drop_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)

"""Session isolation: an invented feature must never escape its session."""

from __future__ import annotations

import polars as pl
import pytest

from tessa.ai.expressions import compile_feature_expression
from tessa.ai.session import (
    AgentSession,
    FeatureLedgerEntry,
    create_session,
    drop_session,
    get_session,
    list_sessions,
)
from tessa.features import default_aggregator_registry, default_feature_registry
from tessa.features.registry import FeatureSpec


def _register(session: AgentSession, name: str, src: str) -> None:
    c = compile_feature_expression(src)
    session.features.register(FeatureSpec(name=name, deps=c.deps, factory=c.factory))


@pytest.fixture
def session(tmp_path):
    s = create_session(tmp_path, output_dir=tmp_path / "out")
    yield s
    drop_session(s.session_id)


def test_session_starts_with_the_stock_aggregators(session):
    # `import tessa.features` alone leaves these unregistered; session.py imports
    # tessa.features.builtins precisely so this holds.
    for name in ("mean", "std", "min", "max", "median", "p05", "p95", "range", "iqr"):
        assert name in session.aggregators


def test_registering_does_not_touch_the_process_wide_registry(session):
    _register(session, "x__vol", 'pl.col("x").rolling_std(5)')
    assert "x__vol" in session.features
    assert "x__vol" not in default_feature_registry()


def test_two_sessions_are_isolated(tmp_path):
    a = create_session(tmp_path, output_dir=tmp_path / "a")
    b = create_session(tmp_path, output_dir=tmp_path / "b")
    try:
        _register(a, "shared_name", 'pl.col("x").abs()')
        assert "shared_name" not in b.features
        # The same name is free in the other session.
        _register(b, "shared_name", 'pl.col("x").diff()')
        assert a.features.get("shared_name").deps == ("x",)
    finally:
        drop_session(a.session_id)
        drop_session(b.session_id)


def test_aggregator_registry_is_cloned_not_shared(session):
    before = set(default_aggregator_registry().names())
    session.aggregators.unregister("iqr")
    assert "iqr" not in session.aggregators
    assert set(default_aggregator_registry().names()) == before


def test_unregister_then_recreate_allows_revision(session):
    _register(session, "f", 'pl.col("x").abs()')
    with pytest.raises(ValueError, match="already registered"):
        _register(session, "f", 'pl.col("x").diff()')
    session.features.unregister("f")
    _register(session, "f", 'pl.col("x").diff()')
    df = pl.DataFrame({"x": [1.0, 4.0]})
    assert df.with_columns(session.features.get("f").expr())["f"].to_list() == [None, 3.0]


# ── store ────────────────────────────────────────────────────────────────────


def test_unknown_session_error_lists_live_ids(session):
    with pytest.raises(KeyError) as exc:
        get_session("deadbeef")
    assert session.session_id in str(exc.value)
    assert session.session_id in list_sessions()


# ── tables and runs ──────────────────────────────────────────────────────────


def test_table_lookup_errors_are_actionable(session):
    with pytest.raises(KeyError, match="No table has been materialized"):
        session.table()
    session.tables["baseline"] = pl.DataFrame({"a": [1]})
    session.active_table = "baseline"
    assert session.table().shape == (1, 1)
    with pytest.raises(KeyError, match="baseline"):
        session.table("nope")


def test_run_is_created_once_per_table(session):
    session.tables["t"] = pl.DataFrame({"class": ["a", "b"], "f": [1.0, 2.0]})
    session.active_table = "t"
    assert session.run_for() is session.run_for()
    session.invalidate_runs("t")
    assert "t" not in session.runs


def test_run_target_col_is_none_without_a_class_column(session):
    session.tables["t"] = pl.DataFrame({"f": [1.0, 2.0]})
    session.active_table = "t"
    assert session.run_for().ctx.target_col is None


# ── holdout + ledger ─────────────────────────────────────────────────────────


def test_visible_filter_excludes_held_out_assets(tmp_path):
    s = create_session(tmp_path, output_dir=tmp_path / "o", holdout_assets=("A3",))
    try:
        s.labels = pl.DataFrame({"asset_id": ["A1", "A2", "A3"], "class": ["x", "y", "z"]})
        assert s.visible_filter() == {"asset_id": ["A1", "A2"]}
    finally:
        drop_session(s.session_id)


def test_visible_filter_is_none_without_holdout(session):
    session.labels = pl.DataFrame({"asset_id": ["A1"], "class": ["x"]})
    assert session.visible_filter() is None


def test_ledger_entries_are_json_serializable(session):
    import json

    session.ledger.append(
        FeatureLedgerEntry(
            name="f",
            kind="feature",
            expression='pl.col("x").abs()',
            rationale="magnitude matters",
            deps=("x",),
        )
    )
    json.dumps(session.ledger_as_dicts())
    assert session.ledger_entry("f").rationale == "magnitude matters"
    assert session.ledger_entry("missing") is None

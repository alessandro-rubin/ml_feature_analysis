"""The tool surface, end to end, with no model and no network.

This is the offline proof that an AI-authored feature reaches the analyses.
"""

from __future__ import annotations

import json

import pytest

from tessa.ai import tools
from tessa.ai.session import drop_session


def _ok(payload: dict) -> dict:
    assert "error" not in payload, payload.get("error")
    json.dumps(payload)  # every payload must survive the wire
    return payload


@pytest.fixture
def sid(synthetic_root, synthetic_labels, tmp_path):
    out = _ok(
        tools.create_session(
            data_root=str(synthetic_root),
            labels_path=str(synthetic_labels),
            output_dir=str(tmp_path / "out"),
        )
    )
    session_id = out["session_id"]
    yield session_id
    drop_session(session_id)


@pytest.fixture
def built(sid):
    _ok(tools.build_events(sid))
    return sid


# ── session + discovery ──────────────────────────────────────────────────────


def test_create_session_reports_classes_and_assets(sid):
    out = _ok(tools.describe_data(sid))
    assert out["class_counts"] == {"calm": 12, "jittery": 12}
    assert out["assets"] == ["A1", "A2", "A3"]
    assert "vibration" in out["numeric_channels"]
    assert out["numeric_channels"] and "timestamp" not in out["numeric_channels"]


def test_create_session_rejects_a_missing_root(tmp_path):
    err = tools.create_session(data_root=str(tmp_path / "nope"))["error"]
    assert err["type"] == "NotFound"


def test_list_capabilities_exposes_the_expression_language(sid):
    out = _ok(tools.list_capabilities(sid))
    assert "rolling_std" in out["expression_language"]["expression_methods"]
    assert "mean" in out["aggregators"]
    assert out["features"] == []
    assert "separability" in out["analyses"]


# ── order enforcement ────────────────────────────────────────────────────────


def test_materialize_before_build_is_an_actionable_error(sid):
    err = tools.materialize(sid)["error"]
    assert err["type"] == "OrderError"
    assert "build_events" in err["hint"]


def test_run_analysis_before_materialize_is_an_actionable_error(built):
    err = tools.run_analysis(built, "importance")["error"]
    assert err["type"] == "OrderError"
    assert "materialize" in err["hint"]


def test_unknown_session_error_is_recoverable(built):
    err = tools.describe_data("nosuchid")["error"]
    assert built in err["message"]


def test_unknown_analysis_lists_valid_names(built):
    _ok(tools.materialize(built))
    err = tools.run_analysis(built, "importance_test")["error"]
    assert err["type"] == "UnknownAnalysis"
    assert "separability" in err["valid"]


def test_unknown_parameter_lists_valid_parameters(built):
    _ok(tools.materialize(built))
    err = tools.run_analysis(built, "pairwise", params={"topn": 3})["error"]
    assert "top_n" in err["message"]


# ── feature invention ────────────────────────────────────────────────────────


def test_preview_does_not_register(built):
    out = _ok(tools.preview_feature(built, 'pl.col("vibration").rolling_std(15)'))
    assert out["deps"] == ["vibration"]
    assert out["null_fraction"] > 0  # rolling window leaves a leading gap
    assert _ok(tools.list_capabilities(built))["features"] == []


def test_preview_rejects_an_unsafe_expression(built):
    err = tools.preview_feature(built, '__import__("os").getcwd()')["error"]
    assert err["type"] == "ExpressionError"
    assert "list_capabilities" in err["hint"]


def test_preview_reports_a_bad_channel_name_usefully(built):
    err = tools.preview_feature(built, 'pl.col("no_such_channel").abs()')["error"]
    assert err["type"] == "PolarsError"
    assert "describe_data" in err["hint"]


def test_create_feature_registers_and_records_a_rationale(built):
    out = _ok(
        tools.create_feature(
            built,
            "vibration__vol",
            'pl.col("vibration").rolling_std(15)',
            rationale="the classes differ in short-timescale volatility",
        )
    )
    assert out["registered"] == "vibration__vol"
    assert "vibration__vol" in _ok(tools.list_capabilities(built))["features"]
    entry = _ok(tools.feature_ledger(built))["entries"][0]
    assert entry["rationale"].startswith("the classes differ")
    assert entry["kind"] == "feature"


def test_create_feature_requires_a_rationale(built):
    err = tools.create_feature(built, "f", 'pl.col("vibration").abs()', rationale=" ")["error"]
    assert err["type"] == "InvalidArgument"


def test_create_feature_refuses_reserved_names(built):
    err = tools.create_feature(built, "asset_id", 'pl.col("vibration").abs()', rationale="x")[
        "error"
    ]
    assert err["type"] == "InvalidArgument"


def test_duplicate_feature_needs_replace(built):
    args = (built, "f", 'pl.col("vibration").abs()')
    _ok(tools.create_feature(*args, rationale="first"))
    err = tools.create_feature(*args, rationale="second")["error"]
    assert err["type"] == "DuplicateName"
    _ok(
        tools.create_feature(
            built, "f", 'pl.col("vibration").diff()', rationale="revised", replace=True
        )
    )


def test_broken_expression_never_enters_the_registry(built):
    tools.create_feature(built, "bad", 'pl.col("vibration").rolling_std()', rationale="x")
    assert "bad" not in _ok(tools.list_capabilities(built))["features"]


def test_custom_aggregator_produces_columns(built):
    _ok(
        tools.create_aggregator(
            built,
            "spread",
            "pl.col(c).quantile(0.9) - pl.col(c).quantile(0.1)",
            rationale="robust spread, insensitive to outliers",
        )
    )
    out = _ok(tools.materialize(built, aggregators=["mean", "spread"]))
    assert any(c.endswith("__spread") for c in out["columns_sample"])


def test_drop_feature_marks_the_ledger(built):
    _ok(tools.create_feature(built, "f", 'pl.col("vibration").abs()', rationale="x"))
    _ok(tools.drop_feature(built, "f"))
    assert _ok(tools.feature_ledger(built))["entries"][0]["verdict"] == "dropped"


def test_seed_builtin_features_registers_a_baseline(built):
    out = _ok(tools.seed_builtin_features(built, sources=["vibration"], windows=[10]))
    assert "vibration__roll_std_10" in out["registered"]
    assert "vibration__diff1" in out["registered"]


# ── materialization ──────────────────────────────────────────────────────────


def test_materialize_excludes_features_by_default(built):
    """`feature_names=None` must mean none — features multiply by aggregators."""
    _ok(
        tools.create_feature(
            built, "vibration__vol", 'pl.col("vibration").rolling_std(15)', rationale="volatility"
        )
    )
    out = _ok(tools.materialize(built))
    assert not any("vibration__vol" in c for c in out["columns_sample"])


def test_materialize_all_features_is_explicit(built):
    _ok(
        tools.create_feature(
            built, "vibration__vol", 'pl.col("vibration").rolling_std(15)', rationale="volatility"
        )
    )
    out = _ok(tools.materialize(built, feature_names="__all__", table_name="aug"))
    assert any(c.startswith("vibration__vol__") for c in out["columns_sample"])


def test_materialize_reports_shape_and_class_counts(built):
    out = _ok(tools.materialize(built))
    assert out["shape"][0] == 24
    assert out["class_counts"] == {"calm": 12, "jittery": 12}


def test_column_budget_refuses_an_explosive_table(built):
    _ok(tools.seed_builtin_features(built, windows=[5, 10, 20, 30]))
    err = tools.materialize(built, feature_names="__all__", max_columns=20)["error"]
    assert err["type"] == "BudgetError"
    assert err["projected_columns"] > 20
    assert "Narrow" in err["hint"]


def test_unknown_feature_and_aggregator_names_are_reported(built):
    assert tools.materialize(built, feature_names=["ghost"])["error"]["type"] == "NotFound"
    assert tools.materialize(built, aggregators=["ghost"])["error"]["type"] == "NotFound"


def test_set_active_table_switches_context(built):
    _ok(tools.materialize(built, table_name="a"))
    _ok(tools.materialize(built, table_name="b"))
    assert _ok(tools.set_active_table(built, "a"))["active_table"] == "a"
    assert tools.set_active_table(built, "ghost")["error"]["type"] == "NotFound"


# ── analysis + results ───────────────────────────────────────────────────────


def test_run_analysis_returns_a_headline_not_the_frames(built):
    _ok(tools.materialize(built))
    out = _ok(tools.run_analysis(built, "importance"))
    assert "frames" not in out
    assert "table" in out["frames_available"]


def test_get_result_pages_into_a_frame(built):
    _ok(tools.materialize(built))
    _ok(tools.run_analysis(built, "importance"))
    listing = _ok(tools.get_result(built, "importance"))
    assert "table" in listing["frames_available"]
    page = _ok(tools.get_result(built, "importance", frame="table", top_n=3))
    assert page["n_rows_returned"] == 3


def test_get_result_before_run_is_an_actionable_error(built):
    _ok(tools.materialize(built))
    err = tools.get_result(built, "importance", frame="table")["error"]
    assert err["type"] == "OrderError"


def test_pairwise_exposes_pairs_long(built):
    """`pairs` is tuple-keyed and lands in `objects`; `pairs_long` is the usable frame."""
    _ok(tools.materialize(built))
    _ok(tools.run_analysis(built, "pairwise"))
    listing = _ok(tools.get_result(built, "pairwise"))
    assert "pairs_long" in listing["frames_available"]
    assert "pairs" not in listing["frames_available"]


def test_characterize_classes_returns_a_digest(built):
    _ok(tools.materialize(built))
    out = _ok(tools.characterize_classes(built))
    assert "verdict" in out["separability"]
    assert "next" in out


# ── the whole point: an invented feature measurably helps ────────────────────


def test_invented_feature_flips_the_separability_verdict(built):
    """The whole thesis, end to end.

    The fixture's classes share every marginal statistic by construction, so a
    period-aggregate table genuinely cannot tell them apart. A single invented
    feature that looks at *ordering* rather than level should move the verdict
    from "not separable" to "separable".
    """
    _ok(tools.materialize(built, aggregators=["mean", "std", "min", "max"], table_name="baseline"))
    _ok(tools.run_analysis(built, "separability", params={"n_permutations": 60}))

    _ok(
        tools.create_feature(
            built,
            "vibration__jerk",
            'pl.col("vibration").diff().abs()',
            rationale="the classes differ in sample-to-sample ordering, not in level, "
            "so a difference-based feature should see what the aggregates cannot",
        )
    )
    _ok(
        tools.materialize(
            built,
            aggregators=["mean", "std", "min", "max"],
            feature_names=["vibration__jerk"],
            table_name="augmented",
        )
    )
    _ok(tools.run_analysis(built, "separability", params={"n_permutations": 60}))

    scored = _ok(tools.score_features(built, table="augmented", baseline_table="baseline"))
    base, aug = scored["baseline_separability"], scored["separability"]

    assert base["verdict"] == "not separable", base
    assert aug["verdict"] == "separable", aug
    assert scored["delta_cv_balanced_accuracy"] > 0.2, scored
    assert aug["perm_p_value"] < 0.05 < base["perm_p_value"]
    assert scored["verdict_changed"] is True

    # The gain is attributed to the feature, and the caution travels with it.
    entry = next(
        e for e in _ok(tools.feature_ledger(built))["entries"] if e["name"] == "vibration__jerk"
    )
    assert "augmented" in entry["metrics"]
    assert entry["metrics"]["augmented"]["best_importance_rank"] == 1
    assert "confirm_on_holdout" in scored["selection_bias_caution"]


# ── persistence ──────────────────────────────────────────────────────────────


def test_save_run_puts_the_ledger_in_the_manifest(built, tmp_path):
    _ok(tools.materialize(built))
    _ok(
        tools.create_feature(
            built, "vibration__vol", 'pl.col("vibration").rolling_std(15)', rationale="volatility"
        )
    )
    _ok(tools.run_analysis(built, "importance"))
    _ok(tools.save_run(built, name="testrun"))

    manifest = json.loads((tmp_path / "out" / "runs" / "testrun" / "manifest.json").read_text())
    ledger = manifest["feature_ledger"]
    assert any(e["expression"] == 'pl.col("vibration").rolling_std(15)' for e in ledger)


def test_save_run_before_any_analysis_is_an_error(built):
    _ok(tools.materialize(built))
    assert tools.save_run(built)["error"]["type"] == "OrderError"


def test_write_report_produces_html(built, tmp_path):
    _ok(tools.materialize(built))
    _ok(tools.run_analysis(built, "importance"))
    out = _ok(tools.write_report(built, path=str(tmp_path / "r.html")))
    assert (tmp_path / "r.html").read_text().lstrip().lower().startswith("<!doctype html")
    assert out["path"].endswith("r.html")


# ── dispatch ─────────────────────────────────────────────────────────────────


def test_call_tool_dispatches_and_validates(built):
    assert _ok(tools.call_tool("list_capabilities", {"session_id": built}))
    assert tools.call_tool("nope", {})["error"]["type"] == "UnknownTool"
    bad = tools.call_tool("list_capabilities", {"wrong_arg": 1})["error"]
    assert bad["type"] == "InvalidArgument" and "Signature" in bad["hint"]


def test_every_tool_has_a_docstring_with_args_and_returns():
    for fn in tools.TOOL_FUNCTIONS:
        doc = fn.__doc__ or ""
        assert "Returns:" in doc, fn.__name__
        params = [p for p in tools.inspect.signature(fn).parameters]
        if params:
            assert "Args:" in doc, fn.__name__


# ── holdout ──────────────────────────────────────────────────────────────────


def test_holdout_assets_are_hidden_then_usable_for_confirmation(
    synthetic_root, synthetic_labels, tmp_path
):
    """Held-out assets must be invisible during exploration and available once."""
    sid = _ok(
        tools.create_session(
            data_root=str(synthetic_root),
            labels_path=str(synthetic_labels),
            output_dir=str(tmp_path / "out"),
            holdout_assets=["A3"],
        )
    )["session_id"]
    try:
        built = _ok(tools.build_events(sid))
        assert built["holdout_excluded"] == ["A3"]
        assert "A3" not in built["events_per_asset"]

        _ok(
            tools.create_feature(
                sid, "vibration__jerk", 'pl.col("vibration").diff().abs()', rationale="ordering"
            )
        )
        _ok(tools.materialize(sid, feature_names=["vibration__jerk"]))

        out = _ok(tools.confirm_on_holdout(sid))
        assert out["holdout_assets"] == ["A3"]
        assert out["n_events"] == 8  # A3 only
        assert "cv_balanced_accuracy" in out["separability"]
    finally:
        drop_session(sid)


def test_confirm_on_holdout_requires_a_reserved_holdout(built):
    _ok(tools.materialize(built))
    err = tools.confirm_on_holdout(built)["error"]
    assert err["type"] == "InvalidArgument"
    assert "holdout_assets" in err["hint"]

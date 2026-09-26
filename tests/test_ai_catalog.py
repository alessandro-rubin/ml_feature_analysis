"""The catalogue is introspected, so it cannot drift from the analyses."""

from __future__ import annotations

import json

import pytest

from tessa.ai.catalog import (
    UnknownAnalysis,
    analysis_catalogue,
    analysis_entry,
    analysis_names,
    analysis_params,
    coerce_params,
    compact_catalogue,
)
from tessa.run import _ANALYSES


def test_catalogue_covers_exactly_the_registered_analyses():
    assert set(analysis_names()) == set(_ANALYSES)


@pytest.mark.parametrize("name", sorted(_ANALYSES))
def test_every_analysis_has_a_description(name):
    """Drift alarm: a new analysis must be documented before it is exposed."""
    assert analysis_entry(name)["description"].strip(), f"{name} has no description"


@pytest.mark.parametrize("name", sorted(_ANALYSES))
def test_parameters_are_real_fields_and_json_expressible(name):
    import dataclasses

    fields = {f.name for f in dataclasses.fields(_ANALYSES[name])}
    for spec in analysis_params(name):
        assert spec.name in fields
        json.dumps(spec.default)


def test_protocol_fields_are_never_exposed_as_parameters():
    for name in analysis_names():
        names = {p.name for p in analysis_params(name)}
        assert not names & {"name", "requires", "needs_labels"}


def test_catalogue_is_json_serializable():
    json.dumps(analysis_catalogue())


def test_compact_catalogue_is_deterministic():
    """It sits in the cached prompt prefix; variation would destroy the cache."""
    assert compact_catalogue() == compact_catalogue()
    assert len(compact_catalogue().splitlines()) == len(_ANALYSES)


def test_pairwise_advertises_pairs_long_not_pairs():
    """`pairs` is tuple-keyed, lands in `objects`, and ResultStore drops it."""
    frames = analysis_entry("pairwise")["result_frames"]
    assert "pairs_long" in frames and "pairs" not in frames


def test_classifier_caveat_mentions_optional_boosting():
    assert "lightgbm" in analysis_entry("classifier")["caveat"]


# ── coercion ─────────────────────────────────────────────────────────────────


def test_json_lists_are_coerced_to_the_tuples_fields_expect():
    assert coerce_params("clustering", {"k_range": [2, 8]}) == {"k_range": (2, 8)}
    assert coerce_params("distributions", {"quantiles": [0.1, 0.9]})["quantiles"] == (0.1, 0.9)
    assert coerce_params("pairwise", {"pairs": [["TP", "FP"]]})["pairs"] == [("TP", "FP")]


def test_plain_values_pass_through_untouched():
    assert coerce_params("separability", {"n_splits": 3}) == {"n_splits": 3}
    assert coerce_params("importance", None) == {}
    assert coerce_params("changepoint", {"channels": ["a", "b"]}) == {"channels": ["a", "b"]}


def test_unknown_parameter_error_lists_the_valid_ones():
    with pytest.raises(ValueError, match="Valid parameters"):
        coerce_params("pairwise", {"topn": 5})


def test_unknown_analysis_error_lists_the_valid_ones():
    with pytest.raises(UnknownAnalysis, match="separability"):
        analysis_params("importance_test")

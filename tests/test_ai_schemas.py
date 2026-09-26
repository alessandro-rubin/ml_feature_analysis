"""Schemas are derived once and adapted per provider, so they cannot drift."""

from __future__ import annotations

import inspect
import json

import pytest

from tessa.ai import tools
from tessa.ai.schemas import to_anthropic_tools, to_openai_tools, tool_schema, tool_schemas


@pytest.mark.parametrize("fn", tools.TOOL_FUNCTIONS, ids=lambda f: f.__name__)
def test_every_tool_gets_a_complete_schema(fn):
    s = tool_schema(fn)
    assert s["name"] == fn.__name__
    assert len(s["description"]) > 30, "description must actually explain the tool"
    params = s["parameters"]
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    json.dumps(s)


@pytest.mark.parametrize("fn", tools.TOOL_FUNCTIONS, ids=lambda f: f.__name__)
def test_schema_properties_match_the_signature(fn):
    props = tool_schema(fn)["parameters"]["properties"]
    assert set(props) == set(inspect.signature(fn).parameters)


@pytest.mark.parametrize("fn", tools.TOOL_FUNCTIONS, ids=lambda f: f.__name__)
def test_required_is_exactly_the_defaultless_parameters(fn):
    sig = inspect.signature(fn)
    expected = [n for n, p in sig.parameters.items() if p.default is inspect.Parameter.empty]
    assert tool_schema(fn)["parameters"]["required"] == expected


@pytest.mark.parametrize("fn", tools.TOOL_FUNCTIONS, ids=lambda f: f.__name__)
def test_every_parameter_is_documented(fn):
    props = tool_schema(fn)["parameters"]["properties"]
    for name, schema in props.items():
        assert schema.get("description"), f"{fn.__name__}.{name} is undocumented"


def test_annotations_resolve_to_json_types():
    """`from __future__ import annotations` makes these strings; they must resolve."""
    props = tool_schema(tools.create_feature)["parameters"]["properties"]
    assert props["session_id"]["type"] == "string"
    assert props["replace"]["type"] == "boolean"
    agg = tool_schema(tools.materialize)["parameters"]["properties"]["aggregators"]
    assert agg["type"] == "array" and agg["items"]["type"] == "string"
    assert tool_schema(tools.materialize)["parameters"]["properties"]["max_columns"]["type"] == (
        "integer"
    )


def test_multi_type_union_leaves_the_type_open():
    """feature_names accepts a list or the string "__all__"; pinning one would lie."""
    schema = tool_schema(tools.materialize)["parameters"]["properties"]["feature_names"]
    assert "type" not in schema
    assert "__all__" in schema["description"]


def test_optional_scalars_keep_their_type():
    schema = tool_schema(tools.get_result)["parameters"]["properties"]["frame"]
    assert schema["type"] == "string"


# ── adapters ─────────────────────────────────────────────────────────────────


def test_openai_and_anthropic_carry_the_same_tools():
    openai = {t["function"]["name"] for t in to_openai_tools()}
    anthropic = {t["name"] for t in to_anthropic_tools()}
    assert openai == anthropic == set(tools.TOOLS_BY_NAME)


def test_openai_shape():
    t = to_openai_tools()[0]
    assert t["type"] == "function"
    assert set(t["function"]) == {"name", "description", "parameters"}
    json.dumps(to_openai_tools())


def test_anthropic_shape():
    t = to_anthropic_tools()[0]
    assert set(t) == {"name", "description", "input_schema"}
    json.dumps(to_anthropic_tools())


def test_tool_order_is_stable():
    """The tool list sits in the cached prompt prefix; reordering breaks the cache."""
    assert [t["name"] for t in tool_schemas()] == [t["name"] for t in tool_schemas()]
    assert [t["name"] for t in tool_schemas()] == [f.__name__ for f in tools.TOOL_FUNCTIONS]


def test_tool_and_schema_modules_import_no_sdk():
    """One implementation must serve every driver, so these stay SDK-free."""
    import subprocess
    import sys

    code = (
        "import sys; import tessa.ai.tools, tessa.ai.schemas; "
        "bad=[m for m in ('anthropic','openai','mcp') if m in sys.modules]; "
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"SDK leaked into the tool import graph: {out.stdout}"

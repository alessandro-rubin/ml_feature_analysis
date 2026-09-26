"""The MCP server must expose exactly the same tools as the agent drivers."""

from __future__ import annotations

import asyncio

import pytest

from tessa.ai.tools import TOOLS_BY_NAME

pytest.importorskip("mcp")

from tessa.ai.mcp_server import build_server  # noqa: E402


@pytest.fixture(scope="module")
def listed():
    server = build_server()
    return asyncio.run(server.list_tools())


def test_mcp_exposes_the_whole_tool_surface(listed):
    assert sorted(t.name for t in listed) == sorted(TOOLS_BY_NAME)


def test_every_mcp_tool_is_described(listed):
    for tool in listed:
        assert tool.description and len(tool.description) > 30, tool.name


def test_required_arguments_survive_registration(listed):
    tool = next(t for t in listed if t.name == "create_feature")
    schema = getattr(tool, "input_schema", None) or tool.inputSchema
    assert set(schema["required"]) == {"session_id", "name", "expression", "rationale"}


def test_server_class_lookup_spans_mcp_versions():
    """mcp 2.x renamed FastMCP to MCPServer; both must work."""
    from tessa.ai.mcp_server import _server_class

    cls = _server_class()
    assert cls.__name__ in {"FastMCP", "MCPServer"}
    assert hasattr(cls, "tool") and hasattr(cls, "run")

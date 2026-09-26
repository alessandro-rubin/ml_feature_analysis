"""MCP server exposing the TESSA tools to any MCP client.

MCP is a protocol, not a vendor: Claude Desktop and Claude Code are two clients,
but so are Cline, Continue, Zed, LibreChat and anything else speaking the spec.
This matters practically — driving these tools from a Claude client uses that
client's own model, so it needs **no API key of your own**.

Every tool is registered from the same ``tessa.ai.tools`` functions the agent
drivers use, so there is exactly one implementation and no schema to keep in
sync.

Run it::

    tessa-mcp
    # or: python -m tessa.ai.mcp_server

Then point a client at it, e.g. in Claude Desktop's config::

    {"mcpServers": {"tessa": {"command": "tessa-mcp"}}}
"""

from __future__ import annotations

import sys

from tessa.ai.tools import TOOL_FUNCTIONS

__all__ = ["build_server", "main"]


def _server_class():
    """Return the FastMCP/MCPServer class across mcp 1.x and 2.x.

    mcp 2.x renamed ``FastMCP`` to ``MCPServer`` and moved the module. The
    surface used here — ``.tool()`` and ``.run()`` — is the same in both.
    """
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2

        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x

        return FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "Could not load an MCP server class from the installed `mcp` package "
            f"({exc}). Install or repair it with:  uv sync --extra ai"
        ) from exc


def build_server(name: str = "tessa"):
    """Create an MCP server with every tool registered.

    Raises
    ------
    RuntimeError
        If the ``mcp`` package is missing or its server API cannot be located,
        with the underlying reason preserved rather than masked.
    """
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The `mcp` package is required to run the MCP server. Install it with:  "
            "uv sync --extra ai"
        ) from exc

    server = _server_class()(name)
    for fn in TOOL_FUNCTIONS:
        # Applied functionally rather than as a decorator, so the very same
        # objects back the MCP surface and the agent drivers.
        server.tool()(fn)
    return server


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - a server loop
    try:
        server = build_server()
    except RuntimeError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

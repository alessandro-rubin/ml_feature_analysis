"""Optional AI layer: drive the TESSA pipeline with a language model.

The tool surface (:mod:`tessa.ai.tools`) is plain Python and imports no model
SDK, so ``import tessa.ai`` works with none of the optional dependencies
installed. The backends and the MCP server are reached by explicit submodule
import and are the only places that need ``openai``, ``anthropic`` or ``mcp``.
"""

from __future__ import annotations

from tessa.ai import catalog, expressions, render, session, tools

__all__ = ["catalog", "expressions", "render", "session", "tools"]

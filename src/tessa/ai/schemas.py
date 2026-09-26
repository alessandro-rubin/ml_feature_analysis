"""One schema source, two wire formats.

Tool schemas are derived from each function's signature, type hints and
Google-style docstring, then adapted to whichever provider is driving. The old
AI layer hand-wrote 374 lines of Anthropic schemas and the Groq variant
mechanically re-serialized them; the two drifted apart, which is the failure this
module exists to prevent.

Deriving here rather than through ``anthropic.beta_tool`` is deliberate: that
helper is Anthropic-only, and the primary driver for this project speaks the
OpenAI-compatible protocol.
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from typing import Any, Callable

from tessa.ai.tools import TOOL_FUNCTIONS

__all__ = [
    "tool_schema",
    "tool_schemas",
    "to_openai_tools",
    "to_anthropic_tools",
]

_ARGS_RE = re.compile(r"^\s*Args:\s*$")
_SECTION_RE = re.compile(r"^\s*(Returns|Raises|Yields|Notes|Examples):\s*$")
_ARG_RE = re.compile(r"^\s{4,}(\w+)\s*:\s*(.*)$")


def _summary(doc: str) -> str:
    """The docstring up to the Args: section — what the tool is for."""
    lines: list[str] = []
    for line in doc.splitlines():
        if _ARGS_RE.match(line) or _SECTION_RE.match(line):
            break
        lines.append(line.strip())
    return " ".join(x for x in lines if x).strip()


def _returns(doc: str) -> str:
    out, capturing = [], False
    for line in doc.splitlines():
        if _SECTION_RE.match(line):
            capturing = line.strip().startswith("Returns")
            continue
        if capturing:
            if _ARGS_RE.match(line):
                break
            out.append(line.strip())
    return " ".join(x for x in out if x).strip()


def _arg_docs(doc: str) -> dict[str, str]:
    """Parse the ``Args:`` block into {name: description}."""
    docs: dict[str, str] = {}
    in_args, current = False, None
    for line in doc.splitlines():
        if _ARGS_RE.match(line):
            in_args, current = True, None
            continue
        if not in_args:
            continue
        if _SECTION_RE.match(line):
            break
        match = _ARG_RE.match(line)
        if match:
            current = match.group(1)
            docs[current] = match.group(2).strip()
        elif current and line.strip():
            docs[current] = f"{docs[current]} {line.strip()}".strip()
    return docs


_SCALARS: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _unwrap_optional(annotation: Any) -> tuple[list[Any], bool]:
    """Split a union into its non-None members plus a nullable flag."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = list(typing.get_args(annotation))
        nullable = type(None) in args
        return [a for a in args if a is not type(None)], nullable
    return [annotation], False


def _json_schema_for(annotation: Any) -> dict[str, Any]:
    members, nullable = _unwrap_optional(annotation)
    if not members or annotation is Any or annotation is inspect.Parameter.empty:
        return {}

    def one(member: Any) -> dict[str, Any]:
        if member in _SCALARS:
            return {"type": _SCALARS[member]}
        origin = typing.get_origin(member)
        if origin in (list, tuple):
            args = typing.get_args(member)
            item = _json_schema_for(args[0]) if args else {}
            return {"type": "array", "items": item or {"type": "string"}}
        if origin is dict:
            return {"type": "object"}
        return {}

    schemas = [one(m) for m in members]
    concrete = [s for s in schemas if s]
    if len(concrete) == 1 and len(schemas) == 1:
        return concrete[0]
    # A genuine multi-type union (e.g. list[str] | str): leave the type open
    # rather than emit something a provider may reject, and rely on the
    # description to say what is accepted.
    if nullable and len(concrete) == 1:
        return concrete[0]
    return {}


def tool_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Derive a provider-neutral JSON Schema description of one tool."""
    doc = inspect.getdoc(fn) or ""
    arg_docs = _arg_docs(doc)
    sig = inspect.signature(fn)
    # `from __future__ import annotations` leaves signature annotations as
    # strings, so resolve them properly rather than reading sig.annotation.
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # pragma: no cover - unresolvable forward reference
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("args", "kwargs"):
            continue
        schema = _json_schema_for(hints.get(name, param.annotation))
        description = arg_docs.get(name, "")
        if param.default is not inspect.Parameter.empty and param.default is not None:
            description = (
                f"{description} Defaults to {param.default!r}.".strip()
                if description
                else f"Defaults to {param.default!r}."
            )
        if description:
            schema = {**schema, "description": description}
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    description = _summary(doc)
    returns = _returns(doc)
    if returns:
        description = f"{description} Returns: {returns}"

    return {
        "name": fn.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def tool_schemas(functions: list[Callable[..., Any]] | None = None) -> list[dict[str, Any]]:
    """Schemas for the whole tool surface, in a stable order.

    Stable order matters: on the Anthropic path the tool list is part of the
    cached prompt prefix, and reordering it would silently destroy the cache.
    """
    return [tool_schema(fn) for fn in (functions or TOOL_FUNCTIONS)]


def to_openai_tools(functions: list[Callable[..., Any]] | None = None) -> list[dict[str, Any]]:
    """Adapt to the OpenAI ``tools`` format (Groq, OpenRouter, OpenAI, local)."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in tool_schemas(functions)
    ]


def to_anthropic_tools(functions: list[Callable[..., Any]] | None = None) -> list[dict[str, Any]]:
    """Adapt to the Anthropic ``tools`` format."""
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["parameters"],
        }
        for s in tool_schemas(functions)
    ]

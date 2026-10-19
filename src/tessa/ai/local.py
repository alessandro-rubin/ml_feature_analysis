"""Local model servers: the checks that make a local run fail fast, not quietly.

A hosted API either answers or returns an error. A local server can be up and
still be wrong in ways that only surface as a bad run many minutes later:

* the model is not downloaded, or not the one being served;
* the model cannot call tools (Ollama reports it; LM Studio reports whether the
  model was trained for tool use);
* the model is not loaded yet, so LM Studio will load it with its default
  context length;
* the context window is smaller than the prompt. Ollama defaults to 4k tokens
  on GPUs with less than 24 GiB of VRAM, and the system prompt plus the tool
  definitions alone take about twice that. Ollama does not reject an oversized
  prompt: it drops the oldest messages without an error, which cannot make room
  when the fixed part alone does not fit;
* the model does not fit in VRAM and runs partly on the CPU, so every turn
  takes minutes;
* the "local" model is an Ollama cloud model, so the data leaves the machine.

:func:`probe` checks these through each server's own endpoints, as far as that
server exposes them, before the first turn. It reports the context window
(Ollama ``/api/ps``, llama.cpp ``/props``, vLLM ``/v1/models``, LM Studio
``/api/v1/models`` or ``/api/v0``) so that the loop can keep the conversation
inside it (``run_loop(context_tokens=...)``). It uses only the standard
library. On Ollama it loads the model, which the first turn would do anyway,
because only a loaded model reports the window it was given.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from tessa.ai.backend_openai import DEFAULT_MAX_TOKENS, LOCAL_TIMEOUT, PROVIDERS, Provider
from tessa.ai.loop import estimate_tokens

__all__ = [
    "RECOMMENDED_CONTEXT",
    "LocalServerError",
    "ProbeResult",
    "probe",
    "static_prompt_tokens",
]

# The window to aim for. Besides the system prompt, the tool definitions and a
# reply, it holds a dozen or more recent tool results before compaction starts.
RECOMMENDED_CONTEXT = 32_768
# The fewest tokens a run needs for the results it reads, once the system
# prompt, the tool definitions and a reply are paid for: two of the largest.
MIN_ROOM = 3_000


class LocalServerError(RuntimeError):
    """The local server cannot run the agent as configured. The message says how to fix it."""


@dataclass(frozen=True)
class _Hints:
    label: str
    start: str
    get_model: str
    set_context: str
    elsewhere: str = "or pass --base-url if it listens elsewhere."


_HINTS: dict[str, _Hints] = {
    "ollama": _Hints(
        label="Ollama",
        start="ollama serve",
        get_model="ollama pull {model}",
        set_context="restart the server with  OLLAMA_CONTEXT_LENGTH=32768 ollama serve  "
        "(in the desktop app: Settings, context length)",
    ),
    "lmstudio": _Hints(
        label="LM Studio",
        start="lms server start",
        get_model="lms get {model}  and then  lms load {model} --context-length 32768",
        set_context="reload the model with  lms load <model> --context-length 32768",
    ),
    "llamacpp": _Hints(
        label="llama.cpp",
        start="llama-server -m <model.gguf> --jinja -c 32768 -np 1",
        get_model="restart llama-server with  -m <model.gguf>  or  -hf <user>/<repo>",
        set_context="restart llama-server with  -c 32768 -np 1",
        elsewhere="or pass --base-url if it listens elsewhere: older builds use port "
        "8080 (--base-url http://localhost:8080/v1).",
    ),
    "vllm": _Hints(
        label="vLLM",
        start="vllm serve <model> --enable-auto-tool-choice --tool-call-parser <parser>",
        get_model="restart vllm serve with that model",
        set_context="restart vllm serve with  --max-model-len 32768",
    ),
}


@dataclass
class ProbeResult:
    """What the probe found out about the server."""

    server: str
    base_url: str
    model: str
    context_length: int | None
    context_source: str = ""
    warnings: list[str] = field(default_factory=list)


def static_prompt_tokens() -> int:
    """Estimated tokens taken by the system prompt and tool definitions on every turn."""
    from tessa.ai.prompts import system_prompt
    from tessa.ai.schemas import to_openai_tools

    return estimate_tokens(system_prompt() + json.dumps(to_openai_tools()))


def _request(url: str, *, body: dict | None = None, headers: dict[str, str], timeout: float) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    return json.loads(text) if text.strip() else None


def _optional(url: str, **kwargs: Any) -> Any:
    """Like ``_request``, but None when the server does not offer the endpoint."""
    try:
        return _request(url, **kwargs)
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 404, 405, 501):
            return None
        raise
    except ValueError:  # not JSON: some other service is answering
        return None


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode("utf-8"))
    except (OSError, ValueError):
        return str(exc)
    error = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(error, dict):
        error = error.get("message", error)
    return str(error)


def _resolve_model(wanted: str | None, served: list[str], spec: Provider, hints: _Hints) -> str:
    if wanted:
        if wanted in served:
            return wanted
        # Ollama lists "qwen3:latest" for a model pulled as "qwen3".
        if spec.server == "ollama" and ":" not in wanted and f"{wanted}:latest" in served:
            return f"{wanted}:latest"
        raise LocalServerError(
            f"{hints.label} does not serve {wanted!r}. Get it with:  "
            f"{hints.get_model.format(model=wanted)}\n"
            f"Models it serves: {', '.join(served) or 'none'}."
        )
    if len(served) == 1:
        return served[0]
    if not served:
        raise LocalServerError(
            f"{hints.label} serves no model. Get one with:  "
            f"{hints.get_model.format(model='<model>')}"
        )
    raise LocalServerError(
        f"{hints.label} serves several models; choose one with --model: {', '.join(served)}."
    )


def _ollama_context(
    root: str,
    model: str,
    *,
    headers: dict[str, str],
    timeout: float,
    load_timeout: float,
    warnings: list[str],
) -> int | None:
    request = {"headers": headers, "timeout": timeout}
    show = _optional(f"{root}/api/show", body={"model": model}, **request)
    if not isinstance(show, dict):
        warnings.append(
            f"no Ollama API at {root}. If this is another server, use its preset "
            "(--provider lmstudio, llamacpp or vllm)."
        )
        return None

    capabilities = show.get("capabilities")
    if isinstance(capabilities, list) and "tools" not in capabilities:
        raise LocalServerError(
            f"{model} cannot call tools in Ollama (its capabilities: "
            f"{', '.join(capabilities)}). Choose a model with the 'tools' tag in the "
            "Ollama library, such as qwen3:14b."
        )
    if show.get("remote_host"):
        warnings.append(
            f"{model} is an Ollama cloud model: prompts and tool results are sent to "
            f"{show['remote_host']}, not kept on this machine."
        )
        return None

    # Load the model so that /api/ps reports the window it was actually given;
    # the first turn would load it anyway.
    try:
        _request(
            f"{root}/api/generate",
            body={"model": model, "stream": False},
            headers=headers,
            timeout=load_timeout,
        )
    except urllib.error.HTTPError as exc:
        raise LocalServerError(f"Ollama could not load {model}: {_error_detail(exc)}") from exc
    except ValueError:
        pass  # loaded; the shape of the reply does not matter

    running = _optional(f"{root}/api/ps", **request) or {}
    entry = next(
        (m for m in running.get("models", []) if model in (m.get("name"), m.get("model"))),
        None,
    )
    if entry is None:
        return None
    size, in_vram = entry.get("size") or 0, entry.get("size_vram")
    if size and in_vram is not None and in_vram < size:
        on_cpu = round(100 * (size - in_vram) / size)
        warnings.append(
            f"{on_cpu}% of {model} runs on the CPU, so turns will be slow. A smaller "
            "model, or a smaller context window, keeps it on the GPU."
        )
    return entry.get("context_length") or None


def _llamacpp_context(root: str, model: str, **request: Any) -> int | None:
    props = _optional(f"{root}/props", **request)
    if not (isinstance(props, dict) and "default_generation_settings" in props):
        # Router mode answers per model.
        props = _optional(f"{root}/props?model={urllib.parse.quote(model, safe='')}", **request)
    if not isinstance(props, dict):
        return None
    return (props.get("default_generation_settings") or {}).get("n_ctx") or None


def _lmstudio_context(root: str, model: str, warnings: list[str], **request: Any) -> int | None:
    not_loaded = (
        f"{model} is not loaded yet, so LM Studio will load it on the first request with "
        f"its default context length. Load it first with:  lms load {model} "
        "--context-length 32768"
    )

    # The native REST API since LM Studio 0.4. `model` is the model's key, or the
    # identifier of one loaded instance (`lms load ... --identifier`).
    listing = _optional(f"{root}/api/v1/models", **request)
    if isinstance(listing, dict) and isinstance(listing.get("models"), list):
        for entry in listing["models"]:
            if not isinstance(entry, dict):
                continue
            instances = [i for i in entry.get("loaded_instances") or [] if isinstance(i, dict)]
            if entry.get("key") != model and not any(i.get("id") == model for i in instances):
                continue
            if (entry.get("capabilities") or {}).get("trained_for_tool_use") is False:
                warnings.append(
                    f"{model} was not trained for tool use, so LM Studio prompts it with a "
                    "generic call format; expect more malformed calls than from a model "
                    "with native tool use (the hammer badge in LM Studio)."
                )
            instance = next((i for i in instances if i.get("id") == model), None)
            instance = instance or (instances[0] if instances else None)
            if instance is None:
                warnings.append(not_loaded)
                return None
            return (instance.get("config") or {}).get("context_length") or None
        return None

    # LM Studio before 0.4: the v0 REST API.
    info = _optional(f"{root}/api/v0/models/{urllib.parse.quote(model, safe='')}", **request)
    if not isinstance(info, dict):
        return None
    if info.get("loaded_context_length"):
        return info["loaded_context_length"]
    if info.get("state") == "not-loaded":
        warnings.append(not_loaded)
    return None


def probe(
    provider: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 10.0,
    load_timeout: float = LOCAL_TIMEOUT,
) -> ProbeResult:
    """Check that a local server can run the agent, before spending a turn on it.

    Parameters
    ----------
    provider : str
        A local preset from ``PROVIDERS``: ``ollama``, ``lmstudio``,
        ``llamacpp``, ``vllm`` (or ``local``, an alias of ``ollama``).
    base_url : str, optional
        Overrides the preset's URL, e.g. for a server on another machine.
    model : str, optional
        The model to run. Without one, the preset's default, or the only model
        the server serves.
    api_key : str, optional
        For a server started with a key. Defaults to the preset's environment
        variable (``LM_API_TOKEN``, ``LLAMA_API_KEY``, ``VLLM_API_KEY``).
    timeout : float
        Seconds to wait for each metadata request.
    load_timeout : float
        Seconds to wait for Ollama to load the model.

    Returns
    -------
    ProbeResult
        The resolved model, the context window when the server reports it, and
        warnings worth showing before the run.

    Raises
    ------
    LocalServerError
        When the run could not work: no server, a missing model, a model that
        cannot call tools (Ollama), or a context window that leaves too little
        room for tool results once the system prompt, the tool definitions and
        a reply are paid for. The message names the command that fixes it.
    """
    spec = PROVIDERS.get(provider)
    if spec is None or not spec.local:
        local = sorted(name for name, p in PROVIDERS.items() if p.local)
        raise ValueError(f"{provider!r} is not a local provider; choose one of {local}.")
    hints = _HINTS[spec.server]
    base = (base_url or spec.base_url or "").rstrip("/")
    root = base.removesuffix("/v1")
    key = api_key or (os.environ.get(spec.key_env) if spec.key_env else None)
    request = {"headers": {"Authorization": f"Bearer {key}"} if key else {}, "timeout": timeout}

    try:
        listing = _request(f"{base}/models", **request)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            remedy = f"Set {spec.key_env} to its key." if spec.key_env else "Disable its API key."
            raise LocalServerError(
                f"{hints.label} at {base} rejected the request. {remedy}"
            ) from exc
        hint = " (the base URL usually ends in /v1)" if exc.code == 404 else ""
        raise LocalServerError(
            f"{hints.label} at {base} answered {exc.code} to GET /models{hint}: "
            f"{_error_detail(exc)}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise LocalServerError(
            f"No {hints.label} server is answering at {base} ({reason}). Start it with:\n"
            f"  {hints.start}\n"
            f"{hints.elsewhere}"
        ) from exc
    except ValueError as exc:
        raise LocalServerError(f"{base}/models did not return JSON; is it {hints.label}?") from exc

    data = listing.get("data") if isinstance(listing, dict) else None
    entries = [e for e in data or [] if isinstance(e, dict) and e.get("id")]
    resolved = _resolve_model(model or spec.default_model, [e["id"] for e in entries], spec, hints)

    warnings: list[str] = []
    context: int | None = None
    try:
        if spec.server == "ollama":
            context = _ollama_context(
                root,
                resolved,
                headers=request["headers"],
                timeout=timeout,
                load_timeout=load_timeout,
                warnings=warnings,
            )
        elif spec.server == "llamacpp":
            context = _llamacpp_context(root, resolved, **request)
        elif spec.server == "vllm":
            entry = next((e for e in entries if e["id"] == resolved), {})
            context = entry.get("max_model_len") or None
        elif spec.server == "lmstudio":
            context = _lmstudio_context(root, resolved, warnings, **request)
    except urllib.error.HTTPError as exc:
        raise LocalServerError(
            f"{hints.label} answered {exc.code} while checking {resolved}: {_error_detail(exc)}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise LocalServerError(
            f"{hints.label} stopped answering while checking {resolved} ({reason})."
        ) from exc

    static = static_prompt_tokens()
    if context is None:
        warnings.append(
            f"{hints.label} did not report the context window, so the history cannot be "
            f"kept inside it. Pass --context-length with the value the model was loaded "
            f"with; the system prompt and tool definitions alone take about {static} tokens."
        )
        return ProbeResult(spec.server, base, resolved, None, warnings=warnings)

    # The reply reserve the CLI will use: a quarter of the window at most.
    reply = min(DEFAULT_MAX_TOKENS, context // 4)
    room = context - static - reply
    if room < MIN_ROOM:
        raise LocalServerError(
            f"{hints.label} runs {resolved} with a {context}-token context window. The "
            f"system prompt and tool definitions take about {static} tokens and a reply "
            f"{reply}, which leaves {max(room, 0)} for the tool results a run reads; it "
            f"needs at least {MIN_ROOM}. To fix it, {hints.set_context}."
        )
    if context < RECOMMENDED_CONTEXT:
        warnings.append(
            f"{resolved} has a {context}-token context window; {RECOMMENDED_CONTEXT} or "
            "more is recommended. Older tool results will be elided to fit. To raise it, "
            f"{hints.set_context}."
        )

    return ProbeResult(
        server=spec.server,
        base_url=base,
        model=resolved,
        context_length=context,
        context_source=f"reported by {hints.label}",
        warnings=warnings,
    )

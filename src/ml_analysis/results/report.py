"""Self-contained static HTML report for a run.

`render_html(results, manifest)` turns `{analysis: AnalysisResult}` into a
single HTML string: manifest header, then one section per analysis with
its scalars, tables (capped rows), and a histogram for every 1-D numeric
array (matplotlib, embedded as base64 PNG — no external assets, the file
can be e-mailed as-is).
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any

import polars as pl

from ml_analysis.results.result import AnalysisResult

_MAX_TABLE_ROWS = 50

_STYLE = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px;
       color: #1a1a1a; }
h1 { border-bottom: 2px solid #444; padding-bottom: .3rem; }
h2 { margin-top: 2.2rem; border-bottom: 1px solid #bbb; padding-bottom: .2rem; }
table { border-collapse: collapse; font-size: .85rem; margin: .6rem 0; }
th, td { border: 1px solid #ccc; padding: .25rem .55rem; text-align: right; }
th { background: #f0f0f0; }
.scalars td:first-child { font-weight: 600; text-align: left; }
.note { color: #777; font-size: .8rem; }
img { max-width: 760px; display: block; margin: .5rem 0; }
"""


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _figures_html(res: AnalysisResult) -> str:
    """Curated charts for an analysis, embedded as standalone PNGs."""
    import matplotlib

    matplotlib.use("Agg")
    from ml_analysis.results.figures import figures_for_result

    parts = []
    for title, fig in figures_for_result(res):
        parts.append(
            f'<img alt="{html.escape(title)}" '
            f'src="data:image/png;base64,{_fig_to_b64(fig)}"/>'
        )
    return "".join(parts)


def _frame_html(name: str, frame: Any) -> str:
    pdf = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame
    note = ""
    if len(pdf) > _MAX_TABLE_ROWS:
        note = f'<p class="note">showing {_MAX_TABLE_ROWS} of {len(pdf)} rows</p>'
        pdf = pdf.head(_MAX_TABLE_ROWS)
    return f"<h3>{html.escape(name)}</h3>{note}{pdf.to_html(border=0)}"


def _scalars_html(scalars: dict) -> str:
    if not scalars:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in scalars.items()
    )
    return f'<table class="scalars">{rows}</table>'


def render_html(
    results: dict[str, AnalysisResult | Any],
    manifest: dict | None = None,
    title: str = "ml_analysis run report",
) -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    if manifest:
        meta = {
            k: manifest[k]
            for k in ("created_at", "target_col", "data_fingerprint", "versions")
            if k in manifest
        }
        parts.append("<h2>Run manifest</h2>")
        parts.append(_scalars_html({k: v for k, v in meta.items()}))

    for name, raw in results.items():
        res = AnalysisResult.from_raw(name, raw)
        parts.append(f"<h2>{html.escape(name)}</h2>")
        parts.append(_scalars_html(res.scalars))
        parts.append(_figures_html(res))
        for fname, frame in res.frames.items():
            parts.append(_frame_html(fname, frame))
        if res.objects:
            parts.append(
                f'<p class="note">not rendered (live objects): '
                f'{html.escape(", ".join(res.objects))}</p>'
            )

    parts.append("</body></html>")
    return "".join(parts)


def write_report(
    path: str | Path,
    results: dict[str, Any],
    manifest: dict | None = None,
    title: str = "ml_analysis run report",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(results, manifest, title), encoding="utf-8")
    return path

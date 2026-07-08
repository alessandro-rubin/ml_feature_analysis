"""Output helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_fig(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p


def output_path(output_dir: str | Path | None, name: str) -> Path:
    base = Path(output_dir) if output_dir else Path("outputs")
    base.mkdir(parents=True, exist_ok=True)
    return base / name

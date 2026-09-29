"""``tessa-dashboard``: launch the run browser from any installed environment.

``streamlit run`` needs a script path, which only exists predictably inside a
clone of the repository. This entry point resolves the installed ``app.py``
and hands it to streamlit, so after ``pip``/``uv`` installing ``tessa`` with
the ``dashboard`` extra, ``tessa-dashboard outputs/runs`` works anywhere.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tessa-dashboard",
        description="Browse runs saved with Run.save() in a local Streamlit app.",
        epilog="Arguments after `--` are passed to `streamlit run` (e.g. -- --server.port 8502).",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="outputs/runs",
        help="ResultStore directory (default: %(default)s)",
    )
    args, streamlit_args = parser.parse_known_args(argv)
    streamlit_args = [a for a in streamlit_args if a != "--"]

    if importlib.util.find_spec("streamlit") is None:
        print(
            "tessa-dashboard needs streamlit: install the `dashboard` extra, "
            "e.g. `uv add 'tessa[dashboard]'` or `pip install 'tessa[dashboard]'`",
            file=sys.stderr,
        )
        return 1

    app = Path(__file__).with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *streamlit_args]
    cmd += ["--", "--root", args.root]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

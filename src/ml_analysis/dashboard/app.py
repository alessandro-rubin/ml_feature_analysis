"""Streamlit app: browse saved analysis runs (manifest, tables, scores).

The dashboard is a thin reader over `ResultStore` run directories — it
never recomputes an analysis. Launch:

    streamlit run src/ml_analysis/dashboard/app.py -- --root outputs/runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

try:
    import streamlit as st
except ImportError as err:  # pragma: no cover
    raise SystemExit(
        "streamlit is required for the dashboard: "
        "pip install 'ml-analysis[dashboard]'"
    ) from err

from ml_analysis.results import ResultStore


def _cli_root() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/runs")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.root


def main() -> None:  # pragma: no cover — manual/UI entry point
    st.set_page_config(page_title="ml_analysis runs", layout="wide")
    st.title("ml_analysis — run browser")

    root = st.sidebar.text_input("Results root", _cli_root())
    if not Path(root).exists():
        st.info(f"No results directory at `{root}` yet. Save a run first: "
                "`Run(...).save(path)`.")
        return
    store = ResultStore(root)
    runs = store.runs()
    if not runs:
        st.info("No runs found in this store.")
        return

    run_name = st.sidebar.selectbox("Run", runs, index=len(runs) - 1)
    manifest = store.load_manifest(run_name)
    results = store.load_run(run_name)

    with st.expander("Run manifest", expanded=False):
        st.json(manifest)

    names = list(results)
    tabs = st.tabs(names)
    for tab, name in zip(tabs, names):
        res = results[name]
        with tab:
            if res.scalars:
                st.write({k: v for k, v in res.scalars.items()})
            for fname, frame in res.frames.items():
                st.subheader(fname)
                pdf = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame
                st.dataframe(pdf, width="stretch")
            for aname, arr in res.arrays.items():
                if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                    st.subheader(aname)
                    counts, edges = np.histogram(arr[np.isfinite(arr)], bins=40)
                    st.bar_chart(
                        pl.DataFrame({"bin": edges[:-1], "count": counts}),
                        x="bin", y="count",
                    )


if __name__ == "__main__":  # streamlit executes scripts as __main__
    main()

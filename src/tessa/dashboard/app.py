"""Streamlit app: browse saved analysis runs (KPIs, charts, tables).

The dashboard is a thin reader over `ResultStore` run directories — it
never recomputes an analysis. All chart logic lives UI-independently in
`tessa.results.figures` (shared with the HTML report and
`Run.figures()`); this module only arranges the matplotlib figures it
returns. Launch:

    streamlit run src/tessa/dashboard/app.py -- --root outputs/runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

try:
    import streamlit as st
except ImportError as err:  # pragma: no cover
    raise SystemExit(
        "streamlit is required for the dashboard: pip install 'ml-analysis[dashboard]'"
    ) from err

from tessa.results import ResultStore
from tessa.results.figures import figures_for_result, headline_metrics


def _cli_root() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/runs")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.root


def main() -> None:  # pragma: no cover — manual/UI entry point
    st.set_page_config(page_title="tessa runs", layout="wide")
    st.title("tessa — run browser")

    root = st.sidebar.text_input("Results root", _cli_root())
    if not Path(root).exists():
        st.info(f"No results directory at `{root}` yet. Save a run first: `Run(...).save(path)`.")
        return
    store = ResultStore(root)
    runs = store.runs()
    if not runs:
        st.info("No runs found in this store.")
        return

    run_name = st.sidebar.selectbox("Run", runs, index=len(runs) - 1)
    manifest = store.load_manifest(run_name)
    results = store.load_run(run_name)

    _render_overview(results)

    with st.expander("Run manifest", expanded=False):
        st.json(manifest)

    names = list(results)
    tabs = st.tabs(names)
    for tab, name in zip(tabs, names):
        with tab:
            _render_analysis(results[name])


def _render_overview(results) -> None:
    metrics = headline_metrics(results)
    if not metrics:
        return
    st.subheader("Overview")
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        col.metric(m["label"], m["value"], help=m.get("help"))


def _render_analysis(res) -> None:
    if res.scalars:
        st.write({k: v for k, v in res.scalars.items()})

    figs = figures_for_result(res)
    if figs:
        st.markdown("#### Charts")
        for left, right in zip(figs[::2], figs[1::2] + [None]):
            cols = st.columns(2)
            for col, item in zip(cols, (left, right)):
                if item is None:
                    continue
                _, fig = item
                col.pyplot(fig)
                plt.close(fig)

    if res.frames:
        st.markdown("#### Tables")
        for fname, frame in res.frames.items():
            with st.expander(fname, expanded=False):
                pdf = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame
                st.dataframe(pdf, width="stretch")


if __name__ == "__main__":  # streamlit executes scripts as __main__
    main()

# TODO: analysis title

TODO: the question, the data it uses (where it lives, which export), and the
conclusion once there is one.

Built on [tessa](https://github.com/alessandro-rubin/ml_feature_analysis).

## The example it ships with

`notebooks/01_anomaly_vs_normal.ipynb` runs end to end on real data: Wind
Farm A of the [CARE to Compare](https://doi.org/10.3390/data9120138) benchmark
(5 wind turbines, 10-minute SCADA data, 22 labelled events). It asks whether
the SCADA signals tell the run-up to a logged fault apart from normal
operation, and walks through the checks that keep the answer honest: a column
that leaks the label, a seasonal confounder, and a corrected representation.

The first run downloads about 160 MB from Zenodo (a minute or two) into
`example_data/`; later runs reuse it. The data is licensed CC BY-SA 4.0 by its
authors (Gück, Roelofs & Faulstich, *Data* 9(12):138, 2024): cite the paper if
you publish results obtained from it. Offline, download `CARE_To_Compare.zip`
from <https://zenodo.org/records/15846963> (the version tessa reads) and pass
its path as `fetch_wind_farm_a(..., archive=...)`.

To point the notebook at your own data, fill in the `TODO`s in its settings
cell and set `USE_EXAMPLE_DATA = False`, then rename it after your question.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and read access to the tessa
repository (SSH key or HTTPS credentials; see `pyproject.toml`).

```bash
uv sync                       # creates .venv with tessa + Jupyter, from uv.lock
uv run jupyter lab            # or open a notebook in VS Code and pick .venv
```

Commit `uv.lock`: it pins every package version, so anyone re-running the
notebooks later gets the same environment.

## Layout

```
notebooks/     analysis notebooks, numbered in reading order
outputs/       saved runs and HTML reports (git-ignored)
example_data/  downloaded by the notebook when USE_EXAMPLE_DATA=True (git-ignored)
```

Raw data and label sheets stay where they are (a shared drive, a data folder
outside the repository); the settings cell at the top of each notebook points
at them.

## Large stores

tessa reads events in batches, so memory follows the batch size rather than
the number of events: hundreds of assets do not need to fit in memory at
once. Each notebook's settings cell has two knobs, passed to `Config`:

- `EVENTS_PER_BATCH`: events read and reduced at the same time (default: one
  per CPU thread);
- `MEMORY_BUDGET`: cap on a batch's estimated working set (default `"auto"`,
  half the machine's memory). Events with long windows, high sampling rates
  or many derived features then run fewer at a time; an event above the
  budget on its own runs alone, with a warning.

If memory still runs out, lower `MEMORY_BUDGET` (or `EVENTS_PER_BATCH`), and
pass `columns=` to `ds.events()` so only the signals you analyse are read.

## Updating tessa

Change the tag in `pyproject.toml` (e.g. `@v0.3.0` to `@v0.4.0`), then
`uv lock --upgrade-package tessa && uv sync`. Re-run the notebooks and check
the results before committing: a new version can change them.

## Browsing results

```bash
uv run tessa-dashboard outputs/runs
```

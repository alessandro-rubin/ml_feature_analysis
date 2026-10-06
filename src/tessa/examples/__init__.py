"""Example data stores for tutorials, the analysis template, and tests.

- :func:`fetch_wind_farm_a` — real SCADA data: Wind Farm A of the CARE to
  Compare benchmark (5 turbines, 10-minute data, labelled fault run-ups and
  normal periods), downloaded from Zenodo once and converted to the layout
  tessa expects. Used by the analysis template.
- :func:`make_example_dataset` — a small synthetic store generated locally,
  with no network access. Used by the tests and ``demo.py``.

Both return an :class:`ExampleDataset` pointing at a data root and an Excel
label sheet, so a notebook can switch between them by changing one line.
"""

from tessa.examples._base import ExampleDataset
from tessa.examples.synthetic import COLUMN_MAP, make_example_dataset
from tessa.examples.wind_turbines import WIND_FARM_A_COLUMN_MAP, fetch_wind_farm_a

__all__ = [
    "COLUMN_MAP",
    "ExampleDataset",
    "WIND_FARM_A_COLUMN_MAP",
    "fetch_wind_farm_a",
    "make_example_dataset",
]

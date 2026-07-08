# TESSA — Time-series Event Slicing and Source Alignment

Lazy, polars-first loading of per-asset time-series that are split across
parquet files along two axes:

- **time** — files sharing a filename prefix (the *source* / dataname) hold
  the same columns for consecutive periods and are vertically concatenated;
- **variables** — files with different prefixes hold different columns over
  the same timeline, possibly at different sampling rates, and are aligned
  with a full outer join on the timestamp column (missing stamps become
  nulls).

## On-disk convention

```
<data_root>/
  <asset_id>/
    <dataname>_<start>_<end>.parquet
```

Dates are `YYMMDD` or `YYYYMMDD`; the end date is inclusive of its whole
day. `<dataname>` may contain underscores (e.g.
`flow_rate_240101_240131.parquet`). The convention is configurable via
`LoaderConfig.filename_pattern` (named groups `source`, `start`, `end`).

## Usage

```python
from datetime import datetime
from tessa import LoaderConfig, load_asset, load_event

# One line: entire history of an asset, all sources joined on timestamp.
df = load_asset("A1", "path/to/data")

# Windowed / column-subset / lazy variants.
df = load_asset("A1", "path/to/data", start=datetime(2024, 1, 1), columns=["x"])
lf = load_asset("A1", "path/to/data", lazy=True)

# Full control via LoaderConfig; load_event returns a LazyFrame.
cfg = LoaderConfig(data_root="path/to/data", timestamp_col="ts")
lf = load_event("A1", datetime(2024, 1, 10), datetime(2024, 1, 12), cfg)
```

Constraints: column names (other than the timestamp) must be unique across
sources, and timestamps should be unique within each source.

## Development

Part of the `ml_feature_analysis` uv workspace. From the repo root:

```bash
uv run --package tessa pytest packages/tessa/tests
```

Only runtime dependency: `polars`.

# asset_loader — multi-source parquet time-series loader

Lazy, polars-first loading of per-asset time-series that are split across
parquet files along two axes:

- **time** — files sharing a filename prefix (the *source* / dataname) hold
  the same columns for consecutive periods and are vertically concatenated;
- **variables** — files with different prefixes hold columns over the same
  timeline, possibly at different sampling rates, and are combined by a
  configurable `merge` strategy (full outer join on the timestamp by
  default, so missing stamps become nulls).

The same column name may appear in several sources; `on_duplicate` decides
whether that is an error, or how the copies are combined. Pass
`with_metadata=True` to get a provenance dict back alongside the frame.

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
from asset_loader import LoaderConfig, load_asset, load_event

# One line: entire history of an asset, all sources joined on timestamp.
df = load_asset("A1", "path/to/data")

# Windowed / column-subset / lazy variants.
df = load_asset("A1", "path/to/data", start=datetime(2024, 1, 1), columns=["x"])
lf = load_asset("A1", "path/to/data", lazy=True)

# Full control via LoaderConfig; load_event returns a LazyFrame.
cfg = LoaderConfig(data_root="path/to/data", timestamp_col="ts")
lf = load_event("A1", datetime(2024, 1, 10), datetime(2024, 1, 12), cfg)
```

### Combining sources — `merge`

How sources covering the same period are put together:

| `merge`      | result                                                              |
| ------------ | ------------------------------------------------------------------- |
| `"outer"`    | *(default)* full outer join on the timestamp — union of all stamps, nulls where a source has no row |
| `"left"`     | keep the anchor source's stamps only; others match exactly           |
| `"inner"`    | keep only stamps present in **every** source                         |
| `"vertical"` | no join — stack the sources' rows (`diagonal_relaxed` concat)        |
| `"asof"`     | `join_asof` each source onto the anchor's grid (`asof_strategy`, `asof_tolerance`) |

The *anchor* of a `left`/`asof` merge is the first source in
`source_order`; sources it doesn't name follow alphabetically.

```python
# A slow source carried forward onto the fast source's grid, no null gaps.
df = load_asset(
    "A1", "path/to/data",
    merge="asof", source_order=["sensor"], asof_tolerance="2h",
)
```

### Repeated column names — `on_duplicate`

When a non-timestamp column appears in more than one source:

| `on_duplicate` | result                                                          |
| -------------- | --------------------------------------------------------------- |
| `"error"`      | *(default)* raise `ValueError`                                    |
| `"rename"`     | keep every copy, suffixed with its source (`x__sensor`, `x__backup`) |
| `"coalesce"`   | one column `x`, first non-null across sources in `source_order` priority |
| `"first"`      | keep the highest-priority source's copy, drop the rest            |
| `"last"`       | keep the lowest-priority source's copy, drop the rest             |

`columns=[...]` always names columns as they appear *on disk*, i.e. before
any renaming, so `columns=["x"]` selects every copy of `x`.

### Provenance — `with_metadata`

```python
df, meta = load_asset("A1", "path/to/data", with_metadata=True)

meta["sources"]["sensor"]["files"]   # exactly which parquet files were read
meta["duplicates"]["coalesced"]      # {"x": ["x__backup", "x__sensor"]}
meta["operations"]                   # ordered log: discover, scan, concat,
                                     # rename/drop, merge, coalesce, filter,
                                     # sort, collect
```

`LoadResult` is a plain `NamedTuple`, so it unpacks as above or is used as
`result.frame` / `result.metadata`. Alongside `operations`, the dict
carries `asset_id`, `requested`, `config`, `source_order`, `sources`,
`merge`, `duplicates`, and the result's `columns` / `schema`.

Constraint: timestamps should be unique within each source — duplicated
stamps fan out through a join.

## Development

Part of the `ml_feature_analysis` uv workspace. From the repo root:

```bash
uv run --package asset_loader pytest packages/asset_loader/tests
```

Only runtime dependency: `polars`.

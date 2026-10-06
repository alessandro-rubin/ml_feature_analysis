"""Real wind-turbine SCADA data: Wind Farm A of the CARE to Compare benchmark.

CARE to Compare (Gück, Roelofs & Faulstich, 2024) is a public benchmark for
early fault detection in wind turbines. Its Wind Farm A comes from the EDP
open-data platform: 5 onshore turbines in Portugal, 10-minute SCADA averages
(plus 10-minute min / max / std for some sensors) of 54 sensors, and 22
labelled events: 12 periods leading up to a logged fault (hydraulic group,
gearbox, generator bearing, transformer) and 10 periods of normal behaviour.
The publishers anonymised the timestamps and scaled the power readings.

:func:`fetch_wind_farm_a` downloads only Wind Farm A (about 160 MB) out of the
5.5 GB archive on Zenodo, using HTTP range requests, and writes:

- ``<root>/data/WT<nn>/scada_<YYYYMMDD>_<YYYYMMDD>.parquet``: one folder per
  turbine, one file per month, in the layout :mod:`asset_loader` expects;
- ``<root>/labels.xlsx``: the event table as a label sheet ("Turbine",
  "Event start", "Event end", "Label", "Fault", "CARE event"), with
  :data:`WIND_FARM_A_COLUMN_MAP` translating its headers;
- ``<root>/signals.csv``: what each column measures, from the dataset's
  feature description.

The download is cached under ``<root>/raw/`` (one parquet file per CARE
dataset), so an interrupted download resumes and later calls are instant.

What the conversion changes, and why
------------------------------------
- **One timeline per turbine.** The benchmark ships one CSV per *event*, each
  holding about a year of the turbine's history before the event, so the CSVs
  of one turbine overlap by months. They are merged and de-duplicated on the
  timestamp. The conversion checks that every duplicated timestamp carries
  identical values in every column, and raises otherwise.
- **Two one-day offsets.** Datasets 71 (turbine 0) and 40 (turbine 10) are
  stamped exactly one day later than every other dataset of the same turbine,
  a side effect of the timestamp anonymisation: shifted back by one day, their
  overlaps with the other datasets match row for row. Their data and their
  event windows are moved back by one day.
- **Readable column names.** ``sensor_12_avg`` becomes ``gearbox_oil_temp_avg``:
  the anonymised sensor name is replaced by a short name taken from the
  feature description, and the statistic suffix is kept. ``signals.csv`` maps
  every column back to its original name.
- **Bookkeeping columns dropped.** ``asset_id`` (it is the folder name), and
  ``id`` and ``train_test`` (they index the per-event files, not the turbine).

``status_type_id`` is kept as published, but it is **not a sensor reading** in
Wind Farm A: it was derived from the same fault logbook as the labels. The
anomaly windows are almost entirely status 4 ("downtime") while the turbine
keeps producing power, so using it as a feature would leak the label.

License and citation
--------------------
The data is licensed CC BY-SA 4.0 by its authors. tessa does not redistribute
it: it is downloaded from Zenodo when this function runs. Cite the paper when
you publish results obtained from it:

    C. Gück, C. M. A. Roelofs, S. Faulstich. "CARE to Compare: A Real-World
    Benchmark Dataset for Early Fault Detection in Wind Turbine Data".
    Data 9(12):138, 2024. https://doi.org/10.3390/data9120138

Dataset: https://doi.org/10.5281/zenodo.10958774 (all versions); this module
fetches the version published as Zenodo record 15846963.
"""

from __future__ import annotations

import dataclasses
import io
import json
import re
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from email.message import Message
from pathlib import Path

import polars as pl

from tessa.examples._base import ExampleDataset

__all__ = ["WIND_FARM_A_COLUMN_MAP", "fetch_wind_farm_a"]

ZENODO_RECORD = "15846963"
ARCHIVE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/CARE_To_Compare.zip/content"
_FARM_DIR = "CARE_To_Compare/Wind Farm A/"

WIND_FARM_A_COLUMN_MAP = {
    "Turbine": "asset_id",
    "Event start": "start",
    "Event end": "end",
    "Label": "class",
    "Fault": "fault",
    "CARE event": "care_event",
}
"""Label-sheet header -> tessa column name, for :class:`tessa.labels.ExcelLabelSource`."""

# CARE event id -> correction to its timestamps (see the module docstring).
_TIME_SHIFTS = {71: timedelta(days=-1), 40: timedelta(days=-1)}

# Anonymised sensor name -> short readable name, from feature_description.csv.
_SENSOR_NAMES = {
    "sensor_0": "ambient_temp",
    "sensor_1": "wind_direction",
    "sensor_2": "wind_direction_rel",
    "wind_speed_3": "wind_speed",
    "wind_speed_4": "wind_speed_est",
    "sensor_5": "pitch_angle",
    "sensor_6": "hub_controller_temp",
    "sensor_7": "nacelle_controller_temp",
    "sensor_8": "choke_coil_temp",
    "sensor_9": "vcp_board_temp",
    "sensor_10": "vcs_cooling_water_temp",
    "sensor_11": "gearbox_bearing_temp",
    "sensor_12": "gearbox_oil_temp",
    "sensor_13": "gen_bearing_de_temp",
    "sensor_14": "gen_bearing_nde_temp",
    "sensor_15": "stator_temp_1",
    "sensor_16": "stator_temp_2",
    "sensor_17": "stator_temp_3",
    "sensor_18": "generator_rpm",
    "sensor_19": "split_ring_temp",
    "sensor_20": "busbar_temp",
    "sensor_21": "igbt_grid_side_temp",
    "sensor_22": "phase_displacement",
    "sensor_23": "current_1",
    "sensor_24": "current_2",
    "sensor_25": "current_3",
    "sensor_26": "grid_frequency",
    "reactive_power_27": "reactive_power_cap_possible",
    "reactive_power_28": "reactive_power_ind_possible",
    "power_29": "active_power_possible",
    "power_30": "grid_power",
    "sensor_31": "grid_reactive_power",
    "sensor_32": "voltage_1",
    "sensor_33": "voltage_2",
    "sensor_34": "voltage_3",
    "sensor_35": "igbt_rotor_side_temp_1",
    "sensor_36": "igbt_rotor_side_temp_2",
    "sensor_37": "igbt_rotor_side_temp_3",
    "sensor_38": "transformer_temp_1",
    "sensor_39": "transformer_temp_2",
    "sensor_40": "transformer_temp_3",
    "sensor_41": "hydraulic_oil_temp",
    "sensor_42": "nacelle_direction",
    "sensor_43": "nacelle_temp",
    "sensor_44": "energy_gen_disconnected",
    "sensor_45": "energy_gen_delta",
    "sensor_46": "energy_gen_star",
    "sensor_47": "reactive_energy_gen_disconnected",
    "sensor_48": "reactive_energy_gen_delta",
    "sensor_49": "reactive_energy_gen_star",
    "sensor_50": "energy_total",
    "sensor_51": "reactive_energy_total",
    "sensor_52": "rotor_rpm",
    "sensor_53": "nose_cone_temp",
}
_STATS = ("avg", "min", "max", "std")
_DROPPED = ("asset_id", "id", "train_test")
_MARKER = "wind_farm_a.json"


# ── Remote archive access ────────────────────────────────────────────────────


class _HTTPRangeFile(io.RawIOBase):
    """Seekable, read-only view of a remote file, read with HTTP range requests.

    Lets :class:`zipfile.ZipFile` read the archive's directory and single
    members without downloading the whole archive.
    """

    def __init__(self, url: str, timeout: float = 60.0, retries: int = 4) -> None:
        self.url, self.timeout, self.retries = url, timeout, retries
        self.pos = 0
        status, headers, _ = self._get(0, 0)
        content_range = headers.get("Content-Range", "")
        if status != 206 or "/" not in content_range:
            raise OSError(f"{url} does not support HTTP range requests")
        self.size = int(content_range.rsplit("/", 1)[1])

    def _get(self, first: int, last: int) -> tuple[int, Message, bytes]:
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={first}-{last}", "User-Agent": "tessa"}
        )
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    # `headers` looks names up case-insensitively; a dict would not.
                    return response.status, response.headers, response.read()
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == self.retries - 1:
                    raise ConnectionError(f"Download from {self.url} failed: {exc}") from exc
                wait = 2**attempt
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                    wait = int(exc.headers.get("Retry-After", 60))  # Zenodo rate limit
                time.sleep(wait)
        raise AssertionError("unreachable")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        _, _, data = self._get(self.pos, self.pos + n - 1)
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _open_archive(archive: str | Path | None) -> zipfile.ZipFile:
    """The CARE archive: a local copy, or the Zenodo file read remotely."""
    if archive is not None:
        return zipfile.ZipFile(Path(archive).expanduser())
    # Large reads: zipfile asks for a member in small pieces, one request each.
    return zipfile.ZipFile(io.BufferedReader(_HTTPRangeFile(ARCHIVE_URL), buffer_size=8 << 20))


def _read_text(data: bytes) -> str:
    """CARE's CSVs are UTF-8 except where they are not (the units use Latin-1)."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _cached_ids(raw: Path) -> list[int] | None:
    """The CARE event ids if the cache is complete, else ``None``."""
    tables = [raw / "event_info.csv", raw / "feature_description.csv"]
    if not all(t.exists() for t in tables):
        return None
    ids = sorted(pl.read_csv(tables[0], separator=";")["event_id"].to_list())
    return ids if all((raw / f"{i}.parquet").exists() for i in ids) else None


def _download(raw: Path, archive: str | Path | None, workers: int, progress: bool) -> list[int]:
    """Cache Wind Farm A under ``raw``: the two tables plus one parquet per dataset.

    Returns the CARE event ids present. Only files not already cached are read,
    and the archive is not opened at all when the cache is complete.
    """
    cached = _cached_ids(raw)
    if cached is not None:
        return cached
    raw.mkdir(parents=True, exist_ok=True)
    with _open_archive(archive) as zf:
        members = [i for i in zf.infolist() if i.filename.startswith(_FARM_DIR)]
        if not members:
            raise ValueError(f"No '{_FARM_DIR}' folder in the CARE archive")
        for name in ("event_info.csv", "feature_description.csv"):
            if not (raw / name).exists():
                # Bytes, not write_text: no newline translation on Windows.
                (raw / name).write_bytes(_read_text(zf.read(_FARM_DIR + name)).encode())
    datasets = {
        int(Path(i.filename).stem): i
        for i in members
        if i.filename.startswith(_FARM_DIR + "datasets/") and i.filename.endswith(".csv")
    }
    todo = [i for i in sorted(datasets) if not (raw / f"{i}.parquet").exists()]
    if todo and progress:
        mb = sum(datasets[i].compress_size for i in todo) / 1e6
        print(f"Wind Farm A: downloading {len(todo)} datasets ({mb:.0f} MB)...", flush=True)

    def fetch(ids: list[int]) -> None:
        # One archive handle per thread: a ZipFile serialises reads on its file.
        with _open_archive(archive) as zf:
            for i in ids:
                data = zf.read(datasets[i].filename)
                frame = pl.read_csv(data, separator=";", infer_schema_length=None)
                tmp = raw / f"{i}.parquet.part"
                frame.write_parquet(tmp)
                tmp.replace(raw / f"{i}.parquet")  # never leave a truncated cache file
                if progress:
                    print(f"  dataset {i} ({frame.height} rows)", flush=True)

    if todo:
        n = max(1, min(workers, len(todo)))
        with ThreadPoolExecutor(n) as pool:
            list(pool.map(fetch, [todo[k::n] for k in range(n)]))
    return sorted(datasets)


# ── Conversion ───────────────────────────────────────────────────────────────


def _rename_map(columns: list[str]) -> dict[str, str]:
    """Original column -> readable name (statistic suffix kept)."""
    out = {}
    for col in columns:
        base, _, stat = col.rpartition("_")
        if stat in _STATS and base in _SENSOR_NAMES:
            out[col] = f"{_SENSOR_NAMES[base]}_{stat}"
        elif col in _SENSOR_NAMES:
            out[col] = _SENSOR_NAMES[col]
    out["time_stamp"] = "timestamp"
    return out


def _signals_table(raw: Path, renames: dict[str, str]) -> pl.DataFrame:
    """One row per data column: original name, description, unit, statistic."""
    desc = pl.read_csv(raw / "feature_description.csv", separator=";", infer_schema_length=None)
    desc = desc.rename(str.strip).with_columns(pl.col(pl.String).str.strip_chars())
    # The published file has U+FFFD where its degree signs were (°C, °).
    desc = desc.with_columns(pl.col("unit").str.replace_all("�", "°"))
    by_sensor = {row["sensor_name"]: row for row in desc.iter_rows(named=True)}
    rows = []
    for original, column in renames.items():
        if original == "time_stamp":
            continue
        base, _, stat = original.rpartition("_")
        sensor = base if stat in _STATS else original
        info = by_sensor.get(sensor, {})
        rows.append(
            {
                "column": column,
                "original": original,
                "description": info.get("description"),
                "unit": info.get("unit"),
                "statistic": stat if stat in _STATS else "avg",
                "is_angle": str(info.get("is_angle")).lower() == "true",
                "is_counter": str(info.get("is_counter")).lower() == "true",
            }
        )
    rows.append(
        {
            "column": "status_type_id",
            "original": "status_type_id",
            "description": "Turbine status (0 normal ... 4 downtime); derived from the "
            "fault logbook in Wind Farm A, so it leaks the label",
            "unit": None,
            "statistic": None,
            "is_angle": False,
            "is_counter": False,
        }
    )
    return pl.DataFrame(rows)


def _turbine(asset: int) -> str:
    return f"WT{asset:02d}"


def _check_data_root(data_root: Path) -> None:
    """Refuse a data root that already holds assets other than these turbines.

    tessa treats every folder under the data root as an asset, so another
    store's folders (an earlier example, real data) would mix with the
    turbines and break whatever reads the store.
    """
    if not data_root.is_dir():
        return
    foreign = sorted(
        p.name
        for p in data_root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and not re.fullmatch(r"WT\d{2,}", p.name)
    )
    if foreign:
        shown = ", ".join(foreign[:5]) + (", ..." if len(foreign) > 5 else "")
        raise ValueError(
            f"{data_root} already holds asset folders that are not Wind Farm A turbines "
            f"({shown}); a Dataset on it would mix the two stores. Point "
            "fetch_wind_farm_a at an empty folder of its own, or remove those folders."
        )


def _build_store(raw: Path, root: Path, ids: list[int], progress: bool) -> pl.DataFrame:
    """Write the per-turbine monthly files and the label sheet; return the signals table."""
    events = pl.read_csv(raw / "event_info.csv", separator=";", infer_schema_length=None)
    events = events.rename(str.strip)  # headers may carry the CR of a CRLF line end
    events = events.rename({"asset": "asset_id"}) if "asset" in events.columns else events
    shift = pl.col("event_id").replace_strict(_TIME_SHIFTS, default=timedelta(0))
    events = events.filter(pl.col("event_id").is_in(ids)).with_columns(
        pl.col("event_start", "event_end").str.to_datetime() + shift
    )

    data_root = root / "data"
    renames: dict[str, str] = {}
    for (asset,), group in events.group_by("asset_id", maintain_order=True):
        frames = []
        for event_id in group["event_id"].to_list():
            frame = pl.read_parquet(raw / f"{event_id}.parquet")
            frame = frame.drop([c for c in _DROPPED if c in frame.columns])
            frame = frame.with_columns(
                pl.col("time_stamp").str.to_datetime() + _TIME_SHIFTS.get(event_id, timedelta(0))
            )
            frames.append(frame)
        merged = pl.concat(frames, how="diagonal_relaxed")
        # Overlapping datasets must agree wherever they share a timestamp.
        n_stamps = merged["time_stamp"].n_unique()
        merged = merged.unique(maintain_order=True)
        if merged.height != n_stamps:
            clash = merged.filter(pl.col("time_stamp").is_duplicated())["time_stamp"]
            raise ValueError(
                f"Turbine {asset}: the CARE datasets {group['event_id'].to_list()} disagree "
                f"at {clash.n_unique()} shared timestamps (first {clash.min()})"
            )
        renames = _rename_map(merged.columns)
        merged = merged.rename(renames).sort("timestamp")

        folder = data_root / _turbine(asset)
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.parquet"):
            old.unlink()
        months = merged.with_columns(_month=pl.col("timestamp").dt.truncate("1mo"))
        for (_,), month in months.group_by("_month", maintain_order=True):
            first, last = month["timestamp"].min(), month["timestamp"].max()
            month.drop("_month").write_parquet(
                folder / f"scada_{first:%Y%m%d}_{last:%Y%m%d}.parquet"
            )
        if progress:
            span = f"{merged['timestamp'].min():%Y-%m-%d} to {merged['timestamp'].max():%Y-%m-%d}"
            print(f"  {_turbine(asset)}: {merged.height} rows, {span}", flush=True)

    labels = events.sort("asset_id", "event_start").select(
        pl.col("asset_id").map_elements(_turbine, return_dtype=pl.String).alias("Turbine"),
        pl.col("event_start").alias("Event start"),
        pl.col("event_end").alias("Event end"),
        pl.col("event_label").alias("Label"),
        pl.col("event_description").fill_null("").alias("Fault"),
        pl.col("event_id").alias("CARE event"),
    )
    labels.write_excel(root / "labels.xlsx", worksheet="events", autofit=True)
    return _signals_table(raw, renames)


def fetch_wind_farm_a(
    root: str | Path,
    *,
    archive: str | Path | None = None,
    force: bool = False,
    workers: int = 4,
    progress: bool = True,
) -> ExampleDataset:
    """Download CARE to Compare Wind Farm A and write it as a tessa data store.

    See the module docstring for what is written, what the conversion changes,
    and the license (CC BY-SA 4.0, cite the paper).

    Parameters
    ----------
    root : str or Path
        Target directory, used for this example only; ``data/``,
        ``labels.xlsx``, ``signals.csv`` and the download cache ``raw/`` are
        created inside.
    archive : str or Path, optional
        A local copy of ``CARE_To_Compare.zip`` to read instead of Zenodo,
        for machines without direct internet access.
    force : bool, default False
        Rebuild the store even if it is already there. The download cache is
        kept either way; delete ``<root>/raw`` to download again.
    workers : int, default 4
        Parallel downloads.
    progress : bool, default True
        Print what is being downloaded and written.

    Returns
    -------
    ExampleDataset
        ``data_root``, ``labels_path``, the ``column_map`` the label sheet
        needs (:data:`WIND_FARM_A_COLUMN_MAP`), and ``signals``, the table
        describing every column.

    Raises
    ------
    ConnectionError
        If Zenodo cannot be reached (pass ``archive=`` to work offline).
    ValueError
        If ``<root>/data`` already holds folders other than the turbines (give
        the example a folder of its own), or if the datasets of one turbine
        disagree where they overlap, which would mean the archive differs from
        the version this was written for.
    """
    root = Path(root)
    _check_data_root(root / "data")
    marker = root / _MARKER
    out = ExampleDataset(
        data_root=root / "data",
        labels_path=root / "labels.xlsx",
        column_map=dict(WIND_FARM_A_COLUMN_MAP),
    )
    if not force and marker.exists() and (root / "signals.csv").exists():
        if json.loads(marker.read_text()).get("zenodo_record") == ZENODO_RECORD:
            return dataclasses.replace(out, signals=pl.read_csv(root / "signals.csv"))

    marker.unlink(missing_ok=True)
    ids = _download(root / "raw", archive, workers, progress)
    if progress:
        print("Wind Farm A: writing the data store...", flush=True)
    signals = _build_store(root / "raw", root, ids, progress)
    signals.write_csv(root / "signals.csv")
    marker.write_text(json.dumps({"zenodo_record": ZENODO_RECORD, "datasets": ids}))
    return dataclasses.replace(out, signals=signals)

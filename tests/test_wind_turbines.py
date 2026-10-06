"""The CARE to Compare converter, on a small fake archive (no network).

The fake archive mimics Wind Farm A: one CSV per event, the CSVs of one
turbine overlapping in time, dataset 71 stamped a day late, a CRLF event table
and degree signs lost to U+FFFD in the feature description.
"""

from __future__ import annotations

import io
import re
import threading
import zipfile
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import polars as pl
import pytest

import tessa.examples.wind_turbines as wt
from tessa import Config, Dataset
from tessa.examples import WIND_FARM_A_COLUMN_MAP, fetch_wind_farm_a
from tessa.labels import ExcelLabelSource

FARM = "CARE_To_Compare/Wind Farm A/"
T0 = datetime(2023, 1, 25)


def _turbine_history(asset: int, days: int = 20) -> pl.DataFrame:
    """Ten-minute history of one turbine, as every CARE dataset of it agrees."""
    n = days * 144
    ts = pl.datetime_range(T0, T0 + timedelta(minutes=10 * (n - 1)), "10m", eager=True)
    i = pl.int_range(n, eager=True)
    return pl.DataFrame(
        {
            "time_stamp": ts,
            "asset_id": asset,
            "status_type_id": (i % 5 == 0).cast(pl.Int64) * 4,
            "sensor_0_avg": (i % 144).cast(pl.Float64) / 10 + asset,
            "wind_speed_3_avg": (i % 7).cast(pl.Float64),
            "wind_speed_3_max": (i % 7).cast(pl.Float64) + 2,
            "sensor_44": (i * 3).cast(pl.Float64),
        }
    )


def _dataset_csv(history: pl.DataFrame, first: int, last: int, shift_days: int = 0) -> bytes:
    """Rows ``first..last`` of a history as a CARE dataset CSV."""
    part = history.slice(first, last - first).with_columns(
        id=pl.int_range(pl.len()),
        train_test=pl.lit("train"),
        time_stamp=(pl.col("time_stamp") + timedelta(days=shift_days)).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )
    cols = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]
    part = part.select(cols + [c for c in part.columns if c not in cols])
    return part.write_csv(separator=";").encode()


EVENTS = (
    "asset;event_id;event_label;event_start;event_start_id;event_end;event_end_id;"
    "event_description\r\n"
    "0;5;anomaly;2023-02-05 00:00:00;1;2023-02-06 00:00:00;2;Gearbox failure\r\n"
    "0;71;normal;2023-02-02 00:00:00;1;2023-02-03 00:00:00;2;\r\n"
    "3;8;normal;2023-01-30 00:00:00;1;2023-01-31 00:00:00;2;\r\n"
)
FEATURES = (
    "sensor_name;statistics_type;description;unit;is_angle;is_counter\n"
    "sensor_0;average;Ambient temperature;�C;False;False\n"
    "wind_speed_3;maximum,average;Windspeed;m/s;False;False\n"
    "sensor_44;average;Active power - generator disconnected;Wh;False;False\n"
)


def _archive(path: Path, *, disagree: bool = False) -> Path:
    """Write a fake CARE_To_Compare.zip; ``disagree`` breaks one overlap."""
    h0, h3 = _turbine_history(0), _turbine_history(3)
    other = h0.with_columns(pl.col("sensor_0_avg") + 1) if disagree else h0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(FARM + "event_info.csv", EVENTS)
        zf.writestr(FARM + "feature_description.csv", FEATURES)
        zf.writestr(FARM + "datasets/5.csv", _dataset_csv(h0, 0, 2000))
        # Dataset 71 overlaps dataset 5 but is stamped a day late, as in CARE.
        zf.writestr(FARM + "datasets/71.csv", _dataset_csv(other, 1000, 2880, shift_days=1))
        zf.writestr(FARM + "datasets/8.csv", _dataset_csv(h3, 0, 1500))
        zf.writestr("CARE_To_Compare/Wind Farm B/event_info.csv", "not read")
    return path


def test_converts_the_archive_into_a_tessa_store(tmp_path: Path):
    ex = fetch_wind_farm_a(tmp_path / "out", archive=_archive(tmp_path / "care.zip"))
    assert ex.column_map == WIND_FARM_A_COLUMN_MAP

    cfg = Config(data_root=ex.data_root)
    ds = Dataset(cfg)
    assert ds.assets == ["WT00", "WT03"]
    files = sorted(p.name for p in (ex.data_root / "WT00").iterdir())
    assert files == ["scada_20230125_20230131.parquet", "scada_20230201_20230213.parquet"]

    # The overlapping datasets merge into one timeline, identical to the source.
    wt00 = pl.read_parquet(ex.data_root / "WT00").sort("timestamp")
    assert wt00.columns == [
        "timestamp",
        "status_type_id",
        "ambient_temp_avg",
        "wind_speed_avg",
        "wind_speed_max",
        "energy_gen_disconnected",
    ]
    truth = _turbine_history(0).head(2880)
    assert wt00["timestamp"].to_list() == truth["time_stamp"].to_list()
    assert wt00["ambient_temp_avg"].to_list() == truth["sensor_0_avg"].to_list()

    labels = ExcelLabelSource(ex.labels_path, sheet="events", column_map=ex.column_map).load(cfg)
    assert labels.columns == ["asset_id", "start", "end", "class", "fault", "care_event"]
    shifted = labels.filter(pl.col("care_event") == 71)
    assert shifted["start"][0] == datetime(2023, 2, 1)  # moved back with its data
    assert labels.filter(pl.col("care_event") == 5)["fault"][0] == "Gearbox failure"
    assert ds.available(labels, check="rows").height == 3

    signals = ex.signals
    temp = signals.filter(pl.col("column") == "ambient_temp_avg").row(0, named=True)
    assert temp["original"] == "sensor_0_avg" and temp["unit"] == "°C"
    assert signals.filter(pl.col("column") == "wind_speed_max")["statistic"][0] == "max"
    assert "status_type_id" in signals["column"].to_list()


def test_second_call_reuses_the_store(tmp_path: Path):
    archive = _archive(tmp_path / "care.zip")
    first = fetch_wind_farm_a(tmp_path / "out", archive=archive, progress=False)
    archive.unlink()  # a rebuild would need it
    again = fetch_wind_farm_a(tmp_path / "out", archive=archive, progress=False)
    assert again.signals.equals(first.signals)
    # Rebuilding works from the download cache alone.
    rebuilt = fetch_wind_farm_a(tmp_path / "out", archive=archive, force=True, progress=False)
    assert rebuilt.signals.equals(first.signals)


def test_disagreeing_overlaps_are_refused(tmp_path: Path):
    archive = _archive(tmp_path / "care.zip", disagree=True)
    with pytest.raises(ValueError, match=r"Turbine 0: the CARE datasets .* disagree"):
        fetch_wind_farm_a(tmp_path / "out", archive=archive, progress=False)


# ── Remote access ────────────────────────────────────────────────────────────


def _serve(payload: bytes, ranges: bool = True):
    """Serve ``payload`` over HTTP on localhost, honouring Range requests."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            m = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
            if ranges and m:
                first, last = int(m.group(1)), min(int(m.group(2)), len(payload) - 1)
                body = payload[first : last + 1]
                self.send_response(206)
                # Lower-case, as Zenodo sends it.
                self.send_header("content-range", f"bytes {first}-{last}/{len(payload)}")
            else:
                body = payload
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/CARE_To_Compare.zip"


def test_reads_members_of_a_remote_archive(tmp_path: Path, monkeypatch):
    payload = _archive(tmp_path / "care.zip").read_bytes()
    server, url = _serve(payload)
    try:
        remote = wt._HTTPRangeFile(url)
        assert remote.size == len(payload)
        with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=4096)) as zf:
            assert zf.read(FARM + "event_info.csv").decode() == EVENTS

        monkeypatch.setattr(wt, "ARCHIVE_URL", url)
        ex = fetch_wind_farm_a(tmp_path / "out", workers=2, progress=False)
        assert Dataset(Config(data_root=ex.data_root)).assets == ["WT00", "WT03"]
    finally:
        server.shutdown()


def test_refuses_a_server_without_range_support(tmp_path: Path):
    server, url = _serve(b"x" * 100, ranges=False)
    try:
        with pytest.raises(OSError, match="does not support HTTP range requests"):
            wt._HTTPRangeFile(url)
    finally:
        server.shutdown()

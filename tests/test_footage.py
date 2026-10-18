"""Recorded footage: photo metadata, offline analysis and track exports."""

import json
import math

import numpy as np
import pytest
from PIL import Image

from pharos.cli import main
from pharos.eval.export import to_csv, to_geojson, to_kml
from pharos.footage import load_photos
from tests.conftest import write_dji_recording

DJI_XMP = (
    b'<x:xmpmeta><rdf:Description drone-dji:RelativeAltitude="+120.50" '
    b'drone-dji:GimbalPitchDegree="-80.0" drone-dji:GimbalRollDegree="0.0" '
    b'drone-dji:GimbalYawDegree="+45.2" drone-dji:FlightYawDegree="+44.0"/></x:xmpmeta>'
)


def write_photo(path, when, lat, lon, alt, xmp=b""):
    img = Image.fromarray(np.full((300, 400, 3), 120, np.uint8))
    exif = img.getexif()
    exif[0x0132] = when
    gps = exif.get_ifd(0x8825)
    to_dms = lambda v: (int(v), int(v * 60 % 60), round(v * 3600 % 60, 4))  # noqa: E731
    gps.update({1: "N", 2: to_dms(lat), 3: "W", 4: to_dms(-lon), 5: 0, 6: alt})
    exif.get_ifd(0x8769)[0xA405] = 24  # 35 mm-equivalent focal length
    img.save(path, exif=exif)
    if xmp:
        with open(path, "ab") as f:  # vendors embed XMP in APP1; appended works for the reader
            f.write(xmp)


def test_dji_photos(tmp_path):
    write_photo(tmp_path / "b.jpg", "2024:05:03 10:00:04", 46.9512, -7.44, 700.0, DJI_XMP)
    write_photo(tmp_path / "a.jpg", "2024:05:03 10:00:00", 46.9510, -7.44, 700.0, DJI_XMP)
    footage = load_photos(tmp_path)
    frames = list(footage.recording)
    assert [f.t for f in frames] == [0.0, 4.0]  # sorted by capture time
    tel = frames[0].telemetry
    assert tel.altitude == 120.5  # DJI relative altitude, not GPS altitude
    assert math.isclose(math.degrees(tel.pitch), 10.0)  # gimbal -80 = tilted 10 degrees
    assert math.isclose(math.degrees(tel.yaw), 45.2)
    assert footage.gnss[0] == pytest.approx((46.9510, -7.44), abs=1e-5)
    # 24 mm equivalent: 84 degrees diagonal, about 71.6 degrees across a 4:3 frame
    cam = footage.recording.camera
    assert 71 < math.degrees(2 * math.atan(cam.width / 2 / cam.fx)) < 72


def test_photos_without_height_need_help(tmp_path):
    write_photo(tmp_path / "a.jpg", "2024:05:03 10:00:00", 46.95, -7.44, 700.0)
    with pytest.raises(ValueError, match="height above ground"):
        load_photos(tmp_path)
    assert next(iter(load_photos(tmp_path, ground_m=600).recording)).telemetry.altitude == 100
    assert next(iter(load_photos(tmp_path, altitude_m=80).recording)).telemetry.altitude == 80


def test_exports():
    records = [
        {"t": 0.0, "status": "lost", "altitude_m": 100},
        {
            "t": 0.5,
            "status": "tracking",
            "lat": 46.95,
            "lon": 7.44,
            "heading_deg": 10,
            "gnss": {"lat": 46.95001, "lon": 7.44, "x": 0, "y": 0},
            "error_m": 1.1,
        },
        {"t": 1.0, "status": "tracking", "lat": 46.951, "lon": 7.441, "heading_deg": 12},
    ]
    geo = json.loads(to_geojson(records))
    kinds = [f["geometry"]["type"] for f in geo["features"]]
    assert kinds.count("LineString") == 1 and kinds.count("Point") == 2
    assert "7.4400000,46.9500000" in to_kml(records)
    rows = to_csv(records).splitlines()
    assert rows[0].startswith("t,status,lat,lon") and len(rows) == 4
    assert "46.95001" in rows[2]


def test_analyze_recorded_dji_video(small_sim, tmp_path):
    geomap, rec = small_sim
    geomap.save(tmp_path / "map")
    video, _ = write_dji_recording(rec, geomap, tmp_path)  # SRT found next to the video
    out = tmp_path / "out"
    main(
        ["analyze", str(video), "--map", str(tmp_path / "map.json"), "--hfov", "70", "-o", str(out)]
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["frames"] == len(rec.frames)
    assert summary["tracking"] > 0.5 and summary["error_median_m"] < 8
    assert summary["confident_wrong"] == 0
    for name in ("track.geojson", "track.kml", "track.csv", "overview.png", "frames.jsonl"):
        assert (out / name).stat().st_size > 0


def test_probe_separates_truth_from_decoys_and_calibrates(small_sim):
    from dataclasses import replace

    from pharos.eval.probe import probe, summarize_probe
    from pharos.footage import Footage
    from pharos.recording import Recording

    geomap, rec = small_sim
    gnss = [geomap.to_latlon(g.x, g.y) for g in rec.truth]
    footage = Footage(Recording(rec.camera, rec.frames, rec.truth), gnss)
    summary = summarize_probe(probe(footage, geomap, every=4))
    assert summary["frames"] == 15
    # at the tracker's threshold (LocalizerConfig.track_min_excess)
    assert summary["true_pass_2.0"] >= 0.8
    assert summary["decoy_pass_2.0"] <= 0.15
    assert summary["calibration"]["mount_offset_deg"] == pytest.approx([0, 0], abs=0.5)
    # the simulated barometer reads 4% high
    assert summary["calibration"]["altitude_scale"] == pytest.approx(1 / 1.04, abs=0.02)

    # telemetry that misses 3 degrees of forward camera tilt
    tilt = math.radians(3)
    frames = [
        replace(f, telemetry=replace(f.telemetry, pitch=f.telemetry.pitch - tilt))
        for f in rec.frames
    ]
    footage = Footage(Recording(rec.camera, frames, rec.truth), gnss)
    calibration = summarize_probe(probe(footage, geomap, every=6))["calibration"]
    assert calibration["mount_offset_deg"] == pytest.approx([3, 0], abs=0.7)


def test_video_window_and_mount_offset(small_sim, tmp_path):
    from pharos.footage import load_footage

    geomap, rec = small_sim
    video, _ = write_dji_recording(rec, geomap, tmp_path)
    full = load_footage(video, hfov_deg=70)
    part = load_footage(video, hfov_deg=70, start_s=5, end_s=10, mount_offset_deg=(2.0, -1.0))
    times = [f.t for f in part.recording]
    assert times[0] == 5.0 and times[-1] == 10.0 and len(times) == 11
    before, after = full.recording.frames.telemetry[10], part.recording.frames.telemetry[0]
    assert math.degrees(after.pitch - before.pitch) == pytest.approx(2.0)
    assert math.degrees(after.roll - before.roll) == pytest.approx(-1.0)
    assert part.gnss == full.gnss[10:21]

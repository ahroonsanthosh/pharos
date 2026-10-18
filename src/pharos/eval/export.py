"""Per-frame result records and track exports (GeoJSON, KML, CSV).

A record is a plain dict, the same for live feeds and offline analysis, so
the dashboard, the API and ``pharos analyze`` all export identical files.
"""

from __future__ import annotations

import csv
import io
import json
import math
from xml.sax.saxutils import escape

from pharos.geomap import GeoMap
from pharos.types import Estimate, Status, Telemetry

TRUSTED = (Status.TRACKING.value,)
CSV_FIELDS = (
    "t",
    "status",
    "lat",
    "lon",
    "heading_deg",
    "sigma_m",
    "score",
    "altitude_m",
    "agl_m",
    "tilt_deg",
    "gnss_lat",
    "gnss_lon",
    "error_m",
    "latency_ms",
)


def record(
    e: Estimate,
    telemetry: Telemetry,
    gnss: tuple[float, float] | None,
    geomap: GeoMap,
    used: Telemetry | None = None,
) -> dict:
    """One frame's result. ``gnss`` is the aircraft's own position, if it reported one.

    ``used`` is the telemetry as the localizer used it: with the height above the
    ground (``agl_m``) worked out from the terrain where the aircraft reported its
    height above takeoff.
    """
    rec: dict = {
        "t": round(e.t, 3),
        "status": e.status.value,
        "altitude_m": round(telemetry.altitude, 1),
        "tilt_deg": round(math.degrees(telemetry.pitch), 1),
        "latency_ms": round(e.latency_ms),
    }
    if used is not None and telemetry.above_takeoff and not used.above_takeoff:
        rec["agl_m"] = round(used.altitude, 1)
    if e.status is not Status.LOST:
        rec |= {
            "x": e.x,
            "y": e.y,
            "heading_deg": round(math.degrees(e.heading) % 360, 1),
            "sigma_m": round(e.sigma_xy, 1),
        }
        if (latlon := geomap.to_latlon(e.x, e.y)) is not None:
            rec["lat"], rec["lon"] = latlon
    if e.fix is not None:
        rec["score"] = round(e.fix.score, 3)
    if gnss is not None and (xy := geomap.from_latlon(*gnss)) is not None:
        rec["gnss"] = {"lat": gnss[0], "lon": gnss[1], "x": xy[0], "y": xy[1]}
        if "x" in rec:
            rec["error_m"] = round(math.hypot(e.x - xy[0], e.y - xy[1]), 1)
    return rec


def to_geojson(records: list[dict], name: str = "pharos") -> str:
    """Trusted positions as a line plus one point per located frame (all statuses)."""
    trusted = [[r["lon"], r["lat"]] for r in records if r["status"] in TRUSTED and "lat" in r]
    features = []
    if len(trusted) > 1:
        line = {"type": "LineString", "coordinates": trusted}
        features.append({"type": "Feature", "geometry": line, "properties": {"name": name}})
    for r in records:
        if "lat" not in r:
            continue
        props = {k: r[k] for k in CSV_FIELDS if k in r and k not in ("lat", "lon")}
        point = {"type": "Point", "coordinates": [r["lon"], r["lat"]]}
        features.append({"type": "Feature", "geometry": point, "properties": props})
    gnss = [[r["gnss"]["lon"], r["gnss"]["lat"]] for r in records if "gnss" in r]
    if len(gnss) > 1:
        line = {"type": "LineString", "coordinates": gnss}
        features.append({"type": "Feature", "geometry": line, "properties": {"name": "GNSS"}})
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=1)


def to_kml(records: list[dict], name: str = "pharos") -> str:
    """Google Earth: the trusted visual track in blue, the aircraft's GNSS in green."""

    def line(coords: list[tuple[float, float]], label: str, color: str) -> str:
        pts = " ".join(f"{lon:.7f},{lat:.7f},0" for lat, lon in coords)
        return (
            f"<Placemark><name>{escape(label)}</name><Style><LineStyle><color>{color}</color>"
            f"<width>3</width></LineStyle></Style><LineString><tessellate>1</tessellate>"
            f"<coordinates>{pts}</coordinates></LineString></Placemark>"
        )

    parts = []
    trusted = [(r["lat"], r["lon"]) for r in records if r["status"] in TRUSTED and "lat" in r]
    if len(trusted) > 1:
        parts.append(line(trusted, f"{name} (visual)", "ffff8c3b"))
    gnss = [(r["gnss"]["lat"], r["gnss"]["lon"]) for r in records if "gnss" in r]
    if len(gnss) > 1:
        parts.append(line(gnss, "aircraft GNSS", "ff7dd04c"))
    body = "".join(parts)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"<name>{escape(name)}</name>{body}</Document></kml>"
    )


def to_csv(records: list[dict]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for r in records:
        row = dict(r)
        if "gnss" in r:
            row["gnss_lat"], row["gnss_lon"] = r["gnss"]["lat"], r["gnss"]["lon"]
        writer.writerow(row)
    return out.getvalue()

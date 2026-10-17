# Pharos

Absolute visual localization for drones: match a downward camera feed against
free satellite or aerial imagery to get position and heading **without GNSS**.

Pharos is built to survive what changes between when the map was captured
and when the drone flies over: arbitrary heading, season, sun angle and
shadows, haze, altitude error, camera tilt, and land changes such as crop
rotation, construction and felling. It runs on a CPU with no learned weights.
A simulator generates realistic test flights so the full pipeline can be
developed and benchmarked before any hardware exists.

## Quick start

```bash
pip install -e ".[dev]"          # add ,geo for GeoTIFF maps

pytest -q                        # unit + small end-to-end tests (~15 s)
pharos bench configs/scenarios/*.yaml -o runs/bench    # full benchmark

# or step by step, through the same on-disk format a real flight uses
pharos simulate configs/scenarios/autumn_low_sun.yaml -o runs/autumn
pharos localize --map runs/autumn/map.json --recording runs/autumn/recording -o runs/autumn/out
```

`localize` writes `estimates.jsonl` (position, lat/lon, heading, uncertainty
and status per frame), `track.png`, and `metrics.json` when the recording
has ground truth.

## How it works

1. **Rectify.** Each frame is warped onto the ground plane at map
   resolution using the reported altitude and camera tilt.
2. **Describe.** Frame and map become dense orientation fields that keep
   edges (roads, field boundaries, buildings, shorelines) and discard
   intensity, contrast polarity, canopy texture and sun-driven shading.
3. **Correlate.** FFT cross-correlation scores every position at once, over a
   set of heading and altitude-scale hypotheses.
4. **Localize.** When lost, a global search proposes candidates. Several are
   followed in parallel until one clearly wins over 120 m of flight. After
   that, a Kalman filter tracks with cheap local searches and slowly learns
   barometric and compass bias.

Each estimate carries a status: `lost`, `converging` (a candidate, not yet
trustworthy) or `tracking`. Details and design rationale are in
[docs/architecture.md](docs/architecture.md).

## Maps and video

Any north-up orthophoto works. Free sources include USGS NAIP, national
programmes such as IGN, swisstopo and PDOK, OpenAerialMap and Sentinel-2.
Video can come from any file or stream OpenCV decodes, plus a telemetry log.
See [docs/data.md](docs/data.md) for sources, licences and the recording
format.

## Benchmark

Ten scenarios in `configs/scenarios/`, each a 60 s flight at 180 m over a
1.6 km × 1.6 km world, localized at 2 Hz against a summer reference map:

<!-- bench:start -->
<!-- bench:end -->

Columns: share of frames in `tracking`, time to first confirmed fix, median
and 95th-percentile position error while tracking, median heading error,
share of all frames within 10 m, share of tracking frames more than 25 m off
(confidently wrong), and median processing time per frame on a 4-core CPU.

## Layout

```
src/pharos/
  geometry.py  geomap.py  types.py  recording.py  config.py  cli.py
  vision/   rectify, features, correlate, matcher
  nav/      filter, localizer
  sim/      world, appearance, trajectory, flight, scenario, procedural
  eval/     metrics, plot, bench
configs/scenarios/   benchmark scenarios (YAML)
docs/                architecture, data sources
tests/
```

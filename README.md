# Pharos

Absolute visual localization for drones: match a downward camera feed against
free satellite or aerial imagery to get position and heading **without GNSS**.

Pharos is built to survive what changes between when the map was captured
and when the drone flies over: arbitrary heading, season, sun angle and
shadows, haze, altitude error, camera tilt, and land changes such as crop
rotation, construction and felling. It runs on a CPU with no learned weights.
A simulator generates realistic test flights so the full pipeline can be
developed and benchmarked before any hardware exists.

## Run it on a laptop

```bash
git clone https://github.com/ahroonsanthosh/pharos.git && cd pharos
docker compose up            # then open http://localhost:8000
```

The dashboard takes feeds from anywhere: a DJI controller or OBS pushing
RTMP/SRT, RTSP/HLS/UDP stream URLs, an HDMI capture card or webcam, recorded
videos with DJI `.SRT` telemetry, or a built-in demo flight. Telemetry
comes from MAVLink (ArduPilot/PX4), DJI subtitle files or manual entry. When
the aircraft reports GPS, it is drawn next to the visual estimate, so every
real flight measures accuracy. Setup for each source is in
[docs/laptop.md](docs/laptop.md).

## Develop

```bash
pip install -e ".[dev,server,geo]"
pytest -q                                  # unit + end-to-end tests
pharos bench -j 4 -o runs/bench            # all built-in scenarios
pharos simulate autumn_low_sun -o runs/autumn      # a flight in the recording format
pharos localize --map runs/autumn/map.json --recording runs/autumn/recording -o runs/autumn/out
pharos serve --open                        # dashboard without Docker
```

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

## Maps

Any north-up orthophoto works. Free sources include USGS NAIP, national
programmes such as IGN, swisstopo and PDOK, OpenAerialMap and Sentinel-2.
See [docs/data.md](docs/data.md) for sources, licences and the recording
format.

## Benchmark

Ten scenarios in `src/pharos/sim/scenarios/`, each a 60 s flight at 180 m
over a 1.6 km × 1.6 km world, localized at 2 Hz against a summer reference
map:

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
  sim/      world, appearance, trajectory, flight, scenario, scenarios/*.yaml
  eval/     metrics, plot, bench
  server/   app, feeds, telemetry, store, static/ (dashboard)
docs/       laptop setup, architecture, data sources
tests/
Dockerfile  compose.yaml
```

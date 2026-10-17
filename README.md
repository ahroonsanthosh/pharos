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

## Analyze recorded footage

The same engine works on archives, not just live feeds: patrol logs, survey
photo sets, old flight videos. Footage is processed as fast as the CPU allows.

```bash
pharos analyze DJI_0042.MP4 --map area.tif --hfov 70        # .SRT next to the video is used
pharos analyze photos/ --map area.tif                       # EXIF GPS/time, DJI or senseFly XMP
pharos analyze old.mp4 --map area.tif --hfov 70 --altitude 120   # no telemetry at all
```

It writes `track.geojson`, `track.kml` (opens in Google Earth), `track.csv`,
an overview image, and `summary.json`. When the footage carries its own GPS,
the summary scores Pharos against it. The dashboard offers the same exports
for every feed, live or recorded, and accepts photo sets as well as videos.

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
   that, a Kalman filter tracks with cheap local searches, driven by visual
   odometry between consecutive frames, and slowly learns barometric and
   compass bias.

Each estimate carries a status: `lost`, `converging` (a candidate, not yet
trustworthy) or `tracking`. Details and design rationale are in
[docs/architecture.md](docs/architecture.md).

## Maps

Any north-up orthophoto works. Free sources include USGS NAIP, national
programmes such as IGN, swisstopo and PDOK, OpenAerialMap and Sentinel-2.
See [docs/data.md](docs/data.md) for sources, licences and the recording
format.

## Benchmark

Twelve scenarios in `src/pharos/sim/scenarios/`, each a 60 s flight at
180 m over a 1.6 km × 1.6 km world, localized at 2 Hz from a cold start (no
initial position) against a summer reference map:

<!-- bench:start -->
| scenario | avail | fix s | err50 m | err95 m | hdg deg | <10 m | wrong | ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| altitude_bias | 84% | 9.5 | 1.9 | 4.3 | 0.1 | 84% | 0.0% | 367 |
| autumn_low_sun | 84% | 9.5 | 2.4 | 5.7 | 0.2 | 84% | 0.0% | 379 |
| dusk_haze | 84% | 9.5 | 3.0 | 21.1 | 0.8 | 73% | 0.0% | 364 |
| grid_farmland | 85% | 9.0 | 2.0 | 4.6 | 0.2 | 85% | 0.0% | 320 |
| kitchen_sink | 84% | 9.5 | 2.7 | 5.4 | 0.9 | 84% | 0.0% | 266 |
| land_change | 84% | 9.5 | 2.7 | 5.1 | 0.1 | 84% | 0.0% | 311 |
| no_compass_spinning | 84% | 9.5 | 2.2 | 3.6 | 0.1 | 84% | 0.0% | 293 |
| nominal | 85% | 9.0 | 1.9 | 3.9 | 0.1 | 85% | 0.0% | 312 |
| sensor_errors | 84% | 9.5 | 7.0 | 9.7 | 0.2 | 81% | 0.0% | 353 |
| spring_clouds_on_map | 72% | 9.5 | 3.3 | 9.0 | 0.6 | 72% | 0.0% | 345 |
| tilted_camera | 85% | 9.0 | 1.7 | 3.6 | 0.1 | 85% | 0.0% | 307 |
| winter_snow | 69% | 18.5 | 2.3 | 4.3 | 0.2 | 69% | 0.0% | 238 |
<!-- bench:end -->

Columns: share of frames in `tracking`, time to first confirmed fix, median
and 95th-percentile position error while tracking, median heading error,
share of all frames within 10 m, share of tracking frames more than 25 m off
(confidently wrong), and median processing time per frame with four scenarios
running in parallel on a 4-core CPU.

Availability tops out near 85% because a candidate must cover 120 m of new
ground (about 10 s) before it is trusted. Lower rows are informative:
`spring_clouds_on_map` flies under clouds baked into the reference map, where
odometry carries the track part of the way; `winter_snow` needs longer to
find a distinctive start under the stricter gate that real imagery required; `sensor_errors` shows that a 2° camera mount
misalignment becomes a ~6 m bias that no single frame can reveal; the
`dusk_haze` tail is one turn over forest where the filter coasts straight.

The simulator is still kinder than reality (flat ground, the map and the
drone view share exact geometry). Expect real cross-season match scores to be
lower and treat these numbers as an upper bound until validated on flights.

## First real-imagery test

Public test data, all free: OpenDroneMap's *Aukerman* survey (CC0): 77 nadir
photos from a senseFly eBee fixed-wing at about 100 m, June 2016, with GPS.
It is matched against USGS NAIP imagery from three years, with no initial
position given:

| reference map | frames trusted | median error | 95th pct | confidently wrong |
|---|---:|---:|---:|---:|
| NAIP 2015 (July, 1 m) | 39% | 11 m | 19 m | 0% |
| NAIP 2019 (October) | 18% | 12 m | 19 m | 0% |
| NAIP 2023 (May) | 31% | 11 m | 19 m | 0% |

Errors are measured against the photos' own GPS (about 1.5 m accurate). How
it got here, and what this footage taught:

* The first version confirmed look-alike places 0.7 to 1.4 km away. Real
  photos match far more weakly than synthetic ones (0.2 to 0.6 vs 0.8 to
  0.9), so fixed score thresholds were unsafe. Matches are now judged
  against the noise of their own search window.
* Visual odometry between photos replaced the constant-velocity guess (it
  needed windows up to 150 m with 4 to 13 s between photos). Its error
  differs by platform (a banking fixed-wing is far noisier than a gimbal), so
  confirmed tracks calibrate it from their own innovations. Together this
  raised coverage from 0 to 4% to 18 to 39%.
* Photo GPS altitude was above the WGS84 ellipsoid, not sea level: 34 m, a
  50% scale error at this altitude. The eBee's camera attitude is recorded,
  but its sign convention had to be determined from the data.
* The map whose season matches the flight wins, even though it predates a
  house and path visible in the photos.

This survey is a hard case (low altitude, tight loops, mostly lawn and
trees). Coverage is the open problem; see below.

## Next steps

* **Multi-frame evidence:** accumulate aligned score maps over several frames
  and stitch frames into larger templates; this is what low-altitude footage
  needs most.
* **Feature scales tuned on real imagery** (canopy and lawn texture, NAIP's
  softer resolution), validated on the Aukerman set and future flights.
* **More real flights:** a DJI MP4 + SRT gives ground truth for free.
* **Calibration helpers** for camera mount offset and video delay, run once
  on a flight with GPS.
* **Uncertainty from the correlation peak** (its curvature) instead of a
  fixed measurement error.
* **Terrain relief** via a free DEM (Copernicus GLO-30) for hilly areas.

## Layout

```
src/pharos/
  geometry.py  geomap.py  types.py  recording.py  config.py  cli.py
  footage.py  analyze.py  telemetry.py
  vision/   rectify, features, correlate, matcher, odometry
  nav/      filter, localizer
  sim/      world, appearance, trajectory, flight, scenario, scenarios/*.yaml
  eval/     metrics, plot, bench, export
  server/   app, feeds, store, static/ (dashboard)
docs/       laptop setup, architecture, data sources
tests/
Dockerfile  compose.yaml
```

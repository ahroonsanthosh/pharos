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

Twelve scenarios in `src/pharos/sim/scenarios/`, each a 60 s flight at
180 m over a 1.6 km × 1.6 km world, localized at 2 Hz from a cold start (no
initial position) against a summer reference map:

<!-- bench:start -->
| scenario | avail | fix s | err50 m | err95 m | hdg deg | <10 m | wrong | ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| altitude_bias | 82% | 10.5 | 1.4 | 2.8 | 0.4 | 82% | 0.0% | 224 |
| autumn_low_sun | 83% | 10.0 | 1.9 | 4.2 | 0.4 | 83% | 0.0% | 221 |
| dusk_haze | 84% | 9.5 | 2.8 | 21.1 | 0.8 | 73% | 0.0% | 227 |
| grid_farmland | 83% | 10.0 | 1.6 | 3.2 | 0.4 | 83% | 0.0% | 234 |
| kitchen_sink | 83% | 10.0 | 2.7 | 6.2 | 1.2 | 82% | 0.0% | 860 |
| land_change | 83% | 10.0 | 2.3 | 3.8 | 0.5 | 83% | 0.0% | 274 |
| no_compass_spinning | 83% | 10.0 | 1.1 | 2.7 | 0.1 | 83% | 0.0% | 779 |
| nominal | 84% | 9.5 | 1.4 | 2.5 | 0.3 | 84% | 0.0% | 261 |
| sensor_errors | 82% | 10.5 | 6.8 | 8.8 | 0.4 | 82% | 0.0% | 213 |
| spring_clouds_on_map | 58% | 10.0 | 3.1 | 8.6 | 0.7 | 57% | 0.0% | 238 |
| tilted_camera | 83% | 10.0 | 1.0 | 2.6 | 0.4 | 83% | 0.0% | 202 |
| winter_snow | 53% | 28.0 | 1.7 | 3.2 | 0.5 | 53% | 0.0% | 217 |
<!-- bench:end -->

Columns: share of frames in `tracking`, time to first confirmed fix, median
and 95th-percentile position error while tracking, median heading error,
share of all frames within 10 m, share of tracking frames more than 25 m off
(confidently wrong), and median processing time per frame with four scenarios
running in parallel on a 4-core CPU.

Availability tops out near 83% because a candidate must cover 120 m of new
ground (about 10 s) before it is trusted. Lower rows are informative:
`spring_clouds_on_map` flies under clouds baked into the reference map and
correctly reports lost there; `winter_snow` needs longer to find a
distinctive start under the stricter gate that real imagery required; `sensor_errors` shows that a 2° camera mount
misalignment becomes a ~6 m bias that no single frame can reveal; the
`dusk_haze` tail is one turn over forest where the filter coasts straight.

The simulator is still kinder than reality (flat ground, the map and the
drone view share exact geometry). Expect real cross-season match scores to be
lower and treat these numbers as an upper bound until validated on flights.

## First real-imagery test

Public test data, all free: OpenDroneMap's *Aukerman* survey (CC0): 77 nadir
photos from a senseFly eBee fixed-wing at about 100 m, June 2016, with GPS.
It was matched against USGS NAIP imagery from 2015, 2019 and 2023, with no
initial position given.

* **Safety:** after adding a window-corrected peak statistic, no frame was
  confidently wrong. Before that, several runs confirmed look-alike places
  0.7 to 1.4 km away. This is the result that matters most.
* **Accuracy when confident:** about 9 m median against the photos' GPS.
* **Coverage: poor, about 4% of frames.** This survey is a hard case: low
  altitude (each photo covers only ~140 m), tight loops with 4 to 13 s
  between photos, mostly lawn and trees, and the house and path in the photos
  were built after the 2015 imagery.
* Two real-world traps turned up and are now handled: photo GPS altitudes
  above the WGS84 ellipsoid rather than sea level (34 m here, i.e. a 50%
  scale error at this altitude), and vendor-specific attitude metadata.

Raising coverage on footage like this is the next milestone (see below).

## Next steps

* **Visual odometry** between consecutive frames as the motion model: small
  search windows instead of 150 m guesses, real distance for confirmation,
  coasting through featureless stretches.
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
  vision/   rectify, features, correlate, matcher
  nav/      filter, localizer
  sim/      world, appearance, trajectory, flight, scenario, scenarios/*.yaml
  eval/     metrics, plot, bench, export
  server/   app, feeds, store, static/ (dashboard)
docs/       laptop setup, architecture, data sources
tests/
Dockerfile  compose.yaml
```

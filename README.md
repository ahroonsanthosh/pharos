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
comes from MAVLink (ArduPilot/PX4), DJI subtitle files or manual entry;
recorded footage can also pair with an ArduPilot log. When
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
pharos analyze cam.mkv --map area.tif --hfov 77 --log flight.bin --log-sync 1.5   # ArduPilot
```

A separate camera and an autopilot log share no clock; `pharos sync` finds
the offset from the footage alone. On a flight with GNSS, `pharos probe`
measures per-frame match quality and calibrates the camera once (mount
misalignment and effective field of view):

```bash
pharos sync cam.mkv --log flight.bin --log-sync 1 --hfov 77 --start 300
pharos probe cam.mkv --map area.tif --hfov 77 --log flight.bin --log-sync 1.5
```

Recorded footage is solved as a whole flight, so a turn or a gap is bridged
from both sides; `--causal` reports only what a live run would have. Give
`--map-accuracy` when the reference map itself may be metres off (e.g. 6 for
NAIP), so reported uncertainties include it.

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
   followed in parallel until one clearly wins over 120 m of flight. A
   confirmed track is the last 40 s of frames solved jointly from visual
   odometry, gyro, compass and map matches, so a look-alike match is
   outvoted; it learns the odometry scale and compass bias as it flies. When
   matches stop, it dead-reckons with an uncertainty that grows honestly and
   looks for the track again only where that leaves the aircraft. Recorded
   footage is solved as a whole flight, bridging gaps from both sides.

Each estimate carries a status: `tracking` (trusted), `coasting` (a
position with an honest uncertainty, not trusted), `converging` (a candidate,
not yet a position) or `lost`. Details and design rationale are in
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
| scenario | avail | fix s | err50 m | err95 m | hdg deg | cover | mislead | gap s | in 3 sd | ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| altitude_bias | 84% | 9.5 | 1.3 | 3.1 | 0.1 | 84% | 0.0% | 10 | 100% | 195 |
| autumn_low_sun | 84% | 9.5 | 2.0 | 4.2 | 0.1 | 84% | 0.0% | 10 | 100% | 217 |
| dusk_haze | 66% | 10.0 | 2.1 | 8.6 | 0.9 | 66% | 0.0% | 10 | 99% | 216 |
| grid_farmland | 85% | 9.0 | 1.6 | 3.5 | 0.2 | 85% | 0.0% | 9 | 100% | 179 |
| kitchen_sink | 68% | 9.5 | 2.3 | 5.3 | 0.6 | 68% | 0.0% | 10 | 100% | 163 |
| land_change | 84% | 9.5 | 2.2 | 3.5 | 0.1 | 84% | 0.0% | 10 | 100% | 195 |
| no_compass_spinning | 84% | 9.5 | 2.0 | 3.9 | 0.1 | 84% | 0.0% | 10 | 100% | 179 |
| nominal | 85% | 9.0 | 1.4 | 2.4 | 0.1 | 85% | 0.0% | 9 | 100% | 167 |
| sensor_errors | 84% | 9.5 | 6.7 | 8.8 | 0.3 | 84% | 0.0% | 10 | 100% | 152 |
| spring_clouds_on_map | 56% | 9.5 | 2.4 | 5.5 | 0.4 | 56% | 0.0% | 17 | 100% | 156 |
| tilted_camera | 85% | 9.0 | 1.2 | 2.5 | 0.1 | 85% | 0.0% | 9 | 100% | 165 |
| winter_snow | 69% | 18.5 | 1.6 | 2.8 | 0.1 | 69% | 0.0% | 18 | 100% | 181 |
<!-- bench:end -->

Columns: share of frames trusted (`tracking`), time to first confirmed
fix, median and 95th-percentile error of trusted positions, median heading
error, accurate coverage (trusted and within 25 m) and the misleading rate
(trusted and further off), both over all frames, the longest stretch without
a trusted position, the share of trusted positions within three reported
sigmas, and median processing time per frame on a loaded 4-core CPU.

No scenario reports a wrong position as trusted, and reported uncertainties
hold. Coverage tops out near 85% because a candidate must cover 120 m of new
ground (about 10 s) before it is trusted. Lower rows report the missing
frames as coasting, not wrong: in `spring_clouds_on_map`, `kitchen_sink` and
`dusk_haze`, clouds on the map, haze and forest leave stretches where too
little matches for a 10 m uncertainty; `winter_snow` needs longer to find a
distinctive start;
`sensor_errors` shows that a 2° camera mount misalignment becomes a ~6 m bias
that no single frame can reveal.

The simulator is still kinder than reality (flat ground, the map and the
drone view share exact geometry). Expect real cross-season match scores to be
lower and treat these numbers as an upper bound until validated on flights.

## Real flights

Three public datasets, all free, each localized from a cold start with no
initial position; GNSS is used only to calibrate a camera once and to score.
Every frame of a flight counts. A frame is **covered** when its position is
trusted and within 25 m of GNSS, and **misleading** when it is trusted and
further off. A trusted position has an uncertainty of at most 10 m and two
independent supporting matches; otherwise the position is still reported, as
coasting, with its honest uncertainty, so the table also scores every
position against its own sigma.

| flight | | covered | misleading | trusted within 3σ | any position within 25 m | any position within 3σ |
|---|---|---:|---:|---:|---:|---:|
| PinPoint 1: farmland, ~105 m, 13 min | live | 52% | 0.3% | 99% | 87% | 99% |
| | whole flight | 86% | 0.4% | 99% | 97% | 99% |
| PinPoint 2: river valley, ~80 m, 15 min | live | 11% | 0.0% | 99% | 43% | 100% |
| | whole flight | 32% | 0.0% | 100% | 57% | 100% |
| Aukerman: 77 photos, ~100 m | live | 19% | 0.0% | 100% | 35% | 100% |
| | whole flight | 35% | 0.0% | 100% | 70% | 100% |

*Live* is what the aircraft would have been told in flight; *whole flight*
is `pharos analyze` on the recording, solved at once. Flight 2 was localized
with the camera calibration from flight 1, out of sample.

The headline is honesty rather than coverage. Reported uncertainties hold
on every flight: 99% or more of positions, trusted or coasting, are within
three sigmas of GNSS, and the misleading rate is at most 0.4%. The previous
version reported 3.4% of flight 2's frames as trusted and wrong, and only a
third of its positions were within their sigma. Flight 2 now carries a
position within 25 m in 57% of its frames (from 33%), most of it coasting.
The cost is trusted coverage where matching was already good: flight 1's
whole-flight coverage fell from 97% to 86% (live, 70% to 52%) and
Aukerman's from 49% to 35%, because positions that were trusted on
optimistic sigmas now coast on honest ones; 97% of flight 1's frames still
carry a position within 25 m.

### Data and workflow

*PinPoint* (Zenodo, CC BY 4.0) is two flights of an ArduPlane VTOL with a
consumer camera (Sony IMX477, 1280×720 at 60 fps) on a pitch/yaw servo mount,
each with its ArduPilot `.bin` log; the camera and autopilot share no clock.
The map is Spain's PNOA orthophoto (IGN, CC BY 4.0, July 2024) over its free
WMS. No openly licensed downward-looking DJI video with its `.SRT` could be
found. *Aukerman* (OpenDroneMap, CC0) is a senseFly eBee survey, photos 4 to
22 s apart, matched against USGS NAIP 2015 (`--map-accuracy 6`: NAIP is good
to about 6 m).

1. `pharos sync` placed flight 1's video 1.54 s after the first GNSS fix
   (published by eye: 1.2 s; odometry steps disagree by 8.2 m at 1.2 s and
   by 1.8 m at 1.54 s). On flight 2 it agreed with the published value.
2. `pharos probe` on flight 1 measured an effective field of view of 76.7°
   (published 72.3°) and a mount offset of 4.4° back, 3.0° right.
3. `pharos analyze` over the cruise part of each flight at 2 frames/s.

### What real footage taught

* **Overlapping frames are one observation.** Consecutive frames see the same
  ground, so a look-alike fools a run of them alike. Counted as independent,
  sixteen such matches in a test outvoted odometry and the solve claimed 3 to
  5 m while 50 m off. Matches less than 40 m apart now share one vote; on flight 1
  this alone cut the whole-flight misleading rate from 1.9% to 0.5%.
* **Lost is not unknown.** A lost track keeps dead reckoning, reported with
  a growing sigma, and the search for a new track is confined to where that
  leaves the aircraft (every third search still covers the whole map, in
  case the lost track was already wrong). An untrusted track that keeps
  rejecting matches pointing to one place gives up its belief and lets that
  place compete for confirmation.
* **Estimator bugs hide behind matches.** Dead reckoning without matches
  exposed four, each of which made the error explode while the sigma
  collapsed: old information was forgotten a little more every frame, a
  missing odometry step at the window's edge erased the position, the
  velocity prior pulled the odometry scale to zero, and a full step could
  push the scale through zero. Six minutes of pure dead reckoning on either
  flight now stays within three sigmas from every start tried.
* **Wind turns with the aircraft.** The crab angle flips from one leg to the
  next (yaw minus track went from +19° to −14° in one turn), so body-frame
  velocity changes in every turn. A stiffer velocity prior left dead
  reckoning 36 m off 10 s into a bank with a 16 m sigma; the looser one,
  21 m.
* **ArduPilot logs nonsense at gimbal lock.** A servo mount pointing straight
  down logs roll −90, yaw −90 for a camera at roll 0, yaw 3. Those samples
  inherit roll and yaw.
* **Ridges.** Along a straight field edge the correlation peak is a line;
  each match reports its peak's spread, and the estimator uses it.
* **Sync is attitude, and so are banks.** 0.3 s of clock error gives every
  frame the bank angle of another moment. Half the confident matches above
  20° of bank were 20 m or more off, so only odometry runs beyond 30°, or
  beyond 20° while rolling.
* **Photo GPS altitude was above the ellipsoid, not sea level** (34 m, a
  50% scale error at 100 m), and the eBee's camera attitude convention had to
  be read from the data. The map whose season matches the flight wins.
* **Tried and rejected:** averaging aligned score maps over several frames
  (flight 1 live coverage 70% to 62%, misleading 0.5% to 1.4%), and a 12 m
  trust threshold (flight 2 misleading 6.3%).

## Next steps

* **Flight 2's long gaps are acquisition.** Over its valley the true place
  stands out from noise in only 43% of frames (`pharos probe`), and minutes
  pass without a candidate winning. Better features for low texture
  (learned matchers behind the `Matcher` protocol) are the lever.
* **Terrain in banks.** On flight 2 the ground a banked camera sees is 25 m
  (rms) from the height under the aircraft; a free DEM (Copernicus GLO-30)
  would correct odometry there. Near nadir it is only 2 to 4 m, so it is not
  why flight 2 fails to acquire.
* **Lens distortion:** consumer cameras need undistorting; it matters most at
  the image edges a banked view relies on.
* **More real flights:** a DJI MP4 + SRT gives ground truth for free.

## Layout

```
src/pharos/
  geometry.py  geomap.py  types.py  recording.py  config.py  cli.py
  footage.py  analyze.py  telemetry.py  sync.py
  vision/   rectify, features, correlate, matcher, odometry
  nav/      filter, window, localizer
  sim/      world, appearance, trajectory, flight, scenario, scenarios/*.yaml
  eval/     metrics, plot, bench, export
  server/   app, feeds, store, static/ (dashboard)
docs/       laptop setup, architecture, data sources
tests/
Dockerfile  compose.yaml
```

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
| scenario | avail | fix s | err50 m | err95 m | hdg deg | <10 m | wrong | ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| altitude_bias | 84% | 9.5 | 1.9 | 4.4 | 0.1 | 84% | 0.0% | 335 |
| autumn_low_sun | 84% | 9.5 | 2.4 | 5.6 | 0.2 | 84% | 0.0% | 358 |
| dusk_haze | 84% | 9.5 | 2.9 | 21.9 | 0.8 | 72% | 0.0% | 344 |
| grid_farmland | 85% | 9.0 | 2.0 | 4.6 | 0.2 | 85% | 0.0% | 251 |
| kitchen_sink | 60% | 9.5 | 3.0 | 7.3 | 1.0 | 60% | 0.0% | 321 |
| land_change | 84% | 9.5 | 2.6 | 5.0 | 0.2 | 84% | 0.0% | 319 |
| no_compass_spinning | 84% | 9.5 | 2.2 | 3.6 | 0.1 | 84% | 0.0% | 315 |
| nominal | 85% | 9.0 | 1.9 | 3.9 | 0.1 | 85% | 0.0% | 299 |
| sensor_errors | 84% | 9.5 | 7.0 | 9.7 | 0.1 | 82% | 0.0% | 298 |
| spring_clouds_on_map | 62% | 9.5 | 2.8 | 6.9 | 0.5 | 62% | 0.0% | 323 |
| tilted_camera | 85% | 9.0 | 1.7 | 3.6 | 0.1 | 85% | 0.0% | 286 |
| winter_snow | 69% | 18.5 | 2.2 | 4.3 | 0.2 | 69% | 0.0% | 338 |
<!-- bench:end -->

Columns: share of frames in `tracking`, time to first confirmed fix, median
and 95th-percentile position error while tracking, median heading error,
share of all frames within 10 m, share of tracking frames more than 25 m off
(confidently wrong), and median processing time per frame on a fully loaded
4-core CPU.

Availability tops out near 85% because a candidate must cover 120 m of new
ground (about 10 s) before it is trusted. Lower rows are informative:
`spring_clouds_on_map` and `kitchen_sink` lose frames to the stricter rules
real video required (a coasting track must fly 40 m on consecutive stronger
matches before it is trusted again, and is dropped after 30 s); they report
those frames as coasting, not wrong. `winter_snow` needs longer to find a
distinctive start; `sensor_errors` shows that a 2° camera mount
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
| NAIP 2015 (July, 1 m) | 36% | 10 m | 20 m | 0% |
| NAIP 2019 (October) | 18% | 11 m | 15 m | 0% |
| NAIP 2023 (May) | 26% | 12 m | 19 m | 0% |

Errors are measured against the photos' own GPS (about 1.5 m accurate). How
it got here, and what this footage taught:

* The first version confirmed look-alike places 0.7 to 1.4 km away. Real
  photos match far more weakly than synthetic ones (0.2 to 0.6 vs 0.8 to
  0.9), so fixed score thresholds were unsafe. Matches are now judged
  against the noise of their own search window.
* Visual odometry between photos replaced the constant-velocity guess (it
  needed windows up to 150 m with 4 to 13 s between photos). Its error
  differs by platform (a banking fixed-wing is far noisier than a gimbal), so
  confirmed tracks calibrate it from their own innovations.
* Tracks were lost at every tight turn, where photos are up to 22 s apart
  and do not overlap. A confirmed track now keeps searching around its
  prediction while that is still feasible (reported as coasting, re-trusted
  only after consecutive matches), and a lost track is re-acquired within the
  area the aircraft can have reached, with fewer confirmations needed.
  Together with odometry this raised coverage from 0 to 4% to 25 to 43%
  (18 to 36% after the stricter re-trust rules the video below needed).
* Per frame, the true location stands out from noise in 55 to 73% of frames
  (`pharos probe`); feature-scale variations tuned on this footage did not
  beat the defaults.
* Photo GPS altitude was above the WGS84 ellipsoid, not sea level: 34 m, a
  50% scale error at this altitude. The eBee's camera attitude is recorded,
  but its sign convention had to be determined from the data.
* The map whose season matches the flight wins, even though it predates a
  house and path visible in the photos.

This survey is a hard case (low altitude, tight loops, mostly lawn and
trees). Coverage is the open problem; see below.

## Real video: a fixed-wing over Spanish farmland

No openly licensed downward-looking DJI video with its `.SRT` could be found,
so the test uses the *PinPoint* validation dataset (Zenodo, CC BY 4.0): two
flights of an ArduPlane VTOL with a consumer camera (Sony IMX477, 1280×720 at
60 fps) on a pitch/yaw servo mount, each with its ArduPilot `.bin` log. The
camera and autopilot share no clock. The reference map is Spain's PNOA
orthophoto (IGN, CC BY 4.0, flown July 2024), fetched over its free WMS.

The workflow is the one a user would follow, from a cold start with no
initial position; GNSS from the log is used only to calibrate once and to
score:

1. `pharos sync` placed the video 1.54 s after the first GNSS fix. The
   dataset publishes 1.2 s, set by eye; at 1.2 s odometry steps disagree by
   8.2 m, at 1.54 s by 1.8 m. On flight 2 it agreed with the published value.
2. `pharos probe` on flight 1 measured an effective field of view of 76.7°
   (published 72.3°; their own by-eye estimate was 78°) and a mount offset of
   4.4° backward, 3.0° right.
3. `pharos analyze` over the cruise part of each flight, 2 frames/s:

| flight | camera calibration | frames trusted | median error | 95th pct | confidently wrong |
|---|---|---:|---:|---:|---:|
| 1: farmland, ~105 m, 13 min | from flight 1 | 63% | 5.6 m | 24 m | 4.7% |
| 2: river valley, ~80 m, ~30 m relief | from flight 1 | 16% | 9.2 m | 44 m | 16% |

Flight 1 works: the track follows the survey lines and coasts through the
turns. It is not yet safe: 4.7% of trusted frames are 25 to 69 m off,
mostly in the seconds after a steeply banked turn. Flight 2, lower over hilly
ground, mostly fails; flat-ground rectification and odometry are wrong by up
to a third of the height there, and a DEM is the missing piece (see next
steps). What these flights taught:

* **ArduPilot logs nonsense at gimbal lock.** A servo mount pointing straight
  down logs roll -90, yaw -90 for a camera at roll 0, yaw 3: turned around in
  half the samples of flight 2. Those samples now inherit roll and yaw.
* **Ridges.** Along a straight field edge the correlation peak is a line,
  and a second bump on it 70 m up-track scored as well as the truth. Each
  match now reports its peak's spread and the filter uses it.
* **Sync is attitude.** 0.3 s of clock error gives every frame the bank
  angle of another moment; odometry error doubled.
* **Banked turns.** Half the confident matches above 20° tilt were 20 m or
  more off (likely lens distortion and attitude timing, which matter most
  there), so only odometry
  runs beyond 30°, with an error model that grows as 1/cos²(tilt).
* **Look-alikes come in runs.** Two consecutive matches 0.5 s apart are one
  observation; re-trusting a track now takes 40 m of flight, and weak matches
  can no longer keep a wrong coasting track alive.

## Next steps

* **Multi-frame evidence:** accumulate aligned score maps over several frames
  and stitch frames into larger templates; this is what low-altitude footage
  needs most.
* **Feature scales tuned on real imagery** (canopy and lawn texture, NAIP's
  softer resolution), validated on the Aukerman set and future flights.
* **Terrain relief** via a free DEM (Copernicus GLO-30): rectification and
  odometry assume flat ground, which is what limits the hilly flight below.
* **Lens distortion:** consumer cameras need undistorting; matching is
  turned off in steep banks partly because the distorted image edges then
  dominate.
* **More real flights:** a DJI MP4 + SRT gives ground truth for free; no
  openly licensed downward-looking DJI video with its SRT was found online.

## Layout

```
src/pharos/
  geometry.py  geomap.py  types.py  recording.py  config.py  cli.py
  footage.py  analyze.py  telemetry.py  sync.py
  vision/   rectify, features, correlate, matcher, odometry
  nav/      filter, localizer
  sim/      world, appearance, trajectory, flight, scenario, scenarios/*.yaml
  eval/     metrics, plot, bench, export
  server/   app, feeds, store, static/ (dashboard)
docs/       laptop setup, architecture, data sources
tests/
Dockerfile  compose.yaml
```

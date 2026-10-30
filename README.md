# Pharos

[![ci](https://github.com/ahroonsanthosh/pharos/actions/workflows/ci.yml/badge.svg)](https://github.com/ahroonsanthosh/pharos/actions/workflows/ci.yml)

**GPS-free navigation for drones.** Pharos works out where a drone is by
matching its downward camera against free satellite and aerial maps. When
GPS is jammed, spoofed or just missing, it keeps going.

It runs on a laptop CPU. No trained models, no paid maps, no special
hardware. Plug in any video feed with MAVLink (ArduPilot, PX4) or DJI
telemetry, live or recorded.

![Pharos's estimate from video alone over the GPS track](docs/img/flight1.jpg)

<sub>A real flight: Pharos's estimate from the video alone (blue trusted,
orange coasting with its uncertainty) over the GPS track (green).
Orthophoto: PNOA © Instituto Geográfico Nacional, CC BY 4.0.</sub>

## Why it matters

GPS jamming and spoofing are now routine, and most drones are lost without
it. Visual navigation exists, but it usually needs expensive maps, a GPU or
a model trained on the area. Pharos needs none of those.

Just as important, it knows when it doesn't know. Every position comes with
an honest uncertainty, and Pharos only calls a position "trusted" when
independent evidence backs it up. A navigation system that is confidently
wrong is worse than none.

## Results

On public real-world flights, with GPS used only to score the result:

| Flight                                  | Trusted within 25 m (after flight) | Live, in flight | Trusted but wrong |
| --------------------------------------- | ---------------------------------: | --------------: | ----------------: |
| Farmland survey, fixed-wing, 13 min     |                                92% |             63% |              none |
| River valley survey, fixed-wing, 15 min |                                91% |             52% |              none |

In simulation, across 31 scenarios (snow, haze, low sun, clouds on the map,
Irish bog and uplands, a frozen video link, GPS jammed at 50 m, a last GPS
fix that was already spoofed and more), Pharos never trusts a wrong
position. Reproduce it with `pharos bench`. The full benchmark table, every
real flight (including DJI footage from 16 to 92 m) and what was tried and
dropped are in [docs/results.md](docs/results.md).

## Try it

```bash
git clone https://github.com/ahroonsanthosh/pharos.git && cd pharos
docker compose up            # then open http://localhost:8000
```

The dashboard takes almost any feed: a DJI controller or OBS over RTMP/SRT,
RTSP/HLS/UDP streams, a capture card or webcam, recorded video with DJI
`.SRT` files, or a built-in demo flight. If the drone reports GPS, it is
drawn beside Pharos's estimate so you can see the accuracy for yourself.

**Start from GPS** takes one fix at takeoff and never uses GPS again, so
later jamming can't pull it off course. The **GNSS check** then flags a
receiver that looks confident but is drifting away, which is what a spoofed
one does. Setup for each source: [docs/laptop.md](docs/laptop.md).

## Analyse recorded footage

```bash
pharos analyze DJI_0042.MP4 --map area.tif --hfov 70     # uses the .SRT beside the video
pharos analyze photos/ --map area.tif                    # photo sets with EXIF/XMP
pharos analyze old.mp4 --map area.tif --hfov 70 --altitude 120   # no telemetry at all
```

Outputs a track as GeoJSON, KML (opens in Google Earth) and CSV, plus a
summary. If the footage has its own GPS, the summary scores Pharos against
it. Run `pharos --help` for syncing a separate camera to an autopilot log,
calibrating a camera, and starting from a known position.

## How it works

1. **Rectify.** Project each camera frame onto the ground, using height and
   tilt, through a terrain model on uneven ground.
2. **Describe.** Turn frame and map into edge maps (roads, field
   boundaries, buildings, shorelines) that ignore colour, season and
   shadows.
3. **Match.** Score every position on the map at once with FFT
   cross-correlation, across headings and scales.
4. **Track.** Follow several candidates until one clearly wins, then fuse
   map matches with visual odometry, gyro and compass. When matches stop,
   dead-reckon with a growing uncertainty.

Any north-up orthophoto works as a map: USGS NAIP, national programmes like
IGN, swisstopo and PDOK, OpenAerialMap or Sentinel-2. More in
[docs/architecture.md](docs/architecture.md).

## Develop

```bash
pip install -e ".[dev,server,geo]"
pytest -q                          # tests
pharos bench -j 4 -o runs/bench    # run every simulated scenario
pharos serve --open                # dashboard without Docker
```

## What's next

- A real flight over Irish pasture or bog (those scenarios are simulated so far)
- Faster first fix at low altitude
- Learned features as an optional matcher
- Finer terrain models and lens distortion correction

## License

Copyright (C) 2026 Ahroon Santhosh. Free software under the
[GNU AGPL v3.0](LICENSE). A commercial licence for proprietary use is
available from Heloria Labs: contact
[@ahroonsanthosh](https://github.com/ahroonsanthosh). Outside contributions
are accepted under a contributor licence agreement.

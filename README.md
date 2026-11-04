<p align="center">
  <img src="docs/img/logo.png" alt="Pharos" width="420">
</p>

<h1 align="center">Pharos 0.10</h1>

<p align="center">
  <a href="https://github.com/ahroonsanthosh/pharos/actions/workflows/ci.yml"><img src="https://github.com/ahroonsanthosh/pharos/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
</p>

I built Pharos because I wanted a drone to know where it is when GPS stops
telling the truth. It looks down through the camera, compares what it sees
with free satellite and aerial maps, and works out its position from that
alone. Jammed, spoofed or just missing GPS, it keeps going.

It runs on a normal laptop CPU. No trained models, no paid maps, no special
hardware. You give it a video feed plus MAVLink (ArduPilot, PX4) or DJI
telemetry, live or recorded, and it gives you a position.

![The Pharos dashboard tracking a demo flight](docs/img/live.jpg)

## What I cared about most

Being honest. Every position comes with an uncertainty, and Pharos only
calls a position "trusted" when independent evidence backs it up. A
navigation system that is confidently wrong is worse than having none, so
that rule shaped almost every decision in here.

On public real flights (GPS used only to score it):

| Flight                                  | Trusted within 25 m | Live, in flight | Trusted but wrong |
| --------------------------------------- | ------------------: | --------------: | ----------------: |
| Farmland survey, fixed-wing, 13 min     |                 92% |             63% |              none |
| River valley survey, fixed-wing, 15 min |                 91% |             52% |              none |

Across 31 simulated scenarios (snow, haze, low sun, Irish bog and uplands,
a frozen video link, GPS jammed at 50 m, a spoofed starting fix and more)
it never trusts a wrong position. You can check that yourself with
`pharos bench`. The full numbers, the DJI footage I tested on and the ideas
I tried and threw away are in [docs/results.md](docs/results.md).

![A real flight: Pharos from video alone over the GPS track](docs/img/flight1.jpg)

<sub>A real flight: Pharos from the video alone (blue trusted, orange
coasting) over the GPS track (green). Orthophoto: PNOA © Instituto
Geográfico Nacional, CC BY 4.0.</sub>

## Try it

```bash
git clone https://github.com/ahroonsanthosh/pharos.git && cd pharos
docker compose up            # then open http://localhost:8000
```

Hit **New feed** and pick the demo flight to see it work straight away. It
also takes a DJI controller or OBS over RTMP/SRT, RTSP/HLS/UDP streams, a
capture card or webcam, or recorded video with DJI `.SRT` files. If the
drone reports GPS, I draw it next to the estimate so you can judge the
accuracy for yourself. Setup for each source is in
[docs/laptop.md](docs/laptop.md).

![Adding a live feed](docs/img/feed.jpg)

To put a public demo online for free, see [docs/deploy.md](docs/deploy.md).

**Start from GPS** takes one fix at takeoff and never uses GPS again, so
jamming later on can't drag it off course. The **GNSS check** then flags a
receiver that looks confident but is drifting away, which is exactly what a
spoofed one does.

## Recorded footage

```bash
pharos analyze DJI_0042.MP4 --map area.tif --hfov 70
pharos analyze photos/ --map area.tif
pharos analyze old.mp4 --map area.tif --hfov 70 --altitude 120
```

You get the track as GeoJSON, KML (opens in Google Earth) and CSV, plus a
summary that scores Pharos against the footage's own GPS if it has any.
`pharos --help` covers syncing a camera to an autopilot log, calibrating a
camera and starting from a known position.

## How it works

1. **Rectify** each frame onto the ground using height, tilt and a terrain
   model.
2. **Describe** frame and map as edges (roads, field boundaries, buildings,
   shorelines) so colour, season and shadows matter less.
3. **Match** every position on the map at once with FFT cross-correlation,
   across headings and scales.
4. **Track** several candidates until one clearly wins, then fuse map
   matches with visual odometry, gyro and compass. When matches dry up it
   dead-reckons with a growing uncertainty.

Any north-up orthophoto works as a map: USGS NAIP, IGN, swisstopo, PDOK,
OpenAerialMap or Sentinel-2. More detail in
[docs/architecture.md](docs/architecture.md).

## Working on it

```bash
pip install -e ".[dev,server,geo]"
pytest -q
pharos bench -j 4 -o runs/bench
pharos serve --open
```

Next on my list: a real flight over Irish pasture or bog, learned features
as an optional matcher, and lens distortion correction. A faster first fix
for low flights is in as an opt-in: in simulation it gets a position in
about 6 s instead of 10, and at 40 m over pasture where it used to find
none, but I want it to hold on real flights before it becomes the default
([details](docs/results.md#confirming-by-frame-lengths-pharos-09-off-by-default)).

## License

Copyright (C) 2026 Ahroon Santhosh. Free software under the
[GNU AGPL v3.0](LICENSE). If you need it for proprietary use, a commercial
licence is available from Heloria Labs: get in touch through
[@ahroonsanthosh](https://github.com/ahroonsanthosh). Outside contributions
are accepted under a contributor licence agreement.

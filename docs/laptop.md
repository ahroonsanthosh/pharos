# Running Pharos on a laptop

Pharos runs as a small local service with a dashboard in your browser. It
needs no internet once installed, and all of it is free.

## Start it

### With Docker (recommended: one command on macOS, Windows and Linux)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   (on Linux, Docker Engine). It is free for personal use, education and
   small businesses.
2. In a terminal:

   ```bash
   git clone https://github.com/ahroonsanthosh/pharos.git
   cd pharos
   docker compose up
   ```

   The first start builds the image (a few minutes). Later starts take
   seconds.
3. Open **http://localhost:8000**.

Stop with `Ctrl+C` (or `docker compose down`). Maps and uploaded videos are
kept in `pharos-data/` next to the code.

The stack runs two containers: **pharos** (dashboard, API, localization) and
**MediaMTX** (receives video pushed over RTMP, SRT or RTSP).

| port | used for |
|---|---|
| 8000/tcp | dashboard and API |
| 1935/tcp | RTMP push (DJI Fly / DJI Pilot live streaming, OBS) |
| 8890/udp | SRT push |
| 8554/tcp | RTSP push or pull |
| 14551/udp | MAVLink telemetry forwarded from a ground station |

If a drone controller or another computer sends to the laptop, allow these
ports through the laptop's firewall.

### Without Docker

Python 3.11 or newer:

```bash
pip install -e ".[server,geo]"
pharos serve --open            # add --host 0.0.0.0 to accept other devices
```

To receive pushed streams, also download and run
[MediaMTX](https://github.com/bluenviron/mediamtx/releases); its defaults use
the ports above. Without Docker, cameras and capture cards plugged into the
laptop can also be opened directly ("Camera or capture card (direct)").

## First run: the demo

Click **New feed**, keep **Demo flight**, press **Start**. A simulated flight
and its map are generated (about 15 s). The dashboard then shows the drone's
view, the estimated track (orange while converging, blue once confirmed) and
the simulated GPS track in green for comparison.

## Your own area

1. Get a free orthophoto of the area as GeoTIFF (see [data.md](data.md)).
   Cover the whole flight area plus a margin; a few km across is ideal.
2. **New feed → Upload a map**: name it, choose the `.tif`, press Upload.
   Imagery finer than 1 m is resampled on import.

## Connecting a feed

Everything below goes through **New feed**. For every live source, set:

* **Horizontal FOV** of the video as streamed. Manufacturers often publish
  the *diagonal* FOV. Convert with
  `hfov = 2 · atan(tan(dfov / 2) · w / √(w² + h²))`; for a 4:3 sensor, a
  diagonal of 82° is about 70° horizontal. A wrong FOV looks like a wrong
  altitude and degrades accuracy.
* **Rate**: 1 to 2 Hz is plenty; higher costs CPU without helping.

### DJI, live

1. Put the laptop and the remote controller on the same Wi-Fi (or connect
   the laptop to the controller's hotspot) and find the laptop's IP address.
2. In Pharos: **Stream pushed to this laptop**, stream name `drone`. The form
   lists the URL to use.
3. In DJI Fly or DJI Pilot 2, open live streaming, choose **RTMP**, and enter
   `rtmp://<laptop-ip>:1935/drone`. Not every model and app version offers
   RTMP live streaming. If yours doesn't, use an HDMI capture card (below) or
   a recorded video.
4. The RTMP stream carries no telemetry, so choose **Manual** telemetry and
   enter the height above ground. Update it from the dashboard if you climb
   or descend.

### DJI, recorded

Turn on **Video Subtitles** in the DJI camera settings. Each video then has
a matching `.SRT` file with height, gimbal angle and GPS for every frame.
Choose **Recorded video**, select the MP4 and the SRT. The SRT's GPS is
drawn in green and gives you the accuracy of every flight.

### ArduPilot / PX4 (MAVLink)

* **Telemetry:** choose **MAVLink**, port `14551`, then forward MAVLink from
  the ground station to the laptop:
  * QGroundControl: *Application Settings → MAVLink → MAVLink forwarding*,
    host `127.0.0.1:14551`.
  * Mission Planner: `Ctrl+F` → *MAVLink* → *UDP Client*, `127.0.0.1`, port `14551`.

  Pharos uses rangefinder height when a downward rangefinder reports, else
  height above home corrected for terrain (the home position comes over
  MAVLink), plus gimbal pitch for camera tilt. Pick **Fixed** mount if the
  camera is bolted to the frame instead of a gimbal.
* **Video:** a companion computer's RTSP/UDP stream as **Stream URL**, or
  push RTMP/SRT to the laptop as above.
* **Video delay:** the video arrives later than the telemetry, typically 0.5
  to 1.5 s through a streaming pipeline. Set it once on a flight that has
  GPS: pick the value that minimizes *Error vs GNSS*.
* **Start from the first GPS fix:** tick this when the aircraft has GPS at
  takeoff and may lose it later. Pharos then begins from that fix instead of
  searching the whole map, and never uses GPS again, so later jamming or
  spoofing cannot pull its position away.

### Archived footage and photo sets

Choose **Recorded video or photos** and select either one video or many
photos. Photos bring their own GPS, time and, for DJI and senseFly, height
and heading. Recorded footage runs as fast as the laptop allows, not in real
time. From a terminal the same thing is `pharos analyze` (see the README).

Every feed, live or recorded, can be downloaded as **GeoJSON**, **KML**
(Google Earth) or **CSV** from the links under the readout.

### HDMI capture card or webcam

Choose **Camera or capture card (through this browser)**. The browser
captures the device and sends frames to Pharos, so this works under Docker on
every OS. A cheap USB HDMI capture dongle on the remote controller's HDMI
output turns any drone with video out into a Pharos feed.

### Anything else

OBS, ffmpeg or any encoder can push RTMP or SRT to the laptop, and any RTSP,
RTMP, HLS or UDP URL can be pulled directly. For example, to replay a file
as a live stream:

```bash
ffmpeg -re -i flight.mp4 -c:v libx264 -tune zerolatency -f flv rtmp://localhost:1935/drone
```

## Reading the dashboard

* **converging**: candidate positions are being checked against each other
  as the drone moves. Don't trust these yet.
* **tracking**: one location won over at least 120 m of flight. Positions are
  shown with a 1-sigma uncertainty circle.
* **coasting**: was tracking, but recent frames did not match; the position
  is dead reckoning and its uncertainty is growing.
* **lost**: no confident match (over featureless ground, or before the first
  fix). Pharos says so instead of guessing.
* **Error vs GNSS** appears whenever the aircraft reports its own GPS.
* **GNSS check** compares that GPS with pharos's own position while pharos
  trusts it. *disagrees* (the status bar turns red) means GPS has been
  further off than both uncertainties allow for more than 3 s: a spoofed
  receiver looks like this, since it keeps a confident fix while it walks
  away. A jammed receiver simply loses its fix.

## Limits to know

* Over terrain, each frame is projected through the elevation model, which
  at 30 m is coarser than cliffs or buildings: those still add error.
* Matching needs visible structure: roads, field edges, buildings, water.
  Over forest, open water or uniform snow it will report lost until
  structure returns.
* Height above ground matters. Barometric height is relative to takeoff, so
  the first time a feed with such heights runs on a map, Pharos downloads the
  free Copernicus terrain model for it (about 40 MB per 1 by 1 degree tile,
  kept in the data folder) and corrects every height for the ground below.
  Without internet access then, it learns a constant scale error instead,
  which does not follow terrain that changes along the track.

# Data: maps and flight recordings

Everything here is free of charge. Licences are noted where they impose
conditions such as attribution; check each provider's current terms before
redistributing imagery.

## Reference maps

Pharos needs a **north-up orthophoto in a projected CRS measured in metres**
(UTM is ideal). Load it with `GeoMap.from_geotiff` (`pip install pharos[geo]`)
or save it as PNG with a JSON sidecar (see `GeoMap.save`).

Resolution should be close to or finer than what the drone sees at the
matching scale. The default matching level is 1 m/px, so 0.1 to 1 m imagery is
ideal. 10 m imagery only works from several kilometres up.

| source | coverage | resolution | licence / cost |
|---|---|---|---|
| USGS **NAIP** | contiguous US | 0.6 to 1 m, 4-band | public domain; free on AWS Open Data and Microsoft Planetary Computer |
| IGN **BD ORTHO** | France | 20 cm | Etalab Open Licence 2.0 (attribution) |
| IGN **PNOA** | Spain | 25 cm, free WMS (4096 px per request) | CC BY 4.0 |
| **basemap.at** Orthofoto | Austria | 30 cm, free WMTS (Web Mercator tiles) | CC BY 4.0 |
| swisstopo **SWISSIMAGE 10 cm** | Switzerland | 10 cm | free since 2021 (attribution) |
| PDOK **Luchtfoto** | Netherlands | 8 to 25 cm | CC-BY 4.0 / CC0 |
| Dataforsyningen **orthophotos** | Denmark | 12 cm | free (attribution) |
| German state geoportals (NRW, Bavaria, Berlin, ...) | per state | 10 to 20 cm | dl-de/zero or dl-de/by |
| Maa-amet **orthophotos** | Estonia | 25 cm | free (attribution) |
| **OpenAerialMap** | patchy, worldwide | varies | CC-BY 4.0 |
| Copernicus **Sentinel-2** | worldwide | 10 m | free and open; only for high altitude |

**Ireland** has no free high-resolution orthophoto: Tailte Éireann's aerial
imagery (the Republic) is licensed, OSNI's (Northern Ireland) is sold apart
from a 10 cm strip along the Northern Irish coast (OpenDataNI), and the EU's
2 to 4 m Copernicus mosaic can only be viewed, not downloaded. Two free routes
remain: map the area yourself with the drone (fly a survey grid once, with
GNSS, and build an orthophoto with the free OpenDroneMap; pharos reads its
GeoTIFF directly), or license the national imagery for the area you operate
in. For terrain, Geological Survey Ireland publishes LiDAR elevation models
(1 to 2 m, CC BY 4.0) where it has flown; pass one with `--dem`.

Avoid Google, Bing and Esri basemap tiles. They are free to view, but their
terms do not allow bulk download, offline caching or use as a navigation
reference.

Typical preparation with GDAL (free):

```bash
# mosaic tiles, reproject to the local UTM zone at 0.5 m, compress
gdalbuildvrt area.vrt tiles/*.tif
gdalwarp -t_srs EPSG:32632 -tr 0.5 0.5 -r average -co COMPRESS=DEFLATE area.vrt area_utm.tif
```

Prefer imagery from the same season as the flights when there is a choice.
The matcher tolerates seasonal change, but the margin shrinks (see the
benchmark).

## Terrain

Barometric heights are above the takeoff point, not the ground, so pharos
also needs the ground's elevation under the map. `pharos analyze`, `probe`
and live feeds fetch it automatically when the telemetry's heights are above
takeoff: **Copernicus GLO-30**, a 30 m surface model of the whole world (ESA,
free, no account, served from AWS Open Data). Tiles are 1 by 1 degree, about
40 MB, downloaded once into `~/.cache/pharos` (or `$PHAROS_CACHE`; the live
service keeps them in its data folder). Over open sea there are no tiles and
the elevation is 0.

`--dem file.tif` uses any other elevation GeoTIFF instead, in any CRS, such
as a national lidar model at 1 m. GLO-30 and most national surface models
include trees and buildings, which over a forest is right: the camera sees
the canopy. At 30 m, though, GLO-30 smooths a canopy down: over a forest west
of Vienna it stood only 3 m above the ground Airdata reports, under crowns
perhaps 20 m tall, so a DJI flying 38 m above the ground measured its steps
1.5 to 2 times too long. Low over forest, give a finer surface model with
`--dem`. `--no-terrain` takes heights above takeoff as heights above
ground, as if the ground were level.

## Flight recordings

A recording is a directory (see `pharos.recording`):

```
recording/
  camera.json       {"width", "height", "fx", "fy", "cx", "cy"} for undistorted frames
  telemetry.jsonl   one line per frame to localize
  frames/*.jpg      or a single video.mp4
  truth.jsonl       optional reference positions for evaluation
```

Each telemetry line holds `t` (seconds), either `image` (path) or `frame`
(index into `video.mp4`), plus `altitude_m` (height above ground),
`yaw_deg` (compass, or `null`) and `pitch_deg`/`roll_deg` (camera tilt from
nadir).

The localizer needs only 1 to 5 frames per second. List every Nth video
frame in `telemetry.jsonl`; the rest are skipped while decoding.

### From real hardware

`pharos analyze` and `pharos probe` read recorded footage directly; the
recording format above is only needed for simulated flights.

* **Video:** anything OpenCV/FFmpeg decodes (mp4, mov, mkv). Live streams and
  capture cards are covered in [laptop.md](laptop.md). `--start`/`--end`
  skip takeoff and landing.
* **Telemetry:** DJI `.SRT` subtitle files (picked up next to the video),
  ArduPilot DataFlash logs (`--log flight.bin`), or live MAVLink. PX4 `.ulg`
  logs are not read yet.
* **Syncing a log to a video:** camera and autopilot rarely share a clock.
  `--log-sync` is the time from the log's first 3D GNSS fix to the first
  video frame. Guess it to within a second or two (from the takeoff, say),
  then `pharos sync video.mkv --log flight.bin --log-sync GUESS --hfov H
  --start S` refines it from two minutes of footage without using GNSS.
  Getting it right matters: 0.3 s off doubled the odometry error on a real
  fixed-wing flight, because every frame got the bank angle of another moment.
* **Calibration, once per camera:** run `pharos probe` on a flight with
  GNSS. Its `calibration` block gives the camera's mount misalignment
  (`--mount-offset FORWARD RIGHT`, degrees) and the scale error, both as
  height ratio and as the equivalent field of view (`--hfov`). Lens
  distortion is not modelled; undistort strongly distorted footage first.
* **Ground truth for free:** the aircraft's own GNSS is never given to the
  localizer, only used to score it, so every real flight is also an
  evaluation run.
* **Height above ground:** barometric height is relative to takeoff, so it
  is wrong wherever the ground is higher or lower. With [terrain](#terrain)
  pharos turns it into height above the ground at its own position estimate,
  which needs the takeoff point's elevation: from `--takeoff LAT LON`, from
  the telemetry when the recording starts on the ground (or MAVLink's home
  position), and otherwise learned from the first confirmed track. On two
  real flights this agreed with ArduPilot's own terrain database to a median
  of 1 to 2 m. From an ArduPilot log pharos takes the barometric height and
  does its own terrain lookup, which needs no GNSS and covered more of a real
  flight. `--log-height terrain` uses the autopilot's terrain database
  instead (logged with `TERRAIN_ENABLE`, looked up at the autopilot's GNSS
  position), which is also the fallback when pharos has no terrain.
* **DJI specifics:** the `.SRT`'s `rel_alt` is height above takeoff even on
  terrain-following missions: on a real Mavic 3T survey it read 46 to 112 m
  while the drone held 50 to 63 m above the ground. With terrain, pharos
  brought that to within 3.5 m (median) of Airdata's height above ground.
  Its `abs_alt` is above the WGS84 ellipsoid (about 47 m above sea level in
  Austria), not sea level. Phantom 4 subtitles (`RTK (lon, lat, alt)` or
  `GPS (...)`, `H` above home, `G.PRY` gimbal angles) are read too.
  `gb_yaw` is the camera's compass heading, so a
  drone flying sideways on a mission with a fixed heading is handled. A video
  started in the air does not show where the drone took off; the first row
  of an Airdata CSV export does (`--takeoff LAT LON`).
* **Flying for pharos:** the more ground a frame shows, the better it
  matches. Fly 100 m or more above the ground with the gimbal straight down,
  over ground with edges (fields, roads, buildings, water) rather than
  continuous forest, and give it a straight stretch of 150 m or more to
  confirm a first position. Below about 80 m, match at `--map-gsd 0.5`.
  How much height matters, over simulated Irish ground with an 80° lens (a
  frame 200 m wide at 120 m, 67 m at 40 m), as the share of a two-minute
  flight located: pasture 90% at 110 m, 68% at 60 m, none at 40 m against a
  1 m map and 41% against a 0.5 m one; bog and coast 90%, 85% and 47%;
  uplands with 200 m of relief 77% at 110 m and 44% at 60 m. None trusted
  a wrong position: lower flights find their first position later or not
  at all. Against a 0.5 m map, pasture covered 90%, 85% and 50% at 110, 60
  and 40 m from a cold start, bog and coast 90%, 90% and 62%; the 40 m
  flights took 45 to 60 s to find their first position. Starting instead
  from a last GNSS fix known to 10 m (`--known-start`, or **Start from GPS**
  live), the same flights covered 95 to 96% at every height, again with no
  wrong position trusted: at 40 m the hard part is the first position, not
  keeping it. Real ground is likely harder still low down, where hedges and
  trees a few metres tall look different from the overhead map.

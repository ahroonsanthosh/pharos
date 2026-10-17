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
| swisstopo **SWISSIMAGE 10 cm** | Switzerland | 10 cm | free since 2021 (attribution) |
| PDOK **Luchtfoto** | Netherlands | 8 to 25 cm | CC-BY 4.0 / CC0 |
| Dataforsyningen **orthophotos** | Denmark | 12 cm | free (attribution) |
| German state geoportals (NRW, Bavaria, Berlin, ...) | per state | 10 to 20 cm | dl-de/zero or dl-de/by |
| Maa-amet **orthophotos** | Estonia | 25 cm | free (attribution) |
| **OpenAerialMap** | patchy, worldwide | varies | CC-BY 4.0 |
| Copernicus **Sentinel-2** | worldwide | 10 m | free and open; only for high altitude |

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

* **Video:** anything OpenCV/FFmpeg decodes (mp4, mov, mkv). Live feeds
  (RTSP/UDP URLs, HDMI capture cards as `/dev/video*`) use the same `Frame`
  interface; a streaming source is a small addition next to
  `recording._DiskFrames`.
* **Intrinsics:** calibrate once with a checkerboard (`cv2.calibrateCamera`)
  and undistort frames, or use the manufacturer's values.
* **Telemetry:** ArduPilot/PX4 `.bin`/`.ulg` logs (MAVLink `GLOBAL_POSITION_INT`,
  `ATTITUDE`, `GIMBAL_DEVICE_ATTITUDE_STATUS`) or DJI `.SRT` subtitle files
  carry everything needed. They have to be converted to `telemetry.jsonl` by
  timestamp; those converters are not written yet and are the first job when
  hardware arrives.
* **Ground truth for free:** the GNSS position in the same logs becomes
  `truth.jsonl`, so every real flight is also an evaluation run.
* **Height above ground:** barometric altitude is relative to takeoff. Over
  varying terrain, subtract a DEM (e.g. Copernicus GLO-30, free) or rely on
  the localizer's altitude-scale estimate.

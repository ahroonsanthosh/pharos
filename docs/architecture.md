# Architecture

Pharos estimates an aircraft's absolute position and heading from a downward
camera by matching each frame against a georeferenced orthophoto. It needs no
GNSS and no learned weights, and it runs on a CPU.

```
 frames + telemetry ──▶ rectify ──▶ features ──▶ correlate ──▶ fixes ──▶ localizer ──▶ estimates
 (video, images,        (ground     (orientation (FFT, over     (x, y,     (hypotheses,
  live stream)           plane,      field)       headings &     heading,   Kalman filter,
                         map GSD)                 scales)        score)     state machine)
                                         ▲
 reference map (GeoTIFF / PNG + JSON) ───┘  (features precomputed once per pyramid level)
```

## Modules

| module | role |
|---|---|
| `geometry` | All coordinate conventions, camera model, ground-plane homographies. Read this first. |
| `geomap` | `GeoMap`: north-up orthophoto + affine georeference, pyramid, GeoTIFF loader, WGS84 output. |
| `types` | Plain dataclasses passed between stages (`Frame`, `Telemetry`, `Fix`, `Estimate`, ...). |
| `recording` | On-disk flight format shared by the simulator and real flights. Lazy frame decoding. |
| `vision.rectify` | Warps a frame onto the ground plane at map resolution using altitude and camera tilt. |
| `vision.features` | Appearance-invariant dense orientation field (see below). |
| `vision.correlate` | Masked normalized cross-correlation via FFT; peak finding and sub-pixel refinement. |
| `vision.matcher` | `Matcher` protocol and `CorrelationMatcher` (global and local search). |
| `nav.filter` | Constant-velocity Kalman filter over x, y, vx, vy, heading with innovation gating. |
| `nav.localizer` | Multi-hypothesis acquisition, confirmation and tracking state machine. |
| `sim.*` | Procedural world, seasonal/lighting appearance, trajectories, sensor models, rendering. |
| `eval.*` | Metrics, track plots, scenario benchmark. |
| `telemetry` | Manual, DJI `.SRT` and live MAVLink telemetry sources. |
| `footage`, `analyze` | Recorded videos and photo sets (EXIF, DJI and senseFly XMP) localized offline, with exports. |
| `server.*` | Live service: feed runner (any video source + telemetry), map store, FastAPI app and dashboard. |
| `cli` | `pharos simulate`, `pharos localize`, `pharos bench`, `pharos serve`. |

## Conventions

* World frame: East-North-Up metres. For a GeoTIFF map this is the map's
  projected CRS (e.g. UTM); in simulation it is a local frame tied to a
  lat/lon origin.
* Heading: radians, compass style (0 = north, clockwise positive).
* Camera tilt: `pitch` tilts the optical axis forward from nadir, `roll` to
  the right. Zero is straight down.
* Pixels: `(u, v)` = (column, row), integers at pixel centres (OpenCV).
* Internally everything is radians and metres; files use degrees with `_deg`
  suffixes and metres with `_m`.

## Why these choices

**Rectify first, then match in 2D.** With altitude and camera attitude known,
a frame can be warped to a nadir view at exactly the map's ground sample
distance. What remains between patch and map is a 2D translation, a heading,
and a small scale error from barometric bias. That is a 4-parameter search
that dense correlation handles exhaustively and cheaply, instead of a
homography estimate that can fail silently.

**Orientation features instead of intensity.** Season, sun angle, sensor and
haze change intensities completely, but field boundaries, roads, buildings and
water edges keep their geometry. Each pixel becomes `|g| e^{2iθ}`, averaged
over a small window, normalized by local gradient magnitude, with the
regional mean removed:

* doubling the angle ignores contrast polarity (a field darker than its
  neighbour in summer may be brighter in winter),
* averaging cancels isotropic texture such as tree crowns,
* normalizing removes illumination, contrast and haze,
* removing the regional mean cancels texture whose direction is set by the
  sun (canopy relief shading) or by the season (crop rows).

These were chosen by measurement, not taste: on a fixed world flown under
five appearances, the regional-mean step alone raised cold-start global
top-1 accuracy from 82% to 91%.

**A disk-shaped template.** The matched patch is a disk centred on the
optical-axis ground point. Rotating the heading hypothesis only rotates the
content inside the disk, so the mask is the same for every hypothesis and the
correlator's normalization is computed once per frame. Each extra
heading/scale hypothesis costs one forward and one inverse FFT.

**Global vs local search.** When lost, the matcher searches the whole map
(or a prior region) at half resolution over all headings, or ±45° around the
compass, then refines the best candidates at full resolution. When tracking,
it searches a window sized from the filter covariance, over a few headings
around the prediction and three altitude-scale hypotheses.

**Judge peaks against their own window.** A raw correlation score does not
transfer between domains: synthetic frames score 0.8 to 0.9 at the true place,
real photos 0.2 to 0.6, while the best noise peak in a 150 m search window
already reaches 0.25 to 0.3. Every match is therefore scored by how far its
peak stands above the score distribution of its own window, minus what the
best of that many noise samples would reach (`correlate.peak_excess`). On the
real Aukerman photos this passes 74% of true windows and 10% of wrong ones per
frame, and confirmation needs several in a row.

**Multiple hypotheses before commitment.** Consecutive frames overlap almost
entirely, so a look-alike place that fools one frame fools the next several
too. The localizer keeps up to four candidate locations, tracks each with
local searches, and confirms one only after it has covered at least 120 m of
new ground, once stood out map-wide, and leads every rival over the frames
both have seen. Candidates at the same spot with opposite headings count as
rivals, because line-dominated scenes (survey grids) can match almost as well
turned around. A wrong position reported as
trustworthy is the worst failure mode, so the benchmark tracks the
`confident_wrong` rate separately.

**Self-calibration.** Accepted fixes slowly update the altitude scale (baro
bias) and the compass offset (magnetic bias). The compass is otherwise used
only for frame-to-frame heading changes and as a loose prior when lost.

## Extending

* **Learned matchers** (LightGlue, LoFTR, RoMa) or **learned retrieval**
  (DINOv2/AnyLoc tiles) plug in by implementing `vision.matcher.Matcher`:
  `match(frame, region, scale) -> list[Fix]`. Keep them in an optional
  extra so the core stays torch-free.
* **Visual odometry / IMU** would feed `PoseFilter.predict` with measured
  motion instead of the constant-velocity model.
* **Terrain relief**: rectification assumes flat ground. Over hilly terrain,
  use a DEM to rectify (and use the DEM in the simulator).
* **Large maps**: map features are precomputed in memory per pyramid level.
  A 10 km × 10 km map at 1 m needs tiling or a coarser global level.

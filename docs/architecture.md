# Architecture

Pharos estimates an aircraft's absolute position and heading from a downward
camera by matching each frame against a georeferenced orthophoto. It needs no
GNSS and no learned weights, and it runs on a CPU.

```
 frames + telemetry ──▶ rectify ──▶ features ──▶ correlate ──▶ fixes ──▶ localizer ──▶ estimates
 (video, images,        (ground     (orientation (FFT, over     (x, y,     (candidates,
  live stream)           plane,      field)       headings &     heading,   pose window,
                         map GSD)                 scales)        spread)    state machine)
                                         ▲
 reference map (GeoTIFF / PNG + JSON) ───┘  (features precomputed once per pyramid level)
```

## Modules

| module | role |
|---|---|
| `geometry` | All coordinate conventions, camera model, ground-plane homographies. Read this first. |
| `geomap` | `GeoMap`: north-up orthophoto + affine georeference, pyramid, GeoTIFF loader, WGS84 output. |
| `terrain` | Ground elevation under a map (Copernicus GLO-30 fetched and cached, or any DEM GeoTIFF). |
| `types` | Plain dataclasses passed between stages (`Frame`, `Telemetry`, `Fix`, `Estimate`, ...). |
| `recording` | On-disk flight format shared by the simulator and real flights. Lazy frame decoding. |
| `vision.rectify` | Warps a frame onto the ground plane at map resolution using altitude and camera tilt. |
| `vision.features` | Appearance-invariant dense orientation field (see below). |
| `vision.correlate` | Masked normalized cross-correlation via FFT; peak finding and sub-pixel refinement. |
| `vision.matcher` | `Matcher` protocol and `CorrelationMatcher` (global and local search). |
| `vision.odometry` | Frame-to-frame motion: SIFT matches projected to the ground, rigid fit in metres. |
| `nav.filter` | Kalman filter that follows each candidate location during acquisition. |
| `nav.window` | Sliding-window robust pose solve over odometry, gyro, compass and map matches; the whole-flight solve for recorded footage. |
| `nav.localizer` | Acquisition, confirmation, tracking, dead reckoning after a loss, and whole-flight smoothing. |
| `sim.*` | Procedural world, seasonal/lighting appearance, trajectories, sensor models, rendering. |
| `eval.*` | Metrics (coverage, misleading rate, sigma calibration), track plots, scenario benchmark. |
| `telemetry` | Manual, DJI `.SRT`, ArduPilot `.bin` log and live MAVLink telemetry sources. |
| `footage`, `analyze` | Recorded videos and photo sets (EXIF, DJI and senseFly XMP) localized offline, with exports. |
| `sync` | Finds where a video starts in its flight log, from the footage alone. |
| `server.*` | Live service: feed runner (any video source + telemetry), map store, FastAPI app and dashboard. |
| `cli` | `pharos analyze`, `probe`, `sync`, `serve`, `simulate`, `localize`, `bench`. |

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

**Match all the ground the frame sees.** The patch is centred on the
optical-axis ground point, so a heading hypothesis only turns its content.
The template was once the largest disk inside the frame, which is the same
for every heading; but a 16:9 frame then uses only its short side. At 80 m
over the real river valley, the frame covers 2.3 times the ground of its
disk, and the true location stood out from noise in 62% of frames instead of
40%; at the stricter threshold this allows, chance peaks also passed less
often. Coverage of that flight rose from 32% to 73% with no misleading frame.
The frame's mask turns with the heading, so the correlator normalizes the
same search image once per heading (three FFTs), and features are computed
once per scale, heading-up, then turned (doubled-angle orientations turn by
twice the heading; 0.999 correlation with features of a patch rendered at
the heading). Each heading/scale hypothesis still costs one forward and one
inverse FFT.

**Global vs local search.** When lost, the matcher searches the whole map,
or a prior region such as where dead reckoning puts a lost track, at half
resolution over all headings or around the expected heading, then refines
the best candidates at full resolution. When tracking,
it searches a window sized from the track's uncertainty, over a few headings
around the prediction and three altitude-scale hypotheses.

**Judge peaks against their own window.** A raw correlation score does not
transfer between domains: synthetic frames score 0.8 to 0.9 at the true place,
real photos 0.2 to 0.6, while the best noise peak in a 150 m search window
already reaches 0.25 to 0.3. Every match is therefore scored by how far its
peak stands above the score distribution of its own window, minus what the
best of that many noise samples would reach (`correlate.peak_excess`). On the
real Aukerman photos this passes 74% of true windows and 10% of wrong ones per
frame, and confirmation needs several in a row.

**Uncertainty from the peak's shape.** Flying along a straight field edge or
road, the correlation peak is a ridge: the match fixes position across the
edge but hardly along it, and a second bump on the same ridge can score as
well as the true one. Each match therefore reports the spread of all offsets
that score within noise of its peak, and the estimator uses that as the
measurement covariance. A ridge corrects only the cross-track position, and a
track whose along-track uncertainty has grown is reported as coasting until a
match with a compact peak pins it down. On real farmland video this removed
a lock-on 70 to 480 m off.

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

**Odometry as the motion model.** Consecutive frames come from the same
camera seconds apart, so plain feature matching works between them even
where frame-to-map matching is hard. Keypoints are projected to the ground
with each frame's own altitude and tilt and a rigid motion is fitted in
metres; it is applied along each pose's own heading, so heading uncertainty
becomes cross-track uncertainty. This keeps search windows small
(fewer noise peaks to be fooled by), follows turns without a compass, and
lets a confirmed track coast through stretches with nothing to match. Its
error model is deliberately honest: attitude noise between frames shifts all
projected points alike and cannot be separated from motion.

**A window, not a filter.** A Kalman filter folds each match in once and
judges it against a single prediction, so a look-alike that arrives first
wins, and an odometry scale error (lens, field of view, height) becomes drift
it cannot see. A confirmed track is instead the last 40 s of frames re-solved
jointly every frame by robust Gauss-Newton (`nav.window`): odometry divided
by a scale that is an unknown of every frame and drifts as a random walk,
gyro heading changes, a smooth body-frame velocity, a loose compass with a
learned offset, and every map match as a max-mixture of "right" and "a
look-alike somewhere around". A wrong match is outvoted by the odometry chain
and the other matches. On a real flight the window found that odometry ran
7% short.

Matches less than 40 m apart see largely the same ground and share their
errors: a look-alike fools a run of overlapping frames alike. Each of n such
matches therefore counts as 1/n of one. Counted as independent, a run of
sixteen look-alike matches in a test outvoted odometry and the solve claimed
3 to 5 m while 50 m off; on a real flight this one rule cut the whole-flight
misleading rate from 1.9% to 0.5% and brought 99% of trusted errors within
three sigmas (from 93%). Frames leaving the window become a prior on the two oldest kept
frames (Schur complement over only the factors that leave with them, so
nothing is counted twice and the velocity across the cut survives),
loosened by 1.5 per window span against linearization error. Each of these
details was a bug before it was a rule: loosened per frame, or kept on one
frame, the prior forgot within a minute where a track without matches was.
Likewise, velocity changes over steps that odometry measured are judged in
odometry units: in metres, shrinking every step shrinks the accelerations
that odometry noise implies, and without matches the scale collapsed. And
the scale is held loosely near 1 (lens, field of view and height errors are
tens of percent) and kept between 0.5 and 2: with nothing else to hold it,
minutes of dead reckoning let it wander through zero. With these, six
minutes of dead reckoning on either real flight stays within three sigmas
from every starting point tried.

**Trust.** A position is reported as `tracking` only when its uncertainty
is at most 10 m and at least two independent inlier matches (40 m of flight
apart, since overlapping frames see the same look-alike) support it;
otherwise it is `coasting`, with its sigma. Reported sigmas include 3 m of
systematic error (map accuracy, mount calibration) that averaging many
matches does not remove; for a less accurate map, say so with
`--map-accuracy`.

**Turns.** In a banked turn the camera looks 40 to 50 degrees sideways. Map
matches there were biased (half of the confident ones on a real fixed-wing
were 20+ m off, against 9% near nadir), so only odometry runs beyond 30
degrees of tilt, and beyond 20 degrees while the tilt changes faster than
5 degrees per second (rolling into or out of a bank, when attitude timing
errors are largest). The window loosens the odometry scale and heading
increments with tilt too: terrain under an oblique view is further from the
assumed height than under nadir. And wind turns with the aircraft (the crab
angle flips between opposite legs), so the body-frame velocity prior allows
4 m/s² of change: at 2, dead reckoning 10 s into a real bank was 36 m off
with a 16 m sigma. Odometry's error model grows with tilt too: an attitude
error moves the ground under the optical axis by h·δ/cos²(tilt), 2.4 times
more at 50 degrees, so a track coasts through a turn instead of claiming
precision it does not have. Steps implying more than `max_speed` are
dropped outright.

**Losing and finding a track.** A confirmed track that stops matching keeps
searching around its prediction (reported as coasting) while that search
stays within `max_search_m` and for at most `max_coast_s` after its last
inlier match; longer coasts with wider searches were tried on real flights
and accepted more look-alikes. An untrusted track that rejects three strong
matches within 10 s, all pointing to one place, gives up its belief sooner:
the place it was pointed to becomes a candidate that must win confirmation
(a run of matches counts as one, so it cannot simply drag the track over).
A lost track is not forgotten: its window
keeps dead reckoning on odometry, gyro and compass, so a position is still
reported, as coasting, with an uncertainty that grows honestly. The search
for a new track is confined to that position's 3-sigma region instead of the
whole map, and while the region is small a candidate in it needs fewer
confirmations: continuity is evidence too. Only when dead reckoning has
grown beyond `max_dead_reckoning_m` does the search widen to the whole map.

**Recorded footage is smoothed.** Offline there is no reason to estimate
causally. `pharos analyze` solves the whole flight at once with the same
factors, so a turn or a gap is bridged from both sides; every frame record
also keeps what a live run would have reported, under `causal`.

**How results are scored.** Two numbers over all frames, against GNSS:
coverage (trusted and within 25 m) and the misleading rate (trusted and
further off). A misleading frame is the failure that matters, so every change
is judged on both. Positions from a confirmed track, trusted or coasting, are
scored as well, with the share within three reported sigmas: a coasting
position with an honest sigma is still usable.

**Self-calibration.** The window learns the odometry scale and the compass
offset (magnetic bias) as unknowns, and accepted fixes slowly update the
altitude scale used to rectify (baro bias).

**Heights follow the terrain.** Barometric heights (DJI's `rel_alt`,
MAVLink's `relative_alt`, ArduPilot's height above home) are measured from
the takeoff point, so over a valley the ground is further away than reported
and over a hill closer: a terrain-following DJI survey read 46 to 112 m
while holding 50 to 63 m above the ground. Telemetry marks such heights
(`above_takeoff`), and over a map with terrain the localizer turns them into
heights above the ground: the aircraft's elevation (the takeoff's plus the
reported height) minus the ground's where the camera looks. Odometry uses
the height at the current position (track, dead reckoning or lead
candidate). Matching uses it wherever the aircraft might be: a local search
at its centre, a global search by rendering the frame for a few heights
20% apart, nearest the expected one first, each scored only where the
terrain puts the aircraft at that height. The coarse search finds the place
with the height up to about 10% off, so 20% steps lose nothing, and places
where the aircraft would be underground are never searched. The takeoff
elevation comes from the terrain at the takeoff point (`--takeoff`, the
first position of a log or video that starts on the ground, or MAVLink's
home position); without one, the first confirmed track supplies it, each of
its matches measuring the height above the ground it saw. Copernicus GLO-30
is a surface model: over forest it gives the canopy top, which is what the
camera sees. Heights the aircraft measured above the ground itself (a
rangefinder) are used as they are. ArduPilot's terrain database is looked up
at the autopilot's GNSS position, so pharos prefers the barometric height and
its own lookup, which also covered more of a real flight (PinPoint 2: 90%
against 77%). Over a steep slope a position error becomes a height error, so
the terrain is averaged over the position's uncertainty: its sigma for
odometry, a third of the search radius for a match.

**Rectify onto the terrain.** Over uneven ground a frame is not a plane: a
valley side rising towards the camera looks larger and shifts outwards
(relief displacement), by its height over the aircraft's height above the
ground, a third on a steep slope. Local searches therefore project every
patch pixel through the terrain into the frame (`rectify` with `relief`)
instead of through the plane under the patch centre, and the template mask
follows what the frame actually shows. The relief is taken where the search
is centred and averaged over its uncertainty (a third of the search radius),
because relief finer than the position is known cannot be placed: for a
tracked aircraft that is the terrain's own detail, for a wide search a
gentle tilted plane. On a steep simulated ramp seen 20 degrees off nadir the
rectified patch correlates 0.98 with the map instead of 0.64. The simulator
ray-traces frames over its terrain to match.

Odometry needs the same, and more than expected: on a slope across the
track the uphill side of a frame is closer and seems to move faster than
the downhill side, which a plane reads as a turn, 1.2 degrees per 8 m step
on a 0.5 slope. Over simulated hills that biased the tracked heading by 3.4
degrees while every match was within 0.3, because the window trusts many
odometry steps over a few match headings. Keypoint rays now meet the terrain
around the aircraft, looked up where it is now (carried forward from the
last pose at its speed), and the bias is gone (0.3 degrees).

**Calibration once per camera.** What cannot be learned in flight is
measured once on a flight with GNSS. `pharos sync` finds the offset between
a separate camera's video and the autopilot log without GNSS: a wrong offset
pairs frames with the attitude of another moment, which makes visual
odometry jerky, so the smoothest offset wins (on a real flight it found the
published sync 0.34 s off). `pharos probe` then reports the mount
misalignment and the effective field of view from confident matches.

## Extending

* **Learned matchers** (LightGlue, LoFTR, RoMa) or **learned retrieval**
  (DINOv2/AnyLoc tiles) plug in by implementing `vision.matcher.Matcher`:
  `match(frame, region, scale, height) -> list[Fix]`. Keep them in an
  optional extra so the core stays torch-free.
* **IMU** rates or airspeed become more factors in `nav.window`.
* **Terrain in the global search**: without a position there is no relief to
  rectify onto, only the height bins; candidates are refined over terrain.
* **Large maps**: map features are precomputed in memory per pyramid level.
  A 10 km × 10 km map at 1 m needs tiling or a coarser global level.

"""Turn a scenario into a reference map and a simulated flight recording.

Over hills, each frame is ray-traced onto the terrain (a valley side rising
towards the camera looks larger), and the aircraft reports its barometric
height above the takeoff point (the first point of the flight).
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import cv2
import numpy as np

from pharos.geomap import GeoMap
from pharos.geometry import Camera, camera_rotation, ground_to_image, wrap_angle
from pharos.recording import Recording
from pharos.sim.appearance import render
from pharos.sim.scenario import CameraSpec, Scenario, TelemetrySpec
from pharos.sim.trajectory import TrajectorySpec, generate_trajectory
from pharos.sim.world import apply_changes, generate_terrain, generate_world, surface_height
from pharos.terrain import Terrain
from pharos.types import Frame, GroundTruth, Telemetry


def simulate(sc: Scenario) -> tuple[GeoMap, Recording]:
    """Build the world, the (older) satellite map and the drone flight over today's world."""
    world = generate_world(sc.world, sc.seed)
    k = round(sc.map_gsd / sc.world.gsd)
    if k < 1 or abs(k * sc.world.gsd - sc.map_gsd) > 1e-6:
        raise ValueError("map_gsd must be an integer multiple of world.gsd")

    geomap = _satellite_map(render(world, sc.map_look), sc.world.gsd, k, sc.geo_origin)
    geomap.terrain = generate_terrain(sc.terrain, sc.world.size_m, sc.seed + 4)
    dx, dy = sc.map_shift_m  # a map whose georeference is off: its image sits elsewhere
    geomap.origin_x, geomap.origin_y = geomap.origin_x + dx, geomap.origin_y + dy
    today = apply_changes(world, sc.changes, sc.seed + 1)
    ground = _as_geomap(render(today, sc.flight_look), sc.world.gsd)
    heights = surface_height(today) if sc.world.object_heights else None
    texture = sc.world.fine_texture

    camera = Camera.from_fov(sc.camera.width, sc.camera.height, sc.camera.hfov_deg)
    truth = generate_trajectory(sc.trajectory, geomap.extent, sc.seed + 2)
    if any(sc.map_crop_m):
        geomap = _cropped(geomap, sc.map_crop_m)
    above, takeoff = None, None
    if geomap.terrain is not None:
        truth, above = _over_terrain(truth, geomap.terrain, sc.trajectory)
        if sc.telemetry.takeoff_known:
            takeoff = (truth[0].x, truth[0].y)
    rng = np.random.default_rng(sc.seed + 3)
    sensors = _SensorModel(sc.telemetry, rng, truth, above)
    drops = _raindrops(camera, sc.camera.raindrops, np.random.default_rng(sc.seed + 5))
    frames, kept, still = [], [], None
    for i, g in enumerate(truth):
        lost = _within(g.t, sc.camera.dropped)
        if not lost and (still is None or not _within(g.t, sc.camera.freezes)):
            still = render_frame(
                ground, camera, g, sc.camera, rng, geomap.terrain, heights, texture, drops
            )
        tel = sensors.read(i)  # after rendering, as the shared random draws always were
        if lost:
            continue  # no frame at all
        frames.append(Frame(g.t, still, tel))  # frozen video repeats its last image
        kept.append(g)
    return geomap, Recording(camera, frames, kept, takeoff)


def _cropped(geomap: GeoMap, crop_m) -> GeoMap:
    """The map without the given metres of its (west, south, east, north) sides."""
    w, s, e, n = (round(m / geomap.gsd) for m in crop_m)
    rows, cols = geomap.shape
    return GeoMap(
        geomap.image[n : rows - s, w : cols - e],
        geomap.gsd,
        origin_x=geomap.origin_x + w * geomap.gsd,
        origin_y=geomap.origin_y - n * geomap.gsd,
        geo_origin=geomap.geo_origin,
        terrain=geomap.terrain,
    )


def _within(t: float, spans) -> bool:
    """Whether ``t`` falls in one of the (start, duration, ...) spans."""
    return any(a <= t < a + d for a, d, *_ in spans)


def _raindrops(
    camera: Camera, n: int, rng: np.random.Generator
) -> list[tuple[float, float, float]]:
    """Drops stuck on the lens: (column, row, radius) in pixels, the same in every frame."""
    scale = min(camera.width, camera.height)
    return [
        (
            rng.uniform(0, camera.width),
            rng.uniform(0, camera.height),
            rng.uniform(0.04, 0.1) * scale,
        )
        for _ in range(n)
    ]


def _over_terrain(
    truth: list[GroundTruth], terrain: Terrain, spec: TrajectorySpec
) -> tuple[list[GroundTruth], list[float]]:
    """Poses with heights above the ground below, and the heights above takeoff."""
    ground = terrain.at(np.array([(g.x, g.y) for g in truth]))
    if spec.follow_terrain:
        above = [g.altitude + z - ground[0] for g, z in zip(truth, ground, strict=True)]
    else:
        above = [g.altitude for g in truth]
    agl = [a + ground[0] - z for a, z in zip(above, ground, strict=True)]
    if min(agl) < 30.0:
        raise ValueError("the hills reach within 30 m of the aircraft: fly higher")
    out = [replace(g, altitude=float(h)) for g, h in zip(truth, agl, strict=True)]
    return out, [float(a) for a in above]


def render_frame(
    ground: GeoMap,
    camera: Camera,
    pose: GroundTruth,
    spec: CameraSpec,
    rng: np.random.Generator,
    terrain: Terrain | None = None,
    heights: np.ndarray | None = None,
    texture: float = 0.0,
    drops: list[tuple[float, float, float]] = (),
) -> np.ndarray:
    """What the camera sees from ``pose`` over the ground image, with sensor effects.

    ``pose.altitude`` is the height above the ground under the aircraft; without
    ``terrain`` the ground is flat. With ``heights`` (metres, aligned with ``ground``)
    what stands on the ground is drawn as layers a metre or two apart: a low camera
    sees tree tops lean outwards and the sides of hedges and buildings (in shade).
    ``texture`` is the contrast of fine detail fixed to the ground (grass, gravel),
    drawn where the frame resolves finer than the ground image. ``drops`` blur and
    brighten discs fixed in the image (raindrops on the lens).
    """
    # Pick a ground pyramid level no finer than the frame so the warp does not alias.
    frame_gsd = pose.altitude / camera.fx
    level = int(np.clip(np.floor(np.log2(frame_gsd / ground.gsd)), 0, 4)) if frame_gsd > 0 else 0
    src, g = _color_level(ground, level)

    def surface(lift: float) -> np.ndarray:
        """World (x, y) each pixel sees on the surface ``lift`` m above the ground."""
        if terrain is None:
            return _plane_seen(camera, pose, lift)
        return _ground_seen(camera, pose, terrain, lift=lift)

    detail = texture * float(np.clip(1.0 - frame_gsd / ground.gsd, 0.0, 1.0))
    xy = surface(0.0) if detail > 0 else None

    def seen(image: np.ndarray, lift: float, interpolation: int) -> np.ndarray:
        """``image`` (at the chosen level) as seen on the surface ``lift`` m above ground."""
        if terrain is None:
            H = ground_to_image(
                camera, (pose.x, pose.y, pose.altitude - lift), pose.heading, pose.pitch, pose.roll
            )
            A = np.array([[g, 0.0, ground.origin_x], [0.0, -g, ground.origin_y], [0.0, 0.0, 1.0]])
            size = (camera.width, camera.height)
            return cv2.warpPerspective(image, H @ A, size, flags=interpolation)
        xy = _ground_seen(camera, pose, terrain, lift=lift)
        u = ((xy[..., 0] - ground.origin_x) / g).astype(np.float32)
        v = ((ground.origin_y - xy[..., 1]) / g).astype(np.float32)
        return cv2.remap(image, u, v, interpolation)

    img = seen(src, 0.0, cv2.INTER_LINEAR)
    if heights is not None:
        tall = cv2.resize(heights, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST)
        walls = tall - cv2.erode(tall, np.ones((3, 3), np.uint8)) > 1.5  # edges beside a drop
        for z in (z for z in _LAYERS_M if z <= min(float(tall.max()), pose.altitude - 5.0)):
            stands = seen(tall, z, cv2.INTER_LINEAR)
            on = stands >= z
            if not on.any():
                continue
            layer = seen(src, z, cv2.INTER_LINEAR)
            # below the top of a wall (a hedge's or building's side), in its own shade
            side = (
                on & (stands >= z + 1.0) & (seen(walls.astype(np.uint8), z, cv2.INTER_NEAREST) > 0)
            )
            img[on] = layer[on]
            img[side] = (0.7 * layer[side]).astype(np.uint8)
            if xy is not None:
                xy[on] = surface(z)[on]

    f = img.astype(np.float32) / 255.0
    if xy is not None:
        f *= (1.0 + detail * _fine_texture(xy))[..., None]
    if drops:
        wet = np.zeros(f.shape[:2], np.float32)
        for u, v, r in drops:
            cv2.circle(wet, (round(u), round(v)), round(r), 1.0, -1, cv2.LINE_AA)
        wet = cv2.GaussianBlur(wet, (0, 0), 3.0)[..., None]
        f = (1 - wet) * f + wet * 1.1 * cv2.GaussianBlur(f, (0, 0), 9.0)
    if spec.blur > 0:
        f = cv2.GaussianBlur(f, (0, 0), spec.blur)
    if spec.motion_blur_px >= 1:
        n = round(spec.motion_blur_px) | 1
        f = cv2.filter2D(f, -1, np.full((n, 1), 1.0 / n, np.float32))  # forward = image up
    yy, xx = np.mgrid[: camera.height, : camera.width].astype(np.float32)
    r2 = ((xx - camera.cx) ** 2 + (yy - camera.cy) ** 2) / (camera.cx**2 + camera.cy**2)
    f *= (1.0 - spec.vignetting * r2)[..., None]
    f *= 1.0 + spec.exposure_jitter * rng.standard_normal()
    f += rng.normal(0, spec.noise, f.shape).astype(np.float32)
    out = (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, spec.jpeg_quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else out


_LAYERS_M = (*range(1, 13), 14, 16, 18, 20)  # metres; a top is drawn at the layer below it
_TEXTURE_M = 0.08  # ground metres per pixel of the fine texture tile


def _plane_seen(camera: Camera, pose: GroundTruth, lift: float = 0.0) -> np.ndarray:
    """World (x, y) each image pixel sees on flat ground raised by ``lift`` m."""
    uu, vv = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
    rays = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(camera.K).T
    rays = rays @ camera_rotation(pose.heading, pose.pitch, pose.roll).T
    down = np.minimum(rays[..., 2], -1e-3)
    return np.array([pose.x, pose.y]) + rays[..., :2] * ((pose.altitude - lift) / -down)[..., None]


@cache
def _texture_tile() -> np.ndarray:
    noise = np.random.default_rng(12345).standard_normal((512, 512)).astype(np.float32)
    tile = cv2.GaussianBlur(np.pad(noise, 8, mode="wrap"), (0, 0), 1.0)[8:-8, 8:-8]
    return tile / tile.std()  # seamless: it repeats every 512 pixels


def _fine_texture(xy: np.ndarray) -> np.ndarray:
    """Detail fixed to world positions ``xy``: two octaves (8 and 32 cm), unit spread."""
    tile = _texture_tile()
    out = np.zeros(xy.shape[:2], np.float32)
    for scale, weight in ((_TEXTURE_M, 0.6), (4 * _TEXTURE_M, 0.8)):
        u = (xy[..., 0] / scale).astype(np.float32)
        v = (xy[..., 1] / scale).astype(np.float32)
        out += weight * cv2.remap(tile, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return out


def _ground_seen(
    camera: Camera,
    pose: GroundTruth,
    terrain: Terrain,
    step: int = 4,
    rounds: int = 12,
    lift: float = 0.0,
) -> np.ndarray:
    """World (x, y) of the terrain (raised by ``lift`` m) each image pixel sees, by
    marching every ray.

    Rays are traced on a grid ``step`` pixels apart and interpolated: the
    terrain is smooth on that scale.
    """
    w, h = -(-camera.width // step), -(-camera.height // step)
    uu, vv = np.meshgrid(step * np.arange(w) + (step - 1) / 2, step * np.arange(h) + (step - 1) / 2)
    rays = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(camera.K).T
    rays = rays @ camera_rotation(pose.heading, pose.pitch, pose.roll).T  # world directions
    down = np.minimum(rays[..., 2], -1e-3)  # rays at or above the horizon: far away
    origin = np.array([pose.x, pose.y])
    top = float(terrain.at(origin)) + pose.altitude
    z = np.full((h, w), top - pose.altitude + lift)
    for _ in range(rounds):  # damped fixed point: where the ray meets the ground's height
        xy = origin + rays[..., :2] * ((top - z) / -down)[..., None]
        z = 0.5 * z + 0.5 * np.minimum(terrain.at(xy) + lift, top - 1.0)
    xy = origin + rays[..., :2] * ((top - z) / -down)[..., None]
    size = (w * step, h * step)
    full = np.stack([cv2.resize(xy[..., k], size, interpolation=cv2.INTER_LINEAR) for k in (0, 1)])
    return np.moveaxis(full, 0, -1)[: camera.height, : camera.width]


class _SensorModel:
    """Turns true poses into the biased, noisy telemetry an aircraft would report."""

    def __init__(
        self,
        spec: TelemetrySpec,
        rng: np.random.Generator,
        truth: list[GroundTruth],
        above_takeoff: list[float] | None = None,
    ):
        """``above_takeoff``: true heights above the takeoff point, reported instead."""
        self.spec = spec
        self.rng = rng
        self.truth = truth
        self.above_takeoff = above_takeoff
        self.compass_bias = np.radians(spec.compass_bias_deg) * rng.choice([-1, 1])
        self.drift_rate = np.radians(spec.compass_drift_deg_per_min) / 60 * rng.choice([-1, 1])
        self.alt_bias = 1.0 + spec.altitude_bias * rng.choice([-1, 1])
        self.alt_offset = spec.altitude_offset_m * rng.choice([-1, 1])
        direction = rng.uniform(0, 2 * np.pi)
        bias = np.radians(spec.attitude_bias_deg)
        self.mount = (bias * np.cos(direction), bias * np.sin(direction))  # pitch, roll
        self.last: Telemetry | None = None

    def read(self, i: int) -> Telemetry:
        """What the aircraft reports alongside frame ``i`` (in order: dropouts repeat)."""
        t = self.truth[i].t
        if self.last is not None and _within(t, self.spec.dropouts):
            return self.last
        self.last = self._measure(i)
        return self.last

    def _measure(self, i: int) -> Telemetry:
        s, rng = self.spec, self.rng
        t = self.truth[i].t
        j = min(range(len(self.truth)), key=lambda k: abs(self.truth[k].t - (t - s.latency_s)))
        g = self.truth[j]
        height = g.altitude if self.above_takeoff is None else self.above_takeoff[j]
        att = np.radians(s.attitude_noise_deg)
        yaw = None
        if s.compass:
            err = self.compass_bias + self.drift_rate * t
            err += sum(np.radians(deg) for a, d, deg in s.compass_jumps if a <= t < a + d)
            yaw = float(
                wrap_angle(g.heading + err + rng.normal(0, np.radians(s.compass_noise_deg)))
            )
        return Telemetry(
            altitude=float(
                height * self.alt_bias + self.alt_offset + rng.normal(0, s.altitude_noise_m)
            ),
            yaw=yaw,
            pitch=float(g.pitch + self.mount[0] + rng.normal(0, att)),
            roll=float(g.roll + self.mount[1] + rng.normal(0, att)),
            above_takeoff=self.above_takeoff is not None,
        )


def _as_geomap(image: np.ndarray, gsd: float) -> GeoMap:
    return GeoMap(image, gsd, origin_x=0.0, origin_y=(image.shape[0] - 1) * gsd)


def _satellite_map(image: np.ndarray, gsd: float, k: int, geo_origin) -> GeoMap:
    """Downsample the rendered world by ``k`` (area average) into the reference map."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (w // k, h // k), interpolation=cv2.INTER_AREA) if k > 1 else image
    half = (k - 1) / 2 * gsd
    return GeoMap(
        small,
        gsd * k,
        origin_x=half,
        origin_y=(h - 1) * gsd - half,
        geo_origin=tuple(geo_origin) if geo_origin else None,
    )


def _color_level(ground: GeoMap, level: int) -> tuple[np.ndarray, float]:
    img = ground.image
    for _ in range(level):
        img = cv2.pyrDown(img)
    return img, ground.gsd * 2**level

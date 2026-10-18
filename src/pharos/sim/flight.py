"""Turn a scenario into a reference map and a simulated flight recording."""

from __future__ import annotations

import cv2
import numpy as np

from pharos.geomap import GeoMap
from pharos.geometry import Camera, ground_to_image, wrap_angle
from pharos.recording import Recording
from pharos.sim.appearance import render
from pharos.sim.scenario import CameraSpec, Scenario, TelemetrySpec
from pharos.sim.trajectory import generate_trajectory
from pharos.sim.world import apply_changes, generate_world
from pharos.types import Frame, GroundTruth, Telemetry


def simulate(sc: Scenario) -> tuple[GeoMap, Recording]:
    """Build the world, the (older) satellite map and the drone flight over today's world."""
    world = generate_world(sc.world, sc.seed)
    k = round(sc.map_gsd / sc.world.gsd)
    if k < 1 or abs(k * sc.world.gsd - sc.map_gsd) > 1e-6:
        raise ValueError("map_gsd must be an integer multiple of world.gsd")

    geomap = _satellite_map(render(world, sc.map_look), sc.world.gsd, k, sc.geo_origin)
    today = apply_changes(world, sc.changes, sc.seed + 1)
    ground = _as_geomap(render(today, sc.flight_look), sc.world.gsd)

    camera = Camera.from_fov(sc.camera.width, sc.camera.height, sc.camera.hfov_deg)
    truth = generate_trajectory(sc.trajectory, geomap.extent, sc.seed + 2)
    rng = np.random.default_rng(sc.seed + 3)
    sensors = _SensorModel(sc.telemetry, rng, truth)
    frames = [
        Frame(g.t, render_frame(ground, camera, g, sc.camera, rng), sensors.read(i))
        for i, g in enumerate(truth)
    ]
    return geomap, Recording(camera, frames, truth)


def render_frame(
    ground: GeoMap, camera: Camera, pose: GroundTruth, spec: CameraSpec, rng: np.random.Generator
) -> np.ndarray:
    """What the camera sees from ``pose`` over the flat ground image, with sensor effects."""
    H = ground_to_image(
        camera, (pose.x, pose.y, pose.altitude), pose.heading, pose.pitch, pose.roll
    )
    # Pick a ground pyramid level no finer than the frame so the warp does not alias.
    frame_gsd = pose.altitude / camera.fx
    level = int(np.clip(np.floor(np.log2(frame_gsd / ground.gsd)), 0, 4)) if frame_gsd > 0 else 0
    src, g = _color_level(ground, level)
    A = np.array([[g, 0.0, ground.origin_x], [0.0, -g, ground.origin_y], [0.0, 0.0, 1.0]])
    img = cv2.warpPerspective(src, H @ A, (camera.width, camera.height), flags=cv2.INTER_LINEAR)

    f = img.astype(np.float32) / 255.0
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


class _SensorModel:
    """Turns true poses into the biased, noisy telemetry an aircraft would report."""

    def __init__(self, spec: TelemetrySpec, rng: np.random.Generator, truth: list[GroundTruth]):
        self.spec = spec
        self.rng = rng
        self.truth = truth
        self.compass_bias = np.radians(spec.compass_bias_deg) * rng.choice([-1, 1])
        self.drift_rate = np.radians(spec.compass_drift_deg_per_min) / 60 * rng.choice([-1, 1])
        self.alt_bias = 1.0 + spec.altitude_bias * rng.choice([-1, 1])
        self.alt_offset = spec.altitude_offset_m * rng.choice([-1, 1])
        direction = rng.uniform(0, 2 * np.pi)
        bias = np.radians(spec.attitude_bias_deg)
        self.mount = (bias * np.cos(direction), bias * np.sin(direction))  # pitch, roll

    def read(self, i: int) -> Telemetry:
        """What the aircraft reports alongside frame ``i``."""
        s, rng = self.spec, self.rng
        t = self.truth[i].t
        g = min(self.truth, key=lambda q: abs(q.t - (t - s.latency_s)))
        att = np.radians(s.attitude_noise_deg)
        yaw = None
        if s.compass:
            err = self.compass_bias + self.drift_rate * t
            yaw = float(
                wrap_angle(g.heading + err + rng.normal(0, np.radians(s.compass_noise_deg)))
            )
        return Telemetry(
            altitude=float(
                g.altitude * self.alt_bias + self.alt_offset + rng.normal(0, s.altitude_noise_m)
            ),
            yaw=yaw,
            pitch=float(g.pitch + self.mount[0] + rng.normal(0, att)),
            roll=float(g.roll + self.mount[1] + rng.normal(0, att)),
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

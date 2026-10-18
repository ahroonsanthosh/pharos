import cv2
import numpy as np

from pharos.geomap import GeoMap
from pharos.geometry import Camera, rot2
from pharos.sim.flight import render_frame
from pharos.sim.scenario import CameraSpec
from pharos.types import GroundTruth
from pharos.vision.features import orientation_field
from pharos.vision.rectify import (
    disk_mask,
    footprint,
    frame_mask,
    nadir_from_patch_center,
    rectify,
)

CLEAN = CameraSpec(
    blur=0, motion_blur_px=0, noise=0, vignetting=0, exposure_jitter=0, jpeg_quality=100
)


def ground_map():
    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur(rng.random((1200, 1200)).astype(np.float32), (0, 0), 4)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return GeoMap(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), gsd=0.5, origin_x=0, origin_y=599.5)


def test_rectified_patch_matches_map():
    ground = ground_map()
    cam = Camera.from_fov(640, 480, 70)
    pose = GroundTruth(0, 300.0, 300.0, 120.0, np.radians(37), np.radians(12), np.radians(-4))
    frame = cv2.cvtColor(
        render_frame(ground, cam, pose, CLEAN, np.random.default_rng(0)), cv2.COLOR_BGR2GRAY
    )
    fp = footprint(cam, pose.altitude, pose.pitch, pose.roll)
    gsd, R = 1.0, 40
    patch = rectify(frame, cam, pose.altitude, pose.pitch, pose.roll, pose.heading, gsd, R, fp)

    center = np.array([pose.x, pose.y]) + rot2(pose.heading) @ fp.center
    gm = GeoMap(cv2.pyrDown(ground.gray), 1.0, origin_x=0.25, origin_y=599.25)
    u, v = np.round(gm.world_to_px(center)).astype(int)
    ref = gm.gray[v - R : v + R + 1, u - R : u + R + 1]

    mask = disk_mask(R)
    a, b = orientation_field(patch, mask), orientation_field(ref, mask)
    cos = np.vdot(a, b).real / (np.linalg.norm(a) * np.linalg.norm(b))
    assert cos > 0.8
    np.testing.assert_allclose(nadir_from_patch_center(center, pose.heading, fp), [pose.x, pose.y])


def test_footprint_of_tilted_camera_moves_forward():
    cam = Camera.from_fov(640, 480, 70)
    nadir = footprint(cam, 100, 0, 0)
    tilted = footprint(cam, 100, np.radians(20), 0)
    np.testing.assert_allclose(nadir.center, [0, 0], atol=1e-9)
    assert tilted.center[1] > 30 and abs(tilted.center[0]) < 1e-6
    # inscribed radius is set by the short (vertical) side of the image
    assert np.isclose(nadir.radius, 100 * 0.75 * np.tan(np.radians(35)), rtol=0.01)


def test_frame_mask_covers_the_whole_image_and_turns_with_heading():
    cam = Camera.from_fov(1280, 720, 77)
    fp = footprint(cam, 80.0, 0.0, 0.0)
    gsd = 1.0
    r = int(fp.extent / gsd)
    north_up = frame_mask(fp, 0.0, gsd, r)
    east_up = frame_mask(fp, np.pi / 2, gsd, r)
    # a wide frame covers much more ground than its inscribed disk
    assert north_up.sum() > 1.8 * np.pi * fp.radius**2
    rows, cols = np.nonzero(north_up)
    assert np.ptp(cols) > 1.6 * np.ptp(rows)  # wide across the heading
    rows, cols = np.nonzero(east_up)
    assert np.ptp(rows) > 1.6 * np.ptp(cols)  # flying east: wide north-south
    assert abs(int(north_up.sum()) - int(east_up.sum())) < 0.03 * north_up.sum()

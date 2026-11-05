"""Lens distortion: straightening frames, measuring a lens, and tracking through one."""

import re
from dataclasses import replace

import cv2
import numpy as np
import pytest

from pharos.cli import main
from pharos.eval.bench import localize
from pharos.eval.metrics import evaluate
from pharos.geometry import Camera
from pharos.recording import load_recording, save_recording
from pharos.sim.flight import simulate
from pharos.sim.scenario import CameraSpec, Scenario
from pharos.sim.trajectory import TrajectorySpec
from pharos.sim.world import WorldSpec
from pharos.vision.lens import Lens, calibrate, distort, margin_needed, pinhole

BARREL = (-0.3, 0.1)


def checkerboard(width: int, height: int, square: int = 24) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    return (((xs // square + ys // square) % 2) * 255).astype(np.uint8)


def test_a_pinhole_camera_needs_no_straightening():
    camera = Camera.from_fov(640, 480, 70)
    assert pinhole(camera) == (camera, None)
    assert pinhole(replace(camera, dist=(0.0, 0.0)))[1] is None


def test_straightening_undoes_the_lens():
    camera = Camera.from_fov(640, 480, 80, dist=BARREL)
    margin = margin_needed(camera)
    ideal = checkerboard(640 + 2 * margin, 480 + 2 * margin)
    seen = distort(ideal, camera, margin)
    lens = Lens(camera)
    straight = lens.straighten(seen)
    # what the straightened pinhole camera sees of the same ideal scene
    shift = np.array([[1, 0, margin], [0, 1, margin], [0, 0, 1]], float)
    H = lens.camera.K @ np.linalg.inv(camera.K) @ np.linalg.inv(shift)
    expected = cv2.warpPerspective(ideal, H, (640, 480), flags=cv2.INTER_LINEAR)
    inner = (slice(20, -20), slice(20, -20))
    differs = np.abs(straight[inner].astype(int) - expected[inner].astype(int)) > 64
    assert differs.mean() < 0.02  # only pixels on the squares' edges disagree
    assert straight.min() == 0 and straight.max() == 255  # no blank corners or fading


def synthetic_board_views(camera: Camera, n: int = 12) -> list[np.ndarray]:
    """Photos of a 9x6 board (10 x 7 squares) from different angles, through ``camera``."""
    square = 60
    board = np.full((7 * square + 2 * square, 10 * square + 2 * square), 255, np.uint8)
    board[square:-square, square:-square] = checkerboard(10 * square, 7 * square, square)
    margin = margin_needed(camera)
    K = camera.K + np.array([[0, 0, margin], [0, 0, margin], [0, 0, 0]])  # the wider view
    h, w = board.shape
    corners = np.float32([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]])  # board pixels, z = 0
    rng = np.random.default_rng(1)
    views = []
    for i in range(n):
        # the board turned up to 35 degrees and held where it fills half or more of the frame
        rvec = rng.uniform(-0.6, 0.6, 3) * np.array([1, 1, 0.5])
        z = camera.fx * w / (camera.width * rng.uniform(0.55, 0.85))
        shift = (
            rng.uniform(-0.25, 0.25, 2) * np.array([camera.width, camera.height]) * z / camera.fx
        )
        R = cv2.Rodrigues(rvec)[0]
        tvec = np.array([shift[0], shift[1], z]) - R @ np.array([w / 2, h / 2, 0])
        quad, _ = cv2.projectPoints(corners, rvec, tvec, K, None)
        flat = np.ascontiguousarray(corners[:, :2])
        H = cv2.getPerspectiveTransform(flat, quad.reshape(4, 2).astype(np.float32))
        size_view = (camera.width + 2 * margin, camera.height + 2 * margin)
        ideal = cv2.warpPerspective(board, H, size_view, borderValue=180)
        views.append(cv2.GaussianBlur(distort(ideal, camera, margin), (0, 0), 0.7 + 0.1 * (i % 3)))
    return views


def test_a_lens_is_measured_from_a_checkerboard():
    camera = Camera.from_fov(800, 600, 75, dist=(-0.2, 0.05))
    result = calibrate(synthetic_board_views(camera), (9, 6))
    assert result.views >= 8 and result.rms_px < 0.5
    assert result.hfov_deg == pytest.approx(75, abs=1.5)
    assert result.camera.dist[0] == pytest.approx(-0.2, abs=0.04)


def test_pharos_lens_prints_what_to_pass(tmp_path, capsys):
    camera = Camera.from_fov(800, 600, 75, dist=(-0.2, 0.05))
    for i, view in enumerate(synthetic_board_views(camera)):
        cv2.imwrite(str(tmp_path / f"board_{i:02d}.png"), view)
    main(["lens", str(tmp_path), "--board", "9x6"])
    out = capsys.readouterr().out
    hfov, k1 = re.search(r"--hfov ([\d.]+) --lens (-?[\d.e-]+)", out).groups()
    assert float(hfov) == pytest.approx(75, abs=1.5)
    assert float(k1) == pytest.approx(-0.2, abs=0.04)


def test_a_recording_keeps_its_lens(small_sim, tmp_path):
    _, rec = small_sim
    rec = replace(rec, camera=replace(rec.camera, dist=BARREL), frames=rec.frames[:2])
    save_recording(rec, tmp_path / "rec")
    assert load_recording(tmp_path / "rec").camera.dist == BARREL


@pytest.fixture(scope="module")
def wide_lens_sim():
    """A tilted camera behind a strong barrel lens, where distortion costs the most."""
    sc = Scenario(
        seed=43,
        world=WorldSpec(size_m=700, villages=2, buildings_per_village=20),
        trajectory=TrajectorySpec(duration_s=30, margin_m=200, altitude=150, tilt_deg=25),
        camera=CameraSpec(distortion=BARREL),
    )
    return simulate(sc)


def test_tracking_through_a_lens_needs_to_be_told_it(wide_lens_sim):
    geomap, rec = wide_lens_sim
    assert rec.camera.dist == BARREL
    told = evaluate(localize(geomap, rec), rec.truth)
    blind = evaluate(localize(geomap, replace(rec, camera=replace(rec.camera, dist=()))), rec.truth)
    assert told["misleading"] == 0
    assert told["err_median_m"] < 3.0
    assert told["err_median_m"] < 0.6 * blind["err_median_m"]

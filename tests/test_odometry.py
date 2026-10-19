import numpy as np

from pharos.geometry import rot2, wrap_angle
from pharos.vision.odometry import VisualOdometry


def test_odometry_follows_simulated_flight(small_sim):
    _, rec = small_sim
    vo = VisualOdometry(rec.camera)
    errors, heading_errors = [], []
    for k, (frame, truth) in enumerate(zip(rec.frames[:20], rec.truth[:20], strict=True)):
        motion = vo.step(frame)
        if k == 0:
            assert motion is None
            continue
        assert motion is not None
        prev = rec.truth[k - 1]
        expected = rot2(prev.heading).T @ np.array([truth.x - prev.x, truth.y - prev.y])
        errors.append(np.hypot(*(motion.d_body - expected)))
        heading_errors.append(abs(wrap_angle(motion.d_heading - (truth.heading - prev.heading))))
        assert 0.8 < motion.scale < 1.2
    # per-step error is dominated by the simulated 0.5 degree attitude noise at 170 m
    assert np.median(errors) < 4.0
    assert np.degrees(np.median(heading_errors)) < 0.5
    assert np.median(errors) < 2 * vo.step(rec.frames[20]).sigma_m  # honest uncertainty


def test_no_overlap_gives_no_motion(small_sim):
    _, rec = small_sim
    vo = VisualOdometry(rec.camera)
    vo.step(rec.frames[0])
    blank = rec.frames[1].image * 0 + 128
    assert vo.step(type(rec.frames[1])(1.0, blank, rec.frames[1].telemetry)) is None


def test_odometry_over_a_slope_across_the_track_sees_no_turn(small_sim):
    """Flying north along a slope rising eastwards: projected onto a plane, the uphill side
    seems to move faster than the downhill side, which reads as a turn."""
    import cv2

    from pharos.geomap import GeoMap
    from pharos.sim.flight import render_frame
    from pharos.sim.scenario import CameraSpec
    from pharos.terrain import Terrain
    from pharos.types import Frame, GroundTruth, Telemetry

    geomap, rec = small_sim
    ground = GeoMap(geomap.image, geomap.gsd, geomap.origin_x, geomap.origin_y)
    xs = np.arange(0, 701, 10.0)
    ramp = Terrain(np.tile(0.5 * xs, (len(xs), 1)).astype(np.float32), 10.0, 0.0, 700.0)
    clean = CameraSpec(blur=0, motion_blur_px=0, noise=0, vignetting=0, exposure_jitter=0)
    vo = VisualOdometry(rec.camera)
    tel = Telemetry(120.0, 0.0)  # straight down, 120 m above the ground below

    def relief(local):  # heading north: right is east, so the ground rises to the right
        return 0.5 * local[:, 0]

    turns = {}
    for name, terrain in (("plane", None), ("terrain", relief)):
        frames = []
        for k, y in enumerate((300.0, 308.0)):
            pose = GroundTruth(k, 350.0, y, 120.0, 0.0)
            image = render_frame(ground, rec.camera, pose, clean, np.random.default_rng(k), ramp)
            frames.append(Frame(float(k), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), tel))
        a, b = vo.features(frames[0].image), vo.features(frames[1].image)
        motion = vo.motion(*vo.correspondences(a, b), tel, tel, relief_a=terrain, relief_b=terrain)
        turns[name] = abs(np.degrees(motion.d_heading))
        assert np.hypot(*(motion.d_body - (0.0, 8.0))) < (1.0 if terrain else 3.0)
    assert turns["plane"] > 1.0 and turns["terrain"] < 0.2

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

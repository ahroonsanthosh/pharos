import numpy as np

from pharos.nav.filter import FilterConfig, PoseFilter

R = np.diag([3.0**2, 3.0**2, np.radians(2) ** 2])


def test_converges_to_constant_velocity():
    kf = PoseFilter(FilterConfig(), 0.0, 0.0, 0.0, pos_sigma=5, heading_sigma=0.1)
    rng = np.random.default_rng(0)
    for k in range(1, 40):
        kf.predict(0.5)
        z = np.array([10.0 * 0.5 * k, -4.0 * 0.5 * k, 0.0]) + rng.normal(0, [1, 1, 0.01])
        assert kf.update(z, R)
    np.testing.assert_allclose(kf.x[2:4], [10.0, -4.0], atol=1.5)
    assert kf.pos_sigma < 3.0


def test_gate_rejects_outlier():
    kf = PoseFilter(FilterConfig(), 0.0, 0.0, 0.0, pos_sigma=3, heading_sigma=0.05)
    kf.predict(0.5)
    before = kf.x.copy()
    assert not kf.update(np.array([300.0, 0.0, 0.0]), R)
    np.testing.assert_array_equal(kf.x, before)


def test_heading_wraps_across_pi():
    kf = PoseFilter(FilterConfig(), 0.0, 0.0, np.pi - 0.02, pos_sigma=3, heading_sigma=0.1)
    kf.predict(0.5)
    assert kf.update(np.array([0.0, 0.0, -np.pi + 0.02]), R)
    assert abs(abs(kf.x[4]) - np.pi) < 0.03


def test_heading_change_input_is_applied():
    kf = PoseFilter(FilterConfig(), 0.0, 0.0, 0.0, pos_sigma=3, heading_sigma=0.05)
    kf.predict(0.5, heading_change=0.3)
    assert np.isclose(kf.x[4], 0.3)


def test_odometry_step_moves_along_heading():
    kf = PoseFilter(FilterConfig(), 0.0, 0.0, np.pi / 2, pos_sigma=1, heading_sigma=np.radians(5))
    kf.predict_motion(2.0, np.array([0.0, 20.0]), np.radians(10), sigma_m=1.0, sigma_heading=0.01)
    np.testing.assert_allclose(kf.x[:2], [20.0, 0.0], atol=1e-9)  # forward = east
    np.testing.assert_allclose(kf.x[2:4], [10.0, 0.0], atol=1e-9)  # velocity from measured rate
    assert np.isclose(kf.x[4], np.pi / 2 + np.radians(10))
    # heading uncertainty becomes cross-track (north-south) position uncertainty
    assert kf.P[1, 1] > kf.P[0, 0] + 1.0

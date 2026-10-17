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

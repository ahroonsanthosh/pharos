import numpy as np
import pytest

from pharos.geometry import (
    Camera,
    apply_h,
    camera_attitude,
    camera_rotation,
    euler_ned,
    ground_to_image,
    image_to_ground_points,
    misaligned,
    rot2,
    rot_x,
    rot_y,
    wrap_angle,
)

CAM = Camera.from_fov(640, 480, 70.0)
PP = np.array([[CAM.cx, CAM.cy]])


def project(xy, pos, yaw, pitch=0.0, roll=0.0):
    return apply_h(ground_to_image(CAM, pos, yaw, pitch, roll), np.atleast_2d(xy))[0]


def test_nadir_looks_straight_down():
    uv = project([100.0, 200.0], (100.0, 200.0, 120.0), yaw=0.3)
    np.testing.assert_allclose(uv, PP[0], atol=1e-9)


@pytest.mark.parametrize(
    ("yaw_deg", "ahead"), [(0, (0, 10)), (90, (10, 0)), (180, (0, -10)), (270, (-10, 0))]
)
def test_point_ahead_appears_above_centre(yaw_deg, ahead):
    uv = project(ahead, (0.0, 0.0, 100.0), np.radians(yaw_deg))
    assert uv[1] < CAM.cy - 10
    assert abs(uv[0] - CAM.cx) < 1e-6


def test_point_to_the_right_appears_right():
    # heading east: the drone's right is south
    uv = project((0.0, -10.0), (0.0, 0.0, 100.0), np.radians(90))
    assert uv[0] > CAM.cx + 10


def test_pitch_looks_forward_and_roll_looks_right():
    H = ground_to_image(CAM, (0.0, 0.0, 100.0), yaw=0.0, pitch=np.radians(30))
    ground = image_to_ground_points(H, PP)[0]
    np.testing.assert_allclose(ground, [0.0, 100 * np.tan(np.radians(30))], atol=1e-6)
    H = ground_to_image(CAM, (0.0, 0.0, 100.0), yaw=0.0, roll=np.radians(20))
    ground = image_to_ground_points(H, PP)[0]
    np.testing.assert_allclose(ground, [100 * np.tan(np.radians(20)), 0.0], atol=1e-6)


def test_image_ground_round_trip():
    H = ground_to_image(CAM, (5.0, -3.0, 80.0), 1.1, np.radians(10), np.radians(-5))
    uv = np.array([[0.0, 0.0], [639.0, 479.0], [320.0, 100.0]])
    np.testing.assert_allclose(apply_h(H, image_to_ground_points(H, uv)), uv, atol=1e-6)


def test_rays_above_horizon_are_nan():
    H = ground_to_image(CAM, (0.0, 0.0, 100.0), 0.0, pitch=np.radians(80))
    assert np.isnan(image_to_ground_points(H, np.array([[CAM.cx, 0.0]]))).all()


def test_rotation_is_proper():
    R = camera_rotation(0.7, 0.2, -0.1)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rot2_maps_forward_to_heading():
    for yaw in np.radians([0, 45, 90, 200]):
        np.testing.assert_allclose(rot2(yaw) @ [0.0, 1.0], [np.sin(yaw), np.cos(yaw)])


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    np.testing.assert_allclose(wrap_angle(np.array([0.0, 2 * np.pi + 0.1])), [0.0, 0.1])


def test_scaled_camera_keeps_geometry():
    small = CAM.scaled(0.5)
    assert (small.width, small.height) == (320, 240)
    assert np.isclose(small.fx, CAM.fx / 2)


def test_camera_attitude_from_autopilot_frames():
    down = rot_y(-np.pi / 2)  # a camera looking along body x, pitched down to the ground
    d = np.radians
    # level, heading east: a nadir view with image "up" pointing east
    np.testing.assert_allclose(
        camera_attitude(euler_ned(0, 0, d(90)) @ down), (d(90), 0, 0), atol=1e-9
    )
    # nose up: a body-fixed camera looks forward
    np.testing.assert_allclose(
        camera_attitude(euler_ned(0, d(5), 0) @ down), (0, d(5), 0), atol=1e-9
    )
    # right wing down: it looks to the left
    np.testing.assert_allclose(
        camera_attitude(euler_ned(d(10), 0, 0) @ down), (0, 0, -d(10)), atol=1e-9
    )


def test_camera_attitude_inverts_camera_rotation():
    rng = np.random.default_rng(3)
    ned_to_enu = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]])
    forward_camera = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    angles = rng.uniform((-3, -1, -1), (3, 1, 1), (20, 3))
    for yaw, pitch, roll in angles:
        r_ned = ned_to_enu.T @ camera_rotation(yaw, pitch, roll) @ forward_camera.T
        np.testing.assert_allclose(camera_attitude(r_ned), (yaw, pitch, roll), atol=1e-9)


def test_mount_misalignment_turns_with_the_camera():
    d = np.radians
    # level: a forward misalignment is just more pitch
    np.testing.assert_allclose(misaligned(0.3, 0.0, 0.0, d(4), 0.0), (0.3, d(4), 0.0), atol=1e-9)
    # banked 45 degrees: the same misalignment is not 4 degrees more pitch
    yaw, pitch, roll = misaligned(0.0, 0.0, d(45), d(-4), 0.0)
    assert abs(pitch - d(-4)) > d(1)
    R = camera_rotation(0.0, 0.0, d(45)) @ rot_x(d(-4))
    np.testing.assert_allclose(camera_rotation(yaw, pitch, roll), R, atol=1e-9)

"""The synthetic data generator itself.

If these are wrong, every other test is measuring the wrong thing.
"""

import math

import numpy as np
import pytest

from slamkit.extrinsics import (
    euler_to_matrix,
    make_transform,
    matrix_to_euler,
    rotation_geodesic_distance,
)
from slamkit.synthetic import (
    GRAVITY,
    apply_motion_distortion,
    circle_trajectory,
    corridor_scene,
    inject_z_ramp,
    inject_z_step,
    open_field_scene,
    perturb_transform,
    room_scene,
    shift_timestamps,
    simulate_imu,
    simulate_scan,
    spinning_lidar_rays,
    straight_line_trajectory,
    tunnel_scene,
    yaw_sweep_trajectory,
)


# ----------------------------------------------------------------- geometry
def test_ray_directions_are_unit_length_and_ordered_by_azimuth():
    dirs, rings, times = spinning_lidar_rays(16, 180, -15.0, 15.0, 0.1)
    assert len(dirs) == 16 * 180
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    assert rings.min() == 0 and rings.max() == 15
    assert np.all(np.diff(times) >= -1e-12)     # non-decreasing firing order
    assert times.max() < 0.1


def test_corridor_scan_has_the_right_width():
    scan = simulate_scan(corridor_scene(length=80.0, width=2.5, height=3.0),
                         make_transform(t=[0.0, 0.0, 1.0]),
                         n_rings=16, n_azimuth=180, max_range=40.0,
                         range_noise=0.0)
    y = scan.points[:, 1]
    assert y.max() == pytest.approx(1.25, abs=0.02)
    assert y.min() == pytest.approx(-1.25, abs=0.02)


def test_open_field_scan_is_all_below_the_sensor():
    scan = simulate_scan(open_field_scene(60.0), make_transform(t=[0, 0, 1.5]),
                         n_rings=32, n_azimuth=90, fov_down_deg=-45.0,
                         fov_up_deg=45.0, max_range=60.0, range_noise=0.0)
    assert len(scan) > 0
    assert np.all(scan.points[:, 2] < 0.0)      # sensor frame: ground is below
    assert np.allclose(scan.points[:, 2], -1.5, atol=1e-6)


def test_room_scan_is_bounded_by_the_room():
    scan = simulate_scan(room_scene(length=8.0, width=6.0, height=3.0),
                         make_transform(t=[0, 0, 1.2]), n_rings=32,
                         n_azimuth=180, fov_down_deg=-45.0, fov_up_deg=45.0,
                         max_range=40.0, range_noise=0.0)
    assert np.all(np.abs(scan.points[:, 0]) <= 4.01)
    assert np.all(np.abs(scan.points[:, 1]) <= 3.01)


def test_tunnel_scan_is_at_constant_radius_from_the_axis():
    scan = simulate_scan(tunnel_scene(length=120.0, radius=2.5, n_facets=48),
                         make_transform(t=[0, 0, 2.5]), n_rings=32,
                         n_azimuth=180, fov_down_deg=-45.0, fov_up_deg=45.0,
                         max_range=40.0, range_noise=0.0)
    radial = np.linalg.norm(scan.points[:, 1:], axis=1)
    # Faceted approximation, so a little under the true radius in places.
    assert radial.max() == pytest.approx(2.5, abs=0.05)
    assert radial.min() > 2.4


def test_scan_carries_ring_and_time_fields():
    scan = simulate_scan(room_scene(), make_transform(t=[0, 0, 1.2]),
                         n_rings=16, n_azimuth=90)
    assert len(scan.rings) == len(scan.points) == len(scan.times)
    assert scan.times.min() >= 0.0
    assert scan.times.max() < 0.1


def test_scan_is_deterministic_for_a_fixed_seed():
    args = dict(pose=make_transform(t=[0, 0, 1.2]), n_rings=16, n_azimuth=90,
                range_noise=0.02, seed=7)
    a = simulate_scan(room_scene(), **args)
    b = simulate_scan(room_scene(), **args)
    assert np.array_equal(a.points, b.points)


# ------------------------------------------------------------- trajectories
def test_straight_line_has_constant_velocity_and_no_rotation():
    traj = straight_line_trajectory(duration=5.0, rate=100.0, speed=2.0)
    assert len(traj) == 500
    assert np.allclose(traj.rotations, np.eye(3))
    assert traj.positions[-1, 0] == pytest.approx(2.0 * traj.times[-1], abs=1e-9)
    assert np.allclose(traj.angular_velocity_body(), 0.0, atol=1e-12)


def test_yaw_sweep_produces_the_commanded_yaw_amplitude():
    traj = yaw_sweep_trajectory(duration=8.0, rate=200.0, amplitude_deg=30.0,
                                period=4.0)
    yaws = np.array([matrix_to_euler(R)[2] for R in traj.rotations])
    assert math.degrees(yaws.max()) == pytest.approx(30.0, abs=0.5)
    assert math.degrees(yaws.min()) == pytest.approx(-30.0, abs=0.5)


def test_circle_trajectory_stays_on_the_circle():
    traj = circle_trajectory(duration=10.0, rate=100.0, radius=3.0, period=10.0)
    r = np.linalg.norm(traj.positions[:, :2], axis=1)
    assert np.allclose(r, 3.0, atol=1e-9)


def test_angular_velocity_matches_a_known_yaw_rate():
    traj = circle_trajectory(duration=10.0, rate=200.0, radius=3.0, period=10.0)
    w = traj.angular_velocity_body()
    expected = 2 * math.pi / 10.0
    assert np.allclose(w[5:-5, 2], expected, atol=1e-4)


# ---------------------------------------------------------------------- IMU
def test_stationary_imu_reads_gravity_up():
    traj = straight_line_trajectory(duration=2.0, rate=100.0, speed=0.0)
    imu = simulate_imu(traj)
    assert np.allclose(imu.accel[:, 2], GRAVITY, atol=1e-9)
    assert np.allclose(imu.accel[:, :2], 0.0, atol=1e-9)
    assert np.allclose(imu.gyro, 0.0, atol=1e-12)


def test_imu_gyro_is_expressed_in_the_imu_frame():
    """With a 90 deg yaw extrinsic, LiDAR yaw appears on the IMU's own Z."""
    traj = circle_trajectory(duration=10.0, rate=200.0, radius=3.0, period=10.0)
    T = make_transform(euler_to_matrix([0.0, 0.0, math.pi / 2]))
    plain = simulate_imu(traj)
    rotated = simulate_imu(traj, T)
    # A yaw-only extrinsic leaves the Z rate alone but swaps X and Y.
    assert np.allclose(plain.gyro[5:-5, 2], rotated.gyro[5:-5, 2], atol=1e-9)
    assert np.allclose(rotated.gyro[5:-5, 0], plain.gyro[5:-5, 1], atol=1e-9)


def test_imu_biases_are_added_exactly():
    traj = straight_line_trajectory(duration=2.0, rate=100.0, speed=0.0)
    imu = simulate_imu(traj, gyro_bias=[0.01, 0.0, -0.02],
                       accel_bias=[0.0, 0.1, 0.0])
    assert np.allclose(imu.gyro[:, 0], 0.01, atol=1e-12)
    assert np.allclose(imu.gyro[:, 2], -0.02, atol=1e-12)
    assert np.allclose(imu.accel[:, 1], 0.1, atol=1e-12)


def test_imu_noise_is_reproducible():
    traj = straight_line_trajectory(duration=2.0, rate=100.0, speed=0.0)
    a = simulate_imu(traj, gyro_noise=0.01, seed=3)
    b = simulate_imu(traj, gyro_noise=0.01, seed=3)
    c = simulate_imu(traj, gyro_noise=0.01, seed=4)
    assert np.array_equal(a.gyro, b.gyro)
    assert not np.array_equal(a.gyro, c.gyro)


# ----------------------------------------------------------- defect injection
def test_perturb_transform_applies_the_requested_error():
    T = make_transform(np.eye(3), [0.1, 0.0, 0.0])
    bad = perturb_transform(T, rpy_error_deg=[0.0, 0.0, 5.0],
                            translation_error=[0.02, 0.0, 0.0])
    angle = math.degrees(rotation_geodesic_distance(T[:3, :3], bad[:3, :3]))
    assert angle == pytest.approx(5.0, abs=1e-9)
    assert bad[0, 3] == pytest.approx(0.12, abs=1e-12)


def test_shift_timestamps_offset_drift_and_jitter():
    t = np.linspace(0.0, 60.0, 601)
    assert np.allclose(shift_timestamps(t, offset=0.05) - t, 0.05)
    drifted = shift_timestamps(t, drift_ppm=100.0)
    assert (drifted[-1] - t[-1]) == pytest.approx(60.0 * 1e-4, rel=1e-9)
    jittered = shift_timestamps(t, jitter_std=0.002, seed=1)
    assert np.std(jittered - t) == pytest.approx(0.002, rel=0.2)


def test_motion_distortion_is_the_inverse_of_deskewing():
    P = np.random.default_rng(0).uniform(-5.0, 5.0, (100, 3))
    undistorted = apply_motion_distortion(P, np.zeros(100), [1.0, 0, 0], [0, 0, 1.0])
    assert np.allclose(undistorted, P)     # zero elapsed time = no distortion


def test_inject_z_ramp_and_step():
    t = np.linspace(0.0, 60.0, 61)
    poses = np.tile(np.eye(4), (61, 1, 1))
    ramped = inject_z_ramp(poses, t, rate_m_per_s=0.01)
    assert ramped[-1, 2, 3] == pytest.approx(0.6, abs=1e-12)
    stepped = inject_z_step(poses, index=30, jump_m=0.4)
    assert stepped[29, 2, 3] == 0.0
    assert stepped[30, 2, 3] == pytest.approx(0.4)
    assert stepped[-1, 2, 3] == pytest.approx(0.4)

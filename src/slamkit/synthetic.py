"""Synthetic scenes, scans, trajectories and IMU streams -- with injectable defects.

Every diagnostic in this toolkit is tested against data generated here, which
means the tests can state *ground truth*: "inject a 45 ms time offset, assert
the estimator recovers 45 +/- 5 ms".  You cannot do that with a real bag.

The scenes are ray-cast against real planar geometry rather than sampled from
a random distribution, so a corridor really does produce a corridor's normal
distribution, and the degeneracy analysis is being tested on the physics
rather than on a mock.

Defect injectors provided:

* :func:`perturb_transform` -- extrinsic rotation and translation error.
* :func:`shift_timestamps` -- constant offset, linear drift, jitter.
* :func:`simulate_imu` -- accelerometer/gyro bias, white noise, gravity error.
* :func:`apply_motion_distortion` -- un-deskewed sweep (the "smeared cloud").
* :func:`inject_z_ramp` / :func:`inject_z_step` -- the two z-drift signatures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .extrinsics import euler_to_matrix, rotation_exp, rotation_log

__all__ = [
    "Panel",
    "Scene",
    "make_panel",
    "corridor_scene",
    "room_scene",
    "open_field_scene",
    "tunnel_scene",
    "Scan",
    "spinning_lidar_rays",
    "simulate_scan",
    "Trajectory",
    "straight_line_trajectory",
    "yaw_sweep_trajectory",
    "circle_trajectory",
    "ImuStream",
    "simulate_imu",
    "perturb_transform",
    "shift_timestamps",
    "apply_motion_distortion",
    "inject_z_ramp",
    "inject_z_step",
    "GRAVITY",
]

GRAVITY = 9.80665
"""Standard gravity, m/s^2. LIO-SAM's ``imuGravity`` defaults to 9.80511."""

_EPS = 1e-12


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
@dataclass
class Panel:
    """A finite planar rectangle in world coordinates."""

    center: np.ndarray
    normal: np.ndarray
    u: np.ndarray
    v: np.ndarray
    half_u: float
    half_v: float


def make_panel(center: Sequence[float], normal: Sequence[float],
               half_u: float, half_v: float,
               up_hint: Sequence[float] = (0.0, 0.0, 1.0)) -> Panel:
    """Build a :class:`Panel` from a centre, a normal and two half-extents."""
    c = np.asarray(center, dtype=float).reshape(3)
    n = np.asarray(normal, dtype=float).reshape(3)
    n = n / max(float(np.linalg.norm(n)), _EPS)
    up = np.asarray(up_hint, dtype=float).reshape(3)
    if abs(float(n @ up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    u = np.cross(up, n)
    u = u / max(float(np.linalg.norm(u)), _EPS)
    v = np.cross(n, u)
    return Panel(c, n, u, v, float(half_u), float(half_v))


class Scene:
    """A set of :class:`Panel` surfaces that a simulated LiDAR can hit."""

    def __init__(self, panels: Sequence[Panel], name: str = "scene") -> None:
        self.panels = list(panels)
        self.name = name

    def cast(self, origin: np.ndarray, directions: np.ndarray,
             max_range: float = 100.0, min_range: float = 0.1
             ) -> Tuple[np.ndarray, np.ndarray]:
        """Cast rays from ``origin`` along unit ``directions`` (``(M, 3)``).

        Returns ``(ranges, hit_mask)``; ``ranges`` is ``inf`` where nothing was
        hit inside ``max_range``.
        """
        o = np.asarray(origin, dtype=float).reshape(3)
        D = np.asarray(directions, dtype=float).reshape(-1, 3)
        best = np.full(len(D), np.inf)
        for p in self.panels:
            denom = D @ p.normal
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((p.center - o) @ p.normal) / denom
            valid = np.isfinite(t) & (np.abs(denom) > 1e-9) & (t > min_range) & (t < max_range)
            if not np.any(valid):
                continue
            hit = o + D[valid] * t[valid, None]
            rel = hit - p.center
            inside = (np.abs(rel @ p.u) <= p.half_u) & (np.abs(rel @ p.v) <= p.half_v)
            idx = np.where(valid)[0][inside]
            tv = t[valid][inside]
            better = tv < best[idx]
            best[idx[better]] = tv[better]
        return best, np.isfinite(best)


def corridor_scene(length: float = 40.0, width: float = 2.5,
                   height: float = 3.0) -> Scene:
    """A straight corridor along +X: two walls, floor, ceiling, two end caps.

    The canonical degenerate scene.  Every normal points along Y or Z; nothing
    points along X except the far end caps, which are usually out of range.
    """
    hl, hw = length / 2.0, width / 2.0
    panels = [
        make_panel([0, -hw, height / 2], [0, 1, 0], hl, height / 2),
        make_panel([0, hw, height / 2], [0, -1, 0], hl, height / 2),
        make_panel([0, 0, 0.0], [0, 0, 1], hl, hw),
        make_panel([0, 0, height], [0, 0, -1], hl, hw),
        make_panel([-hl, 0, height / 2], [1, 0, 0], hw, height / 2),
        make_panel([hl, 0, height / 2], [-1, 0, 0], hw, height / 2),
    ]
    return Scene(panels, "corridor")


def room_scene(length: float = 8.0, width: float = 6.0, height: float = 3.0,
               pillars: bool = True) -> Scene:
    """A closed room, optionally with two pillars. Well constrained on all axes."""
    hl, hw = length / 2.0, width / 2.0
    panels = [
        make_panel([0, -hw, height / 2], [0, 1, 0], hl, height / 2),
        make_panel([0, hw, height / 2], [0, -1, 0], hl, height / 2),
        make_panel([-hl, 0, height / 2], [1, 0, 0], hw, height / 2),
        make_panel([hl, 0, height / 2], [-1, 0, 0], hw, height / 2),
        make_panel([0, 0, 0.0], [0, 0, 1], hl, hw),
        make_panel([0, 0, height], [0, 0, -1], hl, hw),
    ]
    if pillars:
        for cx, cy in ((hl / 2, hw / 2), (-hl / 2, -hw / 2)):
            s = 0.25
            for nx, ny in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                panels.append(
                    make_panel([cx + nx * s, cy + ny * s, height / 2],
                               [nx, ny, 0], s, height / 2)
                )
    return Scene(panels, "room")


def open_field_scene(extent: float = 60.0) -> Scene:
    """Flat ground and nothing else. Z observable, X and Y free."""
    return Scene([make_panel([0, 0, 0.0], [0, 0, 1], extent, extent)], "open_field")


def tunnel_scene(length: float = 60.0, radius: float = 2.5,
                 n_facets: int = 24) -> Scene:
    """A round tunnel along +X, approximated by ``n_facets`` flat strips.

    Worse than a corridor: the cross-section is smooth, so roll about X is
    unobservable as well as translation along X.
    """
    panels: List[Panel] = []
    half_arc = math.pi / n_facets
    strip_half = radius * math.tan(half_arc)
    for i in range(n_facets):
        ang = 2 * math.pi * (i + 0.5) / n_facets
        cy, cz = radius * math.cos(ang), radius * math.sin(ang)
        inward = [0.0, -math.cos(ang), -math.sin(ang)]
        panels.append(
            make_panel([0.0, cy, cz + radius], inward,
                       strip_half, length / 2.0, up_hint=[1.0, 0.0, 0.0])
        )
    return Scene(panels, "tunnel")


# --------------------------------------------------------------------------
# LiDAR simulation
# --------------------------------------------------------------------------
@dataclass
class Scan:
    """One simulated sweep, expressed in the sensor frame."""

    points: np.ndarray
    """``(N, 3)`` metres."""

    rings: np.ndarray
    """``(N,)`` int ring index."""

    times: np.ndarray
    """``(N,)`` seconds relative to the start of the sweep."""

    stamp: float = 0.0
    """Message timestamp (start of sweep), seconds."""

    pose: np.ndarray = field(default_factory=lambda: np.eye(4), repr=False)
    """Ground-truth sensor pose in world at ``stamp``."""

    def __len__(self) -> int:
        return len(self.points)


def spinning_lidar_rays(
    n_rings: int = 16,
    n_azimuth: int = 360,
    fov_down_deg: float = -15.0,
    fov_up_deg: float = 15.0,
    scan_period: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit ray directions for a spinning LiDAR, plus ring index and firing time.

    Returns ``(directions (M, 3), rings (M,), times (M,))`` with times spread
    across ``scan_period`` in azimuth order -- which is what makes motion
    distortion reproducible.
    """
    elev = np.radians(np.linspace(fov_down_deg, fov_up_deg, n_rings))
    az = np.linspace(0.0, 2 * math.pi, n_azimuth, endpoint=False)
    A, E = np.meshgrid(az, elev, indexing="ij")  # azimuth-major = firing order
    dirs = np.stack(
        [np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1
    ).reshape(-1, 3)
    rings = np.tile(np.arange(n_rings), n_azimuth)
    times = np.repeat(np.linspace(0.0, scan_period, n_azimuth, endpoint=False), n_rings)
    return dirs, rings, times


def simulate_scan(
    scene: Scene,
    pose: Optional[np.ndarray] = None,
    n_rings: int = 16,
    n_azimuth: int = 360,
    fov_down_deg: float = -15.0,
    fov_up_deg: float = 15.0,
    max_range: float = 80.0,
    range_noise: float = 0.01,
    scan_period: float = 0.1,
    stamp: float = 0.0,
    seed: int = 0,
) -> Scan:
    """Ray-cast one sweep of a spinning LiDAR placed at ``pose`` in ``scene``.

    ``pose`` is the 4x4 sensor pose in world.  The returned points are in the
    **sensor** frame, as a driver would publish them.
    """
    T = np.eye(4) if pose is None else np.asarray(pose, dtype=float)
    dirs, rings, times = spinning_lidar_rays(
        n_rings, n_azimuth, fov_down_deg, fov_up_deg, scan_period
    )
    world_dirs = dirs @ T[:3, :3].T
    ranges, hit = scene.cast(T[:3, 3], world_dirs, max_range=max_range)
    rng = np.random.default_rng(seed)
    r = ranges[hit]
    if range_noise > 0:
        r = r + rng.normal(0.0, range_noise, size=r.shape)
    pts = dirs[hit] * r[:, None]
    return Scan(points=pts, rings=rings[hit], times=times[hit], stamp=stamp, pose=T)


# --------------------------------------------------------------------------
# Trajectories
# --------------------------------------------------------------------------
@dataclass
class Trajectory:
    """Time-stamped ground-truth poses."""

    times: np.ndarray
    """``(N,)`` seconds."""

    poses: np.ndarray
    """``(N, 4, 4)`` sensor pose in world."""

    def __len__(self) -> int:
        return len(self.times)

    @property
    def positions(self) -> np.ndarray:
        return self.poses[:, :3, 3]

    @property
    def rotations(self) -> np.ndarray:
        return self.poses[:, :3, :3]

    def angular_velocity_body(self) -> np.ndarray:
        """Body-frame angular rate, rad/s, by finite differences of the rotations.

        Uses central differences in the interior and one-sided at the ends, so
        the result is the same length as the trajectory.
        """
        n = len(self)
        w = np.zeros((n, 3), dtype=float)
        R = self.rotations
        t = self.times
        for i in range(n):
            a = max(i - 1, 0)
            b = min(i + 1, n - 1)
            dt = float(t[b] - t[a])
            if dt <= 0:
                continue
            w[i] = rotation_log(R[a].T @ R[b]) / dt
        return w

    def linear_acceleration_world(self) -> np.ndarray:
        """World-frame acceleration, m/s^2, by second differences of position."""
        p = self.positions
        t = self.times
        n = len(t)
        a = np.zeros((n, 3), dtype=float)
        for i in range(1, n - 1):
            dt1 = float(t[i] - t[i - 1])
            dt2 = float(t[i + 1] - t[i])
            if dt1 <= 0 or dt2 <= 0:
                continue
            v1 = (p[i] - p[i - 1]) / dt1
            v2 = (p[i + 1] - p[i]) / dt2
            a[i] = (v2 - v1) / (0.5 * (dt1 + dt2))
        if n > 2:
            a[0] = a[1]
            a[-1] = a[-2]
        return a


def straight_line_trajectory(duration: float = 10.0, rate: float = 100.0,
                             speed: float = 1.0,
                             direction: Sequence[float] = (1.0, 0.0, 0.0),
                             start: Sequence[float] = (0.0, 0.0, 1.0)) -> Trajectory:
    """Constant-velocity straight line with fixed orientation."""
    t = np.arange(0.0, duration, 1.0 / rate)
    d = np.asarray(direction, dtype=float)
    d = d / max(float(np.linalg.norm(d)), _EPS)
    p0 = np.asarray(start, dtype=float)
    poses = np.tile(np.eye(4), (len(t), 1, 1))
    poses[:, :3, 3] = p0 + np.outer(t * speed, d)
    return Trajectory(t, poses)


def yaw_sweep_trajectory(duration: float = 10.0, rate: float = 100.0,
                         amplitude_deg: float = 30.0, period: float = 4.0,
                         speed: float = 0.5,
                         start: Sequence[float] = (0.0, 0.0, 1.0)) -> Trajectory:
    """Forward motion with a sinusoidal yaw. The excitation you need for calibration.

    A pure straight line leaves the extrinsic rotation unobservable; a pure
    yaw leaves roll and pitch of the extrinsic unobservable.  Use
    :func:`circle_trajectory` combined with this, or add a pitch component,
    when you actually want to solve for a full extrinsic.
    """
    t = np.arange(0.0, duration, 1.0 / rate)
    yaw = math.radians(amplitude_deg) * np.sin(2 * math.pi * t / period)
    poses = np.tile(np.eye(4), (len(t), 1, 1))
    p0 = np.asarray(start, dtype=float)
    for i, (ti, y) in enumerate(zip(t, yaw)):
        poses[i, :3, :3] = euler_to_matrix([0.0, 0.0, float(y)])
        poses[i, :3, 3] = p0 + np.array([speed * ti, 0.0, 0.0])
    return Trajectory(t, poses)


def circle_trajectory(duration: float = 20.0, rate: float = 100.0,
                      radius: float = 3.0, period: float = 10.0,
                      height: float = 1.0,
                      tilt_deg: float = 0.0) -> Trajectory:
    """Constant-speed circle with the sensor facing along the path.

    ``tilt_deg`` adds a rolling component so the trajectory excites more than
    one rotation axis, which is what :func:`slamkit.extrinsics.solve_rotation_hand_eye`
    needs for a well-conditioned solve.
    """
    t = np.arange(0.0, duration, 1.0 / rate)
    w = 2 * math.pi / period
    poses = np.tile(np.eye(4), (len(t), 1, 1))
    for i, ti in enumerate(t):
        a = w * ti
        poses[i, :3, 3] = [radius * math.cos(a), radius * math.sin(a), height]
        roll = math.radians(tilt_deg) * math.sin(2 * a)
        pitch = math.radians(tilt_deg) * math.cos(3 * a)
        poses[i, :3, :3] = euler_to_matrix([roll, pitch, a + math.pi / 2])
    return Trajectory(t, poses)


# --------------------------------------------------------------------------
# IMU simulation
# --------------------------------------------------------------------------
@dataclass
class ImuStream:
    """Simulated IMU samples in the IMU body frame."""

    times: np.ndarray
    gyro: np.ndarray
    """``(N, 3)`` rad/s."""
    accel: np.ndarray
    """``(N, 3)`` m/s^2 specific force -- includes gravity, like a real IMU."""

    def __len__(self) -> int:
        return len(self.times)


def simulate_imu(
    traj: Trajectory,
    T_lidar_imu: Optional[np.ndarray] = None,
    gyro_bias: Sequence[float] = (0.0, 0.0, 0.0),
    accel_bias: Sequence[float] = (0.0, 0.0, 0.0),
    gyro_noise: float = 0.0,
    accel_noise: float = 0.0,
    gravity: float = GRAVITY,
    seed: int = 0,
) -> ImuStream:
    """Generate IMU measurements consistent with ``traj``.

    ``traj`` is the *LiDAR* trajectory.  ``T_lidar_imu`` is the extrinsic
    (``R_lidar_from_imu`` in the rotation block), so passing a non-identity
    transform gives you an IMU stream in a genuinely different frame -- which
    is what makes the extrinsic solvers testable.

    The lever arm is applied to the angular rate only (the rate is frame
    invariant up to rotation); the centripetal term on acceleration is
    included so that a bad lever arm produces the right kind of error.
    """
    R_li = np.eye(3) if T_lidar_imu is None else np.asarray(T_lidar_imu)[:3, :3]
    t_li = np.zeros(3) if T_lidar_imu is None else np.asarray(T_lidar_imu)[:3, 3]
    rng = np.random.default_rng(seed)
    w_lidar = traj.angular_velocity_body()
    a_world = traj.linear_acceleration_world()
    g_world = np.array([0.0, 0.0, -gravity])
    n = len(traj)
    gyro = np.zeros((n, 3))
    accel = np.zeros((n, 3))
    R_il = R_li.T  # IMU-from-LiDAR
    for i in range(n):
        R_wl = traj.poses[i, :3, :3]
        w_i = R_il @ w_lidar[i]
        gyro[i] = w_i
        # Specific force at the IMU location, expressed in the IMU frame.
        a_lidar_body = R_wl.T @ (a_world[i] - g_world)
        lever = -R_il @ t_li  # IMU origin in the LiDAR frame, rotated to IMU frame
        centripetal = np.cross(w_i, np.cross(w_i, lever))
        accel[i] = R_il @ a_lidar_body + centripetal
    gyro = gyro + np.asarray(gyro_bias, dtype=float)
    accel = accel + np.asarray(accel_bias, dtype=float)
    if gyro_noise > 0:
        gyro = gyro + rng.normal(0.0, gyro_noise, gyro.shape)
    if accel_noise > 0:
        accel = accel + rng.normal(0.0, accel_noise, accel.shape)
    return ImuStream(traj.times.copy(), gyro, accel)


# --------------------------------------------------------------------------
# Defect injectors
# --------------------------------------------------------------------------
def perturb_transform(T: np.ndarray, rpy_error_deg: Sequence[float] = (0.0, 0.0, 0.0),
                      translation_error: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Apply a small rotation and translation error to a transform.

    Models the realistic case: you measured the mount from a CAD drawing and
    it is a couple of degrees and a centimetre off.
    """
    T = np.asarray(T, dtype=float)
    dR = euler_to_matrix(np.radians(np.asarray(rpy_error_deg, dtype=float)))
    out = T.copy()
    out[:3, :3] = dR @ T[:3, :3]
    out[:3, 3] = T[:3, 3] + np.asarray(translation_error, dtype=float)
    return out


def shift_timestamps(times: np.ndarray, offset: float = 0.0,
                     drift_ppm: float = 0.0, jitter_std: float = 0.0,
                     seed: int = 0) -> np.ndarray:
    """Corrupt a timestamp series the way a real system does.

    Parameters
    ----------
    offset:
        Constant offset, seconds. Positive means these stamps run *late*.
    drift_ppm:
        Linear clock drift in parts per million relative to the first sample.
        A cheap MCU crystal is +/-50 ppm, which is 3 ms per minute -- fine for
        a 30 s bag, fatal for a 20 minute survey.
    jitter_std:
        Zero-mean Gaussian jitter, seconds. USB and non-realtime kernels give
        1-5 ms; that is what makes ROS receive-time stamping unusable for
        LiDAR-inertial fusion.
    """
    t = np.asarray(times, dtype=float).copy()
    t0 = float(t[0]) if len(t) else 0.0
    out = t + offset + (t - t0) * (drift_ppm * 1e-6)
    if jitter_std > 0:
        rng = np.random.default_rng(seed)
        out = out + rng.normal(0.0, jitter_std, out.shape)
    return out


def apply_motion_distortion(points: np.ndarray, rel_times: np.ndarray,
                            linear_velocity: Sequence[float] = (0.0, 0.0, 0.0),
                            angular_velocity: Sequence[float] = (0.0, 0.0, 0.0)
                            ) -> np.ndarray:
    """Add the smear a moving sensor produces when a scan is *not* deskewed.

    This is the forward model: each point is re-expressed in the pose the
    sensor had when that point was fired, using a constant twist.  Feeding the
    result to :func:`slamkit.cloud.deskew_points` with the matching
    ``T_start_end`` recovers the original cloud, which is how the deskewing
    test is written.
    """
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    s = np.asarray(rel_times, dtype=float).reshape(-1)
    if len(s) != len(P):
        raise ValueError("rel_times length must match points")
    v = np.asarray(linear_velocity, dtype=float).reshape(3)
    w = np.asarray(angular_velocity, dtype=float).reshape(3)
    out = np.empty_like(P)
    for i in range(len(P)):
        Ri = rotation_exp(w * s[i])
        out[i] = Ri.T @ (P[i] - v * s[i])
    return out


def inject_z_ramp(poses: np.ndarray, times: np.ndarray,
                  rate_m_per_s: float) -> np.ndarray:
    """Add a constant-slope vertical drift -- the IMU-bias / gravity signature."""
    out = np.array(poses, dtype=float, copy=True)
    t = np.asarray(times, dtype=float)
    out[:, 2, 3] += rate_m_per_s * (t - t[0])
    return out


def inject_z_step(poses: np.ndarray, index: int, jump_m: float) -> np.ndarray:
    """Add a single discontinuity in z -- the loop-closure / plane-snap signature."""
    out = np.array(poses, dtype=float, copy=True)
    out[index:, 2, 3] += jump_m
    return out

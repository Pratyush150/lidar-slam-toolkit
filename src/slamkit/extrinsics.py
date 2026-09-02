"""SE(3) maths and LiDAR->IMU extrinsic calibration checks.

Why this module exists
----------------------
A LiDAR-inertial system has exactly one hard geometric input from you: the
rigid transform between the LiDAR and the IMU.  Everything else the solver
estimates.  If that transform is wrong the optimiser will happily converge --
to a wrong answer -- and the failure looks like a tuning problem.  It is not.

Four mistakes account for nearly every broken extrinsic in the wild:

1. **Transposed rotation.**  You wrote ``R_imu_from_lidar`` where the config
   wanted ``R_lidar_from_imu``.  For a 90 degree yaw offset the two differ,
   for a 180 degree offset they do not, which is why the bug survives the
   first bench test and appears on the robot.
2. **Degrees pasted into a radians field.**  ``[0, 0, 90]`` in a field that
   wants radians is a 90 *radian* yaw = 5156 degrees.
3. **Translation expressed in the wrong frame / wrong direction.**  You
   measured "the IMU is 12 cm behind the LiDAR" and typed ``[0.12, 0, 0]``
   into a field defined as the LiDAR origin expressed in IMU coordinates.
4. **Left-handed axes.**  A vendor drawing with Z down, or a sign flipped by
   hand to "make it look right", gives ``det(R) = -1``.

Conventions used here
---------------------
* A transform is a 4x4 homogeneous matrix ``T_a_b`` that maps a point
  expressed in frame ``b`` into frame ``a``:  ``p_a = T_a_b @ p_b``.
* Rotations are right-handed, ``det(R) = +1``, ``R @ R.T = I``.
* Euler angles are roll-pitch-yaw applied as ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``
  (intrinsic Z-Y-X), which is what ROS, LIO-SAM and ``tf2`` use.
* Quaternions are ``(x, y, z, w)`` -- ROS order, scalar last.  ``tf2`` and
  ``geometry_msgs/Quaternion`` use this order; Eigen's constructor is
  ``Quaterniond(w, x, y, z)``.  Mixing those two is its own bug class, so
  :func:`quaternion_to_matrix` validates the norm and nothing else -- it
  cannot know which order you meant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .findings import Finding, Report, Severity

__all__ = [
    "skew",
    "unskew",
    "euler_to_matrix",
    "matrix_to_euler",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "rotation_log",
    "rotation_exp",
    "rotation_angle",
    "rotation_geodesic_distance",
    "make_transform",
    "invert_transform",
    "compose_transforms",
    "transform_points",
    "transform_vectors",
    "is_rotation_matrix",
    "orthonormalize",
    "kabsch_rotation",
    "solve_rotation_hand_eye",
    "solve_rotation_from_angular_rates",
    "ExtrinsicCheckConfig",
    "check_rotation_valid",
    "check_handedness",
    "check_degrees_as_radians",
    "check_transposed_rotation",
    "check_translation_direction",
    "check_translation_magnitude",
    "validate_extrinsics",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# Small SO(3) helpers
# --------------------------------------------------------------------------
def skew(v: Sequence[float]) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix ``[v]_x`` with ``[v]_x u = v x u``."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float
    )


def unskew(m: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew`: pull the vector back out of ``[v]_x``."""
    m = np.asarray(m, dtype=float)
    return np.array([m[2, 1], m[0, 2], m[1, 0]], dtype=float)


def euler_to_matrix(rpy: Sequence[float], degrees: bool = False) -> np.ndarray:
    """Roll-pitch-yaw to rotation matrix, ``R = Rz(yaw) Ry(pitch) Rx(roll)``.

    Parameters
    ----------
    rpy:
        ``(roll, pitch, yaw)``.
    degrees:
        Interpret ``rpy`` as degrees.  Default is radians, matching every ROS
        config field.  Pass ``degrees=True`` only when you know your source is
        a datasheet drawing.
    """
    r, p, y = (float(a) for a in np.asarray(rpy, dtype=float).reshape(3))
    if degrees:
        r, p, y = math.radians(r), math.radians(p), math.radians(y)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def matrix_to_euler(R: np.ndarray, degrees: bool = False) -> np.ndarray:
    """Rotation matrix to ``(roll, pitch, yaw)``, inverse of :func:`euler_to_matrix`.

    Near pitch = +/-90 degrees roll and yaw are not separable (gimbal lock);
    this returns roll = 0 and puts the whole rotation into yaw, which is the
    conventional choice and is still a correct decomposition.
    """
    R = np.asarray(R, dtype=float)
    sp = -R[2, 0]
    sp = max(-1.0, min(1.0, float(sp)))
    pitch = math.asin(sp)
    if abs(sp) > 1.0 - 1e-9:  # gimbal lock
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    out = np.array([roll, pitch, yaw], dtype=float)
    return np.degrees(out) if degrees else out


def quaternion_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Quaternion ``(x, y, z, w)`` -- ROS order, scalar last -- to a rotation matrix."""
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < _EPS:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion ``(x, y, z, w)``, with ``w >= 0``.

    Uses Shepperd's method (branch on the largest diagonal term) so it stays
    numerically well conditioned at 180 degree rotations, where the naive
    ``w = sqrt(1 + trace)/2`` formula divides by ~0.
    """
    R = np.asarray(R, dtype=float)
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    if q[3] < 0:
        q = -q
    return q / float(np.linalg.norm(q))


def rotation_log(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm: rotation matrix to rotation vector (axis * angle, rad)."""
    R = np.asarray(R, dtype=float)
    cos_theta = (float(np.trace(R)) - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        # Small angle: log(R) ~ (R - R^T)/2, exact to third order.
        return unskew((R - R.T) / 2.0)
    if theta > math.pi - 1e-6:
        # Near pi the antisymmetric part vanishes; recover the axis from
        # the symmetric part instead.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = int(np.argmax(axis))
        if axis[k] > _EPS:
            axis = A[:, k] / axis[k]
        axis = axis / max(float(np.linalg.norm(axis)), _EPS)
        return axis * theta
    return unskew((R - R.T)) * (theta / (2.0 * math.sin(theta)))


def rotation_exp(w: Sequence[float]) -> np.ndarray:
    """SO(3) exponential (Rodrigues): rotation vector (rad) to rotation matrix."""
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + skew(w)
    k = w / theta
    K = skew(k)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def rotation_angle(R: np.ndarray) -> float:
    """Magnitude of the rotation encoded by ``R``, in radians, in ``[0, pi]``."""
    cos_theta = (float(np.trace(np.asarray(R, dtype=float))) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, cos_theta)))


def rotation_geodesic_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Angle in radians of the rotation that takes ``A`` to ``B``."""
    return rotation_angle(np.asarray(A, dtype=float).T @ np.asarray(B, dtype=float))


# --------------------------------------------------------------------------
# SE(3)
# --------------------------------------------------------------------------
def make_transform(R: Optional[np.ndarray] = None,
                   t: Optional[Sequence[float]] = None) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a rotation and a translation."""
    T = np.eye(4, dtype=float)
    if R is not None:
        R = np.asarray(R, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {R.shape}")
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform, using ``R^T`` rather than a general solve.

    ``inv(T_a_b) = T_b_a``.  Note ``t_b_a = -R^T t_a_b``: the *sign flip alone*
    is not the inverse.  Getting this wrong is the source of the classic
    "translation points the wrong way" extrinsic bug, because for a pure
    translation (R = I) the wrong formula happens to be right.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got {T.shape}")
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=float)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Left-to-right chain: ``compose(T_a_b, T_b_c) -> T_a_c``."""
    out = np.eye(4, dtype=float)
    for T in transforms:
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"expected 4x4 transforms, got {T.shape}")
        out = out @ T
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply ``T`` to an ``(N, 3)`` array of points (rotation *and* translation)."""
    T = np.asarray(T, dtype=float)
    P = np.asarray(points, dtype=float)
    single = P.ndim == 1
    P = np.atleast_2d(P)
    if P.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {P.shape}")
    out = P @ T[:3, :3].T + T[:3, 3]
    return out[0] if single else out


def transform_vectors(T: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Apply only the rotation part of ``T``.

    Use this for velocities, accelerations, angular rates and surface normals.
    Rotating an IMU acceleration vector with the full transform (adding the
    translation) is a real and surprisingly common bug: it adds a constant
    metres-per-second-squared offset that the estimator absorbs as an
    enormous accelerometer bias, and the map then pitches slowly.
    """
    T = np.asarray(T, dtype=float)
    V = np.asarray(vectors, dtype=float)
    single = V.ndim == 1
    V = np.atleast_2d(V)
    if V.shape[1] != 3:
        raise ValueError(f"vectors must be (N, 3), got {V.shape}")
    out = V @ T[:3, :3].T
    return out[0] if single else out


def is_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> bool:
    """True if ``R`` is orthonormal with ``det = +1`` to within ``tol``."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    if not np.all(np.isfinite(R)):
        return False
    if float(np.max(np.abs(R.T @ R - np.eye(3)))) > tol:
        return False
    return abs(float(np.linalg.det(R)) - 1.0) <= max(tol, 1e-9)


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest proper rotation to ``R`` in the Frobenius sense (polar/SVD).

    Useful after hand-editing a matrix in a YAML file: rounding nine numbers
    to three decimals leaves a matrix that is not quite orthonormal, and some
    solvers silently accumulate that error into scale drift.
    """
    R = np.asarray(R, dtype=float)
    U, _, Vt = np.linalg.svd(R)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    return U @ D @ Vt


# --------------------------------------------------------------------------
# Rotation estimation
# --------------------------------------------------------------------------
def kabsch_rotation(source: np.ndarray, target: np.ndarray,
                    weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Least-squares rotation ``R`` minimising ``sum_i w_i ||target_i - R source_i||^2``.

    Both arrays are ``(N, 3)``.  The reflection case is handled explicitly so
    the result is always a proper rotation -- without that guard, noisy or
    near-planar data can hand you a ``det = -1`` "rotation" that mirrors your
    map.
    """
    A = np.asarray(source, dtype=float).reshape(-1, 3)
    B = np.asarray(target, dtype=float).reshape(-1, 3)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {B.shape}")
    if A.shape[0] < 2:
        raise ValueError("need at least 2 correspondences (3 non-coplanar for a unique solve)")
    if weights is None:
        w = np.ones(A.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != A.shape[0]:
            raise ValueError("weights length must match correspondences")
    H = (A * w[:, None]).T @ B
    U, _, Vt = np.linalg.svd(H)
    d = float(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, math.copysign(1.0, d) if d != 0 else 1.0])
    return Vt.T @ D @ U.T


def solve_rotation_hand_eye(
    rotations_a: Sequence[np.ndarray],
    rotations_b: Sequence[np.ndarray],
    min_angle_rad: float = math.radians(2.0),
) -> Tuple[np.ndarray, dict]:
    """Estimate ``R_a_b`` from paired incremental rotations (hand-eye ``AX = XB``).

    Given the motion of the LiDAR between two instants (``A_i``, from scan
    matching) and the motion of the IMU over the same interval (``B_i``, from
    integrating the gyro), the extrinsic rotation ``X = R_a_b`` satisfies
    ``A_i = X B_i X^T``.  Taking the SO(3) logarithm of both sides turns that
    into a plain vector correspondence, ``log(A_i) = X log(B_i)``, which is a
    Kabsch problem.

    Motions smaller than ``min_angle_rad`` are dropped: their log axes are
    dominated by noise, and including them biases the solve.  **You need
    rotation about at least two non-parallel axes.**  Driving a ground robot
    that only ever yaws leaves roll and pitch of the extrinsic unobservable --
    the solve will return *an* answer and the residual will look fine.

    Returns
    -------
    (R, info)
        ``R`` is the 3x3 estimate.  ``info`` carries ``n_used``,
        ``residual_rad`` (RMS angular residual of ``log(A) - R log(B)``),
        ``axis_condition`` (condition number of the used-axis scatter matrix;
        > ~50 means the excitation was nearly single-axis and the result is
        only trustworthy about that axis) and ``well_conditioned``.
    """
    a_vecs: List[np.ndarray] = []
    b_vecs: List[np.ndarray] = []
    for A, B in zip(rotations_a, rotations_b):
        la = rotation_log(np.asarray(A, dtype=float))
        lb = rotation_log(np.asarray(B, dtype=float))
        if min(np.linalg.norm(la), np.linalg.norm(lb)) < min_angle_rad:
            continue
        a_vecs.append(la)
        b_vecs.append(lb)
    if len(a_vecs) < 2:
        raise ValueError(
            f"only {len(a_vecs)} motions exceed {math.degrees(min_angle_rad):.1f} deg; "
            "record a segment with real rotation on at least two axes"
        )
    A = np.asarray(b_vecs)  # source: IMU-frame axes
    B = np.asarray(a_vecs)  # target: LiDAR-frame axes
    R = kabsch_rotation(A, B)
    resid = B - A @ R.T
    residual = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
    scatter = A.T @ A / max(len(A), 1)
    eig = np.sort(np.abs(np.linalg.eigvalsh(scatter)))[::-1]
    cond = float(eig[0] / eig[-1]) if eig[-1] > 1e-12 else float("inf")
    info = {
        "n_used": int(len(a_vecs)),
        "n_input": int(min(len(rotations_a), len(rotations_b))),
        "residual_rad": residual,
        "residual_deg": math.degrees(residual),
        "axis_condition": cond,
        "well_conditioned": bool(cond < 50.0),
    }
    return R, info


def solve_rotation_from_angular_rates(
    omega_a: np.ndarray,
    omega_b: np.ndarray,
    min_rate: float = 0.05,
) -> Tuple[np.ndarray, dict]:
    """Estimate ``R_a_b`` from time-aligned angular-rate vectors.

    Angular velocity of a rigid body is frame-covariant: ``w_a = R_a_b w_b``.
    So if you have gyro samples in the IMU frame and scan-matched angular
    rates in the LiDAR frame for the *same* instants, one Kabsch solve gives
    the extrinsic rotation.  This is the cheap version of
    :func:`solve_rotation_hand_eye` and is what :mod:`slamkit.timesync` uses
    once the time offset has been removed.

    **This is only valid after time sync.**  A 20 ms offset during a 1 rad/s
    yaw rotates the correspondence by ~1.1 degrees and biases the extrinsic by
    about that much, which is exactly the failure this toolkit exists to
    untangle: an uncorrected offset masquerades as a bad extrinsic.

    ``min_rate`` (rad/s) drops near-stationary samples, whose direction is
    pure noise.
    """
    Wa = np.asarray(omega_a, dtype=float).reshape(-1, 3)
    Wb = np.asarray(omega_b, dtype=float).reshape(-1, 3)
    if Wa.shape != Wb.shape:
        raise ValueError(f"shape mismatch: {Wa.shape} vs {Wb.shape}")
    mag = np.minimum(np.linalg.norm(Wa, axis=1), np.linalg.norm(Wb, axis=1))
    keep = mag > min_rate
    if int(keep.sum()) < 3:
        raise ValueError(
            f"only {int(keep.sum())} samples above {min_rate} rad/s; "
            "the recording is too static to observe the extrinsic rotation"
        )
    A, B = Wb[keep], Wa[keep]
    R = kabsch_rotation(A, B)
    resid = B - A @ R.T
    scatter = (A / np.linalg.norm(A, axis=1, keepdims=True)).T @ (
        A / np.linalg.norm(A, axis=1, keepdims=True)
    ) / len(A)
    eig = np.sort(np.abs(np.linalg.eigvalsh(scatter)))[::-1]
    cond = float(eig[0] / eig[-1]) if eig[-1] > 1e-12 else float("inf")
    info = {
        "n_used": int(keep.sum()),
        "n_input": int(len(Wa)),
        "residual_rad_s": float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))),
        "axis_condition": cond,
        "well_conditioned": bool(cond < 50.0),
    }
    return R, info


# --------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------
@dataclass
class ExtrinsicCheckConfig:
    """Thresholds for :func:`validate_extrinsics`.

    Defaults are sized for a ground robot or multirotor where the IMU and
    LiDAR share a plate.  Raise ``max_translation_m`` for a car or a boat.
    """

    orthonormal_tol: float = 1e-4
    """Max element-wise deviation of ``R^T R`` from identity before complaining."""

    max_translation_m: float = 2.0
    """Lever arms longer than this on a small robot are usually a units error."""

    min_transpose_margin_rad: float = math.radians(5.0)
    """How much better ``R^T`` must fit the data before we call it transposed."""

    min_direction_margin_m: float = 0.02
    """How much better ``-t`` must fit before we call the translation reversed."""

    degrees_suspicion_rad: float = 2.0 * math.pi
    """|angle| above this in a radians field is almost certainly degrees."""


def check_rotation_valid(R: np.ndarray, cfg: Optional[ExtrinsicCheckConfig] = None) -> Finding:
    """Is this even a rotation matrix?  Run this before anything else."""
    cfg = cfg or ExtrinsicCheckConfig()
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return Finding(
            code="EXTRINSIC_SHAPE",
            severity=Severity.CRITICAL,
            message=f"rotation must be 3x3, got shape {R.shape}",
            symptom="Node exits on startup, or the YAML loader silently pads the matrix "
                    "with zeros and every point lands at the origin.",
            fix="LIO-SAM's extrinsicRot is a flat list of 9 numbers in row-major order.",
        )
    if not np.all(np.isfinite(R)):
        return Finding(
            code="EXTRINSIC_NONFINITE",
            severity=Severity.CRITICAL,
            message="rotation contains NaN or inf",
            symptom="Whole map disappears; RViz shows nothing after the first scan.",
            fix="Check for an unset/empty YAML field parsed as NaN.",
        )
    err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    det = float(np.linalg.det(R))
    if err > cfg.orthonormal_tol:
        scale = float(np.mean(np.linalg.norm(R, axis=0)))
        return Finding(
            code="EXTRINSIC_NOT_ORTHONORMAL",
            severity=Severity.ERROR,
            message=f"R is not orthonormal: max|R^T R - I| = {err:.3e}, "
                    f"det = {det:.6f}, mean column norm = {scale:.6f}",
            symptom="Map scale drifts: the same corridor is a different length on the "
                    "second pass, and loop closure residuals grow with distance "
                    "travelled rather than with time.",
            fix="Do not hand-type rotation matrices to 3 decimals. Generate them with "
                "slamkit.extrinsics.euler_to_matrix(), or repair an existing one with "
                "orthonormalize().",
            data={"orthonormality_error": err, "determinant": det},
        )
    return Finding(
        code="EXTRINSIC_ROTATION_VALID",
        severity=Severity.OK,
        message=f"R is a proper rotation (max|R^T R - I| = {err:.2e}, det = {det:.6f})",
    )


def check_handedness(R: np.ndarray) -> Finding:
    """Catch ``det(R) = -1``: a mirrored, left-handed "rotation"."""
    R = np.asarray(R, dtype=float)
    det = float(np.linalg.det(R))
    if det < 0:
        return Finding(
            code="EXTRINSIC_LEFT_HANDED",
            severity=Severity.CRITICAL,
            message=f"det(R) = {det:+.4f}, which is a reflection, not a rotation",
            symptom="The map is mirrored. Turning left in the real world moves the "
                    "robot right in RViz, loop closure never matches because the "
                    "revisited geometry is chirally flipped, and IMU yaw fights the "
                    "scan matcher on every turn.",
            fix="Someone negated a column or a whole axis to 'make it look right', or "
                "you transcribed a Z-down vendor drawing into a Z-up REP-103 frame. "
                "Rebuild R from roll/pitch/yaw instead of flipping signs; if the sensor "
                "really is Z-down, apply a proper 180 deg roll (Rx(pi)), which has "
                "det = +1.",
            data={"determinant": det},
        )
    return Finding(
        code="EXTRINSIC_HANDEDNESS_OK",
        severity=Severity.OK,
        message=f"right-handed (det = {det:+.6f})",
    )


def check_degrees_as_radians(
    rpy: Sequence[float], cfg: Optional[ExtrinsicCheckConfig] = None
) -> Finding:
    """Catch degrees typed into a field that is documented in radians.

    Two signals are used.  Magnitude: a value above 2*pi cannot be a sane
    radian angle for a sensor mount.  Roundness: values that are integer
    multiples of 5 and larger than 1 (``[0, 0, 90]``, ``[180, 0, 0]``) are
    human-written degrees -- nobody types a radian angle as a round integer.
    """
    cfg = cfg or ExtrinsicCheckConfig()
    v = np.asarray(rpy, dtype=float).reshape(3)
    big = float(np.max(np.abs(v)))
    if big > cfg.degrees_suspicion_rad:
        return Finding(
            code="EXTRINSIC_DEGREES_IN_RADIAN_FIELD",
            severity=Severity.CRITICAL,
            message=f"max |angle| = {big:.3f} rad = {math.degrees(big):.0f} deg; "
                    f"a mounting angle above {cfg.degrees_suspicion_rad:.2f} rad "
                    "(360 deg) is a units error, not a mount",
            symptom="The map is rotated by an arbitrary-looking angle that is not a "
                    "multiple of 90. The first scan looks plausible, then the map "
                    "smears into a fan as soon as the robot yaws, because the IMU "
                    "gravity vector is expressed in a frame nothing else agrees with.",
            fix=f"Convert: {big:.0f} deg = {math.radians(big):.6f} rad. Every ROS "
                "rotation field (URDF <origin rpy>, static_transform_publisher, "
                "LIO-SAM extrinsicRPY) is radians.",
            data={"max_abs_rad": big, "as_degrees": math.degrees(big),
                  "if_degrees_then_rad": [math.radians(x) for x in v]},
        )
    roundish = [
        abs(x) > 1.0 and abs(x % 5.0) < 1e-9 for x in v
    ]
    if any(roundish):
        return Finding(
            code="EXTRINSIC_SUSPECT_DEGREES",
            severity=Severity.WARN,
            message=f"rpy = {np.round(v, 4).tolist()} rad contains round multiples of 5 "
                    f"({math.degrees(big):.0f} deg equivalent); these read like degrees "
                    "that were pasted without conversion",
            symptom="Map rotates about a plausible axis but by the wrong amount; the "
                    "floor comes out tilted and z drifts in one direction.",
            fix="If you meant degrees, use "
                f"{np.round(np.radians(v), 6).tolist()} rad instead.",
            data={"rpy_rad": v.tolist(), "if_degrees_then_rad": np.radians(v).tolist()},
        )
    return Finding(
        code="EXTRINSIC_UNITS_OK",
        severity=Severity.OK,
        message=f"rpy = {np.round(v, 4).tolist()} rad "
                f"({np.round(np.degrees(v), 2).tolist()} deg) is plausible as radians",
    )


def check_transposed_rotation(
    R_claimed: np.ndarray,
    R_estimated: np.ndarray,
    cfg: Optional[ExtrinsicCheckConfig] = None,
) -> Finding:
    """Compare a configured rotation against one estimated from data.

    If ``R_claimed.T`` matches the estimate much better than ``R_claimed``
    does, you have the transform the wrong way round.  This is the single most
    common extrinsic bug, and it is invisible on a bench test where the offset
    is 0 or 180 degrees -- both of which are their own transpose.
    """
    cfg = cfg or ExtrinsicCheckConfig()
    Rc = np.asarray(R_claimed, dtype=float)
    Re = np.asarray(R_estimated, dtype=float)
    d_direct = rotation_geodesic_distance(Rc, Re)
    d_transposed = rotation_geodesic_distance(Rc.T, Re)
    data = {
        "error_as_given_deg": math.degrees(d_direct),
        "error_if_transposed_deg": math.degrees(d_transposed),
    }
    if d_transposed + cfg.min_transpose_margin_rad < d_direct:
        return Finding(
            code="EXTRINSIC_TRANSPOSED",
            severity=Severity.CRITICAL,
            message=f"configured rotation is off by {math.degrees(d_direct):.2f} deg from "
                    f"the data-derived estimate, but its transpose is off by only "
                    f"{math.degrees(d_transposed):.2f} deg -- the matrix is inverted",
            symptom="The map rotates about the wrong axis when you turn: yaw the robot "
                    "and the map pitches or rolls. Straight-line driving looks fine, "
                    "which is why this survives the first test. Walls duplicate at every "
                    "corner and the duplicate offset scales with how much you turned.",
            fix="Transpose the matrix, or -- better -- fix the naming. LIO-SAM's "
                "extrinsicRot is R_lidar_from_imu: it rotates an IMU-frame acceleration "
                "into the LiDAR frame. If you measured 'where the LiDAR is as seen by "
                "the IMU', you built R_imu_from_lidar and must transpose it.",
            data=data,
        )
    if d_direct > math.radians(10.0):
        return Finding(
            code="EXTRINSIC_ROTATION_MISMATCH",
            severity=Severity.ERROR,
            message=f"configured rotation disagrees with the data-derived estimate by "
                    f"{math.degrees(d_direct):.2f} deg (transposing does not help: "
                    f"{math.degrees(d_transposed):.2f} deg)",
            symptom="Persistent map skew that grows with rotation rate; the IMU and the "
                    "scan matcher disagree so the optimiser splits the difference and "
                    "the trajectory develops a slow curve.",
            fix="Re-derive the extrinsic. If the disagreement is close to a multiple of "
                "90 deg you have an axis permutation (a sensor mounted rotated in its "
                "bracket); otherwise remeasure the mount angle.",
            data=data,
        )
    return Finding(
        code="EXTRINSIC_ROTATION_AGREES",
        severity=Severity.OK,
        message=f"configured rotation agrees with the estimate to "
                f"{math.degrees(d_direct):.2f} deg",
        data=data,
    )


def check_translation_direction(
    t_claimed: Sequence[float],
    t_estimated: Sequence[float],
    cfg: Optional[ExtrinsicCheckConfig] = None,
) -> Finding:
    """Catch a lever arm entered with the wrong sign / in the wrong frame."""
    cfg = cfg or ExtrinsicCheckConfig()
    tc = np.asarray(t_claimed, dtype=float).reshape(3)
    te = np.asarray(t_estimated, dtype=float).reshape(3)
    d_direct = float(np.linalg.norm(tc - te))
    d_flipped = float(np.linalg.norm(-tc - te))
    data = {
        "error_as_given_m": d_direct,
        "error_if_negated_m": d_flipped,
        "t_claimed": tc.tolist(),
        "t_estimated": te.tolist(),
    }
    if d_flipped + cfg.min_direction_margin_m < d_direct:
        return Finding(
            code="EXTRINSIC_TRANSLATION_REVERSED",
            severity=Severity.ERROR,
            message=f"lever arm is {d_direct * 100:.1f} cm from the estimate as given, "
                    f"but only {d_flipped * 100:.1f} cm if negated -- the translation "
                    "points the wrong way",
            symptom="The map shears while you translate and doubles while you rotate: "
                    "a wall scanned before and after a turn appears twice, separated by "
                    "roughly twice the lever arm. Stationary the map is perfect.",
            fix="You measured the transform in the opposite direction. Note that "
                "reversing a transform is not just negating t: "
                "t_b_a = -R_a_b^T @ t_a_b. Use invert_transform() on the full 4x4 "
                "instead of flipping the three numbers by hand.",
            data=data,
        )
    if d_direct > 0.10:
        return Finding(
            code="EXTRINSIC_TRANSLATION_MISMATCH",
            severity=Severity.WARN,
            message=f"lever arm differs from the estimate by {d_direct * 100:.1f} cm",
            symptom="Mild ghosting on rotation, proportional to yaw rate. Often "
                    "misdiagnosed as needing a smaller mapping leaf size.",
            fix="Remeasure from the LiDAR optical centre (not the housing) to the IMU "
                "reference point marked on its datasheet.",
            data=data,
        )
    return Finding(
        code="EXTRINSIC_TRANSLATION_AGREES",
        severity=Severity.OK,
        message=f"lever arm agrees to {d_direct * 100:.1f} cm",
        data=data,
    )


def check_translation_magnitude(
    t: Sequence[float], cfg: Optional[ExtrinsicCheckConfig] = None
) -> Finding:
    """Sanity-check the size of the lever arm (catches mm entered as metres)."""
    cfg = cfg or ExtrinsicCheckConfig()
    tv = np.asarray(t, dtype=float).reshape(3)
    n = float(np.linalg.norm(tv))
    if not np.all(np.isfinite(tv)):
        return Finding(
            code="EXTRINSIC_TRANSLATION_NONFINITE",
            severity=Severity.CRITICAL,
            message="translation contains NaN or inf",
            symptom="Every point lands at NaN; RViz shows an empty map.",
            fix="Check the YAML field actually parsed.",
        )
    if n > cfg.max_translation_m:
        return Finding(
            code="EXTRINSIC_TRANSLATION_TOO_LARGE",
            severity=Severity.ERROR,
            message=f"lever arm is {n:.3f} m, above the {cfg.max_translation_m:.1f} m "
                    "sanity limit for a sensor pair on one plate",
            symptom="Huge lever arm turns every yaw into a large apparent translation. "
                    "The trajectory loops outward on every turn and z jumps.",
            fix=f"Almost always millimetres typed into a metres field: "
                f"{n:.3f} m would be {n:.1f} mm = {n / 1000.0:.6f} m. "
                "REP-103 says metres.",
            data={"norm_m": n, "if_mm_then_m": (tv / 1000.0).tolist()},
        )
    return Finding(
        code="EXTRINSIC_TRANSLATION_SANE",
        severity=Severity.OK,
        message=f"lever arm magnitude {n * 100:.1f} cm is plausible",
        data={"norm_m": n},
    )


def validate_extrinsics(
    R: np.ndarray,
    t: Optional[Sequence[float]] = None,
    R_estimated: Optional[np.ndarray] = None,
    t_estimated: Optional[Sequence[float]] = None,
    rpy: Optional[Sequence[float]] = None,
    cfg: Optional[ExtrinsicCheckConfig] = None,
) -> Report:
    """Run every extrinsic check that the supplied inputs make possible.

    Parameters
    ----------
    R:
        The rotation as configured (``R_lidar_from_imu`` by LIO-SAM's
        convention).
    t:
        The configured lever arm, metres.
    R_estimated, t_estimated:
        Optional data-derived estimates -- e.g. from
        :func:`solve_rotation_hand_eye`.  Without these the transposed- and
        reversed-transform checks cannot run, because a transposed rotation is
        a perfectly valid rotation matrix; only data can tell you it is the
        wrong one.
    rpy:
        Optional raw roll/pitch/yaw as written in the config file, used for
        the degrees-vs-radians check.  If omitted it is recovered from ``R``,
        which means an already-converted matrix will (correctly) pass.
    """
    cfg = cfg or ExtrinsicCheckConfig()
    rep = Report(title="LiDAR<-IMU extrinsic validation")
    valid = check_rotation_valid(R, cfg)
    rep.add(valid)
    if valid.severity >= Severity.CRITICAL:
        return rep
    rep.add(check_handedness(R))
    rep.add(check_degrees_as_radians(rpy if rpy is not None else matrix_to_euler(R), cfg))
    if t is not None:
        rep.add(check_translation_magnitude(t, cfg))
    if R_estimated is not None:
        rep.add(check_transposed_rotation(R, R_estimated, cfg))
    if t is not None and t_estimated is not None:
        rep.add(check_translation_direction(t, t_estimated, cfg))
    return rep

"""SE(3) maths and the extrinsic validators.

The important tests here are the last two groups: that a transposed rotation
and a degrees-in-a-radians-field error are actually caught.
"""

import math

import numpy as np
import pytest

from slamkit.extrinsics import (
    ExtrinsicCheckConfig,
    check_degrees_as_radians,
    check_handedness,
    check_rotation_valid,
    check_translation_direction,
    check_translation_magnitude,
    check_transposed_rotation,
    compose_transforms,
    euler_to_matrix,
    invert_transform,
    is_rotation_matrix,
    kabsch_rotation,
    make_transform,
    matrix_to_euler,
    matrix_to_quaternion,
    orthonormalize,
    quaternion_to_matrix,
    rotation_exp,
    rotation_geodesic_distance,
    rotation_log,
    solve_rotation_from_angular_rates,
    solve_rotation_hand_eye,
    transform_points,
    transform_vectors,
    validate_extrinsics,
)
from slamkit.findings import Severity


# ---------------------------------------------------------------- conversions
def test_euler_matrix_roundtrip():
    for rpy in ([0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [-1.2, 0.4, 2.9]):
        R = euler_to_matrix(rpy)
        assert is_rotation_matrix(R)
        assert np.allclose(matrix_to_euler(R), rpy, atol=1e-9)


def test_euler_degrees_flag():
    assert np.allclose(euler_to_matrix([0, 0, 90], degrees=True),
                       euler_to_matrix([0, 0, math.pi / 2]))


def test_euler_yaw_is_right_handed_about_z():
    # +90 deg yaw takes +X to +Y (REP-103: right-handed, Z up).
    R = euler_to_matrix([0.0, 0.0, math.pi / 2])
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_quaternion_roundtrip_including_180_degrees():
    for rpy in ([0.3, 0.2, -0.4], [math.pi, 0.0, 0.0], [0.0, 0.0, math.pi]):
        R = euler_to_matrix(rpy)
        q = matrix_to_quaternion(R)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12
        assert np.allclose(quaternion_to_matrix(q), R, atol=1e-9)


def test_log_exp_roundtrip_small_and_large():
    for w in ([1e-9, 0.0, 0.0], [0.3, -0.2, 0.1], [0.0, 0.0, math.pi - 1e-4]):
        R = rotation_exp(w)
        assert is_rotation_matrix(R)
        assert np.allclose(rotation_log(R), w, atol=1e-6)


def test_geodesic_distance_matches_known_angle():
    A = np.eye(3)
    B = euler_to_matrix([0.0, 0.0, math.radians(30.0)])
    assert abs(math.degrees(rotation_geodesic_distance(A, B)) - 30.0) < 1e-9


# ------------------------------------------------------------------- SE(3)
def test_invert_and_compose_are_consistent():
    T = make_transform(euler_to_matrix([0.2, -0.1, 0.7]), [1.0, -2.0, 0.5])
    assert np.allclose(compose_transforms(T, invert_transform(T)), np.eye(4), atol=1e-12)
    assert np.allclose(compose_transforms(invert_transform(T), T), np.eye(4), atol=1e-12)


def test_inverse_translation_is_not_just_negated():
    """The classic bug: -t is not the inverse translation unless R is identity."""
    T = make_transform(euler_to_matrix([0.0, 0.0, math.pi / 2]), [1.0, 0.0, 0.0])
    Tinv = invert_transform(T)
    assert not np.allclose(Tinv[:3, 3], -T[:3, 3])
    assert np.allclose(Tinv[:3, 3], [0.0, 1.0, 0.0], atol=1e-12)


def test_transform_points_vs_vectors():
    T = make_transform(np.eye(3), [1.0, 2.0, 3.0])
    p = np.array([[0.0, 0.0, 0.0]])
    assert np.allclose(transform_points(T, p), [[1.0, 2.0, 3.0]])
    assert np.allclose(transform_vectors(T, p), [[0.0, 0.0, 0.0]])


def test_orthonormalize_repairs_a_rounded_matrix():
    R = euler_to_matrix([0.3, 0.1, -0.2])
    rounded = np.round(R, 3)
    assert not is_rotation_matrix(rounded, tol=1e-6)
    assert is_rotation_matrix(orthonormalize(rounded), tol=1e-9)


# ------------------------------------------------------------------ solvers
def test_kabsch_recovers_a_known_rotation():
    R_true = euler_to_matrix([0.2, -0.35, 1.1])
    src = np.random.default_rng(0).normal(size=(40, 3))
    dst = src @ R_true.T
    assert np.allclose(kabsch_rotation(src, dst), R_true, atol=1e-9)


def test_kabsch_never_returns_a_reflection():
    src = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0], [0, -1.0, 0]])
    dst = src * np.array([1.0, -1.0, 1.0])  # a mirror, not a rotation
    R = kabsch_rotation(src, dst)
    assert np.linalg.det(R) > 0


def test_hand_eye_recovers_the_extrinsic_rotation():
    X = euler_to_matrix([0.1, -0.2, math.pi / 2])
    motions_b = [euler_to_matrix(v) for v in
                 ([0.25, 0, 0], [0, 0.3, 0], [0, 0, 0.35], [0.15, 0.15, 0.15])]
    motions_a = [X @ B @ X.T for B in motions_b]
    R, info = solve_rotation_hand_eye(motions_a, motions_b)
    assert np.allclose(R, X, atol=1e-8)
    assert info["n_used"] == 4
    assert info["well_conditioned"]
    assert info["residual_deg"] < 1e-6


def test_hand_eye_rejects_a_static_recording():
    tiny = [euler_to_matrix([1e-4, 0, 0])] * 5
    with pytest.raises(ValueError):
        solve_rotation_hand_eye(tiny, tiny)


def test_angular_rate_solver_recovers_the_extrinsic():
    R_true = euler_to_matrix([0.0, 0.0, math.pi / 2])
    rng = np.random.default_rng(3)
    w_imu = rng.normal(scale=0.8, size=(200, 3))
    w_lidar = w_imu @ R_true.T
    R, info = solve_rotation_from_angular_rates(w_lidar, w_imu)
    assert np.allclose(R, R_true, atol=1e-8)
    assert info["n_used"] > 150


# --------------------------------------------------------------- validators
def test_validator_catches_a_transposed_rotation():
    """The headline case: a 90 deg yaw offset entered the wrong way round."""
    R_true = euler_to_matrix([0.0, 0.0, math.pi / 2])
    f = check_transposed_rotation(R_true.T, R_true)
    assert f.code == "EXTRINSIC_TRANSPOSED"
    assert f.severity == Severity.CRITICAL
    assert f.data["error_if_transposed_deg"] < 1e-6
    assert f.data["error_as_given_deg"] > 90.0
    assert "yaw" in f.symptom.lower()


def test_validator_accepts_a_correct_rotation():
    R = euler_to_matrix([0.0, 0.0, math.pi / 2])
    f = check_transposed_rotation(R, R)
    assert f.code == "EXTRINSIC_ROTATION_AGREES"
    assert f.severity == Severity.OK


def test_transpose_check_cannot_fire_on_a_180_degree_mount():
    """A 180 deg rotation is its own transpose -- the check must not claim a bug."""
    R = euler_to_matrix([0.0, 0.0, math.pi])
    assert np.allclose(R, R.T, atol=1e-12)
    assert check_transposed_rotation(R, R).severity == Severity.OK


def test_validator_catches_degrees_in_a_radians_field():
    f = check_degrees_as_radians([0.0, 0.0, 90.0])
    assert f.code == "EXTRINSIC_DEGREES_IN_RADIAN_FIELD"
    assert f.severity == Severity.CRITICAL
    assert abs(f.data["as_degrees"] - math.degrees(90.0)) < 1e-6
    assert abs(f.data["if_degrees_then_rad"][2] - math.pi / 2) < 1e-9


def test_validator_flags_suspiciously_round_radian_values():
    f = check_degrees_as_radians([0.0, 0.0, 5.0])  # < 2*pi but a round multiple of 5
    assert f.code == "EXTRINSIC_SUSPECT_DEGREES"
    assert f.severity == Severity.WARN


def test_validator_passes_real_radian_values():
    f = check_degrees_as_radians([0.0, 0.0, math.pi / 2])
    assert f.severity == Severity.OK


def test_validator_catches_left_handed_rotation():
    R = np.diag([1.0, -1.0, 1.0])
    f = check_handedness(R)
    assert f.code == "EXTRINSIC_LEFT_HANDED"
    assert f.severity == Severity.CRITICAL
    assert f.data["determinant"] < 0


def test_validator_catches_non_orthonormal_matrix():
    R = euler_to_matrix([0.3, 0.1, -0.2]) * 1.05
    f = check_rotation_valid(R)
    assert f.code == "EXTRINSIC_NOT_ORTHONORMAL"
    assert f.severity == Severity.ERROR


def test_validator_catches_reversed_translation():
    f = check_translation_direction([0.12, 0.0, 0.0], [-0.12, 0.0, 0.0])
    assert f.code == "EXTRINSIC_TRANSLATION_REVERSED"
    assert f.severity == Severity.ERROR
    assert f.data["error_if_negated_m"] < f.data["error_as_given_m"]


def test_validator_catches_millimetres_in_a_metres_field():
    f = check_translation_magnitude([120.0, 0.0, 0.0])
    assert f.code == "EXTRINSIC_TRANSLATION_TOO_LARGE"
    assert np.allclose(f.data["if_mm_then_m"], [0.12, 0.0, 0.0])


def test_validate_extrinsics_ranks_the_worst_problem_first():
    R_true = euler_to_matrix([0.0, 0.0, math.pi / 2])
    rep = validate_extrinsics(R_true.T, t=[0.12, 0.0, 0.0], R_estimated=R_true,
                              t_estimated=[-0.12, 0.0, 0.0])
    codes = [f.code for f in rep.problems]
    assert "EXTRINSIC_TRANSPOSED" in codes
    assert "EXTRINSIC_TRANSLATION_REVERSED" in codes
    assert rep.problems[0].severity == Severity.CRITICAL
    assert rep.worst == Severity.CRITICAL


def test_validate_extrinsics_clean_config_has_no_problems():
    R = euler_to_matrix([0.0, 0.0, math.pi / 2])
    rep = validate_extrinsics(R, t=[0.12, 0.0, 0.0], R_estimated=R,
                              t_estimated=[0.12, 0.0, 0.0])
    assert rep.problems == []
    assert rep.worst == Severity.OK


def test_check_config_thresholds_are_respected():
    cfg = ExtrinsicCheckConfig(max_translation_m=0.05)
    assert check_translation_magnitude([0.10, 0, 0], cfg).severity == Severity.ERROR
    assert check_translation_magnitude([0.01, 0, 0], cfg).severity == Severity.OK

"""ATE / RPE and the no-ground-truth trajectory metrics.

The ATE and RPE tests use trajectories simple enough to compute by hand, so a
regression in the maths is caught by inspection rather than by a golden file.
"""

import math

import numpy as np
import pytest

from slamkit.drift import (
    absolute_trajectory_error,
    as_poses,
    find_revisits,
    loop_closure_residual,
    relative_pose_error,
    revisit_consistency,
    umeyama_alignment,
    yaw_drift_rate,
    z_drift_rate,
)
from slamkit.extrinsics import euler_to_matrix, make_transform


def _line(n=4, step=1.0):
    """Straight line along +X at 0, step, 2*step, ..."""
    p = np.zeros((n, 3))
    p[:, 0] = np.arange(n) * step
    return p


# ------------------------------------------------------------------ ATE
def test_ate_matches_a_hand_computed_constant_offset():
    """Every pose off by exactly 0.1 m in +Y -> ATE RMSE is exactly 0.1 m."""
    gt = _line()
    est = gt + np.array([0.0, 0.1, 0.0])
    r = absolute_trajectory_error(est, gt, align=False)
    assert r["translation_rmse"] == pytest.approx(0.1, abs=1e-12)
    assert r["translation"].max == pytest.approx(0.1, abs=1e-12)
    assert np.allclose(r["per_axis_rmse"], [0.0, 0.1, 0.0], atol=1e-12)


def test_ate_alignment_removes_a_rigid_offset():
    """The same offset is a gauge freedom once alignment is enabled."""
    gt = _line()
    est = gt + np.array([0.0, 0.1, 0.0])
    r = absolute_trajectory_error(est, gt, align=True)
    assert r["translation_rmse"] < 1e-12


def test_ate_alignment_removes_a_rigid_rotation():
    gt = _line(n=8)
    R = euler_to_matrix([0.0, 0.0, 0.7])
    est = gt @ R.T + np.array([3.0, -2.0, 1.0])
    r = absolute_trajectory_error(est, gt, align=True)
    assert r["translation_rmse"] < 1e-9


def test_ate_hand_computed_alternating_error():
    """+0.2 / -0.2 in z alternating -> RMSE 0.2, mean 0.2, per-axis z 0.2."""
    gt = _line(n=4)
    est = gt.copy()
    est[:, 2] += np.array([0.2, -0.2, 0.2, -0.2])
    r = absolute_trajectory_error(est, gt, align=False)
    assert r["translation_rmse"] == pytest.approx(0.2, abs=1e-12)
    assert r["per_axis_rmse"][2] == pytest.approx(0.2, abs=1e-12)


# ------------------------------------------------------------------ RPE
def test_rpe_matches_a_hand_computed_scale_error():
    """Steps of 1.1 m where truth is 1.0 m -> RPE = 0.1 m, drift = 10%."""
    gt = _line(n=4, step=1.0)
    est = _line(n=4, step=1.1)
    r = relative_pose_error(est, gt, delta=1)
    assert r["translation_rmse"] == pytest.approx(0.1, abs=1e-9)
    assert r["drift_percent"] == pytest.approx(10.0, abs=1e-6)
    assert r["translation"].n == 3


def test_rpe_is_insensitive_to_a_global_offset():
    gt = _line(n=6)
    est = gt + np.array([5.0, -3.0, 2.0])
    r = relative_pose_error(est, gt, delta=1)
    assert r["translation_rmse"] < 1e-12


def test_rpe_measures_a_known_rotation_error():
    gt = np.tile(np.eye(4), (5, 1, 1))
    est = np.tile(np.eye(4), (5, 1, 1))
    for i in range(5):
        gt[i, :3, :3] = euler_to_matrix([0.0, 0.0, 0.1 * i])
        est[i, :3, :3] = euler_to_matrix([0.0, 0.0, 0.11 * i])
    r = relative_pose_error(est, gt, delta=1)
    assert r["rotation_rmse_deg"] == pytest.approx(math.degrees(0.01), abs=1e-6)


def test_rpe_rejects_an_out_of_range_delta():
    gt = _line(n=4)
    with pytest.raises(ValueError):
        relative_pose_error(gt, gt, delta=4)


# ------------------------------------------------------------- alignment
def test_umeyama_recovers_a_known_transform():
    R_true = euler_to_matrix([0.1, 0.2, -0.4])
    t_true = np.array([1.0, -2.0, 0.5])
    src = np.random.default_rng(1).normal(size=(30, 3))
    dst = src @ R_true.T + t_true
    R, t, s = umeyama_alignment(src, dst)
    assert np.allclose(R, R_true, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-9)
    assert s == pytest.approx(1.0)


def test_umeyama_with_scale_recovers_a_scale_factor():
    src = np.random.default_rng(2).normal(size=(30, 3))
    dst = 2.5 * src
    _, _, s = umeyama_alignment(src, dst, with_scale=True)
    assert s == pytest.approx(2.5, rel=1e-9)


# ------------------------------------------------- no-ground-truth metrics
def test_loop_closure_residual_reports_the_gap():
    poses = np.tile(np.eye(4), (10, 1, 1))
    poses[:, 0, 3] = np.arange(10) * 1.0
    poses[9, :3, 3] = [0.3, 0.0, 0.2]   # "same place" as pose 0, but 0.36 m off
    r = loop_closure_residual(poses, [(0, 9)])
    assert r["n_pairs"] == 1
    assert r["translation_max_m"] == pytest.approx(math.hypot(0.3, 0.2), abs=1e-9)
    assert r["z_residual_max_m"] == pytest.approx(0.2, abs=1e-9)


def test_find_revisits_needs_both_proximity_and_a_time_gap():
    t = np.arange(0.0, 60.0, 1.0)
    # Out and back: returns to the origin at the end.
    x = np.concatenate([np.arange(30) * 1.0, np.arange(30)[::-1] * 1.0])
    pos = np.zeros((60, 3))
    pos[:, 0] = x
    pairs = find_revisits(pos, t, radius=1.0, min_time_gap_s=20.0)
    assert len(pairs) > 0
    for i, j in pairs:
        assert t[j] - t[i] >= 20.0
        assert np.linalg.norm(pos[j] - pos[i]) <= 1.0


def test_revisit_consistency_measures_an_injected_loop_error():
    t = np.arange(0.0, 60.0, 1.0)
    pos = np.zeros((60, 3))
    pos[:, 0] = np.concatenate([np.arange(30) * 1.0, np.arange(30)[::-1] * 1.0])
    pos[30:, 2] += 0.4        # the return leg comes back 40 cm high
    r = revisit_consistency(as_poses(pos), t, radius=1.0, min_time_gap_s=20.0)
    assert r["n_pairs"] > 0
    assert r["z_residual_max_m"] == pytest.approx(0.4, abs=1e-9)


def test_z_drift_rate_matches_an_injected_slope():
    t = np.linspace(0.0, 120.0, 240)
    z = 0.005 * t          # 0.005 m/s = 0.30 m/min
    r = z_drift_rate(t, z)
    assert r["rate_m_per_min"] == pytest.approx(0.30, abs=1e-9)
    assert r["r_squared"] > 0.999
    assert r["total_change_m"] == pytest.approx(0.6, abs=1e-9)


def test_yaw_drift_rate_from_a_linear_fit():
    t = np.linspace(0.0, 60.0, 120)
    poses = np.tile(np.eye(4), (120, 1, 1))
    for i, ti in enumerate(t):
        poses[i, :3, :3] = euler_to_matrix([0.0, 0.0, math.radians(0.1) * ti])
    r = yaw_drift_rate(t, poses)
    assert r["method"] == "linear_fit"
    assert r["rate_deg_per_min"] == pytest.approx(6.0, abs=1e-6)


def test_yaw_drift_rate_from_revisits_uses_the_pairs():
    t = np.linspace(0.0, 60.0, 120)
    poses = np.tile(np.eye(4), (120, 1, 1))
    poses[-1, :3, :3] = euler_to_matrix([0.0, 0.0, math.radians(2.0)])
    r = yaw_drift_rate(t, poses, revisit_pairs=[(0, 119)])
    assert r["method"] == "revisit"
    assert r["rate_deg_per_min"] == pytest.approx(2.0, abs=0.05)


def test_as_poses_accepts_all_three_input_shapes():
    xyz = _line(3)
    assert as_poses(xyz).shape == (3, 4, 4)
    assert np.allclose(as_poses(xyz)[:, :3, 3], xyz)
    quat = np.zeros((3, 7))
    quat[:, 3] = 1.0        # qx=1 -> 180 deg roll
    assert as_poses(quat).shape == (3, 4, 4)
    T = make_transform(euler_to_matrix([0.1, 0.2, 0.3]), [1, 2, 3])
    assert np.allclose(as_poses(np.array([T]))[0], T)
    with pytest.raises(ValueError):
        as_poses(np.zeros((3, 5)))

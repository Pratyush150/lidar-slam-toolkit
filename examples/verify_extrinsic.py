#!/usr/bin/env python3
"""Worked example: verify a LiDAR->IMU extrinsic against recorded motion.

Run it directly -- it generates its own data, so it needs nothing but numpy:

    python3 examples/verify_extrinsic.py

The workflow it demonstrates is the one to use on real data:

  1. Fix time sync FIRST. An uncorrected offset biases the extrinsic estimate
     by roughly (angular rate) * (offset), so calibrating before syncing bakes
     the time error into your mount angle.
  2. Estimate the extrinsic rotation from paired motions (hand-eye).
  3. Check the conditioning. Single-axis excitation cannot observe all three
     angles, and the solver will still return an answer.
  4. Compare the configured value against the estimate, which is the only way
     to catch a transposed matrix -- a transposed rotation is still a perfectly
     valid rotation.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from slamkit.extrinsics import (  # noqa: E402
    euler_to_matrix,
    matrix_to_euler,
    rotation_exp,
    solve_rotation_hand_eye,
    validate_extrinsics,
)
from slamkit.synthetic import circle_trajectory, simulate_imu  # noqa: E402
from slamkit.timesync import estimate_lidar_imu_offset  # noqa: E402


def incremental_rotations(times, poses, imu_times, gyro):
    """Pair each inter-scan LiDAR rotation with the integrated gyro rotation."""
    lidar, imu = [], []
    for i in range(len(times) - 1):
        t0, t1 = float(times[i]), float(times[i + 1])
        sel = (imu_times >= t0) & (imu_times < t1)
        if int(sel.sum()) < 2:
            continue
        ts, ws = imu_times[sel], gyro[sel]
        R_b = np.eye(3)
        for j in range(len(ts) - 1):
            R_b = R_b @ rotation_exp(ws[j] * (ts[j + 1] - ts[j]))
        lidar.append(poses[i, :3, :3].T @ poses[i + 1, :3, :3])
        imu.append(R_b)
    return lidar, imu


def main() -> int:
    # --- ground truth we are pretending not to know ------------------------
    true_rpy = [0.0, 0.0, math.pi / 2]        # IMU yawed 90 deg in its bracket
    R_true = euler_to_matrix(true_rpy)
    T_lidar_imu = np.eye(4)
    T_lidar_imu[:3, :3] = R_true
    T_lidar_imu[:3, 3] = [0.12, 0.0, -0.05]

    # --- a calibration recording: rotation about all three axes ------------
    traj = circle_trajectory(duration=40.0, rate=200.0, radius=6.0,
                             period=10.0, tilt_deg=12.0)
    imu = simulate_imu(traj, T_lidar_imu, gyro_noise=0.002, seed=0)
    scan_idx = np.arange(0, len(traj), 20)    # 10 Hz scan-matched poses
    scan_times, scan_poses = traj.times[scan_idx], traj.poses[scan_idx]

    # --- step 1: time sync -------------------------------------------------
    est = estimate_lidar_imu_offset(scan_times, scan_poses, imu.times, imu.gyro,
                                    max_offset_s=0.1)
    print(f"[1] time offset  : {est.offset_ms:+.2f} ms "
          f"(correlation {est.correlation:.3f}, peak width {est.peak_width_ms:.0f} ms)")
    if abs(est.offset_ms) > 5.0:
        print("    Offset is large. Fix it before calibrating -- otherwise the")
        print("    time error is absorbed into the mount angle.")

    # --- step 2 and 3: hand-eye solve and conditioning ---------------------
    lidar_motions, imu_motions = incremental_rotations(
        scan_times, scan_poses, imu.times, imu.gyro)
    R_est, info = solve_rotation_hand_eye(lidar_motions, imu_motions)
    print(f"[2] estimated rpy: "
          f"{np.round(np.degrees(matrix_to_euler(R_est)), 3).tolist()} deg "
          f"from {info['n_used']} motions")
    print(f"[3] conditioning : axis condition {info['axis_condition']:.1f}, "
          f"residual {info['residual_deg']:.3f} deg, "
          f"well conditioned: {info['well_conditioned']}")

    # --- step 4: check the config that is actually deployed ----------------
    print("\n[4a] a CORRECT config:")
    for f in validate_extrinsics(R_true, t=[0.12, 0.0, -0.05],
                                 R_estimated=R_est).ranked():
        print("   ", f)

    print("\n[4b] the same config entered TRANSPOSED (the classic bug):")
    for f in validate_extrinsics(R_true.T, t=[0.12, 0.0, -0.05],
                                 R_estimated=R_est).problems:
        print("   ", f)

    print("\n[4c] the same angles pasted in DEGREES into a radians field:")
    for f in validate_extrinsics(euler_to_matrix([0.0, 0.0, 90.0]),
                                 rpy=[0.0, 0.0, 90.0]).problems:
        print("   ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

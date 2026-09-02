"""``slam-doctor`` -- run the whole diagnostic battery and rank what to fix first.

The ordering of the output is the point.  A broken LiDAR SLAM setup produces a
dozen symptoms from one root cause, and if you work the list bottom-up you fix
nothing.  So findings are ranked by severity, and the report ends with the
bring-up order (TF, then time, then extrinsics, then tuning) because that is
the order in which fixes actually stick.

Run ``slam-doctor --demo`` to see it work on synthetic data with known
injected defects -- a transposed extrinsic, a 45 ms time offset, a z-drift
ramp with a step in it, and a corridor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import cloud as cloud_mod
from . import degeneracy as degen_mod
from . import drift as drift_mod
from . import extrinsics as ext
from . import rosbag_check as bag_mod
from . import timesync as ts_mod
from .findings import Finding, Report, Severity

__all__ = ["DoctorResult", "load_table", "diagnose", "run_demo", "format_report", "main"]

_BRINGUP = """\
BRING-UP ORDER (do not skip ahead)
  1. TF          -- every sensor has a frame_id, one parent per frame, chains resolve.
  2. Time sync   -- sensor-clock stamping, offset under a few ms, no drift.
  3. Extrinsics  -- rotation direction and lever arm verified against data.
  4. Tuning      -- leaf sizes, thresholds, loop-closure gates.
Tuning before steps 1-3 is wasted work: you are fitting parameters to a defect,
and every parameter you touch will need re-tuning once the defect is fixed."""


@dataclass
class DoctorResult:
    """Everything ``slam-doctor`` computed, plus the ranked report."""

    report: Report = field(default_factory=lambda: Report(title="slam-doctor"))
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    @property
    def exit_code(self) -> int:
        return 1 if self.report.worst >= Severity.ERROR else 0

    def to_dict(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for f in self.report.findings:
            counts[f.severity.name] = counts.get(f.severity.name, 0) + 1
        return {
            "source": self.source,
            "worst_severity": self.report.worst.name,
            "counts": counts,
            "findings": [f.to_dict() for f in self.report.ranked()],
            "details": _jsonable(self.details),
            "bringup_order": [line.strip() for line in _BRINGUP.splitlines()[1:5]],
        }


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy types so ``json.dumps`` does not choke."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "to_dict"):
        return _jsonable(obj.to_dict())
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


# --------------------------------------------------------------------------
# Input loading
# --------------------------------------------------------------------------
def load_table(path: str) -> np.ndarray:
    """Load a numeric table from ``.npy``, ``.csv`` or whitespace-delimited text.

    Header lines starting with ``#`` are skipped, and so is a single leading
    line of column names.
    """
    if path.endswith(".npy"):
        return np.asarray(np.load(path), dtype=float)
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()
                 and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{path} contains no data rows")
    delim = "," if "," in lines[0] else None
    try:
        float(lines[0].split(delim)[0] if delim else lines[0].split()[0])
    except ValueError:
        lines = lines[1:]  # drop a column-name header
    rows = [[float(x) for x in (ln.split(delim) if delim else ln.split())]
            for ln in lines]
    return np.asarray(rows, dtype=float)


def _trajectory_from_table(tab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``t x y z [qx qy qz qw]`` -> ``(times, (N, 4, 4) poses)``."""
    if tab.ndim != 2 or tab.shape[1] < 4:
        raise ValueError("trajectory needs at least 4 columns: t x y z")
    t = tab[:, 0]
    if tab.shape[1] >= 8:
        poses = drift_mod.as_poses(tab[:, 1:8])
    else:
        poses = drift_mod.as_poses(tab[:, 1:4])
    return t, poses


def _imu_from_table(tab: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``t gx gy gz ax ay az`` -> ``(times, gyro, accel)``."""
    if tab.ndim != 2 or tab.shape[1] < 4:
        raise ValueError("IMU table needs at least 4 columns: t gx gy gz")
    t = tab[:, 0]
    gyro = tab[:, 1:4]
    accel = tab[:, 4:7] if tab.shape[1] >= 7 else np.zeros_like(gyro)
    return t, gyro, accel


# --------------------------------------------------------------------------
# The battery
# --------------------------------------------------------------------------
def diagnose(
    scan_times: Optional[np.ndarray] = None,
    scan_poses: Optional[np.ndarray] = None,
    imu_times: Optional[np.ndarray] = None,
    gyro: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    ground_truth_poses: Optional[np.ndarray] = None,
    extrinsic_R: Optional[np.ndarray] = None,
    extrinsic_t: Optional[Sequence[float]] = None,
    bag_info: Optional[bag_mod.BagInfo] = None,
    required_tf_chains: Sequence[Tuple[str, str]] = (),
    lidar_rate_hz: float = 10.0,
    imu_rate_hz: float = 200.0,
    max_offset_s: float = 0.2,
    source: str = "user data",
) -> DoctorResult:
    """Run every check the supplied inputs make possible.

    Every argument is optional.  Pass what you have; the checks that need
    something you did not supply are skipped rather than faked.

    ``max_offset_s`` bounds the time-offset search.  Widen it if you suspect a
    whole-scan (100 ms+) misalignment; narrowing it makes the search cheaper on
    long recordings.
    """
    res = DoctorResult(source=source)
    rep = res.report
    rep.title = f"slam-doctor: {source}"

    # ---- 1. TF and bag topology ----------------------------------------
    if bag_info is not None:
        bag_report = bag_mod.check_bag(bag_info, required_chains=required_tf_chains)
        rep.extend(bag_report.findings)
        res.details["bag"] = bag_info.to_dict()

    # ---- 2. Time ---------------------------------------------------------
    if scan_times is not None:
        tsr = ts_mod.analyze_time_sync(
            scan_times=scan_times,
            scan_rotations=scan_poses,
            imu_times=imu_times,
            gyro=gyro,
            lidar_rate_hz=lidar_rate_hz,
            imu_rate_hz=imu_rate_hz,
            max_offset_s=max_offset_s,
        )
        rep.extend(tsr.findings.findings)
        res.details["timesync"] = tsr.to_dict()

    # ---- 3. Extrinsics ---------------------------------------------------
    R_est = None
    if scan_poses is not None and imu_times is not None and gyro is not None:
        try:
            R_est, info = _estimate_extrinsic_rotation(
                np.asarray(scan_times, dtype=float), np.asarray(scan_poses, dtype=float),
                np.asarray(imu_times, dtype=float), np.asarray(gyro, dtype=float))
            res.details["extrinsic_estimate"] = {
                "R": R_est.tolist(),
                "rpy_deg": np.degrees(ext.matrix_to_euler(R_est)).tolist(),
                **info,
            }
            if not info.get("well_conditioned", True):
                rep.add(Finding(
                    code="EXTRINSIC_POORLY_EXCITED",
                    severity=Severity.WARN,
                    message=f"the recording rotates about essentially one axis "
                            f"(axis condition number {info['axis_condition']:.0f}); the "
                            "estimated extrinsic is only trustworthy about that axis",
                    symptom="Calibration 'converges' and the map is still wrong in "
                            "roll or pitch.",
                    fix="Record a calibration segment that rotates about all three "
                        "axes: yaw left/right, then pitch up/down, then roll. 30 s is "
                        "enough. A robot that only ever yaws cannot observe the roll "
                        "and pitch of its own extrinsic.",
                    data=info,
                ))
        except ValueError as exc:
            rep.add(Finding(
                code="EXTRINSIC_NOT_ESTIMABLE",
                severity=Severity.INFO,
                message=f"could not estimate the extrinsic rotation from data: {exc}",
                fix="Record a segment with real rotation on more than one axis.",
            ))
    if extrinsic_R is not None:
        ext_report = ext.validate_extrinsics(
            np.asarray(extrinsic_R, dtype=float),
            t=extrinsic_t,
            R_estimated=R_est,
        )
        rep.extend(ext_report.findings)

    # ---- 4. Geometry -----------------------------------------------------
    if points is not None and len(points) >= 10:
        dg = degen_mod.analyze_degeneracy(points, voxel_size=0.3)
        rep.extend(degen_mod.degeneracy_findings(dg))
        res.details["degeneracy"] = dg.to_dict()
        plane = cloud_mod.ransac_ground_plane(points, distance_threshold=0.08)
        res.details["ground_plane"] = {
            "normal": plane.normal.tolist(),
            "offset": plane.offset,
            "inliers": plane.n_inliers,
            "tilt_deg": plane.tilt_deg,
        }
        if plane.n_inliers > 50 and plane.tilt_deg > 3.0:
            rep.add(Finding(
                code="GROUND_PLANE_TILTED",
                severity=Severity.WARN,
                message=f"the dominant ground plane is tilted {plane.tilt_deg:.1f} deg "
                        f"from horizontal in the sensor frame "
                        f"({plane.n_inliers} inliers)",
                symptom="The whole map tips; z drifts as you drive because forward "
                        "motion has a vertical component in the map frame.",
                fix="Either the sensor really is mounted at that angle -- in which case "
                    "put it in the extrinsic roll/pitch -- or your gravity alignment is "
                    "off. Check both against a spirit level before touching any "
                    "estimator parameter.",
                data=res.details["ground_plane"],
            ))

    # ---- 5. Trajectory ---------------------------------------------------
    if scan_poses is not None and scan_times is not None:
        P = drift_mod.as_poses(scan_poses)
        t = np.asarray(scan_times, dtype=float)
        xy = P[:, :2, 3]
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        zr = drift_mod.analyze_z_drift(t, P[:, 2, 3], horizontal_distance=dist)
        rep.extend(drift_mod.z_drift_findings(zr))
        res.details["z_drift"] = zr.to_dict()
        rc = drift_mod.revisit_consistency(P, t)
        res.details["revisits"] = rc
        if rc.get("n_pairs", 0) > 0 and rc["translation_rmse_m"] > 0.5:
            rep.add(Finding(
                code="LOOP_RESIDUAL_LARGE",
                severity=Severity.WARN,
                message=f"{rc['n_pairs']} revisit(s) disagree by "
                        f"{rc['translation_rmse_m']:.2f} m RMS "
                        f"(max {rc['translation_max_m']:.2f} m, "
                        f"z {rc['z_residual_max_m']:.2f} m)",
                symptom="Ghosted / doubled walls where the robot passed twice.",
                fix="This is accumulated odometry drift that loop closure has not "
                    "removed. Check whether loop closure is firing at all "
                    "(loopClosureEnableFlag, historyKeyframeSearchRadius must exceed "
                    "your drift) before assuming the front end is at fault.",
                data={k: v for k, v in rc.items() if k != "pairs"},
            ))
        if ground_truth_poses is not None:
            G = drift_mod.as_poses(ground_truth_poses)
            ate = drift_mod.absolute_trajectory_error(P, G)
            rpe = drift_mod.relative_pose_error(P, G, delta=1)
            res.details["ate"] = {
                "translation_rmse_m": ate["translation_rmse"],
                "rotation_rmse_deg": ate["rotation_rmse_deg"],
                "per_axis_rmse_m": ate["per_axis_rmse"].tolist(),
            }
            res.details["rpe"] = {
                "translation_rmse_m": rpe["translation_rmse"],
                "rotation_rmse_deg": rpe["rotation_rmse_deg"],
                "drift_percent": rpe["drift_percent"],
            }
            rep.add(Finding(
                code="TRAJECTORY_ATE",
                severity=Severity.INFO,
                message=f"ATE {ate['translation_rmse']:.3f} m / "
                        f"{ate['rotation_rmse_deg']:.2f} deg RMSE after rigid "
                        f"alignment; per-axis RMSE "
                        f"{np.round(ate['per_axis_rmse'], 3).tolist()} m. "
                        f"RPE {rpe['translation_rmse']:.4f} m per step "
                        f"({rpe['drift_percent']:.2f}% of distance travelled)",
                data={**res.details["ate"], **res.details["rpe"]},
            ))
    return res


def _estimate_extrinsic_rotation(scan_times, scan_poses, imu_times, gyro):
    """Hand-eye solve using scan-matched rotations and integrated gyro rotations."""
    P = drift_mod.as_poses(scan_poses)
    A: List[np.ndarray] = []
    B: List[np.ndarray] = []
    g = np.asarray(gyro, dtype=float).reshape(-1, 3)
    it = np.asarray(imu_times, dtype=float).reshape(-1)
    for i in range(len(scan_times) - 1):
        t0, t1 = float(scan_times[i]), float(scan_times[i + 1])
        sel = (it >= t0) & (it < t1)
        if int(sel.sum()) < 2:
            continue
        # Integrate the gyro over the interval to get the IMU's incremental rotation.
        ts = it[sel]
        ws = g[sel]
        R_b = np.eye(3)
        for j in range(len(ts) - 1):
            R_b = R_b @ ext.rotation_exp(ws[j] * (ts[j + 1] - ts[j]))
        A.append(P[i, :3, :3].T @ P[i + 1, :3, :3])
        B.append(R_b)
    return ext.solve_rotation_hand_eye(A, B)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
def run_demo(seed: int = 0) -> DoctorResult:
    """Build a synthetic dataset with known defects and diagnose it.

    Injected defects, all of which the report should recover:

    * IMU timestamps 45 ms early.
    * Configured extrinsic rotation is the transpose of the true one.
    * z drifts by a 0.30 m/min ramp with a 0.40 m step partway through.
    * The mapping segment is a corridor (degenerate along its axis).
    * The bag is missing the ``base_link -> imu_link`` static transform.
    """
    from . import synthetic as syn

    true_R = ext.euler_to_matrix([0.0, 0.0, math.pi / 2])  # IMU yawed 90 deg
    T_lidar_imu = ext.make_transform(true_R, [0.12, 0.0, -0.05])

    # --- calibration segment: rotation about several axes -----------------
    traj = syn.circle_trajectory(duration=30.0, rate=200.0, radius=6.0,
                                 period=10.0, tilt_deg=10.0)
    imu = syn.simulate_imu(traj, T_lidar_imu, gyro_noise=0.002, accel_noise=0.02,
                           seed=seed)
    imu_times_bad = syn.shift_timestamps(imu.times, offset=-0.045)  # IMU stamps early

    scan_idx = np.arange(0, len(traj), 20)  # 10 Hz
    scan_times = traj.times[scan_idx]
    scan_poses = traj.poses[scan_idx].copy()

    # --- inject z drift into the "estimated" trajectory -------------------
    scan_poses = syn.inject_z_ramp(scan_poses, scan_times, rate_m_per_s=0.30 / 60.0)
    scan_poses = syn.inject_z_step(scan_poses, index=len(scan_poses) // 2, jump_m=0.40)

    # --- mapping segment: a corridor --------------------------------------
    corridor = syn.corridor_scene(length=80.0, width=2.5, height=3.0)
    scan = syn.simulate_scan(corridor, ext.make_transform(t=[0.0, 0.0, 1.0]),
                             n_rings=32, n_azimuth=180,
                             fov_down_deg=-45.0, fov_up_deg=45.0,
                             max_range=40.0, range_noise=0.01, seed=seed)

    # --- a bag description with a missing static transform ----------------
    bag = bag_mod.load_json({
        "path": "demo_bag",
        "duration_s": 30.0,
        "message_count": 6300,
        "topics": [
            {"name": "/points", "type": "sensor_msgs/msg/PointCloud2", "count": 300},
            {"name": "/imu/data", "type": "sensor_msgs/msg/Imu", "count": 6000},
        ],
        "frame_ids": {"/points": "velodyne", "/imu/data": "imu_link"},
        "tf": [
            {"parent": "base_link", "child": "velodyne", "static": True},
            {"parent": "odom", "child": "base_link", "static": False},
        ],
    })

    res = diagnose(
        scan_times=scan_times,
        scan_poses=scan_poses,
        imu_times=imu_times_bad,
        gyro=imu.gyro,
        points=scan.points,
        extrinsic_R=true_R.T,           # the classic transposed-config bug
        extrinsic_t=[0.12, 0.0, -0.05],
        bag_info=bag,
        required_tf_chains=[("base_link", "velodyne"), ("base_link", "imu_link")],
        max_offset_s=0.12,
        source="--demo (synthetic data with known injected defects)",
    )
    res.details["injected_defects"] = {
        "imu_time_offset_ms": 45.0,
        "extrinsic": "configured rotation is the transpose of the truth",
        "z_ramp_m_per_min": 0.30,
        "z_step_m": 0.40,
        "environment": "corridor, degenerate along X",
        "tf": "base_link -> imu_link static transform missing",
    }
    return res


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------
_COLORS = {
    Severity.CRITICAL: "\033[1;31m",
    Severity.ERROR: "\033[31m",
    Severity.WARN: "\033[33m",
    Severity.INFO: "\033[36m",
    Severity.OK: "\033[32m",
}
_RESET = "\033[0m"


def format_report(res: DoctorResult, show_ok: bool = False, color: bool = True) -> str:
    """Render a :class:`DoctorResult` as the terminal report."""
    use_color = color and sys.stdout.isatty()

    def paint(sev: Severity, text: str) -> str:
        return f"{_COLORS[sev]}{text}{_RESET}" if use_color else text

    lines: List[str] = []
    lines.append("=" * 74)
    lines.append("slam-doctor")
    lines.append(f"source: {res.source}")
    lines.append("=" * 74)
    problems = res.report.problems
    shown = res.report.ranked() if show_ok else problems
    counts: Dict[str, int] = {}
    for f in res.report.findings:
        counts[f.severity.name] = counts.get(f.severity.name, 0) + 1
    if not shown:
        lines.append("")
        lines.append("No defects found by the checks that could run.")
    else:
        lines.append("")
        lines.append(f"RANKED DIAGNOSIS ({len(problems)} issue(s), worst first)")
        lines.append("")
        for i, f in enumerate(shown, 1):
            lines.append(f"{i:2d}. {paint(f.severity, '[' + f.severity.name + ']')} "
                         f"{f.code}")
            lines.append(f"    {f.message}")
            if f.symptom:
                lines.append(f"    symptom : {_wrap(f.symptom)}")
            if f.fix:
                lines.append(f"    fix     : {_wrap(f.fix)}")
            lines.append("")
    order = ["CRITICAL", "ERROR", "WARN", "INFO", "OK"]
    summary = ", ".join(f"{counts[k]} {k.lower()}" for k in order if counts.get(k))
    lines.append(f"summary: {summary or 'nothing checked'}")
    lines.append("")
    lines.append(_BRINGUP)
    lines.append("")
    return "\n".join(lines)


def _wrap(text: str, width: int = 66, indent: str = " " * 14) -> str:
    import textwrap

    wrapped = textwrap.wrap(text, width=width)
    if not wrapped:
        return ""
    return ("\n" + indent).join(wrapped)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slam-doctor",
        description="Diagnose a LiDAR SLAM setup: TF, time sync, extrinsics, "
                    "geometry, drift. Ranked output with concrete fixes.",
        epilog="Try: slam-doctor --demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--demo", action="store_true",
                   help="run on synthetic data with known injected defects")
    p.add_argument("--bag", metavar="PATH",
                   help="rosbag2 directory, `ros2 bag info` text, or JSON dump")
    p.add_argument("--trajectory", metavar="FILE",
                   help="estimated trajectory: t x y z [qx qy qz qw] (csv/txt/npy)")
    p.add_argument("--ground-truth", metavar="FILE",
                   help="ground-truth trajectory in the same format; enables ATE/RPE")
    p.add_argument("--imu", metavar="FILE",
                   help="IMU samples: t gx gy gz [ax ay az] (csv/txt/npy)")
    p.add_argument("--cloud", metavar="FILE",
                   help="one point cloud as x y z rows (csv/txt/npy)")
    p.add_argument("--extrinsic", metavar="FILE",
                   help='JSON with {"rotation": [9 row-major] or {"rpy": [r,p,y]}, '
                        '"translation": [x,y,z]}')
    p.add_argument("--tf-chain", action="append", default=[], metavar="FROM:TO",
                   help="TF chain that must resolve; repeatable")
    p.add_argument("--lidar-rate", type=float, default=10.0, metavar="HZ")
    p.add_argument("--imu-rate", type=float, default=200.0, metavar="HZ")
    p.add_argument("--max-offset", type=float, default=0.2, metavar="SECONDS",
                   help="bound on the LiDAR/IMU time-offset search (default 0.2 s)")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p.add_argument("--all", action="store_true",
                   help="show passing checks too, not just problems")
    p.add_argument("--no-color", action="store_true")
    return p


def _load_extrinsic(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    rot = data.get("rotation", data.get("extrinsicRot"))
    if isinstance(rot, dict) and "rpy" in rot:
        R = ext.euler_to_matrix(rot["rpy"], degrees=bool(rot.get("degrees", False)))
    elif rot is not None:
        arr = np.asarray(rot, dtype=float)
        R = arr.reshape(3, 3) if arr.size == 9 else ext.euler_to_matrix(arr)
    else:
        raise ValueError('extrinsic file needs a "rotation" key')
    t = data.get("translation", data.get("extrinsicTrans"))
    return R, (np.asarray(t, dtype=float) if t is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``slam-doctor`` console script."""
    args = build_parser().parse_args(argv)

    if args.demo:
        res = run_demo()
    else:
        kwargs: Dict[str, Any] = {"lidar_rate_hz": args.lidar_rate,
                                  "imu_rate_hz": args.imu_rate,
                                  "max_offset_s": args.max_offset}
        sources: List[str] = []
        if args.trajectory:
            t, poses = _trajectory_from_table(load_table(args.trajectory))
            kwargs["scan_times"] = t
            kwargs["scan_poses"] = poses
            sources.append(os.path.basename(args.trajectory))
        if args.ground_truth:
            _, gt = _trajectory_from_table(load_table(args.ground_truth))
            kwargs["ground_truth_poses"] = gt
            sources.append(os.path.basename(args.ground_truth))
        if args.imu:
            it, g, _ = _imu_from_table(load_table(args.imu))
            kwargs["imu_times"] = it
            kwargs["gyro"] = g
            sources.append(os.path.basename(args.imu))
        if args.cloud:
            kwargs["points"] = load_table(args.cloud)[:, :3]
            sources.append(os.path.basename(args.cloud))
        if args.extrinsic:
            R, t = _load_extrinsic(args.extrinsic)
            kwargs["extrinsic_R"] = R
            kwargs["extrinsic_t"] = t
            sources.append(os.path.basename(args.extrinsic))
        if args.bag:
            kwargs["bag_info"] = bag_mod.load_any(args.bag)
            sources.append(os.path.basename(args.bag))
        chains = []
        for spec in args.tf_chain:
            if ":" not in spec:
                print(f"--tf-chain expects FROM:TO, got {spec!r}", file=sys.stderr)
                return 2
            a, b = spec.split(":", 1)
            chains.append((a, b))
        kwargs["required_tf_chains"] = chains
        if len(kwargs) <= 4:
            build_parser().print_help()
            print("\nNothing to analyse. Start with:  slam-doctor --demo",
                  file=sys.stderr)
            return 2
        kwargs["source"] = ", ".join(sources)
        res = diagnose(**kwargs)

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(format_report(res, show_ok=args.all, color=not args.no_color))
    return res.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

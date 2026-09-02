"""The slam-doctor battery and CLI.

The demo injects known defects; these tests assert every one of them is found
and that the ranking puts the root causes at the top.
"""

import json
import math

import numpy as np
import pytest

from slamkit.doctor import (
    DoctorResult,
    build_parser,
    diagnose,
    format_report,
    load_table,
    main,
    run_demo,
)
from slamkit.findings import Severity
from slamkit.rosbag_check import load_json


@pytest.fixture(scope="module")
def demo():
    return run_demo(seed=0)


def test_demo_runs_and_returns_a_result(demo):
    assert isinstance(demo, DoctorResult)
    assert len(demo.report) > 10
    assert demo.exit_code == 1          # defects were injected, so non-zero


def test_demo_finds_the_injected_time_offset(demo):
    f = demo.report.get("TIME_OFFSET_LARGE")
    assert f is not None
    assert f.severity == Severity.CRITICAL
    assert abs(f.data["offset_ms"] - 45.0) < 3.0


def test_demo_finds_the_transposed_extrinsic(demo):
    f = demo.report.get("EXTRINSIC_TRANSPOSED")
    assert f is not None
    assert f.severity == Severity.CRITICAL
    assert f.data["error_if_transposed_deg"] < f.data["error_as_given_deg"]


def test_demo_finds_the_missing_tf_chain(demo):
    f = demo.report.get("TF_CHAIN_MISSING")
    assert f is not None
    assert f.data["to"] == "imu_link"


def test_demo_finds_the_corridor(demo):
    f = demo.report.get("DEGENERACY_CORRIDOR")
    assert f is not None
    assert demo.details["degeneracy"]["weakest_axis"] == "x"


def test_demo_finds_the_z_drift_step(demo):
    z = demo.details["z_drift"]
    assert z["n_steps"] >= 1
    assert z["largest_step_m"] == pytest.approx(0.40, abs=0.05)
    assert z["ramp_m_per_min"] == pytest.approx(0.30, abs=0.05)


def test_demo_ranks_critical_findings_first(demo):
    ranked = demo.report.ranked()
    assert ranked[0].severity == Severity.CRITICAL
    severities = [int(f.severity) for f in ranked]
    assert severities == sorted(severities, reverse=True)


def test_demo_result_is_json_serialisable(demo):
    text = json.dumps(demo.to_dict())
    data = json.loads(text)
    assert data["worst_severity"] == "CRITICAL"
    assert data["counts"]["CRITICAL"] >= 3
    assert len(data["bringup_order"]) == 4


def test_demo_is_deterministic():
    a = run_demo(seed=1)
    b = run_demo(seed=1)
    assert [f.code for f in a.report.ranked()] == [f.code for f in b.report.ranked()]


def test_formatted_report_mentions_the_bringup_order(demo):
    text = format_report(demo, color=False)
    assert "RANKED DIAGNOSIS" in text
    assert "BRING-UP ORDER" in text
    assert "Tuning before steps 1-3 is wasted work" in text


def test_formatted_report_can_include_passing_checks(demo):
    short = format_report(demo, show_ok=False, color=False)
    long = format_report(demo, show_ok=True, color=False)
    assert len(long) > len(short)
    assert "DEGENERACY" in long


# --------------------------------------------------------------- diagnose()
def test_diagnose_with_no_inputs_produces_an_empty_report():
    res = diagnose()
    assert len(res.report) == 0
    assert res.exit_code == 0


def test_diagnose_with_only_a_bag_runs_bag_checks():
    bag = load_json({
        "duration_s": 60.0,
        "topics": [{"name": "/points", "type": "sensor_msgs/msg/PointCloud2",
                    "count": 600}],
    })
    res = diagnose(bag_info=bag)
    assert res.report.has("BAG_MISSING_IMU")
    assert not res.report.has("TIME_LIDAR_MONOTONIC")   # no timestamps supplied


def test_diagnose_reports_ate_when_ground_truth_is_supplied():
    t = np.linspace(0.0, 10.0, 50)
    gt = np.tile(np.eye(4), (50, 1, 1))
    gt[:, 0, 3] = t
    est = gt.copy()
    est[:, 1, 3] += 0.05
    res = diagnose(scan_times=t, scan_poses=est, ground_truth_poses=gt)
    f = res.report.get("TRAJECTORY_ATE")
    assert f is not None
    assert res.details["ate"]["translation_rmse_m"] < 1e-6   # alignment removes it
    assert res.details["rpe"]["translation_rmse_m"] < 1e-9


def test_diagnose_flags_a_tilted_ground_plane():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-10.0, 10.0, (3000, 2))
    z = 0.15 * xy[:, 0] - 1.5 + rng.normal(0.0, 0.01, 3000)   # ~8.5 deg tilt
    pts = np.column_stack([xy, z])
    res = diagnose(points=pts)
    f = res.report.get("GROUND_PLANE_TILTED")
    assert f is not None
    assert f.data["tilt_deg"] == pytest.approx(math.degrees(math.atan(0.15)), abs=1.0)


def test_diagnose_warns_when_calibration_motion_is_single_axis():
    """A robot that only yaws cannot observe the roll/pitch of its extrinsic."""
    from slamkit import synthetic as syn

    traj = syn.yaw_sweep_trajectory(duration=20.0, rate=200.0, amplitude_deg=40.0,
                                    period=3.0)
    imu = syn.simulate_imu(traj, gyro_noise=0.0005, seed=0)
    idx = np.arange(0, len(traj), 20)
    res = diagnose(scan_times=traj.times[idx], scan_poses=traj.poses[idx],
                   imu_times=imu.times, gyro=imu.gyro)
    assert res.report.has("EXTRINSIC_POORLY_EXCITED")


# --------------------------------------------------------------------- CLI
def test_cli_demo_exits_nonzero(capsys):
    code = main(["--demo", "--no-color"])
    out = capsys.readouterr().out
    assert code == 1
    assert "slam-doctor" in out
    assert "EXTRINSIC_TRANSPOSED" in out


def test_cli_json_output_is_parsable(capsys):
    main(["--demo", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["worst_severity"] == "CRITICAL"
    assert any(f["code"] == "TIME_OFFSET_LARGE" for f in data["findings"])


def test_cli_with_no_arguments_prints_help(capsys):
    code = main([])
    err = capsys.readouterr().err
    assert code == 2
    assert "slam-doctor --demo" in err


def test_cli_rejects_a_malformed_tf_chain(capsys):
    code = main(["--tf-chain", "base_link"])
    assert code == 2
    assert "FROM:TO" in capsys.readouterr().err


def test_cli_reads_a_trajectory_file(tmp_path, capsys):
    path = tmp_path / "traj.csv"
    lines = ["t,x,y,z"]
    for i in range(60):
        lines.append(f"{i * 0.1},{i * 0.1},0.0,{i * 0.001}")
    path.write_text("\n".join(lines))
    code = main(["--trajectory", str(path), "--no-color"])
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "slam-doctor" in out


def test_load_table_handles_headers_comments_and_whitespace(tmp_path):
    csv = tmp_path / "a.csv"
    csv.write_text("# comment\nt,x,y\n0,1,2\n1,3,4\n")
    assert np.allclose(load_table(str(csv)), [[0, 1, 2], [1, 3, 4]])
    txt = tmp_path / "b.txt"
    txt.write_text("0 1 2\n1 3 4\n")
    assert np.allclose(load_table(str(txt)), [[0, 1, 2], [1, 3, 4]])


def test_parser_exposes_the_documented_flags():
    opts = {a.dest for a in build_parser()._actions}
    for name in ("demo", "bag", "trajectory", "ground_truth", "imu", "cloud",
                 "extrinsic", "json", "all"):
        assert name in opts

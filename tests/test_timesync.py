"""Time-sync diagnostics, tested against known injected offsets."""

import numpy as np
import pytest

from slamkit.findings import Severity
from slamkit.synthetic import shift_timestamps, simulate_imu, yaw_sweep_trajectory
from slamkit.timesync import (
    analyze_time_sync,
    detect_clock_drift,
    detect_jitter,
    detect_non_monotonic,
    detect_receive_time_mismatch,
    estimate_lidar_imu_offset,
    estimate_offset_xcorr,
    rotation_rate_from_poses,
)


def _dataset(duration=30.0, imu_rate=200.0, seed=0):
    """A yaw-sweeping trajectory with 10 Hz scans and a 200 Hz IMU."""
    traj = yaw_sweep_trajectory(duration=duration, rate=imu_rate,
                                amplitude_deg=40.0, period=3.0)
    imu = simulate_imu(traj, gyro_noise=0.002, seed=seed)
    step = int(round(imu_rate / 10.0))
    idx = np.arange(0, len(traj), step)
    return traj.times[idx], traj.poses[idx], imu


@pytest.mark.parametrize("true_offset_s", [0.0, 0.045, -0.030, 0.012])
def test_offset_estimator_recovers_injected_offset(true_offset_s):
    """The core claim: inject a known offset, recover it within 2 ms."""
    scan_t, scan_R, imu = _dataset()
    # Negative shift => the IMU stamps run early => +true_offset must be added.
    imu_t = shift_timestamps(imu.times, offset=-true_offset_s)
    est = estimate_lidar_imu_offset(scan_t, scan_R, imu_t, imu.gyro,
                                    max_offset_s=0.1)
    assert abs(est.offset_s - true_offset_s) < 0.002
    assert est.correlation > 0.9
    assert est.trustworthy


def test_offset_sign_convention_is_documented_and_correct():
    """offset_s is what you ADD to the second series' timestamps."""
    t = np.arange(0.0, 20.0, 0.005)
    sig = np.sin(2 * np.pi * 0.7 * t)
    est = estimate_offset_xcorr(t, sig, t - 0.050, sig, max_offset_s=0.2)
    assert abs(est.offset_s - 0.050) < 0.002


def test_offset_estimate_is_flagged_unreliable_without_rotation():
    """A recording with no rotation cannot localise the correlation peak."""
    t = np.arange(0.0, 20.0, 0.005)
    rng = np.random.default_rng(0)
    a = rng.normal(size=len(t)) * 0.0 + 1.0  # constant
    est = estimate_offset_xcorr(t, a, t, a, max_offset_s=0.1)
    assert not est.trustworthy


def test_rotation_rate_from_poses_matches_the_commanded_rate():
    traj = yaw_sweep_trajectory(duration=10.0, rate=100.0, amplitude_deg=30.0,
                                period=4.0)
    t, rate = rotation_rate_from_poses(traj.times, traj.poses)
    expected_peak = np.radians(30.0) * 2 * np.pi / 4.0
    assert abs(rate.max() - expected_peak) < 0.02
    assert len(t) == len(rate) == len(traj) - 1


def test_detect_clock_drift_finds_an_injected_ramp():
    scan_t, scan_R, imu = _dataset(duration=90.0)
    # 200 ppm is 12 ms per minute -- large but well inside the search window.
    imu_t = shift_timestamps(imu.times, offset=0.0, drift_ppm=-200.0)
    t_l, rate_l = rotation_rate_from_poses(scan_t, scan_R)
    rate_i = np.linalg.norm(imu.gyro, axis=1)
    d = detect_clock_drift(t_l, rate_l, imu_t, rate_i, n_windows=6,
                           max_offset_s=0.08, step_s=0.002)
    assert d["significant"]
    assert abs(d["drift_ppm"] - 200.0) < 60.0
    assert d["r_squared"] > 0.9


def test_detect_clock_drift_is_quiet_on_a_clean_clock():
    scan_t, scan_R, imu = _dataset(duration=90.0)
    t_l, rate_l = rotation_rate_from_poses(scan_t, scan_R)
    rate_i = np.linalg.norm(imu.gyro, axis=1)
    d = detect_clock_drift(t_l, rate_l, imu.times, rate_i, n_windows=6,
                           max_offset_s=0.08, step_s=0.002)
    assert not d["significant"]


def test_detect_jitter_measures_injected_jitter():
    t = np.arange(0.0, 30.0, 0.1)
    noisy = shift_timestamps(t, jitter_std=0.004, seed=1)
    clean = detect_jitter(t, expected_rate_hz=10.0)
    dirty = detect_jitter(noisy, expected_rate_hz=10.0)
    assert clean["jitter_std_ms"] < 0.1
    assert 4.0 < dirty["jitter_std_ms"] < 8.0
    assert abs(clean["mean_rate_hz"] - 10.0) < 0.01


def test_detect_jitter_counts_dropped_messages():
    t = np.concatenate([np.arange(0.0, 5.0, 0.1), np.arange(5.5, 8.0, 0.1)])
    j = detect_jitter(t, expected_rate_hz=10.0)
    # The series jumps from 4.9 s to 5.5 s: a 0.6 s gap where 0.1 s was due,
    # so five messages are missing.
    assert j["estimated_dropped"] == 5
    assert j["max_gap_s"] == pytest.approx(0.6, abs=1e-9)


def test_detect_non_monotonic_finds_a_backwards_step():
    t = np.arange(0.0, 5.0, 0.1)
    t[20] = t[19] - 0.35
    r = detect_non_monotonic(t)
    assert not r["monotonic"]
    assert r["n_backwards"] >= 1
    assert r["worst_backstep_s"] > 0.3


def test_detect_non_monotonic_passes_a_clean_series():
    r = detect_non_monotonic(np.arange(0.0, 5.0, 0.1))
    assert r["monotonic"]
    assert r["n_backwards"] == 0 and r["n_duplicate"] == 0


def test_receive_time_mismatch_detects_now_stamping():
    """A driver stamping with now() gives identical jitter on both series."""
    base = np.arange(0.0, 20.0, 0.1)
    receive = shift_timestamps(base, offset=0.020, jitter_std=0.004, seed=2)
    r = detect_receive_time_mismatch(receive, receive, expected_rate_hz=10.0)
    assert r["stamped_on_receive"]
    assert r["latency_std_ms"] < 0.01


def test_receive_time_mismatch_accepts_hardware_stamping():
    sensor = np.arange(0.0, 20.0, 0.1)
    receive = shift_timestamps(sensor, offset=0.015, jitter_std=0.003, seed=3)
    r = detect_receive_time_mismatch(sensor, receive, expected_rate_hz=10.0)
    assert not r["stamped_on_receive"]
    assert 10.0 < r["mean_latency_ms"] < 20.0
    assert not r["negative_latency"]


def test_receive_time_mismatch_flags_a_sensor_clock_ahead_of_the_host():
    sensor = np.arange(0.0, 20.0, 0.1) + 0.05
    receive = np.arange(0.0, 20.0, 0.1)
    r = detect_receive_time_mismatch(sensor, receive, expected_rate_hz=10.0)
    assert r["negative_latency"]


def test_analyze_time_sync_reports_a_critical_offset_with_a_verdict():
    scan_t, scan_R, imu = _dataset()
    imu_t = shift_timestamps(imu.times, offset=-0.045)
    rep = analyze_time_sync(scan_t, scan_R, imu_t, imu.gyro,
                            lidar_rate_hz=10.0, imu_rate_hz=200.0)
    assert rep.findings.has("TIME_OFFSET_LARGE")
    assert rep.findings.worst >= Severity.CRITICAL
    assert abs(rep.offset_ms - 45.0) < 2.0
    assert "BROKEN" in rep.verdict


def test_analyze_time_sync_is_clean_on_synchronised_data():
    scan_t, scan_R, imu = _dataset()
    rep = analyze_time_sync(scan_t, scan_R, imu.times, imu.gyro)
    assert rep.findings.has("TIME_OFFSET_OK")
    assert "synchronised" in rep.verdict
    assert not any(f.code.startswith("TIME_OFFSET_LARGE") for f in rep.findings)


def test_analyze_time_sync_flags_non_monotonic_lidar_stamps():
    scan_t, scan_R, imu = _dataset()
    scan_t = scan_t.copy()
    scan_t[10] = scan_t[9] - 0.5
    rep = analyze_time_sync(scan_t, lidar_rate_hz=10.0)
    assert rep.findings.has("TIME_LIDAR_NON_MONOTONIC")
    assert rep.findings.worst == Severity.CRITICAL


def test_analyze_time_sync_flags_a_slow_imu():
    scan_t, scan_R, imu = _dataset(imu_rate=50.0)
    rep = analyze_time_sync(scan_t, scan_R, imu.times, imu.gyro, imu_rate_hz=50.0)
    assert rep.findings.has("TIME_IMU_RATE_LOW")


def test_report_is_json_serialisable():
    import json

    scan_t, scan_R, imu = _dataset()
    rep = analyze_time_sync(scan_t, scan_R, imu.times, imu.gyro)
    text = json.dumps(rep.to_dict(), default=float)
    assert "verdict" in text


def test_smooth_motion_gives_a_wide_peak_and_is_not_trusted():
    """A slow sinusoid localises the offset poorly; the report must say so."""
    t = np.arange(0.0, 20.0, 0.005)
    slow = np.sin(2 * np.pi * 0.2 * t)
    fast = np.sin(2 * np.pi * 2.0 * t)
    slow_est = estimate_offset_xcorr(t, slow, t, slow, max_offset_s=0.2)
    fast_est = estimate_offset_xcorr(t, fast, t, fast, max_offset_s=0.2)
    assert slow_est.peak_width_ms > fast_est.peak_width_ms
    assert not slow_est.trustworthy
    assert fast_est.trustworthy


def test_peak_width_does_not_depend_on_the_search_window():
    scan_t, scan_R, imu = _dataset()
    wide = estimate_lidar_imu_offset(scan_t, scan_R, imu.times, imu.gyro,
                                     max_offset_s=0.2)
    narrow = estimate_lidar_imu_offset(scan_t, scan_R, imu.times, imu.gyro,
                                       max_offset_s=0.1)
    assert abs(wide.peak_width_ms - narrow.peak_width_ms) < 1e-6
    # ...whereas the raw sharpness statistic does.
    assert wide.sharpness > narrow.sharpness

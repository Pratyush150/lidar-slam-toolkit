"""The z-drift analyser: can it tell a ramp from a step?

That distinction is the whole point. A ramp and a step both show up as "the map
floats upward" and they need opposite fixes -- IMU/attitude work for a ramp,
loop-closure or geometry work for a step.
"""

import numpy as np
import pytest

from slamkit.drift import ZDriftReport, analyze_z_drift, z_drift_findings
from slamkit.findings import Severity


def _times(duration=180.0, n=360):
    return np.linspace(0.0, duration, n)


def test_pure_ramp_is_reported_as_a_ramp():
    t = _times()
    rng = np.random.default_rng(0)
    z = 0.004 * t + rng.normal(0.0, 0.002, len(t))     # 0.24 m/min
    r = analyze_z_drift(t, z)
    assert r.n_steps == 0
    assert r.ramp_share > 0.95
    assert r.step_share < 0.05
    assert r.ramp_m_per_min == pytest.approx(0.24, abs=0.02)
    assert "step" not in r.likely_cause.lower()


def test_pure_step_is_reported_as_a_step():
    t = _times()
    rng = np.random.default_rng(1)
    z = rng.normal(0.0, 0.002, len(t))
    z[len(t) // 2:] += 0.40
    r = analyze_z_drift(t, z)
    assert r.n_steps == 1
    assert r.step_share > 0.9
    assert r.largest_step_m == pytest.approx(0.40, abs=0.02)
    assert abs(r.ramp_m_per_min) < 0.02
    assert "step" in r.likely_cause.lower() or "loop closure" in r.likely_cause.lower()


def test_step_index_is_located_correctly():
    t = _times(n=200)
    z = np.zeros(200)
    z[120:] += 0.5
    r = analyze_z_drift(t, z)
    assert r.step_indices == [120]
    assert r.step_sizes_m[0] == pytest.approx(0.5, abs=1e-9)


def test_multiple_steps_are_all_found():
    t = _times(n=300)
    z = np.zeros(300)
    z[100:] += 0.3
    z[200:] -= 0.5
    r = analyze_z_drift(t, z)
    assert r.n_steps == 2
    assert r.total_step_m == pytest.approx(0.8, abs=1e-6)
    assert r.largest_step_m == pytest.approx(0.5, abs=1e-6)


def test_ramp_estimate_is_not_corrupted_by_a_step():
    """The reason steps are removed before fitting: one jump would tilt the line."""
    t = _times()
    z = 0.004 * t.copy()
    z[len(t) // 2:] += 1.0
    r = analyze_z_drift(t, z)
    assert r.ramp_m_per_min == pytest.approx(0.24, abs=0.03)
    assert r.n_steps == 1
    assert r.largest_step_m == pytest.approx(1.0, abs=0.01)


def test_mixed_ramp_and_step_is_reported_as_mixed():
    t = _times()
    z = 0.004 * t.copy()                # ~0.72 m over 180 s
    z[len(t) // 2:] += 0.7              # comparable step
    r = analyze_z_drift(t, z)
    assert 0.3 < r.ramp_share < 0.7
    assert "mixed" in r.likely_cause.lower()
    assert r.confidence == "low"


def test_distance_correlated_ramp_is_blamed_on_attitude():
    """z growing with distance travelled, not with time, means a tilted map."""
    t = _times(duration=120.0, n=240)
    speed = np.where(t < 60.0, 1.0, 0.2)          # slows down halfway
    dist = np.concatenate([[0.0], np.cumsum(speed[:-1] * np.diff(t))])
    z = 0.02 * dist                                # 2 cm per metre = 1.15 deg tilt
    r = analyze_z_drift(t, z, horizontal_distance=dist)
    assert "attitude" in r.likely_cause.lower()
    assert r.distance_correlation == pytest.approx(0.02, abs=1e-3)
    assert r.confidence in ("high", "medium")


def test_time_correlated_ramp_is_not_blamed_on_attitude():
    """Same total drift but the robot is stationary for half of it."""
    t = _times(duration=120.0, n=240)
    speed = np.where(t < 60.0, 1.0, 0.0)
    dist = np.concatenate([[0.0], np.cumsum(speed[:-1] * np.diff(t))])
    z = 0.004 * t
    r = analyze_z_drift(t, z, horizontal_distance=dist)
    assert "attitude" not in r.likely_cause.lower()
    assert "bias" in r.likely_cause.lower() or "gravity" in r.likely_cause.lower()


def test_quadratic_growth_is_blamed_on_bias_integration():
    t = _times(duration=120.0, n=240)
    z = 1e-4 * t ** 2                     # double-integrated constant bias
    r = analyze_z_drift(t, z)
    assert "bias" in r.likely_cause.lower()
    assert r.curvature == pytest.approx(1e-4, rel=0.2)


def test_degeneracy_hint_raises_confidence_on_the_geometry_hypothesis():
    t = _times()
    z = np.zeros(len(t))
    z[len(t) // 2:] += 0.5
    r = analyze_z_drift(t, z, degeneracy_fraction=0.6)
    assert "degenerate" in r.likely_cause.lower()
    assert r.confidence == "high"


def test_flat_trace_reports_no_drift():
    t = _times()
    r = analyze_z_drift(t, np.zeros(len(t)))
    assert r.n_steps == 0
    assert "no measurable" in r.likely_cause.lower()
    assert z_drift_findings(r)[0].severity == Severity.OK


def test_findings_escalate_with_the_drift_rate():
    t = _times()
    slow = z_drift_findings(analyze_z_drift(t, 0.0005 * t))     # 0.03 m/min
    fast = z_drift_findings(analyze_z_drift(t, 0.02 * t))       # 1.2 m/min
    assert slow[0].severity == Severity.OK
    assert fast[0].severity == Severity.ERROR
    assert fast[0].fix != ""


def test_findings_carry_a_cause_specific_fix():
    t = _times(duration=120.0, n=240)
    dist = np.linspace(0.0, 100.0, 240)
    attitude = z_drift_findings(analyze_z_drift(t, 0.02 * dist,
                                                horizontal_distance=dist))[0]
    z = np.zeros(240)
    z[120:] += 0.6
    steps = z_drift_findings(analyze_z_drift(t, z))[0]
    assert attitude.code == "ZDRIFT_ATTITUDE"
    assert steps.code == "ZDRIFT_STEPS"
    assert "extrinsic" in attitude.fix.lower()
    assert "loop closure" in steps.fix.lower()


def test_report_serialises():
    t = _times()
    r = analyze_z_drift(t, 0.004 * t)
    d = r.to_dict()
    assert isinstance(r, ZDriftReport)
    assert set(["ramp_m_per_min", "n_steps", "likely_cause", "evidence"]) <= set(d)


def test_rejects_too_few_samples():
    with pytest.raises(ValueError):
        analyze_z_drift(np.arange(3.0), np.zeros(3))

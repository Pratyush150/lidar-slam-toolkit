"""Time-synchronisation diagnostics for LiDAR + IMU.

Why 10 ms matters (and why "it looks fine on the bench" proves nothing)
----------------------------------------------------------------------
A LiDAR-inertial system fuses two things: where the IMU says you rotated, and
where the geometry says you rotated.  It assumes both statements refer to the
same instant.  An un-modelled time offset ``dt`` breaks that assumption, and
the resulting error is

    angular error  ~  omega * dt
    position error ~  v * dt   +   omega * dt * r

where ``omega`` is angular rate, ``v`` linear speed and ``r`` the range to the
structure you are matching against.

Put numbers in.  A handheld or ground robot turning at 1 rad/s (57 deg/s --
not fast) with a 10 ms offset gets 0.57 deg of angular error *per scan*.
Matching a wall 20 m away, 0.57 deg is 20 cm of apparent displacement.  The
optimiser sees the IMU and the scan matcher disagreeing by 20 cm, splits the
difference, and writes the residual into the accelerometer and gyro bias
states.  Those biases are supposed to change over minutes; now they are being
yanked around every time you turn.  Once the bias estimate is corrupted, the
gravity direction it implies is wrong, so z drifts -- and you go and file a
bug about z drift.

Now stand still.  ``omega = 0`` and ``v = 0``, so both error terms vanish.
The map is perfect.  This is the entire reason time sync bugs survive testing:
**the failure is proportional to motion**.  "It works fine until I turn" is
almost never a tuning problem.

Rules of thumb for a 10 Hz spinning LiDAR and a 200 Hz IMU:

===============  ==========================================================
< 1 ms           Fine. This is what PTP or a PPS-disciplined sensor gives.
1-5 ms           Acceptable for slow ground robots. Visible on fast yaw.
5-20 ms          Degrades noticeably. Ghosting on turns, biased gravity.
> 20 ms          Broken. Fix this before touching any other parameter.
> 100 ms         Usually a whole-scan misalignment: you are stamping with
                 end-of-sweep time while the consumer assumes start-of-sweep,
                 or you are using ROS receive time on a buffered USB link.
===============  ==========================================================

What this module measures
-------------------------
* Constant offset, by cross-correlating IMU angular-rate magnitude against
  the rotation rate implied by consecutive scan-matched poses.  Angular rate
  is the right signal: it is frame-covariant, so it works before you know the
  extrinsic rotation, and it has sharp features that correlate well.
* Clock drift, by repeating that estimate in windows and fitting a line.
* Jitter and dropped messages from the timestamp series alone.
* Non-monotonic timestamps, which make most estimators either crash or
  silently discard data.
* Sensor-clock vs ROS-receive-time stamping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .extrinsics import rotation_log
from .findings import Finding, Report, Severity

__all__ = [
    "OffsetEstimate",
    "estimate_offset_xcorr",
    "rotation_rate_from_poses",
    "estimate_lidar_imu_offset",
    "detect_clock_drift",
    "detect_jitter",
    "detect_non_monotonic",
    "detect_receive_time_mismatch",
    "TimeSyncReport",
    "analyze_time_sync",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# Cross-correlation offset estimation
# --------------------------------------------------------------------------
@dataclass
class OffsetEstimate:
    """Result of a cross-correlation time-offset search."""

    offset_s: float
    """Seconds to ADD to the second (``b``) timestamp series to align it with ``a``."""

    correlation: float
    """Peak normalised correlation in ``[-1, 1]``. Below ~0.5 do not trust the offset."""

    sharpness: float
    """Peak minus the median of the correlation curve. Descriptive only: it
    depends on how wide a search window you asked for, so it is not a good
    gate on its own -- use :attr:`peak_width_ms`."""

    peak_width_ms: float = float("inf")
    """Width of the correlation peak within 0.02 of its maximum, milliseconds.

    This is the honest measure of how well the data localises the offset, and
    unlike :attr:`sharpness` it does not depend on the search window. A narrow
    peak (tens of ms) means sharp motion pinned the answer down; a wide peak
    (hundreds of ms) means the motion was too smooth and small amounts of noise
    will move the estimate around. Record yaw reversals, not gentle sweeps."""

    curve: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    candidates: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    n_samples: int = 0

    @property
    def offset_ms(self) -> float:
        return self.offset_s * 1000.0

    @property
    def trustworthy(self) -> bool:
        """Coarse gate: strong correlation, a localised peak, enough samples."""
        return (self.correlation > 0.5
                and self.peak_width_ms < 250.0
                and self.n_samples > 20)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = float(np.std(x))
    if s < _EPS:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / s


def estimate_offset_xcorr(
    t_a: np.ndarray,
    sig_a: np.ndarray,
    t_b: np.ndarray,
    sig_b: np.ndarray,
    max_offset_s: float = 0.2,
    step_s: float = 0.001,
) -> OffsetEstimate:
    """Find the time offset that best aligns two irregularly sampled signals.

    Both signals are resampled onto a uniform grid (linear interpolation) over
    the overlapping, offset-padded interval, then correlated over a dense
    sweep of candidate offsets.  The peak is refined by fitting a parabola to
    the three samples around it, which gets you well below ``step_s``.

    Returns an :class:`OffsetEstimate` whose ``offset_s`` should be **added to
    ``t_b``**.  So a result of ``-0.045`` means the ``b`` stamps are 45 ms
    late.

    A dense sweep rather than an FFT because the series are short (a few
    thousand samples), irregular, and we want the shape of the correlation
    curve to judge whether the peak is real.
    """
    ta = np.asarray(t_a, dtype=float).reshape(-1)
    tb = np.asarray(t_b, dtype=float).reshape(-1)
    ya = np.asarray(sig_a, dtype=float).reshape(-1)
    yb = np.asarray(sig_b, dtype=float).reshape(-1)
    if len(ta) != len(ya) or len(tb) != len(yb):
        raise ValueError("timestamp and signal lengths must match")
    if len(ta) < 4 or len(tb) < 4:
        raise ValueError("need at least 4 samples in each series")
    lo = max(float(ta[0]), float(tb[0])) + max_offset_s
    hi = min(float(ta[-1]), float(tb[-1])) - max_offset_s
    if hi - lo <= 10 * step_s:
        raise ValueError(
            "the two series barely overlap once padded for the offset search; "
            "record a longer segment or reduce max_offset_s"
        )
    grid = np.arange(lo, hi, step_s)
    a_g = _zscore(np.interp(grid, ta, ya))
    candidates = np.arange(-max_offset_s, max_offset_s + 0.5 * step_s, step_s)
    corr = np.empty(len(candidates), dtype=float)
    for i, d in enumerate(candidates):
        # d is added to t_b, so sampling b at (grid - d) in b's own clock
        # yields the value b reports at grid time after correction.
        b_g = _zscore(np.interp(grid - d, tb, yb))
        corr[i] = float(np.mean(a_g * b_g))
    k = int(np.argmax(corr))
    peak = float(corr[k])
    off = float(candidates[k])
    if 0 < k < len(corr) - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            off += float(np.clip(delta, -1.0, 1.0)) * step_s
    sharpness = peak - float(np.median(corr))
    # Width of the contiguous run around the peak that stays within 0.02 of it.
    lo_i = hi_i = k
    while lo_i > 0 and corr[lo_i - 1] >= peak - 0.02:
        lo_i -= 1
    while hi_i < len(corr) - 1 and corr[hi_i + 1] >= peak - 0.02:
        hi_i += 1
    width_ms = (hi_i - lo_i + 1) * step_s * 1000.0
    return OffsetEstimate(
        offset_s=off,
        correlation=peak,
        sharpness=sharpness,
        peak_width_ms=width_ms,
        curve=corr,
        candidates=candidates,
        n_samples=len(grid),
    )


def rotation_rate_from_poses(times: np.ndarray, rotations: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Angular-rate magnitude implied by a sequence of scan-matched rotations.

    Returns ``(midpoint_times, rate_rad_s)``.  Magnitude rather than the
    vector so the estimate does not depend on knowing the LiDAR->IMU extrinsic
    rotation -- which is the point: **fix time before you fix extrinsics.**
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    R = np.asarray(rotations, dtype=float)
    if R.ndim == 3 and R.shape[1:] == (4, 4):
        R = R[:, :3, :3]
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"rotations must be (N, 3, 3) or (N, 4, 4), got {R.shape}")
    if len(t) != len(R):
        raise ValueError("times and rotations must be the same length")
    if len(t) < 3:
        raise ValueError("need at least 3 poses")
    dt = np.diff(t)
    good = dt > _EPS
    mids = (t[:-1] + t[1:]) / 2.0
    rate = np.zeros(len(dt), dtype=float)
    for i in range(len(dt)):
        if not good[i]:
            continue
        rate[i] = float(np.linalg.norm(rotation_log(R[i].T @ R[i + 1]))) / dt[i]
    return mids[good], rate[good]


def estimate_lidar_imu_offset(
    scan_times: np.ndarray,
    scan_rotations: np.ndarray,
    imu_times: np.ndarray,
    gyro: np.ndarray,
    max_offset_s: float = 0.2,
    step_s: float = 0.001,
) -> OffsetEstimate:
    """Estimate the LiDAR-vs-IMU time offset from rotation.

    ``scan_rotations`` are per-scan orientations (from scan matching, or from
    ground truth in a test).  ``gyro`` is ``(M, 3)`` rad/s.

    The returned offset is what you must **add to the IMU timestamps** to line
    them up with the LiDAR.  Feed it to ``imuTopic`` republishing, to a
    ``message_filters`` time offset, or -- properly -- fix the driver.

    The LiDAR series is short (10 Hz) and the IMU series is long (100-400 Hz).
    That is fine: both are resampled onto a 1 kHz grid before correlating, so
    the resolution of the estimate is set by ``step_s``, not by the scan rate.
    """
    t_l, rate_l = rotation_rate_from_poses(scan_times, scan_rotations)
    g = np.asarray(gyro, dtype=float).reshape(-1, 3)
    t_i = np.asarray(imu_times, dtype=float).reshape(-1)
    if len(t_i) != len(g):
        raise ValueError("imu_times and gyro must be the same length")
    rate_i = np.linalg.norm(g, axis=1)
    return estimate_offset_xcorr(t_l, rate_l, t_i, rate_i,
                                 max_offset_s=max_offset_s, step_s=step_s)


# --------------------------------------------------------------------------
# Drift, jitter, monotonicity
# --------------------------------------------------------------------------
def detect_clock_drift(
    t_a: np.ndarray,
    sig_a: np.ndarray,
    t_b: np.ndarray,
    sig_b: np.ndarray,
    n_windows: int = 5,
    max_offset_s: float = 0.2,
    step_s: float = 0.001,
) -> Dict[str, object]:
    """Detect a linearly growing offset by estimating it in successive windows.

    Two free-running crystals differ by tens of ppm.  50 ppm is 3 ms/minute,
    which is invisible on a 30 s test bag and fatal on a 20 minute survey: the
    offset walks through the whole "acceptable" band and out the other side,
    so the map is good at the start and progressively worse.

    Returns a dict with ``drift_ppm``, ``drift_ms_per_min``, ``r_squared``,
    the per-window offsets, and ``significant``.
    """
    ta = np.asarray(t_a, dtype=float).reshape(-1)
    tb = np.asarray(t_b, dtype=float).reshape(-1)
    lo = max(float(ta[0]), float(tb[0]))
    hi = min(float(ta[-1]), float(tb[-1]))
    if n_windows < 3:
        raise ValueError("need at least 3 windows to fit a drift line")
    edges = np.linspace(lo, hi, n_windows + 1)
    centers: List[float] = []
    offsets: List[float] = []
    corrs: List[float] = []
    for i in range(n_windows):
        w0, w1 = edges[i], edges[i + 1]
        ma = (ta >= w0) & (ta <= w1)
        mb = (tb >= w0 - max_offset_s) & (tb <= w1 + max_offset_s)
        if ma.sum() < 6 or mb.sum() < 6:
            continue
        try:
            est = estimate_offset_xcorr(
                ta[ma], np.asarray(sig_a)[ma], tb[mb], np.asarray(sig_b)[mb],
                max_offset_s=max_offset_s, step_s=step_s,
            )
        except ValueError:
            continue
        centers.append(0.5 * (w0 + w1))
        offsets.append(est.offset_s)
        corrs.append(est.correlation)
    out: Dict[str, object] = {
        "window_centers": centers,
        "window_offsets_ms": [o * 1000.0 for o in offsets],
        "window_correlations": corrs,
        "n_windows_used": len(centers),
    }
    if len(centers) < 3:
        out.update({"drift_ppm": 0.0, "drift_ms_per_min": 0.0, "r_squared": 0.0,
                    "significant": False,
                    "reason": "not enough usable windows to fit a drift line"})
        return out
    c = np.asarray(centers) - centers[0]
    o = np.asarray(offsets)
    A = np.vstack([c, np.ones_like(c)]).T
    coef, *_ = np.linalg.lstsq(A, o, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(np.sum((o - pred) ** 2))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _EPS else 0.0
    span = float(c[-1] - c[0])
    total_change_ms = abs(slope) * span * 1000.0
    out.update({
        "drift_ppm": slope * 1e6,
        "drift_ms_per_min": slope * 60.0 * 1000.0,
        "intercept_ms": intercept * 1000.0,
        "r_squared": r2,
        "total_change_ms": total_change_ms,
        # Require both a good line fit and a change bigger than the estimator's
        # own resolution, otherwise every noisy estimate looks like drift.
        "significant": bool(r2 > 0.7 and total_change_ms > 4.0 * step_s * 1000.0),
    })
    return out


def detect_jitter(stamps: np.ndarray, expected_rate_hz: Optional[float] = None
                  ) -> Dict[str, object]:
    """Characterise the inter-message interval of a timestamp series.

    Returns median/mean rate, jitter standard deviation, the worst gap, and an
    estimate of how many messages were dropped (gaps that are near-integer
    multiples of the nominal period).

    Jitter above ~20% of the nominal period on a LiDAR topic almost always
    means the stamps are ROS receive time rather than sensor time; a spinning
    LiDAR's motor is far more stable than that.
    """
    t = np.asarray(stamps, dtype=float).reshape(-1)
    if len(t) < 3:
        raise ValueError("need at least 3 timestamps")
    dt = np.diff(t)
    pos = dt[dt > 0]
    median_dt = float(np.median(pos)) if len(pos) else float("nan")
    nominal = 1.0 / expected_rate_hz if expected_rate_hz else median_dt
    jitter = float(np.std(pos)) if len(pos) else float("nan")
    max_gap = float(np.max(dt)) if len(dt) else float("nan")
    dropped = 0
    if nominal and np.isfinite(nominal) and nominal > 0:
        multiples = np.round(pos / nominal)
        dropped = int(np.sum(np.maximum(multiples - 1, 0)))
    return {
        "n": int(len(t)),
        "duration_s": float(t[-1] - t[0]),
        "median_dt_s": median_dt,
        "mean_rate_hz": float(1.0 / median_dt) if median_dt > 0 else float("nan"),
        "nominal_dt_s": float(nominal) if nominal else float("nan"),
        "jitter_std_s": jitter,
        "jitter_std_ms": jitter * 1000.0,
        "jitter_ratio": float(jitter / nominal) if nominal else float("nan"),
        "max_gap_s": max_gap,
        "max_gap_ratio": float(max_gap / nominal) if nominal else float("nan"),
        "estimated_dropped": dropped,
    }


def detect_non_monotonic(stamps: np.ndarray) -> Dict[str, object]:
    """Find timestamps that go backwards or repeat.

    Backwards stamps come from: a bag recorded across a clock step (NTP
    correcting the system clock mid-run), two publishers on one topic, or a
    driver that stamps in a worker thread without a lock.  Most estimators
    respond by discarding data silently, so the symptom is "SLAM ignores
    half my scans" rather than a crash.
    """
    t = np.asarray(stamps, dtype=float).reshape(-1)
    dt = np.diff(t)
    back = np.where(dt < 0)[0]
    dup = np.where(dt == 0)[0]
    return {
        "n": int(len(t)),
        "n_backwards": int(len(back)),
        "n_duplicate": int(len(dup)),
        "backwards_indices": back[:20].tolist(),
        "duplicate_indices": dup[:20].tolist(),
        "worst_backstep_s": float(-dt[back].min()) if len(back) else 0.0,
        "monotonic": bool(len(back) == 0 and len(dup) == 0),
    }


def detect_receive_time_mismatch(
    sensor_stamps: np.ndarray,
    receive_stamps: np.ndarray,
    expected_rate_hz: Optional[float] = None,
) -> Dict[str, object]:
    """Compare a sensor's own clock against the time the message was received.

    ``receive_stamps`` is what ``rclcpp::Clock().now()`` returns in the
    subscriber callback (rosbag2 records it as the message's receive time).
    ``sensor_stamps`` is ``msg.header.stamp``.

    Three outcomes matter:

    * ``latency`` roughly constant with < 1 ms spread -- the sensor is
      hardware-timestamped and the two clocks are disciplined together. Good.
    * ``latency`` spread of several ms -- normal transport jitter. Fine, as
      long as ``header.stamp`` is the *sensor* clock.
    * ``latency`` spread near zero *and* the sensor stamps as jittery as the
      receive stamps -- the driver is stamping with ``now()`` in the callback.
      That is the killer: your "sensor time" is actually USB/Ethernet arrival
      time plus kernel scheduling, and it wanders by milliseconds per message.
    * ``latency`` negative -- the sensor clock is ahead of the host. Untreated,
      this makes TF lookups fail with "requested time is in the future".
    """
    s = np.asarray(sensor_stamps, dtype=float).reshape(-1)
    r = np.asarray(receive_stamps, dtype=float).reshape(-1)
    if len(s) != len(r):
        raise ValueError("sensor_stamps and receive_stamps must be the same length")
    if len(s) < 3:
        raise ValueError("need at least 3 message pairs")
    lat = r - s
    s_jit = detect_jitter(s, expected_rate_hz)
    r_jit = detect_jitter(r, expected_rate_hz)
    lat_std = float(np.std(lat))
    same_jitter = (
        s_jit["jitter_std_s"] > 0
        and r_jit["jitter_std_s"] > 0
        and abs(s_jit["jitter_std_s"] - r_jit["jitter_std_s"])
        < 0.2 * max(s_jit["jitter_std_s"], r_jit["jitter_std_s"])
    )
    stamped_on_receive = bool(lat_std < 1e-4 and same_jitter and s_jit["jitter_std_s"] > 5e-4)
    return {
        "mean_latency_s": float(np.mean(lat)),
        "mean_latency_ms": float(np.mean(lat)) * 1000.0,
        "latency_std_ms": lat_std * 1000.0,
        "min_latency_ms": float(np.min(lat)) * 1000.0,
        "max_latency_ms": float(np.max(lat)) * 1000.0,
        "sensor_jitter_ms": s_jit["jitter_std_ms"],
        "receive_jitter_ms": r_jit["jitter_std_ms"],
        "negative_latency": bool(np.min(lat) < 0),
        "stamped_on_receive": stamped_on_receive,
    }


# --------------------------------------------------------------------------
# Aggregate report
# --------------------------------------------------------------------------
@dataclass
class TimeSyncReport:
    """Everything :func:`analyze_time_sync` measured, plus a verdict."""

    offset: Optional[OffsetEstimate] = None
    drift: Optional[Dict[str, object]] = None
    lidar_jitter: Optional[Dict[str, object]] = None
    imu_jitter: Optional[Dict[str, object]] = None
    monotonic_lidar: Optional[Dict[str, object]] = None
    monotonic_imu: Optional[Dict[str, object]] = None
    receive_mismatch: Optional[Dict[str, object]] = None
    findings: Report = field(default_factory=lambda: Report(title="Time sync"))

    @property
    def offset_ms(self) -> float:
        return self.offset.offset_ms if self.offset else float("nan")

    @property
    def verdict(self) -> str:
        """One-line human verdict."""
        if self.offset is None:
            return "no offset estimate (insufficient rotation in the data)"
        ms = abs(self.offset.offset_ms)
        if not self.offset.trustworthy:
            return (f"offset estimate {self.offset.offset_ms:+.1f} ms is NOT reliable "
                    f"(correlation {self.offset.correlation:.2f}, peak width "
                    f"{self.offset.peak_width_ms:.0f} ms); the recording has too little "
                    "rotation to localise the peak -- redo it with sharp yaw reversals")
        if ms < 1.0:
            return f"offset {self.offset.offset_ms:+.1f} ms -- synchronised, no action needed"
        if ms < 5.0:
            return (f"offset {self.offset.offset_ms:+.1f} ms -- acceptable for slow motion, "
                    "visible as mild ghosting on fast turns")
        if ms < 20.0:
            return (f"offset {self.offset.offset_ms:+.1f} ms -- degrading the solution; "
                    "expect ghosting on turns and a biased gravity estimate")
        if ms < 100.0:
            return (f"offset {self.offset.offset_ms:+.1f} ms -- BROKEN. Fix time sync "
                    "before tuning anything else")
        return (f"offset {self.offset.offset_ms:+.1f} ms -- whole-scan misalignment. "
                "Check start-of-sweep vs end-of-sweep stamping and whether you are "
                "using ROS receive time")

    def to_dict(self) -> Dict[str, object]:
        return {
            "verdict": self.verdict,
            "offset_ms": self.offset_ms,
            "offset_correlation": self.offset.correlation if self.offset else None,
            "offset_peak_width_ms": self.offset.peak_width_ms if self.offset else None,
            "offset_trustworthy": self.offset.trustworthy if self.offset else False,
            "drift": self.drift,
            "lidar_jitter": self.lidar_jitter,
            "imu_jitter": self.imu_jitter,
            "monotonic_lidar": self.monotonic_lidar,
            "monotonic_imu": self.monotonic_imu,
            "receive_mismatch": self.receive_mismatch,
            "findings": self.findings.to_dict(),
        }


def analyze_time_sync(
    scan_times: np.ndarray,
    scan_rotations: Optional[np.ndarray] = None,
    imu_times: Optional[np.ndarray] = None,
    gyro: Optional[np.ndarray] = None,
    lidar_rate_hz: float = 10.0,
    imu_rate_hz: float = 200.0,
    scan_receive_times: Optional[np.ndarray] = None,
    max_offset_s: float = 0.2,
    step_s: float = 0.001,
    check_drift: bool = True,
) -> TimeSyncReport:
    """Run the whole time-sync battery and produce ranked findings.

    Only ``scan_times`` is required; every other input unlocks more checks.
    """
    rep = TimeSyncReport()
    r = rep.findings

    # --- monotonicity: cheapest check, and it invalidates everything else ---
    rep.monotonic_lidar = detect_non_monotonic(scan_times)
    if not rep.monotonic_lidar["monotonic"]:
        r.add(Finding(
            code="TIME_LIDAR_NON_MONOTONIC",
            severity=Severity.CRITICAL,
            message=f"{rep.monotonic_lidar['n_backwards']} backwards and "
                    f"{rep.monotonic_lidar['n_duplicate']} duplicate LiDAR timestamps "
                    f"(worst backstep {rep.monotonic_lidar['worst_backstep_s'] * 1000:.1f} ms)",
            symptom="SLAM appears to ignore data: the map updates in bursts, or the "
                    "node logs 'message dropped, old timestamp' and the trajectory has "
                    "gaps.",
            fix="Usually a clock step during recording (NTP correcting the system "
                "clock) or two nodes publishing on one topic. Record with the machine "
                "clock already settled, or use `ros2 bag record --use-sim-time` "
                "consistently. Check `ros2 topic info -v` for a second publisher.",
            data=rep.monotonic_lidar,
        ))
    else:
        r.add(Finding(code="TIME_LIDAR_MONOTONIC", severity=Severity.OK,
                      message=f"{rep.monotonic_lidar['n']} LiDAR timestamps are strictly "
                              "increasing"))

    rep.lidar_jitter = detect_jitter(scan_times, lidar_rate_hz)
    jr = rep.lidar_jitter["jitter_ratio"]
    if np.isfinite(jr) and jr > 0.2:
        r.add(Finding(
            code="TIME_LIDAR_JITTER",
            severity=Severity.ERROR,
            message=f"LiDAR inter-scan jitter is {rep.lidar_jitter['jitter_std_ms']:.1f} ms "
                    f"({jr * 100:.0f}% of the {1000.0 / lidar_rate_hz:.0f} ms nominal period)",
            symptom="Deskewing is wrong by a variable amount every scan, so the cloud "
                    "smears differently each frame and the map looks 'fuzzy' rather "
                    "than doubled.",
            fix="A spinning LiDAR's motor is stable to well under 1%. This much jitter "
                "means header.stamp is ROS receive time. Configure the driver to use "
                "the sensor's own clock (Ouster: timestamp_mode TIME_FROM_PTP_1588 or "
                "TIME_FROM_SYNC_PULSE_IN; Velodyne: enable GPS/PPS).",
            data=rep.lidar_jitter,
        ))
    else:
        r.add(Finding(code="TIME_LIDAR_RATE_OK", severity=Severity.OK,
                      message=f"LiDAR at {rep.lidar_jitter['mean_rate_hz']:.2f} Hz, jitter "
                              f"{rep.lidar_jitter['jitter_std_ms']:.2f} ms",
                      data=rep.lidar_jitter))
    if rep.lidar_jitter["estimated_dropped"] > 0:
        r.add(Finding(
            code="TIME_LIDAR_DROPS",
            severity=Severity.WARN,
            message=f"about {rep.lidar_jitter['estimated_dropped']} LiDAR messages are "
                    f"missing (worst gap {rep.lidar_jitter['max_gap_s'] * 1000:.0f} ms)",
            symptom="The trajectory jumps at the gaps; if the robot was turning, the "
                    "map gets a visible seam there.",
            fix="Dropped messages are almost always QoS or bandwidth: a BEST_EFFORT "
                "publisher with a RELIABLE subscriber, a small queue depth, or a "
                "1 Gbit link saturated by a 64-beam sensor. Check `ros2 topic hz` at "
                "the source and at the consumer.",
            data=rep.lidar_jitter,
        ))

    if imu_times is not None:
        rep.monotonic_imu = detect_non_monotonic(imu_times)
        rep.imu_jitter = detect_jitter(imu_times, imu_rate_hz)
        if not rep.monotonic_imu["monotonic"]:
            r.add(Finding(
                code="TIME_IMU_NON_MONOTONIC",
                severity=Severity.CRITICAL,
                message=f"{rep.monotonic_imu['n_backwards']} backwards IMU timestamps",
                symptom="IMU preintegration resets constantly; the estimator falls back "
                        "to scan matching alone and drifts in exactly the places the "
                        "IMU was supposed to help.",
                fix="Same causes as the LiDAR case. Also check that the driver is not "
                    "reusing one stamp for a whole batch of samples.",
                data=rep.monotonic_imu,
            ))
        if rep.imu_jitter["mean_rate_hz"] < 100.0:
            r.add(Finding(
                code="TIME_IMU_RATE_LOW",
                severity=Severity.ERROR,
                message=f"IMU is publishing at {rep.imu_jitter['mean_rate_hz']:.0f} Hz",
                symptom="Preintegration between scans is built from too few samples, so "
                        "fast rotations are under-integrated and the map lags on turns.",
                fix="LIO-SAM expects 200 Hz or more (it warns below that). Raise the "
                    "driver's output rate; if the IMU cannot do it, do not expect "
                    "LiDAR-inertial performance from it.",
                data=rep.imu_jitter,
            ))
        else:
            r.add(Finding(code="TIME_IMU_RATE_OK", severity=Severity.OK,
                          message=f"IMU at {rep.imu_jitter['mean_rate_hz']:.0f} Hz, jitter "
                                  f"{rep.imu_jitter['jitter_std_ms']:.2f} ms",
                          data=rep.imu_jitter))

    if scan_receive_times is not None:
        rep.receive_mismatch = detect_receive_time_mismatch(
            scan_times, scan_receive_times, lidar_rate_hz)
        rm = rep.receive_mismatch
        if rm["stamped_on_receive"]:
            r.add(Finding(
                code="TIME_STAMPED_ON_RECEIVE",
                severity=Severity.ERROR,
                message=f"header.stamp tracks the receive time exactly (latency spread "
                        f"{rm['latency_std_ms']:.3f} ms) while both series jitter by "
                        f"{rm['sensor_jitter_ms']:.1f} ms",
                symptom="Works with a recorded bag replayed on one machine, fails live "
                        "or on a different machine -- because the transport delay you "
                        "baked into the stamps is different there.",
                fix="Stamp with the sensor clock. This is the single most common reason "
                    "a setup that 'works with the bag' fails on the robot.",
                data=rm,
            ))
        elif rm["negative_latency"]:
            r.add(Finding(
                code="TIME_SENSOR_CLOCK_AHEAD",
                severity=Severity.ERROR,
                message=f"sensor stamps are ahead of receive time by up to "
                        f"{-rm['min_latency_ms']:.1f} ms",
                symptom="TF lookups fail with 'requested time is in the future'; RViz "
                        "shows the cloud flickering in and out.",
                fix="The sensor clock is not disciplined to the host. Enable PTP "
                    "(ptp4l/phc2sys) or feed the sensor a PPS. Do not 'fix' it by "
                    "adding a constant offset unless you also handle drift.",
                data=rm,
            ))
        else:
            r.add(Finding(code="TIME_RECEIVE_LATENCY_OK", severity=Severity.OK,
                          message=f"transport latency {rm['mean_latency_ms']:.1f} +/- "
                                  f"{rm['latency_std_ms']:.1f} ms",
                          data=rm))

    # --- the main event: constant offset between LiDAR and IMU ---
    if scan_rotations is not None and imu_times is not None and gyro is not None:
        try:
            rep.offset = estimate_lidar_imu_offset(
                scan_times, scan_rotations, imu_times, gyro,
                max_offset_s=max_offset_s, step_s=step_s)
        except ValueError as exc:
            r.add(Finding(
                code="TIME_OFFSET_NOT_ESTIMABLE",
                severity=Severity.INFO,
                message=f"could not estimate the offset: {exc}",
                symptom="",
                fix="Record 20-30 s with deliberate yaw reversals -- rotate, stop, "
                    "rotate the other way. Smooth constant motion carries almost no "
                    "information about time offset.",
            ))
        if rep.offset is not None:
            est = rep.offset
            ms = abs(est.offset_ms)
            if not est.trustworthy:
                sev = Severity.WARN
                code = "TIME_OFFSET_UNRELIABLE"
            elif ms >= 20.0:
                sev = Severity.CRITICAL
                code = "TIME_OFFSET_LARGE"
            elif ms >= 5.0:
                sev = Severity.ERROR
                code = "TIME_OFFSET_SIGNIFICANT"
            elif ms >= 1.0:
                sev = Severity.WARN
                code = "TIME_OFFSET_SMALL"
            else:
                sev = Severity.OK
                code = "TIME_OFFSET_OK"
            r.add(Finding(
                code=code,
                severity=sev,
                message=rep.verdict,
                symptom="Stationary the map is perfect. As soon as you rotate, walls "
                        "double and the trajectory kicks; the faster the turn, the "
                        "worse it gets. Error scales as omega * dt."
                        if sev >= Severity.WARN else "",
                fix=(f"Add {est.offset_ms:+.1f} ms to the IMU timestamps, or better, fix "
                     "the source: hardware-sync the two sensors (PTP/PPS), and make the "
                     "driver stamp with the sensor clock rather than with now(). Only "
                     "use a constant software offset when you have also checked for "
                     "drift.") if sev >= Severity.WARN else "",
                data={"offset_ms": est.offset_ms, "correlation": est.correlation,
                      "sharpness": est.sharpness,
                      "peak_width_ms": est.peak_width_ms,
                      "n_samples": est.n_samples},
            ))
        if check_drift and rep.offset is not None:
            t_l, rate_l = rotation_rate_from_poses(scan_times, scan_rotations)
            rate_i = np.linalg.norm(np.asarray(gyro, dtype=float).reshape(-1, 3), axis=1)
            try:
                rep.drift = detect_clock_drift(
                    t_l, rate_l, np.asarray(imu_times, dtype=float), rate_i,
                    max_offset_s=max_offset_s, step_s=step_s)
            except ValueError:
                rep.drift = None
            if rep.drift and rep.drift.get("significant"):
                r.add(Finding(
                    code="TIME_CLOCK_DRIFT",
                    severity=Severity.ERROR,
                    message=f"the offset is not constant: it changes by "
                            f"{rep.drift['drift_ms_per_min']:+.1f} ms/min "
                            f"({rep.drift['drift_ppm']:+.0f} ppm, R^2="
                            f"{rep.drift['r_squared']:.2f})",
                    symptom="The map is good for the first minute and degrades steadily. "
                            "Re-running the same bag from the start looks fine again, "
                            "which makes people blame the loop closure.",
                    fix="A constant software offset cannot fix drift. Discipline the "
                        "sensor clock: PTP over Ethernet for Ouster/Hesai, PPS+NMEA for "
                        "Velodyne, or run the IMU off the same MCU that timestamps the "
                        "LiDAR. Free-running crystals are +/-50 ppm and there is nothing "
                        "to tune.",
                    data=rep.drift,
                ))
    return rep

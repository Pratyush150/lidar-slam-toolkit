"""Trajectory error analysis, with and without ground truth.

With ground truth you get ATE and RPE.  With a customer's bag you get neither,
because there is no ground truth -- and that is the normal case.  So this
module also implements the things you *can* measure from the estimate alone:

* loop-closure residual (the map disagrees with itself at a revisit),
* z-drift rate, split into ramp and step components,
* yaw-drift rate at revisits,
* revisit consistency across the whole run.

The z-drift analysis is the centrepiece because "the map slowly floats
upward" is the single most common LiDAR SLAM complaint, and it has three
completely different causes that need three completely different fixes.  A
number alone ("0.4 m/min") does not tell you which one you have.  The *shape*
of the drift does:

===================  ===========================  ==========================
shape                cause                        fix
===================  ===========================  ==========================
smooth ramp,         gravity / attitude           Fix the extrinsic roll and
proportional to      misalignment: the whole      pitch, or let the system
horizontal distance  map is tilted, so driving    initialise level and
                     "forward" also drives "up"   stationary.
smooth ramp or       accelerometer bias, or an    Calibrate the IMU bias, or
curve, proportional  IMU noise model that lets    raise imuAccBiasN so the
to time              the bias state wander        estimator stops chasing it.
discrete steps       degenerate geometry or a     Fix the geometry problem or
                     loop closure yanking the     the loop-closure gating;
                     graph                        tuning the IMU will not help
===================  ===========================  ==========================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .extrinsics import invert_transform, matrix_to_euler, rotation_angle
from .findings import Finding, Severity

__all__ = [
    "as_poses",
    "umeyama_alignment",
    "TrajectoryError",
    "absolute_trajectory_error",
    "relative_pose_error",
    "loop_closure_residual",
    "find_revisits",
    "revisit_consistency",
    "z_drift_rate",
    "yaw_drift_rate",
    "ZDriftReport",
    "analyze_z_drift",
    "z_drift_findings",
]

_EPS = 1e-12


def as_poses(poses: np.ndarray) -> np.ndarray:
    """Accept ``(N, 4, 4)``, ``(N, 3)`` positions, or ``(N, 7)`` xyz+quat(xyzw)."""
    P = np.asarray(poses, dtype=float)
    if P.ndim == 3 and P.shape[1:] == (4, 4):
        return P
    if P.ndim == 2 and P.shape[1] == 3:
        out = np.tile(np.eye(4), (len(P), 1, 1))
        out[:, :3, 3] = P
        return out
    if P.ndim == 2 and P.shape[1] == 7:
        from .extrinsics import quaternion_to_matrix

        out = np.tile(np.eye(4), (len(P), 1, 1))
        out[:, :3, 3] = P[:, :3]
        for i in range(len(P)):
            out[i, :3, :3] = quaternion_to_matrix(P[i, 3:])
        return out
    raise ValueError(f"cannot interpret pose array of shape {P.shape}")


# --------------------------------------------------------------------------
# Ground-truth comparison
# --------------------------------------------------------------------------
def umeyama_alignment(source: np.ndarray, target: np.ndarray,
                      with_scale: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity transform mapping ``source`` onto ``target``.

    Returns ``(R, t, s)`` such that ``target ~ s * R @ source + t``.

    ATE is normally reported after this alignment because an odometry estimate
    starts at its own arbitrary origin.  **Leave ``with_scale=False`` for
    LiDAR SLAM**: LiDAR is metric, so a scale factor is not a free gauge
    freedom -- if fitting one improves the error, that is itself a finding
    (an extrinsic that is not orthonormal, or a range calibration problem).
    """
    S = np.asarray(source, dtype=float).reshape(-1, 3)
    T = np.asarray(target, dtype=float).reshape(-1, 3)
    if S.shape != T.shape:
        raise ValueError(f"shape mismatch {S.shape} vs {T.shape}")
    n = len(S)
    if n < 3:
        raise ValueError("need at least 3 points to align")
    mu_s, mu_t = S.mean(axis=0), T.mean(axis=0)
    Sc, Tc = S - mu_s, T - mu_t
    cov = Tc.T @ Sc / n
    U, D, Vt = np.linalg.svd(cov)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1.0
    R = U @ W @ Vt
    if with_scale:
        var_s = float(np.mean(np.sum(Sc ** 2, axis=1)))
        s = float(np.trace(np.diag(D) @ W) / var_s) if var_s > _EPS else 1.0
    else:
        s = 1.0
    t = mu_t - s * R @ mu_s
    return R, t, s


@dataclass
class TrajectoryError:
    """Summary statistics of a per-pose error series."""

    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    n: int
    errors: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    @classmethod
    def from_errors(cls, e: np.ndarray) -> "TrajectoryError":
        e = np.asarray(e, dtype=float).reshape(-1)
        if e.size == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, e)
        return cls(
            rmse=float(np.sqrt(np.mean(e ** 2))),
            mean=float(np.mean(e)),
            median=float(np.median(e)),
            std=float(np.std(e)),
            min=float(np.min(e)),
            max=float(np.max(e)),
            n=int(e.size),
            errors=e,
        )

    def to_dict(self) -> Dict[str, float]:
        return {"rmse": self.rmse, "mean": self.mean, "median": self.median,
                "std": self.std, "min": self.min, "max": self.max, "n": self.n}


def absolute_trajectory_error(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    align: bool = True,
    with_scale: bool = False,
) -> Dict[str, object]:
    """Absolute Trajectory Error: global consistency after rigid alignment.

    ATE answers "how far is the whole trajectory from the truth", which makes
    it the right metric for a mapping run and the wrong metric for odometry
    quality -- a single early yaw error inflates ATE for the rest of the run
    even if every subsequent increment is perfect.  Use
    :func:`relative_pose_error` for that.

    Returns a dict with ``translation`` and ``rotation``
    :class:`TrajectoryError` objects, the alignment used, and the per-axis
    RMSE (which is where you see z-specific problems).
    """
    E = as_poses(estimated)
    G = as_poses(ground_truth)
    if len(E) != len(G):
        raise ValueError(f"trajectory lengths differ: {len(E)} vs {len(G)}")
    pe, pg = E[:, :3, 3], G[:, :3, 3]
    if align:
        R, t, s = umeyama_alignment(pe, pg, with_scale=with_scale)
    else:
        R, t, s = np.eye(3), np.zeros(3), 1.0
    aligned = (s * (pe @ R.T)) + t
    diff = aligned - pg
    trans = TrajectoryError.from_errors(np.linalg.norm(diff, axis=1))
    rot_err = np.array([
        rotation_angle((R @ E[i, :3, :3]).T @ G[i, :3, :3]) for i in range(len(E))
    ])
    rot = TrajectoryError.from_errors(rot_err)
    return {
        "translation": trans,
        "rotation": rot,
        "translation_rmse": trans.rmse,
        "rotation_rmse_deg": math.degrees(rot.rmse),
        "per_axis_rmse": np.sqrt(np.mean(diff ** 2, axis=0)),
        "aligned_positions": aligned,
        "alignment": {"R": R, "t": t, "scale": s},
        "aligned": bool(align),
    }


def relative_pose_error(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    delta: int = 1,
) -> Dict[str, object]:
    """Relative Pose Error over a fixed frame gap: local odometry quality.

    For each ``i``, compares the motion ``T_i -> T_{i+delta}`` in the estimate
    against the same motion in ground truth.  Independent of any global
    alignment, so it is the metric that tells you whether the *odometry* is
    good, separately from whether the map is globally consistent.

    Returns ``translation`` and ``rotation`` :class:`TrajectoryError` objects,
    plus ``drift_percent`` (translation RMSE as a fraction of the ground-truth
    distance covered per interval) which is the number people actually quote.
    """
    E = as_poses(estimated)
    G = as_poses(ground_truth)
    if len(E) != len(G):
        raise ValueError(f"trajectory lengths differ: {len(E)} vs {len(G)}")
    if delta < 1 or delta >= len(E):
        raise ValueError(f"delta must be in [1, {len(E) - 1}]")
    t_err: List[float] = []
    r_err: List[float] = []
    gt_dist: List[float] = []
    for i in range(len(E) - delta):
        de = invert_transform(E[i]) @ E[i + delta]
        dg = invert_transform(G[i]) @ G[i + delta]
        err = invert_transform(dg) @ de
        t_err.append(float(np.linalg.norm(err[:3, 3])))
        r_err.append(rotation_angle(err[:3, :3]))
        gt_dist.append(float(np.linalg.norm(dg[:3, 3])))
    trans = TrajectoryError.from_errors(np.asarray(t_err))
    rot = TrajectoryError.from_errors(np.asarray(r_err))
    mean_dist = float(np.mean(gt_dist)) if gt_dist else 0.0
    return {
        "translation": trans,
        "rotation": rot,
        "translation_rmse": trans.rmse,
        "rotation_rmse_deg": math.degrees(rot.rmse),
        "delta": delta,
        "mean_interval_distance_m": mean_dist,
        "drift_percent": 100.0 * trans.rmse / mean_dist if mean_dist > _EPS else float("nan"),
    }


# --------------------------------------------------------------------------
# No ground truth
# --------------------------------------------------------------------------
def loop_closure_residual(poses: np.ndarray,
                          pairs: Sequence[Tuple[int, int]]) -> Dict[str, object]:
    """How badly the trajectory disagrees with itself at known revisits.

    ``pairs`` are ``(i, j)`` index pairs the operator knows to be the same
    physical place (start/end of a loop, a marked waypoint).  The residual is
    the pose difference the estimator failed to close.  This is the closest
    thing to ground truth you get from a customer bag.
    """
    P = as_poses(poses)
    trans: List[float] = []
    rot: List[float] = []
    dz: List[float] = []
    detail: List[Dict[str, float]] = []
    for i, j in pairs:
        d = invert_transform(P[i]) @ P[j]
        tn = float(np.linalg.norm(d[:3, 3]))
        rn = rotation_angle(d[:3, :3])
        trans.append(tn)
        rot.append(rn)
        dz.append(float(P[j][2, 3] - P[i][2, 3]))
        detail.append({"i": int(i), "j": int(j), "translation_m": tn,
                       "rotation_deg": math.degrees(rn), "dz_m": dz[-1]})
    if not trans:
        return {"n_pairs": 0}
    return {
        "n_pairs": len(trans),
        "translation_rmse_m": float(np.sqrt(np.mean(np.square(trans)))),
        "translation_max_m": float(np.max(trans)),
        "rotation_rmse_deg": float(math.degrees(np.sqrt(np.mean(np.square(rot))))),
        "rotation_max_deg": float(math.degrees(np.max(rot))),
        "z_residual_rmse_m": float(np.sqrt(np.mean(np.square(dz)))),
        "z_residual_max_m": float(np.max(np.abs(dz))),
        "pairs": detail,
    }


def find_revisits(positions: np.ndarray, times: np.ndarray,
                  radius: float = 2.0, min_time_gap_s: float = 20.0,
                  max_pairs: int = 200) -> List[Tuple[int, int]]:
    """Candidate revisit pairs: close in space, far apart in time.

    Detected from the *estimated* trajectory, so it is biased -- if the
    estimate has drifted more than ``radius`` the true revisit will be missed.
    That bias is one-sided and useful: a run whose loop is obviously closed on
    the map but produces no revisit candidates here has already drifted past
    ``radius``, which is itself the answer.
    """
    P = np.asarray(positions, dtype=float).reshape(-1, 3)
    t = np.asarray(times, dtype=float).reshape(-1)
    pairs: List[Tuple[int, int]] = []
    n = len(P)
    if n < 2:
        return pairs
    d2 = radius * radius
    for i in range(n):
        # Only look forward, and only past the time gap.
        j0 = int(np.searchsorted(t, t[i] + min_time_gap_s))
        if j0 >= n:
            break
        seg = P[j0:] - P[i]
        hits = np.where(np.sum(seg ** 2, axis=1) <= d2)[0]
        if hits.size:
            pairs.append((i, int(j0 + hits[int(np.argmin(np.sum(seg[hits] ** 2, axis=1)))])))
        if len(pairs) >= max_pairs:
            break
    # Thin out near-duplicate pairs from consecutive poses.
    thinned: List[Tuple[int, int]] = []
    for p in pairs:
        if not thinned or p[0] - thinned[-1][0] > 5:
            thinned.append(p)
    return thinned


def revisit_consistency(poses: np.ndarray, times: np.ndarray,
                        radius: float = 2.0, min_time_gap_s: float = 20.0
                        ) -> Dict[str, object]:
    """Find revisits automatically and measure the disagreement at each."""
    P = as_poses(poses)
    pairs = find_revisits(P[:, :3, 3], times, radius, min_time_gap_s)
    out = loop_closure_residual(P, pairs)
    out["radius_m"] = radius
    out["min_time_gap_s"] = min_time_gap_s
    return out


def z_drift_rate(times: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    """Least-squares slope of z against time, reported in m/min."""
    t = np.asarray(times, dtype=float).reshape(-1)
    zz = np.asarray(z, dtype=float).reshape(-1)
    if len(t) != len(zz) or len(t) < 3:
        raise ValueError("need at least 3 matching samples")
    A = np.vstack([t - t[0], np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, zz, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((zz - pred) ** 2))
    ss_tot = float(np.sum((zz - zz.mean()) ** 2))
    return {
        "slope_m_per_s": float(coef[0]),
        "rate_m_per_min": float(coef[0]) * 60.0,
        "intercept_m": float(coef[1]),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > _EPS else 0.0,
        "total_change_m": float(zz[-1] - zz[0]),
        "duration_s": float(t[-1] - t[0]),
    }


def yaw_drift_rate(times: np.ndarray, rotations: np.ndarray,
                   revisit_pairs: Optional[Sequence[Tuple[int, int]]] = None
                   ) -> Dict[str, float]:
    """Yaw drift, in deg/min.

    With ``revisit_pairs``, the heading difference at each revisit is divided
    by the elapsed time -- this is the meaningful measurement, because the
    robot is by construction back at the same heading.

    Without pairs, a line is fitted to the unwrapped yaw.  **That is only
    meaningful for a stationary sensor or a closed loop**: on a run that
    genuinely turns, the fitted slope is the mission, not the drift.  The
    returned dict says which method was used.
    """
    R = as_poses(rotations)[:, :3, :3]
    t = np.asarray(times, dtype=float).reshape(-1)
    yaw = np.unwrap(np.array([matrix_to_euler(r)[2] for r in R]))
    if revisit_pairs:
        rates = []
        for i, j in revisit_pairs:
            dt = float(t[j] - t[i])
            if dt <= 0:
                continue
            dy = float(np.arctan2(np.sin(yaw[j] - yaw[i]), np.cos(yaw[j] - yaw[i])))
            rates.append(math.degrees(dy) / dt * 60.0)
        if rates:
            return {"method": "revisit", "n": len(rates),
                    "rate_deg_per_min": float(np.mean(rates)),
                    "max_abs_deg_per_min": float(np.max(np.abs(rates)))}
    A = np.vstack([t - t[0], np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, yaw, rcond=None)
    return {"method": "linear_fit", "n": len(t),
            "rate_deg_per_min": math.degrees(float(coef[0])) * 60.0,
            "max_abs_deg_per_min": abs(math.degrees(float(coef[0])) * 60.0),
            "caveat": "linear fit over the whole run; only meaningful if the platform "
                      "ends at the heading it started with"}


# --------------------------------------------------------------------------
# The z-drift analyser
# --------------------------------------------------------------------------
@dataclass
class ZDriftReport:
    """Decomposition of a z trace into ramp + steps, with a cause hypothesis."""

    ramp_m_per_min: float
    ramp_r_squared: float
    n_steps: int
    step_indices: List[int]
    step_sizes_m: List[float]
    total_step_m: float
    largest_step_m: float
    total_change_m: float
    ramp_share: float
    """Fraction of the total z change explained by the smooth ramp (0-1)."""
    step_share: float
    """Fraction explained by discrete jumps (0-1)."""
    likely_cause: str
    confidence: str
    evidence: List[str] = field(default_factory=list)
    distance_correlation: Optional[float] = None
    """Slope of z against horizontal distance (m/m), when distance was given."""
    curvature: Optional[float] = None
    """Quadratic coefficient of z against time (m/s^2), when the fit is used."""

    def to_dict(self) -> Dict[str, object]:
        return {
            "ramp_m_per_min": self.ramp_m_per_min,
            "ramp_r_squared": self.ramp_r_squared,
            "n_steps": self.n_steps,
            "step_indices": self.step_indices,
            "step_sizes_m": self.step_sizes_m,
            "total_step_m": self.total_step_m,
            "largest_step_m": self.largest_step_m,
            "total_change_m": self.total_change_m,
            "ramp_share": self.ramp_share,
            "step_share": self.step_share,
            "likely_cause": self.likely_cause,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "distance_correlation": self.distance_correlation,
            "curvature": self.curvature,
        }


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation, scaled to be a standard-deviation estimate."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(x - float(np.median(x)))))


def analyze_z_drift(
    times: np.ndarray,
    z: np.ndarray,
    horizontal_distance: Optional[np.ndarray] = None,
    step_sigma: float = 6.0,
    min_step_m: float = 0.05,
    degeneracy_fraction: Optional[float] = None,
) -> ZDriftReport:
    """Separate ramp drift from step jumps in a z trace and name the cause.

    Parameters
    ----------
    times, z:
        The trajectory's timestamps and vertical coordinate.
    horizontal_distance:
        Optional cumulative horizontal path length at each pose.  Supplying it
        is what lets the analyser separate *attitude* error (z grows with
        distance travelled) from *bias* error (z grows with time, and curves).
        Compute it as ``np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy,
        axis=0), axis=1))])``.
    step_sigma:
        A z increment is called a step when it exceeds this many robust
        standard deviations of the increment distribution.
    min_step_m:
        Absolute floor so that a very smooth trace does not produce
        "steps" out of millimetre noise.
    degeneracy_fraction:
        Optional: fraction of the run spent in geometrically degenerate
        surroundings (from :mod:`slamkit.degeneracy`).  Used only to raise
        confidence in the geometry hypothesis.

    Returns
    -------
    :class:`ZDriftReport`
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    zz = np.asarray(z, dtype=float).reshape(-1)
    if len(t) != len(zz):
        raise ValueError("times and z must be the same length")
    if len(t) < 5:
        raise ValueError("need at least 5 samples")

    dz = np.diff(zz)
    sigma = _mad(dz)
    thresh = max(step_sigma * sigma, min_step_m)
    step_mask = np.abs(dz) > thresh
    step_idx = (np.where(step_mask)[0] + 1).tolist()
    step_sizes = dz[step_mask].tolist()
    total_step = float(np.sum(np.abs(step_sizes)))
    largest_step = float(np.max(np.abs(step_sizes))) if step_sizes else 0.0

    # Rebuild a step-free trace by zeroing the jump increments, then fit the
    # ramp to what is left. Doing it the other way round (fit first, call the
    # residual steps) lets one big jump tilt the ramp estimate.
    dz_smooth = np.where(step_mask, 0.0, dz)
    z_smooth = np.concatenate([[zz[0]], zz[0] + np.cumsum(dz_smooth)])
    ramp = z_drift_rate(t, z_smooth)
    ramp_change = abs(ramp["slope_m_per_s"]) * float(t[-1] - t[0])
    total_change = float(zz[-1] - zz[0])
    denom = ramp_change + total_step
    ramp_share = float(ramp_change / denom) if denom > _EPS else 0.0
    step_share = float(total_step / denom) if denom > _EPS else 0.0

    evidence: List[str] = []
    # Curvature: fit a quadratic to the step-free trace. An uncorrected
    # accelerometer bias integrates twice, so it curves; an attitude error
    # is linear in distance.
    A = np.vstack([(t - t[0]) ** 2, (t - t[0]), np.ones_like(t)]).T
    qcoef, *_ = np.linalg.lstsq(A, z_smooth, rcond=None)
    curvature = float(qcoef[0])
    quad_pred = A @ qcoef
    ss_tot = float(np.sum((z_smooth - z_smooth.mean()) ** 2))
    quad_r2 = 1.0 - float(np.sum((z_smooth - quad_pred) ** 2)) / ss_tot if ss_tot > _EPS else 0.0

    dist_slope: Optional[float] = None
    dist_r2 = 0.0
    if horizontal_distance is not None:
        d = np.asarray(horizontal_distance, dtype=float).reshape(-1)
        if len(d) == len(t) and float(d[-1] - d[0]) > 1e-6:
            Ad = np.vstack([d - d[0], np.ones_like(d)]).T
            dcoef, *_ = np.linalg.lstsq(Ad, z_smooth, rcond=None)
            dist_slope = float(dcoef[0])
            pred = Ad @ dcoef
            dist_r2 = 1.0 - float(np.sum((z_smooth - pred) ** 2)) / ss_tot if ss_tot > _EPS else 0.0

    # --- hypothesis selection -------------------------------------------
    if denom < 1e-6:
        cause = "no measurable z drift"
        confidence = "high"
        evidence.append(f"total z change {total_change:+.3f} m over "
                        f"{float(t[-1] - t[0]):.0f} s")
    elif step_share > 0.6:
        if degeneracy_fraction is not None and degeneracy_fraction > 0.3:
            cause = "degenerate geometry (the estimator loses vertical constraint, " \
                    "then snaps back when structure reappears)"
            confidence = "high"
            evidence.append(f"{degeneracy_fraction * 100:.0f}% of the run is "
                            "geometrically degenerate")
        else:
            cause = "discrete pose-graph corrections: loop closure or a plane/ground " \
                    "constraint snapping the trajectory"
            confidence = "medium"
        evidence.append(f"{len(step_sizes)} step(s) totalling {total_step:.3f} m "
                        f"account for {step_share * 100:.0f}% of the z change; "
                        f"largest single jump {largest_step:.3f} m")
        evidence.append(f"residual smooth ramp is only "
                        f"{ramp['rate_m_per_min']:+.3f} m/min")
    elif ramp_share > 0.6:
        evidence.append(f"smooth ramp of {ramp['rate_m_per_min']:+.3f} m/min accounts "
                        f"for {ramp_share * 100:.0f}% of the z change "
                        f"(linear fit R^2 {ramp['r_squared']:.3f})")
        if step_sizes:
            evidence.append(f"plus {len(step_sizes)} small step(s) totalling "
                            f"{total_step:.3f} m")
        if dist_slope is not None and dist_r2 > max(ramp["r_squared"], 0.5) - 0.02:
            tilt_deg = math.degrees(math.atan(dist_slope))
            cause = ("attitude / gravity misalignment: the map is tilted, so horizontal "
                     "travel is partly interpreted as vertical travel")
            confidence = "high" if dist_r2 > 0.9 else "medium"
            evidence.append(
                f"z tracks horizontal distance at {dist_slope * 100:+.2f} cm per metre "
                f"(R^2 {dist_r2:.3f}), i.e. an effective map tilt of {tilt_deg:+.2f} deg"
            )
        elif quad_r2 > ramp["r_squared"] + 0.05 and abs(curvature) > 1e-5:
            cause = ("accelerometer bias / gravity magnitude error: the vertical error "
                     "is growing faster than linearly, which is the signature of a bias "
                     "being integrated twice")
            confidence = "medium"
            evidence.append(f"quadratic fit (R^2 {quad_r2:.3f}) beats linear "
                            f"(R^2 {ramp['r_squared']:.3f}); curvature "
                            f"{curvature:.2e} m/s^2")
        else:
            cause = ("constant-rate vertical drift: IMU accelerometer bias or an "
                     "imuGravity value that does not match your location")
            confidence = "medium"
            evidence.append("the ramp is linear in time and no horizontal-distance "
                            "correlation was supplied to separate attitude error from "
                            "bias -- pass horizontal_distance to disambiguate")
    else:
        cause = "mixed: both a smooth ramp and discrete jumps are present"
        confidence = "low"
        evidence.append(f"ramp {ramp['rate_m_per_min']:+.3f} m/min "
                        f"({ramp_share * 100:.0f}%) and {len(step_sizes)} step(s) "
                        f"totalling {total_step:.3f} m ({step_share * 100:.0f}%)")

    return ZDriftReport(
        ramp_m_per_min=float(ramp["rate_m_per_min"]),
        ramp_r_squared=float(ramp["r_squared"]),
        n_steps=len(step_sizes),
        step_indices=step_idx,
        step_sizes_m=[float(s) for s in step_sizes],
        total_step_m=total_step,
        largest_step_m=largest_step,
        total_change_m=total_change,
        ramp_share=ramp_share,
        step_share=step_share,
        likely_cause=cause,
        confidence=confidence,
        evidence=evidence,
        distance_correlation=dist_slope,
        curvature=curvature,
    )


def z_drift_findings(report: ZDriftReport,
                     warn_m_per_min: float = 0.1,
                     error_m_per_min: float = 0.5) -> List[Finding]:
    """Turn a :class:`ZDriftReport` into ranked findings with concrete fixes."""
    out: List[Finding] = []
    rate = abs(report.ramp_m_per_min)
    fixes = {
        "attitude": (
            "Fix the attitude, not the z axis. In order: (1) verify the IMU-to-LiDAR "
            "extrinsic roll and pitch -- a 1 deg error is 1.7 cm of z per metre "
            "travelled; (2) let the system initialise stationary and level so gravity "
            "is observed cleanly (LIO-SAM uses the first IMU samples for this); "
            "(3) check imuRPYWeight -- if it is 0 the scan matcher is free to tilt the "
            "map and nothing pulls it back."
        ),
        "bias": (
            "Calibrate or re-estimate the IMU bias: leave the robot still for 10-20 s "
            "at startup. If the bias genuinely wanders, raise imuAccBiasN/imuGyrBiasN "
            "so the estimator is allowed to track it. Check imuGravity matches your "
            "latitude and altitude (9.79-9.83); a 0.02 m/s^2 mismatch is 0.6 m of z "
            "after 8 s of open-loop integration."
        ),
        "steps": (
            "Steps are not drift -- do not fix them with IMU tuning. Find which "
            "correction produced each jump: loop closure (tighten "
            "historyKeyframeFitnessScore, raise historyKeyframeSearchRadius only if "
            "your odometry is good enough to justify it) or a ground-plane constraint. "
            "In LIO-SAM, z_tollerance clamps vertical motion and will produce exactly "
            "this if it is set too tight for your platform."
        ),
        "geometry": (
            "The environment is the problem. See slamkit.degeneracy: in a corridor or "
            "an open field the vertical channel can lose its constraint and the "
            "estimator free-runs on the IMU until structure reappears. Add an "
            "independent height source (barometer, wheel odometry, GNSS) or accept and "
            "bound the drift with loop closure."
        ),
    }
    key = "bias"
    cl = report.likely_cause.lower()
    if "attitude" in cl or "gravity misalignment" in cl:
        key = "attitude"
    elif "degenerate" in cl:
        key = "geometry"
    elif "step" in cl or "loop closure" in cl or "pose-graph" in cl:
        key = "steps"
    if rate >= error_m_per_min or report.largest_step_m >= 0.5:
        sev = Severity.ERROR
    elif rate >= warn_m_per_min or report.largest_step_m >= 0.1:
        sev = Severity.WARN
    else:
        sev = Severity.OK
    out.append(Finding(
        code="ZDRIFT_" + key.upper(),
        severity=sev,
        message=(f"z drift: ramp {report.ramp_m_per_min:+.3f} m/min "
                 f"({report.ramp_share * 100:.0f}% of the change), "
                 f"{report.n_steps} step(s) totalling {report.total_step_m:.3f} m "
                 f"({report.step_share * 100:.0f}%). Likely cause: "
                 f"{report.likely_cause} [confidence: {report.confidence}]"),
        symptom="The map floats upward (or sinks) while walls stay crisp; on a second "
                "pass the floor of the first pass is visibly above or below the second.",
        fix=fixes[key] if sev >= Severity.WARN else "",
        data=report.to_dict(),
    ))
    return out

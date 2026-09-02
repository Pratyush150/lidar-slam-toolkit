"""Geometric degeneracy detection: the honest answer to "why does it slide?".

The question
------------
"My SLAM works in the lab and slides down the corridor."  The algorithm is
fine.  The corridor is the problem.

Point-to-plane scan matching solves a linear system whose translation block is

    H_t = sum_i n_i n_i^T

over the surface normals ``n_i`` of the matched points.  That matrix is the
*information* the geometry gives you about translation.  In a straight
corridor every normal points at a wall (+/-Y), the floor (+/-Z) or the ceiling
(+/-Z).  Nothing points along the corridor (X).  So ``H_t`` has a near-zero
eigenvalue along X, the solver is free to slide, and it will -- by exactly as
much as the IMU or the odometry prior lets it.

The rotational block is

    H_r = sum_i (p_i x n_i)(p_i x n_i)^T

which is why a featureless *tunnel* is worse than a corridor: the circular
cross-section also removes the constraint on roll about the tunnel axis.

What this module does
---------------------
Compute both blocks from a cloud, eigen-decompose them, and report a
normalised observability score per world axis plus the weakest direction.  The
scores are relative (best axis = 1.0), because the absolute magnitude just
scales with how many points you fed in.

What it does not do
-------------------
It does not tell you the metric drift you will suffer.  That depends on your
IMU, your update rate and how long you stay in the degenerate stretch.  It
tells you *which* degree of freedom is unconstrained, which is the part people
get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cloud import estimate_normals
from .findings import Finding, Severity

__all__ = [
    "DegeneracyReport",
    "translation_information",
    "rotation_information",
    "analyze_degeneracy",
    "degeneracy_findings",
]

_EPS = 1e-12
_AXES = ("x", "y", "z")


@dataclass
class DegeneracyReport:
    """Result of :func:`analyze_degeneracy`."""

    translation_scores: Dict[str, float]
    """Per-world-axis translation observability in ``[0, 1]``, 1 = best axis."""

    rotation_scores: Dict[str, float]
    """Per-world-axis rotation observability (about X, Y, Z) in ``[0, 1]``."""

    translation_eigenvalues: np.ndarray = field(repr=False,
                                                default_factory=lambda: np.zeros(3))
    translation_eigenvectors: np.ndarray = field(repr=False,
                                                 default_factory=lambda: np.eye(3))
    rotation_eigenvalues: np.ndarray = field(repr=False,
                                             default_factory=lambda: np.zeros(3))
    condition_number: float = 1.0
    """``lambda_max / lambda_min`` of the translation block. > ~100 is trouble."""

    weakest_direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Unit vector along the least observable translation direction."""

    weakest_axis: str = "x"
    """Whichever of x/y/z is closest to :attr:`weakest_direction`."""

    environment: str = "unknown"
    """One of ``well_constrained``, ``corridor``, ``open_field``, ``tunnel``,
    ``plane_only``, ``sparse``."""

    n_points: int = 0
    degenerate: bool = False
    explanation: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "environment": self.environment,
            "degenerate": self.degenerate,
            "weakest_axis": self.weakest_axis,
            "weakest_direction": [round(float(v), 4) for v in self.weakest_direction],
            "condition_number": float(self.condition_number),
            "translation_scores": {k: float(v) for k, v in self.translation_scores.items()},
            "rotation_scores": {k: float(v) for k, v in self.rotation_scores.items()},
            "n_points": int(self.n_points),
            "explanation": self.explanation,
        }


def translation_information(normals: np.ndarray,
                            weights: Optional[np.ndarray] = None) -> np.ndarray:
    """``sum_i w_i n_i n_i^T`` -- the point-to-plane translation information matrix."""
    N = np.asarray(normals, dtype=float).reshape(-1, 3)
    n = np.linalg.norm(N, axis=1, keepdims=True)
    N = N / np.maximum(n, _EPS)
    if weights is None:
        return N.T @ N / max(len(N), 1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    return (N * w[:, None]).T @ N / max(float(w.sum()), _EPS)


def rotation_information(points: np.ndarray, normals: np.ndarray,
                         weights: Optional[np.ndarray] = None) -> np.ndarray:
    """``sum_i w_i (p_i x n_i)(p_i x n_i)^T`` -- the rotation information matrix.

    ``points`` must be expressed relative to the rotation centre (the sensor
    origin).  Scaled by the mean squared range so the eigenvalues stay
    comparable between a 5 m room and a 50 m yard.
    """
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    N = np.asarray(normals, dtype=float).reshape(-1, 3)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), _EPS)
    C = np.cross(P, N)
    scale = float(np.mean(np.sum(P ** 2, axis=1)))
    scale = scale if scale > _EPS else 1.0
    if weights is None:
        return C.T @ C / (max(len(C), 1) * scale)
    w = np.asarray(weights, dtype=float).reshape(-1)
    return (C * w[:, None]).T @ C / (max(float(w.sum()), _EPS) * scale)


def _axis_scores(M: np.ndarray) -> Dict[str, float]:
    """Diagonal of ``M`` normalised so the strongest axis is 1.0."""
    d = np.abs(np.diag(M))
    m = float(d.max())
    if m <= _EPS:
        return {a: 0.0 for a in _AXES}
    return {a: float(v / m) for a, v in zip(_AXES, d)}


def analyze_degeneracy(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    k: int = 12,
    degenerate_score: float = 0.10,
    degenerate_condition: float = 25.0,
    voxel_size: Optional[float] = None,
    viewpoint: Sequence[float] = (0.0, 0.0, 0.0),
) -> DegeneracyReport:
    """Score how well a cloud constrains each degree of freedom.

    Parameters
    ----------
    points:
        ``(N, 3)`` in the sensor frame (the sensor at the origin), so that the
        rotation block is computed about the right centre.
    normals:
        Precomputed normals.  If omitted they are estimated with local PCA.
    k:
        Neighbourhood size for normal estimation.
    degenerate_score:
        An axis is called degenerate below this normalised score.  0.10 means
        "this axis gets under a tenth of the constraint the best axis gets".
    degenerate_condition:
        Also flag degeneracy when the translation block's condition number
        exceeds this, which catches the case where the weak direction is not
        aligned with a world axis (a diagonal corridor).
    voxel_size:
        Optional pre-downsample.  **Use it.**  Raw spinning-LiDAR clouds are
        wildly non-uniform in density -- the ground right under the sensor can
        carry a third of the points -- and that density bias alone will tell
        you the vertical axis is the best constrained one in every scene.
        Voxelising equalises the vote.

    Returns
    -------
    :class:`DegeneracyReport`
    """
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    if voxel_size is not None and len(P) > 0:
        from .cloud import voxel_downsample

        if normals is not None:
            raise ValueError("pass either voxel_size or precomputed normals, not both")
        P = voxel_downsample(P, voxel_size)
    if len(P) < 10:
        return DegeneracyReport(
            translation_scores={a: 0.0 for a in _AXES},
            rotation_scores={a: 0.0 for a in _AXES},
            n_points=len(P),
            environment="sparse",
            degenerate=True,
            explanation=f"only {len(P)} points; nothing can be concluded and nothing "
                        "can be matched either",
        )
    if normals is None:
        normals = estimate_normals(P, k=k, viewpoint=viewpoint)
    Ht = translation_information(normals)
    Hr = rotation_information(P, normals)
    t_evals, t_evecs = np.linalg.eigh(Ht)
    r_evals = np.linalg.eigvalsh(Hr)
    t_evals = np.abs(t_evals)
    r_evals = np.abs(r_evals)
    lam_max = float(t_evals.max())
    lam_min = float(t_evals.min())
    cond = lam_max / lam_min if lam_min > _EPS else float("inf")
    weak_dir = t_evecs[:, 0] / max(float(np.linalg.norm(t_evecs[:, 0])), _EPS)
    if weak_dir[int(np.argmax(np.abs(weak_dir)))] < 0:
        weak_dir = -weak_dir
    weak_axis = _AXES[int(np.argmax(np.abs(weak_dir)))]

    t_scores = _axis_scores(Ht)
    r_scores = _axis_scores(Hr)
    # Work in the eigenbasis, not the world axes: a corridor that runs
    # diagonally across the map is just as degenerate, and a per-axis
    # threshold would miss it.
    norm_evals = t_evals / lam_max if lam_max > _EPS else np.zeros(3)
    r0, r1 = float(norm_evals[0]), float(norm_evals[1])
    # Rotational observability about the weakest translation direction.
    rot_about_weak = float(weak_dir @ Hr @ weak_dir)
    rot_max = float(np.max(np.abs(np.diag(Hr))))
    rot_about_weak_norm = rot_about_weak / rot_max if rot_max > _EPS else 0.0
    degenerate = bool(r0 < degenerate_score or cond > degenerate_condition)

    env, why = _classify(
        r0=r0, r1=r1, cond=cond, thr=degenerate_score,
        cond_thr=degenerate_condition,
        weak_dir=weak_dir, weak_axis=weak_axis,
        strong_dir=t_evecs[:, 2],
        rot_about_weak=rot_about_weak_norm,
    )
    return DegeneracyReport(
        translation_scores=t_scores,
        rotation_scores=r_scores,
        translation_eigenvalues=t_evals,
        translation_eigenvectors=t_evecs,
        rotation_eigenvalues=r_evals,
        condition_number=cond,
        weakest_direction=weak_dir,
        weakest_axis=weak_axis,
        environment=env,
        n_points=int(len(P)),
        degenerate=degenerate,
        explanation=why,
    )


def _classify(r0, r1, cond, thr, cond_thr, weak_dir, weak_axis, strong_dir,
              rot_about_weak):
    """Turn the eigen-spectrum into an environment label and a sentence.

    ``r0 <= r1 <= 1`` are the translation-information eigenvalues normalised by
    the largest.  ``rot_about_weak`` is the rotational information about the
    weakest translation direction, normalised the same way.
    """
    wa = weak_axis.upper()
    if r0 >= thr and cond <= cond_thr:
        return (
            "well_constrained",
            f"Surface normals span all three axes (weakest/strongest information "
            f"ratio {r0:.2f}, condition number {cond:.1f}). Translation and rotation "
            "are both observable from geometry alone.",
        )
    if r0 < thr <= r1:
        # Exactly one unconstrained translation direction.
        if r1 > 0.5 and rot_about_weak < 0.10:
            return (
                "tunnel",
                f"One unconstrained translation direction ({np.round(weak_dir, 3).tolist()}, "
                f"nearest {wa}) and the cross-section perpendicular to it is close to "
                f"isotropic (remaining eigenvalue ratio {r1:.2f}). Rotation about that "
                f"same direction is also unobserved ({rot_about_weak:.3f}). The geometry "
                "is invariant under both sliding and rolling along the axis: the worst "
                "case for a LiDAR-only front end.",
            )
        return (
            "corridor",
            f"Exactly one unconstrained translation direction "
            f"({np.round(weak_dir, 3).tolist()}, nearest {wa}) at {r0:.3f} of the "
            f"best-observed direction; the perpendicular directions are constrained "
            f"({r1:.2f} and 1.00). Parallel walls and a floor pin two axes, nothing "
            "pins motion along the corridor. Scan matching will slide, and how far "
            "depends entirely on your motion prior.",
        )
    # Two or more unconstrained translation directions.
    if abs(float(strong_dir[2])) > 0.8:
        return (
            "open_field",
            f"Normals collapse onto the vertical ({np.round(strong_dir, 3).tolist()}): "
            "only the ground plane is visible. Z, roll and pitch are observed, both "
            "horizontal axes and yaw are free. Position wanders laterally while the "
            "map still looks flat and plausible -- which is why this gets reported as "
            "'the map is fine but the pose is wrong'.",
        )
    return (
        "plane_only",
        f"Normals collapse onto a single direction "
        f"({np.round(strong_dir, 3).tolist()}); every surface in view is parallel. "
        f"Two translation directions are free (eigenvalue ratios {r0:.3f}, {r1:.3f}) "
        "and at least one rotation with them.",
    )


def degeneracy_findings(report: DegeneracyReport) -> List[Finding]:
    """Convert a :class:`DegeneracyReport` into ranked :class:`Finding` objects."""
    out: List[Finding] = []
    if report.environment == "sparse":
        out.append(Finding(
            code="DEGENERACY_TOO_FEW_POINTS",
            severity=Severity.ERROR,
            message=report.explanation,
            symptom="Scan matching diverges or the node reports 'not enough features'.",
            fix="Check lidarMinRange/lidarMaxRange, the ring field, and whether a "
                "downstream filter is deleting the cloud.",
            data=report.to_dict(),
        ))
        return out
    if not report.degenerate:
        out.append(Finding(
            code="DEGENERACY_NONE",
            severity=Severity.OK,
            message=f"geometry is well constrained (condition number "
                    f"{report.condition_number:.1f}, weakest axis "
                    f"{report.weakest_axis.upper()} at "
                    f"{report.translation_scores[report.weakest_axis]:.2f})",
            data=report.to_dict(),
        ))
        return out
    weak = report.weakest_axis.upper()
    fixes = {
        "corridor": (
            "Do not fix this by tuning the scan matcher. Options, in order of how "
            "well they work: (1) feed wheel odometry or a velocity prior so the "
            "unconstrained axis has a source of information -- in LIO-SAM this means "
            "trusting IMU preintegration more, in Cartographer it means raising "
            "TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight; "
            "(2) detect degeneracy online and project the update out of the weak "
            "direction (the Zhang & Singh solution-remapping approach); "
            "(3) add reflectors or make sure doorways and wall breaks stay in view "
            "by widening lidarMaxRange."
        ),
        "tunnel": (
            "A LiDAR-only front end cannot solve this. You need an odometry source "
            "with an independent scale: wheel encoders, a radar, or a camera with "
            "enough texture. Increase IMU trust and accept bounded drift over the "
            "tunnel length, then rely on loop closure at the exit."
        ),
        "open_field": (
            "Ground-only geometry constrains z, roll and pitch, nothing else. Use "
            "GNSS if you have it (LIO-SAM's gpsTopic exists for exactly this), or "
            "keep structure in view by planning the route along a treeline or fence."
        ),
        "plane_only": (
            "Point the sensor so that at least two non-parallel surfaces are in view. "
            "If the sensor is horizontal and the scene is flat, tilting it 15-20 deg "
            "recovers vertical constraint at almost no cost."
        ),
        "ill_conditioned": (
            "The weak direction is diagonal to the world axes. Check the reported "
            "weakest_direction vector: it points along the corridor or aisle. Same "
            "remedies as the corridor case."
        ),
    }
    out.append(Finding(
        code="DEGENERACY_" + report.environment.upper(),
        severity=Severity.WARN if report.environment != "tunnel" else Severity.ERROR,
        message=f"{report.environment.replace('_', ' ')}: translation observability "
                f"x={report.translation_scores['x']:.3f} "
                f"y={report.translation_scores['y']:.3f} "
                f"z={report.translation_scores['z']:.3f}, "
                f"condition number {report.condition_number:.1f}, weakest direction "
                f"{np.round(report.weakest_direction, 3).tolist()} (nearest {weak})",
        symptom=f"The map slides along {weak}. Walls stay crisp, the robot's reported "
                "position creeps, and when you re-enter structured space the map snaps "
                "back with a step -- which then gets reported as 'loop closure jump'.",
        fix=fixes.get(report.environment, ""),
        data=report.to_dict(),
    ))
    weak_rot = [a for a, s in report.rotation_scores.items() if s < 0.10]
    if weak_rot:
        out.append(Finding(
            code="DEGENERACY_ROTATION",
            severity=Severity.WARN,
            message=f"rotation about {'/'.join(a.upper() for a in weak_rot)} is weakly "
                    f"observed (scores "
                    f"{ {a: round(report.rotation_scores[a], 3) for a in weak_rot} })",
            symptom="Heading or roll drifts with no visible cause; the map tilts "
                    "gradually over a long straight run.",
            fix="Rely on the IMU for the unobserved rotation: check that gravity "
                "alignment is good (imuRPYWeight in LIO-SAM) before blaming the "
                "scan matcher.",
            data={"rotation_scores": report.rotation_scores},
        ))
    return out

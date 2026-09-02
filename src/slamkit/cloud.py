"""Pure-numpy point-cloud utilities used by the rest of the toolkit.

Nothing here needs PCL, Open3D or ROS.  That is deliberate: when a customer
sends you a bag and says "the map is smeared", you want to be able to answer
from a laptop, offline, in five minutes.

Everything operates on plain ``(N, 3)`` float arrays.  The per-point ``ring``
and ``time`` fields that ship inside a ``sensor_msgs/PointCloud2`` are handled
separately (see :func:`detect_timestamp_format` and :func:`assign_rings`),
because those two fields are where most sensor-specific breakage lives:

* Velodyne publishes ``time`` as float32 **seconds relative to the start of
  the scan**, and it is *negative* in some drivers.
* Ouster publishes ``t`` as uint32 **nanoseconds relative to the scan start**.
* Hesai publishes ``timestamp`` as float64 **absolute UNIX seconds**.
* Livox in its ROS2 CustomMsg publishes ``offset_time`` as uint32 nanoseconds.

Feed the wrong one into a deskewing stage and you either get no correction at
all (the values are ~0 in the units it expected) or a wildly overcorrected
cloud that looks like a fan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "GridIndex",
    "voxel_downsample",
    "radius_outlier_removal",
    "statistical_outlier_removal",
    "estimate_normals",
    "PlaneModel",
    "ransac_ground_plane",
    "remove_ground",
    "detect_timestamp_format",
    "normalize_point_times",
    "assign_rings",
    "ring_statistics",
    "deskew_points",
    "bounds",
]

_EPS = 1e-12


def _as_points(points: np.ndarray) -> np.ndarray:
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got shape {P.shape}")
    return P


def bounds(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned ``(min_xyz, max_xyz)`` of a cloud."""
    P = _as_points(points)
    if P.shape[0] == 0:
        raise ValueError("empty cloud has no bounds")
    return P.min(axis=0), P.max(axis=0)


# --------------------------------------------------------------------------
# Spatial index
# --------------------------------------------------------------------------
class GridIndex:
    """Uniform-voxel spatial hash for radius and k-nearest queries.

    A hash grid rather than a KD-tree because LiDAR clouds are close to
    uniformly dense in space, which is the case where a grid wins and where a
    tree's build cost is wasted.  Queries are exact: the k-nearest search
    expands the searched ring of cells until the k-th distance is inside the
    radius already guaranteed to be covered.
    """

    def __init__(self, points: np.ndarray, cell_size: float) -> None:
        if cell_size <= 0:
            raise ValueError("cell_size must be > 0")
        self.points = _as_points(points)
        self.cell_size = float(cell_size)
        self.origin = self.points.min(axis=0) if len(self.points) else np.zeros(3)
        self.cells = np.floor((self.points - self.origin) / self.cell_size).astype(np.int64)
        table: Dict[Tuple[int, int, int], List[int]] = {}
        for i, c in enumerate(map(tuple, self.cells)):
            table.setdefault(c, []).append(i)
        self.table: Dict[Tuple[int, int, int], np.ndarray] = {
            k: np.asarray(v, dtype=np.int64) for k, v in table.items()
        }

    def _gather(self, cell: Tuple[int, int, int], ring: int) -> np.ndarray:
        """Indices of all points in the (2*ring+1)^3 block of cells around ``cell``."""
        out: List[np.ndarray] = []
        cx, cy, cz = cell
        rng = range(-ring, ring + 1)
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    idx = self.table.get((cx + dx, cy + dy, cz + dz))
                    if idx is not None:
                        out.append(idx)
        if not out:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(out)

    def radius_counts(self, radius: float) -> np.ndarray:
        """Number of *other* points within ``radius`` of each point."""
        if radius > self.cell_size:
            raise ValueError(
                "radius must be <= cell_size for an exact single-ring query; "
                "build the index with cell_size >= radius"
            )
        counts = np.zeros(len(self.points), dtype=np.int64)
        r2 = radius * radius
        for cell, members in self.table.items():
            cand = self._gather(cell, 1)
            if cand.size == 0:
                continue
            d2 = np.sum(
                (self.points[members][:, None, :] - self.points[cand][None, :, :]) ** 2,
                axis=2,
            )
            counts[members] = np.sum(d2 <= r2, axis=1) - 1  # drop self
        return counts

    def knn(self, k: int, max_ring: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """Exact k-nearest neighbours (excluding self).

        Returns ``(indices, distances)``, each ``(N, k)``.  Points with fewer
        than ``k`` neighbours in the cloud get ``-1`` indices and ``inf``
        distances in the unfilled slots.
        """
        n = len(self.points)
        if k < 1:
            raise ValueError("k must be >= 1")
        idx_out = np.full((n, k), -1, dtype=np.int64)
        dist_out = np.full((n, k), np.inf, dtype=float)
        for cell, members in self.table.items():
            pending = members
            ring = 1
            while pending.size and ring <= max_ring:
                cand = self._gather(cell, ring)
                if cand.size <= 1:
                    ring += 1
                    continue
                d2 = np.sum(
                    (self.points[pending][:, None, :] - self.points[cand][None, :, :]) ** 2,
                    axis=2,
                )
                # Mask self-matches.
                self_mask = cand[None, :] == pending[:, None]
                d2 = np.where(self_mask, np.inf, d2)
                kk = min(k, d2.shape[1])
                order = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
                rows = np.arange(len(pending))[:, None]
                sel_d2 = d2[rows, order]
                srt = np.argsort(sel_d2, axis=1)
                sel_d2 = np.take_along_axis(sel_d2, srt, axis=1)
                sel_idx = cand[np.take_along_axis(order, srt, axis=1)]
                d = np.sqrt(sel_d2)
                idx_out[pending, :kk] = sel_idx
                dist_out[pending, :kk] = d
                # A result is exact once the k-th distance is inside the
                # sphere fully covered by the searched block.
                covered = ring * self.cell_size
                worst = d[:, kk - 1] if kk == k else np.full(len(pending), np.inf)
                done = np.isfinite(worst) & (worst <= covered)
                pending = pending[~done]
                ring += 1
        return idx_out, dist_out


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    method: str = "centroid",
) -> np.ndarray:
    """Reduce a cloud to one point per occupied voxel.

    Parameters
    ----------
    voxel_size:
        Edge length, metres.  This is the same number as LIO-SAM's
        ``odometrySurfLeafSize`` / ``mappingSurfLeafSize`` and Cartographer's
        ``voxel_filter_size``.  Too large and you destroy the thin structures
        that constrain rotation (door frames, poles) -- the map then rotates
        freely in open areas.  Too small and the mapping thread falls behind
        real time, scans queue up, and the estimator starts extrapolating,
        which looks exactly like an IMU problem.
    method:
        ``"centroid"`` averages the points in each voxel (default; slightly
        smoother normals).  ``"first"`` keeps the first point encountered,
        which preserves original measurements including their ring/time
        association.

    Returns
    -------
    (M, 3) array with ``M <= N``.  With ``"centroid"`` the output bounding box
    is always contained in the input bounding box, since every output point is
    a convex combination of input points.
    """
    P = _as_points(points)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    if P.shape[0] == 0:
        return P.copy()
    keys = np.floor(P / voxel_size).astype(np.int64)
    # Lexicographic unique over the 3 integer coordinates.
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    m = len(counts)
    if method == "centroid":
        sums = np.zeros((m, 3), dtype=float)
        np.add.at(sums, inverse, P)
        return sums / counts[:, None]
    if method == "first":
        first = np.full(m, -1, dtype=np.int64)
        # Reverse iteration leaves the lowest index per voxel.
        first[inverse[::-1]] = np.arange(len(P) - 1, -1, -1)
        return P[first]
    raise ValueError(f"unknown method {method!r}; use 'centroid' or 'first'")


def radius_outlier_removal(
    points: np.ndarray, radius: float, min_neighbors: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop points with fewer than ``min_neighbors`` others inside ``radius``.

    Kills isolated returns: rain, dust, retro-reflector blooming and the
    "ghost ring" some sensors emit at close range.  Those points are a real
    problem for LiDAR-inertial odometry because they are *stable in the sensor
    frame*, so the scan matcher happily uses them as if they were structure
    and the map slowly follows the robot.

    Returns ``(filtered_points, keep_mask)``.
    """
    P = _as_points(points)
    if P.shape[0] == 0:
        return P.copy(), np.zeros(0, dtype=bool)
    grid = GridIndex(P, cell_size=max(radius, 1e-6))
    counts = grid.radius_counts(radius)
    keep = counts >= min_neighbors
    return P[keep], keep


def statistical_outlier_removal(
    points: np.ndarray, k: int = 12, std_ratio: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop points whose mean k-NN distance is an outlier for the cloud.

    Density-adaptive, unlike :func:`radius_outlier_removal`: it keeps sparse
    far-field returns that a fixed radius would delete.  For a spinning LiDAR
    the point spacing grows linearly with range, so a fixed-radius filter set
    for 10 m will strip everything past 30 m and quietly remove the long-range
    structure that constrains your yaw.

    Returns ``(filtered_points, keep_mask)``.
    """
    P = _as_points(points)
    n = len(P)
    if n <= k:
        return P.copy(), np.ones(n, dtype=bool)
    span = float(np.max(P.max(axis=0) - P.min(axis=0)))
    cell = max(span / 20.0, 1e-3)
    grid = GridIndex(P, cell_size=cell)
    _, dist = grid.knn(k)
    finite = np.isfinite(dist)
    counts = finite.sum(axis=1)
    summed = np.where(finite, dist, 0.0).sum(axis=1)
    mean_d = np.where(counts > 0, summed / np.maximum(counts, 1), np.inf)
    valid = np.isfinite(mean_d)
    mu = float(np.mean(mean_d[valid]))
    sigma = float(np.std(mean_d[valid]))
    threshold = mu + std_ratio * sigma
    keep = mean_d <= threshold
    return P[keep], keep


# --------------------------------------------------------------------------
# Normals
# --------------------------------------------------------------------------
def estimate_normals(
    points: np.ndarray,
    k: int = 12,
    viewpoint: Optional[Sequence[float]] = None,
    return_curvature: bool = False,
):
    """Surface normals by PCA over each point's k nearest neighbours.

    The normal is the eigenvector of the local covariance with the smallest
    eigenvalue; curvature is reported as ``lambda_0 / (lambda_0 + lambda_1 +
    lambda_2)``, i.e. how far from planar the neighbourhood is.

    Normals are sign-ambiguous.  If ``viewpoint`` is given (use the sensor
    origin, usually ``[0, 0, 0]`` in the LiDAR frame) every normal is flipped
    to face it.  **Consistent orientation matters** for the degeneracy
    analysis in :mod:`slamkit.degeneracy`: it works on the *distribution* of
    normal directions, and an unoriented set of normals is symmetric about the
    origin, which halves nothing but does make some sanity checks confusing.

    Returns ``normals`` ``(N, 3)``, or ``(normals, curvature)`` if
    ``return_curvature``.
    """
    P = _as_points(points)
    n = len(P)
    if n < 3:
        raise ValueError("need at least 3 points to estimate a normal")
    k = min(k, n - 1)
    span = float(np.max(P.max(axis=0) - P.min(axis=0)))
    cell = max(span / 20.0, 1e-3)
    grid = GridIndex(P, cell_size=cell)
    idx, dist = grid.knn(k)
    normals = np.zeros((n, 3), dtype=float)
    curvature = np.zeros(n, dtype=float)
    for i in range(n):
        nb = idx[i][idx[i] >= 0]
        if len(nb) < 2:
            normals[i] = np.array([0.0, 0.0, 1.0])
            curvature[i] = 1.0 / 3.0
            continue
        neigh = np.vstack([P[i], P[nb]])
        centred = neigh - neigh.mean(axis=0)
        cov = centred.T @ centred / len(neigh)
        evals, evecs = np.linalg.eigh(cov)
        normals[i] = evecs[:, 0]
        total = float(np.sum(np.abs(evals)))
        curvature[i] = float(abs(evals[0]) / total) if total > _EPS else 0.0
    if viewpoint is not None:
        vp = np.asarray(viewpoint, dtype=float).reshape(3)
        to_vp = vp[None, :] - P
        flip = np.sum(normals * to_vp, axis=1) < 0.0
        normals[flip] *= -1.0
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, _EPS)
    if return_curvature:
        return normals, curvature
    return normals


# --------------------------------------------------------------------------
# Ground plane
# --------------------------------------------------------------------------
@dataclass
class PlaneModel:
    """A plane ``normal . p + offset = 0`` with unit normal, oriented +Z up."""

    normal: np.ndarray
    offset: float
    inlier_mask: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, bool))
    n_inliers: int = 0
    n_iterations: int = 0

    @property
    def height(self) -> float:
        """Vertical distance from the sensor origin down to the plane, metres.

        This is the sensor mounting height when the plane is the ground. It is
        the *vertical* distance (along Z), not the perpendicular distance --
        the two differ by ``1 / cos(tilt)`` on a sloped plane, and it is the
        vertical one you compare against a tape measure.
        """
        nz = float(self.normal[2])
        if abs(nz) < _EPS:
            return float("nan")
        return float(self.offset) / nz

    @property
    def tilt_deg(self) -> float:
        """Angle between the plane normal and +Z, degrees."""
        c = max(-1.0, min(1.0, float(self.normal[2])))
        return math.degrees(math.acos(c))

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed point-to-plane distance for each row of ``points``."""
        P = _as_points(points)
        return P @ self.normal + self.offset


def ransac_ground_plane(
    points: np.ndarray,
    distance_threshold: float = 0.05,
    max_iterations: int = 200,
    max_tilt_deg: Optional[float] = 30.0,
    min_inliers: int = 10,
    seed: int = 0,
    refine: bool = True,
) -> PlaneModel:
    """RANSAC plane fit, constrained to be roughly horizontal.

    Parameters
    ----------
    distance_threshold:
        Inlier band, metres.  Set it to about 3x the sensor's range noise.
        Too tight and the ground splits into strips (one per ring) and the
        fit locks onto a single ring; too loose and a shallow ramp or a parked
        car roof gets absorbed into "ground".
    max_tilt_deg:
        Reject candidate planes whose normal is more than this far from +Z.
        Without it, RANSAC on an indoor cloud reliably picks a *wall*, because
        walls have more points than the floor once the floor is occluded by
        the robot's own body.  Pass ``None`` to disable.
    seed:
        Fixed by default so results are reproducible; this matters when the
        plane height feeds a z-drift diagnosis.

    Returns
    -------
    :class:`PlaneModel`.  ``n_inliers == 0`` means no plane met the criteria.
    """
    P = _as_points(points)
    n = len(P)
    if n < 3:
        return PlaneModel(np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(n, bool), 0, 0)
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0
    best_normal = np.array([0.0, 0.0, 1.0])
    best_offset = 0.0
    cos_limit = math.cos(math.radians(max_tilt_deg)) if max_tilt_deg is not None else -1.0
    iterations = 0
    for _ in range(int(max_iterations)):
        iterations += 1
        i, j, k = rng.choice(n, size=3, replace=False)
        v1 = P[j] - P[i]
        v2 = P[k] - P[i]
        nrm = np.cross(v1, v2)
        nn = float(np.linalg.norm(nrm))
        if nn < 1e-9:
            continue  # collinear sample
        nrm = nrm / nn
        if nrm[2] < 0:
            nrm = -nrm
        if nrm[2] < cos_limit:
            continue  # too steep to be the ground
        off = -float(nrm @ P[i])
        d = np.abs(P @ nrm + off)
        mask = d <= distance_threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
            best_normal = nrm
            best_offset = off
    if best_count < min_inliers:
        return PlaneModel(best_normal, best_offset, best_mask, best_count, iterations)
    if refine and best_count >= 3:
        # Least-squares refit on the inliers: RANSAC picks the support set,
        # a total-least-squares fit gets the geometry right.
        Q = P[best_mask]
        c = Q.mean(axis=0)
        _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
        nrm = Vt[2]
        if nrm[2] < 0:
            nrm = -nrm
        off = -float(nrm @ c)
        d = np.abs(P @ nrm + off)
        mask = d <= distance_threshold
        if int(mask.sum()) >= best_count:
            best_normal, best_offset = nrm, off
            best_mask, best_count = mask, int(mask.sum())
    return PlaneModel(best_normal, best_offset, best_mask, best_count, iterations)


def remove_ground(
    points: np.ndarray, plane: PlaneModel, margin: float = 0.10
) -> np.ndarray:
    """Return the points more than ``margin`` above ``plane``."""
    P = _as_points(points)
    return P[plane.distance(P) > margin]


# --------------------------------------------------------------------------
# Ring and per-point timestamp handling
# --------------------------------------------------------------------------
_TIME_FORMATS = {
    "relative_seconds": 1.0,
    "relative_milliseconds": 1e-3,
    "relative_microseconds": 1e-6,
    "relative_nanoseconds": 1e-9,
    "absolute_seconds": 1.0,
    "absolute_nanoseconds": 1e-9,
}


def detect_timestamp_format(values: np.ndarray, scan_period: float = 0.1) -> Dict[str, object]:
    """Guess the units and datum of a per-point timestamp field.

    There is no standard.  ``sensor_msgs/PointCloud2`` says nothing about what
    a field called ``t`` or ``time`` or ``timestamp`` contains, so every driver
    picked something different.  This function looks at the magnitude and the
    span of the values and reports what they must be.

    Parameters
    ----------
    values:
        The raw field, as read out of the cloud.
    scan_period:
        Expected time to complete one full sweep, seconds (0.1 for a 10 Hz
        spinning LiDAR).  The *span* of a per-point time field over one scan
        should be close to this.

    Returns
    -------
    dict with ``format``, ``scale_to_seconds``, ``span_seconds``, ``absolute``
    and ``confidence`` (0-1).  ``format == "unknown"`` means the span did not
    match any plausible unit -- usually because the field is all zeros (a
    driver that does not populate it) or because you handed it the ``intensity``
    field by mistake.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return {"format": "unknown", "reason": "empty field", "confidence": 0.0,
                "scale_to_seconds": 1.0, "span_seconds": 0.0, "absolute": False}
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"format": "unknown", "reason": "all values non-finite", "confidence": 0.0,
                "scale_to_seconds": 1.0, "span_seconds": 0.0, "absolute": False}
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    mag = max(abs(lo), abs(hi))
    if span <= 0.0:
        return {"format": "unknown",
                "reason": "field is constant; the driver is not populating per-point time "
                          "(deskewing will silently do nothing)",
                "confidence": 0.9, "scale_to_seconds": 1.0, "span_seconds": 0.0,
                "absolute": mag > 1e8}
    absolute = mag > 1e8  # a UNIX epoch in seconds is ~1.7e9
    candidates: List[Tuple[str, float, float]] = []
    for name, scale in _TIME_FORMATS.items():
        if name.startswith("absolute") != absolute:
            continue
        span_s = span * scale
        # Score by how close the span is to one scan period (log ratio).
        ratio = span_s / max(scan_period, 1e-9)
        if ratio <= 0:
            continue
        score = 1.0 / (1.0 + abs(math.log10(ratio)))
        candidates.append((name, scale, score))
    if not candidates:
        return {"format": "unknown", "reason": "no unit matches the observed span",
                "confidence": 0.0, "scale_to_seconds": 1.0, "span_seconds": span,
                "absolute": absolute}
    candidates.sort(key=lambda c: -c[2])
    name, scale, score = candidates[0]
    return {
        "format": name,
        "scale_to_seconds": scale,
        "span_seconds": span * scale,
        "absolute": absolute,
        "min": lo,
        "max": hi,
        "confidence": float(score),
        "negative_offsets": bool(lo < 0),
    }


def normalize_point_times(
    values: np.ndarray,
    scan_period: float = 0.1,
    fmt: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Convert any per-point time field to seconds relative to the scan start.

    Returns ``(rel_seconds, info)`` where ``rel_seconds`` starts at 0 and
    ``info`` is the output of :func:`detect_timestamp_format` (or a synthetic
    equivalent when ``fmt`` is forced).
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    if fmt is None:
        info = detect_timestamp_format(v, scan_period=scan_period)
        scale = float(info["scale_to_seconds"])
    else:
        if fmt not in _TIME_FORMATS:
            raise ValueError(f"unknown format {fmt!r}; one of {sorted(_TIME_FORMATS)}")
        scale = _TIME_FORMATS[fmt]
        info = {"format": fmt, "scale_to_seconds": scale, "forced": True}
    rel = (v - v.min()) * scale
    info = dict(info)
    info["normalized_span_seconds"] = float(rel.max() - rel.min()) if rel.size else 0.0
    return rel, info


def assign_rings(
    points: np.ndarray,
    n_rings: int,
    fov_down_deg: float,
    fov_up_deg: float,
) -> np.ndarray:
    """Recover a ring index per point from elevation angle.

    Use when the driver does not publish a ``ring`` field (common with generic
    ``PointCloud2`` republishers and with some Livox setups).  LIO-SAM's
    feature extraction indexes into ``N_SCAN`` rows and will crash or silently
    drop the whole cloud without one.

    Returns an ``int`` array in ``[0, n_rings - 1]``; ring 0 is the lowest
    elevation.
    """
    P = _as_points(points)
    if n_rings < 1:
        raise ValueError("n_rings must be >= 1")
    r_xy = np.linalg.norm(P[:, :2], axis=1)
    elev = np.degrees(np.arctan2(P[:, 2], np.maximum(r_xy, _EPS)))
    span = fov_up_deg - fov_down_deg
    if span <= 0:
        raise ValueError("fov_up_deg must exceed fov_down_deg")
    idx = np.floor((elev - fov_down_deg) / span * n_rings).astype(int)
    return np.clip(idx, 0, n_rings - 1)


def ring_statistics(rings: np.ndarray, expected_rings: Optional[int] = None) -> Dict[str, object]:
    """Summarise a ring field: which rings are present, and how populated.

    A missing or nearly empty ring is a dead laser or a blocked window, and it
    is worth knowing before you spend an afternoon tuning ``edgeThreshold``:
    LIO-SAM computes curvature along a ring, so a ring with holes generates
    spurious edge features exactly at the holes.
    """
    r = np.asarray(rings).reshape(-1).astype(int)
    present, counts = np.unique(r, return_counts=True)
    out: Dict[str, object] = {
        "n_points": int(r.size),
        "rings_present": present.tolist(),
        "counts": counts.tolist(),
        "min_ring": int(present.min()) if present.size else -1,
        "max_ring": int(present.max()) if present.size else -1,
    }
    if expected_rings is not None:
        missing = sorted(set(range(expected_rings)) - set(present.tolist()))
        out["expected_rings"] = int(expected_rings)
        out["missing_rings"] = missing
        median = float(np.median(counts)) if counts.size else 0.0
        out["sparse_rings"] = [
            int(p) for p, c in zip(present, counts) if median > 0 and c < 0.3 * median
        ]
    return out


def deskew_points(
    points: np.ndarray,
    rel_times: np.ndarray,
    T_start_end: np.ndarray,
    target: str = "start",
) -> np.ndarray:
    """Motion-compensate ("deskew") one scan.

    A spinning LiDAR samples over ~100 ms.  If the sensor moved during that
    window, every point was measured from a different pose, so the raw cloud
    is a smear.  This undoes it by interpolating the sensor pose over the
    sweep and re-expressing every point in a single frame.

    Parameters
    ----------
    points:
        ``(N, 3)`` in the sensor frame, as measured.
    rel_times:
        ``(N,)`` in **seconds relative to the start of the sweep** -- run the
        raw field through :func:`normalize_point_times` first.
    T_start_end:
        4x4 pose of the sensor at sweep end expressed in the sweep-start
        frame.  In a LIO pipeline this comes from IMU preintegration over the
        sweep; that is the whole reason a LiDAR-inertial system deskews better
        than a LiDAR-only one.
    target:
        ``"start"`` or ``"end"`` -- which instant to express the output in.
        LIO-SAM deskews to the sweep start and stamps the cloud accordingly;
        if your downstream node assumes end-of-sweep stamping you get a
        constant one-sweep offset, which shows up as a fixed lag between the
        map and the robot.

    Interpolation is ``exp(s * log(R))`` for rotation and linear for
    translation -- constant-twist, which is correct to first order over 100 ms.
    """
    from .extrinsics import rotation_exp, rotation_log  # local import: avoid cycle

    P = _as_points(points)
    s = np.asarray(rel_times, dtype=float).reshape(-1)
    if len(s) != len(P):
        raise ValueError(f"rel_times length {len(s)} != points length {len(P)}")
    T = np.asarray(T_start_end, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("T_start_end must be 4x4")
    span = float(s.max() - s.min()) if len(s) else 0.0
    frac = (s - s.min()) / span if span > _EPS else np.zeros_like(s)
    if target == "end":
        frac = frac - 1.0
    elif target != "start":
        raise ValueError("target must be 'start' or 'end'")
    w = rotation_log(T[:3, :3])
    t = T[:3, 3]
    out = np.empty_like(P)
    # Group identical fractions to keep this vectorised for real clouds where
    # all points on a ring share a timestamp.
    uniq, inverse = np.unique(np.round(frac, 9), return_inverse=True)
    for u_i, u in enumerate(uniq):
        sel = inverse == u_i
        R_u = rotation_exp(w * u)
        out[sel] = P[sel] @ R_u.T + t * u
    return out

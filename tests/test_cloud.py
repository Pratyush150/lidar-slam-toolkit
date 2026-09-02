"""Point-cloud utilities, tested on clouds with known ground truth."""

import math

import numpy as np
import pytest

from slamkit.cloud import (
    GridIndex,
    assign_rings,
    bounds,
    deskew_points,
    detect_timestamp_format,
    estimate_normals,
    normalize_point_times,
    radius_outlier_removal,
    ransac_ground_plane,
    remove_ground,
    ring_statistics,
    statistical_outlier_removal,
    voxel_downsample,
)
from slamkit.extrinsics import euler_to_matrix, make_transform
from slamkit.synthetic import apply_motion_distortion


def _random_cloud(n=3000, extent=5.0, seed=0):
    return np.random.default_rng(seed).uniform(-extent, extent, (n, 3))


# ------------------------------------------------------- voxel downsample
def test_voxel_downsample_reduces_count_and_preserves_bounds():
    P = _random_cloud()
    D = voxel_downsample(P, 1.0)
    assert len(D) < len(P)
    lo_p, hi_p = bounds(P)
    lo_d, hi_d = bounds(D)
    # Every output point is a centroid of input points, so the output box is
    # contained in the input box.
    assert np.all(lo_d >= lo_p - 1e-9)
    assert np.all(hi_d <= hi_p + 1e-9)
    # 10 m of extent at a 1 m voxel is at most 11^3 occupied voxels.
    assert len(D) <= 11 ** 3


def test_voxel_downsample_is_monotone_in_voxel_size():
    P = _random_cloud()
    assert len(voxel_downsample(P, 0.5)) > len(voxel_downsample(P, 2.0))


def test_voxel_downsample_first_keeps_original_points():
    P = _random_cloud(n=500)
    D = voxel_downsample(P, 1.0, method="first")
    for row in D[:20]:
        assert np.any(np.all(np.isclose(P, row), axis=1))


def test_voxel_downsample_rejects_bad_arguments():
    with pytest.raises(ValueError):
        voxel_downsample(_random_cloud(10), 0.0)
    with pytest.raises(ValueError):
        voxel_downsample(_random_cloud(10), 1.0, method="nope")


# ----------------------------------------------------------- outlier removal
def test_radius_outlier_removal_drops_isolated_points():
    dense = np.random.default_rng(0).normal(0.0, 0.2, (400, 3))
    outliers = np.array([[10.0, 10.0, 10.0], [-8.0, 4.0, 2.0]])
    P = np.vstack([dense, outliers])
    filtered, keep = radius_outlier_removal(P, radius=0.5, min_neighbors=5)
    assert keep[-1] == False  # noqa: E712
    assert keep[-2] == False  # noqa: E712
    assert len(filtered) >= 380


def test_statistical_outlier_removal_keeps_the_bulk():
    dense = np.random.default_rng(1).normal(0.0, 0.5, (600, 3))
    outliers = np.random.default_rng(2).uniform(20.0, 30.0, (10, 3))
    P = np.vstack([dense, outliers])
    filtered, keep = statistical_outlier_removal(P, k=10, std_ratio=2.0)
    assert keep[:600].sum() > 570
    assert keep[600:].sum() == 0
    assert len(filtered) == int(keep.sum())


# ------------------------------------------------------------- grid index
def test_grid_knn_matches_brute_force():
    P = _random_cloud(n=400, extent=3.0, seed=5)
    grid = GridIndex(P, cell_size=1.0)
    idx, dist = grid.knn(5)
    brute = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(brute, np.inf)
    expected = np.sort(brute, axis=1)[:, :5]
    assert np.allclose(np.sort(dist, axis=1), expected, atol=1e-9)


def test_grid_radius_counts_match_brute_force():
    P = _random_cloud(n=300, extent=2.0, seed=6)
    grid = GridIndex(P, cell_size=0.8)
    counts = grid.radius_counts(0.8)
    brute = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    expected = (brute <= 0.8).sum(axis=1) - 1
    assert np.array_equal(counts, expected)


# ---------------------------------------------------------------- normals
def test_normals_of_a_plane_point_along_its_normal():
    rng = np.random.default_rng(3)
    xy = rng.uniform(-5.0, 5.0, (600, 2))
    P = np.column_stack([xy, np.zeros(600)])
    n = estimate_normals(P, k=10, viewpoint=[0.0, 0.0, 10.0])
    assert np.allclose(np.abs(n[:, 2]), 1.0, atol=1e-6)
    assert np.all(n[:, 2] > 0)          # oriented towards the viewpoint


def test_normals_of_a_tilted_plane():
    rng = np.random.default_rng(4)
    xy = rng.uniform(-5.0, 5.0, (600, 2))
    z = 0.5 * xy[:, 0]
    P = np.column_stack([xy, z])
    n = estimate_normals(P, k=12, viewpoint=[0.0, 0.0, 20.0])
    expected = np.array([-0.5, 0.0, 1.0])
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(n.mean(axis=0), expected, atol=1e-3)


def test_normal_curvature_is_low_on_a_plane():
    rng = np.random.default_rng(7)
    xy = rng.uniform(-5.0, 5.0, (400, 2))
    P = np.column_stack([xy, np.zeros(400)])
    _, curv = estimate_normals(P, k=10, return_curvature=True)
    assert float(np.median(curv)) < 1e-6


# ------------------------------------------------------------ ground plane
def test_ransac_recovers_a_known_plane():
    """Plane 0.3x - 0.2y - z - 1.5 = 0, plus 15% uniform clutter."""
    rng = np.random.default_rng(8)
    xy = rng.uniform(-10.0, 10.0, (2000, 2))
    z = 0.3 * xy[:, 0] - 0.2 * xy[:, 1] - 1.5 + rng.normal(0.0, 0.01, 2000)
    ground = np.column_stack([xy, z])
    clutter = rng.uniform(-10.0, 10.0, (300, 3))
    P = np.vstack([ground, clutter])

    plane = ransac_ground_plane(P, distance_threshold=0.06, max_iterations=400,
                                seed=0)
    expected = np.array([-0.3, 0.2, 1.0])
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(plane.normal, expected, atol=0.01)
    assert plane.offset == pytest.approx(1.5 / np.linalg.norm([0.3, -0.2, -1.0]),
                                         abs=0.02)
    assert plane.n_inliers > 1900
    assert plane.tilt_deg == pytest.approx(math.degrees(math.acos(expected[2])),
                                           abs=0.5)


def test_ransac_is_deterministic_for_a_fixed_seed():
    P = _random_cloud(n=800, seed=9)
    a = ransac_ground_plane(P, seed=42, max_iterations=100)
    b = ransac_ground_plane(P, seed=42, max_iterations=100)
    assert np.allclose(a.normal, b.normal)
    assert a.n_inliers == b.n_inliers


def test_ransac_tilt_constraint_rejects_a_wall():
    """A vertical wall with more points than the floor must not win."""
    rng = np.random.default_rng(10)
    wall_yz = rng.uniform(-5.0, 5.0, (1500, 2))
    wall = np.column_stack([np.zeros(1500), wall_yz])
    floor_xy = rng.uniform(-5.0, 5.0, (500, 2))
    floor = np.column_stack([floor_xy, np.full(500, -1.0)])
    P = np.vstack([wall, floor])
    plane = ransac_ground_plane(P, distance_threshold=0.05, max_tilt_deg=20.0,
                                seed=1, max_iterations=300)
    assert plane.normal[2] > 0.9
    assert plane.n_inliers == pytest.approx(500, abs=20)


def test_remove_ground_keeps_only_what_is_above():
    rng = np.random.default_rng(11)
    xy = rng.uniform(-5.0, 5.0, (800, 2))
    floor = np.column_stack([xy, np.zeros(800)])
    above = np.column_stack([rng.uniform(-5.0, 5.0, (100, 2)), np.full(100, 2.0)])
    P = np.vstack([floor, above])
    plane = ransac_ground_plane(P, distance_threshold=0.05, seed=2)
    rest = remove_ground(P, plane, margin=0.5)
    assert len(rest) == 100


# ------------------------------------------------- ring / timestamp fields
@pytest.mark.parametrize("values,expected", [
    (np.linspace(0.0, 0.0999, 100), "relative_seconds"),
    (np.linspace(0.0, 99_900_000.0, 100), "relative_nanoseconds"),
    (np.linspace(0.0, 99_900.0, 100), "relative_microseconds"),
    (np.linspace(0.0, 0.0999, 100) + 1.7e9, "absolute_seconds"),
    (np.linspace(0.0, 99_900_000.0, 100) + 1.7e18, "absolute_nanoseconds"),
])
def test_timestamp_format_detection(values, expected):
    info = detect_timestamp_format(values, scan_period=0.1)
    assert info["format"] == expected
    assert info["span_seconds"] == pytest.approx(0.0999, rel=0.01)


def test_timestamp_detection_flags_an_unpopulated_field():
    info = detect_timestamp_format(np.zeros(100))
    assert info["format"] == "unknown"
    assert "not populating" in str(info["reason"])


def test_timestamp_detection_flags_negative_relative_offsets():
    info = detect_timestamp_format(np.linspace(-0.1, 0.0, 100))
    assert info["negative_offsets"]


def test_normalize_point_times_returns_seconds_from_zero():
    raw = np.linspace(0.0, 99_900_000.0, 50) + 1.7e18
    rel, info = normalize_point_times(raw, scan_period=0.1)
    assert rel[0] == 0.0
    assert rel[-1] == pytest.approx(0.0999, rel=1e-6)
    assert info["format"] == "absolute_nanoseconds"


def test_normalize_point_times_can_be_forced():
    rel, info = normalize_point_times(np.arange(10.0), fmt="relative_milliseconds")
    assert info["forced"]
    assert rel[-1] == pytest.approx(0.009)
    with pytest.raises(ValueError):
        normalize_point_times(np.arange(10.0), fmt="furlongs")


def test_assign_rings_recovers_elevation_bands():
    elev = np.radians(np.linspace(-15.0, 15.0, 16))
    r = 10.0
    P = np.column_stack([r * np.cos(elev), np.zeros(16), r * np.sin(elev)])
    rings = assign_rings(P, 16, -15.0, 15.0)
    assert rings.min() == 0
    assert rings.max() == 15
    assert np.all(np.diff(rings) >= 0)


def test_ring_statistics_reports_missing_and_sparse_rings():
    rings = np.concatenate([np.full(100, i) for i in range(16) if i != 7])
    rings = np.concatenate([rings, np.full(5, 3)])
    stats = ring_statistics(rings, expected_rings=16)
    assert stats["missing_rings"] == [7]
    assert stats["max_ring"] == 15


# --------------------------------------------------------------- deskewing
def test_deskew_undoes_injected_motion_distortion():
    rng = np.random.default_rng(12)
    P = rng.uniform(-10.0, 10.0, (500, 3))
    rel_t = np.linspace(0.0, 0.1, 500)
    v = np.array([2.0, 0.0, 0.0])
    w = np.array([0.0, 0.0, 0.5])
    distorted = apply_motion_distortion(P, rel_t, v, w)
    T = make_transform(euler_to_matrix(w * 0.1), v * 0.1)
    recovered = deskew_points(distorted, rel_t, T)
    assert np.allclose(recovered, P, atol=1e-9)


def test_deskew_is_identity_when_the_sensor_did_not_move():
    P = np.random.default_rng(13).uniform(-5.0, 5.0, (200, 3))
    rel_t = np.linspace(0.0, 0.1, 200)
    assert np.allclose(deskew_points(P, rel_t, np.eye(4)), P, atol=1e-12)


def test_deskew_validates_its_inputs():
    P = np.zeros((10, 3))
    with pytest.raises(ValueError):
        deskew_points(P, np.zeros(5), np.eye(4))
    with pytest.raises(ValueError):
        deskew_points(P, np.zeros(10), np.eye(3))
    with pytest.raises(ValueError):
        deskew_points(P, np.zeros(10), np.eye(4), target="middle")


def test_plane_height_is_the_vertical_sensor_height():
    """Ground 1.5 m below a slightly tilted sensor -> height = 1.5 m."""
    rng = np.random.default_rng(20)
    xy = rng.uniform(-10.0, 10.0, (2000, 2))
    z = 0.1 * xy[:, 0] - 1.5
    plane = ransac_ground_plane(np.column_stack([xy, z]),
                                distance_threshold=0.02, seed=0)
    assert plane.height == pytest.approx(1.5, abs=0.01)
    assert plane.tilt_deg == pytest.approx(math.degrees(math.atan(0.1)), abs=0.2)


def test_plane_distance_is_signed():
    plane = ransac_ground_plane(
        np.column_stack([np.random.default_rng(21).uniform(-5, 5, (500, 2)),
                         np.zeros(500)]), distance_threshold=0.02, seed=0)
    above = plane.distance(np.array([[0.0, 0.0, 2.0]]))
    below = plane.distance(np.array([[0.0, 0.0, -2.0]]))
    assert above[0] > 0 and below[0] < 0

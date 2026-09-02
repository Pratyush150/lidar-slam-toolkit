"""Degeneracy scoring, tested on ray-cast scenes rather than on mocked normals.

The claim being tested: a corridor is degenerate along its own axis and a room
is not. Both clouds come from the same simulated sensor, so the difference is
the geometry and nothing else.
"""

import numpy as np
import pytest

from slamkit.degeneracy import (
    analyze_degeneracy,
    degeneracy_findings,
    rotation_information,
    translation_information,
)
from slamkit.extrinsics import make_transform
from slamkit.findings import Severity
from slamkit.synthetic import (
    corridor_scene,
    open_field_scene,
    room_scene,
    simulate_scan,
    tunnel_scene,
)

# A wide vertical FOV (OS0-class, 90 deg) so the floor and ceiling are actually
# in view. A VLP-16's +/-15 deg genuinely sees very little floor indoors, which
# is a real limitation, not an artefact of the analysis.
SENSOR = dict(n_rings=32, n_azimuth=180, fov_down_deg=-45.0, fov_up_deg=45.0,
              range_noise=0.005)


def _scan(scene, height, max_range):
    return simulate_scan(scene, make_transform(t=[0.0, 0.0, height]),
                         max_range=max_range, seed=0, **SENSOR)


# ------------------------------------------------------ information matrices
def test_translation_information_of_a_single_normal_direction():
    """All normals along +Z -> only Z is observable."""
    n = np.tile([0.0, 0.0, 1.0], (100, 1))
    M = translation_information(n)
    assert np.allclose(M, np.diag([0.0, 0.0, 1.0]), atol=1e-12)


def test_translation_information_of_isotropic_normals_is_isotropic():
    rng = np.random.default_rng(0)
    n = rng.normal(size=(4000, 3))
    M = translation_information(n)
    assert np.allclose(np.diag(M), 1.0 / 3.0, atol=0.03)


def test_rotation_information_is_range_normalised():
    """Scaling the scene must not change the relative rotation observability."""
    rng = np.random.default_rng(1)
    p = rng.normal(size=(500, 3)) * 5.0
    n = rng.normal(size=(500, 3))
    a = rotation_information(p, n)
    b = rotation_information(p * 10.0, n)
    assert np.allclose(a, b, atol=1e-9)


# --------------------------------------------------------------- scenes
def test_corridor_is_degenerate_along_its_own_axis():
    scan = _scan(corridor_scene(length=80.0, width=2.5, height=3.0), 1.0, 40.0)
    r = analyze_degeneracy(scan.points, voxel_size=0.3)
    assert r.degenerate
    assert r.environment == "corridor"
    assert r.weakest_axis == "x"
    # The corridor runs along +X, so the free direction is +/-X.
    assert abs(r.weakest_direction[0]) > 0.95
    assert r.translation_scores["x"] < 0.05
    assert r.translation_scores["y"] > 0.5
    assert r.condition_number > 25.0


def test_room_is_well_constrained():
    scan = _scan(room_scene(length=8.0, width=6.0, height=3.0), 1.2, 40.0)
    r = analyze_degeneracy(scan.points, voxel_size=0.3)
    assert not r.degenerate
    assert r.environment == "well_constrained"
    assert min(r.translation_scores.values()) > 0.1
    assert r.condition_number < 25.0


def test_corridor_ranks_worse_than_a_room():
    corridor = analyze_degeneracy(
        _scan(corridor_scene(length=80.0), 1.0, 40.0).points, voxel_size=0.3)
    room = analyze_degeneracy(
        _scan(room_scene(), 1.2, 40.0).points, voxel_size=0.3)
    assert corridor.condition_number > 10.0 * room.condition_number
    assert min(corridor.translation_scores.values()) < min(room.translation_scores.values())


def test_open_field_leaves_both_horizontal_axes_free():
    scan = _scan(open_field_scene(extent=60.0), 1.5, 60.0)
    r = analyze_degeneracy(scan.points, voxel_size=0.3)
    assert r.degenerate
    assert r.environment == "open_field"
    assert r.translation_scores["z"] > 0.9
    assert r.translation_scores["x"] < 0.05
    assert r.translation_scores["y"] < 0.05
    # Yaw is unobservable on a flat plane.
    assert r.rotation_scores["z"] < 0.05


def test_tunnel_loses_rotation_about_its_axis_too():
    scan = _scan(tunnel_scene(length=120.0, radius=2.5), 2.5, 40.0)
    r = analyze_degeneracy(scan.points, voxel_size=0.3)
    assert r.degenerate
    assert r.environment == "tunnel"
    assert r.translation_scores["x"] < 0.05
    assert r.rotation_scores["x"] < 0.10


def test_tunnel_is_reported_more_severely_than_a_corridor():
    tunnel = analyze_degeneracy(_scan(tunnel_scene(length=120.0), 2.5, 40.0).points,
                                voxel_size=0.3)
    corridor = analyze_degeneracy(_scan(corridor_scene(length=80.0), 1.0, 40.0).points,
                                  voxel_size=0.3)
    t_sev = degeneracy_findings(tunnel)[0].severity
    c_sev = degeneracy_findings(corridor)[0].severity
    assert t_sev > c_sev


# --------------------------------------------------------------- findings
def test_findings_name_the_weak_axis_and_offer_a_fix():
    r = analyze_degeneracy(_scan(corridor_scene(length=80.0), 1.0, 40.0).points,
                           voxel_size=0.3)
    f = degeneracy_findings(r)[0]
    assert f.code == "DEGENERACY_CORRIDOR"
    assert f.severity >= Severity.WARN
    assert "slides along X" in f.symptom
    assert "odometry" in f.fix or "prior" in f.fix


def test_well_constrained_scene_produces_an_ok_finding():
    r = analyze_degeneracy(_scan(room_scene(), 1.2, 40.0).points, voxel_size=0.3)
    f = degeneracy_findings(r)[0]
    assert f.code == "DEGENERACY_NONE"
    assert f.severity == Severity.OK


def test_sparse_cloud_is_reported_rather_than_crashing():
    r = analyze_degeneracy(np.zeros((5, 3)))
    assert r.environment == "sparse"
    assert r.degenerate
    assert degeneracy_findings(r)[0].code == "DEGENERACY_TOO_FEW_POINTS"


def test_report_is_serialisable_and_carries_the_numbers():
    r = analyze_degeneracy(_scan(corridor_scene(length=80.0), 1.0, 40.0).points,
                           voxel_size=0.3)
    d = r.to_dict()
    assert d["environment"] == "corridor"
    assert len(d["weakest_direction"]) == 3
    assert set(d["translation_scores"]) == {"x", "y", "z"}


def test_voxel_size_and_normals_are_mutually_exclusive():
    pts = _scan(room_scene(), 1.2, 40.0).points
    with pytest.raises(ValueError):
        analyze_degeneracy(pts, normals=np.zeros_like(pts), voxel_size=0.3)

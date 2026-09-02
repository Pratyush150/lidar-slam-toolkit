"""slamkit -- diagnostics for LiDAR SLAM setups (LIO-SAM, Cartographer, RTAB-Map).

SLAM rarely fails because the algorithm is wrong. It fails because of
extrinsics, time sync, IMU units and point-cloud timestamps. This package is
the toolkit for finding out which one it is, offline, from a laptop.

Modules
-------
``extrinsics``
    SE(3) maths plus validators for the four classic extrinsic mistakes.
``timesync``
    Offset, drift, jitter, monotonicity and receive-time-stamping diagnostics.
``drift``
    ATE/RPE against ground truth; loop residual, z-drift and revisit
    consistency without it.
``degeneracy``
    Eigen-analysis of the surface-normal distribution: which axis is free.
``cloud``
    Pure-numpy voxel filter, outlier removal, normals, RANSAC ground plane,
    ring/timestamp field handling, deskewing.
``synthetic``
    Ray-cast scenes, trajectories and IMU streams with injectable defects.
``rosbag_check``
    Topic/rate/frame_id/TF checks; degrades to a JSON or text bag dump when
    rosbag2 is not installed.
``doctor``
    The ``slam-doctor`` CLI that runs the whole battery and ranks the output.

Nothing here imports ROS at module level, so ``import slamkit`` works on any
machine with numpy.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .findings import Finding, Report, Severity  # noqa: F401

__all__ = [
    "__version__",
    "Finding",
    "Report",
    "Severity",
    "cloud",
    "degeneracy",
    "doctor",
    "drift",
    "extrinsics",
    "findings",
    "rosbag_check",
    "synthetic",
    "timesync",
]


def __getattr__(name: str):
    """Lazy submodule import so `import slamkit` stays cheap."""
    if name in __all__:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

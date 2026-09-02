# lidar-slam-toolkit

**SLAM rarely fails because the algorithm is wrong. It fails because of
extrinsics, time sync, IMU units and point-cloud timestamps.**

LIO-SAM, Cartographer and RTAB-Map are all correct implementations of
well-understood mathematics. When a map ghosts, tilts, floats upward or slides
down a corridor, the cause is almost always in the data you fed them — a
transposed rotation matrix, a 40 ms clock offset, degrees pasted into a radians
field, or a per-point timestamp field read in the wrong units.

This repo is the toolkit for finding out which one it is:

- **`slamkit/`** — pure-numpy diagnostics that run offline, with no ROS 2
  installation, on a laptop, from a customer's exported CSVs.
- **`configs/`** — reference configurations for LIO-SAM, Cartographer 2D/3D and
  RTAB-Map where **every parameter carries a comment saying what it does and
  what breaks when it is wrong**, plus per-sensor starting profiles.
- **`docs/SLAM_DEBUGGING.md`** — a symptom-driven field guide: you have a
  symptom, it gives you the cause, the check, and the fix.

---

## Quickstart

```bash
pip install numpy
python3 tools/slam-doctor --demo
```

`--demo` builds a synthetic dataset with four defects deliberately injected — a
45 ms IMU time offset, a transposed extrinsic rotation, a z-drift ramp with a
step in it, and a corridor — plus a bag description missing a static transform.
It then finds all of them and ranks them:

```
RANKED DIAGNOSIS (6 issue(s), worst first)

 1. [CRITICAL] TF_CHAIN_MISSING
    no TF path from 'base_link' to 'imu_link'
    symptom : 'Could not find a connection between [a] and [b] because they are
              not part of the same tree'. The SLAM node blocks or drops every
              message.
    fix     : Add the missing link. ...

 2. [CRITICAL] TIME_OFFSET_LARGE
    offset +44.9 ms -- BROKEN. Fix time sync before tuning anything else
    symptom : Stationary the map is perfect. As soon as you rotate, walls double
              and the trajectory kicks; the faster the turn, the worse it gets.
              Error scales as omega * dt.
    fix     : Add +44.9 ms to the IMU timestamps, or better, fix the source ...

 3. [CRITICAL] EXTRINSIC_TRANSPOSED
    configured rotation is off by 177.42 deg from the data-derived estimate, but
    its transpose is off by only 2.58 deg -- the matrix is inverted
    ...
```

Every finding carries the measured number, the **symptom** (what it looks like
in RViz, because that is how the problem was reported to you), and the **fix**.

On real data:

```bash
ros2 bag info my_bag > baginfo.txt          # on the robot

slam-doctor --bag baginfo.txt \
            --trajectory est_traj.csv \
            --imu imu.csv \
            --cloud one_scan.csv \
            --extrinsic extrinsic.json \
            --tf-chain base_link:velodyne \
            --tf-chain base_link:imu_link

slam-doctor --demo --json                   # machine-readable, for CI
```

Every argument is optional. Pass what you have; checks that need something you
did not supply are skipped rather than faked.

---

## What it checks

| Area | Check | Catches |
|---|---|---|
| **TF** | chain resolution, single parent, static/dynamic conflict, disconnected roots | "frame does not exist", robot jittering between two poses |
| **Bag** | topic presence by message type, average rate, dropped messages, frame_ids | node starts and silently receives nothing |
| **Time** | LiDAR/IMU offset by cross-correlation, clock drift, jitter, non-monotonic stamps, `now()`-stamping | "fine stationary, breaks when moving", "works with the bag, fails live" |
| **Extrinsics** | transposed rotation, degrees-in-radians, reversed lever arm, left-handed axes, non-orthonormal matrix | "map rotates when I turn", mirrored map, ghosting at corners |
| **Geometry** | per-axis observability from the surface-normal distribution | "why does it slide in a corridor" |
| **Drift** | ATE / RPE against ground truth; loop residual, z-drift ramp vs step, yaw drift, revisit consistency without it | "z drifts upward", "loop closure never fires" |
| **Cloud** | ring field completeness, per-point timestamp units and datum, ground-plane tilt | smeared clouds, deskewing that silently does nothing |

### The z-drift analyser

"The map floats upward" is the most common complaint and the one where a single
number tells you nothing. `analyze_z_drift()` splits the trace into a smooth
ramp and discrete steps, then tests whether the ramp tracks **time** or
**distance travelled** — which is what separates three causes needing three
different fixes:

| shape | cause | fix |
|---|---|---|
| ramp ∝ horizontal distance | attitude / gravity misalignment — the map is tilted, so driving forward also drives up | fix the extrinsic roll/pitch, or initialise level and stationary |
| ramp or curve ∝ time | accelerometer bias, or `imuGravity` wrong for your location | calibrate the bias; **raise** `imuAccBiasN` if it genuinely wanders |
| discrete steps | loop closure, a z clamp (`z_tollerance`), or degenerate geometry | not drift — IMU tuning will not touch it |

### The degeneracy analyser

Point-to-plane scan matching solves a system whose translation block is
`Σ nᵢnᵢᵀ` over the matched surface normals. In a straight corridor every normal
points at a wall or the floor; **nothing points along the corridor**. That
matrix has a near-zero eigenvalue along the corridor axis, the solver is free to
slide, and it will.

`analyze_degeneracy()` computes both the translation and rotation information
matrices, eigen-decomposes them, and reports a per-axis observability score plus
the weakest direction — which is the honest answer to "why does my SLAM slide in
a corridor", and tells you whether to reach for wheel odometry (corridor), an
independent scale source (tunnel), or GNSS (open field).

---

## Supported stacks and sensors

| Stack | Reference config | Notes |
|---|---|---|
| LIO-SAM | `configs/lio_sam/params.yaml` | Full parameter set. Long annotated section on `extrinsicRot` / `extrinsicRPY` / `extrinsicTrans`, which is where most setups break. |
| Cartographer 2D | `configs/cartographer/cartographer_2d.lua` | 2D LiDAR + wheel odometry, optional IMU. |
| Cartographer 3D | `configs/cartographer/cartographer_3d.lua` | 3D LiDAR + mandatory IMU. |
| RTAB-Map | `configs/rtabmap/rtabmap_params.yaml` | ICP registration and proximity loop closure for LiDAR-only rigs. |

| Sensor | Profile | Rings | Rate | Per-point time field | The gotcha |
|---|---|---|---|---|---|
| Velodyne VLP-16 | `sensor_profiles/velodyne_vlp16.yaml` | 16 | 10 Hz | `time`, float32 seconds, relative (sometimes negative) | dual-return mode silently doubles the point count and corrupts the range image |
| Ouster OS1-32/64 | `sensor_profiles/ouster_os1_32_64.yaml` | 32 / 64 | 10–20 Hz | `t`, uint32 **nanoseconds**, relative | `timestamp_mode`, and the non-identity `os_lidar`↔`os_imu` extrinsic that people assume is identity |
| Livox Mid-360 | `sensor_profiles/livox_mid360.yaml` | n/a (non-repetitive) | 10 Hz | `offset_time` ns relative, or `timestamp` float64 **absolute** | no ring field; a single frame is sparse; the driver applies its own extrinsic in degrees and millimetres |
| RPLIDAR (2D) | `sensor_profiles/rplidar_2d.yaml` | 1 | 5.5–20 Hz | none — only `scan_time` / `time_increment` | rotation direction vs the ROS convention: the map comes out mirrored and self-consistent |

---

## Library use

Everything is importable without ROS.

```python
import numpy as np
from slamkit.extrinsics import euler_to_matrix, solve_rotation_hand_eye, validate_extrinsics

# Estimate the LiDAR->IMU rotation from paired incremental motions,
# then check the configured one against it.
R_est, info = solve_rotation_hand_eye(lidar_motions, imu_motions)
print(info["axis_condition"], info["well_conditioned"])

report = validate_extrinsics(
    R=configured_rotation,
    t=configured_translation,
    R_estimated=R_est,
)
for f in report.problems:
    print(f)
```

```python
from slamkit.timesync import analyze_time_sync
rep = analyze_time_sync(scan_times, scan_rotations, imu_times, gyro)
print(rep.verdict)          # "offset +44.9 ms -- BROKEN. Fix time sync before ..."
```

```python
from slamkit.cloud import detect_timestamp_format, ransac_ground_plane
print(detect_timestamp_format(cloud["t"], scan_period=0.1))
plane = ransac_ground_plane(points, distance_threshold=0.08)
print(plane.tilt_deg)
```

```python
from slamkit.synthetic import corridor_scene, simulate_scan
from slamkit.degeneracy import analyze_degeneracy
scan = simulate_scan(corridor_scene(length=80.0), n_rings=32, n_azimuth=180)
print(analyze_degeneracy(scan.points, voxel_size=0.3).explanation)
```

Every diagnostic returns `Finding` objects with a stable `code`, a `severity`, a
`message` containing the measured number, a `symptom` and a `fix` — so the same
code drives both the terminal report and `--json` output for CI.

---

## Layout

```
configs/
  lio_sam/params.yaml                 full annotated parameter set
  cartographer/cartographer_2d.lua    2D reference config
  cartographer/cartographer_3d.lua    3D reference config
  rtabmap/rtabmap_params.yaml         ICP + proximity loop closure
  sensor_profiles/                    VLP-16, OS1-32/64, Mid-360, RPLIDAR
src/slamkit/
  extrinsics.py    SE(3) maths, hand-eye rotation solve, the four validators
  timesync.py      offset / drift / jitter / monotonicity / receive-time checks
  drift.py         ATE, RPE, loop residual, z-drift decomposition
  degeneracy.py    per-axis observability from the normal distribution
  cloud.py         voxel filter, outlier removal, normals, RANSAC ground plane,
                   ring and timestamp field handling, deskewing
  synthetic.py     ray-cast scenes, trajectories, IMU, injectable defects
  rosbag_check.py  topics, rates, frame_ids, TF completeness (guarded imports)
  doctor.py        the slam-doctor CLI
tools/slam-doctor  launcher that works from a bare checkout
docs/SLAM_DEBUGGING.md
tests/             205 tests, 430+ assertions, offline and deterministic
examples/
```

## Running the tests

```bash
pip install numpy pytest
python3 -m pytest -q
```

All tests are offline, deterministic and seeded. Nothing touches the network and
nothing needs ROS. The synthetic generator ray-casts against real planar
geometry, so the degeneracy tests are run against the physics of a corridor
rather than against a mock.

---

## What this is / what it isn't

**It is:**

- A diagnostic layer that runs before and alongside your SLAM stack, offline.
- Reference configurations whose comments explain the failure mode of every
  parameter, so the config teaches as well as configures.
- A written record of the failure modes that actually consume time on real
  LiDAR SLAM bring-ups.

**It isn't:**

- A SLAM implementation. It will not build you a map. Use LIO-SAM,
  Cartographer, RTAB-Map, FAST-LIO or Point-LIO for that.
- A replacement for a proper calibration toolbox. The hand-eye rotation solve
  here is for **verifying** an extrinsic and catching a transposed or
  wrong-units one — it estimates rotation only, needs multi-axis excitation, and
  reports its own conditioning so you know when not to trust it. For a
  production extrinsic including the lever arm, use a dedicated LiDAR-IMU
  calibration package.
- Tested against every driver and firmware combination. The sensor profiles are
  starting points with the gotchas written down; verify the field names and
  units against **your** driver version, which is what
  `slamkit.cloud.detect_timestamp_format()` is for.
- A performance-tuning tool. It will tell you that the mapping thread is behind;
  it will not tell you which line of C++ to optimise.

**Known limitations:**

- The rosbag2 reader path needs `rosbag2_py`. Without it, everything still works
  from a `ros2 bag info` text dump or a JSON description — that is the intended
  workflow, since the analysis usually happens on a different machine from the
  robot.
- The time-offset estimator needs rotational excitation. Smooth or purely
  translational motion produces a broad correlation peak; the estimator reports
  `peak_width_ms` and marks itself untrustworthy rather than returning a
  confident wrong answer.
- Degeneracy scores are relative (best axis = 1.0) and describe observability,
  not the metric drift you will suffer. How far you actually slide depends on
  your IMU, your update rate, and how long you stay in the degenerate stretch.
- The spatial index is a uniform hash grid. It is exact and fast on
  LiDAR-density clouds; it is not the right structure for a cloud with extreme
  density variation across many orders of magnitude.
- 2D SLAM support is limited to the RPLIDAR profile and the Cartographer 2D
  config. The trajectory and degeneracy analysis assume 3D poses.

---

## Related repositories

- [`ros2-drone-bringup`](https://github.com/Pratyush150/ros2-drone-bringup) —
  ROS 2 bring-up, TF trees and launch structure for a PX4 drone. The TF checks
  here exist because of the failures documented there.
- [`flight-log-analyzer`](https://github.com/Pratyush150/flight-log-analyzer) —
  PX4/ArduPilot log analysis. Same approach applied to flight logs: measure
  first, then tune.
- [`drone-control-toolkit`](https://github.com/Pratyush150/drone-control-toolkit) —
  estimation and control primitives, including the complementary filter and EKF
  machinery that the IMU discussion here assumes.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Pratyush Vatsa.

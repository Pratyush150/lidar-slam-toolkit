# LiDAR SLAM debugging: a symptom-driven field guide

SLAM rarely fails because the algorithm is wrong. LIO-SAM, Cartographer and
RTAB-Map are all correct implementations of well-understood mathematics. What
fails is the data you feed them: the transform between the sensors, the clocks
that stamp the data, the units in a config field, and the per-point timestamps
inside the cloud.

This guide is organised by what you *see*, because that is what you have when
the problem starts. Each entry gives the cause, the check to run, and the fix.

Every check below has a corresponding function in `slamkit`, so you can run it
offline on a laptop with no ROS installation.

---

## Contents

- [Bring-up order, and why tuning first is wasted effort](#bring-up-order)
- [Symptom: map duplicates / ghosting](#symptom-map-duplicates--ghosting)
- [Symptom: the map rotates when I turn](#symptom-the-map-rotates-when-i-turn)
- [Symptom: z drifts upward](#symptom-z-drifts-upward)
- [Symptom: fine stationary, breaks when moving](#symptom-fine-stationary-breaks-when-moving)
- [Symptom: loop closure never fires](#symptom-loop-closure-never-fires)
- [Symptom: works with the bag, fails live](#symptom-works-with-the-bag-fails-live)
- [Symptom: the point cloud looks smeared](#symptom-the-point-cloud-looks-smeared)
- [Symptom: the map slides in a corridor](#symptom-the-map-slides-in-a-corridor)
- [Symptom: the map is mirrored](#symptom-the-map-is-mirrored)
- [Symptom: SLAM node starts and does nothing](#symptom-slam-node-starts-and-does-nothing)
- [The 20-minute triage](#the-20-minute-triage)

---

## Bring-up order

**TF → time sync → extrinsics → tuning.** In that order. No exceptions.

```
1. TF          Every sensor message has a frame_id. Every frame has exactly one
               parent. The chains your SLAM node will look up all resolve.
2. Time sync   Stamps come from the sensor clock, not from now() in a callback.
               LiDAR/IMU offset is under a few ms. No drift over the run.
3. Extrinsics  The LiDAR→IMU rotation direction is verified against data, not
               against a drawing. The lever arm is in metres and points the
               right way.
4. Tuning      Leaf sizes, feature thresholds, loop-closure gates, weights.
```

### Why tuning first is wasted effort

Tuning a parameter means finding the value that produces the best output *given
everything else*. If "everything else" includes a 45 ms time offset, you will
find the leaf size that best compensates for a 45 ms time offset. It will work
— at the speed you tested at. Drive faster and it breaks, because the error you
were compensating for scales with velocity and your compensation does not.

Then you fix the time offset, and every parameter you tuned is now wrong.

The three layers below tuning have a property tuning does not: they are either
right or wrong, and you can *check* which. There is no judgement involved in
whether `base_link → imu_link` resolves. Settle the checkable things first and
the tuning becomes a short, stable job instead of an endless one.

There is a second reason, specific to LiDAR-inertial systems. The estimator has
bias states, and bias states absorb model error. A wrong extrinsic or a time
offset does not show up as a large residual — it gets quietly written into the
accelerometer bias, which corrupts the estimated gravity direction, which tilts
the map, which drifts z. By the time you see the symptom it is three steps
downstream of the cause, and the parameter that "fixes" it (`imuAccBiasN`,
usually) is nowhere near the actual problem.

---

## Symptom: map duplicates / ghosting

Walls appear twice, offset by a few centimetres to a metre. Often the
duplication is worse near corners, or worse on a second pass through the same
space.

### Ghosting that appears at corners, scaling with how much you turned

**Cause:** extrinsic rotation error, or a LiDAR/IMU time offset. Both produce
error proportional to rotation, which is why the artefact concentrates at
corners.

**Check:**

```python
from slamkit.timesync import analyze_time_sync
rep = analyze_time_sync(scan_times, scan_rotations, imu_times, gyro)
print(rep.verdict)
```

Then, once time is clean:

```python
from slamkit.extrinsics import solve_rotation_hand_eye, check_transposed_rotation
R_est, info = solve_rotation_hand_eye(lidar_motions, imu_motions)
print(check_transposed_rotation(R_configured, R_est))
```

Do these in that order. An uncorrected time offset biases the extrinsic
estimate — during a 1 rad/s turn, a 20 ms offset looks like 1.1° of extrinsic
error. Fix time first or you will "calibrate" the offset into your extrinsic
and then wonder why the calibration is speed-dependent.

**Fix:** correct the offset at the source (PTP/PPS), then re-solve the
extrinsic.

### Ghosting that appears during straight-line motion, scaling with speed

**Cause:** the sweep is not being deskewed, or is being deskewed with the wrong
per-point timestamps. See [the point cloud looks
smeared](#symptom-the-point-cloud-looks-smeared).

### Doubling separated by roughly twice the sensor spacing

**Cause:** the lever arm (`extrinsicTrans`) is entered with the wrong sign or
in the wrong frame direction. The same wall gets mapped from two positions that
differ by 2× the lever arm.

**Check:**

```python
from slamkit.extrinsics import check_translation_direction
print(check_translation_direction(t_configured, t_estimated))
```

**Fix:** reverse it — but reverse the *transform*, not the three numbers.
`t_b_a = -R_a_b.T @ t_a_b`, not `-t_a_b`. Use
`slamkit.extrinsics.invert_transform()` on the full 4×4. (For a pure
translation the two happen to be the same, which is exactly why this bug ships.)

### Ghosting only where the robot passed twice, minutes apart

**Cause:** accumulated odometry drift that loop closure has not corrected. This
is not a data-integrity bug, it is a drift-plus-loop-closure problem.

**Check:**

```python
from slamkit.drift import revisit_consistency
print(revisit_consistency(poses, times, radius=3.0, min_time_gap_s=30.0))
```

**Fix:** see [loop closure never fires](#symptom-loop-closure-never-fires).

---

## Symptom: the map rotates when I turn

You yaw the robot and the map pitches or rolls. Or you yaw left and the map
yaws right. Driving in a straight line looks perfectly fine.

**Cause:** the extrinsic rotation is transposed. `extrinsicRot` is
`R_lidar_from_imu` — it takes a vector in the IMU frame and expresses it in the
LiDAR frame. If you wrote down "where the LiDAR is, as seen from the IMU", you
built `R_imu_from_lidar` and it needs transposing.

Both matrices are perfectly valid rotations. Nothing will warn you.

**Why it survives your bench test:** a 0° or 180° mount is its own transpose.
If you first tested with the IMU mounted in the same orientation as the LiDAR,
the bug was invisible. It appears the day someone rotates the IMU 90° in its
bracket.

**Check:**

```python
from slamkit.extrinsics import check_transposed_rotation
f = check_transposed_rotation(R_configured, R_estimated_from_data)
print(f.message)   # "...off by 89.9 deg, but its transpose is off by 0.4 deg"
```

**30-second manual check** (do this on every new robot):

1. Rotate the robot slowly counter-clockwise (yaw left, viewed from above).
2. Echo the converted IMU angular rate. `z` must be **positive**
   (REP-103: X forward, Y left, Z up, right-handed).
3. Pitch the nose up. Converted `y` must be **negative**.

If the axes are right but a sign is inverted, you are transposed. If the axes
are permuted, you have the wrong mount angle entirely.

**Fix:** transpose the matrix — and rename your variables so it does not happen
again. `R_lidar_from_imu` is unambiguous; `R_lidar_imu` is not.

---

## Symptom: z drifts upward

The most common complaint, and the one with the most different causes. The
number ("0.4 m/min") tells you nothing. The **shape** of the drift tells you
everything.

```python
from slamkit.drift import analyze_z_drift
import numpy as np

xy = poses[:, :2, 3]
dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
r = analyze_z_drift(times, poses[:, 2, 3], horizontal_distance=dist)
print(r.likely_cause)
for line in r.evidence:
    print(" -", line)
```

The analyser splits the trace into a smooth ramp and discrete steps, then tests
whether the ramp tracks **time** or **distance travelled**.

### Shape: smooth ramp, proportional to horizontal distance travelled

**Cause:** attitude / gravity misalignment. The map is tilted, so driving
"forward" also drives "up". A 1° tilt is 1.7 cm of z per metre travelled — 1.7 m
over a 100 m run.

Where the tilt comes from, in order of likelihood:

1. The extrinsic roll/pitch between LiDAR and IMU is wrong.
2. The system initialised while moving, so gravity was never observed cleanly.
3. `imuRPYWeight` is 0 with a 9-axis IMU, so nothing anchors the map to gravity.
4. The IMU is genuinely mounted at an angle that nobody put in the config.

**Check:** fit the ground plane in the sensor frame and look at its tilt.

```python
from slamkit.cloud import ransac_ground_plane
plane = ransac_ground_plane(points, distance_threshold=0.08)
print(plane.tilt_deg)     # should match your intended mount angle, not exceed it
```

**Fix:** fix the attitude, not the z axis. Do not enable GPS elevation to paper
over it.

### Shape: smooth ramp or curve, proportional to time, independent of motion

**Cause:** accelerometer bias, or `imuGravity` not matching your location.
A 0.02 m/s² gravity mismatch is ~0.6 m of z after 8 s of open-loop integration.
A curve (quadratic in time) rather than a straight line is the signature of a
constant bias being integrated twice.

**Fix:** leave the robot stationary for 10–20 s at startup so the bias is
observed. Set `imuGravity` for your latitude and altitude (9.79–9.83). If the
bias genuinely wanders with temperature, **raise** `imuAccBiasN` so the
estimator is allowed to track it — the common instinct to lower it freezes the
bias state and pushes the error into tilt instead.

### Shape: discrete steps

**Cause:** something is yanking the pose graph. Candidates:

- A loop closure firing (correctly or not).
- A ground-plane or z constraint snapping the trajectory.
- LIO-SAM's `z_tollerance` clamping vertical motion on a platform that actually
  moves vertically. (Note the upstream misspelling.)
- Degenerate geometry: the estimator loses vertical constraint, free-runs on the
  IMU, then snaps back when structure reappears.

**Fix:** identify which. Steps are *not* drift, and IMU tuning will not touch
them. `analyze_z_drift()` reports the index of each step so you can go and look
at what happened at that moment.

---

## Symptom: fine stationary, breaks when moving

The map is beautiful while the robot sits still. Start driving and it degrades;
turn and it falls apart.

**This is almost always time sync.** The error terms in a LiDAR-inertial system
are

```
angular error  ≈ ω · Δt
position error ≈ v · Δt  +  ω · Δt · r
```

Stationary, `ω = 0` and `v = 0`, so both vanish regardless of how bad `Δt` is.
The failure is *proportional to motion*. That is why it survives testing.

Put numbers in: turning at 1 rad/s (57 °/s — not fast) with a 10 ms offset gives
0.57° of angular error per scan. Against a wall 20 m away that is 20 cm of
apparent displacement. The optimiser sees the IMU and the scan matcher
disagreeing by 20 cm, splits the difference, and writes the residual into the
bias states — which then corrupts gravity, which drifts z.

**Check:**

```bash
slam-doctor --trajectory est.csv --imu imu.csv
```

or directly:

```python
from slamkit.timesync import estimate_lidar_imu_offset
est = estimate_lidar_imu_offset(scan_times, scan_rotations, imu_times, gyro)
print(f"{est.offset_ms:+.1f} ms  (correlation {est.correlation:.2f}, "
      f"peak width {est.peak_width_ms:.0f} ms)")
```

The offset is what you must **add to the IMU timestamps**.

| offset | verdict |
|---|---|
| < 1 ms | fine — what PTP or a PPS-disciplined sensor gives you |
| 1–5 ms | acceptable for slow ground robots; visible on fast yaw |
| 5–20 ms | degrading; ghosting on turns, biased gravity |
| > 20 ms | broken; fix before touching anything else |
| > 100 ms | whole-scan misalignment — start-of-sweep vs end-of-sweep stamping, or ROS receive time on a buffered link |

**Record the right data for this check.** Cross-correlation needs *sharp*
motion. Yaw left, stop, yaw right, stop — 20–30 s. A gentle constant sweep
produces a broad correlation peak and an estimate that noise moves around;
`peak_width_ms` tells you when that has happened.

**Other candidates, if time is clean:**

- Motion distortion (no deskewing) — see [smeared point
  cloud](#symptom-the-point-cloud-looks-smeared).
- The mapping thread missing real time, so the estimator extrapolates. Check CPU
  and `mappingProcessInterval`.
- Dropped messages under load: `slamkit.timesync.detect_jitter()` reports
  `estimated_dropped`.

---

## Symptom: loop closure never fires

You drive a loop, come back to the start, and nothing happens. The map stays
split.

Work through these in order — the first three are binary and take a minute each.

### 1. Is it enabled?

- LIO-SAM: `loopClosureEnableFlag: true`. It is `false` in some forks.
- Cartographer: `POSE_GRAPH.optimize_every_n_nodes` must not be 0.
- RTAB-Map: on a LiDAR-only setup, `RGBD/ProximityBySpace` must be `true`.
  RTAB-Map's default loop closure is appearance-based over camera features; with
  no camera, that machinery does nothing and proximity detection is the only
  mechanism you have.

### 2. Is the search radius bigger than your drift?

If you have accumulated 20 m of drift and the search radius is 15 m, the true
match is outside the window. Loop closure is not failing — it is never being
offered the right candidate.

```python
from slamkit.drift import revisit_consistency
print(revisit_consistency(poses, times, radius=30.0, min_time_gap_s=60.0))
```

Measure the drift, then set:

- LIO-SAM: `historyKeyframeSearchRadius` above it.
- Cartographer: `POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window`.
- RTAB-Map: `RGBD/LocalRadius`.

In Cartographer 3D, also check `linear_z_search_window`. If you have z drift,
candidates that match perfectly in x and y get rejected on height alone. The two
symptoms — z drift and no loop closure — are the same bug.

### 3. Is the acceptance threshold impossible?

- LIO-SAM: `historyKeyframeFitnessScore` is a mean squared distance, so **lower
  is stricter**. 0.1 rejects everything; 0.3 is a normal starting point.
- Cartographer: `POSE_GRAPH.constraint_builder.min_score`, higher is stricter.
- RTAB-Map: `Icp/CorrespondenceRatio`, higher is stricter.

Loosen it in small steps and watch the constraint list. **Err on the strict
side**: a false loop closure folds the map onto itself in a single optimisation
step and is unrecoverable.

### 4. Are the candidates even being sampled?

Cartographer's `constraint_builder.sampling_ratio` defaults to 0.3 in 2D and
0.03 in 3D. On a small map the true pair may simply never be drawn. Raise it and
watch the CPU.

### 5. Are the submaps blurry?

If local drift accumulates *inside* one submap, the submap comes out smeared and
nothing can match it. Lower `TRAJECTORY_BUILDER_*.submaps.num_range_data` before
touching any loop-closure parameter. A blurred submap looks like a loop-closure
problem and is a local-SLAM problem.

---

## Symptom: works with the bag, fails live

The single most diagnostic symptom in the whole list, because it rules out
almost everything. The algorithm, the parameters and the geometry are identical.
What changed is *timing*.

### Cause 1: the driver stamps with `now()` in the callback

`header.stamp` is not the sensor clock — it is the time the message finished
arriving, which includes USB or Ethernet scheduling. When you record a bag, that
latency is baked into the stamps. Replay it and everything is consistent with
itself. Run live on a different machine, or under different load, and the
latency is different.

**Check:**

```python
from slamkit.timesync import detect_receive_time_mismatch
print(detect_receive_time_mismatch(sensor_stamps, receive_stamps, expected_rate_hz=10.0))
```

`stamped_on_receive: True` means the sensor stamps track the receive stamps
exactly while both jitter — the definition of `now()` stamping.

You can also spot it from the stamps alone: a spinning LiDAR's motor is stable
to well under 1% of its nominal period. Several milliseconds of jitter on a
10 Hz topic is not a motor, it is a scheduler.

```python
from slamkit.timesync import detect_jitter
print(detect_jitter(scan_times, expected_rate_hz=10.0)["jitter_std_ms"])
```

**Fix:** configure the sensor's own clock.

- Ouster: `timestamp_mode: TIME_FROM_PTP_1588` (or `TIME_FROM_SYNC_PULSE_IN`).
  `TIME_FROM_INTERNAL_OSC` starts at zero on power-up, which puts your stamps in
  1970 and breaks every TF lookup.
- Velodyne: feed it PPS + NMEA.
- Livox: set the driver's timestamp source and check it against the host clock.

### Cause 2: `use_sim_time`

Replaying with `--clock` and `use_sim_time: true` gives every node the same
monotone clock. Live, they use the system clock, which NTP may step mid-run.
A backwards step makes most estimators silently discard data.

```python
from slamkit.timesync import detect_non_monotonic
print(detect_non_monotonic(scan_times))
```

### Cause 3: QoS

Bag playback often publishes with different QoS than the driver. A `BEST_EFFORT`
publisher and a `RELIABLE` subscriber will not connect at all in ROS 2 — the
subscription exists, the topic lists, and no message ever arrives. Check with
`ros2 topic info -v`.

### Cause 4: CPU

The bag plays on a workstation; live runs on a Jetson. If the mapping thread
misses real time, scans queue up and the estimator extrapolates. Symptoms look
like an IMU problem. Check the queue and the actual processing rate before
blaming anything else.

---

## Symptom: the point cloud looks smeared

Individual walls look like they were painted with a wide brush. The smear is
worse when moving faster, and worse at longer range.

**Cause:** motion distortion — the sweep is not being deskewed. A spinning LiDAR
samples over ~100 ms; if the sensor moved during that window, every point was
measured from a different pose. At 2 m/s, that is 20 cm of smear.

Deskewing needs three things, and any one of them missing breaks it silently:

1. **A per-point timestamp field in the cloud.** Not all drivers publish one.
2. **The correct interpretation of that field.** There is no standard:

   | sensor | field | type | units | datum |
   |---|---|---|---|---|
   | Velodyne | `time` | float32 | seconds | relative to sweep start (sometimes negative) |
   | Ouster | `t` | uint32 | **nanoseconds** | relative to sweep start |
   | Hesai | `timestamp` | float64 | seconds | **absolute** UNIX epoch |
   | Livox (PointCloud2) | `timestamp` | float64 | seconds | **absolute** UNIX epoch |
   | Livox (CustomMsg) | `offset_time` | uint32 | nanoseconds | relative to `timebase` |

3. **A pose estimate over the sweep**, which is what the IMU is for.

**Check:**

```python
from slamkit.cloud import detect_timestamp_format
print(detect_timestamp_format(cloud_time_field, scan_period=0.1))
```

It reports the units and datum from the magnitude and span of the values, and
tells you when the field is constant — a driver that is not populating it, in
which case your deskewing stage is doing nothing at all and reporting success.

**Fix:**

- LIO-SAM: `sensor: velodyne | ouster | livox` selects the field handling. Set
  it correctly. This is not cosmetic.
- Cartographer: `num_subdivisions_per_laser_scan`. With 1, the whole sweep is
  treated as instantaneous. 10 is a good value for a 10 Hz sensor.
- If the field genuinely is not there, synthesise it from azimuth — the sensor
  spins at a known rate, so the angle around the sweep *is* the time.

**Related failure:** deskewing to the wrong reference instant. LIO-SAM deskews
to sweep start and stamps accordingly. If a downstream node assumes
end-of-sweep, you get a constant one-sweep lag between the map and the robot.

---

## Symptom: the map slides in a corridor

Long featureless corridor, tunnel, warehouse aisle or open field. The walls stay
crisp — the map looks *good* — but the reported position creeps, and when you
reach a junction the map snaps back with a jump.

**Cause:** geometric degeneracy. This is not a bug and not a tuning failure. The
environment genuinely does not constrain that degree of freedom.

Point-to-plane scan matching solves a system whose translation block is
`H = Σ nᵢnᵢᵀ` over the matched surface normals. In a straight corridor every
normal points at a wall (±Y) or the floor and ceiling (±Z). **Nothing points
along the corridor.** `H` has a near-zero eigenvalue along X, the solver is free
to slide, and it will — by exactly as much as your motion prior allows.

A round tunnel is worse: the smooth cross-section also removes the constraint on
roll about the tunnel axis.

**Check:**

```python
from slamkit.degeneracy import analyze_degeneracy
r = analyze_degeneracy(points, voxel_size=0.3)
print(r.environment, r.weakest_axis, r.condition_number)
print(r.explanation)
```

`voxel_size` matters here. Raw LiDAR clouds are wildly non-uniform in density —
the ground right under the sensor can carry a third of the points — and that
bias alone will tell you the vertical axis is best-constrained in every scene.
Voxelising equalises the vote.

**Fix**, in order of how well it works:

1. **Give the free axis an independent source of information.** Wheel odometry,
   a velocity prior, GNSS, a barometer for z. This is the real fix. In LIO-SAM,
   that means trusting IMU preintegration more; in Cartographer, raising
   `ceres_scan_matcher.translation_weight`.
2. **Detect the degeneracy online and project the update out of the weak
   direction** (Zhang & Singh's solution remapping). Correct, and more work.
3. **Change what the sensor sees.** Widen `lidarMaxRange` so the far end of the
   corridor stays in view. Tilt the sensor so the floor and ceiling contribute.
   Keep doorways and wall breaks in the field of view.
4. **Accept bounded drift and close the loop at the exit.**

What does **not** work: tuning the scan matcher. There is no information in the
data to recover; a different weight just changes which wrong answer you get.

---

## Symptom: the map is mirrored

Turning left in the real world moves the robot right in RViz. In a symmetric
room this is nearly invisible until you try to navigate.

**Cause:** a left-handed transform somewhere — `det(R) = −1`. Usually one of:

- Someone negated an axis "to make it look right".
- A vendor drawing with Z down was transcribed into a Z-up REP-103 frame.
- A 2D LiDAR whose scan direction convention is reversed (RPLIDAR spins
  clockwise; ROS LaserScan angles increase counter-clockwise).

**Check:**

```python
from slamkit.extrinsics import check_handedness
print(check_handedness(R))
```

**Why it survives:** a mirrored map is *self-consistent*. Loop closure still
fires. Scan matching still converges. Everything looks healthy except the
relationship to the real world.

**Fix:** find and remove the reflection. If a sensor really is mounted upside
down, express that as a proper 180° roll (`Rx(π)`, which has `det = +1`), not as
a sign flip. For a mirrored 2D scan, fix the driver's `inverted` parameter —
do not compensate downstream with a negated matrix.

---

## Symptom: SLAM node starts and does nothing

No error, no crash, no map. The node is running.

Work down this list; each item takes under a minute.

1. **Is the topic name right?** Subscribing to a topic nobody publishes is legal
   in ROS 2 and produces no warning. `ros2 topic list`, then check that a launch
   file remap is not overriding your parameter.

2. **Is the QoS compatible?** `ros2 topic info -v`. A `BEST_EFFORT` publisher
   and a `RELIABLE` subscriber never connect.

3. **Is the parameter file even loaded?** LIO-SAM's ROS 2 branch nests
   parameters under `/**: ros__parameters:`. A mis-nested YAML leaves every
   parameter at its compiled-in default and says nothing.
   `ros2 param dump /lio_sam` to confirm.

4. **Is TF complete?**

   ```python
   from slamkit.rosbag_check import check_tf_completeness
   for f in check_tf_completeness(edges, [("base_link", "velodyne"),
                                          ("base_link", "imu_link")]):
       print(f)
   ```

   Also check for two parents on one frame — the classic being a
   `robot_state_publisher` and a hand-written `static_transform_publisher` for
   the same joint. Symptom is TF warnings and a robot that jitters between two
   positions.

5. **Is it waiting for a message that will never come?**
   - Cartographer 3D **requires** an IMU. With `use_imu_data = true` and no IMU
     topic it blocks forever.
   - RTAB-Map with `subscribe_depth: true` and no depth camera waits for a
     synchronised set that cannot be formed. Nothing is processed.

6. **Are the frame_ids populated?** An empty `header.frame_id` means the node
   cannot look up any transform for that data. RViz says
   `Frame [] does not exist`.

7. **Is `use_sim_time` consistent?** If some nodes use sim time and others do
   not, message timestamps are decades apart and every synchroniser drops
   everything.

---

## The 20-minute triage

What to do when a bag lands on your desk with "SLAM doesn't work".

```bash
# 0. What is actually in the bag?
ros2 bag info my_bag > baginfo.txt
slam-doctor --bag baginfo.txt

# 1. Everything at once, on exported CSVs
slam-doctor --bag baginfo.txt \
            --trajectory est_traj.csv \
            --imu imu.csv \
            --cloud one_scan.csv \
            --extrinsic extrinsic.json \
            --tf-chain base_link:velodyne \
            --tf-chain base_link:imu_link
```

Then work the ranked list from the top. The output is ordered so that the first
item, once fixed, usually makes several of the others disappear.

Manual order if you are doing it by hand:

1. **Topics and rates.** LiDAR at its nominal rate? IMU at ≥ 200 Hz? Gaps?
2. **TF.** All chains resolve? One parent per frame? No static/dynamic conflict?
3. **Timestamps.** Monotonic? Jitter under 1% of the period? Sensor clock or
   `now()`?
4. **LiDAR/IMU offset.** Under a few ms? Constant over the run?
5. **Extrinsics.** Rotation direction verified against data? Lever arm sane?
6. **Cloud contents.** Ring field present? Per-point time field present, and in
   the units the pipeline expects?
7. **Geometry.** Is the environment degenerate where the failure happens?
8. **Only now:** parameters.

---

## What this guide does not cover

- Multi-LiDAR calibration and extrinsic estimation between two LiDARs.
- GNSS/RTK fusion and datum handling.
- Dynamic-object rejection in crowded scenes.
- Real-time performance tuning on specific embedded hardware.

Those are real problems; they are just not the ones that account for most
failures, and they need per-project work rather than a checklist.

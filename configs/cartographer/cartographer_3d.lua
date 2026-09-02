-- ============================================================================
-- Cartographer 3D -- annotated reference configuration
-- ============================================================================
-- For a 3D spinning LiDAR (VLP-16 / OS1-32 / OS1-64) plus a mandatory IMU.
--
-- READ THIS FIRST: 3D Cartographer REQUIRES an IMU. There is no option to run
-- without one. If your IMU is missing, badly stamped, or in the wrong frame,
-- nothing else in this file will help. Work in this order:
--   1. TF: tracking_frame exists and connects to every sensor frame.
--   2. Time: sensor-clock stamps, offset under a few ms, monotonic.
--   3. Extrinsics: the LiDAR->IMU mount is right.
--   4. Only then: the parameters below.
--
-- Load with:
--   ros2 run cartographer_ros cartographer_node \
--     -configuration_directory <this dir> \
--     -configuration_basename cartographer_3d.lua
-- ============================================================================

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  map_frame = "map",

  -- MUST be the IMU frame in 3D. Cartographer places the IMU at the origin of
  -- tracking_frame and does not model an offset to it.
  -- WRONG: gravity is expressed in the wrong frame, the whole map tilts, and z
  -- drifts in proportion to horizontal distance travelled. That specific
  -- signature -- z proportional to distance, not to time -- is what
  -- slamkit.drift.analyze_z_drift() calls "attitude / gravity misalignment".
  tracking_frame = "imu_link",

  published_frame = "base_link",
  odom_frame = "odom",

  -- In 3D, Cartographer normally provides the odom frame itself because its
  -- local SLAM result is better than wheel odometry.
  provide_odom_frame = true,

  -- Never flatten in 3D.
  publish_frame_projected_to_2d = false,

  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,

  num_laser_scans = 0,
  num_multi_echo_laser_scans = 0,

  -- One PointCloud2 topic (/points2). Set to 2 if you have two LiDARs; they
  -- must be on /points2 and /points2_2.
  -- WRONG (0): nothing is subscribed and the node sits idle producing no map.
  num_point_clouds = 1,

  -- Split each sweep into this many pieces, each motion-compensated with its
  -- own interpolated pose. THIS IS CARTOGRAPHER'S DESKEWING KNOB.
  -- WRONG (1 with a 10 Hz LiDAR on a moving platform): the whole 100 ms sweep
  -- is treated as instantaneous. Walls come out curved or doubled during turns,
  -- and it looks exactly like a bad extrinsic.
  -- 10 is a good default for a 10 Hz sensor; raise it for faster motion at the
  -- cost of CPU.
  num_subdivisions_per_laser_scan = 10,

  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,

  rangefinder_sampling_ratio = 1.0,
  odometry_sampling_ratio = 1.0,
  fixed_frame_pose_sampling_ratio = 1.0,
  imu_sampling_ratio = 1.0,
  landmarks_sampling_ratio = 1.0,
}

MAP_BUILDER.use_trajectory_builder_3d = true
MAP_BUILDER.num_background_threads = 7

-- ----------------------------------------------------------------------------
-- Local SLAM
-- ----------------------------------------------------------------------------
-- Range gating, metres.
-- WRONG (min too small): the robot's own body is mapped as structure that
-- moves with it. In 3D this is worse than in 2D because a roof rack or a mast
-- gives a large, well-shaped surface for the matcher to lock onto.
TRAJECTORY_BUILDER_3D.min_range = 1.0
-- WRONG (max too small): you delete the distant geometry that constrains yaw
-- and along-corridor position.
TRAJECTORY_BUILDER_3D.max_range = 60.0

-- Free space is carved out to this range along beams that returned nothing.
TRAJECTORY_BUILDER_3D.missing_data_ray_length = 5.0

-- 3D SLAM matches against two resolutions: a coarse map for robustness and a
-- fine one for accuracy. These are the input filters for each.
-- WRONG (high_resolution too coarse): fine structure is lost and the fine
-- matching stage stops adding anything.
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.max_length = 2.0
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.min_num_points = 150
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.max_range = 15.0
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.max_length = 4.0
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.min_num_points = 200
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.max_range = 60.0

-- The brute-force pre-matcher. Expensive in 3D; leave it off unless the
-- estimate is jumping.
-- WRONG (on, with a large search window, on a Jetson): local SLAM misses real
-- time, the queue grows, and the pose lags further and further behind.
TRAJECTORY_BUILDER_3D.use_online_correlative_scan_matching = false
TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher.linear_search_window = 0.15
TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher.angular_search_window = math.rad(1.0)
TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 1e-1
TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1e-1

-- Ceres matcher weights: relative trust between the two map resolutions and
-- the IMU-driven motion prior.
-- WRONG (translation_weight too low in a tunnel or corridor): nothing holds
-- the pose along the free axis and the map slides. Raising it is a mitigation,
-- not a fix -- confirm the degeneracy first with slamkit.degeneracy.
-- WRONG (rotation_weight too high): the matcher cannot correct an IMU yaw
-- error, so heading drift accumulates unchecked.
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.occupied_space_weight_0 = 1.0   -- high-res map
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.occupied_space_weight_1 = 6.0   -- low-res map
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.translation_weight = 5.0
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.rotation_weight = 4e2
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.ceres_solver_options.max_num_iterations = 12

-- Motion filter: skip inserting a scan if the platform barely moved.
TRAJECTORY_BUILDER_3D.motion_filter.max_time_seconds = 0.5
TRAJECTORY_BUILDER_3D.motion_filter.max_distance_meters = 0.1
TRAJECTORY_BUILDER_3D.motion_filter.max_angle_radians = math.rad(0.5)

-- Time constant of the gravity low-pass in the IMU tracker, seconds.
-- Larger = slower to trust the accelerometer, so vehicle acceleration is less
-- likely to be mistaken for a change in "down".
-- WRONG (too small on a car or drone): braking and cornering accelerations are
-- interpreted as tilt, the map pitches, and z drifts on every acceleration.
-- WRONG (too large): a genuine tilt takes tens of seconds to be recognised, so
-- driving up a ramp bends the map.
TRAJECTORY_BUILDER_3D.imu_gravity_time_constant = 10.0

-- Rotational histogram used to propose orientations for loop closure.
TRAJECTORY_BUILDER_3D.rotational_histogram_size = 120

-- Submap size and resolutions, metres.
-- WRONG (num_range_data too large): the submap accumulates internal drift and
-- comes out blurred, so loop closure cannot match it. Blurry submaps are the
-- usual reason 3D loop closure "never fires".
TRAJECTORY_BUILDER_3D.submaps.num_range_data = 160
TRAJECTORY_BUILDER_3D.submaps.high_resolution = 0.10
TRAJECTORY_BUILDER_3D.submaps.low_resolution = 0.45
TRAJECTORY_BUILDER_3D.submaps.range_data_inserter.hit_probability = 0.55
TRAJECTORY_BUILDER_3D.submaps.range_data_inserter.miss_probability = 0.49
-- Points closer than this are inserted into the high-resolution map only.
TRAJECTORY_BUILDER_3D.high_resolution_range = 15.0

-- ----------------------------------------------------------------------------
-- Global SLAM
-- ----------------------------------------------------------------------------
-- WRONG (0): global optimisation and therefore loop closure are disabled.
POSE_GRAPH.optimize_every_n_nodes = 320

-- 3D constraint search is expensive, so the default sampling ratio is tiny.
-- WRONG (left at the default on a small indoor map): the true loop pair is
-- rarely sampled and loop closure appears not to work. Raise it to 0.03-0.3 on
-- small maps and watch the CPU.
POSE_GRAPH.constraint_builder.sampling_ratio = 0.03

POSE_GRAPH.constraint_builder.min_score = 0.62
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.66

-- Search window for loop candidates. Must exceed your accumulated drift or the
-- true match is never inside it.
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d.linear_xy_search_window = 5.0
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d.linear_z_search_window = 1.0
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d.angular_search_window = math.rad(15.)
-- WRONG (z window too small when you have z drift): every loop candidate is
-- rejected on height alone, even though x and y match perfectly. If z drift is
-- your complaint AND loop closure never fires, this is the connection.
POSE_GRAPH.constraint_builder.ceres_scan_matcher_3d.occupied_space_weight_0 = 5.0
POSE_GRAPH.constraint_builder.ceres_scan_matcher_3d.occupied_space_weight_1 = 30.0
POSE_GRAPH.constraint_builder.ceres_scan_matcher_3d.translation_weight = 10.0
POSE_GRAPH.constraint_builder.ceres_scan_matcher_3d.rotation_weight = 1.0

-- Trust in the local SLAM result inside the global optimisation.
POSE_GRAPH.optimization_problem.local_slam_pose_translation_weight = 5.0
POSE_GRAPH.optimization_problem.local_slam_pose_rotation_weight = 5.0
POSE_GRAPH.optimization_problem.odometry_translation_weight = 0.0
POSE_GRAPH.optimization_problem.odometry_rotation_weight = 0.0

-- How hard the optimiser pulls the estimate towards the IMU's gravity
-- direction. This is the parameter that anchors z.
-- WRONG (too low): nothing stops the map tilting and z ramps steadily upward.
-- WRONG (too high on a dynamic platform): acceleration is read as tilt.
POSE_GRAPH.optimization_problem.acceleration_weight = 1.1e2
POSE_GRAPH.optimization_problem.rotation_weight = 1.6e4

-- Huber loss on loop constraints: the guard against one false loop closure
-- folding the map.
POSE_GRAPH.optimization_problem.huber_scale = 5e2
POSE_GRAPH.max_num_final_iterations = 200

return options

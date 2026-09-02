-- ============================================================================
-- Cartographer 2D -- annotated reference configuration
-- ============================================================================
-- For a differential-drive robot with a single 2D LiDAR (RPLIDAR / Hokuyo /
-- Sick), wheel odometry, and optionally a 6-axis IMU.
--
-- Every option carries what it does and the symptom of setting it wrong.
--
-- Load it with:
--   ros2 run cartographer_ros cartographer_node \
--     -configuration_directory <this dir> \
--     -configuration_basename cartographer_2d.lua
--
-- The single most common cause of "Cartographer just does not start" is a TF
-- problem, not a tuning problem: tracking_frame must exist and must connect to
-- the frame every sensor message is stamped with, at the time the message was
-- stamped. Check `ros2 run tf2_tools view_frames` FIRST.
-- ============================================================================

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  -- Frame Cartographer publishes the global (loop-closed) pose in.
  map_frame = "map",

  -- THE FRAME EVERYTHING IS TRACKED IN. All sensor data is transformed into
  -- this frame before use.
  -- If you use an IMU this MUST be the IMU frame -- Cartographer assumes the
  -- IMU is at the origin of tracking_frame and does not model a lever arm to
  -- it. Without an IMU, "base_link" is the normal choice.
  -- WRONG: with imu_data enabled and tracking_frame = "base_link", the gravity
  -- vector is rotated by the base->imu transform. The map tilts, and in 2D that
  -- shows up as scan ranges that shrink as the robot rolls -- walls appear to
  -- breathe in and out.
  tracking_frame = "imu_link",

  -- The frame whose pose Cartographer publishes. Set it to the frame your
  -- odometry publishes as its child (usually base_link), so that Cartographer
  -- publishes map->odom and does not fight the odometry publisher.
  -- WRONG: if you set this to base_link while also setting provide_odom_frame
  -- = false and something else publishes odom->base_link, you get two parents
  -- for base_link and TF_MULTIPLE_PARENTS warnings; the robot visibly jitters
  -- between two positions.
  published_frame = "base_link",

  odom_frame = "odom",

  -- Have Cartographer publish odom_frame itself (local, non-loop-closed pose).
  -- Set false when the robot's own wheel odometry already publishes odom.
  -- WRONG (true when something else also publishes odom->base_link): two
  -- publishers, one child. See above.
  provide_odom_frame = false,

  -- Flatten the published transform to 2D (zero z, roll, pitch).
  -- Set true for 2D SLAM on a wheeled robot; navigation stacks expect it.
  publish_frame_projected_to_2d = true,

  -- Subscribe to a NavSatFix topic for global constraints.
  use_nav_sat = false,

  -- Subscribe to landmark observations (fiducials).
  use_landmarks = false,

  -- Number of sensor_msgs/LaserScan topics. 1 for a single 2D LiDAR.
  -- WRONG (0 with a scan connected): Cartographer subscribes to nothing and
  -- silently does nothing. The node runs, the map stays empty.
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,

  -- Split each incoming scan into this many pieces before insertion. Raise it
  -- when the sensor spins slowly relative to the robot's motion, so each piece
  -- can be motion-compensated with its own pose.
  -- WRONG (1 on a fast-rotating robot with a 5 Hz LiDAR): the scan is inserted
  -- as if it were instantaneous. Walls come out curved on every turn.
  num_subdivisions_per_laser_scan = 1,

  -- Number of PointCloud2 topics. 0 in a pure 2D LaserScan setup.
  num_point_clouds = 0,

  -- How long to wait for a transform before giving up, seconds.
  -- WRONG (too small): messages are dropped with "Dropped ... because
  -- transform ... is not available", and the map fills in only sporadically.
  -- Raise it to 0.2-0.5 if your TF publisher is slow or jittery. Raising it
  -- does not fix a MISSING transform, only a LATE one.
  lookup_transform_timeout_sec = 0.2,

  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,      -- 200 Hz
  trajectory_publish_period_sec = 30e-3,

  -- Fraction of each input stream actually used. 1.0 = use everything.
  -- Lower these only when CPU-bound, and lower rangefinder_sampling_ratio last
  -- -- range data is the thing that constrains the map.
  rangefinder_sampling_ratio = 1.0,
  odometry_sampling_ratio = 1.0,
  fixed_frame_pose_sampling_ratio = 1.0,
  imu_sampling_ratio = 1.0,
  landmarks_sampling_ratio = 1.0,
}

-- ----------------------------------------------------------------------------
-- Global build settings
-- ----------------------------------------------------------------------------
-- 2D SLAM.
MAP_BUILDER.use_trajectory_builder_2d = true

-- Background threads for the pose-graph optimiser.
-- WRONG (more than physical cores): optimisation competes with the scan
-- matcher for CPU and the local pose starts lagging.
MAP_BUILDER.num_background_threads = 4

-- ----------------------------------------------------------------------------
-- Local SLAM: how each scan is matched and inserted
-- ----------------------------------------------------------------------------
-- Use IMU data. In 2D this is optional; when true, tracking_frame MUST be the
-- IMU frame and the IMU MUST be publishing.
-- WRONG (true with no /imu topic): Cartographer blocks forever waiting for IMU
-- data and never produces a single submap. The log line is easy to miss.
TRAJECTORY_BUILDER_2D.use_imu_data = true

-- Range gating, metres. Set min above the robot's own footprint.
-- WRONG (min too small): the robot's own chassis is scanned as a static
-- obstacle that moves with it; the scan matcher locks onto it and the map stops
-- tracking the robot's motion.
TRAJECTORY_BUILDER_2D.min_range = 0.15
-- WRONG (max too small): in a corridor you cut off the far wall that provides
-- the only along-corridor constraint. See slamkit.degeneracy.
TRAJECTORY_BUILDER_2D.max_range = 12.0

-- Beams that return nothing are inserted as free space out to this distance.
-- WRONG (too large): glass and dark surfaces produce misses, and those misses
-- carve free space through real walls -- the wall disappears from the map.
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 5.0

-- Vertical slab of a 3D cloud kept when feeding 3D data into 2D SLAM. Ignored
-- for a true 2D LaserScan.
TRAJECTORY_BUILDER_2D.min_z = -0.8
TRAJECTORY_BUILDER_2D.max_z = 2.0

-- Voxel size applied to incoming range data, metres.
-- WRONG (too large): thin features vanish and matching becomes unconstrained
-- in open areas.
TRAJECTORY_BUILDER_2D.voxel_filter_size = 0.025

-- Adaptive voxel filter used specifically for scan matching.
-- max_length is the coarsest voxel it will grow to; min_num_points is the point
-- count it tries to keep.
-- WRONG (min_num_points too low): matching runs on too few points and produces
-- confident but wrong poses, usually visible as sudden 90 degree flips in a
-- symmetric room.
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_length = 0.5
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.min_num_points = 200
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_range = 50.0

-- Real-time correlative scan matcher: a brute-force search that gives the Ceres
-- matcher a good initial guess.
-- Turn it ON when you have no odometry, when odometry is poor, or when the
-- robot moves fast. It costs CPU.
-- WRONG (off with bad odometry): the Ceres matcher starts from a bad guess,
-- falls into a local minimum, and the map jumps a whole wall spacing.
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.)
-- Cost multipliers: how much the search penalises deviating from the prior.
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 1e-1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1e-1

-- Ceres scan matcher weights. These are RELATIVE trust between the scan match
-- and the motion prior (odometry + IMU).
-- occupied_space_weight: trust in the range data.
-- translation_weight / rotation_weight: trust in the prior.
-- WRONG (translation_weight too low in a corridor): nothing holds the pose
-- along the corridor axis and the robot slides. Raising it is the standard
-- corridor mitigation -- but only after you have confirmed the geometry really
-- is degenerate, because raising it also makes the map ignore real corrections.
-- WRONG (occupied_space_weight too high): the scan matcher overrides good
-- odometry and the map twitches on every noisy scan.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.occupied_space_weight = 1.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.max_num_iterations = 20

-- Motion filter: a scan is dropped if the robot has not moved enough since the
-- last inserted scan. Prevents the submap from being burned in while parked.
-- WRONG (thresholds too large): real motion is discarded and the trajectory
-- becomes a staircase.
-- WRONG (too small): a stationary robot keeps inserting the same scan, the
-- submap saturates, and matching becomes over-confident.
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 5.0
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.1
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(1.0)

-- Gravity constant used by the IMU tracker.
TRAJECTORY_BUILDER_2D.imu_gravity_time_constant = 10.0

-- How many scans go into one submap. A submap is the unit of loop closure.
-- WRONG (too large): the submap accumulates local drift internally and comes
-- out blurred, so loop closure cannot match it. This is the classic cause of
-- "the map is blurry and loop closure never fires" -- lower it before touching
-- the loop-closure parameters.
-- WRONG (too small): huge numbers of submaps, slow optimisation, and weak
-- individual constraints.
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 90
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05
TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter.hit_probability = 0.55
TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter.miss_probability = 0.49

-- ----------------------------------------------------------------------------
-- Global SLAM: the pose graph and loop closure
-- ----------------------------------------------------------------------------
-- Run the global optimisation every N nodes. 0 disables global SLAM entirely.
-- WRONG (0): no loop closure at all, ever. If loop closure "never fires", check
-- this before anything else.
-- WRONG (too small): the optimiser runs constantly and the map visibly jumps.
POSE_GRAPH.optimize_every_n_nodes = 90

-- Fraction of node/submap pairs actually tested for a loop constraint.
-- WRONG (too low, e.g. 0.003 on a small map): the true loop pair is simply
-- never sampled and loop closure never fires. Raise it to 0.3 on small maps.
POSE_GRAPH.constraint_builder.sampling_ratio = 0.3

-- Minimum matching score to accept a loop constraint, 0..1.
-- WRONG (too high, e.g. 0.75): no loop is accepted; the map drifts forever.
-- WRONG (too low, e.g. 0.4): a false constraint is accepted between two
-- similar-looking corridors and the pose graph folds the map in half. This is
-- unrecoverable in one optimisation step -- always err high, then lower it
-- slowly while watching the constraint list in RViz.
POSE_GRAPH.constraint_builder.min_score = 0.62
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.66

-- How far from the current pose loop candidates are searched.
-- WRONG (smaller than your accumulated drift): the true match is outside the
-- window and loop closure genuinely cannot fire. Measure the drift first --
-- slamkit.drift.revisit_consistency() -- and set this above it.
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 7.0
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(30.)
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.branch_and_bound_depth = 7

-- Relative weights in the global optimisation.
-- Raising the odometry weights tells the optimiser to trust wheel odometry
-- more than loop closures -- correct on a robot with good encoders in a
-- feature-poor building, wrong almost everywhere else.
POSE_GRAPH.optimization_problem.odometry_translation_weight = 1e5
POSE_GRAPH.optimization_problem.odometry_rotation_weight = 1e5
POSE_GRAPH.optimization_problem.local_slam_pose_translation_weight = 1e5
POSE_GRAPH.optimization_problem.local_slam_pose_rotation_weight = 1e5

-- Huber loss scale for loop constraints: how hard an outlier constraint is
-- down-weighted.
-- WRONG (too large): one false loop closure drags the whole map.
POSE_GRAPH.optimization_problem.huber_scale = 1e1

-- Refuse constraints whose residual exceeds these after matching. This is the
-- safety net against a false loop closure destroying the map.
POSE_GRAPH.constraint_builder.max_constraint_distance = 15.0
POSE_GRAPH.max_num_final_iterations = 200

return options

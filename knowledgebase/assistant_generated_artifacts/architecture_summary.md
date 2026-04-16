# UAV Navigation System — ROS2 Architecture Summary

---

## Custom Message Types (`uav_nav_msgs` package)

| Message | Fields |
|---|---|
| `VioStatus.msg` | `Header header`, `int8 status` (TRACKING=0, DEGRADED=1, LOST=2), `int32 tracked_features` |
| `LocalizationMode.msg` | `Header header`, `int8 mode` (VIO_ACTIVE=0, TRANSITIONING=1, DR_ONLY=2) |
| `LocalizationHealth.msg` | `Header header`, `int8 health` (OK=0, DEGRADED=1, CRITICAL=2) |
| `MissionState.msg` | `Header header`, `int8 state` (PREFLIGHT=0, PLANNING=1, TAKEOFF=2, NAVIGATE=3, REPLANNING=4, CLIMB_FOR_VIEW=5, LAND_APPROACH=6, PRECISION_LAND=7, LANDED=8, EMERGENCY=9) |
| `ReplanReason.msg` | `Header header`, `int8 reason` (OBSTACLE_DETECTED=0, DEVIATION_EXCEEDED=1, LOC_DEGRADED=2, BUDGET_EXCEEDED=3) |
| `PathConstraints.msg` | `Header header`, `geometry_msgs/PoseStamped[] waypoints`, `float32[] max_imu_only_duration_s` |
| `LandmarkArray.msg` | `Header header`, `geometry_msgs/PoseStamped[] positions`, `float32[] match_confidence` |

---

## Nodes

---

### SENSORS

---

#### `camera_driver`
**Package:** `uav_sensors`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| OUT | `/sensors/camera/image_raw` | `sensor_msgs/Image` | Raw frames @ 30 Hz |
| OUT | `/sensors/camera/camera_info` | `sensor_msgs/CameraInfo` | Intrinsic matrix K, distortion coeffs |

**Key params:** `camera_calibration_file`, `frame_rate`, `resolution`

---

#### `imu_driver`
**Package:** `uav_sensors`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| OUT | `/sensors/imu/data` | `sensor_msgs/Imu` | Linear acceleration + angular velocity @ ~200 Hz |

**Key params:** `gyro_bias_file`, `accel_bias_file`, `publish_rate_hz`

---

#### `georef_map_loader`
**Package:** `uav_mapping`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| OUT | `/map/georef_image` | `sensor_msgs/Image` | Published once at startup |
| OUT | `/map/georef_metadata` | `uav_nav_msgs/GeoRefMetadata` | Origin, scale, coordinate frame |

**Key params:** `georef_image_path`, `coordinate_frame`, `resolution_m_per_pixel`

---

### LOCALIZATION

---

#### `vio_node`
**Package:** `uav_localization` | **Wraps:** OpenVINS

| Direction | Topic | Type | Notes |
|---|---|---|---|
| IN | `/sensors/camera/image_raw` | `sensor_msgs/Image` | Feature extraction and tracking |
| IN | `/sensors/camera/camera_info` | `sensor_msgs/CameraInfo` | Required for 3D back-projection |
| IN | `/sensors/imu/data` | `sensor_msgs/Imu` | EKF prediction step at 200 Hz |
| OUT | `/localization/vio/pose` | `geometry_msgs/PoseWithCovarianceStamped` | 6-DOF pose + covariance |
| OUT | `/localization/vio/status` | `uav_nav_msgs/VioStatus` | TRACKING / DEGRADED / LOST |
| OUT | `/localization/vio/feature_count` | `std_msgs/Int32` | Tracked features — confidence proxy |

**Key params:** `camera_calibration_file`, `imu_noise_model`, `min_feature_count_threshold`

---

#### `imu_dead_reckoning_node`
**Package:** `uav_localization`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| IN | `/sensors/imu/data` | `sensor_msgs/Imu` | Sole input during VIO loss |
| IN | `/localization/vio/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Initialization only — last known pose when VIO drops |
| OUT | `/localization/dr/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Dead-reckoned pose, covariance grows over time |
| OUT | `/localization/dr/drift_estimate` | `std_msgs/Float32` | Accumulated drift in meters |

**Key params:** `imu_accel_noise_density`, `imu_gyro_noise_density`, `gravity_vector`

---

#### `state_estimator_node`
**Package:** `uav_localization`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| IN | `/localization/vio/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Primary source when VIO healthy |
| IN | `/localization/vio/status` | `uav_nav_msgs/VioStatus` | Triggers mode switching |
| IN | `/localization/dr/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Fallback source when VIO lost |
| IN | `/map/georef_landmarks` | `uav_nav_msgs/LandmarkArray` | Absolute position corrections (drift reset) |
| OUT | `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | **Authoritative pose — all nodes subscribe only to this** |
| OUT | `/uav/odometry` | `nav_msgs/Odometry` | Pose + velocity for MPC dynamics |
| OUT | `/uav/localization_mode` | `uav_nav_msgs/LocalizationMode` | Current active mode |

**Key params:** `vio_to_dr_covariance_threshold`, `blend_window_duration_s`, `min_landmark_match_confidence`

---

#### `localization_monitor_node`
**Package:** `uav_localization`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| IN | `/localization/vio/feature_count` | `std_msgs/Int32` | Low count → health warning |
| IN | `/localization/dr/drift_estimate` | `std_msgs/Float32` | Exceeds threshold → emergency flag |
| IN | `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Covariance trace as secondary indicator |
| OUT | `/monitoring/localization_health` | `uav_nav_msgs/LocalizationHealth` | OK / DEGRADED / CRITICAL |
| OUT | `/monitoring/max_drift_exceeded` | `std_msgs/Bool` | Triggers emergency response in mode manager |

**Key params:** `min_features_warning`, `min_features_critical`, `max_allowed_drift_m`

---

### MAPPING

---

#### `feature_density_map_node`
**Package:** `uav_mapping`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| IN | `/map/georef_image` | `sensor_msgs/Image` | Processed at startup |
| IN | `/sensors/camera/image_raw` | `sensor_msgs/Image` | Live updates during flight |
| IN | `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Projects live features to map frame |
| OUT | `/map/feature_density_grid` | `nav_msgs/OccupancyGrid` | 0–100 VIO quality score per cell, used as RRT* cost layer |
| OUT | `/map/georef_landmarks` | `uav_nav_msgs/LandmarkArray` | Matched landmarks for state estimator corrections |

**Key params:** `grid_resolution_m`, `feature_detector (ORB/FAST)`, `min_density_threshold`

---

#### `local_occupancy_node`
**Package:** `uav_mapping`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| IN | `/sensors/camera/image_raw` | `sensor_msgs/Image` | Obstacle detection from live frames |
| IN | `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Projects obstacles into map frame |
| OUT | `/map/local_costmap` | `nav_msgs/OccupancyGrid` | Live obstacle map @ ~5 Hz, ~30m radius around UAV |

**Key params:** `map_radius_m`, `map_resolution_m`, `obstacle_inflation_radius_m`, `decay_time_s`

---

### PLANNING

---

#### `mission_manager_node`
**Package:** `uav_planning`

| Direction | Topic / Interface | Type | Notes |
|---|---|---|---|
| IN | Service: `SetMission` | `uav_nav_msgs/SetMission` (custom) | Called once pre-flight by operator |
| OUT | `/mission/goal` | `geometry_msgs/PoseStamped` | Target destination |
| OUT | Action client: `PlanPath` | `uav_nav_msgs/PlanPath` (custom) | Calls global_planner to initiate planning |

**Key params:** `default_cruise_altitude_m`, `home_position`

---

#### `global_planner_node`
**Package:** `uav_planning` | **Algorithm:** RRT*

| Direction | Topic / Interface | Type | Notes |
|---|---|---|---|
| IN | `/map/feature_density_grid` | `nav_msgs/OccupancyGrid` | VIO quality cost layer |
| IN | `/mission/goal` | `geometry_msgs/PoseStamped` | Destination |
| IN | `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Start node for replanning |
| IN | Action server: `PlanPath` (goal) | `uav_nav_msgs/PlanPath` | Triggers planning |
| OUT | `/planning/global_path` | `nav_msgs/Path` | RRT* waypoint sequence |
| OUT | `/planning/path_feature_budget` | `uav_nav_msgs/PathConstraints` | Max IMU-only duration per segment |
| OUT | Action server: `PlanPath` (feedback) | `uav_nav_msgs/PlanPath` | Planning progress % |

**Key params:** `max_iterations`, `step_size_m`, `goal_bias_prob`, `feature_density_weight`, `path_length_weight`

---

#### `path_manager_node`
**Package:** `uav_planning`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| IN | `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Cross-track error computation |
| IN | `/planning/global_path` | `nav_msgs/Path` | Active path to track |
| IN | `/planning/path_feature_budget` | `uav_nav_msgs/PathConstraints` | Budget enforcement per segment |
| IN | `/map/local_costmap` | `nav_msgs/OccupancyGrid` | Detects new obstacles ahead on path |
| IN | `/monitoring/localization_health` | `uav_nav_msgs/LocalizationHealth` | DEGRADED triggers reroute to feature-rich area |
| OUT | `/planning/current_waypoint` | `geometry_msgs/PoseStamped` | Active waypoint fed to MPC |
| OUT | `/planning/replan_trigger` | `uav_nav_msgs/ReplanReason` | Fires when RRT* replanning is needed |

**Key params:** `max_cross_track_error_m`, `waypoint_reached_radius_m`, `lookahead_distance_m`

---

### CONTROL

---

#### `mpc_node`
**Package:** `uav_control` | **Library:** `do-mpc` or `acados`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| IN | `/uav/odometry` | `nav_msgs/Odometry` | Current state: position + velocity |
| IN | `/planning/current_waypoint` | `geometry_msgs/PoseStamped` | Reference target for tracking |
| IN | `/map/local_costmap` | `nav_msgs/OccupancyGrid` | Obstacle avoidance within prediction horizon |
| IN | `/mission/state` | `uav_nav_msgs/MissionState` | Switches cost function tuning per flight mode |
| OUT | `/control/mpc_setpoint` | `geometry_msgs/TwistStamped` | Velocity commands [vx, vy, vz, yaw_rate] @ 20–50 Hz |
| OUT | `/control/mpc_predicted_trajectory` | `nav_msgs/Path` | Predicted trajectory over horizon (debug/viz) |

**Key params:** `horizon_steps N`, `dt`, `Q matrix (state cost)`, `R matrix (control cost)`, `obstacle_cost_weight`, `uav_mass_kg`

---

#### `flight_mode_manager`
**Package:** `uav_control`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| IN | `/uav/localization_mode` | `uav_nav_msgs/LocalizationMode` | DR_ONLY → CLIMB_FOR_VIEW state |
| IN | `/monitoring/localization_health` | `uav_nav_msgs/LocalizationHealth` | CRITICAL → emergency response |
| IN | `/monitoring/max_drift_exceeded` | `std_msgs/Bool` | → EMERGENCY_HOVER or EMERGENCY_LAND |
| IN | `/planning/replan_trigger` | `uav_nav_msgs/ReplanReason` | → REPLANNING state |
| IN | `/landing/zone_detected` | `std_msgs/Bool` | True + low altitude → PRECISION_LAND |
| OUT | `/mission/state` | `uav_nav_msgs/MissionState` | **Broadcast to ALL nodes @ ~10 Hz** |

**Key params:** `state_transition_timeouts`, `climb_altitude_m`, `precision_land_activation_altitude_m`

---

#### `command_bridge_node`
**Package:** `uav_control`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| IN | `/control/mpc_setpoint` | `geometry_msgs/TwistStamped` | Active during NAVIGATE |
| IN | `/landing/visual_servo_setpoint` | `geometry_msgs/TwistStamped` | **Overrides MPC during PRECISION_LAND** |
| IN | `/mission/state` | `uav_nav_msgs/MissionState` | Determines which input is active |
| OUT | `/mavros/setpoint_velocity/cmd_vel` | `geometry_msgs/TwistStamped` | → MAVROS → PX4 flight controller |

**Key params:** `max_velocity_mps`, `max_acceleration_mps2`, `geofence_polygon`, `command_timeout_ms`

---

### LANDING

---

#### `landing_zone_detector`
**Package:** `uav_landing`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| IN | `/sensors/camera/image_raw` | `sensor_msgs/Image` | Detection runs only in LAND_APPROACH / PRECISION_LAND |
| IN | `/mission/state` | `uav_nav_msgs/MissionState` | Gates processing — inactive during cruise |
| OUT | `/landing/zone_detected` | `std_msgs/Bool` | True when target confidently found |
| OUT | `/landing/zone_pose` | `geometry_msgs/PoseStamped` | Relative pose of target in camera frame |

**Key params:** `detector_type (aruco/learned)`, `aruco_marker_id`, `aruco_marker_size_m`

---

#### `precision_landing_node`
**Package:** `uav_landing`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| IN | `/landing/zone_pose` | `geometry_msgs/PoseStamped` | Error signal for visual servo loop |
| IN | `/uav/odometry` | `nav_msgs/Odometry` | Velocity for derivative term |
| IN | `/mission/state` | `uav_nav_msgs/MissionState` | Only active in PRECISION_LAND |
| OUT | `/landing/visual_servo_setpoint` | `geometry_msgs/TwistStamped` | Velocity corrections → command_bridge, overrides MPC |

**Key params:** `visual_servo_gains_kp`, `visual_servo_gains_kd`, `activation_altitude_m`, `descent_velocity_profile`

---

## Complete Topic Registry

| Topic | Type | Publisher | Subscribers |
|---|---|---|---|
| `/sensors/camera/image_raw` | `sensor_msgs/Image` | `camera_driver` | `vio_node`, `feat_map`, `occ_map`, `land_det` |
| `/sensors/camera/camera_info` | `sensor_msgs/CameraInfo` | `camera_driver` | `vio_node` |
| `/sensors/imu/data` | `sensor_msgs/Imu` | `imu_driver` | `vio_node`, `imu_dr` |
| `/map/georef_image` | `sensor_msgs/Image` | `georef_loader` | `feat_map` |
| `/map/georef_metadata` | `uav_nav_msgs/GeoRefMetadata` | `georef_loader` | `feat_map` |
| `/localization/vio/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `vio_node` | `state_estimator`, `imu_dr` |
| `/localization/vio/status` | `uav_nav_msgs/VioStatus` | `vio_node` | `state_estimator`, `loc_mon` |
| `/localization/vio/feature_count` | `std_msgs/Int32` | `vio_node` | `loc_mon` |
| `/localization/dr/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `imu_dr` | `state_estimator` |
| `/localization/dr/drift_estimate` | `std_msgs/Float32` | `imu_dr` | `loc_mon` |
| `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `state_estimator` | `mpc`, `path_mgr`, `prec_land`, `loc_mon`, `occ_map`, `feat_map`, `global_plan` |
| `/uav/odometry` | `nav_msgs/Odometry` | `state_estimator` | `mpc`, `prec_land` |
| `/uav/localization_mode` | `uav_nav_msgs/LocalizationMode` | `state_estimator` | `flight_mode_manager` |
| `/map/feature_density_grid` | `nav_msgs/OccupancyGrid` | `feat_map` | `global_planner` |
| `/map/georef_landmarks` | `uav_nav_msgs/LandmarkArray` | `feat_map` | `state_estimator` |
| `/map/local_costmap` | `nav_msgs/OccupancyGrid` | `occ_map` | `mpc`, `path_mgr` |
| `/mission/goal` | `geometry_msgs/PoseStamped` | `mission_manager` | `global_planner` |
| `/planning/global_path` | `nav_msgs/Path` | `global_planner` | `path_mgr` |
| `/planning/path_feature_budget` | `uav_nav_msgs/PathConstraints` | `global_planner` | `path_mgr` |
| `/planning/current_waypoint` | `geometry_msgs/PoseStamped` | `path_mgr` | `mpc` |
| `/planning/replan_trigger` | `uav_nav_msgs/ReplanReason` | `path_mgr` | `global_planner`, `flight_mode_manager` |
| `/mission/state` | `uav_nav_msgs/MissionState` | `flight_mode_manager` | **ALL nodes** |
| `/control/mpc_setpoint` | `geometry_msgs/TwistStamped` | `mpc` | `command_bridge` |
| `/control/mpc_predicted_trajectory` | `nav_msgs/Path` | `mpc` | viz only |
| `/landing/zone_detected` | `std_msgs/Bool` | `land_det` | `flight_mode_manager` |
| `/landing/zone_pose` | `geometry_msgs/PoseStamped` | `land_det` | `prec_land` |
| `/landing/visual_servo_setpoint` | `geometry_msgs/TwistStamped` | `prec_land` | `command_bridge` |
| `/monitoring/localization_health` | `uav_nav_msgs/LocalizationHealth` | `loc_mon` | `path_mgr`, `flight_mode_manager` |
| `/monitoring/max_drift_exceeded` | `std_msgs/Bool` | `loc_mon` | `flight_mode_manager` |
| `/mavros/setpoint_velocity/cmd_vel` | `geometry_msgs/TwistStamped` | `command_bridge` | PX4 via MAVROS |

---

## Package Structure

```
uav_nav_msgs/          # custom message definitions only — no logic
uav_sensors/           # camera_driver, imu_driver, georef_map_loader
uav_localization/      # vio_node, imu_dead_reckoning_node, state_estimator_node, localization_monitor_node
uav_mapping/           # feature_density_map_node, local_occupancy_node
uav_planning/          # global_planner_node, path_manager_node, mission_manager_node
uav_control/           # mpc_node, flight_mode_manager, command_bridge_node
uav_landing/           # landing_zone_detector, precision_landing_node
```

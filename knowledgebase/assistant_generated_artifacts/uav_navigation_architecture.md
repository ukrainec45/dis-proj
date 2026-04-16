# UAV Hybrid Navigation System — ROS2 Architecture

**Context:** GPS-denied outdoor navigation system for UAV.  
**Primary tasks:** Point A → Point B navigation, precise landing at destination.  
**Tech stack:** ROS2 Humble, Python, OpenCV, PX4 flight controller via MAVROS.  
**Core algorithms:** RRT* (global path planning), MPC (trajectory tracking and local obstacle avoidance).  
**Localization approach:** Visual-Inertial Odometry + IMU dead reckoning + Terrain-Aided Navigation + geo-referenced image landmarks.

---

## Design Principles

**RRT* usage policy.** RRT* runs pre-flight and only re-runs in-flight on major events: significant obstacle blocking the path, excessive cross-track deviation, localization degradation requiring a feature-richer route, or IMU-only drift budget exceeded. All local obstacle reactions are handled by MPC within its prediction horizon at control frequency. RRT* is never called at MPC frequency.

**Single source of truth for localization.** No node subscribes directly to VIO, dead reckoning, or terrain correction outputs. Every node that needs the UAV's position subscribes only to `/uav/pose` published by `state_estimator_node`. This node owns all switching and fusion logic.

**Graceful degradation.** The localization stack has four independent correction sources with different failure modes. They are prioritized and fused so that losing any one source degrades accuracy without losing navigation capability entirely.

**Feature-density-aware planning.** RRT* cost function penalizes paths through zones with low visual feature density, because VIO would degrade there. The planner prefers routes where localization will be reliable.

---

## Package Structure

```
uav_nav_msgs/          ← custom message and service definitions (built first)
uav_sensors/           ← hardware driver nodes
uav_localization/      ← all localization nodes
uav_mapping/           ← map loading and processing nodes
uav_planning/          ← RRT* and path management nodes
uav_control/           ← MPC, state machine, hardware bridge
uav_landing/           ← precision landing subsystem
```

---

## Node Reference

### SENSORS

---

#### `camera_driver`
**Package:** `uav_sensors`

Hardware abstraction for the camera. Wraps USB/CSI/GigE camera into ROS2. No image processing — purely a hardware interface. Camera calibration (intrinsic matrix K, distortion coefficients) is loaded from a YAML file and published continuously as `camera_info`. Calibration must be performed before any VIO work begins using the ROS2 `camera_calibration` package and a checkerboard target.

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/sensors/camera/image_raw` | `sensor_msgs/Image` | Raw frames @ 30 Hz |
| `/sensors/camera/camera_info` | `sensor_msgs/CameraInfo` | Intrinsic matrix K, distortion coefficients |

**Key parameters:** `camera_calibration_file.yaml`, `frame_rate`, `resolution`, `encoding`

---

#### `imu_driver`
**Package:** `uav_sensors`

Hardware abstraction for the IMU. Publishes raw linear acceleration (m/s²) and angular velocity (rad/s) at high frequency. IMU bias calibration (the constant offset the sensor reports when stationary) must be performed and stored before flight. Bias is used by `vio_node` and `imu_dead_reckoning_node` internally.

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/sensors/imu/data` | `sensor_msgs/Imu` | Linear acceleration + angular velocity @ ~200 Hz |

**Key parameters:** `publish_rate_hz`, `gyro_bias_file`, `accel_bias_file`

---

#### `rangefinder_driver`
**Package:** `uav_sensors`

Hardware abstraction for the downward-facing 1D LiDAR (e.g. Benewake TF-Luna, Garmin LiDAR-Lite). Measures distance to ground directly below the UAV. Used exclusively by `terrain_aided_nav_node`. Most sensors have existing open-source ROS2 driver packages — do not write this from scratch.

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/sensors/rangefinder/range` | `sensor_msgs/Range` | Distance-to-ground @ 50–100 Hz |

**Key parameters:** `sensor_model`, `min_range_m`, `max_range_m`, `field_of_view_rad`

---

#### `georef_map_loader`
**Package:** `uav_mapping`

Loads geo-referenced imagery from disk at startup (GeoTIFF or image + world file). Runs once at launch. Publishes the map image and its geographic metadata so that `feature_density_map_node` can process it. Exposes a service for other nodes to query what a given position looks like in the reference map.

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/map/georef_image` | `sensor_msgs/Image` | Reference image, published once at startup |
| `/map/georef_metadata` | `custom: GeoRefMetadata` | Origin, scale, coordinate reference system |

**Key parameters:** `georef_image_path`, `coordinate_frame (ENU/NED)`, `resolution_m_per_pixel`

---

#### `height_map_loader`
**Package:** `uav_mapping`

Analogous to `georef_map_loader` but for terrain elevation data. Loads a Digital Elevation Model (DEM) in GeoTIFF format at startup. Publishes the elevation grid and exposes a query service used by `global_planner_node` (terrain clearance during planning) and `terrain_aided_nav_node` (expected ground elevation at current position). The DEM coordinate frame must match the geo-referenced imagery frame exactly.

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/map/elevation_grid` | `nav_msgs/OccupancyGrid` | Terrain elevation per cell |
| Service: `query_elevation` | `custom: QueryElevation.srv` | Returns `float64 elevation_m` for given `(x, y)` |

**Key parameters:** `dem_file_path`, `coordinate_frame`, `resolution_m_per_cell`, `no_data_value`

---

### LOCALIZATION

---

#### `vio_node`
**Package:** `uav_localization`

Visual-Inertial Odometry. Estimates UAV pose relative to its starting position using a tightly-coupled EKF that fuses camera frames and IMU data. Recommended implementation: OpenVINS (ROS2 branch).

**How it works.** Camera frames provide feature tracking — corners and edges are extracted per frame and tracked across consecutive frames. Motion of features in the image reveals camera motion in the world. IMU provides high-rate motion prediction between frames (200 Hz vs 30 Hz camera) and handles fast motion where images blur. The EKF fuses both: IMU predicts, camera corrects drift.

**State vector maintained internally:** `[position(x,y,z), velocity(vx,vy,vz), orientation(roll,pitch,yaw), IMU_biases(6)]`

**Key failure mode.** Smoke, darkness, featureless terrain → feature count drops → status degrades → `state_estimator_node` switches to dead reckoning. The `min_feature_count_threshold` parameter controls when this switch is triggered.

**What it is not.** VIO does not know absolute position on Earth. It knows displacement from start. Drift accumulates over time. Absolute corrections come from `state_estimator_node` using geo-referenced landmarks and terrain corrections.

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/sensors/camera/image_raw` | `sensor_msgs/Image` | `camera_driver` |
| `/sensors/camera/camera_info` | `sensor_msgs/CameraInfo` | `camera_driver` |
| `/sensors/imu/data` | `sensor_msgs/Imu` | `imu_driver` |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/localization/vio/pose` | `geometry_msgs/PoseWithCovarianceStamped` | 6-DOF pose + covariance |
| `/localization/vio/status` | `custom: VioStatus` | `TRACKING / DEGRADED / LOST` |
| `/localization/vio/feature_count` | `std_msgs/Int32` | Tracked feature count — confidence proxy |

**Key parameters:** `camera_calibration_file`, `imu_noise_model`, `min_feature_count_threshold`, `feature_detector (FAST/ORB)`

---

#### `imu_dead_reckoning_node`
**Package:** `uav_localization`

IMU-only position estimation used as fallback when VIO is lost. Integrates corrected acceleration twice: acceleration → velocity → position. Drift grows quadratically with time (typical MEMS IMU: ~0.5m after 10s, ~4m after 30s, ~16m after 60s).

**Initialization.** When `vio_node` status switches to `LOST`, this node initializes from the last known VIO pose. It does not restart from zero.

**Purpose.** This is a bridge, not a long-term solution. The system must reacquire VIO, apply a terrain correction, or land before drift exceeds `max_allowed_drift_m`. The `localization_monitor_node` enforces this limit.

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/sensors/imu/data` | `sensor_msgs/Imu` | `imu_driver` |
| `/localization/vio/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `vio_node` — initialization only |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/localization/dr/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Dead-reckoned pose, covariance grows over time |
| `/localization/dr/drift_estimate` | `std_msgs/Float32` | Accumulated drift in meters |

**Key parameters:** `imu_accel_noise_density`, `imu_gyro_noise_density`, `gravity_vector`, `drift_growth_rate_model`

---

#### `terrain_aided_nav_node`
**Package:** `uav_localization`

Terrain-Aided Navigation. Fuses downward-facing rangefinder readings with the DEM to produce position corrections independent of visual conditions. Operates through smoke, darkness, and featureless terrain — conditions that defeat VIO. Complementary failure modes make TAN and VIO a robust combination.

**Two correction types:**

*Vertical correction (always active).* Corrected rangefinder reading = AGL height. DEM elevation at (x,y) is known. Therefore absolute z = DEM_elevation + AGL_height. Applied as a continuous EKF update to `state_estimator_node`. Effective even in flat terrain.

*Horizontal correction (terrain-dependent).* As the UAV moves, a buffer of (rangefinder_reading, estimated_position) pairs accumulates a terrain profile. This profile is matched against the DEM using correlation search or a particle filter to find the (x,y) offset that minimizes mismatch. Only reliable when terrain has sufficient elevation variation (`min_terrain_variation_m` threshold). Published with a confidence score — `state_estimator_node` ignores low-confidence matches.

**Attitude correction.** Rangefinder measures slant range when UAV is tilted. True vertical distance = `range × cos(pitch) × cos(roll)` using IMU attitude.

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/sensors/rangefinder/range` | `sensor_msgs/Range` | `rangefinder_driver` |
| `/sensors/imu/data` | `sensor_msgs/Imu` | `imu_driver` — for attitude correction |
| `/map/elevation_grid` | `nav_msgs/OccupancyGrid` | `height_map_loader` |
| `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `state_estimator_node` — for DEM lookup at current position |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/localization/terrain/altitude_agl` | `std_msgs/Float32` | Absolute height above ground level |
| `/localization/terrain/correction` | `custom: TerrainCorrection` | Position correction with confidence score |
| `/localization/terrain/status` | `custom: TerrainNavStatus` | `VALID / FLAT_TERRAIN / OUT_OF_RANGE` |

**Key parameters:** `min_terrain_variation_m`, `correction_confidence_threshold`, `profile_length_m`, `rangefinder_mount_offset (extrinsic)`

**When TAN helps most vs least:**

| Scenario | Value |
|---|---|
| Varied terrain (hills, buildings) | High — both vertical and horizontal corrections valid |
| Flat open land | Vertical correction only |
| Smoke / darkness (VIO failed) | High — unaffected by visibility |
| Altitude < 2m | Unreliable — rangefinder multipath effects |
| Outdated DEM (new construction) | Horizontal match may fail locally; vertical still works |

---

#### `state_estimator_node`
**Package:** `uav_localization`

**The single source of truth for UAV state.** All nodes that need position subscribe only to `/uav/pose` from this node. No node subscribes directly to VIO, dead reckoning, or terrain correction outputs.

**Responsibility.** Decides which sources to trust, blends them during transitions, applies corrections, and publishes a single consistent state with a calibrated covariance.

**Correction source hierarchy (priority order):**

1. VIO pose — primary, full 6-DOF, trusted when `VioStatus == TRACKING`
2. Terrain vertical correction — continuous, always applied when rangefinder is in range
3. Terrain horizontal correction — applied when `TerrainNavStatus == VALID` and confidence above threshold
4. Georef landmark correction — opportunistic, when `feature_density_map_node` reports a recognized landmark match
5. Dead reckoning — fallback when all above corrections are unavailable

**Operating modes:**

`VIO_ACTIVE` — passes through VIO pose with its covariance. Terrain vertical correction applied on top as a z-axis EKF update.

`TRANSITIONING` — covariance-weighted blend of VIO and DR. As VIO confidence drops (covariance grows), DR weight increases. Prevents hard position jump at switchover.

`DR_ONLY` — publishes dead reckoning pose. Covariance grows each cycle. Terrain vertical correction continues to constrain z-axis drift. Broadcasts `LocalizationMode = DR_ONLY` to `flight_mode_manager`, which responds (climb for VIO reacquisition, loiter, or emergency land).

**Full state vector published:**
`[x, y, z, vx, vy, vz, roll, pitch, yaw]`
Packed as `PoseWithCovarianceStamped` (position + orientation + uncertainty) and `nav_msgs/Odometry` (position + velocity for MPC dynamics).

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/localization/vio/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `vio_node` |
| `/localization/vio/status` | `custom: VioStatus` | `vio_node` |
| `/localization/dr/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `imu_dead_reckoning_node` |
| `/localization/terrain/correction` | `custom: TerrainCorrection` | `terrain_aided_nav_node` |
| `/localization/terrain/altitude_agl` | `std_msgs/Float32` | `terrain_aided_nav_node` |
| `/map/georef_landmarks` | `custom: LandmarkArray` | `feature_density_map_node` |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Authoritative position + orientation + covariance |
| `/uav/odometry` | `nav_msgs/Odometry` | Pose + velocity — used by MPC for dynamics prediction |
| `/uav/localization_mode` | `custom: LocalizationMode` | `VIO_ACTIVE / TRANSITIONING / DR_ONLY` |

**Key parameters:** `vio_to_dr_covariance_threshold`, `blend_window_duration_s`, `terrain_correction_weight`, `min_terrain_confidence_for_fusion`, `min_landmark_match_confidence`, `correction_max_jump_m`

---

#### `localization_monitor_node`
**Package:** `uav_localization`

Watchdog. Reads VIO feature count, pose covariance magnitude, DR drift estimate, and terrain nav status every cycle. Compares against thresholds. Publishes health status and triggers emergency flags. Pure monitoring — no effect on localization itself. `flight_mode_manager` and `path_manager_node` act on its outputs.

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/localization/vio/feature_count` | `std_msgs/Int32` | `vio_node` |
| `/localization/dr/drift_estimate` | `std_msgs/Float32` | `imu_dead_reckoning_node` |
| `/localization/terrain/status` | `custom: TerrainNavStatus` | `terrain_aided_nav_node` |
| `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `state_estimator_node` |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/monitoring/localization_health` | `custom: LocalizationHealth` | `OK / DEGRADED / CRITICAL` |
| `/monitoring/max_drift_exceeded` | `std_msgs/Bool` | True when DR drift exceeds safety limit |

**Key parameters:** `min_features_warning`, `min_features_critical`, `max_allowed_drift_m`, `covariance_warning_threshold`, `check_rate_hz`

---

### MAPPING

---

#### `feature_density_map_node`
**Package:** `uav_mapping`

Pre-processes geo-referenced imagery at startup using an OpenCV feature detector (ORB or FAST) to compute, per grid cell, how many trackable visual features exist in that area. Output is a cost layer for RRT*: cells with low feature density get high cost because VIO would degrade when flying over them. Updated during flight with live camera frames. Also performs matching between live camera view and map features, publishing confirmed matches as landmarks for `state_estimator_node` to use as absolute position corrections.

Uses elevation data from `height_map_loader` to make the density map altitude-aware: features visible from 30m AGL may not be detectable from 80m AGL, so density scores are adjusted per planned flight altitude.

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/map/georef_image` | `sensor_msgs/Image` | `georef_map_loader` |
| `/map/elevation_grid` | `nav_msgs/OccupancyGrid` | `height_map_loader` — for altitude-aware density scoring |
| `/sensors/camera/image_raw` | `sensor_msgs/Image` | `camera_driver` — live update during flight |
| `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `state_estimator_node` — for map-frame projection |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/map/feature_density_grid` | `nav_msgs/OccupancyGrid` | 0–100 VIO quality score per cell. Used by RRT* as cost. |
| `/map/georef_landmarks` | `custom: LandmarkArray` | Recognized landmark positions for drift correction |

**Key parameters:** `grid_resolution_m`, `feature_detector (ORB/FAST/SIFT)`, `min_density_threshold`, `live_update_enabled`, `planned_flight_altitude_m`

---

#### `local_occupancy_node`
**Package:** `uav_mapping`

Builds a local obstacle map from live camera data. This is the real-time obstacle layer — separate from the geo-referenced global map. Captures dynamic obstacles (people, vehicles) and map changes (new construction, downed trees) not present in pre-flight reference imagery. MPC reads this at each control cycle to detect and avoid obstacles within its prediction horizon.

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/sensors/camera/image_raw` | `sensor_msgs/Image` | `camera_driver` |
| `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `state_estimator_node` — for map-frame projection |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/map/local_costmap` | `nav_msgs/OccupancyGrid` | Live obstacle map updated ~5 Hz, radius ~20–50m around UAV |

**Key parameters:** `map_radius_m`, `map_resolution_m`, `obstacle_inflation_radius_m`, `depth_method (mono_depth_net/stereo)`, `decay_time_s`

---

### PLANNING

---

#### `mission_manager_node`
**Package:** `uav_planning`

Top-level mission interface. Holds goal coordinates and mission-level constraints. Calls the global planner Action at mission start and again if forced replan is requested. Exposes a `SetMission` service called by an operator or ground control station before takeoff.

**Inputs:**

| Interface | Type | Source |
|---|---|---|
| Service: `SetMission` | `custom: SetMission.srv` | Operator / GCS |

**Outputs:**

| Topic / Interface | Type | Description |
|---|---|---|
| `/mission/goal` | `geometry_msgs/PoseStamped` | Target destination for `global_planner_node` |
| Action client: `PlanPath` | `custom: PlanPath.action` | Calls `global_planner_node` to initiate planning |

**Key parameters:** `default_cruise_altitude_m`, `max_flight_time_s`, `home_position`

---

#### `global_planner_node`
**Package:** `uav_planning`

Implements RRT* global path planning. Exposed as a ROS2 Action (not a service) because planning takes seconds and the caller needs progress feedback and cancellation capability.

**RRT* cost function — three weighted terms:**

```
cost = w1 * path_length
     + w2 * feature_density_penalty    (low density = high cost)
     + w3 * obstacle_clearance_penalty (proximity to known obstacles)
```

**Terrain clearance constraint.** For every RRT* sampled point, queries `height_map_loader` via `query_elevation` service and enforces `waypoint_z > terrain_elevation + min_terrain_clearance_m`. Prevents routing into hillsides. Optionally, terrain variation richness is added as a soft bonus term to prefer routes where TAN horizontal correction will be reliable.

**PathConstraints output.** For each path segment, computes minimum feature density along the segment and derives the maximum allowed IMU-only transit duration. Also flags whether terrain correction is expected to be available along each segment. This budget is passed to `path_manager_node` and enforced during flight.

**In-flight replanning.** Warm-starts using the existing RRT* tree — only branches that are now invalid are discarded. Replanning is faster than a full cold start. Triggered only on major events (see RRT* usage policy), not at control frequency.

**Inputs:**

| Topic / Interface | Type | Source |
|---|---|---|
| `/map/feature_density_grid` | `nav_msgs/OccupancyGrid` | `feature_density_map_node` |
| `/map/elevation_grid` | `nav_msgs/OccupancyGrid` | `height_map_loader` — terrain clearance |
| `/mission/goal` | `geometry_msgs/PoseStamped` | `mission_manager_node` |
| `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `state_estimator_node` — start node for replanning |
| Action server: `PlanPath` | `custom: PlanPath.action` | Called by `mission_manager_node` or `path_manager_node` |

**Outputs:**

| Topic / Interface | Type | Description |
|---|---|---|
| `/planning/global_path` | `nav_msgs/Path` | Sequence of waypoints from current position to goal |
| `/planning/path_feature_budget` | `custom: PathConstraints` | Per-segment max IMU-only duration + terrain correction availability |
| Action feedback: `PlanPath` | `custom: PlanPath.action` | Planning progress %, current best path length |

**Key parameters:** `max_iterations`, `step_size_m`, `goal_bias_prob`, `feature_density_weight w2`, `path_length_weight w1`, `rewire_radius_m`, `min_terrain_clearance_m`, `prefer_varied_terrain (bool)`

**RRT* replan triggers (from `path_manager_node`):**

| Reason | Description |
|---|---|
| `OBSTACLE_DETECTED` | New obstacle blocks a significant portion of the path ahead |
| `DEVIATION_EXCEEDED` | Cross-track error beyond `max_cross_track_error_m` — returning to path not sensible |
| `LOC_DEGRADED` | Localization health dropped — need route through higher feature-density area |
| `BUDGET_EXCEEDED` | Approaching segment requiring longer IMU-only transit than safety budget allows |

---

#### `path_manager_node`
**Package:** `uav_planning`

In-flight supervisor of path execution. Runs continuously during navigation. At each cycle: computes cross-track error between UAV position and planned path, checks if the next costmap segment is clear, checks localization health, and enforces feature budget constraints. Advances the current waypoint pointer as waypoints are reached. Triggers RRT* replanning only on the four major events listed above.

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/uav/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `state_estimator_node` |
| `/planning/global_path` | `nav_msgs/Path` | `global_planner_node` |
| `/planning/path_feature_budget` | `custom: PathConstraints` | `global_planner_node` |
| `/map/local_costmap` | `nav_msgs/OccupancyGrid` | `local_occupancy_node` |
| `/monitoring/localization_health` | `custom: LocalizationHealth` | `localization_monitor_node` |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/planning/current_waypoint` | `geometry_msgs/PoseStamped` | Active waypoint fed to MPC |
| `/planning/replan_trigger` | `custom: ReplanReason` | `OBSTACLE_DETECTED / DEVIATION_EXCEEDED / LOC_DEGRADED / BUDGET_EXCEEDED` |

**Key parameters:** `max_cross_track_error_m`, `waypoint_reached_radius_m`, `lookahead_distance_m`, `replan_costmap_change_threshold`

---

### CONTROL

---

#### `mpc_node`
**Package:** `uav_control`

Model Predictive Control. At each cycle (~20–50 Hz), solves an optimization problem over a short future horizon to find the best velocity commands.

**How MPC works.** Given current state x(t), the solver simulates N steps into the future using a UAV dynamics model, finds control inputs that minimize the total predicted cost, applies only the first control input, then repeats next cycle with fresh measurements. The prediction horizon is what separates MPC from PID — it anticipates obstacles and starts avoiding them before they are immediately dangerous.

**Cost function:**
1. Waypoint tracking error (position)
2. Control effort (smooth commands, penalise jerky motion)
3. Obstacle proximity (from `local_costmap`)
4. Velocity and acceleration constraints (actuator limits)

MPC never calls RRT*. All local obstacle reactions are handled within the prediction horizon at control frequency.

**Recommended implementation:** `do-mpc` library (Python, easier to prototype) or `acados` (C code generation, faster — better for final system).

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/uav/odometry` | `nav_msgs/Odometry` | `state_estimator_node` — pose + velocity for dynamics |
| `/planning/current_waypoint` | `geometry_msgs/PoseStamped` | `path_manager_node` |
| `/map/local_costmap` | `nav_msgs/OccupancyGrid` | `local_occupancy_node` |
| `/mission/state` | `custom: MissionState` | `flight_mode_manager` |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/control/mpc_setpoint` | `geometry_msgs/TwistStamped` | Velocity commands [vx, vy, vz, yaw_rate] @ 20–50 Hz |
| `/control/mpc_predicted_trajectory` | `nav_msgs/Path` | Predicted future path over horizon N (RViz debug) |

**Key parameters:** `horizon_steps N`, `dt`, `Q matrix (state tracking weights)`, `R matrix (control effort weights)`, `obstacle_cost_weight`, `max_velocity_mps`, `uav_mass_kg`

---

#### `flight_mode_manager`
**Package:** `uav_control`

Top-level Finite State Machine. All nodes check `/mission/state` to decide what they should be doing. Broadcasts at ~10 Hz.

**State graph:**

```
PREFLIGHT
    ↓ plan_complete
PLANNING
    ↓ path_ready
TAKEOFF
    ↓ altitude_reached
NAVIGATE ←──────────────────────────────┐
    ↓ replan_trigger          REPLANNING ┘
    ↓ loc_degraded + no replan possible
CLIMB_FOR_VIEW  (gain altitude to reacquire VIO features or terrain variation)
    ↓ features_reacquired
NAVIGATE
    ↓ max_drift_exceeded
EMERGENCY_HOVER
    ↓ manual override or timeout
EMERGENCY_LAND
    ↓ goal_reached
LAND_APPROACH
    ↓ zone_detected + altitude < threshold
PRECISION_LAND
    ↓ landed
LANDED

Any state → EMERGENCY_LAND on critical_failure
```

**How nodes use `/mission/state`:**
- `mpc_node` — switches cost function tuning for `LAND_APPROACH` vs `NAVIGATE`
- `command_bridge_node` — disables all commands in `PREFLIGHT`
- `landing_zone_detector` — only runs processing in `LAND_APPROACH` and `PRECISION_LAND`
- `localization_monitor_node` — tightens thresholds in `PRECISION_LAND`

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/uav/localization_mode` | `custom: LocalizationMode` | `state_estimator_node` |
| `/monitoring/localization_health` | `custom: LocalizationHealth` | `localization_monitor_node` |
| `/monitoring/max_drift_exceeded` | `std_msgs/Bool` | `localization_monitor_node` |
| `/planning/replan_trigger` | `custom: ReplanReason` | `path_manager_node` |
| `/landing/zone_detected` | `std_msgs/Bool` | `landing_zone_detector` |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/mission/state` | `custom: MissionState` | Broadcast to all nodes |

**Key parameters:** `state_transition_timeouts`, `climb_altitude_for_view_reacquisition_m`, `emergency_land_on_critical_loc`, `precision_land_activation_altitude_m`

---

#### `command_bridge_node`
**Package:** `uav_control`

Final translation layer from ROS2 velocity setpoints to MAVLink/PX4 commands via MAVROS. Implements command source priority: `precision_landing_node` setpoints override MPC setpoints when `mission_state == PRECISION_LAND`. Enforces hard safety limits (max velocity, geofence polygon) as last-resort software fence — applied regardless of what upstream nodes computed.

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/control/mpc_setpoint` | `geometry_msgs/TwistStamped` | `mpc_node` |
| `/landing/visual_servo_setpoint` | `geometry_msgs/TwistStamped` | `precision_landing_node` — overrides MPC in PRECISION_LAND |
| `/mission/state` | `custom: MissionState` | `flight_mode_manager` |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/mavros/setpoint_velocity/cmd_vel` | `geometry_msgs/TwistStamped` | → MAVROS → PX4 via MAVLink |

**Key parameters:** `max_velocity_mps (hard limit)`, `max_acceleration_mps2 (hard limit)`, `geofence_polygon_wkt`, `command_timeout_ms`

---

### LANDING

---

#### `landing_zone_detector`
**Package:** `uav_landing`

Detects the landing target visually. Only active in `LAND_APPROACH` and `PRECISION_LAND` states — gated by `/mission/state` to avoid wasting compute during cruise. Recommended approach: ArUco marker at the landing site (`cv2.aruco` in OpenCV). Outputs relative pose of the target in the camera frame.

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/sensors/camera/image_raw` | `sensor_msgs/Image` | `camera_driver` |
| `/mission/state` | `custom: MissionState` | `flight_mode_manager` |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/landing/zone_detected` | `std_msgs/Bool` | True when target confidently detected |
| `/landing/zone_pose` | `geometry_msgs/PoseStamped` | Relative pose of landing target in camera frame |

**Key parameters:** `detector_type (aruco/learned/h_pattern)`, `aruco_marker_id`, `aruco_marker_size_m`, `detection_confidence_threshold`

---

#### `precision_landing_node`
**Package:** `uav_landing`

Visual servoing loop. Runs below `activation_altitude_m` (e.g. 5m). Uses landing zone pose to generate corrective velocity commands that center the UAV over the target. Commands bypass MPC and go directly to `command_bridge_node` with priority.

**How visual servoing works.** Error = target pose offset from camera center. Converted to world-frame position error. PD controller applies proportional and derivative gains to produce velocity corrections. Descent velocity (vz) follows an altitude-dependent profile — faster at 5m, slower at 1m, near-zero at 0.3m.

**Inputs:**

| Topic | Type | Source |
|---|---|---|
| `/landing/zone_pose` | `geometry_msgs/PoseStamped` | `landing_zone_detector` |
| `/uav/odometry` | `nav_msgs/Odometry` | `state_estimator_node` — velocity for derivative term |
| `/mission/state` | `custom: MissionState` | `flight_mode_manager` |

**Outputs:**

| Topic | Type | Description |
|---|---|---|
| `/landing/visual_servo_setpoint` | `geometry_msgs/TwistStamped` | Velocity commands → `command_bridge_node`, overrides MPC |

**Key parameters:** `visual_servo_gains_kp`, `visual_servo_gains_kd`, `activation_altitude_m`, `max_lateral_correction_mps`, `descent_velocity_profile_params`, `landing_complete_threshold_m`

---

## Complete Topic Registry

| Topic | Type | Publisher | Subscribers | Rate |
|---|---|---|---|---|
| `/sensors/camera/image_raw` | `sensor_msgs/Image` | `camera_driver` | `vio_node`, `feat_map`, `occ_map`, `land_det` | 30 Hz |
| `/sensors/camera/camera_info` | `sensor_msgs/CameraInfo` | `camera_driver` | `vio_node` | 30 Hz |
| `/sensors/imu/data` | `sensor_msgs/Imu` | `imu_driver` | `vio_node`, `imu_dr`, `terrain_aided_nav` | ~200 Hz |
| `/sensors/rangefinder/range` | `sensor_msgs/Range` | `rangefinder_driver` | `terrain_aided_nav` | 50–100 Hz |
| `/map/georef_image` | `sensor_msgs/Image` | `georef_map_loader` | `feature_density_map` | once |
| `/map/georef_metadata` | `custom: GeoRefMetadata` | `georef_map_loader` | `feature_density_map` | once |
| `/map/elevation_grid` | `nav_msgs/OccupancyGrid` | `height_map_loader` | `global_planner`, `terrain_aided_nav`, `feature_density_map` | once |
| `/map/feature_density_grid` | `nav_msgs/OccupancyGrid` | `feature_density_map` | `global_planner` | on change |
| `/map/georef_landmarks` | `custom: LandmarkArray` | `feature_density_map` | `state_estimator` | ~5 Hz |
| `/map/local_costmap` | `nav_msgs/OccupancyGrid` | `local_occupancy` | `mpc_node`, `path_manager` | ~5 Hz |
| `/localization/vio/pose` | `PoseWithCovarianceStamped` | `vio_node` | `state_estimator`, `imu_dr` | 30 Hz |
| `/localization/vio/status` | `custom: VioStatus` | `vio_node` | `state_estimator`, `loc_mon` | 10 Hz |
| `/localization/vio/feature_count` | `std_msgs/Int32` | `vio_node` | `loc_mon` | 10 Hz |
| `/localization/dr/pose` | `PoseWithCovarianceStamped` | `imu_dead_reckoning` | `state_estimator` | 50 Hz |
| `/localization/dr/drift_estimate` | `std_msgs/Float32` | `imu_dead_reckoning` | `loc_mon` | 10 Hz |
| `/localization/terrain/altitude_agl` | `std_msgs/Float32` | `terrain_aided_nav` | `state_estimator` | 50 Hz |
| `/localization/terrain/correction` | `custom: TerrainCorrection` | `terrain_aided_nav` | `state_estimator` | 5–10 Hz |
| `/localization/terrain/status` | `custom: TerrainNavStatus` | `terrain_aided_nav` | `loc_mon` | 10 Hz |
| `/uav/pose` | `PoseWithCovarianceStamped` | `state_estimator` | `mpc`, `path_mgr`, `prec_land`, `loc_mon`, `occ_map`, `feat_map`, `terrain_aided_nav` | 30 Hz |
| `/uav/odometry` | `nav_msgs/Odometry` | `state_estimator` | `mpc`, `prec_land` | 30 Hz |
| `/uav/localization_mode` | `custom: LocalizationMode` | `state_estimator` | `flight_mode_manager` | 10 Hz |
| `/planning/global_path` | `nav_msgs/Path` | `global_planner` | `path_manager`, `mpc` | on change |
| `/planning/path_feature_budget` | `custom: PathConstraints` | `global_planner` | `path_manager` | on change |
| `/planning/current_waypoint` | `geometry_msgs/PoseStamped` | `path_manager` | `mpc` | 10 Hz |
| `/planning/replan_trigger` | `custom: ReplanReason` | `path_manager` | `global_planner`, `mode_mgr` | on event |
| `/mission/goal` | `geometry_msgs/PoseStamped` | `mission_manager` | `global_planner` | once |
| `/mission/state` | `custom: MissionState` | `flight_mode_manager` | **all nodes** | 10 Hz |
| `/control/mpc_setpoint` | `geometry_msgs/TwistStamped` | `mpc_node` | `command_bridge` | 20–50 Hz |
| `/control/mpc_predicted_trajectory` | `nav_msgs/Path` | `mpc_node` | monitoring/RViz | 10 Hz |
| `/monitoring/localization_health` | `custom: LocalizationHealth` | `loc_mon` | `path_manager`, `mode_mgr` | 10 Hz |
| `/monitoring/max_drift_exceeded` | `std_msgs/Bool` | `loc_mon` | `mode_mgr` | on event |
| `/landing/zone_detected` | `std_msgs/Bool` | `land_det` | `mode_mgr` | 10 Hz |
| `/landing/zone_pose` | `geometry_msgs/PoseStamped` | `land_det` | `prec_land` | 10 Hz |
| `/landing/visual_servo_setpoint` | `geometry_msgs/TwistStamped` | `prec_land` | `command_bridge` | 20 Hz |
| `/mavros/setpoint_velocity/cmd_vel` | `geometry_msgs/TwistStamped` | `command_bridge` | PX4 | 20–50 Hz |

---

## Custom Message Definitions (`uav_nav_msgs`)

```
VioStatus.msg
    std_msgs/Header header
    int8 status
    int32 tracked_features
    int8 TRACKING=0
    int8 DEGRADED=1
    int8 LOST=2

LocalizationMode.msg
    std_msgs/Header header
    int8 mode
    int8 VIO_ACTIVE=0
    int8 TRANSITIONING=1
    int8 DR_ONLY=2

LocalizationHealth.msg
    std_msgs/Header header
    int8 health
    string reason
    int8 OK=0
    int8 DEGRADED=1
    int8 CRITICAL=2

TerrainNavStatus.msg
    std_msgs/Header header
    int8 status
    float32 horizontal_confidence
    int8 VALID=0
    int8 FLAT_TERRAIN=1
    int8 OUT_OF_RANGE=2

TerrainCorrection.msg
    std_msgs/Header header
    geometry_msgs/Point position_correction
    float32 confidence
    bool horizontal_valid
    bool vertical_valid

MissionState.msg
    std_msgs/Header header
    int8 state
    int8 PREFLIGHT=0
    int8 PLANNING=1
    int8 TAKEOFF=2
    int8 NAVIGATE=3
    int8 REPLANNING=4
    int8 CLIMB_FOR_VIEW=5
    int8 EMERGENCY_HOVER=6
    int8 LAND_APPROACH=7
    int8 PRECISION_LAND=8
    int8 LANDED=9
    int8 EMERGENCY_LAND=10

PathConstraints.msg
    std_msgs/Header header
    float32[] max_imu_only_duration_s       # one entry per path segment
    float32[] min_feature_density           # one entry per path segment
    bool[]    terrain_correction_available  # one entry per path segment

ReplanReason.msg
    std_msgs/Header header
    int8 reason
    string detail
    int8 OBSTACLE_DETECTED=0
    int8 DEVIATION_EXCEEDED=1
    int8 LOC_DEGRADED=2
    int8 BUDGET_EXCEEDED=3

LandmarkArray.msg
    std_msgs/Header header
    geometry_msgs/Point[] world_positions
    float32[] confidence_scores

GeoRefMetadata.msg
    std_msgs/Header header
    geometry_msgs/Point origin
    float32 resolution_m_per_pixel
    string coordinate_reference_system
    uint32 width_px
    uint32 height_px
```

---

## Services and Actions

```
SetMission.srv          (mission_manager_node)
    geometry_msgs/PoseStamped goal
    float32 cruise_altitude_m
    float32 max_flight_time_s
    ---
    bool accepted
    string message

QueryElevation.srv      (height_map_loader)
    float64 x
    float64 y
    ---
    float64 elevation_m
    bool valid

PlanPath.action         (global_planner_node)
    Goal:
        geometry_msgs/PoseStamped start
        geometry_msgs/PoseStamped goal
        bool warm_start
    Feedback:
        float32 progress_percent
        float32 current_best_path_length_m
    Result:
        nav_msgs/Path path
        PathConstraints constraints
        bool success
        string message
```

---

## Localization Fallback Chain

```
Normal flight:
  VIO (primary, full 6-DOF)
  + terrain vertical correction (continuous, z-axis, independent of visibility)
  + terrain horizontal correction (when terrain varied enough)
  + georef landmark corrections (opportunistic)

VIO degraded (few features, smoke beginning):
  Blend: VIO + DR weighted by covariance
  Terrain vertical correction continues
  Terrain horizontal correction if terrain has sufficient variation
  path_manager triggers replan to find feature-richer route

VIO lost:
  DR only
  + terrain vertical correction (limits z-axis drift accumulation)
  + terrain horizontal correction if terrain varied enough
  flight_mode_manager responds: CLIMB_FOR_VIEW or EMERGENCY_HOVER
  loc_mon enforces max_allowed_drift_m budget

All corrections lost (flat featureless terrain, VIO lost, rangefinder out of range):
  Pure dead reckoning
  max_drift_exceeded → EMERGENCY_LAND
```

---

## Implementation Notes

**Build order.** Define all custom messages in `uav_nav_msgs` first — every other package depends on them. Then sensor drivers and map loaders. Then localization nodes. Then planning. Then control. Landing last.

**Test strategy.** Implement algorithm classes (RRT*, MPC, EKF, terrain matching) as pure Python with no ROS2 dependency. Test and validate with matplotlib. Wrap in ROS2 nodes only after the algorithm is correct. Use a `sim_state_publisher` fake node publishing `/uav/pose` on a timer during early development so planning and control nodes can be developed before localization is built.

**DEM data sources.** SRTM (30m resolution, global, free), Copernicus DEM (25m EU, free), or local survey data. For urban environments where building heights matter, consider OpenStreetMap 3D building data as a supplementary obstacle layer.

**Rangefinder recommendation.** Benewake TF-Luna (40m range, 100 Hz, lightweight, ROS2 driver available). Mount rigidly downward with a known extrinsic offset from the IMU frame. This offset must be accounted for in `terrain_aided_nav_node`.

**Calibration requirements before any real flight.**
- Camera intrinsics: ROS2 `camera_calibration` package + checkerboard
- Camera-IMU extrinsics: Kalibr (spatial offset + time offset between camera and IMU)
- IMU bias: static measurement, stored to file
- Rangefinder extrinsics: physical measurement of mount offset from IMU
- DEM alignment: verify DEM coordinate frame matches georef imagery frame

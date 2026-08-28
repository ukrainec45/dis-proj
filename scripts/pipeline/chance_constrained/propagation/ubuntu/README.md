# EuRoC/OpenVINS baseline on Ubuntu 24.04 / ROS 2 Jazzy

This directory contains the Ubuntu portion of the
`ins_openvins_model.npz` build pipeline. The platform-independent Python code
can already convert logs into segments, fit the model, and validate it. Actual
numeric parameters are produced only after the 12 measured replays described
below.

## 1. One-time setup

Install ROS 2 Jazzy and the OpenVINS dependencies, then build the pinned
OpenVINS `v2.7` release. The Jazzy patch contains the upstream Ceres 2.2
adaptation from OpenVINS PR #520 and the required ROS 2 header and shutdown
compatibility changes from PR #500; it does not change estimator behavior:

```bash
mkdir -p ~/openvins_ws/src
cd ~/openvins_ws/src
git clone --branch v2.7 --depth 1 https://github.com/rpng/open_vins.git
cd open_vins
git rev-parse HEAD
git apply /absolute/path/to/proj/scripts/pipeline/chance_constrained/propagation/ubuntu/openvins_v2.7_diagnostics.patch
git apply /absolute/path/to/proj/scripts/pipeline/chance_constrained/propagation/ubuntu/openvins_v2.7_jazzy.patch
cd ~/openvins_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source ~/openvins_ws/install/setup.bash
```

The expected commit is
`93adc241390d13e99232652cf05cbe18a93c7bea`. The patch does not change the
estimation algorithm. It only appends tracked/accepted feature counts and a
visual-update acceptance flag to the standard timing CSV.

Download the ROS2 versions of `V2_01_easy`, `V2_02_medium`, and
`V2_03_difficult`, together with the official ground-truth files
(`ov_data/euroc_mav/V2_*.txt`). Keep all datasets outside Git. If you downloaded
a ROS1 `.bag`, first convert it with the OpenVINS-recommended
`rosbags-convert` utility. The runner expects a rosbag2 directory.

## 2. Twelve deterministic replays

Source ROS2 and the OpenVINS workspace in every terminal before running the
experiment. The runner refuses to overwrite an existing run directory. Its
arguments are: repository root, OpenVINS root, rosbag2 path, experiment root,
run ID, frame stride, base regime, and blackout JSON.

Run the following four variants for each sequence:

```bash
runner=/absolute/path/to/proj/scripts/pipeline/chance_constrained/propagation/ubuntu/run_euroc_replay.sh
repo=/absolute/path/to/proj
ov=~/openvins_ws/src/open_vins
root=/absolute/path/to/euroc_propagation
bag=/absolute/path/to/V2_01_easy_ros2

bash "$runner" "$repo" "$ov" "$bag" "$root" V2_01_easy_nominal_20hz 1 nominal_20hz '{"starts":[],"durations":[]}'
bash "$runner" "$repo" "$ov" "$bag" "$root" V2_01_easy_degraded_10hz 2 degraded_10hz '{"starts":[],"durations":[]}'
bash "$runner" "$repo" "$ov" "$bag" "$root" V2_01_easy_degraded_5hz 4 degraded_5hz '{"starts":[],"durations":[]}'
bash "$runner" "$repo" "$ov" "$bag" "$root" V2_01_easy_blackout 1 nominal_20hz '{"starts":[8,15,23],"durations":[2,5,10]}'
```

Repeat the four commands for `V2_02_medium` and `V2_03_difficult`, changing the
`bag` path and run-ID prefix. For the longer V2_02/V2_03 sequences, use blackout
start times `[15,30,50]`. All three blackout windows must end before the replay
finishes so that recovery can be observed. The runner never filters the IMU
stream.

## 3. Converting raw logs

Run the converter from the repository root for every replay. The following is
one example:

```bash
export PYTHONPATH=/absolute/path/to/proj
python3 -m scripts.pipeline.chance_constrained.propagation.openvins_adapter \
  --odometry "$root/raw/V2_01_easy_nominal_20hz/odometry.csv" \
  --diagnostics "$root/raw/V2_01_easy_nominal_20hz/diagnostics.csv" \
  --ground-truth "$ov/ov_data/euroc_mav/V2_01_easy.txt" \
  --frame-schedule "$root/raw/V2_01_easy_nominal_20hz/frame_schedule.csv" \
  --output "$root/run_logs/V2_01_easy_nominal_20hz.npz"
```

The converter synchronizes ground truth with odometry, rotates the local
OpenVINS velocity and its covariance into the global coordinate frame, and
stores the reported covariance. Repeat this command for all 12 run IDs.

Create `source_manifest.json` by supplying the three datasets and all 12
`raw/<run>/config/config_manifest.json` files through the repeatable
`--openvins-config-manifest` argument:

```bash
python3 scripts/pipeline/chance_constrained/propagation/ubuntu/create_source_manifest.py \
  --dataset V2_01_easy /abs/V2_01_easy_ros2 "$ov/ov_data/euroc_mav/V2_01_easy.txt" \
  --dataset V2_02_medium /abs/V2_02_medium_ros2 "$ov/ov_data/euroc_mav/V2_02_medium.txt" \
  --dataset V2_03_difficult /abs/V2_03_difficult_ros2 "$ov/ov_data/euroc_mav/V2_03_difficult.txt" \
  --openvins-config-manifest "$root/raw/V2_01_easy_nominal_20hz/config/config_manifest.json" \
  --output "$root/source_manifest.json"
```

Add the remaining 11 `--openvins-config-manifest` arguments to the actual
command. The resulting manifest records absolute paths and SHA-256 hashes.

## 4. Building the model

```bash
export EUROC_PROPAGATION_ROOT=/absolute/path/to/euroc_propagation
python3 -m scripts.pipeline.chance_constrained.propagation.build_model \
  --config configs/chance_constrained/openvins/euroc_monocular_baseline.example.json \
  --output-dir "$EUROC_PROPAGATION_ROOT/model"
```

The command always writes diagnostic results. However,
`InsOpenVinsPropagationModel` refuses to load the artifact by default if the
held-out V2_03 validation, 95% post-initialization coverage, blackout-mode
classification, or VIO-versus-INS drift comparison fails. Numeric parameters
are never adjusted automatically to make validation pass.

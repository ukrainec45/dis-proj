#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 REPO OPENVINS_ROOT ROS2_BAG OUTPUT_ROOT RUN_ID STRIDE BASE_REGIME BLACKOUT_JSON"
  echo 'BLACKOUT_JSON example: {"starts":[10,17,23],"durations":[2,5,10]}'
  exit 2
fi

repo=$(realpath "$1")
openvins_root=$(realpath "$2")
bag=$(realpath "$3")
output_root=$(realpath -m "$4")
run_id=$5
stride=$6
base_regime=$7
blackout_json=$8
run_output="$output_root/raw/$run_id"
config_output="$run_output/config"

if [[ -e "$run_output" ]]; then
  echo "refusing to overwrite $run_output" >&2
  exit 1
fi
mkdir -p "$run_output"

starts=$(python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1])["starts"]))' "$blackout_json")
durations=$(python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1])["durations"]))' "$blackout_json")

python3 "$repo/scripts/pipeline/chance_constrained/propagation/ubuntu/prepare_openvins_config.py" \
  --openvins-root "$openvins_root" \
  --config-output "$config_output" \
  --run-output "$run_output"

filter_pid=""
recorder_pid=""
openvins_pid=""
cleanup() {
  for pid in "$openvins_pid" "$recorder_pid" "$filter_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python3 "$repo/scripts/pipeline/chance_constrained/propagation/ubuntu/frame_filter_node.py" \
  --ros-args \
  -r input:=/cam0/image_raw \
  -r output:=/openvins/cam0_filtered \
  -p stride:="$stride" \
  -p base_regime:="$base_regime" \
  -p "blackout_starts_s:='$starts'" \
  -p "blackout_durations_s:='$durations'" \
  -p schedule_csv:="$run_output/frame_schedule.csv" \
  >"$run_output/frame_filter.log" 2>&1 &
filter_pid=$!

python3 "$repo/scripts/pipeline/chance_constrained/propagation/ubuntu/odometry_recorder_node.py" \
  --ros-args \
  -r input:=/ov_msckf/odomimu \
  -p output_csv:="$run_output/odometry.csv" \
  >"$run_output/odometry_recorder.log" 2>&1 &
recorder_pid=$!

ros2 launch ov_msckf subscribe.launch.py \
  config_path:="$config_output/estimator_config.yaml" \
  use_stereo:=false max_cameras:=1 save_total_state:=false rviz_enable:=false \
  >"$run_output/openvins.log" 2>&1 &
openvins_pid=$!

sleep 3
ros2 bag play "$bag"
sleep 2
cleanup
trap - EXIT INT TERM

for required in diagnostics.csv frame_schedule.csv odometry.csv; do
  if [[ ! -s "$run_output/$required" ]]; then
    echo "missing or empty replay output: $run_output/$required" >&2
    exit 1
  fi
done
echo "$run_output"

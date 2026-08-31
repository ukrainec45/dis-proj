#!/usr/bin/env python3
"""Create a pinned monocular MUN-FRL Bell412 OpenVINS configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


OPENVINS_COMMIT = "93adc241390d13e99232652cf05cbe18a93c7bea"
CALIBRATION_URL = "https://mun-frl-vil-dataset.readthedocs.io/en/latest/Calibration.html"
VINS_CONFIG_COMMIT = "162f2eb6e3a8ba32a21d5091760523c1e07903f1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def replace_yaml_value(text, key, value):
    pattern = rf"(?m)^({re.escape(key)}\s*:\s*)[^#\r\n]+"
    updated, count = re.subn(
        pattern, lambda match: f"{match.group(1)}{value} ", text, count=1
    )
    if count != 1:
        raise ValueError(f"could not set {key} in official estimator_config.yaml")
    return updated


IMUCAM_CONFIG = """%YAML:1.0

cam0:
  # MUN-FRL body_T_camDown: camera coordinates to IMU/body coordinates.
  T_imu_cam:
    - [-0.01039363,  0.99994595,  0.00027614, -0.27939175]
    - [-0.99982944, -0.01038820, -0.01527021,  0.00394073]
    - [-0.01526652, -0.00043481,  0.99988337, -0.00039529]
    - [0.0, 0.0, 0.0, 1.0]
  cam_overlaps: []
  camera_model: pinhole
  # Kannala-Brandt k2..k5 are the four equidistant polynomial terms.
  distortion_coeffs: [-0.0764245, 0.0322856, -0.0445168, 0.0163317]
  distortion_model: equidistant
  intrinsics: [829.224, 829.454, 833.937, 562.509]
  resolution: [1440, 1080]
  rostopic: /openvins/cam0_filtered
  # image timestamp + shift = timestamp in the IMU clock.
  timeshift_cam_imu: -0.0297405129318
"""


IMU_CONFIG = """%YAML:1.0

imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: 0.08
  accelerometer_random_walk: 0.00004
  gyroscope_noise_density: 0.004
  gyroscope_random_walk: 0.0001
  rostopic: /imu/data
  time_offset: 0.0
  update_rate: 400.0
  model: "kalibr"
  Tw:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  R_IMUtoGYRO:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  Ta:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  R_IMUtoACC:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  Tg:
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.0]
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvins-root", required=True)
    parser.add_argument("--config-output", required=True)
    parser.add_argument("--run-output", required=True)
    args = parser.parse_args()

    root = Path(args.openvins_root).expanduser().resolve()
    source = root / "config" / "euroc_mav"
    output = Path(args.config_output).expanduser().resolve()
    run_output = Path(args.run_output).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"OpenVINS template configuration is missing: {source}")
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != OPENVINS_COMMIT:
        raise RuntimeError(f"expected OpenVINS {OPENVINS_COMMIT}, found {commit}")
    diagnostics_marker = "accepted_visual_features,tracked_features,visual_update_accepted"
    manager_source = (root / "ov_msckf" / "src" / "core" / "VioManager.cpp").read_text()
    if diagnostics_marker not in manager_source:
        raise RuntimeError("the v2.7 diagnostics patch has not been applied")

    estimator_source = source / "estimator_config.yaml"
    text = estimator_source.read_text(encoding="utf-8")
    replacements = {
        "use_stereo": "false",
        "max_cameras": "1",
        "calib_cam_extrinsics": "false",
        "calib_cam_intrinsics": "false",
        "calib_cam_timeoffset": "false",
        "gravity_mag": "9.803",
        "track_frequency": "15.0",
        "record_timing_information": "true",
        "record_timing_filepath": f'"{run_output / "diagnostics.csv"}"',
        # ROS2Visualizer cannot receive the three state paths through the
        # upstream subscribe launch file. Odometry and diagnostics are the
        # required replay artifacts, so disable this optional duplicate log.
        "save_total_state": "false",
        "filepath_est": f'"{run_output / "openvins_state.txt"}"',
        "filepath_std": f'"{run_output / "openvins_state_std.txt"}"',
        "filepath_gt": f'"{run_output / "openvins_groundtruth.txt"}"',
        "init_dyn_use": "true",
        "init_dyn_mle_max_time": "0.5",
    }
    for key, value in replacements.items():
        text = replace_yaml_value(text, key, value)

    if output.exists():
        raise FileExistsError(f"refusing to overwrite configuration directory: {output}")
    shutil.copytree(source, output)
    run_output.mkdir(parents=True, exist_ok=True)
    (output / "estimator_config.yaml").write_text(text, encoding="utf-8")
    (output / "kalibr_imucam_chain.yaml").write_text(IMUCAM_CONFIG, encoding="utf-8")
    (output / "kalibr_imu_chain.yaml").write_text(IMU_CONFIG, encoding="utf-8")

    manifest = {
        "format_version": 1,
        "dataset": "MUN-FRL bell412_dataset1 benchmarking sample",
        "openvins_commit": commit,
        "source_template": str(source),
        "generated_config": str(output),
        "run_output": str(run_output),
        "calibration_source": CALIBRATION_URL,
        "reference_vins_config_commit": VINS_CONFIG_COMMIT,
        "source_sha256": {path.name: sha256(path) for path in sorted(source.iterdir())},
        "generated_sha256": {path.name: sha256(path) for path in sorted(output.iterdir())},
        "fixed_parameters": replacements,
        "camera_topic": "/openvins/cam0_filtered",
        "imu_topic": "/imu/data",
        "lidar_used": False,
        "standalone_ins_model": False,
    }
    (output / "config_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(output / "estimator_config.yaml")


if __name__ == "__main__":
    main()

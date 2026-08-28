#!/usr/bin/env python3
"""Create a pinned monocular EuRoC OpenVINS configuration without editing upstream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


OPENVINS_COMMIT = "93adc241390d13e99232652cf05cbe18a93c7bea"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def replace_yaml_value(text, key, value):
    pattern = rf"(?m)^({re.escape(key)}\s*:\s*)[^#\r\n]+"
    updated, count = re.subn(
        pattern, lambda match: f"{match.group(1)}{value} ", text, count=1)
    if count != 1:
        raise ValueError(f"could not set {key} in official estimator_config.yaml")
    return updated


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
        raise FileNotFoundError(f"official EuRoC configuration is missing: {source}")
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "-C", str(root),
         "rev-parse", "HEAD"], text=True).strip()
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
        "record_timing_information": "true",
        "record_timing_filepath": f'"{run_output / "diagnostics.csv"}"',
        "save_total_state": "true",
        "filepath_est": f'"{run_output / "openvins_state.txt"}"',
        "filepath_std": f'"{run_output / "openvins_state_std.txt"}"',
        "filepath_gt": f'"{run_output / "openvins_groundtruth.txt"}"',
    }
    for key, value in replacements.items():
        text = replace_yaml_value(text, key, value)

    imucam_source = source / "kalibr_imucam_chain.yaml"
    imucam_text = imucam_source.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(?m)^(\s*rostopic:\s*)/cam0/image_raw\s*$",
        r"\g<1>/openvins/cam0_filtered", imucam_text, count=1)
    if count != 1:
        raise ValueError("could not remap cam0 in official calibration")

    if output.exists():
        raise FileExistsError(f"refusing to overwrite configuration directory: {output}")
    shutil.copytree(source, output)
    run_output.mkdir(parents=True, exist_ok=True)
    estimator = output / "estimator_config.yaml"
    estimator.write_text(text, encoding="utf-8")
    imucam = output / "kalibr_imucam_chain.yaml"
    imucam.write_text(updated, encoding="utf-8")

    manifest = {
        "format_version": 1,
        "openvins_commit": commit,
        "source_config": str(source),
        "generated_config": str(output),
        "run_output": str(run_output),
        "source_sha256": {path.name: sha256(path) for path in sorted(source.iterdir())},
        "generated_sha256": {path.name: sha256(path) for path in sorted(output.iterdir())},
        "fixed_parameters": replacements,
        "cam0_topic": "/openvins/cam0_filtered",
    }
    (output / "config_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(estimator)


if __name__ == "__main__":
    main()

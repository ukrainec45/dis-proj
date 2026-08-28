"""Convert Ubuntu/OpenVINS replay exports into the standardized run-log format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ..artifacts import source_fingerprint
from .artifacts import atomic_json
from .run_log import RunLog, save_run_log


ODOMETRY_BASE_FIELDS = (
    "timestamp_s", "qx", "qy", "qz", "qw", "px_m", "py_m", "pz_m",
    "vx_mps", "vy_mps", "vz_mps",
)
POSE_COVARIANCE_FIELDS = tuple(f"pose_cov_{index}" for index in range(36))
TWIST_COVARIANCE_FIELDS = tuple(f"twist_cov_{index}" for index in range(36))
ODOMETRY_FIELDS = ODOMETRY_BASE_FIELDS + POSE_COVARIANCE_FIELDS + TWIST_COVARIANCE_FIELDS


def _yaw_from_quaternion(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _rotation_matrices(qx, qy, qz, qw):
    """Return local-to-global rotation matrices for Hamilton quaternions."""
    quaternions = np.column_stack([qx, qy, qz, qw]).astype(float)
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("odometry contains an invalid orientation quaternion")
    quaternions /= norms[:, None]
    x, y, z, w = quaternions.T
    rotations = np.empty((len(quaternions), 3, 3), dtype=float)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotations


def _read_csv_columns(path, required_fields):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(required_fields) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{Path(path).name} is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{Path(path).name} contains no data rows")
    return {field: np.asarray([float(row[field]) for row in rows], dtype=float)
            for field in required_fields}


def load_odometry_csv(path):
    columns = _read_csv_columns(path, ODOMETRY_FIELDS)
    timestamps = columns["timestamp_s"]
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("odometry timestamps must be strictly increasing")
    rotations = _rotation_matrices(
        columns["qx"], columns["qy"], columns["qz"], columns["qw"])
    local_velocity = np.column_stack([
        columns["vx_mps"], columns["vy_mps"], columns["vz_mps"]])
    global_velocity = np.einsum("nij,nj->ni", rotations, local_velocity)
    estimated = np.column_stack([
        columns["px_m"], columns["py_m"], global_velocity[:, 0], global_velocity[:, 1],
        _yaw_from_quaternion(columns["qx"], columns["qy"], columns["qz"], columns["qw"]),
    ])
    pose = np.column_stack([columns[field] for field in POSE_COVARIANCE_FIELDS]).reshape(-1, 6, 6)
    twist = np.column_stack([columns[field] for field in TWIST_COVARIANCE_FIELDS]).reshape(-1, 6, 6)
    covariance = np.zeros((len(timestamps), 5, 5), dtype=float)
    covariance[:, :2, :2] = pose[:, :2, :2]
    global_velocity_covariance = np.einsum(
        "nij,njk,nlk->nil", rotations, twist[:, :3, :3], rotations)
    covariance[:, 2:4, 2:4] = global_velocity_covariance[:, :2, :2]
    covariance[:, 4, 4] = pose[:, 5, 5]
    for index in range(len(covariance)):
        covariance[index] = .5 * (covariance[index] + covariance[index].T)
        values, vectors = np.linalg.eigh(covariance[index])
        values = np.maximum(values, 0.0)
        covariance[index] = (vectors * values) @ vectors.T
    return timestamps, estimated, covariance


def load_euroc_ground_truth(path, query_timestamps):
    raw = np.loadtxt(Path(path), delimiter=",", comments="#")
    if raw.ndim != 2 or raw.shape[1] < 11:
        raise ValueError("EuRoC ground truth must contain at least 11 columns")
    timestamps = raw[:, 0] * 1e-9
    if query_timestamps[0] < timestamps[0] or query_timestamps[-1] > timestamps[-1]:
        raise ValueError("odometry timestamps extend beyond EuRoC ground-truth coverage")
    yaw = np.unwrap(_yaw_from_quaternion(raw[:, 5], raw[:, 6], raw[:, 7], raw[:, 4]))
    values = np.column_stack([raw[:, 1], raw[:, 2], raw[:, 8], raw[:, 9], yaw])
    interpolated = np.column_stack([
        np.interp(query_timestamps, timestamps, values[:, column])
        for column in range(values.shape[1])
    ])
    interpolated[:, 4] = (interpolated[:, 4] + np.pi) % (2 * np.pi) - np.pi
    return interpolated


def load_diagnostics_csv(path, odometry_timestamps):
    raw = np.loadtxt(Path(path), delimiter=",", comments="#")
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] < 9:
        raise ValueError(
            "OpenVINS diagnostics must include timestamp, timing and the three patched diagnostic columns")
    timestamps = raw[:, 0]
    total_ms = raw[:, -4] * 1000.0
    tracked = raw[:, -2].astype(int)
    accepted = raw[:, -1].astype(bool)
    previous = np.searchsorted(timestamps, odometry_timestamps, side="right") - 1
    valid = previous >= 0
    processing = np.full(len(odometry_timestamps), total_ms[0], dtype=float)
    tracked_out = np.full(len(odometry_timestamps), tracked[0], dtype=np.int32)
    processing[valid] = total_ms[previous[valid]]
    tracked_out[valid] = tracked[previous[valid]]
    accepted_out = np.zeros(len(odometry_timestamps), dtype=bool)
    accepted_times = timestamps[accepted]
    for timestamp in accepted_times:
        index = int(np.searchsorted(odometry_timestamps, timestamp, side="left"))
        if index < len(accepted_out):
            accepted_out[index] = True
    return accepted_out, tracked_out, processing


def load_frame_schedule(path, odometry_timestamps):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp_s", "camera_available", "visual_regime"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"frame schedule is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("frame schedule contains no rows")
    timestamps = np.asarray([float(row["timestamp_s"]) for row in rows])
    available = np.asarray([row["camera_available"].strip().lower() in {"1", "true", "yes"}
                            for row in rows], dtype=bool)
    regimes = np.asarray([row["visual_regime"] for row in rows], dtype=str)
    previous = np.searchsorted(timestamps, odometry_timestamps, side="right") - 1
    if np.all(previous < 0):
        raise ValueError("frame schedule starts after the complete odometry stream")
    if np.any(previous < 0):
        first_valid = int(np.flatnonzero(previous >= 0)[0])
        previous[:first_valid] = previous[first_valid]
    return available[previous], regimes[previous], timestamps


def _post_initialization_coverage(odometry_timestamps, schedule_timestamps):
    remaining_duration = float(schedule_timestamps[-1] - odometry_timestamps[0])
    if remaining_duration <= 0:
        raise ValueError("frame schedule does not extend beyond OpenVINS initialization")
    period = float(np.median(np.diff(odometry_timestamps)))
    if not np.isfinite(period) or period <= 0:
        raise ValueError("cannot infer the OpenVINS odometry publication period")
    expected_samples = max(1.0, remaining_duration / period + 1.0)
    return min(1.0, float(len(odometry_timestamps) / expected_samples))


def convert_openvins_run(*, odometry_csv, diagnostics_csv, ground_truth_csv,
                         frame_schedule_csv, output_path):
    timestamps, estimated, covariance = load_odometry_csv(odometry_csv)
    truth = load_euroc_ground_truth(ground_truth_csv, timestamps)
    accepted, tracked, processing = load_diagnostics_csv(diagnostics_csv, timestamps)
    available, regimes, schedule_timestamps = load_frame_schedule(
        frame_schedule_csv, timestamps)
    log = RunLog(
        timestamp_s=timestamps,
        estimated_state=estimated,
        ground_truth_state=truth,
        reported_covariance=covariance,
        initialized=np.ones(len(timestamps), dtype=bool),
        visual_update_accepted=accepted,
        tracked_feature_count=tracked,
        processing_time_ms=processing,
        visual_regime=regimes,
        forced_camera_available=available,
        post_initialization_coverage=_post_initialization_coverage(
            timestamps, schedule_timestamps),
    )
    save_run_log(output_path, log)
    manifest_path = Path(output_path).with_suffix(".manifest.json")
    manifest = {
        "format_version": 1,
        "artifact": Path(output_path).name,
        "source_fingerprints": {
            "odometry": source_fingerprint(odometry_csv),
            "diagnostics": source_fingerprint(diagnostics_csv),
            "ground_truth": source_fingerprint(ground_truth_csv),
            "frame_schedule": source_fingerprint(frame_schedule_csv),
        },
    }
    atomic_json(manifest_path, manifest)
    return log


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert one OpenVINS EuRoC replay to a standardized run log")
    parser.add_argument("--odometry", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--frame-schedule", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    log = convert_openvins_run(
        odometry_csv=args.odometry,
        diagnostics_csv=args.diagnostics,
        ground_truth_csv=args.ground_truth,
        frame_schedule_csv=args.frame_schedule,
        output_path=args.output,
    )
    print(f"wrote {len(log.timestamp_s)} synchronized samples to {args.output}")


if __name__ == "__main__":
    main()

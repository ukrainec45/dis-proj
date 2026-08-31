#!/usr/bin/env python3
"""Synchronize and rigidly align a MUN-FRL OpenVINS run to ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_columns(path, required):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return {
        field: np.asarray([float(row[field]) for row in rows], dtype=float)
        for field in required
    }


def _validate_timestamps(name, timestamps):
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{name} timestamps must be finite and strictly increasing")


def _rigid_alignment(source, target):
    """Find target ~= rotation @ source + translation, with scale fixed to one."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _rotation_matrices(quaternions):
    quaternions = np.asarray(quaternions, dtype=float)
    quaternions /= np.linalg.norm(quaternions, axis=1)[:, None]
    x, y, z, w = quaternions.T
    result = np.empty((len(quaternions), 3, 3), dtype=float)
    result[:, 0, 0] = 1 - 2 * (y*y + z*z)
    result[:, 0, 1] = 2 * (x*y - z*w)
    result[:, 0, 2] = 2 * (x*z + y*w)
    result[:, 1, 0] = 2 * (x*y + z*w)
    result[:, 1, 1] = 1 - 2 * (x*x + z*z)
    result[:, 1, 2] = 2 * (y*z - x*w)
    result[:, 2, 0] = 2 * (x*z - y*w)
    result[:, 2, 1] = 2 * (y*z + x*w)
    result[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return result


def _percentile(values, percentile):
    return float(np.percentile(values, percentile))


def evaluate(odometry, ground_truth, output_dir):
    odometry = Path(odometry).expanduser().resolve()
    ground_truth = Path(ground_truth).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    odom_fields = ("timestamp_s", "px_m", "py_m", "pz_m", "qx", "qy", "qz", "qw")
    truth_fields = ("timestamp_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
                    "orientation_valid")
    odom = _read_columns(odometry, odom_fields)
    truth = _read_columns(ground_truth, truth_fields)
    odom_time = odom["timestamp_s"]
    truth_time = truth["timestamp_s"]
    _validate_timestamps("odometry", odom_time)
    _validate_timestamps("ground truth", truth_time)

    overlap = (odom_time >= truth_time[0]) & (odom_time <= truth_time[-1])
    if np.count_nonzero(overlap) < 3:
        raise ValueError("fewer than three odometry samples overlap ground truth")
    timestamps = odom_time[overlap]
    estimated = np.column_stack([odom[field][overlap] for field in ("px_m", "py_m", "pz_m")])
    reference = np.column_stack([
        np.interp(timestamps, truth_time, truth[field]) for field in ("x_m", "y_m", "z_m")
    ])
    rotation, translation = _rigid_alignment(estimated, reference)
    aligned = (rotation @ estimated.T).T + translation
    residual = aligned - reference
    error_3d = np.linalg.norm(residual, axis=1)
    error_horizontal = np.linalg.norm(residual[:, :2], axis=1)
    path_length = float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum())

    metrics = {
        "format_version": 1,
        "odometry": str(odometry),
        "ground_truth": str(ground_truth),
        "sample_count": int(len(timestamps)),
        "overlap_start_s": float(timestamps[0]),
        "overlap_end_s": float(timestamps[-1]),
        "overlap_duration_s": float(timestamps[-1] - timestamps[0]),
        "alignment": "full-trajectory rigid SE(3), scale fixed to 1",
        "alignment_rotation": rotation.tolist(),
        "alignment_translation_m": translation.tolist(),
        "reference_path_length_m": path_length,
        "position_3d_rmse_m": float(np.sqrt(np.mean(error_3d**2))),
        "position_3d_median_m": _percentile(error_3d, 50),
        "position_3d_p95_m": _percentile(error_3d, 95),
        "position_3d_max_m": float(error_3d.max()),
        "position_3d_final_m": float(error_3d[-1]),
        "horizontal_rmse_m": float(np.sqrt(np.mean(error_horizontal**2))),
        "final_error_over_path_length": (
            float(error_3d[-1] / path_length) if path_length > 0 else None),
    }

    orientation_valid = bool(np.all(truth["orientation_valid"] > 0.5))
    orientation_error_deg = None
    if orientation_valid:
        nearest = np.searchsorted(truth_time, timestamps, side="left")
        nearest = np.clip(nearest, 0, len(truth_time) - 1)
        previous = np.maximum(nearest - 1, 0)
        use_previous = (np.abs(timestamps - truth_time[previous])
                        < np.abs(truth_time[nearest] - timestamps))
        nearest[use_previous] = previous[use_previous]
        estimated_q = np.column_stack([odom[field][overlap] for field in ("qx", "qy", "qz", "qw")])
        truth_q = np.column_stack([truth[field][nearest] for field in ("qx", "qy", "qz", "qw")])
        estimated_rotation = np.einsum("ij,njk->nik", rotation, _rotation_matrices(estimated_q))
        truth_rotation = _rotation_matrices(truth_q)
        relative = np.einsum("nij,njk->nik", np.transpose(truth_rotation, (0, 2, 1)),
                             estimated_rotation)
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        orientation_error_deg = np.degrees(np.arccos(cosine))
        metrics.update({
            "orientation_rmse_deg": float(np.sqrt(np.mean(orientation_error_deg**2))),
            "orientation_median_deg": _percentile(orientation_error_deg, 50),
            "orientation_p95_deg": _percentile(orientation_error_deg, 95),
        })
    else:
        metrics["orientation_metrics"] = "unavailable in selected ground truth"

    output_dir.mkdir(parents=True)
    with (output_dir / "trajectory_errors.csv").open("x", newline="", encoding="utf-8") as stream:
        fields = ["timestamp_s", "estimated_x_m", "estimated_y_m", "estimated_z_m",
                  "reference_x_m", "reference_y_m", "reference_z_m",
                  "error_x_m", "error_y_m", "error_z_m", "error_3d_m",
                  "error_horizontal_m"]
        if orientation_error_deg is not None:
            fields.append("orientation_error_deg")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, timestamp in enumerate(timestamps):
            row = {
                "timestamp_s": f"{timestamp:.9f}",
                "estimated_x_m": aligned[index, 0], "estimated_y_m": aligned[index, 1],
                "estimated_z_m": aligned[index, 2], "reference_x_m": reference[index, 0],
                "reference_y_m": reference[index, 1], "reference_z_m": reference[index, 2],
                "error_x_m": residual[index, 0], "error_y_m": residual[index, 1],
                "error_z_m": residual[index, 2], "error_3d_m": error_3d[index],
                "error_horizontal_m": error_horizontal[index],
            }
            if orientation_error_deg is not None:
                row["orientation_error_deg"] = orientation_error_deg[index]
            writer.writerow(row)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate an OpenVINS MUN-FRL replay")
    parser.add_argument("--odometry", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    metrics = evaluate(args.odometry, args.ground_truth, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

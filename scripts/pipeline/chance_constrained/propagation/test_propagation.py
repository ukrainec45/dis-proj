from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from .build_model import build_propagation_model
from .config import ModePolicy, PropagationBuildConfig, RunSpec, ValidationPolicy
from .contracts import PropagationMode, PropagationQuery
from .fitting import fit_records
from .model import InsOpenVinsPropagationModel
from .openvins_adapter import ODOMETRY_FIELDS, load_odometry_csv
from .run_log import RunLog, load_run_log, save_run_log
from .segments import SegmentRecord, classify_modes, extract_segments


MODE_POLICY = ModePolicy(.25, .6, 2.0, .6)
VALIDATION_POLICY = ValidationPolicy(.95, .9, 1, 1)


def make_log(*, blackout=False, drift_rate=.01):
    timestamps = np.arange(0.0, 12.0001, .1)
    count = len(timestamps)
    available = np.ones(count, dtype=bool)
    regime = np.full(count, "nominal_20hz", dtype="<U24")
    if blackout:
        unavailable = (timestamps >= 2.0) & (timestamps < 10.0)
        available[unavailable] = False
        regime[unavailable] = "blackout"
    accepted = available.copy()
    truth = np.column_stack([
        timestamps, np.zeros(count), np.ones(count), np.zeros(count),
        np.zeros(count),
    ])
    error_rate = np.full(count, drift_rate)
    if blackout:
        error_rate[~available] = .5
    accumulated_error = np.cumsum(error_rate) * .1
    estimated = truth.copy()
    estimated[:, 0] += accumulated_error
    estimated[:, 2] += error_rate
    covariance = np.repeat((np.eye(5) * .05)[None, :, :], count, axis=0)
    return RunLog(
        timestamp_s=timestamps,
        estimated_state=estimated,
        ground_truth_state=truth,
        reported_covariance=covariance,
        initialized=np.ones(count, dtype=bool),
        visual_update_accepted=accepted,
        tracked_feature_count=np.where(available, 80, 0),
        processing_time_ms=np.full(count, 4.0),
        visual_regime=regime,
        forced_camera_available=available,
        post_initialization_coverage=1.0,
    )


def make_config(tmp_path, runs, *, durations=(2.0,)):
    return PropagationBuildConfig(
        config_path=tmp_path / "config.json",
        model_id="test-model",
        durations_s=durations,
        quantile_levels=(.5, .9, .95),
        minimum_fit_samples=2,
        mode_policy=MODE_POLICY,
        validation=VALIDATION_POLICY,
        runs=tuple(runs),
        source_manifest_path=None,
        raw={"schema_version": 1},
    )


def test_query_rejects_invalid_covariance():
    with pytest.raises(ValueError, match="positive semidefinite"):
        PropagationQuery(2.0, np.diag([1, 1, 1, 1, -1]),
                         PropagationMode.VIO_OPERATIONAL, "nominal_20hz")


def test_run_log_round_trip_and_mode_recovery(tmp_path):
    log = make_log(blackout=True)
    path = tmp_path / "run.npz"
    save_run_log(path, log)
    loaded = load_run_log(path)
    modes = classify_modes(loaded, MODE_POLICY)
    assert loaded.post_initialization_coverage == 1.0
    assert np.all(modes[(loaded.timestamp_s >= 2) & (loaded.timestamp_s < 10)] == "ins_only")
    assert modes[np.searchsorted(loaded.timestamp_s, 10.0)] == "reinitialized"
    assert modes[-1] == "vio_operational"


def test_relative_segment_is_aligned_at_window_start(tmp_path):
    log = make_log(drift_rate=.1)
    run = RunSpec("run", "dataset", "nominal_20hz", "fit", tmp_path / "unused.npz")
    records = extract_segments(log, run, make_config(tmp_path, [run]))
    assert records
    np.testing.assert_allclose(records[0].relative_error[:2], [0.2, 0.0], atol=1e-10)
    np.testing.assert_allclose(records[0].relative_error[2:4], [0.0, 0.0], atol=1e-10)


def test_odometry_adapter_rotates_local_velocity_and_covariance(tmp_path):
    path = tmp_path / "odometry.csv"
    pose_covariance = np.eye(6).reshape(-1)
    twist_covariance = np.diag([4.0, 1.0, 9.0, 1.0, 1.0, 1.0]).reshape(-1)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ODOMETRY_FIELDS)
        writer.writeheader()
        for timestamp in (1.0, 1.1):
            row = {
                "timestamp_s": timestamp,
                "qx": 0.0,
                "qy": 0.0,
                "qz": np.sqrt(.5),
                "qw": np.sqrt(.5),
                "px_m": 0.0,
                "py_m": 0.0,
                "pz_m": 0.0,
                "vx_mps": 1.0,
                "vy_mps": 0.0,
                "vz_mps": 0.0,
            }
            row.update({f"pose_cov_{index}": value
                        for index, value in enumerate(pose_covariance)})
            row.update({f"twist_cov_{index}": value
                        for index, value in enumerate(twist_covariance)})
            writer.writerow(row)
    _, estimated, covariance = load_odometry_csv(path)
    np.testing.assert_allclose(estimated[:, 2:4], [[0, 1], [0, 1]], atol=1e-12)
    np.testing.assert_allclose(covariance[0, 2:4, 2:4], np.diag([1, 4]), atol=1e-12)


def test_fitted_model_propagates_incoming_covariance(tmp_path):
    runs = [RunSpec("run", "dataset", "nominal_20hz", "fit", tmp_path / "unused")]
    config = make_config(tmp_path, runs, durations=(2.0, 5.0))
    records = []
    for duration in config.durations_s:
        for index in range(4):
            records.append(SegmentRecord(
                run_id="run", dataset_id="dataset", split="fit",
                visual_regime="nominal_20hz", duration_s=duration,
                start_timestamp_s=index * duration,
                end_timestamp_s=(index + 1) * duration,
                start_mode="vio_operational", end_mode="vio_operational",
                relative_error=np.asarray([.01 * index, 0, 0, 0, 0]),
                reported_covariance=np.eye(5), mean_processing_time_ms=1.0,
                minimum_tracked_features=20,
            ))
    arrays = fit_records(records, config)
    arrays["model_id"] = np.asarray("test-model")
    arrays["validation_passed"] = np.asarray(True)
    model = InsOpenVinsPropagationModel(arrays)
    incoming = np.eye(5) * .1
    outcome = model.propagate(PropagationQuery(
        3.5, incoming, "vio_operational", "nominal_20hz"))
    assert outcome.outgoing_covariance[0, 0] > incoming[0, 0]
    assert np.isclose(sum(outcome.outgoing_mode_probabilities.values()), 1.0)


def test_build_pipeline_is_deterministic_and_passes_fixture_checks(tmp_path, monkeypatch):
    run_definitions = [
        ("fit_nominal", "V2_01_easy", "nominal_20hz", "fit", False),
        ("fit_blackout", "V2_02_medium", "nominal_20hz", "fit", True),
        ("validation_nominal", "V2_03_difficult", "nominal_20hz", "validation", False),
        ("validation_blackout", "V2_03_difficult", "nominal_20hz", "validation", True),
    ]
    runs = []
    for run_id, dataset, regime, split, blackout in run_definitions:
        path = tmp_path / f"{run_id}.npz"
        save_run_log(path, make_log(blackout=blackout))
        path.with_suffix(".manifest.json").write_text(json.dumps({
            "format_version": 1,
            "artifact": path.name,
            "source_fingerprints": {
                "odometry": {}, "diagnostics": {}, "ground_truth": {},
                "frame_schedule": {},
            },
        }), encoding="utf-8")
        runs.append({
            "id": run_id, "dataset_id": dataset, "visual_regime": regime,
            "split": split, "run_log": str(path),
        })
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(json.dumps({
        "format_version": 1,
        "datasets": {
            "V2_01_easy": {}, "V2_02_medium": {}, "V2_03_difficult": {},
        },
    }), encoding="utf-8")
    config = {
        "schema_version": 1,
        "model": {
            "id": "fixture-only", "durations_s": [2],
            "quantile_levels": [.5, .9, .95], "minimum_fit_samples": 2,
        },
        "mode_policy": {
            "accepted_update_timeout_s": .25,
            "degraded_update_timeout_s": .6,
            "recovery_duration_s": 2,
            "maximum_recovery_gap_s": .6,
        },
        "validation": {
            "confidence": .95, "minimum_coverage": .7,
            "minimum_group_samples": 1, "minimum_evaluable_groups": 1,
        },
        "provenance": {
            "openvins_version": "fixture",
            "openvins_commit": "fixture",
            "source_manifest": str(source_manifest),
        },
        "runs": runs,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _, summary = build_propagation_model(config_path, tmp_path / "first")
    build_propagation_model(config_path, tmp_path / "second")
    assert summary["validation"]["passed"]
    assert (tmp_path / "first" / "segments.csv").read_bytes() == (
        tmp_path / "second" / "segments.csv").read_bytes()
    assert json.loads((tmp_path / "first" / "manifest.json").read_text()) == json.loads(
        (tmp_path / "second" / "manifest.json").read_text())
    InsOpenVinsPropagationModel.load(tmp_path / "first" / "ins_openvins_model.npz")

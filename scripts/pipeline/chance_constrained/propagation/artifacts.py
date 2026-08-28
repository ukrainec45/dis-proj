"""Persistence and validation for versioned INS/OpenVINS model artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np

from ..artifacts import source_fingerprint
from .config import MODEL_ARTIFACT_VERSION, PropagationBuildConfig
from .contracts import ERROR_STATE_DIM, ERROR_STATE_FIELDS, PROPAGATION_MODES


REQUIRED_MODEL_ARRAYS = {
    "artifact_version", "model_id", "state_fields", "mode_names",
    "openvins_version", "openvins_commit", "dataset_ids",
    "configuration_sha256", "degradation_mode_ids", "baseline_only",
    "target_platform_calibrated",
    "visual_regime_ids", "duration_bins_s", "quantile_levels",
    "error_metric_names", "sample_count", "drift_mean",
    "process_noise_covariance", "absolute_error_quantiles",
    "reported_covariance_inflation_95", "mode_transition_count",
    "mode_transition_probability", "mode_transition_lcb",
    "mode_transition_ucb", "supported", "validation_passed",
}


def atomic_npz(path, arrays):
    path = Path(path)
    handle, temporary = tempfile.mkstemp(suffix=".npz", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_json(path, payload):
    path = Path(path)
    handle, temporary = tempfile.mkstemp(suffix=".json", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_manifest(config: PropagationBuildConfig, validation):
    sources = {}
    for run in config.runs:
        sources[f"{run.run_id}:run_log"] = source_fingerprint(run.run_log_path)
        if run.run_manifest_path is not None:
            sources[f"{run.run_id}:capture_manifest"] = source_fingerprint(
                run.run_manifest_path)
    if config.source_manifest_path is not None:
        sources["dataset_and_openvins_source_manifest"] = source_fingerprint(
            config.source_manifest_path)
    return {
        "artifact_format_version": MODEL_ARTIFACT_VERSION,
        "configuration_schema_version": config.raw["schema_version"],
        "artifact": "ins_openvins_model.npz",
        "model_id": config.model_id,
        "baseline_only": True,
        "target_platform_calibrated": False,
        "validation": validation,
        "configuration_path": str(config.config_path),
        "configuration": config.raw,
        "source_fingerprints": sources,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


def validate_model_arrays(arrays):
    missing = REQUIRED_MODEL_ARRAYS - set(arrays)
    if missing:
        raise ValueError(f"model artifact is missing arrays: {', '.join(sorted(missing))}")
    version = int(np.asarray(arrays["artifact_version"]).item())
    if version != MODEL_ARTIFACT_VERSION:
        raise ValueError(f"unsupported INS/OpenVINS artifact version: {version}")
    state_fields = tuple(map(str, np.asarray(arrays["state_fields"])))
    modes = tuple(map(str, np.asarray(arrays["mode_names"])))
    if state_fields != ERROR_STATE_FIELDS:
        raise ValueError("model artifact uses an unsupported error-state order")
    if modes != PROPAGATION_MODES:
        raise ValueError("model artifact uses an unsupported propagation-mode order")
    regimes = np.asarray(arrays["visual_regime_ids"])
    degradation_modes = np.asarray(arrays["degradation_mode_ids"])
    durations = np.asarray(arrays["duration_bins_s"], dtype=float)
    quantiles = np.asarray(arrays["quantile_levels"], dtype=float)
    if regimes.ndim != 1 or not len(regimes):
        raise ValueError("visual_regime_ids must be non-empty")
    if tuple(map(str, degradation_modes)) != tuple(map(str, regimes)):
        raise ValueError("degradation_mode_ids must match the visual-regime axis")
    if not bool(np.asarray(arrays["baseline_only"]).item()):
        raise ValueError("EuRoC INS/OpenVINS artifact must be marked as baseline-only")
    if bool(np.asarray(arrays["target_platform_calibrated"]).item()):
        raise ValueError("EuRoC baseline cannot be marked as target-platform calibrated")
    if durations.ndim != 1 or not len(durations) or np.any(np.diff(durations) <= 0):
        raise ValueError("duration_bins_s must be finite and strictly increasing")
    shape = (len(regimes), len(modes), len(durations))
    expected = {
        "sample_count": shape,
        "drift_mean": shape + (ERROR_STATE_DIM,),
        "process_noise_covariance": shape + (ERROR_STATE_DIM, ERROR_STATE_DIM),
        "absolute_error_quantiles": shape + (len(quantiles), 3),
        "reported_covariance_inflation_95": shape,
        "mode_transition_count": shape + (len(modes),),
        "mode_transition_probability": shape + (len(modes),),
        "mode_transition_lcb": shape + (len(modes),),
        "mode_transition_ucb": shape + (len(modes),),
        "supported": shape,
    }
    for name, expected_shape in expected.items():
        if np.asarray(arrays[name]).shape != expected_shape:
            raise ValueError(f"{name} has shape {np.asarray(arrays[name]).shape}; expected {expected_shape}")
    for index in np.argwhere(np.asarray(arrays["supported"], dtype=bool)):
        selected = tuple(map(int, index))
        covariance = np.asarray(arrays["process_noise_covariance"])[selected]
        probabilities = np.asarray(arrays["mode_transition_probability"])[selected]
        if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T):
            raise ValueError("supported model groups must contain finite symmetric covariance")
        if np.linalg.eigvalsh(covariance).min() < -1e-9:
            raise ValueError("supported process-noise covariance must be positive semidefinite")
        if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("supported mode-transition rows must be finite and sum to one")
    return True


def load_model_artifact(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_model_arrays(arrays)
    return arrays

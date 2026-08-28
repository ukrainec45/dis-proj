"""Standardized, platform-independent output of one OpenVINS replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import ERROR_STATE_DIM


RUN_LOG_VERSION = 1
REQUIRED_RUN_ARRAYS = {
    "format_version", "timestamp_s", "estimated_state", "ground_truth_state",
    "reported_covariance", "initialized", "visual_update_accepted",
    "tracked_feature_count", "processing_time_ms", "visual_regime",
    "forced_camera_available", "post_initialization_coverage",
}


@dataclass(frozen=True)
class RunLog:
    timestamp_s: np.ndarray
    estimated_state: np.ndarray
    ground_truth_state: np.ndarray
    reported_covariance: np.ndarray
    initialized: np.ndarray
    visual_update_accepted: np.ndarray
    tracked_feature_count: np.ndarray
    processing_time_ms: np.ndarray
    visual_regime: np.ndarray
    forced_camera_available: np.ndarray
    post_initialization_coverage: float

    def __post_init__(self):
        timestamps = np.asarray(self.timestamp_s, dtype=float)
        count = len(timestamps)
        if timestamps.ndim != 1 or count < 2 or not np.isfinite(timestamps).all():
            raise ValueError("timestamp_s must be a finite vector with at least two samples")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamp_s must be strictly increasing")
        states = {
            "estimated_state": np.asarray(self.estimated_state, dtype=float),
            "ground_truth_state": np.asarray(self.ground_truth_state, dtype=float),
        }
        for label, value in states.items():
            if value.shape != (count, ERROR_STATE_DIM) or not np.isfinite(value).all():
                raise ValueError(f"{label} must have shape (sample, {ERROR_STATE_DIM}) and be finite")
        covariance = np.asarray(self.reported_covariance, dtype=float)
        if covariance.shape != (count, ERROR_STATE_DIM, ERROR_STATE_DIM):
            raise ValueError("reported_covariance has an invalid shape")
        if not np.isfinite(covariance).all():
            raise ValueError("reported_covariance must be finite")
        if not np.allclose(covariance, np.swapaxes(covariance, 1, 2), atol=1e-9):
            raise ValueError("reported_covariance must be symmetric")
        vector_names = (
            "initialized", "visual_update_accepted", "tracked_feature_count",
            "processing_time_ms", "visual_regime", "forced_camera_available",
        )
        for name in vector_names:
            if np.asarray(getattr(self, name)).shape != (count,):
                raise ValueError(f"{name} must have one value per timestamp")
        processing = np.asarray(self.processing_time_ms, dtype=float)
        if not np.isfinite(processing).all() or np.any(processing < 0):
            raise ValueError("processing_time_ms must be finite and non-negative")
        coverage = float(self.post_initialization_coverage)
        if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("post_initialization_coverage must lie inside [0, 1]")


def save_run_log(path, log: RunLog):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray(RUN_LOG_VERSION, dtype=np.int16),
        timestamp_s=np.asarray(log.timestamp_s, dtype=float),
        estimated_state=np.asarray(log.estimated_state, dtype=float),
        ground_truth_state=np.asarray(log.ground_truth_state, dtype=float),
        reported_covariance=np.asarray(log.reported_covariance, dtype=float),
        initialized=np.asarray(log.initialized, dtype=bool),
        visual_update_accepted=np.asarray(log.visual_update_accepted, dtype=bool),
        tracked_feature_count=np.asarray(log.tracked_feature_count, dtype=np.int32),
        processing_time_ms=np.asarray(log.processing_time_ms, dtype=float),
        visual_regime=np.asarray(log.visual_regime, dtype=str),
        forced_camera_available=np.asarray(log.forced_camera_available, dtype=bool),
        post_initialization_coverage=np.asarray(
            log.post_initialization_coverage, dtype=float),
    )


def load_run_log(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_RUN_ARRAYS - set(archive.files)
        if missing:
            raise ValueError(f"run log is missing arrays: {', '.join(sorted(missing))}")
        version = int(np.asarray(archive["format_version"]).item())
        if version != RUN_LOG_VERSION:
            raise ValueError(f"unsupported run-log format version: {version}")
        values = {name: archive[name] for name in REQUIRED_RUN_ARRAYS
                  if name != "format_version"}
        values["post_initialization_coverage"] = float(
            np.asarray(values["post_initialization_coverage"]).item())
    return RunLog(**values)

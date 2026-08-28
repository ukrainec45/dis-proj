"""Mode classification and relative-error segmentation for OpenVINS replays."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ModePolicy, PropagationBuildConfig, RunSpec
from .contracts import ERROR_STATE_DIM, ERROR_STATE_FIELDS, PropagationMode
from .run_log import RunLog


def wrap_angle_rad(value):
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def classify_modes(log: RunLog, policy: ModePolicy):
    """Classify every replay timestamp using accepted updates and forced outages."""
    modes = np.full(len(log.timestamp_s), PropagationMode.INS_ONLY.value, dtype="<U24")
    last_accepted = -np.inf
    recovery_start = None
    recovering = False
    previous_available = bool(log.forced_camera_available[0])

    for index, timestamp in enumerate(log.timestamp_s):
        if not bool(log.initialized[index]):
            previous_available = bool(log.forced_camera_available[index])
            continue

        available = bool(log.forced_camera_available[index])
        if previous_available and not available:
            recovering = True
            recovery_start = None
        elif not previous_available and available:
            recovering = True
            recovery_start = None
        previous_available = available

        if bool(log.visual_update_accepted[index]):
            gap = timestamp - last_accepted
            if recovery_start is None or gap > policy.maximum_recovery_gap_s:
                recovery_start = float(timestamp)
            last_accepted = float(timestamp)

        if not available:
            modes[index] = PropagationMode.INS_ONLY.value
            continue

        update_age = float(timestamp - last_accepted)
        if update_age > policy.degraded_update_timeout_s:
            modes[index] = PropagationMode.INS_ONLY.value
            continue

        if recovering:
            if (recovery_start is not None and
                    timestamp - recovery_start >= policy.recovery_duration_s):
                recovering = False
            else:
                modes[index] = PropagationMode.REINITIALIZED.value
                continue

        regime = str(log.visual_regime[index])
        degraded_input = regime.startswith("degraded_")
        if degraded_input or update_age > policy.accepted_update_timeout_s:
            modes[index] = PropagationMode.VIO_DEGRADED.value
        else:
            modes[index] = PropagationMode.VIO_OPERATIONAL.value
    return modes


@dataclass(frozen=True)
class SegmentRecord:
    run_id: str
    dataset_id: str
    split: str
    visual_regime: str
    duration_s: float
    start_timestamp_s: float
    end_timestamp_s: float
    start_mode: str
    end_mode: str
    relative_error: np.ndarray
    reported_covariance: np.ndarray
    mean_processing_time_ms: float
    minimum_tracked_features: int

    @property
    def horizontal_position_error_m(self):
        return float(np.linalg.norm(self.relative_error[:2]))

    @property
    def horizontal_velocity_error_mps(self):
        return float(np.linalg.norm(self.relative_error[2:4]))

    @property
    def absolute_yaw_error_rad(self):
        return float(abs(self.relative_error[4]))


def _relative_delta(state, start, end):
    yaw0 = float(state[start, 4])
    cosine, sine = np.cos(yaw0), np.sin(yaw0)
    world_to_start = np.asarray([[cosine, sine], [-sine, cosine]])
    position_delta = world_to_start @ (state[end, :2] - state[start, :2])
    velocity_delta = world_to_start @ (state[end, 2:4] - state[start, 2:4])
    yaw_delta = float(wrap_angle_rad(state[end, 4] - state[start, 4]))
    return np.r_[position_delta, velocity_delta, yaw_delta]


def _segment_regime(log, start, end, default_regime):
    available = np.asarray(log.forced_camera_available[start:end + 1], dtype=bool)
    if len(available) and np.count_nonzero(~available) >= len(available) / 2.0:
        return "blackout"
    values = np.asarray(log.visual_regime[start:end + 1], dtype=str)
    if len(values):
        unique, counts = np.unique(values, return_counts=True)
        selected = str(unique[int(np.argmax(counts))])
        if selected:
            return selected
    return default_regime


def extract_segments(log: RunLog, run: RunSpec,
                     config: PropagationBuildConfig):
    """Create deterministic non-overlapping windows at every configured duration."""
    modes = classify_modes(log, config.mode_policy)
    valid = np.flatnonzero(np.asarray(log.initialized, dtype=bool))
    if not len(valid):
        raise ValueError(f"run {run.run_id} contains no initialized OpenVINS samples")
    first_time = float(log.timestamp_s[valid[0]])
    last_time = float(log.timestamp_s[valid[-1]])
    records = []

    for requested_duration in config.durations_s:
        start_time = first_time
        while start_time + requested_duration <= last_time + 1e-9:
            start = int(np.searchsorted(log.timestamp_s, start_time, side="left"))
            end_target = start_time + requested_duration
            end = int(np.searchsorted(log.timestamp_s, end_target, side="left"))
            if end >= len(log.timestamp_s):
                break
            actual_duration = float(log.timestamp_s[end] - log.timestamp_s[start])
            tolerance = max(.02, requested_duration * .01)
            if (bool(log.initialized[start]) and bool(log.initialized[end]) and
                    abs(actual_duration - requested_duration) <= tolerance):
                estimated_delta = _relative_delta(log.estimated_state, start, end)
                truth_delta = _relative_delta(log.ground_truth_state, start, end)
                error = estimated_delta - truth_delta
                error[4] = float(wrap_angle_rad(error[4]))
                covariance = np.asarray(log.reported_covariance[end], dtype=float)
                processing = np.asarray(log.processing_time_ms[start:end + 1], dtype=float)
                tracked = np.asarray(log.tracked_feature_count[start:end + 1], dtype=int)
                records.append(SegmentRecord(
                    run_id=run.run_id,
                    dataset_id=run.dataset_id,
                    split=run.split,
                    visual_regime=_segment_regime(log, start, end, run.visual_regime),
                    duration_s=float(requested_duration),
                    start_timestamp_s=float(log.timestamp_s[start]),
                    end_timestamp_s=float(log.timestamp_s[end]),
                    start_mode=str(modes[start]),
                    end_mode=str(modes[end]),
                    relative_error=np.asarray(error, dtype=float),
                    reported_covariance=covariance,
                    mean_processing_time_ms=float(np.nanmean(processing)),
                    minimum_tracked_features=int(np.min(tracked)),
                ))
            start_time += requested_duration
    return records


SEGMENT_FIELDS = (
    "run_id", "dataset_id", "split", "visual_regime", "duration_s",
    "start_timestamp_s", "end_timestamp_s", "start_mode", "end_mode",
    *ERROR_STATE_FIELDS,
    "horizontal_position_error_m", "horizontal_velocity_error_mps",
    "absolute_yaw_error_rad", "mean_processing_time_ms",
    "minimum_tracked_features",
    *(f"reported_covariance_{row}_{column}"
      for row in range(ERROR_STATE_DIM) for column in range(ERROR_STATE_DIM)),
)


def write_segments_csv(path, records):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SEGMENT_FIELDS)
        writer.writeheader()
        for record in records:
            row = {
                "run_id": record.run_id,
                "dataset_id": record.dataset_id,
                "split": record.split,
                "visual_regime": record.visual_regime,
                "duration_s": record.duration_s,
                "start_timestamp_s": record.start_timestamp_s,
                "end_timestamp_s": record.end_timestamp_s,
                "start_mode": record.start_mode,
                "end_mode": record.end_mode,
                "horizontal_position_error_m": record.horizontal_position_error_m,
                "horizontal_velocity_error_mps": record.horizontal_velocity_error_mps,
                "absolute_yaw_error_rad": record.absolute_yaw_error_rad,
                "mean_processing_time_ms": record.mean_processing_time_ms,
                "minimum_tracked_features": record.minimum_tracked_features,
            }
            row.update(zip(ERROR_STATE_FIELDS, record.relative_error))
            for matrix_row in range(ERROR_STATE_DIM):
                for matrix_column in range(ERROR_STATE_DIM):
                    row[f"reported_covariance_{matrix_row}_{matrix_column}"] = (
                        record.reported_covariance[matrix_row, matrix_column])
            writer.writerow(row)

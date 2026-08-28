"""Validated configuration for building an empirical INS/OpenVINS model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


CONFIG_SCHEMA_VERSION = 1
MODEL_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    dataset_id: str
    visual_regime: str
    split: str
    run_log_path: Path
    run_manifest_path: Path | None = None


@dataclass(frozen=True)
class ModePolicy:
    accepted_update_timeout_s: float
    degraded_update_timeout_s: float
    recovery_duration_s: float
    maximum_recovery_gap_s: float


@dataclass(frozen=True)
class ValidationPolicy:
    confidence: float
    minimum_coverage: float
    minimum_group_samples: int
    minimum_evaluable_groups: int


@dataclass(frozen=True)
class PropagationBuildConfig:
    config_path: Path
    model_id: str
    durations_s: tuple[float, ...]
    quantile_levels: tuple[float, ...]
    minimum_fit_samples: int
    mode_policy: ModePolicy
    validation: ValidationPolicy
    runs: tuple[RunSpec, ...]
    source_manifest_path: Path | None
    raw: dict[str, Any]

    @property
    def visual_regimes(self):
        return tuple(dict.fromkeys(run.visual_regime for run in self.runs))


def _object(parent, key):
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _positive_values(parent, key, *, inside_unit=False):
    values = parent.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} must be a non-empty JSON array")
    result = tuple(float(value) for value in values)
    if not np.isfinite(result).all() or any(value <= 0 for value in result):
        raise ValueError(f"{key} values must be finite and positive")
    if inside_unit and any(value >= 1 for value in result):
        raise ValueError(f"{key} values must lie inside (0, 1)")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError(f"{key} values must be strictly increasing")
    return result


def _finite_positive(parent, key):
    value = float(parent.get(key, 0.0))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and positive")
    return value


def _resolve_path(value, config_path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("run_log must be a non-empty path")
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_build_config(path, *, require_run_logs=True):
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a JSON object")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")

    model = _object(raw, "model")
    model_id = str(model.get("id", "")).strip()
    if not model_id:
        raise ValueError("model.id must be non-empty")
    durations = _positive_values(model, "durations_s")
    quantiles = _positive_values(model, "quantile_levels", inside_unit=True)
    minimum_fit_samples = int(model.get("minimum_fit_samples", 2))
    if minimum_fit_samples < 2:
        raise ValueError("model.minimum_fit_samples must be at least 2")

    mode_raw = _object(raw, "mode_policy")
    mode_policy = ModePolicy(
        accepted_update_timeout_s=_finite_positive(mode_raw, "accepted_update_timeout_s"),
        degraded_update_timeout_s=_finite_positive(mode_raw, "degraded_update_timeout_s"),
        recovery_duration_s=_finite_positive(mode_raw, "recovery_duration_s"),
        maximum_recovery_gap_s=_finite_positive(mode_raw, "maximum_recovery_gap_s"),
    )
    if mode_policy.accepted_update_timeout_s > mode_policy.degraded_update_timeout_s:
        raise ValueError("accepted_update_timeout_s must not exceed degraded_update_timeout_s")

    validation_raw = _object(raw, "validation")
    confidence = float(validation_raw.get("confidence", .95))
    minimum_coverage = float(validation_raw.get("minimum_coverage", .9))
    minimum_group_samples = int(validation_raw.get("minimum_group_samples", 3))
    minimum_evaluable_groups = int(validation_raw.get("minimum_evaluable_groups", 1))
    if not 0 < confidence < 1 or not 0 < minimum_coverage <= 1:
        raise ValueError("validation probabilities must lie inside (0, 1]")
    if not any(np.isclose(confidence, level) for level in quantiles):
        raise ValueError("model.quantile_levels must contain validation.confidence")
    if minimum_group_samples < 1 or minimum_evaluable_groups < 1:
        raise ValueError("validation sample limits must be positive")
    validation = ValidationPolicy(confidence, minimum_coverage,
                                  minimum_group_samples, minimum_evaluable_groups)

    provenance = raw.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a JSON object when provided")
    openvins_version = str(provenance.get("openvins_version", "")).strip()
    openvins_commit = str(provenance.get("openvins_commit", "")).strip()
    if not openvins_version or not openvins_commit:
        raise ValueError("provenance.openvins_version and openvins_commit are required")
    source_manifest_value = provenance.get("source_manifest")
    source_manifest_path = None
    if source_manifest_value is None:
        raise ValueError("provenance.source_manifest is required")
    source_manifest_path = _resolve_path(source_manifest_value, config_path)
    if require_run_logs and not source_manifest_path.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {source_manifest_path}")

    run_items = raw.get("runs")
    if not isinstance(run_items, list) or not run_items:
        raise ValueError("runs must be a non-empty JSON array")
    runs = []
    run_ids = set()
    splits = set()
    for index, item in enumerate(run_items):
        if not isinstance(item, dict):
            raise ValueError(f"runs[{index}] must be an object")
        run_id = str(item.get("id", "")).strip()
        dataset_id = str(item.get("dataset_id", "")).strip()
        visual_regime = str(item.get("visual_regime", "")).strip()
        split = str(item.get("split", "")).strip()
        if not run_id or run_id in run_ids:
            raise ValueError("run ids must be non-empty and unique")
        if not dataset_id or not visual_regime or split not in {"fit", "validation"}:
            raise ValueError(f"run {run_id}: dataset_id, visual_regime and fit/validation split are required")
        run_log_path = _resolve_path(item.get("run_log"), config_path)
        run_manifest_path = run_log_path.with_suffix(".manifest.json")
        if require_run_logs and not run_log_path.is_file():
            raise FileNotFoundError(f"run {run_id} log does not exist: {run_log_path}")
        if require_run_logs and not run_manifest_path.is_file():
            raise FileNotFoundError(
                f"run {run_id} capture manifest does not exist: {run_manifest_path}")
        if require_run_logs:
            with run_manifest_path.open("r", encoding="utf-8") as stream:
                capture_manifest = json.load(stream)
            expected_sources = {
                "odometry", "diagnostics", "ground_truth", "frame_schedule"}
            if (capture_manifest.get("format_version") != 1
                    or capture_manifest.get("artifact") != run_log_path.name
                    or not expected_sources.issubset(
                        capture_manifest.get("source_fingerprints", {}))):
                raise ValueError(f"run {run_id} capture manifest is incomplete")
        run_ids.add(run_id)
        splits.add(split)
        runs.append(RunSpec(
            run_id, dataset_id, visual_regime, split, run_log_path,
            run_manifest_path))
    if splits != {"fit", "validation"}:
        raise ValueError("runs must include both fit and validation splits")
    if require_run_logs:
        with source_manifest_path.open("r", encoding="utf-8") as stream:
            source_manifest = json.load(stream)
        if source_manifest.get("format_version") != 1:
            raise ValueError("source manifest format_version must be 1")
        datasets = source_manifest.get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError("source manifest must contain a datasets object")
        missing_datasets = {run.dataset_id for run in runs} - set(datasets)
        if missing_datasets:
            raise ValueError(
                "source manifest is missing datasets: "
                + ", ".join(sorted(missing_datasets)))

    return PropagationBuildConfig(
        config_path=config_path,
        model_id=model_id,
        durations_s=durations,
        quantile_levels=quantiles,
        minimum_fit_samples=minimum_fit_samples,
        mode_policy=mode_policy,
        validation=validation,
        runs=tuple(runs),
        source_manifest_path=source_manifest_path,
        raw=raw,
    )

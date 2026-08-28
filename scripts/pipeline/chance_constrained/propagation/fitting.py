"""Fit and validate a conditional empirical INS/OpenVINS propagation model."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import beta, chi2

from .config import MODEL_ARTIFACT_VERSION, PropagationBuildConfig
from .contracts import ERROR_STATE_DIM, ERROR_STATE_FIELDS, PROPAGATION_MODES


ERROR_METRICS = ("horizontal_position_m", "horizontal_velocity_mps", "absolute_yaw_rad")


def _one_sided_lower(k, n, confidence):
    if n == 0 or k == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, k, n - k + 1))


def _one_sided_upper(k, n, confidence):
    if n == 0 or k == n:
        return 1.0
    return float(beta.ppf(confidence, k + 1, n - k))


def _regularized_covariance(samples):
    covariance = np.cov(np.asarray(samples, dtype=float), rowvar=False, ddof=1)
    covariance = .5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(values))), 1.0)
    values = np.maximum(values, scale * 1e-12)
    return (vectors * values) @ vectors.T


def _regime_axis(config, records):
    values = list(config.visual_regimes)
    for record in records:
        if record.visual_regime not in values:
            values.append(record.visual_regime)
    return tuple(values)


def fit_records(records, config: PropagationBuildConfig, *, regimes=None):
    regimes = tuple(regimes or _regime_axis(config, records))
    modes = PROPAGATION_MODES
    durations = config.durations_s
    shape = (len(regimes), len(modes), len(durations))
    regime_index = {value: index for index, value in enumerate(regimes)}
    mode_index = {value: index for index, value in enumerate(modes)}
    duration_index = {float(value): index for index, value in enumerate(durations)}

    sample_count = np.zeros(shape, dtype=np.int32)
    drift_mean = np.full(shape + (ERROR_STATE_DIM,), np.nan, dtype=float)
    process_noise = np.full(shape + (ERROR_STATE_DIM, ERROR_STATE_DIM), np.nan, dtype=float)
    quantiles = np.full(shape + (len(config.quantile_levels), len(ERROR_METRICS)),
                        np.nan, dtype=float)
    covariance_inflation = np.full(shape, np.nan, dtype=float)
    transition_count = np.zeros(shape + (len(modes),), dtype=np.int32)
    transition_probability = np.full(shape + (len(modes),), np.nan, dtype=float)
    transition_lcb = np.full_like(transition_probability, np.nan)
    transition_ucb = np.full_like(transition_probability, np.nan)

    grouped = defaultdict(list)
    for record in records:
        key = (record.visual_regime, record.start_mode, float(record.duration_s))
        if key[0] in regime_index and key[1] in mode_index and key[2] in duration_index:
            grouped[key].append(record)

    for (regime, start_mode, duration), selected in grouped.items():
        index = (regime_index[regime], mode_index[start_mode], duration_index[duration])
        count = len(selected)
        sample_count[index] = count
        for record in selected:
            transition_count[index + (mode_index[record.end_mode],)] += 1
        transition_probability[index] = transition_count[index] / count
        for end_index in range(len(modes)):
            successes = int(transition_count[index + (end_index,)])
            transition_lcb[index + (end_index,)] = _one_sided_lower(
                successes, count, config.validation.confidence)
            transition_ucb[index + (end_index,)] = _one_sided_upper(
                successes, count, config.validation.confidence)

        if count < config.minimum_fit_samples:
            continue
        errors = np.vstack([record.relative_error for record in selected])
        mean = errors.mean(axis=0)
        covariance = _regularized_covariance(errors)
        drift_mean[index] = mean
        process_noise[index] = covariance
        metrics = np.column_stack([
            np.linalg.norm(errors[:, :2], axis=1),
            np.linalg.norm(errors[:, 2:4], axis=1),
            np.abs(errors[:, 4]),
        ])
        quantiles[index] = np.quantile(metrics, config.quantile_levels, axis=0)

        mahalanobis = []
        for record, centred in zip(selected, errors - mean):
            reported = .5 * (record.reported_covariance + record.reported_covariance.T)
            reported = reported + np.eye(ERROR_STATE_DIM) * 1e-12
            mahalanobis.append(float(centred @ np.linalg.pinv(reported) @ centred))
        reference = float(chi2.ppf(config.validation.confidence, ERROR_STATE_DIM))
        covariance_inflation[index] = max(
            1.0, float(np.quantile(mahalanobis, config.validation.confidence)) / reference)

    return {
        "artifact_version": np.asarray(MODEL_ARTIFACT_VERSION, dtype=np.int16),
        "state_fields": np.asarray(ERROR_STATE_FIELDS),
        "mode_names": np.asarray(modes),
        "visual_regime_ids": np.asarray(regimes),
        "duration_bins_s": np.asarray(durations, dtype=float),
        "quantile_levels": np.asarray(config.quantile_levels, dtype=float),
        "error_metric_names": np.asarray(ERROR_METRICS),
        "sample_count": sample_count,
        "drift_mean": drift_mean,
        "process_noise_covariance": process_noise,
        "absolute_error_quantiles": quantiles,
        "reported_covariance_inflation_95": covariance_inflation,
        "mode_transition_count": transition_count,
        "mode_transition_probability": transition_probability,
        "mode_transition_lcb": transition_lcb,
        "mode_transition_ucb": transition_ucb,
        "supported": sample_count >= config.minimum_fit_samples,
    }


def validate_records(model, records, config: PropagationBuildConfig):
    regimes = list(map(str, model["visual_regime_ids"]))
    modes = list(map(str, model["mode_names"]))
    durations = list(map(float, model["duration_bins_s"]))
    grouped = defaultdict(list)
    for record in records:
        key = (record.visual_regime, record.start_mode, float(record.duration_s))
        grouped[key].append(record)

    threshold = float(chi2.ppf(config.validation.confidence, ERROR_STATE_DIM))
    quantile_index = int(np.flatnonzero(np.isclose(
        np.asarray(model["quantile_levels"], dtype=float),
        config.validation.confidence))[0])
    groups = []
    for key, selected in sorted(grouped.items()):
        if len(selected) < config.validation.minimum_group_samples:
            continue
        if key[0] not in regimes or key[1] not in modes or key[2] not in durations:
            continue
        index = (regimes.index(key[0]), modes.index(key[1]), durations.index(key[2]))
        if not bool(model["supported"][index]):
            continue
        mean = model["drift_mean"][index]
        covariance = model["process_noise_covariance"][index]
        inverse = np.linalg.pinv(covariance)
        distances = []
        error_metrics = []
        for record in selected:
            centred = record.relative_error - mean
            distances.append(float(centred @ inverse @ centred))
            error_metrics.append([
                record.horizontal_position_error_m,
                record.horizontal_velocity_error_mps,
                record.absolute_yaw_error_rad,
            ])
        coverage = float(np.mean(np.asarray(distances) <= threshold))
        metric_thresholds = np.asarray(
            model["absolute_error_quantiles"][index][quantile_index], dtype=float)
        metric_coverage = np.mean(
            np.asarray(error_metrics, dtype=float) <= metric_thresholds, axis=0)
        passed = bool(
            coverage + 1e-12 >= config.validation.minimum_coverage
            and np.all(metric_coverage + 1e-12 >= config.validation.minimum_coverage))
        groups.append({
            "visual_regime": key[0],
            "start_mode": key[1],
            "duration_s": key[2],
            "sample_count": len(selected),
            "joint_mahalanobis_coverage": coverage,
            "position_error_quantile_coverage": float(metric_coverage[0]),
            "velocity_error_quantile_coverage": float(metric_coverage[1]),
            "yaw_error_quantile_coverage": float(metric_coverage[2]),
            "passed": passed,
        })

    enough_groups = len(groups) >= config.validation.minimum_evaluable_groups
    passed = enough_groups and all(group["passed"] for group in groups)
    return {
        "passed": bool(passed),
        "confidence": config.validation.confidence,
        "minimum_coverage": config.validation.minimum_coverage,
        "evaluable_group_count": len(groups),
        "minimum_evaluable_groups": config.validation.minimum_evaluable_groups,
        "groups": groups,
        "reason": (None if passed else
                   "held-out coverage is insufficient or too few groups were evaluable"),
    }

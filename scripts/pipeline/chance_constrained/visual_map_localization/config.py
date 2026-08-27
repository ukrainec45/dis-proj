"""Validated configuration for a versioned synthetic VML experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .camera_model import PinholeCamera


SCHEMA_VERSION = 1
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class Degradation:
    scenario_id: str
    contrast: float
    blur_sigma_px: float
    haze_fraction: float
    jpeg_quality: int


@dataclass(frozen=True)
class SiftPolicy:
    nfeatures: int
    ratio_threshold: float
    ransac_threshold_px: float
    ransac_confidence: float
    minimum_good_matches: int
    minimum_inliers: int
    minimum_inlier_ratio: float
    maximum_median_reprojection_error_px: float


@dataclass(frozen=True)
class OutcomePolicy:
    maximum_position_error_m: float
    maximum_yaw_error_deg: float


@dataclass(frozen=True)
class VmlExperimentConfig:
    config_path: Path
    scene_artifact_path: Path
    scene_manifest_path: Path
    camera: PinholeCamera
    sample_edges: int
    altitudes_agl_m: tuple[float, ...]
    yaw_offsets_deg: tuple[float, ...]
    search_radii_m: tuple[float, ...]
    repetitions: int
    matching_gsd_m: float
    seed: int
    degradations: tuple[Degradation, ...]
    matcher: SiftPolicy
    outcome: OutcomePolicy
    raw: dict


def _object(parent, key):
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _positive_list(parent, key):
    values = parent.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} must be a non-empty JSON array")
    result = tuple(float(value) for value in values)
    if any(value <= 0 or not np.isfinite(value) for value in result):
        raise ValueError(f"{key} values must be finite and positive")
    return result


def _number(parent, key, *, minimum=None, maximum=None):
    if key not in parent:
        raise ValueError(f"{key} is required")
    value = float(parent[key])
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be at most {maximum}")
    return value


def _path(value, config_path, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_vml_config(path):
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    scene = _object(raw, "scene")
    scene_artifact = _path(scene.get("artifact"), config_path, "scene.artifact")
    scene_manifest = _path(scene.get("manifest"), config_path, "scene.manifest")

    camera_raw = _object(raw, "camera")
    resolution = camera_raw.get("resolution_px")
    if (not isinstance(resolution, list) or len(resolution) != 2 or
            any(int(value) < 2 for value in resolution)):
        raise ValueError("camera.resolution_px must contain width and height")
    camera = PinholeCamera(
        model=str(camera_raw.get("model", "")).strip(),
        width_px=int(resolution[0]), height_px=int(resolution[1]),
        pixel_size_um=_number(camera_raw, "pixel_size_um", minimum=1e-9),
        focal_length_mm=_number(camera_raw, "focal_length_mm", minimum=1e-9),
        frame_rate_hz=_number(camera_raw, "frame_rate_hz", minimum=1e-9),
    )
    if not camera.model:
        raise ValueError("camera.model must be non-empty")
    if camera_raw.get("mounting") != "nadir":
        raise ValueError("the version-1 synthetic renderer requires mounting=nadir")

    experiment = _object(raw, "experiment")
    sample_edges = int(experiment.get("sample_edges", 0))
    repetitions = int(experiment.get("repetitions", 0))
    if sample_edges <= 0 or repetitions <= 0:
        raise ValueError("sample_edges and repetitions must be positive integers")
    altitudes = _positive_list(experiment, "altitudes_agl_m")
    search_radii = _positive_list(experiment, "search_radii_m")
    yaw_offsets = experiment.get("yaw_offsets_deg")
    if not isinstance(yaw_offsets, list) or not yaw_offsets:
        raise ValueError("yaw_offsets_deg must be a non-empty JSON array")
    yaw_offsets = tuple(float(value) for value in yaw_offsets)
    if not all(np.isfinite(yaw_offsets)):
        raise ValueError("yaw_offsets_deg values must be finite")
    matching_gsd = _number(experiment, "matching_gsd_m", minimum=1e-6)
    seed = int(experiment.get("seed", 42))

    scenario_items = experiment.get("scenarios")
    if not isinstance(scenario_items, list) or not scenario_items:
        raise ValueError("experiment.scenarios must be a non-empty array")
    degradations = []
    identifiers = set()
    for item in scenario_items:
        if not isinstance(item, dict):
            raise ValueError("each VML scenario must be an object")
        identifier = str(item.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise ValueError("VML scenario ids must be non-empty and unique")
        identifiers.add(identifier)
        quality = int(item.get("jpeg_quality", 100))
        if not 1 <= quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        degradations.append(Degradation(
            identifier,
            _number(item, "contrast", minimum=0.0),
            _number(item, "blur_sigma_px", minimum=0.0),
            _number(item, "haze_fraction", minimum=0.0, maximum=1.0),
            quality,
        ))

    matcher_raw = _object(raw, "matcher")
    matcher = SiftPolicy(
        nfeatures=int(_number(matcher_raw, "nfeatures", minimum=1)),
        ratio_threshold=_number(matcher_raw, "ratio_threshold", minimum=0.0,
                                maximum=1.0),
        ransac_threshold_px=_number(matcher_raw, "ransac_threshold_px", minimum=0.0),
        ransac_confidence=_number(matcher_raw, "ransac_confidence", minimum=0.0,
                                  maximum=1.0),
        minimum_good_matches=int(_number(matcher_raw, "minimum_good_matches", minimum=4)),
        minimum_inliers=int(_number(matcher_raw, "minimum_inliers", minimum=4)),
        minimum_inlier_ratio=_number(matcher_raw, "minimum_inlier_ratio", minimum=0.0,
                                     maximum=1.0),
        maximum_median_reprojection_error_px=_number(
            matcher_raw, "maximum_median_reprojection_error_px", minimum=0.0),
    )
    outcome_raw = _object(raw, "outcome")
    outcome = OutcomePolicy(
        _number(outcome_raw, "maximum_position_error_m", minimum=0.0),
        _number(outcome_raw, "maximum_yaw_error_deg", minimum=0.0, maximum=180.0),
    )
    return VmlExperimentConfig(
        config_path, scene_artifact, scene_manifest, camera, sample_edges,
        altitudes, yaw_offsets, search_radii, repetitions, matching_gsd,
        seed, tuple(degradations), matcher, outcome, raw)

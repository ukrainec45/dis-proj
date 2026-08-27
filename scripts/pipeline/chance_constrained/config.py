"""Versioned JSON configuration for chance-constrained input preparation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS


SCHEMA_VERSION = 1
FORMAT_VERSION = 1


@dataclass(frozen=True)
class GridConfig:
    target_crs: str
    pixel_size_m: float
    cell_size_m: float
    edge_profile_samples: int


@dataclass(frozen=True)
class VehicleConfig:
    max_wind_mps: float
    minimum_terrain_clearance_m: float


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    visibility_scale: float
    wind: dict[str, Any]


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    source_paths: dict[str, Path | None]
    grid: GridConfig
    delivery_radius_m: float
    vehicle: VehicleConfig
    altitudes_agl_m: tuple[float, ...]
    airspeeds_mps: tuple[float, ...]
    scenarios: tuple[ScenarioConfig, ...]
    visibility_zone_value: float
    raw: dict[str, Any]


def _mapping(parent, key, required=True):
    value = parent.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _positive_sequence(parent, key, *, allow_zero=False):
    values = parent.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} must be a non-empty JSON array")
    result = tuple(float(value) for value in values)
    invalid = any(value < 0 if allow_zero else value <= 0 for value in result)
    if invalid or not all(np.isfinite(result)):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{key} values must be finite and {qualifier}")
    return result


def _resolve_source_paths(sources, config_path):
    required = ("rgb_image", "dem", "aoi", "mission_points")
    optional = ("nfz", "visibility_zones")
    result: dict[str, Path | None] = {}
    for key in required + optional:
        value = sources.get(key)
        if value is None:
            if key in required:
                raise ValueError(f"sources.{key} is required")
            result[key] = None
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"sources.{key} must be a non-empty path string")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"sources.{key} does not exist: {path}")
        result[key] = path
    return result


def _validate_wind(wind, scenario_id):
    if not isinstance(wind, dict):
        raise ValueError(f"scenario {scenario_id}: wind must be an object")
    mode = wind.get("mode")
    if mode == "constant":
        for key in ("east_mps", "north_mps"):
            if key not in wind or not np.isfinite(float(wind[key])):
                raise ValueError(f"scenario {scenario_id}: wind.{key} must be finite")
    elif mode == "synthetic":
        defaults = {"base_speed_mps": 0.0, "base_direction_deg": 0.0,
                    "perturbation_scale_mps": 0.0, "smoothness": 1.0, "seed": 42}
        for key, default in defaults.items():
            wind.setdefault(key, default)
        if float(wind["base_speed_mps"]) < 0:
            raise ValueError(f"scenario {scenario_id}: base_speed_mps must be non-negative")
        if float(wind["perturbation_scale_mps"]) < 0 or float(wind["smoothness"]) < 0:
            raise ValueError(f"scenario {scenario_id}: wind spread values must be non-negative")
        int(wind["seed"])
    else:
        raise ValueError(f"scenario {scenario_id}: wind.mode must be constant or synthetic")
    return dict(wind)


def load_config(path):
    """Load, resolve, and validate a version-1 JSON configuration."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a JSON object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    sources = _resolve_source_paths(_mapping(raw, "sources"), config_path)
    grid_raw = _mapping(raw, "grid")
    target_crs = str(grid_raw.get("target_crs", "EPSG:32635"))
    crs = CRS.from_user_input(target_crs)
    if not crs.is_projected or any(axis.unit_name.lower() not in {"metre", "meter"}
                                   for axis in crs.axis_info[:2]):
        raise ValueError("grid.target_crs must be a projected metre-based CRS")
    pixel_size = float(grid_raw.get("pixel_size_m", 0.0))
    cell_size = float(grid_raw.get("cell_size_m", 0.0))
    if pixel_size <= 0 or cell_size <= 0:
        raise ValueError("grid pixel_size_m and cell_size_m must be positive")
    ratio = cell_size / pixel_size
    if not np.isclose(ratio, round(ratio)):
        raise ValueError("grid.cell_size_m must be an integer multiple of pixel_size_m")
    profile_samples = int(grid_raw.get("edge_profile_samples", 33))
    if profile_samples < 2:
        raise ValueError("grid.edge_profile_samples must be at least 2")

    radius = float(_mapping(raw, "delivery_zone").get("radius_m", 0.0))
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("delivery_zone.radius_m must be finite and positive")

    vehicle_raw = _mapping(raw, "vehicle")
    max_wind = float(vehicle_raw.get("max_wind_mps", 0.0))
    clearance = float(vehicle_raw.get("minimum_terrain_clearance_m", -1.0))
    if max_wind <= 0 or clearance < 0:
        raise ValueError("vehicle max_wind_mps must be positive and clearance non-negative")

    points = _mapping(raw, "operating_points")
    altitudes = _positive_sequence(points, "altitudes_agl_m", allow_zero=True)
    speeds = _positive_sequence(points, "airspeeds_mps")
    if any(altitude < clearance for altitude in altitudes):
        raise ValueError("all operating altitudes must meet minimum terrain clearance")

    scenario_items = raw.get("scenarios")
    if not isinstance(scenario_items, list) or not scenario_items:
        raise ValueError("scenarios must be a non-empty JSON array")
    scenarios = []
    identifiers = set()
    for index, item in enumerate(scenario_items):
        if not isinstance(item, dict):
            raise ValueError(f"scenario {index} must be an object")
        identifier = str(item.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise ValueError("scenario ids must be non-empty and unique")
        identifiers.add(identifier)
        visibility_scale = float(item.get("visibility_scale", 1.0))
        if not 0.0 <= visibility_scale <= 1.0:
            raise ValueError(f"scenario {identifier}: visibility_scale must be in [0, 1]")
        scenarios.append(ScenarioConfig(identifier, visibility_scale,
                                        _validate_wind(dict(item.get("wind", {})), identifier)))

    environment = _mapping(raw, "environment", required=False)
    zone_value = float(environment.get("visibility_zone_value", 1.0))
    if not 0.0 <= zone_value <= 1.0:
        raise ValueError("environment.visibility_zone_value must be in [0, 1]")
    if sources["visibility_zones"] is not None and "visibility_zone_value" not in environment:
        raise ValueError("environment.visibility_zone_value is required with visibility_zones")
    for optional in ("sensors", "navigation_models", "monte_carlo"):
        _mapping(raw, optional, required=False)

    return PipelineConfig(
        config_path=config_path,
        source_paths=sources,
        grid=GridConfig(target_crs, pixel_size, cell_size, profile_samples),
        delivery_radius_m=radius,
        vehicle=VehicleConfig(max_wind, clearance),
        altitudes_agl_m=altitudes,
        airspeeds_mps=speeds,
        scenarios=tuple(scenarios),
        visibility_zone_value=zone_value,
        raw=raw,
    )

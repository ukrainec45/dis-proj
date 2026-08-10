"""Reproducible synthetic, real-AOI, weather, vehicle, and scale campaigns."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .scenarios import BenchmarkCase, npz_case
from scripts.moa import synthetic


WEATHER_PROFILES = {
    "calm_clear": {"wind": "calm", "speed_fraction": 0.0, "visibility_factor": 1.0},
    "headwind_clear": {"wind": "headwind", "speed_fraction": 0.65, "visibility_factor": 1.0},
    "crosswind_clear": {"wind": "crosswind", "speed_fraction": 0.65, "visibility_factor": 1.0},
    "moderate_degradation": {"wind": "crosswind", "speed_fraction": 0.25, "visibility_factor": 0.60},
    "severe_degradation": {"wind": "headwind", "speed_fraction": 0.35, "visibility_factor": 0.25},
}
VEHICLE_PROFILES = {
    "multirotor": 0.0,
    "fixed_wing_vtol": 40.0,
}
SYNTHETIC_FAMILIES = {
    "lake": synthetic.lake_map,
    "foggy_valley": synthetic.foggy_valley_map,
    "terrain": synthetic.terrain_map,
    "realistic": synthetic.realistic_map,
}


def _clone(cost_map, **changes):
    fields = {name: getattr(cost_map, name) for name in cost_map.__dataclass_fields__}
    for name in ("dem", "nav_density", "visibility", "wind_field", "occupancy", "landing_sites"):
        if fields[name] is not None:
            fields[name] = fields[name].copy()
    fields.update(changes)
    clone = type(cost_map)(**fields)
    for name in ("visual_richness", "terrain_richness"):
        value = getattr(cost_map, name, None)
        if value is not None:
            setattr(clone, name, value.copy())
    return clone


def apply_weather(cost_map, profile, heading=None):
    """Apply a named frozen forecast while retaining the map's spatial layers."""
    if profile not in WEATHER_PROFILES:
        raise ValueError(f"unknown weather profile: {profile}")
    specification = WEATHER_PROFILES[profile]
    if heading is None:
        dx = cost_map.goal[0] - cost_map.start[0]
        dy = -(cost_map.goal[1] - cost_map.start[1])
        heading = np.array([dx, dy], dtype=float)
        heading /= max(np.linalg.norm(heading), 1e-12)
    cross = np.array([-heading[1], heading[0]])
    direction = {"calm": np.zeros(2), "headwind": -heading,
                 "crosswind": cross}[specification["wind"]]
    wind = cost_map.wind_field.astype(float, copy=True) + \
        direction * specification["speed_fraction"] * cost_map.v_max
    magnitude = np.linalg.norm(wind, axis=2, keepdims=True)
    wind *= np.minimum(1.0, 0.90 * cost_map.v_max / np.maximum(magnitude, 1e-12))
    visibility = np.clip(cost_map.visibility * specification["visibility_factor"], 0.0, 1.0)
    return _clone(cost_map, wind_field=wind, visibility=visibility)


def apply_vehicle(cost_map, profile):
    if profile not in VEHICLE_PROFILES:
        raise ValueError(f"unknown vehicle profile: {profile}")
    return _clone(cost_map, min_turn_radius_m=VEHICLE_PROFILES[profile])


def apply_navigation_ablation(cost_map, mode):
    """Create full, visual-only, or terrain-only navigation-density cases.

    Real archives may include optional ``visual_richness`` and ``rugosity``
    arrays in their manifest.  The current CostMap stores only nav_density, so
    an unavailable component is rejected instead of silently fabricating data.
    """
    if mode == "full":
        return cost_map
    visual = getattr(cost_map, "visual_richness", None)
    terrain = getattr(cost_map, "terrain_richness", None)
    if mode == "visual_only" and visual is not None:
        return _clone(cost_map, nav_density=visual)
    if mode == "terrain_only" and terrain is not None:
        return _clone(cost_map, nav_density=terrain)
    raise ValueError(f"{mode} ablation requires the corresponding source layer")


def synthetic_campaign(variants=30, seed=2026, vehicles=("multirotor", "fixed_wing_vtol")):
    """Four seeded conflict-rich families with controlled weather perturbations."""
    rng = np.random.default_rng(seed)
    cases = []
    weather_names = tuple(WEATHER_PROFILES)
    for family, factory in SYNTHETIC_FAMILIES.items():
        for index in range(variants):
            base = factory(seed=int(rng.integers(0, 2**31))) if family == "terrain" else factory()
            weather = weather_names[index % len(weather_names)]
            base = apply_weather(base, weather)
            for vehicle in vehicles:
                cm = apply_vehicle(base, vehicle)
                cases.append(BenchmarkCase(
                    f"synthetic_{family}_{index:02d}_{weather}_{vehicle}", cm,
                    f"synthetic:{family}",
                    {"case_kind": "synthetic_campaign", "family": family, "seed": seed + index,
                     "variant": index, "weather": weather, "vehicle": vehicle,
                     "navigation_model": "full", "grid_scale": 1},
                ))
    return cases


def load_aoi_manifest(path):
    """Load five AOIs, two missions each, from a versioned JSON manifest."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    aois = data.get("aois", [])
    if len(aois) != 5:
        raise ValueError("AOI manifest must contain exactly five AOIs")
    for aoi in aois:
        if not {"id", "archive", "missions", "land_cover"} <= set(aoi):
            raise ValueError("each AOI requires id, archive, land_cover, and missions")
        if len(aoi["missions"]) != 2:
            raise ValueError(f"AOI {aoi['id']} must contain exactly two missions")
    return data


def real_aoi_campaign(manifest_path, vehicles=("multirotor", "fixed_wing_vtol"),
                      weather_profiles=tuple(WEATHER_PROFILES), ablations=("full",)):
    data = load_aoi_manifest(manifest_path)
    root = Path(manifest_path).parent
    cases = []
    for aoi in data["aois"]:
        archive = (root / aoi["archive"]).resolve()
        for mission in aoi["missions"]:
            base = npz_case(aoi["id"], archive).cost_map
            base = _clone(base, start=tuple(mission["start"]), goal=tuple(mission["goal"]))
            for weather in weather_profiles:
                forecast = apply_weather(base, weather)
                for vehicle in vehicles:
                    aircraft = apply_vehicle(forecast, vehicle)
                    for ablation in ablations:
                        try:
                            cm = apply_navigation_ablation(aircraft, ablation)
                        except ValueError:
                            if ablation != "full":
                                continue
                            raise
                        name = f"real_{aoi['id']}_{mission['id']}_{weather}_{vehicle}_{ablation}"
                        cases.append(BenchmarkCase(name, cm, str(archive), {
                            "case_kind": "real_aoi", "aoi": aoi["id"], "land_cover": aoi["land_cover"],
                            "mission": mission["id"], "weather": weather, "vehicle": vehicle,
                            "navigation_model": ablation, "grid_scale": 1,
                        }))
    return cases


def scale_case(case, factor):
    """Increase grid resolution while preserving physical extent and constraints."""
    if factor == 1:
        return case
    cm = case.cost_map
    if factor < 1 or int(factor) != factor:
        raise ValueError("grid scale must be a positive integer")
    factor = int(factor)
    wind = np.repeat(np.repeat(cm.wind_field, factor, axis=0), factor, axis=1)
    changes = {
        "dem": np.repeat(np.repeat(cm.dem, factor, axis=0), factor, axis=1),
        "nav_density": np.repeat(np.repeat(cm.nav_density, factor, axis=0), factor, axis=1),
        "visibility": np.repeat(np.repeat(cm.visibility, factor, axis=0), factor, axis=1),
        "wind_field": wind,
        "occupancy": np.repeat(np.repeat(cm.occupancy, factor, axis=0), factor, axis=1),
        "landing_sites": None if cm.landing_sites is None else
            np.repeat(np.repeat(cm.landing_sites, factor, axis=0), factor, axis=1),
        "resolution_m": cm.resolution_m / factor,
        "start": tuple(value * factor for value in cm.start),
        "goal": tuple(value * factor for value in cm.goal),
    }
    metadata = dict(case.metadata or {})
    metadata["grid_scale"] = factor
    return BenchmarkCase(f"{case.name}_scale{factor}", _clone(cm, **changes), case.source, metadata)

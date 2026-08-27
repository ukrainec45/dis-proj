"""Geospatial scene preparation for the chance-constrained edge model."""

from __future__ import annotations

import numpy as np

from .config import FORMAT_VERSION, PipelineConfig
from .edges import build_directed_edges
from ..utils.grid import (delivery_zone_mask, occupancy_from_zones,
                          point_to_cell)
from ..utils.raster import read_dem_grid, read_rgb_aoi
from ..utils.terrain import patch_centre_height
from ..utils.visibility import visibility_from_zones
from ..utils.wind import generate_synthetic_wind


def _read_vector(path, target_crs, *, geometry_types, label):
    import geopandas as gpd

    frame = gpd.read_file(path)
    if frame.empty or frame.geometry.is_empty.any():
        raise ValueError(f"{label} must contain non-empty geometries")
    if frame.crs is None:
        raise ValueError(f"{label} must declare a CRS")
    if not frame.geometry.geom_type.isin(geometry_types).all():
        expected = ", ".join(sorted(geometry_types))
        raise ValueError(f"{label} must contain only {expected} geometries")
    return frame.to_crs(target_crs)


def _mission_points(path, target_crs):
    points = _read_vector(path, target_crs, geometry_types={"Point"},
                          label="mission_points")
    if "role" not in points.columns:
        raise ValueError("mission_points must contain a role field")
    selected = {}
    for role in ("start", "goal"):
        matches = points.loc[points["role"] == role]
        if len(matches) != 1:
            raise ValueError(f"mission_points must contain exactly one {role} point")
        selected[role] = matches.geometry.iloc[0]
    return selected


def _cell_dem_layers(dem_fine, rows, cols, steps):
    centres = np.full((rows, cols), np.nan, dtype=float)
    maxima = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        for col in range(cols):
            patch = dem_fine[row * steps:(row + 1) * steps,
                             col * steps:(col + 1) * steps]
            centres[row, col] = patch_centre_height(patch)
            finite = patch[np.isfinite(patch)]
            if finite.size:
                maxima[row, col] = float(np.max(finite))
    return centres, maxima


def _wind_for_scenario(scenario, rows, cols):
    wind = scenario.wind
    if wind["mode"] == "constant":
        u = np.full((rows, cols), float(wind["east_mps"]), dtype=float)
        v = np.full((rows, cols), float(wind["north_mps"]), dtype=float)
        return np.dstack((u, v))
    u, v = generate_synthetic_wind(
        rows, cols,
        base_speed=float(wind["base_speed_mps"]),
        base_dir_deg=float(wind["base_direction_deg"]),
        perturbation_scale=float(wind["perturbation_scale_mps"]),
        smoothness=float(wind["smoothness"]),
        seed=int(wind["seed"]),
    )
    return np.dstack((u, v))


def _inside_extent(point, extent):
    xmin, xmax, ymin, ymax = map(float, extent)
    return xmin <= point.x < xmax and ymin < point.y <= ymax


def build_scene_edges(config: PipelineConfig):
    """Build the first-increment scene and directed-edge artifact arrays."""
    target_crs = config.grid.target_crs
    aoi = _read_vector(config.source_paths["aoi"], target_crs,
                       geometry_types={"Polygon", "MultiPolygon"}, label="AOI")
    mission = _mission_points(config.source_paths["mission_points"], target_crs)
    nfz = (_read_vector(config.source_paths["nfz"], target_crs,
                        geometry_types={"Polygon", "MultiPolygon"}, label="NFZ")
           if config.source_paths["nfz"] else None)
    visibility_zones = (
        _read_vector(config.source_paths["visibility_zones"], target_crs,
                     geometry_types={"Polygon", "MultiPolygon"},
                     label="visibility_zones")
        if config.source_paths["visibility_zones"] else None
    )

    rgb, raw_extent, transform = read_rgb_aoi(
        config.source_paths["rgb_image"], aoi, config.grid.pixel_size_m)
    steps = int(round(config.grid.cell_size_m / config.grid.pixel_size_m))
    rows, cols = rgb.shape[0] // steps, rgb.shape[1] // steps
    if rows <= 0 or cols <= 0:
        raise ValueError("AOI does not contain a complete planner cell")
    fine_rows, fine_cols = rows * steps, cols * steps
    rgb = rgb[:fine_rows, :fine_cols]
    xmin, _, _, ymax = map(float, raw_extent)
    extent = np.asarray([xmin, xmin + cols * config.grid.cell_size_m,
                         ymax - rows * config.grid.cell_size_m, ymax], dtype=float)
    dem_fine = read_dem_grid(config.source_paths["dem"], aoi.crs, transform,
                             rgb.shape[0], rgb.shape[1])[:fine_rows, :fine_cols]
    dem_center, dem_max = _cell_dem_layers(dem_fine, rows, cols, steps)
    occupancy = occupancy_from_zones(extent, rows, cols, nfz)
    occupancy |= ~np.isfinite(dem_center)

    if visibility_zones is None:
        base_visibility = np.ones((rows, cols), dtype=float)
    else:
        base_visibility = visibility_from_zones(
            extent, rows, cols, visibility_zones, config.visibility_zone_value)
    visibility = np.stack([
        np.clip(base_visibility * scenario.visibility_scale, 0.0, 1.0)
        for scenario in config.scenarios
    ])
    wind_field = np.stack([
        _wind_for_scenario(scenario, rows, cols) for scenario in config.scenarios
    ])

    for role, point in mission.items():
        if not _inside_extent(point, extent):
            raise ValueError(f"{role} point lies outside the complete planner grid")
    start_xy = np.asarray([mission["start"].x, mission["start"].y], dtype=float)
    goal_xy = np.asarray([mission["goal"].x, mission["goal"].y], dtype=float)
    start_cell = np.asarray(point_to_cell(*start_xy, extent, rows, cols), dtype=np.int32)
    if occupancy[start_cell[1], start_cell[0]]:
        raise ValueError("start point lies in an occupied planner cell")
    zone_mask = delivery_zone_mask(extent, rows, cols, goal_xy,
                                   config.delivery_radius_m)

    arrays = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int16),
        "crs": np.asarray(target_crs),
        "extent": extent,
        "transform": np.asarray(tuple(transform)[:6], dtype=float),
        "grid_shape": np.asarray((rows, cols), dtype=np.int32),
        "aligned_raster_shape": np.asarray((fine_rows, fine_cols), dtype=np.int32),
        "pixel_size_m": np.asarray(config.grid.pixel_size_m, dtype=float),
        "cell_size_m": np.asarray(config.grid.cell_size_m, dtype=float),
        "source_pixels_per_cell": np.asarray(steps, dtype=np.int32),
        "dem_center_m": dem_center,
        "dem_max_m": dem_max,
        "occupancy": occupancy,
        "base_visibility": base_visibility,
        "visibility": visibility,
        "wind_field": wind_field,
        "scenario_ids": np.asarray([item.scenario_id for item in config.scenarios]),
        "altitudes_agl_m": np.asarray(config.altitudes_agl_m, dtype=float),
        "airspeeds_mps": np.asarray(config.airspeeds_mps, dtype=float),
        "max_wind_mps": np.asarray(config.vehicle.max_wind_mps, dtype=float),
        "minimum_terrain_clearance_m": np.asarray(
            config.vehicle.minimum_terrain_clearance_m, dtype=float),
        "start_cell_xy": start_cell,
        "start_map_xy": start_xy,
        "delivery_center_xy": goal_xy,
        "delivery_radius_m": np.asarray(config.delivery_radius_m, dtype=float),
        "delivery_zone_mask": zone_mask,
        "start_inside_delivery_zone": np.asarray(
            np.linalg.norm(start_xy - goal_xy) <= config.delivery_radius_m),
    }
    arrays.update(build_directed_edges(
        dem_fine=dem_fine,
        dem_center=dem_center,
        occupancy=occupancy,
        visibility=visibility,
        wind_field=wind_field,
        extent=extent,
        pixel_size_m=config.grid.pixel_size_m,
        cell_size_m=config.grid.cell_size_m,
        altitudes_agl_m=config.altitudes_agl_m,
        airspeeds_mps=config.airspeeds_mps,
        max_wind_mps=config.vehicle.max_wind_mps,
        minimum_terrain_clearance_m=config.vehicle.minimum_terrain_clearance_m,
        delivery_center_xy=goal_xy,
        delivery_radius_m=config.delivery_radius_m,
        profile_samples=config.grid.edge_profile_samples,
    ))
    if (not bool(arrays["start_inside_delivery_zone"]) and
            not np.isfinite(arrays["edge_goal_entry_fraction"]).any()):
        raise ValueError(
            "delivery zone does not intersect the grid-edge geometry; "
            "increase its radius or reduce planner cell size")
    arrays["edge_count"] = np.asarray(len(arrays["edge_start_cell_xy"]),
                                      dtype=np.int32)
    return arrays

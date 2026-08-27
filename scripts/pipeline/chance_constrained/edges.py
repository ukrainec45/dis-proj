"""Construction of auditable directed edges for the navigation pipeline."""

from __future__ import annotations

import numpy as np

from ..utils.flight import edge_travel_time_s
from ..utils.grid import (NEIGHBOR_OFFSETS, cell_center_xy,
                          segment_circle_entry_fraction)
from ..utils.terrain import finite_bilinear_profile


def _mean_endpoint(layer, start_row, start_col, end_row, end_col):
    return 0.5 * (float(layer[start_row, start_col]) +
                  float(layer[end_row, end_col]))


def _edge_profile(fine_dem, steps, start_row, start_col, end_row, end_col,
                  sample_count):
    start_r = start_row * steps + (steps - 1) / 2.0
    start_c = start_col * steps + (steps - 1) / 2.0
    end_r = end_row * steps + (steps - 1) / 2.0
    end_c = end_col * steps + (steps - 1) / 2.0
    fractions = np.linspace(0.0, 1.0, sample_count)
    return finite_bilinear_profile(
        fine_dem,
        start_r + fractions * (end_r - start_r),
        start_c + fractions * (end_c - start_c),
    )


def build_directed_edges(*, dem_fine, dem_center, occupancy, visibility,
                         wind_field, extent, pixel_size_m, cell_size_m,
                         altitudes_agl_m, airspeeds_mps, max_wind_mps,
                         minimum_terrain_clearance_m, delivery_center_xy,
                         delivery_radius_m, profile_samples):
    """Build all in-bounds directed 8-neighbour edges and their base physics.

    Arrays use the axis order ``edge, scenario, altitude, speed`` where all
    axes are present. Spatially invalid edges remain in the artifact with a
    false feasibility flag so preprocessing decisions remain auditable.
    """
    dem_fine = np.asarray(dem_fine, dtype=float)
    dem_center = np.asarray(dem_center, dtype=float)
    occupancy = np.asarray(occupancy, dtype=bool)
    visibility = np.asarray(visibility, dtype=float)
    wind_field = np.asarray(wind_field, dtype=float)
    rows, cols = dem_center.shape
    scenario_count = visibility.shape[0]
    if occupancy.shape != (rows, cols):
        raise ValueError("occupancy must match dem_center")
    if visibility.shape[1:] != (rows, cols):
        raise ValueError("visibility must have shape (S, H, W)")
    if wind_field.shape != (scenario_count, rows, cols, 2):
        raise ValueError("wind_field must have shape (S, H, W, 2)")
    steps = int(round(cell_size_m / pixel_size_m))
    if dem_fine.shape[0] != rows * steps or dem_fine.shape[1] != cols * steps:
        raise ValueError("fine DEM dimensions must exactly match the planner grid")

    starts, ends, directions = [], [], []
    for row in range(rows):
        for col in range(cols):
            for direction_index, (dx, dy) in enumerate(NEIGHBOR_OFFSETS):
                end_row, end_col = row + dy, col + dx
                if 0 <= end_row < rows and 0 <= end_col < cols:
                    starts.append((col, row))
                    ends.append((end_col, end_row))
                    directions.append(direction_index)
    start_cells = np.asarray(starts, dtype=np.int32)
    end_cells = np.asarray(ends, dtype=np.int32)
    direction_indices = np.asarray(directions, dtype=np.int8)
    edge_count = len(start_cells)
    altitude_count = len(altitudes_agl_m)
    speed_count = len(airspeeds_mps)

    start_map = np.empty((edge_count, 2), dtype=float)
    end_map = np.empty((edge_count, 2), dtype=float)
    offsets = np.empty((edge_count, 2), dtype=np.int8)
    azimuth = np.empty(edge_count, dtype=float)
    horizontal_length = np.empty(edge_count, dtype=float)
    length_3d = np.empty(edge_count, dtype=float)
    profiles = np.empty((edge_count, profile_samples), dtype=np.float32)
    terrain_excess = np.empty(edge_count, dtype=float)
    source_windows = np.empty((edge_count, 4), dtype=np.int32)
    mean_visibility = np.empty((edge_count, scenario_count), dtype=float)
    wind_along = np.empty((edge_count, scenario_count), dtype=float)
    goal_fraction = np.full(edge_count, np.nan, dtype=float)
    minimum_clearance = np.full((edge_count, altitude_count), np.nan, dtype=float)
    spatial_feasible = np.zeros((edge_count, altitude_count), dtype=bool)
    travel_time = np.full((edge_count, scenario_count, altitude_count, speed_count),
                          np.nan, dtype=float)
    terminal_time = np.full_like(travel_time, np.nan)
    feasible = np.zeros_like(travel_time, dtype=bool)

    for edge_index, (start, end) in enumerate(zip(start_cells, end_cells)):
        start_col, start_row = map(int, start)
        end_col, end_row = map(int, end)
        dx, dy = end_col - start_col, end_row - start_row
        offsets[edge_index] = (dx, dy)
        p0 = cell_center_xy(start_row, start_col, extent, rows, cols)
        p1 = cell_center_xy(end_row, end_col, extent, rows, cols)
        start_map[edge_index], end_map[edge_index] = p0, p1
        displacement = p1 - p0
        horizontal_length[edge_index] = np.linalg.norm(displacement)
        azimuth[edge_index] = (np.degrees(np.arctan2(displacement[0],
                                                     displacement[1])) + 360.0) % 360.0
        z0 = float(dem_center[start_row, start_col])
        z1 = float(dem_center[end_row, end_col])
        length_3d[edge_index] = np.hypot(horizontal_length[edge_index], z1 - z0)
        profile = _edge_profile(dem_fine, steps, start_row, start_col,
                                end_row, end_col, profile_samples)
        profiles[edge_index] = profile
        baseline = np.linspace(z0, z1, profile_samples)
        finite_profile = np.isfinite(profile) & np.isfinite(baseline)
        terrain_excess[edge_index] = (
            float(np.max(profile[finite_profile] - baseline[finite_profile]))
            if finite_profile.any() else np.inf
        )
        source_windows[edge_index] = (
            min(start_row, end_row) * steps,
            (max(start_row, end_row) + 1) * steps,
            min(start_col, end_col) * steps,
            (max(start_col, end_col) + 1) * steps,
        )
        goal_fraction[edge_index] = segment_circle_entry_fraction(
            p0, p1, delivery_center_xy, delivery_radius_m)

        base_valid = (not occupancy[start_row, start_col] and
                      not occupancy[end_row, end_col] and
                      np.isfinite(z0) and np.isfinite(z1) and finite_profile.all())
        if dx != 0 and dy != 0:
            base_valid = base_valid and not (
                occupancy[start_row, end_col] or occupancy[end_row, start_col]
            )
        for altitude_index, altitude in enumerate(altitudes_agl_m):
            minimum_clearance[edge_index, altitude_index] = (
                float(altitude) - terrain_excess[edge_index]
            )
            spatial_feasible[edge_index, altitude_index] = (
                base_valid and minimum_clearance[edge_index, altitude_index] + 1e-9
                >= minimum_terrain_clearance_m
            )

        unit_track = displacement / horizontal_length[edge_index]
        for scenario_index in range(scenario_count):
            mean_visibility[edge_index, scenario_index] = _mean_endpoint(
                visibility[scenario_index], start_row, start_col, end_row, end_col)
            wind = 0.5 * (wind_field[scenario_index, start_row, start_col] +
                          wind_field[scenario_index, end_row, end_col])
            wind_along[edge_index, scenario_index] = float(wind @ unit_track)
            wind_valid = np.linalg.norm(wind) < max_wind_mps
            for altitude_index in range(altitude_count):
                for speed_index, airspeed in enumerate(airspeeds_mps):
                    value = edge_travel_time_s(displacement[0], displacement[1],
                                               z1 - z0, wind, airspeed)
                    if value is not None:
                        travel_time[edge_index, scenario_index,
                                    altitude_index, speed_index] = value
                        if np.isfinite(goal_fraction[edge_index]):
                            terminal_time[edge_index, scenario_index,
                                          altitude_index, speed_index] = (
                                value * goal_fraction[edge_index]
                            )
                    feasible[edge_index, scenario_index, altitude_index, speed_index] = (
                        spatial_feasible[edge_index, altitude_index]
                        and wind_valid and value is not None
                    )

    return {
        "edge_start_cell_xy": start_cells,
        "edge_end_cell_xy": end_cells,
        "edge_direction_index": direction_indices,
        "edge_offset_xy": offsets,
        "edge_start_map_xy": start_map,
        "edge_end_map_xy": end_map,
        "edge_azimuth_deg": azimuth,
        "edge_horizontal_length_m": horizontal_length,
        "edge_length_3d_m": length_3d,
        "edge_dem_profile_m": profiles,
        "edge_terrain_excess_m": terrain_excess,
        "edge_minimum_clearance_m": minimum_clearance,
        "edge_source_window_px": source_windows,
        "edge_mean_visibility": mean_visibility,
        "edge_wind_along_mps": wind_along,
        "edge_goal_entry_fraction": goal_fraction,
        "edge_spatial_feasible": spatial_feasible,
        "edge_travel_time_s": travel_time,
        "edge_goal_entry_time_s": terminal_time,
        "edge_feasible": feasible,
    }

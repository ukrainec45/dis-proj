"""Persist planner routes and derived characteristics without rerunning EMOA*."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .topsis import topsis


FORMAT_VERSION = 1


def build_route_results(cost_map, solutions, weights=(0.5, 0.3, 0.2),
                        n_expanded=0, n_generated=0):
    """Return numeric arrays describing all Pareto routes and their metrics."""
    if not solutions:
        raise ValueError("cannot save an empty planner result")

    costs = np.asarray([cost for _, cost in solutions], dtype=float)
    weights = np.asarray(weights, dtype=float)
    order, closeness = topsis(costs, weights)
    lengths = np.asarray([len(path) for path, _ in solutions], dtype=np.int32)
    paths = np.full((len(solutions), int(lengths.max()), 2), -1, dtype=np.int32)

    horizontal = np.zeros(len(solutions), dtype=float)
    spatial = np.zeros(len(solutions), dtype=float)
    minimum_altitude = np.zeros(len(solutions), dtype=float)
    maximum_altitude = np.zeros(len(solutions), dtype=float)

    for route_id, (path, _) in enumerate(solutions):
        paths[route_id, :len(path)] = np.asarray(path, dtype=np.int32)
        altitudes = np.asarray([cost_map.altitude_m(cell) for cell in path], dtype=float)
        minimum_altitude[route_id] = altitudes.min()
        maximum_altitude[route_id] = altitudes.max()
        for start, end in zip(path, path[1:]):
            dx = (end[0] - start[0]) * cost_map.resolution_m
            dy = (end[1] - start[1]) * cost_map.resolution_m
            dz = (float(cost_map.dem[end[1], end[0]]) -
                  float(cost_map.dem[start[1], start[0]]))
            horizontal[route_id] += np.hypot(dx, dy)
            spatial[route_id] += np.sqrt(dx * dx + dy * dy + dz * dz)

    flight_time = costs[:, 0]
    average_speed = np.divide(horizontal, flight_time,
                              out=np.zeros_like(horizontal), where=flight_time > 0)
    average_navigation = 1.0 - np.divide(
        costs[:, 1], flight_time, out=np.zeros_like(flight_time), where=flight_time > 0)
    average_visibility = 1.0 - np.divide(
        costs[:, 2], flight_time, out=np.zeros_like(flight_time), where=flight_time > 0)
    # A zero-duration route stays in one cell and therefore has no deficit.
    average_navigation = np.where(flight_time > 0, average_navigation, 1.0)
    average_visibility = np.where(flight_time > 0, average_visibility, 1.0)

    energy = np.full(len(solutions), np.nan, dtype=float)
    remaining = np.full(len(solutions), np.nan, dtype=float)
    if cost_map.cruise_power_w is not None:
        energy = cost_map.cruise_power_w * flight_time / 3600.0
        if cost_map.battery_energy_wh is not None:
            remaining = cost_map.battery_energy_wh - energy

    return {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int16),
        "paths": paths,
        "path_lengths": lengths,
        "costs": costs,
        "weights": weights,
        "topsis_order": np.asarray(order, dtype=np.int32),
        "topsis_closeness": np.asarray(closeness, dtype=float),
        "topsis_best": np.asarray(int(order[0]), dtype=np.int32),
        "horizontal_length_m": horizontal,
        "terrain_following_length_m": spatial,
        "flight_time_s": flight_time,
        "average_ground_speed_mps": average_speed,
        "navigation_deficit_s": costs[:, 1],
        "average_navigation_quality": average_navigation,
        "visibility_deficit_s": costs[:, 2],
        "average_visibility": average_visibility,
        "energy_used_wh": energy,
        "battery_remaining_wh": remaining,
        "minimum_altitude_msl_m": minimum_altitude,
        "maximum_altitude_msl_m": maximum_altitude,
        "resolution_m": np.asarray(cost_map.resolution_m, dtype=float),
        "start": np.asarray(cost_map.start, dtype=np.int32),
        "goal": np.asarray(cost_map.goal, dtype=np.int32),
        "n_expanded": np.asarray(n_expanded, dtype=np.int64),
        "n_generated": np.asarray(n_generated, dtype=np.int64),
    }


def save_route_results(path, cost_map, solutions, weights=(0.5, 0.3, 0.2),
                       n_expanded=0, n_generated=0):
    """Save routes and characteristics to a compressed, pickle-free NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = build_route_results(cost_map, solutions, weights,
                                  n_expanded=n_expanded, n_generated=n_generated)
    np.savez_compressed(path, **results)
    return results


def load_route_results(path):
    """Load a saved result into ordinary arrays; paths remain padded."""
    with np.load(path, allow_pickle=False) as data:
        results = {key: data[key] for key in data.files}
    version = int(results.get("format_version", -1))
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported planner-result format version: {version}")
    return results


def route_path(results, route_id):
    """Extract one unpadded ``(N, 2)`` integer route from loaded results."""
    length = int(results["path_lengths"][route_id])
    return results["paths"][route_id, :length].copy()


CSV_FIELDS = (
    "route_id", "topsis_selected", "topsis_rank", "topsis_closeness", "steps",
    "horizontal_length_m", "terrain_following_length_m", "flight_time_s",
    "average_ground_speed_mps", "navigation_deficit_s",
    "average_navigation_quality", "visibility_deficit_s", "average_visibility",
    "energy_used_wh", "battery_remaining_wh", "minimum_altitude_msl_m",
    "maximum_altitude_msl_m",
)


def characteristics_rows(results):
    """Yield flat records suitable for CSV/data-frame analysis."""
    count = len(results["path_lengths"])
    ranks = np.empty(count, dtype=int)
    ranks[results["topsis_order"]] = np.arange(1, count + 1)
    best = int(results["topsis_best"])
    array_fields = CSV_FIELDS[5:]
    for route_id in range(count):
        row = {
            "route_id": route_id,
            "topsis_selected": route_id == best,
            "topsis_rank": int(ranks[route_id]),
            "topsis_closeness": float(results["topsis_closeness"][route_id]),
            "steps": int(results["path_lengths"][route_id]),
        }
        row.update({field: float(results[field][route_id]) for field in array_fields})
        yield row


def write_characteristics_csv(path, results):
    """Write all saved route characteristics without invoking the planner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(characteristics_rows(results))
    return path

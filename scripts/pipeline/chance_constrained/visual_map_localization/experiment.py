"""Run and persist a spatially sampled synthetic VML campaign."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from ..artifacts import load_scene_edges, read_manifest
from .config import ARTIFACT_VERSION, load_vml_config
from .matcher import SiftMapMatcher, circular_error_deg
from .raster_sampler import OrthophotoSampler, degrade_image


TRIAL_FIELDS = (
    "trial_id", "edge_index", "midpoint_x_m", "midpoint_y_m",
    "edge_azimuth_deg", "altitude_agl_m", "yaw_deg", "yaw_offset_deg",
    "search_radius_m", "prior_error_m", "prior_x_m", "prior_y_m",
    "scenario_id", "repetition", "outcome", "reject_reason",
    "position_error_m", "yaw_error_deg", "estimated_x_m", "estimated_y_m",
    "estimated_yaw_deg", "query_keypoints", "reference_keypoints",
    "good_matches", "inliers", "inlier_ratio",
    "median_reprojection_error_px", "runtime_ms", "query_valid_fraction",
    "reference_valid_fraction",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_edges(arrays, camera, altitudes, search_radii,
                     source_coverage=None):
    feasible = np.asarray(arrays["edge_feasible"], dtype=bool)
    feasible_any = feasible.reshape(feasible.shape[0], -1).any(axis=1)
    starts = np.asarray(arrays["edge_start_map_xy"], dtype=float)
    ends = np.asarray(arrays["edge_end_map_xy"], dtype=float)
    midpoints = 0.5 * (starts + ends)
    maximum_footprint = max(np.hypot(*camera.footprint_m(altitude))
                            for altitude in altitudes)
    margin = max(search_radii) + maximum_footprint / 2.0 + 5.0
    xmin, xmax, ymin, ymax = map(float, arrays["extent"])
    inside = ((midpoints[:, 0] >= xmin + margin) &
              (midpoints[:, 0] <= xmax - margin) &
              (midpoints[:, 1] >= ymin + margin) &
              (midpoints[:, 1] <= ymax - margin))
    eligible = feasible_any & inside
    if source_coverage is not None:
        coverage = np.asarray(source_coverage, dtype=bool)
        if coverage.shape != eligible.shape:
            raise ValueError("source_coverage must have one value per edge")
        eligible &= coverage
    return np.flatnonzero(eligible), midpoints


def select_spatial_edges(arrays, camera, altitudes, search_radii, count, seed,
                         source_coverage=None):
    """Select directed edges with farthest-point spatial coverage."""
    candidates, midpoints = _candidate_edges(
        arrays, camera, altitudes, search_radii, source_coverage)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} edges have sufficient map margin; "
                         f"cannot sample {count}")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(candidates)))
    selected_positions = [first]
    minimum_distance_sq = np.sum(
        (midpoints[candidates] - midpoints[candidates[first]]) ** 2, axis=1)
    for _ in range(1, count):
        next_position = int(np.argmax(minimum_distance_sq))
        selected_positions.append(next_position)
        distance_sq = np.sum(
            (midpoints[candidates] - midpoints[candidates[next_position]]) ** 2,
            axis=1)
        minimum_distance_sq = np.minimum(minimum_distance_sq, distance_sq)
        minimum_distance_sq[selected_positions] = -1.0
    return candidates[np.asarray(selected_positions, dtype=int)]


def _one_sided_lower(k, n, confidence=.95):
    if n == 0 or k == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, k, n - k + 1))


def _one_sided_upper(k, n, confidence=.95):
    if n == 0 or k == n:
        return 1.0
    return float(beta.ppf(confidence, k + 1, n - k))


def aggregate_trials(rows, edge_indices, scenarios, altitudes, radii):
    shape = (len(edge_indices), len(scenarios), len(altitudes), len(radii))
    edge_axis = {int(value): index for index, value in enumerate(edge_indices)}
    scenario_axis = {value: index for index, value in enumerate(scenarios)}
    altitude_axis = {float(value): index for index, value in enumerate(altitudes)}
    radius_axis = {float(value): index for index, value in enumerate(radii)}
    counts = {name: np.zeros(shape, dtype=np.int32)
              for name in ("trial", "correct", "reject", "false_accept")}
    runtime_sum = np.zeros(shape, dtype=float)
    for row in rows:
        index = (edge_axis[int(row["edge_index"])],
                 scenario_axis[row["scenario_id"]],
                 altitude_axis[float(row["altitude_agl_m"])],
                 radius_axis[float(row["search_radius_m"])])
        counts["trial"][index] += 1
        counts[row["outcome"]][index] += 1
        runtime_sum[index] += float(row["runtime_ms"])
    trials = counts["trial"]
    safe_trials = np.maximum(trials, 1)
    result = {
        "artifact_version": np.asarray(ARTIFACT_VERSION, dtype=np.int16),
        "edge_indices": np.asarray(edge_indices, dtype=np.int32),
        "scenario_ids": np.asarray(scenarios),
        "altitudes_agl_m": np.asarray(altitudes, dtype=float),
        "search_radii_m": np.asarray(radii, dtype=float),
        "trial_count": trials,
        "correct_count": counts["correct"],
        "reject_count": counts["reject"],
        "false_accept_count": counts["false_accept"],
        "p_correct": counts["correct"] / safe_trials,
        "p_reject": counts["reject"] / safe_trials,
        "p_false_accept": counts["false_accept"] / safe_trials,
        "mean_runtime_ms": runtime_sum / safe_trials,
    }
    lower = np.zeros(shape, dtype=float)
    false_upper = np.ones(shape, dtype=float)
    for index in np.ndindex(shape):
        lower[index] = _one_sided_lower(
            int(counts["correct"][index]), int(trials[index]))
        false_upper[index] = _one_sided_upper(
            int(counts["false_accept"][index]), int(trials[index]))
    result["p_correct_lcb95"] = lower
    result["p_false_accept_ucb95"] = false_upper
    return result


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path, rows):
    import matplotlib.pyplot as plt

    outcome_colours = {"correct": "#2ca02c", "reject": "#7f7f7f",
                       "false_accept": "#d62728"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for outcome, colour in outcome_colours.items():
        selected = [row for row in rows if row["outcome"] == outcome]
        if selected:
            axes[0].scatter([row["midpoint_x_m"] for row in selected],
                            [row["midpoint_y_m"] for row in selected],
                            s=14, alpha=.55, c=colour, label=outcome)
    axes[0].set_title("Spatial trial outcomes")
    axes[0].set_aspect("equal")
    axes[0].legend()
    labels = list(outcome_colours)
    values = [sum(row["outcome"] == label for row in rows) for label in labels]
    axes[1].bar(labels, values, color=[outcome_colours[label] for label in labels])
    axes[1].set_title("Outcome counts")
    axes[1].set_ylabel("trials")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyse_rows(rows):
    """Create an auditable descriptive summary without reliability claims."""
    outcome_names = ("correct", "reject", "false_accept")

    def group(field):
        values = sorted({str(row[field]) for row in rows})
        result = {}
        for value in values:
            selected = [row for row in rows if str(row[field]) == value]
            counts = {name: sum(row["outcome"] == name for row in selected)
                      for name in outcome_names}
            result[value] = {
                "trials": len(selected), "counts": counts,
                "rates": {name: counts[name] / len(selected)
                          for name in outcome_names},
            }
        return result

    counts = {name: sum(row["outcome"] == name for row in rows)
              for name in outcome_names}
    reject_reasons = {}
    for row in rows:
        if row["outcome"] == "reject":
            reason = row["reject_reason"]
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
    runtime = np.asarray([float(row["runtime_ms"]) for row in rows], dtype=float)
    correct = [row for row in rows if row["outcome"] == "correct"]

    def percentiles(field):
        values = np.asarray([float(row[field]) for row in correct], dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return {"p50": None, "p95": None, "maximum": None}
        return {"p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "maximum": float(values.max())}

    return {
        "trials": len(rows), "counts": counts,
        "rates": {name: counts[name] / len(rows) for name in outcome_names},
        "reject_reasons": reject_reasons,
        "by_scenario": group("scenario_id"),
        "by_altitude_agl_m": group("altitude_agl_m"),
        "by_search_radius_m": group("search_radius_m"),
        "by_edge": group("edge_index"),
        "correct_position_error_m": percentiles("position_error_m"),
        "correct_yaw_error_deg": percentiles("yaw_error_deg"),
        "runtime_ms": {
            "p50": float(np.percentile(runtime, 50)),
            "p95": float(np.percentile(runtime, 95)),
            "maximum": float(runtime.max()),
        },
    }


def run_experiment(config_path, output_dir, *, smoke=False, force=False):
    config = load_vml_config(config_path)
    arrays = load_scene_edges(config.scene_artifact_path,
                              manifest_path=config.scene_manifest_path,
                              verify_sources=True)
    scene_manifest = read_manifest(config.scene_manifest_path)
    rgb_path = Path(scene_manifest["resolved_sources"]["rgb_image"])
    altitudes = config.altitudes_agl_m
    yaw_offsets = config.yaw_offsets_deg
    radii = config.search_radii_m
    degradations = config.degradations
    repetitions = config.repetitions
    output = Path(output_dir)
    expected = [output / name for name in
                ("trials.csv", "vml_sample_outcomes.npz", "manifest.json",
                 "summary.json", "outcomes.png")]
    existing = [path for path in expected if path.exists()]
    if existing and not force:
        raise FileExistsError("refusing to overwrite existing VML artifacts: " +
                              ", ".join(path.name for path in existing))
    output.mkdir(parents=True, exist_ok=True)

    matcher = SiftMapMatcher(config.matcher)
    rng = np.random.default_rng(config.seed)
    starts = np.asarray(arrays["edge_start_map_xy"], dtype=float)
    ends = np.asarray(arrays["edge_end_map_xy"], dtype=float)
    azimuths = np.asarray(arrays["edge_azimuth_deg"], dtype=float)
    rows = []
    with OrthophotoSampler(rgb_path, str(np.asarray(arrays["crs"]).item())) as sampler:
        all_midpoints = 0.5 * (starts + ends)
        largest_half_diagonal = max(
            np.hypot(*config.camera.footprint_m(altitude)) / 2.0
            for altitude in config.altitudes_agl_m)
        coverage_fraction = sampler.coverage_fraction_around_points(
            all_midpoints,
            max(config.search_radii_m) + largest_half_diagonal + 5.0)
        source_coverage = coverage_fraction >= (8.0 / 9.0)
        selected_edges = select_spatial_edges(
            arrays, config.camera, config.altitudes_agl_m,
            config.search_radii_m, config.sample_edges, config.seed,
            source_coverage=source_coverage)
        if smoke:
            selected_edges = selected_edges[:2]
            altitudes = altitudes[:1]
            yaw_offsets = yaw_offsets[:1]
            radii = radii[:1]
            degradations = degradations[:1]
            repetitions = 1
        trial_id = 0
        for edge_index in selected_edges:
            midpoint = 0.5 * (starts[edge_index] + ends[edge_index])
            edge_azimuth = float(azimuths[edge_index])
            for altitude in altitudes:
                footprint = config.camera.footprint_m(altitude)
                half_diagonal = np.hypot(*footprint) / 2.0
                for yaw_offset in yaw_offsets:
                    yaw = (edge_azimuth + yaw_offset) % 360.0
                    query, query_valid, _ = sampler.render_query(
                        config.camera, midpoint, altitude, yaw,
                        config.matching_gsd_m)
                    for radius in radii:
                        for degradation in degradations:
                            degraded_query = degrade_image(query, degradation)
                            for repetition in range(repetitions):
                                bearing = rng.uniform(0.0, 2.0 * np.pi)
                                magnitude = radius * np.sqrt(rng.uniform(0.0, 1.0))
                                prior = midpoint + magnitude * np.asarray(
                                    [np.sin(bearing), np.cos(bearing)])
                                reference, reference_valid, reference_transform = (
                                    sampler.render_reference(
                                        prior, radius + half_diagonal + 5.0,
                                        config.matching_gsd_m))
                                q_fraction = float(query_valid.mean())
                                r_fraction = float(reference_valid.mean())
                                if q_fraction < .98 or r_fraction < .90:
                                    result = None
                                    outcome = "reject"
                                    reject_reason = "insufficient_map_coverage"
                                    position_error = yaw_error = np.nan
                                else:
                                    result = matcher.localize(
                                        degraded_query, reference,
                                        reference_transform)
                                    reject_reason = result.reason if not result.accepted else ""
                                    if not result.accepted:
                                        outcome = "reject"
                                        position_error = yaw_error = np.nan
                                    else:
                                        position_error = float(np.linalg.norm(
                                            np.asarray(result.estimated_xy) - midpoint))
                                        yaw_error = circular_error_deg(
                                            result.estimated_yaw_deg, yaw)
                                        correct = (
                                            position_error <= config.outcome.maximum_position_error_m
                                            and yaw_error <= config.outcome.maximum_yaw_error_deg)
                                        outcome = "correct" if correct else "false_accept"
                                row = {
                                    "trial_id": trial_id,
                                    "edge_index": int(edge_index),
                                    "midpoint_x_m": float(midpoint[0]),
                                    "midpoint_y_m": float(midpoint[1]),
                                    "edge_azimuth_deg": edge_azimuth,
                                    "altitude_agl_m": float(altitude),
                                    "yaw_deg": float(yaw),
                                    "yaw_offset_deg": float(yaw_offset),
                                    "search_radius_m": float(radius),
                                    "prior_error_m": float(magnitude),
                                    "prior_x_m": float(prior[0]),
                                    "prior_y_m": float(prior[1]),
                                    "scenario_id": degradation.scenario_id,
                                    "repetition": repetition,
                                    "outcome": outcome,
                                    "reject_reason": reject_reason,
                                    "position_error_m": position_error,
                                    "yaw_error_deg": yaw_error,
                                    "estimated_x_m": (np.nan if result is None or
                                                       result.estimated_xy is None else
                                                       result.estimated_xy[0]),
                                    "estimated_y_m": (np.nan if result is None or
                                                       result.estimated_xy is None else
                                                       result.estimated_xy[1]),
                                    "estimated_yaw_deg": (np.nan if result is None else
                                                          result.estimated_yaw_deg),
                                    "query_keypoints": 0 if result is None else result.query_keypoints,
                                    "reference_keypoints": 0 if result is None else result.reference_keypoints,
                                    "good_matches": 0 if result is None else result.good_matches,
                                    "inliers": 0 if result is None else result.inliers,
                                    "inlier_ratio": 0.0 if result is None else result.inlier_ratio,
                                    "median_reprojection_error_px": (np.nan if result is None else
                                                                      result.median_reprojection_error_px),
                                    "runtime_ms": 0.0 if result is None else result.runtime_ms,
                                    "query_valid_fraction": q_fraction,
                                    "reference_valid_fraction": r_fraction,
                                }
                                rows.append(row)
                                trial_id += 1

    scenario_ids = tuple(item.scenario_id for item in degradations)
    aggregates = aggregate_trials(rows, selected_edges, scenario_ids,
                                  altitudes, radii)
    _write_csv(output / "trials.csv", rows)
    np.savez_compressed(output / "vml_sample_outcomes.npz", **aggregates)
    summary = analyse_rows(rows)
    summary.update({
        "smoke": bool(smoke),
        "camera_horizontal_fov_deg": config.camera.horizontal_fov_deg,
        "camera_vertical_fov_deg": config.camera.vertical_fov_deg,
        "selected_edges": list(map(int, selected_edges)),
    })
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_same_orthophoto": True,
        "claim_limit": "feasibility and spatial ranking; not real-flight probability",
        "configuration_path": str(config.config_path),
        "configuration_sha256": _sha256(config.config_path),
        "configuration": config.raw,
        "scene_artifact": str(config.scene_artifact_path),
        "scene_artifact_sha256": _sha256(config.scene_artifact_path),
        "scene_manifest": str(config.scene_manifest_path),
        "source_fingerprints": scene_manifest.get("source_fingerprints", {}),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_plot(output / "outcomes.png", rows)
    return summary

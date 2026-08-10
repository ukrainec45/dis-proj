"""Tile-level landmark quality measures for planning and diagnostics."""

import numpy as np


def spatial_coverage(features, core, bins=3):
    """Fraction of a coarse tile grid containing at least one feature."""
    if not features:
        return 0.0
    r0, r1, c0, c1 = core
    occupied = set()
    for feature in features:
        x = min(bins - 1, max(0, int((feature.col - c0) * bins / max(c1 - c0, 1))))
        y = min(bins - 1, max(0, int((feature.row - r0) * bins / max(r1 - r0, 1))))
        occupied.add((x, y))
    return len(occupied) / float(bins * bins)


def assign_quality(tile_metrics, max_features):
    """Add a 0..1 quality score using density, coverage, and response strength."""
    if not tile_metrics:
        return tile_metrics
    responses = np.asarray([item["mean_response"] for item in tile_metrics], dtype=float)
    response_scale = float(np.max(responses))
    for item in tile_metrics:
        density = min(1.0, item["feature_count"] / float(max_features))
        response = 0.0 if response_scale <= 0 else item["mean_response"] / response_scale
        item["quality"] = float(0.6 * density + 0.3 * item["coverage"] + 0.1 * response)
    return tile_metrics

"""Synthetic smoke-density and visibility layers from the notebook."""

import numpy as np


def generate_smoke_sources(n_sources, x_range, y_range, sigma_range=(500, 2500),
                           seed=None, rng=None):
    """Generate plume sources, optionally continuing an existing RNG stream."""
    if rng is None:
        rng = np.random.RandomState(seed)
    return [{"x0": rng.uniform(*x_range), "y0": rng.uniform(*y_range),
             "sigma": rng.uniform(*sigma_range), "min_vis": rng.uniform(0.2, 0.4)}
            for _ in range(n_sources)]


def gaussian_density(x, y, x0, y0, sigma):
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma**2))


def combine_sources(density_stack):
    return 1.0 - np.prod(1.0 - density_stack, axis=0)


def compute_visibility(density_stack, sources):
    scaled = np.zeros_like(density_stack)
    for i, source in enumerate(sources):
        scaled[i] = (1.0 - source["min_vis"]) * density_stack[i]
    return 1.0 - combine_sources(scaled)


def aggregate_to_planner_grid(layer, upscale):
    """Mean-pool a fine layer into complete square planner cells."""
    rows, cols = layer.shape
    planner_rows, planner_cols = rows // upscale, cols // upscale
    out = np.zeros((planner_rows, planner_cols))
    for row in range(planner_rows):
        for col in range(planner_cols):
            out[row, col] = np.mean(layer[row * upscale:(row + 1) * upscale,
                                         col * upscale:(col + 1) * upscale])
    return out

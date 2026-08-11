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


def generate_local_smoke_sources(x_range, y_range, rng):
    """Generate a sparse, AOI-scaled synthetic visibility-degradation field.

    The original notebook's 500--2500 m plume radii were suitable for a much
    larger Sentinel-2 scene, but cover a small orthophoto AOI almost entirely.
    Use one to three local plumes whose standard deviations are 4--10% of the
    shorter AOI dimension, preserving clear regions for route trade-offs.
    """
    short_dimension = min(abs(x_range[1] - x_range[0]),
                          abs(y_range[1] - y_range[0]))
    if short_dimension <= 0:
        raise ValueError("smoke-source ranges must span a positive AOI area")
    return generate_smoke_sources(rng.randint(1, 4), x_range, y_range,
                                  sigma_range=(.04 * short_dimension,
                                               .10 * short_dimension), rng=rng)


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

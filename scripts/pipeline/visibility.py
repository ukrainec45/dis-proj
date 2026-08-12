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


def generate_local_smoke_sources(x_range, y_range, rng, n_sources=None):
    """Generate a sparse, AOI-scaled synthetic visibility-degradation field.

    The original notebook's 500--2500 m plume radii were suitable for a much
    larger Sentinel-2 scene, but cover a small orthophoto AOI almost entirely.
    The caller may request an exact number of local plumes.  Otherwise one to
    three plumes are sampled for backwards-compatible exploratory use.
    """
    short_dimension = min(abs(x_range[1] - x_range[0]),
                          abs(y_range[1] - y_range[0]))
    if short_dimension <= 0:
        raise ValueError("smoke-source ranges must span a positive AOI area")
    if n_sources is None:
        n_sources = rng.randint(1, 4)
    if n_sources < 1:
        raise ValueError("n_sources must be positive")
    return generate_smoke_sources(n_sources, x_range, y_range,
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


def visibility_from_zones(extent, rows, cols, zones, zone_visibility):
    """Return a planner-grid visibility layer from intersecting zone polygons.

    A zone touching any part of a planner cell applies the configured value to
    that cell.  The minimum operation makes the policy conservative if callers
    later provide features with different visibility values.
    """
    from shapely.geometry import box

    if not 0.0 <= zone_visibility <= 1.0:
        raise ValueError("zone_visibility must be in [0, 1]")
    if rows <= 0 or cols <= 0:
        raise ValueError("planner grid dimensions must be positive")
    visibility = np.ones((rows, cols), dtype=float)
    xmin, xmax, ymin, ymax = extent
    cell_width, cell_height = (xmax - xmin) / cols, (ymax - ymin) / rows
    for row in range(rows):
        y_top, y_bottom = ymax - row * cell_height, ymax - (row + 1) * cell_height
        for col in range(cols):
            x_left, x_right = xmin + col * cell_width, xmin + (col + 1) * cell_width
            if zones.intersects(box(x_left, y_bottom, x_right, y_top)).any():
                visibility[row, col] = min(visibility[row, col], zone_visibility)
    return visibility

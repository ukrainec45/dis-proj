"""Unit tests for dependency-light extracted notebook processing stages."""

import numpy as np

from .cell_vectors import to_planner_layers
from .raster import create_grid, normalize
from .visibility import (aggregate_to_planner_grid, combine_sources,
                         compute_visibility, gaussian_density,
                         generate_smoke_sources)
from .wind import get_wind_direction, get_wind_speed


def test_normalize_and_grid_partition_preserve_layout():
    image = np.arange(16, dtype=float).reshape(4, 4, 1)
    grid = create_grid(image, cell_size_m=2, pixel_size_m=1)
    assert len(grid) == len(grid[0]) == 2
    assert np.array_equal(grid[1][0][:, :, 0], image[2:4, 0:2, 0])
    assert normalize(np.ones((2, 2))).sum() == 0


def test_smoke_visibility_and_aggregation_are_bounded_and_deterministic():
    sources = generate_smoke_sources(2, (0, 10), (0, 10), seed=7)
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    stack = np.stack([gaussian_density(x, y, source["x0"], source["y0"], source["sigma"])
                      for source in sources])
    visibility = compute_visibility(stack, sources)
    assert np.all((visibility >= 0) & (visibility <= 1))
    assert np.allclose(combine_sources(stack), 1 - np.prod(1 - stack, axis=0))
    assert aggregate_to_planner_grid(visibility, 2).shape == (2, 2)


def test_wind_helpers_and_layer_conversion():
    assert get_wind_speed(3, 4) == 5
    assert get_wind_direction(1, 0) == 90
    dtype = [("z_dem", float), ("phi_vis", float), ("phi_ter", float),
             ("phi_vsb", float), ("u_wind", float), ("v_wind", float), ("in_nfz", bool)]
    cells = np.array([(10, .2, .5, .8, 1, 2, False), (20, .9, .1, .7, 3, 4, True)], dtype=dtype)
    layers = to_planner_layers(cells, 1, 2)
    assert np.array_equal(layers["nav_density"], [[.5, .9]])
    assert layers["occupancy"].tolist() == [[False, True]]
    assert layers["wind_field"].shape == (1, 2, 2)

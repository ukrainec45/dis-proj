"""Unit tests for dependency-light extracted notebook processing stages."""

import numpy as np
import pytest

from .cell_vectors import (build_cell_vectors, derive_landing_sites,
                           to_planner_layers)
from .nav_quality import compute_rgb_localization_metric
from .raster import create_grid, normalize, read_rgb_aoi
from .visibility import (aggregate_to_planner_grid, combine_sources,
                         compute_visibility, gaussian_density,
                         generate_local_smoke_sources, generate_smoke_sources)
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


def test_smoke_source_rng_can_continue_the_notebook_sequence():
    rng = np.random.RandomState(42)
    n_sources = rng.randint(3, 11)
    actual = generate_smoke_sources(n_sources, (0, 10), (0, 10), rng=rng)
    expected_rng = np.random.RandomState(42)
    expected_n = expected_rng.randint(3, 11)
    expected = generate_smoke_sources(expected_n, (0, 10), (0, 10), rng=expected_rng)
    assert n_sources == expected_n
    assert actual == expected


def test_local_smoke_sources_are_sparse_and_scaled_to_the_aoi():
    sources = generate_local_smoke_sources((100.0, 1100.0), (0.0, 2000.0),
                                           rng=np.random.RandomState(42))
    assert 1 <= len(sources) <= 3
    assert all(40.0 <= source["sigma"] <= 100.0 for source in sources)


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


def test_landing_sites_require_flat_visible_non_nfz_cells():
    dem = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]])
    visibility = np.full((3, 3), 0.8)
    visibility[0, 0] = 0.6
    occupancy = np.zeros((3, 3), dtype=bool)
    occupancy[2, 2] = True
    sites = derive_landing_sites(dem, visibility, occupancy, resolution_m=10.0,
                                 min_visibility=0.7, max_slope=0.05)
    assert not sites[0, 0]
    assert not sites[2, 2]
    assert sites[0, 2]


def test_nfz_intersection_blocks_a_cell_touched_at_its_boundary():
    pytest.importorskip("shapely")
    from shapely.geometry import box

    class BoundaryNfz:
        def intersects(self, cell):
            return np.asarray([cell.intersects(box(1.0, 0.2, 1.2, 0.8))])

    patches = [[np.zeros((2, 2, 1)), np.zeros((2, 2, 1))]]
    cells = build_cell_vectors((0.0, 2.0, 0.0, 1.0), 1, 2, patches,
                               np.ones((1, 2)), np.ones((1, 2)),
                               np.ones((1, 2)), np.zeros((1, 2)),
                               np.zeros((1, 2)), BoundaryNfz())
    assert cells[0]["in_nfz"] and cells[1]["in_nfz"]


def test_rgb_localization_metric_is_bounded_and_uses_rgb_cells():
    pytest.importorskip("skimage")
    checkerboard = (np.indices((16, 16)).sum(axis=0) % 2 * 255).astype(np.uint8)
    rgb = np.dstack((checkerboard, checkerboard, checkerboard))
    metric, corners, texture = compute_rgb_localization_metric([[rgb, rgb], [rgb, rgb]])
    assert metric.shape == (2, 2)
    assert np.all((metric >= 0) & (metric <= 1))
    assert np.all((corners >= 0) & (corners <= 1))
    assert np.all((texture >= 0) & (texture <= 1))


def test_read_rgb_aoi_reprojects_to_metric_aoi_grid(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    class Aoi:
        crs = "EPSG:32635"
        total_bounds = (500000.0, 5500000.0, 500004.0, 5500004.0)

    path = tmp_path / "rgb.tif"
    with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=3,
                       dtype="uint8", crs=Aoi.crs,
                       transform=from_origin(500000.0, 5500004.0, 1.0, 1.0)) as dst:
        dst.write(np.full((3, 4, 4), 100, dtype=np.uint8))
    rgb, extent, transform = read_rgb_aoi(path, Aoi(), pixel_size_m=2.0)
    assert rgb.shape == (2, 2, 3)
    assert extent == [500000.0, 500004.0, 5500000.0, 5500004.0]
    assert transform.a == 2.0 and transform.e == -2.0

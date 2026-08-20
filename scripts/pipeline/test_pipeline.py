"""Unit tests for dependency-light extracted notebook processing stages."""

import numpy as np
import pytest

from .cell_vectors import (build_cell_vectors, build_edge_terrain_excess,
                           derive_landing_sites,
                           landing_site_coordinates, select_landing_sites,
                           to_planner_layers)
from .nav_quality import compute_rgb_localization_metric
from .raster import create_grid, normalize, read_rgb_aoi
from .visibility import (aggregate_to_planner_grid, combine_sources,
                         compute_visibility, gaussian_density,
                         generate_local_smoke_sources, generate_smoke_sources,
                         visibility_from_zones)
from .build_layers import (_read_visibility_zones, _validate_energy_options,
                           _validate_terrain_clearance_options,
                           _validate_visibility_options)
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


def test_visibility_zones_degrade_intersecting_cells_with_conservative_overlap():
    pytest.importorskip("geopandas")
    import geopandas as gpd
    from shapely.geometry import box

    zones = gpd.GeoDataFrame(geometry=[box(.8, .8, 1.2, 1.2)], crs="EPSG:32635")
    visibility = visibility_from_zones((0, 2, 0, 2), 2, 2, zones, .35)
    assert np.array_equal(visibility, np.full((2, 2), .35))
    assert np.array_equal(visibility_from_zones((0, 2, 0, 2), 2, 2,
                                                 zones.iloc[0:0], .35), np.ones((2, 2)))


def test_visibility_options_default_to_clear_and_reject_invalid_combinations():
    _validate_visibility_options(None, None, None, None)
    with pytest.raises(ValueError, match="required"):
        _validate_visibility_options(object(), None, None, None)
    with pytest.raises(ValueError, match="together"):
        _validate_visibility_options(None, None, 1, None)
    with pytest.raises(ValueError, match="cannot"):
        _validate_visibility_options(object(), .3, 1, .2)


def test_visibility_zone_loader_assumes_utm_when_crs_is_missing(monkeypatch):
    pytest.importorskip("geopandas")
    import geopandas as gpd
    from shapely.geometry import box

    zones = gpd.GeoDataFrame(geometry=[box(500000, 5500000, 500010, 5500010)])
    monkeypatch.setattr(gpd, "read_file", lambda _: zones)
    loaded = _read_visibility_zones("zones.geojson")
    assert str(loaded.crs) == "EPSG:32635"


def test_visibility_zone_loader_reprojects_a_declared_crs(monkeypatch):
    pytest.importorskip("geopandas")
    import geopandas as gpd
    from shapely.geometry import box

    zones = gpd.GeoDataFrame(geometry=[box(28.0, 50.0, 28.01, 50.01)], crs="EPSG:4326")
    monkeypatch.setattr(gpd, "read_file", lambda _: zones)
    loaded = _read_visibility_zones("zones.geojson")
    assert str(loaded.crs) == "EPSG:32635"
    assert loaded.total_bounds[0] > 500000


def test_energy_options_require_a_positive_capacity_power_and_valid_reserve():
    _validate_energy_options(300.0, 40.0, 450.0)
    with pytest.raises(ValueError, match="battery_energy_wh"):
        _validate_energy_options(0.0, 0.0, 450.0)
    with pytest.raises(ValueError, match="smaller"):
        _validate_energy_options(300.0, 300.0, 450.0)
    with pytest.raises(ValueError, match="cruise_power_w"):
        _validate_energy_options(300.0, 40.0, 0.0)


def test_terrain_clearance_defaults_to_cruise_agl_and_rejects_invalid_values():
    assert _validate_terrain_clearance_options(120.0, None) == 120.0
    assert _validate_terrain_clearance_options(120.0, 80.0) == 80.0
    with pytest.raises(ValueError, match="non-negative"):
        _validate_terrain_clearance_options(-1.0, None)
    with pytest.raises(ValueError, match="cannot exceed"):
        _validate_terrain_clearance_options(120.0, 121.0)


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


def test_dem_layers_use_cell_centre_and_preserve_cell_maximum():
    patch = np.array([[1.0, 2.0, 9.0],
                      [3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0]])[:, :, None]
    cells = build_cell_vectors((0.0, 1.0, 0.0, 1.0), 1, 1, [[patch]],
                               np.ones((1, 1)), np.ones((1, 1)),
                               np.ones((1, 1)), np.zeros((1, 1)),
                               np.zeros((1, 1)))
    layers = to_planner_layers(cells, 1, 1)
    assert layers["dem"][0, 0] == 4.0
    assert layers["dem_max"][0, 0] == 9.0


def test_detailed_dem_edge_profile_detects_a_between_vertex_peak():
    left = np.zeros((3, 3, 1), dtype=float)
    right = np.zeros((3, 3, 1), dtype=float)
    left[1, 2, 0] = 5.0
    excess = build_edge_terrain_excess([[left, right]])
    assert excess.shape == (1, 2, 3, 3)
    assert excess[0, 0, 1, 2] == pytest.approx(5.0)
    assert excess[0, 1, 1, 0] == pytest.approx(5.0)


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


def test_landing_sites_are_limited_safe_and_distributed_along_corridor():
    dem = np.zeros((11, 11), dtype=float)
    visibility = np.ones_like(dem)
    occupancy = np.zeros_like(dem, dtype=bool)
    occupancy[5, 5] = True
    sites = select_landing_sites(dem, visibility, occupancy, resolution_m=10.0,
                                 start=(0, 5), goal=(10, 5), max_sites=5,
                                 corridor_width_m=10.0)
    assert 1 <= sites.sum() <= 5
    assert not sites[5, 5]
    selected_columns = np.where(sites)[1]
    assert selected_columns.min() <= 2
    assert selected_columns.max() >= 8


def test_landing_site_coordinates_are_cell_centres():
    sites = np.zeros((2, 2), dtype=bool)
    sites[0, 1] = True
    sites[1, 0] = True
    coordinates = landing_site_coordinates(sites, (100.0, 120.0, 200.0, 220.0))
    assert np.allclose(coordinates, [[115.0, 215.0], [105.0, 205.0]])


def test_landing_site_limit_must_be_positive():
    grid = np.zeros((2, 2), dtype=float)
    with pytest.raises(ValueError, match="positive integer"):
        select_landing_sites(grid, np.ones_like(grid), grid.astype(bool), 10.0,
                             start=(0, 0), goal=(1, 1), max_sites=0)


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

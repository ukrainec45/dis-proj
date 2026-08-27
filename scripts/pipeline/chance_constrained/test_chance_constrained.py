import json
from pathlib import Path

import numpy as np
import pytest

from .artifacts import load_scene_edges
from .build_inputs import build_inputs, main
from .config import load_config
from .edges import build_directed_edges
from ..utils.grid import segment_circle_entry_fraction


def _base_config(tmp_path):
    sources = {}
    for name in ("rgb.tif", "dem.tif", "aoi.geojson", "mission.geojson"):
        path = tmp_path / name
        path.write_bytes(b"placeholder")
        sources[name] = path
    return {
        "schema_version": 1,
        "sources": {
            "rgb_image": sources["rgb.tif"].name,
            "dem": sources["dem.tif"].name,
            "aoi": sources["aoi.geojson"].name,
            "mission_points": sources["mission.geojson"].name,
        },
        "grid": {"target_crs": "EPSG:32635", "pixel_size_m": 1,
                 "cell_size_m": 2, "edge_profile_samples": 5},
        "delivery_zone": {"radius_m": 1},
        "vehicle": {"max_wind_mps": 12,
                    "minimum_terrain_clearance_m": 5},
        "operating_points": {"altitudes_agl_m": [10, 20],
                             "airspeeds_mps": [5, 10]},
        "scenarios": [{"id": "nominal", "visibility_scale": 1,
                       "wind": {"mode": "constant", "east_mps": 0,
                                "north_mps": 0}}],
        "sensors": {"status": "model_assumptions"},
        "monte_carlo": {"seed": 42},
    }


def test_config_rejects_invalid_delivery_radius_and_grid_ratio(tmp_path):
    data = _base_config(tmp_path)
    config_path = tmp_path / "config.json"
    data["delivery_zone"]["radius_m"] = 0
    config_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="radius_m"):
        load_config(config_path)
    data["delivery_zone"]["radius_m"] = 1
    data["grid"]["cell_size_m"] = 2.5
    config_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="integer multiple"):
        load_config(config_path)


def test_segment_circle_entry_is_continuous_and_directional():
    assert segment_circle_entry_fraction((0, 0), (10, 0), (7, 0), 2) == pytest.approx(.5)
    assert segment_circle_entry_fraction((10, 0), (0, 0), (7, 0), 2) == pytest.approx(.1)
    assert segment_circle_entry_fraction((6, 0), (10, 0), (7, 0), 2) == 0
    assert np.isnan(segment_circle_entry_fraction((0, 0), (4, 0), (7, 0), 2))


def test_directed_edges_capture_wind_goal_entry_and_corner_blocking():
    dem_fine = np.zeros((4, 4), dtype=float)
    dem_center = np.zeros((2, 2), dtype=float)
    occupancy = np.zeros((2, 2), dtype=bool)
    occupancy[0, 1] = True
    visibility = np.ones((1, 2, 2), dtype=float)
    wind = np.zeros((1, 2, 2, 2), dtype=float)
    wind[..., 0] = 2.0
    result = build_directed_edges(
        dem_fine=dem_fine, dem_center=dem_center, occupancy=occupancy,
        visibility=visibility, wind_field=wind, extent=(0, 20, 0, 20),
        pixel_size_m=5, cell_size_m=10, altitudes_agl_m=(10,),
        airspeeds_mps=(5,), max_wind_mps=12,
        minimum_terrain_clearance_m=5, delivery_center_xy=(10, 5),
        delivery_radius_m=2, profile_samples=5)
    assert len(result["edge_start_cell_xy"]) == 12

    def edge_index(start, end):
        matches = np.all(result["edge_start_cell_xy"] == start, axis=1)
        matches &= np.all(result["edge_end_cell_xy"] == end, axis=1)
        return int(np.flatnonzero(matches)[0])

    east = edge_index((0, 1), (1, 1))
    west = edge_index((1, 1), (0, 1))
    assert result["edge_travel_time_s"][east, 0, 0, 0] < result["edge_travel_time_s"][west, 0, 0, 0]
    assert result["edge_goal_entry_fraction"][east] == pytest.approx(.3)
    diagonal = edge_index((0, 0), (1, 1))
    assert not result["edge_spatial_feasible"][diagonal, 0]


def _write_geospatial_fixture(tmp_path):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import Point, box

    rgb_path, dem_path = tmp_path / "rgb.tif", tmp_path / "dem.tif"
    transform = from_origin(0, 6, 1, 1)
    rgb = np.arange(3 * 6 * 6, dtype=np.uint8).reshape(3, 6, 6)
    with rasterio.open(rgb_path, "w", driver="GTiff", width=6, height=6,
                       count=3, dtype="uint8", crs="EPSG:32635",
                       transform=transform) as dst:
        dst.write(rgb)
    with rasterio.open(dem_path, "w", driver="GTiff", width=6, height=6,
                       count=1, dtype="float32", crs="EPSG:32635",
                       transform=transform) as dst:
        dst.write(np.zeros((1, 6, 6), dtype=np.float32))
    aoi_path = tmp_path / "aoi.geojson"
    mission_path = tmp_path / "mission.geojson"
    gpd.GeoDataFrame(geometry=[box(0, 0, 6, 6)], crs="EPSG:32635").to_file(
        aoi_path, driver="GeoJSON")
    gpd.GeoDataFrame({"role": ["start", "goal"]},
                     geometry=[Point(1, 1), Point(5, 5)],
                     crs="EPSG:32635").to_file(mission_path, driver="GeoJSON")
    return rgb_path, dem_path, aoi_path, mission_path


def test_end_to_end_artifact_round_trip_and_source_verification(tmp_path, capsys):
    data = _base_config(tmp_path)
    rgb, dem, aoi, mission = _write_geospatial_fixture(tmp_path)
    data["sources"] = {"rgb_image": rgb.name, "dem": dem.name,
                       "aoi": aoi.name, "mission_points": mission.name}
    data["delivery_zone"]["radius_m"] = .75
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    output_dir = tmp_path / "output"

    arrays, artifact_path, manifest_path = build_inputs(config_path, output_dir)
    assert tuple(arrays["grid_shape"]) == (3, 3)
    assert int(arrays["edge_count"]) == 40
    assert arrays["delivery_zone_mask"][0, 2]
    assert arrays["edge_feasible"].shape == (40, 1, 2, 2)
    loaded = load_scene_edges(artifact_path, manifest_path=manifest_path,
                              verify_sources=True)
    np.testing.assert_array_equal(loaded["edge_start_cell_xy"],
                                  arrays["edge_start_cell_xy"])
    with pytest.raises(FileExistsError):
        build_inputs(config_path, output_dir)

    cli_output = tmp_path / "cli-output"
    assert main(["--config", str(config_path),
                 "--output-dir", str(cli_output)]) == 0
    assert "3x3 cells, 40 directed edges" in capsys.readouterr().out
    assert (cli_output / "scene_edges.npz").is_file()

    aoi.write_text(aoi.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        load_scene_edges(artifact_path, manifest_path=manifest_path,
                         verify_sources=True)


def test_unknown_artifact_version_is_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, format_version=np.asarray(99), crs=np.asarray("EPSG:32635"))
    with pytest.raises(ValueError, match="unsupported"):
        load_scene_edges(path)


def test_legacy_imports_reexport_shared_helpers():
    from scripts.pipeline.raster import create_grid as legacy_grid
    from scripts.pipeline.utils.grid import create_grid
    from scripts.pipeline.visibility import compute_visibility as legacy_visibility
    from scripts.pipeline.utils.visibility import compute_visibility
    from scripts.pipeline.wind import get_wind_speed as legacy_wind
    from scripts.pipeline.utils.wind import get_wind_speed

    assert legacy_grid is create_grid
    assert legacy_visibility is compute_visibility
    assert legacy_wind is get_wind_speed

"""Tests for landmark SQLite schema and geospatial build behaviour."""

from pathlib import Path

import numpy as np
import pytest

from .database import LandmarkDatabase, LandmarkDatabaseWriter, SCHEMA_VERSION
from .build_database import _tiles
from .geospatial import GeospatialInputError, validate_metric_crs


def _metadata():
    return {"schema_version": SCHEMA_VERSION, "grid_origin_e": 100.0,
            "grid_origin_n": 200.0, "planner_cell_size_m": 10.0}


def _small_database(path):
    writer = LandmarkDatabaseWriter(path)
    try:
        writer.write_metadata(_metadata())
        west = writer.write_tile({"row": 0, "col": 0, "min_e": 100.0, "min_n": 190.0,
                                  "max_e": 110.0, "max_n": 200.0, "center_e": 105.0,
                                  "center_n": 195.0, "mean_elevation": 20.0, "feature_count": 1,
                                  "coverage": 1 / 9, "mean_response": 0.1, "quality": 0.4})
        east = writer.write_tile({"row": 0, "col": 1, "min_e": 110.0, "min_n": 190.0,
                                  "max_e": 120.0, "max_n": 200.0, "center_e": 115.0,
                                  "center_n": 195.0, "mean_elevation": 21.0, "feature_count": 1,
                                  "coverage": 1 / 9, "mean_response": 0.2, "quality": 0.9})
        descriptor_a = bytes(range(32))
        descriptor_b = bytes(reversed(range(32)))
        writer.write_landmarks(west, [(105.0, 195.0, 20.0, 5.0, 5.0, 31.0, 0.0, 0.1, 0,
                                       0.2, 0.0, 0.0, 1.0, descriptor_a)])
        writer.write_landmarks(east, [(115.0, 195.0, 21.0, 15.0, 5.0, 31.0, 0.0, 0.2, 0,
                                       0.3, 0.0, 0.0, 1.0, descriptor_b)])
    finally:
        writer.close()


def test_reader_uses_tile_bounded_radius_query(tmp_path):
    path = tmp_path / "landmarks.sqlite"
    _small_database(path)
    with LandmarkDatabase(path) as database:
        assert database.tile_at(105.0, 195.0)["col"] == 0
        batch = database.query_nearby(105.0, 195.0, 6.0, 10)
        assert batch.map_xyz.shape == (1, 3)
        assert batch.descriptors.shape == (1, 32)
        assert batch.descriptors.dtype == np.uint8
        assert batch.tile_quality.tolist() == [pytest.approx(0.4)]
        assert database.query_nearby(105.0, 195.0, 50.0, 1).map_xyz.shape == (1, 3)


def test_reader_rejects_invalid_query_limit(tmp_path):
    path = tmp_path / "landmarks.sqlite"
    _small_database(path)
    with LandmarkDatabase(path) as database:
        with pytest.raises(ValueError, match="max_landmarks"):
            database.query_nearby(105.0, 195.0, 1.0, 0)
        with pytest.raises(ValueError, match="radius_m"):
            database.query_nearby(105.0, 195.0, -1.0, 1)


def test_tiles_require_an_integer_number_of_source_pixels():
    class Transform:
        a, e, c, f = 2.0, -2.0, 0.0, 100.0
    with pytest.raises(ValueError, match="integer multiple"):
        list(_tiles((20, 20), Transform(), 5.0))


def test_geographic_reference_crs_is_rejected():
    rasterio = pytest.importorskip("rasterio")
    with pytest.raises(GeospatialInputError, match="projected metric"):
        validate_metric_crs(rasterio.crs.CRS.from_epsg(4326))


def test_geotiff_build_attaches_coordinates_elevation_and_descriptors(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("cv2")
    from shapely.geometry import box
    from rasterio.transform import from_origin
    from .build_database import build_landmark_database

    transform = from_origin(500000, 1000, 1, 1)
    image = np.indices((80, 80)).sum(axis=0).astype(np.uint8)
    image = ((image // 5) % 2 * 255).astype(np.uint8)  # repeatable corner-rich checkerboard
    image_path, dem_path, aoi_path = (tmp_path / "reference.tif", tmp_path / "dem.tif", tmp_path / "aoi.geojson")
    profile = {"driver": "GTiff", "height": 80, "width": 80, "count": 1,
               "dtype": "uint8", "crs": "EPSG:32635", "transform": transform}
    with rasterio.open(image_path, "w", **profile) as destination:
        destination.write(image, 1)
    with rasterio.open(dem_path, "w", **{**profile, "dtype": "float32"}) as destination:
        destination.write(np.arange(6400, dtype=np.float32).reshape(80, 80), 1)
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(500000, 920, 500080, 1000)], crs="EPSG:32635").to_file(aoi_path, driver="GeoJSON")
    output = tmp_path / "landmarks.sqlite"
    result = build_landmark_database(image_path, dem_path, aoi_path, output, 20, max_features_per_tile=50)
    assert Path(result["database"]).is_file()
    assert Path(result["quality"]).is_file()
    with LandmarkDatabase(output) as database:
        metadata = database.metadata()
        assert metadata["image_sha256"]
        batch = database.query_nearby(500040, 960, 100, 500)
        assert len(batch.map_xyz) > 0
        assert batch.descriptors.shape[1] == 32
        assert np.all(np.isfinite(batch.map_xyz))

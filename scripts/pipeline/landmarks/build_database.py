"""Build a portable 2.5D ORB landmark database from GeoTIFF, DEM, and AOI.

Example:
    python -m scripts.pipeline.landmarks.build_database --image reference.tif \
        --dem dem.tif --aoi aoi.geojson --output landmarks.sqlite \
        --planner-cell-size-m 50
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .database import LandmarkDatabaseWriter, SCHEMA_VERSION
from .extractor import extract_orb_tile, prepare_feature_image
from .geospatial import load_reference_data
from .quality import assign_quality, spatial_coverage


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _terrain_context(dem, pixel_size_m):
    """Compute slope and upward normals on the aligned DEM raster."""
    filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    north_derivative, east_derivative = np.gradient(filled, pixel_size_m[1], pixel_size_m[0])
    slope = np.hypot(east_derivative, north_derivative)
    normal = np.dstack((-east_derivative, -north_derivative, np.ones_like(dem)))
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-12)
    return slope.astype(np.float32), normal.astype(np.float32)


def _tiles(shape, transform, cell_size_m):
    """Yield core pixel windows and metric bounds on a regular planner-aligned grid."""
    rows, cols = shape
    px_e, px_n = abs(float(transform.a)), abs(float(transform.e))
    if not np.isclose(px_e, px_n, rtol=0, atol=max(px_e, px_n) * 1e-6):
        raise ValueError("reference GeoTIFF must have square pixels for planner-cell tiling")
    pixels = cell_size_m / px_e
    if not np.isclose(pixels, round(pixels), rtol=0, atol=1e-6):
        raise ValueError("planner_cell_size_m must be an integer multiple of the GeoTIFF pixel size")
    step = int(round(pixels))
    if step < 1:
        raise ValueError("planner_cell_size_m must be at least one GeoTIFF pixel")
    for row, r0 in enumerate(range(0, rows, step)):
        r1 = min(rows, r0 + step)
        for col, c0 in enumerate(range(0, cols, step)):
            c1 = min(cols, c0 + step)
            min_e = transform.c + c0 * transform.a
            max_e = transform.c + c1 * transform.a
            max_n = transform.f + r0 * transform.e
            min_n = transform.f + r1 * transform.e
            yield row, col, (r0, r1, c0, c1), (min_e, min_n, max_e, max_n)


def _coordinate(transform, col, row):
    east, north = transform * (col + 0.5, row + 0.5)
    return float(east), float(north)


def build_landmark_database(image_path, dem_path, aoi_path, output_path, planner_cell_size_m,
                            max_features_per_tile=300, feature_band=None, halo_px=32,
                            overwrite=False):
    """Create SQLite landmark package and adjacent planner-aligned quality NPZ."""
    if planner_cell_size_m <= 0:
        raise ValueError("planner_cell_size_m must be positive")
    reference = load_reference_data(image_path, dem_path, aoi_path)
    feature_image = prepare_feature_image(reference.image, reference.valid_mask, feature_band)
    slope, normal = _terrain_context(reference.dem, reference.pixel_size_m)
    pending = []
    for row, col, core, bounds in _tiles(feature_image.shape, reference.transform, planner_cell_size_m):
        features = extract_orb_tile(feature_image, reference.valid_mask, core, halo_px, max_features_per_tile)
        r0, r1, c0, c1 = core
        core_dem = reference.dem[r0:r1, c0:c1]
        pending.append({
            "row": row, "col": col, "core": core, "bounds": bounds, "features": features,
            "min_e": float(bounds[0]), "min_n": float(bounds[1]), "max_e": float(bounds[2]), "max_n": float(bounds[3]),
            "center_e": float((bounds[0] + bounds[2]) / 2), "center_n": float((bounds[1] + bounds[3]) / 2),
            "mean_elevation": float(np.nanmean(core_dem)) if np.any(np.isfinite(core_dem)) else None,
            "feature_count": len(features), "coverage": spatial_coverage(features, core),
            "mean_response": float(np.mean([item.response for item in features])) if features else 0.0,
        })
    assign_quality(pending, max_features_per_tile)

    output_path = Path(output_path)
    writer = LandmarkDatabaseWriter(output_path, overwrite=overwrite)
    quality_grid = np.zeros((max(item["row"] for item in pending) + 1, max(item["col"] for item in pending) + 1), dtype=np.float32)
    count_grid = np.zeros_like(quality_grid, dtype=np.int32)
    try:
        writer.write_metadata({
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "crs_wkt": reference.crs_wkt,
            "bounds": reference.bounds,
            "grid_origin_e": reference.bounds[0],
            "grid_origin_n": reference.bounds[3],
            "pixel_size_m": reference.pixel_size_m,
            "planner_cell_size_m": float(planner_cell_size_m),
            "feature_detector": "ORB",
            "descriptor_bytes": 32,
            "max_features_per_tile": int(max_features_per_tile),
            "halo_px": int(halo_px),
            "feature_band": feature_band,
            "image_sha256": _sha256(image_path),
            "dem_sha256": _sha256(dem_path),
            "aoi_sha256": _sha256(aoi_path),
        })
        for tile in pending:
            tile_id = writer.write_tile(tile)
            quality_grid[tile["row"], tile["col"]] = tile["quality"]
            count_grid[tile["row"], tile["col"]] = tile["feature_count"]
            landmark_rows = []
            for feature in tile["features"]:
                pixel_row = int(np.clip(round(feature.row), 0, reference.dem.shape[0] - 1))
                pixel_col = int(np.clip(round(feature.col), 0, reference.dem.shape[1] - 1))
                east, north = _coordinate(reference.transform, feature.col, feature.row)
                landmark_rows.append((
                    east, north, float(reference.dem[pixel_row, pixel_col]), feature.col, feature.row,
                    feature.size, feature.angle, feature.response, feature.octave,
                    float(slope[pixel_row, pixel_col]), *[float(value) for value in normal[pixel_row, pixel_col]],
                    feature.descriptor,
                ))
            writer.write_landmarks(tile_id, landmark_rows)
    finally:
        writer.close()

    quality_path = output_path.with_suffix(".quality.npz")
    np.savez_compressed(
        quality_path, landmark_quality=quality_grid, landmark_count=count_grid,
        extent=np.asarray(reference.bounds, dtype=np.float64), resolution_m=float(planner_cell_size_m),
        crs_wkt=np.asarray(reference.crs_wkt),
    )
    return {
        "database": str(output_path), "quality": str(quality_path), "tiles": len(pending),
        "landmarks": int(sum(item["feature_count"] for item in pending)),
        "metadata": json.loads(json.dumps({"crs_wkt": reference.crs_wkt})),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build an onboard ORB landmark SQLite package")
    parser.add_argument("--image", required=True, help="projected metric reference GeoTIFF")
    parser.add_argument("--dem", required=True, help="DEM raster")
    parser.add_argument("--aoi", required=True, help="AOI polygon GeoJSON/vector file")
    parser.add_argument("--output", required=True, help="output SQLite database path")
    parser.add_argument("--planner-cell-size-m", required=True, type=float)
    parser.add_argument("--max-features-per-tile", type=int, default=300)
    parser.add_argument("--feature-band", type=int, help="1-based source band used for feature extraction")
    parser.add_argument("--halo-px", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = build_landmark_database(
        args.image, args.dem, args.aoi, args.output, args.planner_cell_size_m,
        args.max_features_per_tile, args.feature_band, args.halo_px, args.overwrite,
    )
    print(f"wrote {result['database']}: {result['tiles']} tiles, {result['landmarks']} landmarks")
    print(f"wrote {result['quality']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

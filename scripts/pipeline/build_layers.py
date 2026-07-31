"""Executable notebook-equivalent preprocessing pipeline.

Example:
    python -m scripts.pipeline.build_layers --r10m-dir ... --dem dem.tif \
        --aoi aoi.geojson --nfz no_fly_zone.geojson --output planning_layers.npz
"""

import argparse

import numpy as np

from .cell_vectors import build_cell_vectors, to_planner_layers
from .nav_quality import compute_localization_metric, terrain_grid_quality
from .raster import create_grid, get_path, read_dem_aoi, read_image_aoi
from .visibility import (aggregate_to_planner_grid, combine_sources,
                         compute_visibility, gaussian_density,
                         generate_smoke_sources)
from .wind import generate_synthetic_wind


def build_layers(r10m_dir, dem_path, aoi_path, output_path, nfz_path=None,
                 mission_points_path=None,
                 cell_size_m=1000, pixel_size_m=10, smoke_cell_size_m=100,
                 wind_seed=42, smoke_seed=42):
    """Build and save the planner layers from the notebook's input products."""
    import geopandas as gpd

    aoi = gpd.read_file(aoi_path).to_crs("EPSG:32635")
    nfz = gpd.read_file(nfz_path).to_crs("EPSG:32635") if nfz_path else None
    points = (gpd.read_file(mission_points_path).to_crs("EPSG:32635")
              if mission_points_path else None)
    paths = {band: get_path(r10m_dir, band) for band in ("B02", "B03", "B04", "B08")}
    if any(path is None for path in paths.values()):
        missing = [band for band, path in paths.items() if path is None]
        raise FileNotFoundError(f"Sentinel-2 bands not found: {', '.join(missing)}")

    b04, extent = read_image_aoi(paths["B04"], aoi)
    b08, _ = read_image_aoi(paths["B08"], aoi)
    b02, _ = read_image_aoi(paths["B02"], aoi)
    b03, _ = read_image_aoi(paths["B03"], aoi)
    dem, _ = read_dem_aoi(dem_path, paths["B04"], aoi)

    b4_grid = create_grid(b04[:, :, np.newaxis], cell_size_m, pixel_size_m)
    b8_grid = create_grid(b08[:, :, np.newaxis], cell_size_m, pixel_size_m)
    dem_grid = create_grid(dem[:, :, np.newaxis], cell_size_m, pixel_size_m)
    rows, cols = len(b4_grid), len(b4_grid[0])
    phi_vis, _, _ = compute_localization_metric(b4_grid, b8_grid)
    weights = {"slope": .25, "tri": .25, "tpi": .15, "std": .20, "roughness": .15}
    phi_ter = terrain_grid_quality(dem_grid, pixel_size_m, 7, weights)
    u_wind, v_wind = generate_synthetic_wind(rows, cols, base_speed=6.0,
                                               base_dir_deg=240, perturbation_scale=2.5,
                                               smoothness=1.5, seed=wind_seed)

    mission_width, mission_height = extent[1] - extent[0], extent[3] - extent[2]
    smoke_rows, smoke_cols = int(round(mission_height / smoke_cell_size_m)), int(round(mission_width / smoke_cell_size_m))
    x_smoke = np.linspace(extent[0], extent[1], smoke_cols)
    y_smoke = np.linspace(extent[2], extent[3], smoke_rows)
    x_grid, y_grid = np.meshgrid(x_smoke, y_smoke)
    rng = np.random.RandomState(smoke_seed)
    sources = generate_smoke_sources(rng.randint(3, 11), (extent[0], extent[1]),
                                     (extent[2], extent[3]), seed=smoke_seed)
    density_stack = np.stack([gaussian_density(x_grid, y_grid, src["x0"], src["y0"], src["sigma"])
                              for src in sources])
    visibility = compute_visibility(density_stack, sources)
    visibility_planner = aggregate_to_planner_grid(visibility, int(cell_size_m / smoke_cell_size_m))
    if visibility_planner.shape != (rows, cols):
        raise ValueError("AOI dimensions do not produce matching image and smoke planner grids")

    cells = build_cell_vectors(extent, rows, cols, dem_grid, phi_vis, phi_ter,
                               visibility_planner, u_wind, v_wind, nfz)
    layers = to_planner_layers(cells, rows, cols)
    if points is not None:
        def point_to_cell(point):
            x = int((point.x - extent[0]) * cols / (extent[1] - extent[0]))
            y = int((extent[3] - point.y) * rows / (extent[3] - extent[2]))
            return (int(np.clip(x, 0, cols - 1)), int(np.clip(y, 0, rows - 1)))
        layers["start"] = np.asarray(point_to_cell(points.loc[points.role == "start"].geometry.iloc[0]))
        layers["goal"] = np.asarray(point_to_cell(points.loc[points.role == "goal"].geometry.iloc[0]))
    np.savez_compressed(output_path, **layers, resolution_m=float(cell_size_m), extent=np.asarray(extent))
    return layers


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build MOA* raster layers from Sentinel-2, DEM, NFZ, wind, and smoke inputs")
    parser.add_argument("--r10m-dir", required=True, help="Sentinel-2 IMG_DATA/R10m directory")
    parser.add_argument("--dem", required=True, help="DEM raster path")
    parser.add_argument("--aoi", required=True, help="AOI GeoJSON path")
    parser.add_argument("--nfz", help="optional no-fly-zone GeoJSON path")
    parser.add_argument("--mission-points", help="optional start/goal GeoJSON with a role field")
    parser.add_argument("--output", required=True, help="output .npz file")
    parser.add_argument("--cell-size-m", type=float, default=1000)
    parser.add_argument("--pixel-size-m", type=float, default=10)
    parser.add_argument("--smoke-cell-size-m", type=float, default=100)
    args = parser.parse_args(argv)
    layers = build_layers(args.r10m_dir, args.dem, args.aoi, args.output, args.nfz,
                          args.mission_points, args.cell_size_m, args.pixel_size_m,
                          args.smoke_cell_size_m)
    print(f"wrote {args.output}: {layers['dem'].shape[0]}x{layers['dem'].shape[1]} planner grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

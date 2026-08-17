"""Executable notebook-equivalent preprocessing pipeline.

Example:
    python -m scripts.pipeline.build_layers --r10m-dir ... --dem dem.tif \
        --aoi aoi.geojson --nfz no_fly_zone.geojson --output planning_layers.npz

    python -m scripts.pipeline.build_layers --rgb-image orthophoto.tif --dem dem.tif \
        --aoi aoi.geojson --output planning_layers.npz --pixel-size-m 1
"""

import argparse

import numpy as np

from .cell_vectors import (build_cell_vectors, landing_site_coordinates,
                           select_landing_sites, to_planner_layers)
from .nav_quality import (compute_localization_metric,
                          compute_rgb_localization_metric, terrain_grid_quality)
from .raster import (create_grid, get_path, read_dem_aoi, read_dem_grid,
                     read_image_aoi, read_rgb_aoi)
from .visibility import (aggregate_to_planner_grid, combine_sources,
                         compute_visibility, gaussian_density,
                         generate_local_smoke_sources, visibility_from_zones)
from .wind import generate_synthetic_wind


def build_layers(r10m_dir, dem_path, aoi_path, output_path, nfz_path=None,
                 mission_points_path=None,
                 cell_size_m=1000, pixel_size_m=10, smoke_cell_size_m=100,
                 wind_seed=42, smoke_seed=42, rgb_image_path=None,
                 visibility_zones_path=None, zone_visibility=None,
                 synthetic_smoke_sources=None, synthetic_smoke_min_visibility=None,
                 max_landing_sites=10, battery_energy_wh=300.0,
                 energy_reserve_wh=40.0, cruise_power_w=450.0):
    """Build and save the planner layers from the notebook's input products."""
    import geopandas as gpd

    aoi = gpd.read_file(aoi_path).to_crs("EPSG:32635")
    nfz = gpd.read_file(nfz_path).to_crs("EPSG:32635") if nfz_path else None
    zones = _read_visibility_zones(visibility_zones_path) if visibility_zones_path else None
    _validate_visibility_options(zones, zone_visibility, synthetic_smoke_sources,
                                 synthetic_smoke_min_visibility)
    _validate_energy_options(battery_energy_wh, energy_reserve_wh, cruise_power_w)
    if mission_points_path is None:
        raise ValueError("mission_points_path is required to export a runnable planner map")
    points = gpd.read_file(mission_points_path).to_crs("EPSG:32635")
    if rgb_image_path is not None:
        rgb, extent, transform = read_rgb_aoi(rgb_image_path, aoi, pixel_size_m)
        rgb_grid = create_grid(rgb, cell_size_m, pixel_size_m)
        dem = read_dem_grid(dem_path, aoi.crs, transform, rgb.shape[0], rgb.shape[1])
        rows, cols = len(rgb_grid), len(rgb_grid[0])
        phi_vis, _, _ = compute_rgb_localization_metric(rgb_grid)
    else:
        if r10m_dir is None:
            raise ValueError("provide either r10m_dir or rgb_image_path")
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
        rows, cols = len(b4_grid), len(b4_grid[0])
        phi_vis, _, _ = compute_localization_metric(b4_grid, b8_grid)
    dem_grid = create_grid(dem[:, :, np.newaxis], cell_size_m, pixel_size_m)
    weights = {"slope": .25, "tri": .25, "tpi": .15, "std": .20, "roughness": .15}
    phi_ter = terrain_grid_quality(dem_grid, pixel_size_m, 7, weights)
    u_wind, v_wind = generate_synthetic_wind(rows, cols, base_speed=6.0,
                                               base_dir_deg=240, perturbation_scale=2.5,
                                               smoothness=1.5, seed=wind_seed)

    if zones is not None:
        visibility_planner = visibility_from_zones(extent, rows, cols, zones, zone_visibility)
        visibility_source = "zones"
    elif synthetic_smoke_sources is not None:
        mission_width, mission_height = extent[1] - extent[0], extent[3] - extent[2]
        smoke_rows, smoke_cols = (int(round(mission_height / smoke_cell_size_m)),
                                  int(round(mission_width / smoke_cell_size_m)))
        x_smoke = np.linspace(extent[0], extent[1], smoke_cols)
        y_smoke = np.linspace(extent[2], extent[3], smoke_rows)
        x_grid, y_grid = np.meshgrid(x_smoke, y_smoke)
        rng = np.random.RandomState(smoke_seed)
        sources = generate_local_smoke_sources((extent[0], extent[1]),
                                               (extent[2], extent[3]), rng=rng,
                                               n_sources=synthetic_smoke_sources)
        for source in sources:
            source["min_vis"] = synthetic_smoke_min_visibility
        density_stack = np.stack([gaussian_density(x_grid, y_grid, src["x0"], src["y0"], src["sigma"])
                                  for src in sources])
        visibility = compute_visibility(density_stack, sources)
        visibility_planner = aggregate_to_planner_grid(visibility, int(cell_size_m / smoke_cell_size_m))
        if visibility_planner.shape != (rows, cols):
            raise ValueError("AOI dimensions do not produce matching image and smoke planner grids")
        visibility_source = "synthetic_smoke"
    else:
        visibility_planner = np.ones((rows, cols), dtype=float)
        visibility_source = "clear"

    cells = build_cell_vectors(extent, rows, cols, dem_grid, phi_vis, phi_ter,
                               visibility_planner, u_wind, v_wind, nfz)
    layers = to_planner_layers(cells, rows, cols)
    # Preserve the two sources separately for thesis ablations.  The planner
    # continues to consume only nav_density = max(visual_richness, rugosity).
    layers["visual_richness"] = phi_vis.astype(float)
    layers["rugosity"] = phi_ter.astype(float)
    def point_to_cell(point):
        x = int((point.x - extent[0]) * cols / (extent[1] - extent[0]))
        y = int((extent[3] - point.y) * rows / (extent[3] - extent[2]))
        return (int(np.clip(x, 0, cols - 1)), int(np.clip(y, 0, rows - 1)))
    layers["start"] = np.asarray(point_to_cell(points.loc[points.role == "start"].geometry.iloc[0]))
    layers["goal"] = np.asarray(point_to_cell(points.loc[points.role == "goal"].geometry.iloc[0]))
    layers["landing_sites"] = select_landing_sites(
        layers["dem"], layers["visibility"], layers["occupancy"], cell_size_m,
        tuple(layers["start"]), tuple(layers["goal"]), max_sites=max_landing_sites
    )
    layers["landing_site_coordinates"] = landing_site_coordinates(
        layers["landing_sites"], extent
    )
    np.savez_compressed(output_path, **layers, resolution_m=float(cell_size_m), extent=np.asarray(extent),
                        visibility_source=np.asarray(visibility_source),
                        visibility_zone_value=np.asarray(np.nan if zone_visibility is None else zone_visibility),
                        landing_site_crs=np.asarray("EPSG:32635"),
                        max_landing_sites=np.asarray(max_landing_sites, dtype=int),
                        battery_energy_wh=np.asarray(battery_energy_wh, dtype=float),
                        energy_reserve_wh=np.asarray(energy_reserve_wh, dtype=float),
                        cruise_power_w=np.asarray(cruise_power_w, dtype=float))
    return layers


def _read_visibility_zones(path):
    """Load valid visibility polygons, assuming EPSG:32635 only when absent."""
    import geopandas as gpd

    zones = gpd.read_file(path)
    if zones.empty or zones.geometry.is_empty.any():
        raise ValueError("visibility zones must contain non-empty geometries")
    allowed = {"Polygon", "MultiPolygon"}
    invalid = ~zones.geometry.geom_type.isin(allowed)
    if invalid.any():
        raise ValueError("visibility zones must contain only Polygon or MultiPolygon geometries")
    if zones.crs is None:
        zones = zones.set_crs("EPSG:32635", allow_override=True)
    return zones.to_crs("EPSG:32635")


def _validate_visibility_options(zones, zone_visibility, synthetic_smoke_sources,
                                 synthetic_smoke_min_visibility):
    if zones is not None:
        if zone_visibility is None:
            raise ValueError("zone_visibility is required when visibility_zones_path is supplied")
        if synthetic_smoke_sources is not None or synthetic_smoke_min_visibility is not None:
            raise ValueError("visibility zones and synthetic smoke cannot be used together")
        if not 0.0 <= zone_visibility <= 1.0:
            raise ValueError("zone_visibility must be in [0, 1]")
        return
    if (synthetic_smoke_sources is None) != (synthetic_smoke_min_visibility is None):
        raise ValueError("synthetic_smoke_sources and synthetic_smoke_min_visibility must be supplied together")
    if synthetic_smoke_sources is not None:
        if synthetic_smoke_sources < 1:
            raise ValueError("synthetic_smoke_sources must be positive")
        if not 0.0 <= synthetic_smoke_min_visibility <= 1.0:
            raise ValueError("synthetic_smoke_min_visibility must be in [0, 1]")


def _validate_energy_options(battery_energy_wh, energy_reserve_wh, cruise_power_w):
    if battery_energy_wh <= 0:
        raise ValueError("battery_energy_wh must be positive")
    if energy_reserve_wh < 0:
        raise ValueError("energy_reserve_wh must be non-negative")
    if energy_reserve_wh >= battery_energy_wh:
        raise ValueError("energy_reserve_wh must be smaller than battery_energy_wh")
    if cruise_power_w <= 0:
        raise ValueError("cruise_power_w must be positive")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build MOA* raster layers from Sentinel-2, DEM, NFZ, wind, and smoke inputs")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--r10m-dir", help="Sentinel-2 IMG_DATA/R10m directory")
    source_group.add_argument("--rgb-image", help="three-band RGB orthophoto")
    parser.add_argument("--dem", required=True, help="DEM raster path")
    parser.add_argument("--aoi", required=True, help="AOI GeoJSON path")
    parser.add_argument("--nfz", help="optional no-fly-zone GeoJSON path")
    parser.add_argument("--visibility-zones", help="polygon GeoJSON defining low-visibility areas")
    parser.add_argument("--zone-visibility", type=float,
                        help="visibility [0, 1] applied to all visibility-zone polygons")
    parser.add_argument("--synthetic-smoke-sources", type=int,
                        help="number of AOI-scaled synthetic smoke sources")
    parser.add_argument("--synthetic-smoke-min-visibility", type=float,
                        help="visibility [0, 1] at each synthetic smoke-source centre")
    parser.add_argument("--mission-points", required=True,
                        help="start/goal GeoJSON with a role field")
    parser.add_argument("--output", required=True, help="output .npz file")
    parser.add_argument("--cell-size-m", type=float, default=1000)
    parser.add_argument("--pixel-size-m", type=float, default=10)
    parser.add_argument("--smoke-cell-size-m", type=float, default=100)
    parser.add_argument("--max-landing-sites", type=int, default=10,
                        help="maximum emergency landing sites selected along the mission corridor")
    parser.add_argument("--battery-energy-wh", type=float, default=300.0,
                        help="total usable battery model capacity in Wh (default: 300)")
    parser.add_argument("--energy-reserve-wh", type=float, default=40.0,
                        help="battery reserve excluded from mission use in Wh (default: 40)")
    parser.add_argument("--cruise-power-w", type=float, default=450.0,
                        help="constant cruise power used by the energy model (default: 450)")
    args = parser.parse_args(argv)
    layers = build_layers(args.r10m_dir, args.dem, args.aoi, args.output, args.nfz,
                          args.mission_points, args.cell_size_m, args.pixel_size_m,
                          args.smoke_cell_size_m, rgb_image_path=args.rgb_image,
                          visibility_zones_path=args.visibility_zones,
                          zone_visibility=args.zone_visibility,
                          synthetic_smoke_sources=args.synthetic_smoke_sources,
                          synthetic_smoke_min_visibility=args.synthetic_smoke_min_visibility,
                          max_landing_sites=args.max_landing_sites,
                          battery_energy_wh=args.battery_energy_wh,
                          energy_reserve_wh=args.energy_reserve_wh,
                          cruise_power_w=args.cruise_power_w)
    print(f"wrote {args.output}: {layers['dem'].shape[0]}x{layers['dem'].shape[1]} planner grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

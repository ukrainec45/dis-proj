"""Slice an RGB orthophoto and aligned DEM into the pipeline's planner cells.

The output is a compact, visualization-oriented NPZ.  Each cell is retained as
an individually addressable preview at ``rgb_cells[row, col]`` and
``dem_cells[row, col]``.  Cell indices therefore match MOA* coordinates
``(column, row)`` exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .raster import create_grid, read_dem_grid, read_rgb_aoi


FORMAT_VERSION = 1


def _rgb_uint8(rgb):
    """Apply global per-channel 2--98% display scaling."""
    rgb = np.asarray(rgb)
    output = np.empty(rgb.shape, dtype=np.uint8)
    for band in range(3):
        values = rgb[..., band].astype(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            output[..., band] = 0
            continue
        low, high = np.percentile(finite, (2, 98))
        if high <= low:
            output[..., band] = 0
        else:
            output[..., band] = np.clip(
                (values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return output


def _resize_cell(cell, size, is_rgb):
    if size is None or cell.shape[:2] == (size, size):
        return cell
    from scipy.ndimage import zoom
    factors = (size / cell.shape[0], size / cell.shape[1])
    if is_rgb:
        resized = zoom(cell, (*factors, 1), order=1)
        return np.clip(resized, 0, 255).astype(np.uint8)
    return zoom(cell.astype(np.float32), factors, order=1).astype(np.float32)


def stack_cell_previews(grid, preview_pixels_per_cell=None, is_rgb=False):
    """Convert the pipeline's nested cell grid into one indexed numeric array."""
    rows = []
    for grid_row in grid:
        rows.append([_resize_cell(cell, preview_pixels_per_cell, is_rgb)
                     for cell in grid_row])
    return np.asarray(rows)


def slice_inputs(rgb_image_path, dem_path, aoi_path, output_path,
                 pixel_size_m=0.5, cell_size_m=50.0,
                 preview_pixels_per_cell=32, planner_map_path=None):
    """Build aligned RGB/DEM cell stacks using the production pipeline logic."""
    import geopandas as gpd

    if pixel_size_m <= 0 or cell_size_m <= 0:
        raise ValueError("pixel_size_m and cell_size_m must be positive")
    if cell_size_m < pixel_size_m:
        raise ValueError("cell_size_m must be at least pixel_size_m")
    if preview_pixels_per_cell is not None and preview_pixels_per_cell < 1:
        raise ValueError("preview_pixels_per_cell must be positive or None")

    aoi = gpd.read_file(aoi_path).to_crs("EPSG:32635")
    rgb, extent, transform = read_rgb_aoi(rgb_image_path, aoi, pixel_size_m)
    dem = read_dem_grid(dem_path, aoi.crs, transform, rgb.shape[0], rgb.shape[1])
    rgb_grid = create_grid(_rgb_uint8(rgb), cell_size_m, pixel_size_m)
    dem_grid = create_grid(dem[..., np.newaxis], cell_size_m, pixel_size_m)
    rows, cols = len(rgb_grid), len(rgb_grid[0])

    if planner_map_path:
        with np.load(planner_map_path, allow_pickle=False) as planner_map:
            planner_shape = tuple(planner_map["dem"].shape)
            if planner_shape != (rows, cols):
                raise ValueError(
                    f"cell slices {(rows, cols)} do not match planner map {planner_shape}; "
                    "use the same --pixel-size-m and --cell-size-m as build_layers")

    rgb_cells = stack_cell_previews(rgb_grid, preview_pixels_per_cell, is_rgb=True)
    dem_cells = stack_cell_previews(
        [[np.squeeze(cell, axis=-1) for cell in row] for row in dem_grid],
        preview_pixels_per_cell, is_rgb=False)
    valid_dem = np.isfinite(dem_cells)
    dem_fill = float(np.nanmedian(dem_cells)) if valid_dem.any() else 0.0
    dem_cells = np.nan_to_num(dem_cells, nan=dem_fill)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        format_version=np.asarray(FORMAT_VERSION, dtype=np.int16),
        rgb_cells=rgb_cells,
        dem_cells=dem_cells.astype(np.float32),
        extent=np.asarray(extent, dtype=float),
        crs=np.asarray("EPSG:32635"),
        grid_shape=np.asarray((rows, cols), dtype=np.int32),
        pixel_size_m=np.asarray(pixel_size_m, dtype=float),
        cell_size_m=np.asarray(cell_size_m, dtype=float),
        source_pixels_per_cell=np.asarray(int(cell_size_m // pixel_size_m), dtype=np.int32),
        preview_pixels_per_cell=np.asarray(rgb_cells.shape[2], dtype=np.int32),
    )
    return rows, cols


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Slice orthophoto and DEM into cells matching build_layers")
    parser.add_argument("--rgb-image", required=True)
    parser.add_argument("--dem", required=True)
    parser.add_argument("--aoi", required=True)
    parser.add_argument("--output", required=True, help="output cell-slices NPZ")
    parser.add_argument("--pixel-size-m", type=float, default=0.5)
    parser.add_argument("--cell-size-m", type=float, default=50.0)
    parser.add_argument("--preview-pixels-per-cell", type=int, default=32,
                        help="stored preview width/height per cell; use 0 for full resolution")
    parser.add_argument("--planner-map",
                        help="optional map NPZ used to verify the planner grid shape")
    args = parser.parse_args(argv)
    preview_size = None if args.preview_pixels_per_cell == 0 else args.preview_pixels_per_cell
    rows, cols = slice_inputs(
        args.rgb_image, args.dem, args.aoi, args.output,
        pixel_size_m=args.pixel_size_m, cell_size_m=args.cell_size_m,
        preview_pixels_per_cell=preview_size, planner_map_path=args.planner_map)
    print(f"wrote {args.output}: {rows}x{cols} cells; cell index is (column, row)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

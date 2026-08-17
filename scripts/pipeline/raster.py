"""Raster loading, alignment, normalisation, and planner-grid partitioning."""

import os

import numpy as np


def get_path(folder, band_suffix, resolution="10m"):
    """Return the first Sentinel band path matching the notebook convention."""
    for filename in os.listdir(folder):
        if filename.endswith(f"_{band_suffix}_{resolution}.jp2"):
            return os.path.join(folder, filename)
    return None


def read_image_aoi(path, aoi):
    """Read one raster band over an AOI, preserving the notebook's bounds logic."""
    from rasterio.windows import bounds, from_bounds
    import rasterio

    with rasterio.open(path) as src:
        image_aoi = aoi.to_crs(src.crs) if aoi.crs != src.crs else aoi
        left, bottom, right, top = image_aoi.total_bounds
        window = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
        subset = src.read(1, window=window)
        win_left, win_bottom, win_right, win_top = bounds(window, src.transform)
    return subset, [win_left, win_right, win_bottom, win_top]


def read_rgb_aoi(path, aoi, pixel_size_m):
    """Read a three-band orthophoto into the AOI CRS at a metric resolution.

    Orthophotos are commonly supplied in a geographic or web-map CRS.  This
    function creates a north-up grid in the AOI's projected CRS so the returned
    pixels have the requested metre-based size and align with the DEM layer.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT

    if aoi.crs is None:
        raise ValueError("AOI must have a CRS before reading an RGB orthophoto")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    left, bottom, right, top = aoi.total_bounds
    width = int(np.ceil((right - left) / pixel_size_m))
    height = int(np.ceil((top - bottom) / pixel_size_m))
    if width <= 0 or height <= 0:
        raise ValueError("AOI must have a positive area")
    transform = from_origin(left, top, pixel_size_m, pixel_size_m)
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError("RGB orthophoto must contain at least three bands")
        with WarpedVRT(src, crs=aoi.crs, transform=transform, width=width,
                       height=height, resampling=Resampling.bilinear) as vrt:
            rgb = np.moveaxis(vrt.read((1, 2, 3)), 0, -1)
    return rgb, [left, right, bottom, top], transform


def normalize(array):
    """2nd--98th percentile contrast normalisation used by the notebook."""
    p2, p98 = np.percentile(array, (2, 98))
    if p98 == p2:
        return np.zeros_like(array, dtype=float)
    return np.clip((array - p2) / (p98 - p2), 0, 1)


def read_dem_aoi(dem_path, image_path, aoi):
    """Reproject DEM onto the reference image grid and crop it to the AOI."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import bounds, from_bounds, transform as window_transform

    left, bottom, right, top = aoi.total_bounds
    with rasterio.open(image_path) as ref:
        ref_transform = ref.transform
        window = from_bounds(left, bottom, right, top, transform=ref_transform).round_offsets().round_lengths()
        dst_crs, dst_transform = ref.crs, window_transform(window, ref_transform)
        dst_width, dst_height = int(window.width), int(window.height)
    aligned = read_dem_grid(dem_path, dst_crs, dst_transform, dst_height, dst_width)
    win_left, win_bottom, win_right, win_top = bounds(window, ref_transform)
    return aligned, [win_left, win_right, win_bottom, win_top]


def read_dem_grid(dem_path, dst_crs, dst_transform, dst_height, dst_width):
    """Reproject a DEM to a supplied destination grid."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(dem_path) as src:
        aligned = np.zeros((dst_height, dst_width), dtype=np.float32)
        reproject(source=src.read(1), destination=aligned,
                  src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_nodata=np.nan,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  resampling=Resampling.bilinear)
    return aligned


def create_grid(image, cell_size_m, pixel_size_m):
    """Split an image into complete, ordered planner-cell patches."""
    steps = int(cell_size_m // pixel_size_m)
    if steps <= 0:
        raise ValueError("cell_size_m must be at least pixel_size_m")
    height, width = image.shape[:2]
    cells_2d = []
    for y in range(0, height, steps):
        row_cells = []
        for x in range(0, width, steps):
            cell = image[y:y + steps, x:x + steps, ...]
            if cell.shape[0] == steps and cell.shape[1] == steps:
                row_cells.append(cell if np.sum(cell) > 0 else np.zeros_like(cell))
        if row_cells:
            cells_2d.append(row_cells)
    if not cells_2d:
        raise ValueError("image does not contain a complete planner cell")
    if any(len(row) != len(cells_2d[0]) for row in cells_2d):
        raise ValueError("incomplete grid rows; crop input to a rectangular AOI")
    return cells_2d

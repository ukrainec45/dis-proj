"""Finite-aware DEM sampling, edge profiles, and terrain descriptors."""

import numpy as np


def finite_bilinear(array, row, col):
    """Sample a 2-D array while ignoring nodata values in the stencil."""
    array = np.asarray(array, dtype=float)
    height, width = array.shape
    row = float(np.clip(row, 0.0, height - 1.0))
    col = float(np.clip(col, 0.0, width - 1.0))
    r0, c0 = int(np.floor(row)), int(np.floor(col))
    r1, c1 = min(r0 + 1, height - 1), min(c0 + 1, width - 1)
    dr, dc = row - r0, col - c0
    samples = ((array[r0, c0], (1.0 - dr) * (1.0 - dc)),
               (array[r0, c1], (1.0 - dr) * dc),
               (array[r1, c0], dr * (1.0 - dc)),
               (array[r1, c1], dr * dc))
    finite = [(float(value), weight) for value, weight in samples
              if np.isfinite(value) and weight > 0.0]
    if not finite:
        return np.nan
    weight_sum = sum(weight for _, weight in finite)
    return sum(value * weight for value, weight in finite) / weight_sum


def finite_bilinear_profile(array, rows, cols):
    """Vectorised finite-aware bilinear sampling for an edge profile."""
    array = np.asarray(array, dtype=float)
    rows = np.clip(np.asarray(rows, dtype=float), 0.0, array.shape[0] - 1.0)
    cols = np.clip(np.asarray(cols, dtype=float), 0.0, array.shape[1] - 1.0)
    r0, c0 = np.floor(rows).astype(int), np.floor(cols).astype(int)
    r1, c1 = np.minimum(r0 + 1, array.shape[0] - 1), np.minimum(c0 + 1, array.shape[1] - 1)
    dr, dc = rows - r0, cols - c0
    values = np.stack((array[r0, c0], array[r0, c1],
                       array[r1, c0], array[r1, c1]))
    weights = np.stack(((1.0 - dr) * (1.0 - dc), (1.0 - dr) * dc,
                        dr * (1.0 - dc), dr * dc))
    valid = np.isfinite(values) & (weights > 0.0)
    weighted = np.where(valid, values * weights, 0.0).sum(axis=0)
    weight_sum = np.where(valid, weights, 0.0).sum(axis=0)
    return np.divide(weighted, weight_sum, out=np.full_like(weighted, np.nan),
                     where=weight_sum > 0.0)


def as_dem_patch(patch):
    patch = np.asarray(patch, dtype=float)
    if patch.ndim == 3 and patch.shape[-1] == 1:
        patch = patch[..., 0]
    if patch.ndim != 2 or not patch.size:
        raise ValueError("each DEM cell must be a non-empty 2-D patch")
    return patch


def patch_centre_height(patch):
    patch = as_dem_patch(patch)
    return finite_bilinear(patch, (patch.shape[0] - 1) / 2.0,
                           (patch.shape[1] - 1) / 2.0)


def build_edge_terrain_excess(dem_grid):
    """Return detailed-DEM terrain excess for every directed 8-grid edge."""
    rows, cols = len(dem_grid), len(dem_grid[0])
    patches = [[as_dem_patch(cell) for cell in row] for row in dem_grid]
    patch_shape = patches[0][0].shape
    if any(cell.shape != patch_shape for row in patches for cell in row):
        raise ValueError("all DEM planner-cell patches must have the same 2-D shape")
    fine_dem = np.concatenate([np.concatenate(row, axis=1) for row in patches], axis=0)
    patch_rows, patch_cols = patch_shape
    centres = np.asarray([[patch_centre_height(cell) for cell in row]
                          for row in patches], dtype=float)
    excess = np.full((rows, cols, 3, 3), np.nan, dtype=float)
    excess[:, :, 1, 1] = 0.0
    for row in range(rows):
        for col in range(cols):
            start_height = centres[row, col]
            start_r = row * patch_rows + (patch_rows - 1) / 2.0
            start_c = col * patch_cols + (patch_cols - 1) / 2.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    end_row, end_col = row + dy, col + dx
                    if ((dx == 0 and dy == 0) or not (0 <= end_row < rows) or
                            not (0 <= end_col < cols)):
                        continue
                    end_height = centres[end_row, end_col]
                    end_r = end_row * patch_rows + (patch_rows - 1) / 2.0
                    end_c = end_col * patch_cols + (patch_cols - 1) / 2.0
                    count = int(np.ceil(np.hypot(end_r - start_r, end_c - start_c))) + 1
                    t = np.linspace(0.0, 1.0, count)
                    terrain = finite_bilinear_profile(
                        fine_dem, start_r + t * (end_r - start_r),
                        start_c + t * (end_c - start_c))
                    baseline = start_height + t * (end_height - start_height)
                    finite = (terrain - baseline)[np.isfinite(terrain - baseline)]
                    excess[row, col, dy + 1, dx + 1] = (
                        float(np.max(finite)) if finite.size else np.inf)
    return excess


def compute_slope(dem, px_res=10):
    dy, dx = np.gradient(dem, px_res)
    return np.sqrt(dx**2 + dy**2)


def compute_tri(dem, window_size=7):
    from scipy.ndimage import generic_filter
    return generic_filter(dem, lambda values: np.sqrt(np.mean(
        (values - values[len(values) // 2])**2)), size=window_size)


def compute_tpi(dem, window_size=7):
    from scipy.ndimage import uniform_filter
    return dem - uniform_filter(dem, size=window_size)


def compute_std(dem, window_size=7):
    from scipy.ndimage import uniform_filter
    first, second = uniform_filter(dem, window_size), uniform_filter(dem * dem, window_size)
    return np.sqrt(np.maximum(second - first * first, 0))


def compute_roughness(dem, window_size=7):
    from scipy.ndimage import generic_filter
    return generic_filter(dem, lambda values: np.max(values) - np.min(values),
                          size=window_size)


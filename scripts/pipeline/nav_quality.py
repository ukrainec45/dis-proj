"""Visual and terrain navigation-quality metrics extracted from the notebook."""

import numpy as np


def robust_normalize(array, p_low=2, p_high=98):
    p_min, p_max = np.percentile(array, (p_low, p_high))
    if p_min == p_max:
        return np.zeros_like(array)
    clipped = np.clip(array, p_min, p_max)
    return (clipped - p_min) / (p_max - p_min)


def compute_localization_metric(grid_b4, grid_b8, window=7, alpha=0.9, beta=0.6):
    """Compute notebook ``phi_vis`` from Shi--Tomasi and NDVI entropy signals."""
    from scipy.ndimage import sobel, uniform_filter
    from skimage.filters.rank import entropy as sk_entropy
    from skimage.morphology import disk

    rows, cols = len(grid_b4), len(grid_b4[0])
    geom_scores, tex_scores = np.zeros((rows, cols)), np.zeros((rows, cols))

    def shi_tomasi_lambda_min(band):
        ix, iy = sobel(band, axis=1), sobel(band, axis=0)
        ixx, iyy, ixy = (uniform_filter(ix * ix, size=window),
                          uniform_filter(iy * iy, size=window),
                          uniform_filter(ix * iy, size=window))
        trace, determinant = ixx + iyy, ixx * iyy - ixy**2
        return 0.5 * (trace - np.sqrt(np.maximum(trace**2 - 4 * determinant, 0)))

    for row in range(rows):
        for col in range(cols):
            b4 = np.squeeze(grid_b4[row][col]).astype(np.float32)
            b8 = np.squeeze(grid_b8[row][col]).astype(np.float32)
            geom_scores[row, col] = np.max(shi_tomasi_lambda_min(b4) + shi_tomasi_lambda_min(b8))
            denominator = b8 + b4
            ndvi = np.divide(b8 - b4, denominator, out=np.zeros_like(denominator), where=denominator != 0)
            tex_scores[row, col] = np.mean(sk_entropy(((ndvi + 1) * 127.5).clip(0, 255).astype(np.uint8), disk(5)))

    geom_norm, tex_norm = robust_normalize(geom_scores), robust_normalize(tex_scores)
    return 1.0 - (1.0 - alpha * geom_norm) * (1.0 - beta * tex_norm), geom_norm, tex_norm


def compute_slope(dem, px_res=10):
    dy, dx = np.gradient(dem, px_res)
    return np.sqrt(dx**2 + dy**2)


def compute_tri(dem, window_size=7):
    from scipy.ndimage import generic_filter
    return generic_filter(dem, lambda values: np.sqrt(np.mean((values - values[len(values) // 2])**2)), size=window_size)


def compute_tpi(dem, window_size=7):
    from scipy.ndimage import uniform_filter
    return dem - uniform_filter(dem, size=window_size)


def compute_std(dem, window_size=7):
    from scipy.ndimage import uniform_filter
    first, second = uniform_filter(dem, window_size), uniform_filter(dem * dem, window_size)
    return np.sqrt(np.maximum(second - first * first, 0))


def compute_roughness(dem, window_size=7):
    from scipy.ndimage import generic_filter
    return generic_filter(dem, lambda values: np.max(values) - np.min(values), size=window_size)


def terrain_quality(dem, px_res=10, window_size=7,
                    weights=None):
    """Return notebook terrain quality and its normalized component layers."""
    weights = weights or {"slope": .25, "tri": .25, "tpi": .15, "std": .20, "roughness": .15}
    raw = {"slope": compute_slope(dem, px_res), "tri": compute_tri(dem, window_size),
           "tpi": compute_tpi(dem, window_size), "std": compute_std(dem, window_size),
           "roughness": compute_roughness(dem, window_size)}
    normalized = {name: robust_normalize(values) for name, values in raw.items()}
    combined = sum(weights[name] * normalized[name] for name in weights)
    return robust_normalize(combined), raw, normalized


def calculate_cell_quality(dem_patch, px_res, window_size, weights):
    patch = np.squeeze(dem_patch)
    components = [compute_slope(patch, px_res), compute_tri(patch, window_size),
                  compute_tpi(patch, window_size), compute_std(patch, window_size),
                  compute_roughness(patch, window_size)]
    names = ("slope", "tri", "tpi", "std", "roughness")
    return np.mean(sum(weights[name] * robust_normalize(value)
                       for name, value in zip(names, components)))


def terrain_grid_quality(dem_grid, px_res, window_size, weights):
    out = np.zeros((len(dem_grid), len(dem_grid[0])))
    for row in range(len(dem_grid)):
        for col in range(len(dem_grid[0])):
            out[row, col] = calculate_cell_quality(dem_grid[row][col], px_res, window_size, weights)
    return robust_normalize(out)

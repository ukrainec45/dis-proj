"""Planner-cell attributes and notebook-compatible edge evaluation helpers."""

import numpy as np


def build_cell_vectors(extent, grid_rows, grid_cols, dem_grid, phi_vis, phi_ter,
                       visibility_planner, u_wind, v_wind, nfz_gdf=None):
    """Build the structured planner-cell vectors from the notebook."""
    from shapely.geometry import box

    xmin, xmax, ymin, ymax = extent
    dtype = np.dtype([( "r", np.int32), ("c", np.int32), ("x", np.float64), ("y", np.float64),
                      ("z_dem", np.float64), ("phi_vis", np.float64), ("phi_ter", np.float64),
                      ("phi_vsb", np.float64), ("u_wind", np.float64), ("v_wind", np.float64),
                      ("in_nfz", bool)])
    cells = np.empty(grid_rows * grid_cols, dtype=dtype)
    index = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            cell_width = (xmax - xmin) / grid_cols
            cell_height = (ymax - ymin) / grid_rows
            x_left, x_right = xmin + col * cell_width, xmin + (col + 1) * cell_width
            y_top, y_bottom = ymax - row * cell_height, ymax - (row + 1) * cell_height
            x, y = (x_left + x_right) / 2.0, (y_top + y_bottom) / 2.0
            # Conservative policy: a cell is blocked even when an NFZ merely
            # touches its boundary, preventing a raster path through an NFZ edge.
            in_nfz = bool(nfz_gdf.intersects(box(x_left, y_bottom, x_right, y_top)).any()) if nfz_gdf is not None else False
            cells[index] = (row, col, x, y, float(np.mean(np.squeeze(dem_grid[row][col]))),
                            float(phi_vis[row, col]), float(phi_ter[row, col]), float(visibility_planner[row, col]),
                            float(u_wind[row, col]), float(v_wind[row, col]), in_nfz)
            index += 1
    return cells


def derive_landing_sites(dem, visibility, occupancy, resolution_m,
                         min_visibility=0.7, max_slope=0.05):
    """Return all safe emergency-landing cells from terrain and visibility.

    A candidate must be outside an NFZ, have visibility at least
    ``min_visibility``, and have local terrain slope no greater than
    ``max_slope`` (metres of elevation change per horizontal metre).
    """
    if dem.shape != visibility.shape or dem.shape != occupancy.shape:
        raise ValueError("dem, visibility, and occupancy must share a shape")
    north_gradient, east_gradient = np.gradient(dem.astype(float), resolution_m)
    slope = np.hypot(east_gradient, north_gradient)
    return (~occupancy & (visibility >= min_visibility) & (slope <= max_slope))


def to_planner_layers(cells, rows, cols):
    """Convert structured cell vectors to the raster layers accepted by MOA*."""
    shaped = cells.reshape(rows, cols)
    return {"dem": shaped["z_dem"].astype(float),
            "nav_density": np.maximum(shaped["phi_vis"], shaped["phi_ter"]).astype(float),
            "visibility": shaped["phi_vsb"].astype(float),
            "wind_field": np.dstack((shaped["u_wind"], shaped["v_wind"])),
            "occupancy": shaped["in_nfz"].astype(bool)}

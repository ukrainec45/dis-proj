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


def select_landing_sites(dem, visibility, occupancy, resolution_m, start, goal,
                         max_sites=10, min_visibility=0.7, max_slope=0.05,
                         corridor_width_m=None):
    """Select safe, distributed emergency sites along the mission corridor.

    The straight start-to-goal line is divided into ``max_sites`` equal
    progress bands.  At most one safe candidate is selected per band, ranked
    deterministically by visibility, slope, distance to the corridor centre,
    and distance to the band centre.  This must happen before route planning,
    because the selected sites participate in the energy-reserve constraint.
    """
    if not isinstance(max_sites, (int, np.integer)) or max_sites < 1:
        raise ValueError("max_sites must be a positive integer")
    candidates = derive_landing_sites(dem, visibility, occupancy, resolution_m,
                                      min_visibility, max_slope)
    rows, cols = dem.shape
    for name, cell in (("start", start), ("goal", goal)):
        if len(cell) != 2 or not (0 <= cell[0] < cols and 0 <= cell[1] < rows):
            raise ValueError(f"{name} must be a valid planner cell")

    start_xy = np.asarray(start, dtype=float) * resolution_m
    goal_xy = np.asarray(goal, dtype=float) * resolution_m
    route = goal_xy - start_xy
    route_length = float(np.linalg.norm(route))
    if route_length <= 0:
        raise ValueError("start and goal must differ when selecting landing sites")
    direction = route / route_length
    if corridor_width_m is None:
        corridor_width_m = max(3.0 * resolution_m, 0.2 * route_length)
    if corridor_width_m <= 0:
        raise ValueError("corridor_width_m must be positive")

    north_gradient, east_gradient = np.gradient(dem.astype(float), resolution_m)
    slope = np.hypot(east_gradient, north_gradient)
    candidate_rows, candidate_cols = np.where(candidates)
    xy = np.column_stack((candidate_cols, candidate_rows)).astype(float) * resolution_m
    relative = xy - start_xy
    along_m = relative @ direction
    perpendicular_m = np.abs(relative[:, 0] * direction[1] - relative[:, 1] * direction[0])
    in_corridor = ((along_m >= 0.0) & (along_m <= route_length) &
                   (perpendicular_m <= corridor_width_m))

    selected = np.zeros_like(candidates)
    if not in_corridor.any():
        return selected
    indices = np.flatnonzero(in_corridor)
    progress = np.clip(along_m[indices] / route_length, 0.0, 1.0)
    bands = np.minimum((progress * max_sites).astype(int), max_sites - 1)
    for band in range(max_sites):
        members = indices[bands == band]
        if not len(members):
            continue
        band_centre = (band + 0.5) / max_sites
        # np.lexsort uses the final key as primary: prefer high visibility,
        # then flat terrain, proximity to the corridor, and even progression.
        order = np.lexsort((np.abs(along_m[members] / route_length - band_centre),
                            perpendicular_m[members],
                            slope[candidate_rows[members], candidate_cols[members]],
                            -visibility[candidate_rows[members], candidate_cols[members]]))
        chosen = members[int(order[0])]
        selected[candidate_rows[chosen], candidate_cols[chosen]] = True
    return selected


def landing_site_coordinates(landing_sites, extent):
    """Return selected planner-cell centres as projected ``(x, y)`` points."""
    rows, cols = landing_sites.shape
    xmin, xmax, ymin, ymax = extent
    cell_width, cell_height = (xmax - xmin) / cols, (ymax - ymin) / rows
    coordinates = []
    for row, col in np.argwhere(landing_sites):
        coordinates.append((xmin + (col + 0.5) * cell_width,
                            ymax - (row + 0.5) * cell_height))
    return np.asarray(coordinates, dtype=float).reshape(-1, 2)


def to_planner_layers(cells, rows, cols):
    """Convert structured cell vectors to the raster layers accepted by MOA*."""
    shaped = cells.reshape(rows, cols)
    return {"dem": shaped["z_dem"].astype(float),
            "nav_density": np.maximum(shaped["phi_vis"], shaped["phi_ter"]).astype(float),
            "visibility": shaped["phi_vsb"].astype(float),
            "wind_field": np.dstack((shaped["u_wind"], shaped["v_wind"])),
            "occupancy": shaped["in_nfz"].astype(bool)}

"""Planner-cell attributes and notebook-compatible edge evaluation helpers."""

import numpy as np


def _finite_bilinear(array, row, col):
    """Sample a 2-D array while ignoring nodata values in the stencil."""
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


def _finite_bilinear_profile(array, rows, cols):
    """Vectorised finite-aware bilinear sampling for an edge profile."""
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


def _as_dem_patch(patch):
    patch = np.asarray(patch, dtype=float)
    if patch.ndim == 3 and patch.shape[-1] == 1:
        patch = patch[..., 0]
    if patch.ndim != 2 or not patch.size:
        raise ValueError("each DEM cell must be a non-empty 2-D patch")
    return patch


def _patch_centre_height(patch):
    patch = _as_dem_patch(patch)
    return _finite_bilinear(patch, (patch.shape[0] - 1) / 2.0,
                            (patch.shape[1] - 1) / 2.0)


def build_edge_terrain_excess(dem_grid):
    """Return detailed-DEM terrain excess for every directed 8-grid edge.

    ``out[row, col, dy + 1, dx + 1]`` is the maximum amount by which the
    detailed DEM profile rises above the linear ground-elevation profile
    joining the two planner-cell centres.  Therefore the minimum AGL clearance
    on an edge is ``cruise_altitude_agl_m - terrain_excess``.
    """
    rows, cols = len(dem_grid), len(dem_grid[0])
    patches = [[_as_dem_patch(cell) for cell in row]
               for row in dem_grid]
    patch_shape = patches[0][0].shape
    if len(patch_shape) != 2 or any(cell.shape != patch_shape for row in patches for cell in row):
        raise ValueError("all DEM planner-cell patches must have the same 2-D shape")
    fine_dem = np.concatenate([np.concatenate(row, axis=1) for row in patches], axis=0)
    patch_rows, patch_cols = patch_shape
    centres = np.asarray([[_patch_centre_height(cell) for cell in row]
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
                    count = int(np.ceil(np.hypot(end_r - start_r,
                                                 end_c - start_c))) + 1
                    t = np.linspace(0.0, 1.0, count)
                    terrain = _finite_bilinear_profile(
                        fine_dem, start_r + t * (end_r - start_r),
                        start_c + t * (end_c - start_c))
                    baseline = start_height + t * (end_height - start_height)
                    profile_excess = terrain - baseline
                    finite = profile_excess[np.isfinite(profile_excess)]
                    excess[row, col, dy + 1, dx + 1] = (
                        float(np.max(finite)) if finite.size else np.inf)
    return excess


def build_cell_vectors(extent, grid_rows, grid_cols, dem_grid, phi_vis, phi_ter,
                       visibility_planner, u_wind, v_wind, nfz_gdf=None):
    """Build the structured planner-cell vectors from the notebook."""
    from shapely.geometry import box

    xmin, xmax, ymin, ymax = extent
    dtype = np.dtype([( "r", np.int32), ("c", np.int32), ("x", np.float64), ("y", np.float64),
                      ("z_dem_center", np.float64), ("z_dem_max", np.float64),
                      ("phi_vis", np.float64), ("phi_ter", np.float64),
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
            dem_patch = _as_dem_patch(dem_grid[row][col])
            finite_dem = dem_patch[np.isfinite(dem_patch)]
            cells[index] = (row, col, x, y, _patch_centre_height(dem_patch),
                            float(np.max(finite_dem)) if finite_dem.size else np.nan,
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
    centre_field = "z_dem_center" if "z_dem_center" in shaped.dtype.names else "z_dem"
    maximum = (shaped["z_dem_max"].astype(float)
               if "z_dem_max" in shaped.dtype.names else shaped[centre_field].astype(float))
    return {"dem": shaped[centre_field].astype(float),
            "dem_max": maximum,
            "nav_density": np.maximum(shaped["phi_vis"], shaped["phi_ter"]).astype(float),
            "visibility": shaped["phi_vsb"].astype(float),
            "wind_field": np.dstack((shaped["u_wind"], shaped["v_wind"])),
            "occupancy": shaped["in_nfz"].astype(bool)}

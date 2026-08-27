"""Planner-grid geometry and continuous delivery-zone helpers."""

from __future__ import annotations

import numpy as np


NEIGHBOR_OFFSETS = tuple(
    (dx, dy)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dx, dy) != (0, 0)
)


def create_grid(image, cell_size_m, pixel_size_m):
    """Split an image into complete, ordered square planner-cell patches."""
    ratio = float(cell_size_m) / float(pixel_size_m)
    steps = int(round(ratio))
    if steps <= 0 or not np.isclose(ratio, steps):
        raise ValueError("cell_size_m must be an integer multiple of pixel_size_m")
    height, width = image.shape[:2]
    cells_2d = []
    for y in range(0, height, steps):
        row_cells = []
        for x in range(0, width, steps):
            cell = image[y:y + steps, x:x + steps, ...]
            if cell.shape[0] == steps and cell.shape[1] == steps:
                row_cells.append(cell if np.any(cell) else np.zeros_like(cell))
        if row_cells:
            cells_2d.append(row_cells)
    if not cells_2d:
        raise ValueError("image does not contain a complete planner cell")
    if any(len(row) != len(cells_2d[0]) for row in cells_2d):
        raise ValueError("incomplete grid rows; crop input to a rectangular AOI")
    return cells_2d


def cell_center_xy(row, col, extent, rows, cols):
    """Return a planner-cell centre in projected map coordinates."""
    xmin, xmax, ymin, ymax = map(float, extent)
    return np.array(
        [xmin + (col + 0.5) * (xmax - xmin) / cols,
         ymax - (row + 0.5) * (ymax - ymin) / rows],
        dtype=float,
    )


def point_to_cell(x, y, extent, rows, cols):
    """Map projected coordinates to a clipped ``(column, row)`` cell index."""
    xmin, xmax, ymin, ymax = map(float, extent)
    col = int(np.floor((float(x) - xmin) * cols / (xmax - xmin)))
    row = int(np.floor((ymax - float(y)) * rows / (ymax - ymin)))
    return int(np.clip(col, 0, cols - 1)), int(np.clip(row, 0, rows - 1))


def delivery_zone_mask(extent, rows, cols, center_xy, radius_m):
    """Return cells whose centres lie in a continuous circular delivery zone."""
    if radius_m <= 0:
        raise ValueError("delivery radius must be positive")
    center = np.asarray(center_xy, dtype=float)
    mask = np.zeros((rows, cols), dtype=bool)
    for row in range(rows):
        for col in range(cols):
            mask[row, col] = np.linalg.norm(
                cell_center_xy(row, col, extent, rows, cols) - center
            ) <= radius_m
    return mask


def segment_circle_entry_fraction(start_xy, end_xy, center_xy, radius_m):
    """Return the first segment fraction entering a circle, or ``nan``.

    A segment that begins inside the circle has entry fraction zero. Tangential
    contact counts as entry. The calculation is continuous and therefore does
    not depend on planner-cell resolution.
    """
    if radius_m <= 0:
        raise ValueError("delivery radius must be positive")
    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    center = np.asarray(center_xy, dtype=float)
    offset = start - center
    direction = end - start
    c = float(offset @ offset - radius_m**2)
    if c <= 0.0:
        return 0.0
    a = float(direction @ direction)
    if a <= 0.0:
        return np.nan
    b = 2.0 * float(offset @ direction)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return np.nan
    root = np.sqrt(max(discriminant, 0.0))
    candidates = ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
    valid = [value for value in candidates if -1e-12 <= value <= 1.0 + 1e-12]
    return float(np.clip(min(valid), 0.0, 1.0)) if valid else np.nan


def occupancy_from_zones(extent, rows, cols, zones):
    """Rasterise polygons conservatively: touching a cell blocks that cell."""
    occupancy = np.zeros((rows, cols), dtype=bool)
    if zones is None:
        return occupancy
    from shapely.geometry import box

    xmin, xmax, ymin, ymax = map(float, extent)
    cell_width = (xmax - xmin) / cols
    cell_height = (ymax - ymin) / rows
    for row in range(rows):
        y_top = ymax - row * cell_height
        y_bottom = y_top - cell_height
        for col in range(cols):
            x_left = xmin + col * cell_width
            x_right = x_left + cell_width
            occupancy[row, col] = bool(
                zones.intersects(box(x_left, y_bottom, x_right, y_top)).any()
            )
    return occupancy


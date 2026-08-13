import numpy as np

from scripts.moa.route_results import save_route_results
from scripts.moa.synthetic import demo_map
from scripts.moa.moa_star import EmoaStarLateBS
from scripts.moa.visualize_route_context import plot_route_context

from .slice_inputs import stack_cell_previews


def test_cell_previews_keep_row_column_indexing():
    grid = [[np.full((4, 4, 3), row * 10 + col, dtype=np.uint8)
             for col in range(3)] for row in range(2)]
    cells = stack_cell_previews(grid, preview_pixels_per_cell=2, is_rgb=True)
    assert cells.shape == (2, 3, 2, 2, 3)
    assert np.all(cells[1, 2] == 12)


def test_route_context_uses_saved_artifacts(tmp_path):
    cost_map = demo_map()
    solutions = EmoaStarLateBS(cost_map).solve()
    map_path = tmp_path / "map.npz"
    np.savez_compressed(
        map_path, dem=cost_map.dem, nav_density=cost_map.nav_density,
        visibility=cost_map.visibility, wind_field=cost_map.wind_field,
        occupancy=cost_map.occupancy, resolution_m=cost_map.resolution_m,
        start=cost_map.start, goal=cost_map.goal)
    result_path = tmp_path / "results.npz"
    save_route_results(result_path, cost_map, solutions)
    rows, cols = cost_map.shape
    cells_path = tmp_path / "cells.npz"
    np.savez_compressed(
        cells_path,
        rgb_cells=np.zeros((rows, cols, 2, 2, 3), dtype=np.uint8),
        dem_cells=np.broadcast_to(cost_map.dem[:, :, None, None],
                                  (rows, cols, 2, 2)).copy(),
        cell_size_m=np.asarray(cost_map.resolution_m))
    output = tmp_path / "context.png"

    written, selected = plot_route_context(
        map_path, result_path, cells_path, output, label_route_cells=True)

    assert written == output
    assert selected >= 0
    assert output.is_file() and output.stat().st_size > 0

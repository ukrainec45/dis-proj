"""Render saved planner routes beside their aligned orthophoto and DEM cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .route_results import load_route_results, route_path
from .run_planner import load_npz


def _mosaic(cells):
    """Join an array shaped [rows, cols, cell_y, cell_x, ...] for display."""
    rows, cols, cell_h, cell_w = cells.shape[:4]
    axes = (0, 2, 1, 3) + tuple(range(4, cells.ndim))
    transposed = cells.transpose(axes)
    return transposed.reshape(rows * cell_h, cols * cell_w, *cells.shape[4:])


def _overlay_route(axis, path, cell_pixels, label_cells=False):
    xy = (path.astype(float) + 0.5) * cell_pixels
    axis.plot(xy[:, 0], xy[:, 1], color="black", linewidth=5, alpha=0.35, zorder=3)
    axis.plot(xy[:, 0], xy[:, 1], color="yellow", linewidth=2.2, zorder=4)
    axis.scatter(xy[:, 0], xy[:, 1], s=13, color="yellow", edgecolor="black",
                 linewidth=0.35, zorder=5)
    axis.scatter([xy[0, 0]], [xy[0, 1]], marker="o", s=80, color="white",
                 edgecolor="black", zorder=6)
    axis.scatter([xy[-1, 0]], [xy[-1, 1]], marker="*", s=160, color="red",
                 edgecolor="black", zorder=6)
    if label_cells:
        for sequence, ((col, row), (x, y)) in enumerate(zip(path, xy)):
            axis.annotate(f"{sequence}:({col},{row})", (x, y), xytext=(3, 3),
                          textcoords="offset points", fontsize=5.5, color="black",
                          bbox=dict(facecolor="white", alpha=0.65, pad=0.5,
                                    edgecolor="none"), zorder=7)


def plot_route_context(map_path, results_path, cells_path, output_path,
                       route_id=None, label_route_cells=False, dpi=150):
    """Create planner/RGB/DEM panels strictly from previously saved artifacts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cost_map = load_npz(map_path)
    results = load_route_results(results_path)
    with np.load(cells_path, allow_pickle=False) as data:
        rgb_cells = data["rgb_cells"]
        dem_cells = data["dem_cells"]
        cell_size_m = float(data["cell_size_m"])
    if rgb_cells.shape[:2] != cost_map.shape or dem_cells.shape[:2] != cost_map.shape:
        raise ValueError("cell slices and planner map have different grid shapes")
    if not np.isclose(cell_size_m, cost_map.resolution_m):
        raise ValueError("cell slices and planner map have different cell sizes")

    selected = int(results["topsis_best"]) if route_id is None else int(route_id)
    if not 0 <= selected < len(results["path_lengths"]):
        raise ValueError(f"route_id must be between 0 and {len(results['path_lengths']) - 1}")
    path = route_path(results, selected)
    rgb = _mosaic(rgb_cells)
    dem = _mosaic(dem_cells)
    preview_pixels = rgb_cells.shape[2]
    rows, cols = cost_map.shape

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    axis = axes[0]
    nav = np.ma.masked_array(cost_map.nav_density, mask=cost_map.occupancy)
    image = axis.imshow(nav, cmap="viridis", vmin=0, vmax=1,
                        extent=(-0.5, cols - 0.5, rows - 0.5, -0.5))
    for row, col in np.argwhere(cost_map.visibility < 0.6):
        axis.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1,
                                 facecolor=(0.35, 0.6, 1, 0.25),
                                 edgecolor="tab:blue", hatch="//", linewidth=0.5))
    axis.plot(path[:, 0], path[:, 1], color="yellow", linewidth=2.5,
              marker="o", markersize=3, markeredgecolor="black")
    axis.plot(path[0, 0], path[0, 1], "o", color="white", mec="black", ms=8)
    axis.plot(path[-1, 0], path[-1, 1], "*", color="red", mec="black", ms=14)
    axis.set(title=f"Planner result: P{selected}\nbackground = navigation density",
             xlabel="column", ylabel="row")
    axis.set_xticks(np.arange(cols)); axis.set_yticks(np.arange(rows))
    axis.grid(True, color="white", alpha=0.25, linewidth=0.35)
    fig.colorbar(image, ax=axis, fraction=0.046, label="navigation density")

    axes[1].imshow(rgb)
    _overlay_route(axes[1], path, preview_pixels, label_route_cells)
    axes[1].set_title("Orthophoto cells\nyellow = cells used by route")

    dem_image = axes[2].imshow(dem, cmap="terrain")
    _overlay_route(axes[2], path, preview_pixels, label_route_cells)
    axes[2].set_title("Aligned DEM cells\nyellow = cells used by route")
    fig.colorbar(dem_image, ax=axes[2], fraction=0.046, label="elevation MSL (m)")

    for axis in axes[1:]:
        axis.set_xlim(0, cols * preview_pixels)
        axis.set_ylim(rows * preview_pixels, 0)
        axis.set_xticks(np.arange(cols + 1) * preview_pixels, minor=True)
        axis.set_yticks(np.arange(rows + 1) * preview_pixels, minor=True)
        axis.grid(which="minor", color="white", alpha=0.25, linewidth=0.35)
        axis.set_xticks((np.arange(cols) + 0.5) * preview_pixels)
        axis.set_yticks((np.arange(rows) + 0.5) * preview_pixels)
        # Label only moderate grids; route labels remain available for large ones.
        if max(rows, cols) <= 40:
            axis.set_xticklabels(np.arange(cols), fontsize=6)
            axis.set_yticklabels(np.arange(rows), fontsize=6)
        else:
            axis.set_xticklabels([]); axis.set_yticklabels([])
        axis.set_xlabel("planner cells (columns)")
        axis.set_ylabel("planner cells (rows)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path, selected


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Show a saved route beside aligned orthophoto and DEM cells")
    parser.add_argument("--map", required=True, help="planning-layer NPZ")
    parser.add_argument("--results", required=True, help="saved planner-results NPZ")
    parser.add_argument("--cells", required=True, help="NPZ produced by slice_inputs")
    parser.add_argument("--output", required=True, help="output PNG")
    parser.add_argument("--route-id", type=int,
                        help="route P index; default is the TOPSIS-selected route")
    parser.add_argument("--label-route-cells", action="store_true",
                        help="annotate each used cell as sequence:(column,row)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    output, selected = plot_route_context(
        args.map, args.results, args.cells, args.output,
        route_id=args.route_id, label_route_cells=args.label_route_cells,
        dpi=args.dpi)
    print(f"wrote {output}; displayed route P{selected}; no planning was run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

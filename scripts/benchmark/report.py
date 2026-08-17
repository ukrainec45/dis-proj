"""Persist benchmark tables, routes, and visual comparison figures."""

import csv
import json
from pathlib import Path

import numpy as np

from scripts.moa.topsis import topsis


TOPSIS_PROFILES = {
    "emergency": (0.7, 0.2, 0.1),
    "balanced": (0.5, 0.3, 0.2),
    "localization_visibility": (0.2, 0.5, 0.3),
}


def selected_paths(solutions):
    if not solutions:
        return {}
    costs = np.asarray([cost for _, cost in solutions], dtype=float)
    out = {}
    for name, weights in TOPSIS_PROFILES.items():
        order, closeness = topsis(costs, np.asarray(weights))
        index = int(order[0])
        out[name] = {"index": index, "closeness": float(closeness[index]),
                     "weights": list(weights), "path": [list(cell) for cell in solutions[index][0]],
                     "cost": list(map(float, solutions[index][1]))}
    return out


def result_json(result):
    return {
        "method": result.method,
        "feasible": result.feasible,
        "runtime_ms": result.runtime_ms,
        "n_expanded": result.n_expanded,
        "n_generated": result.n_generated,
        "peak_memory_kib": result.peak_memory_kib,
        "details": result.details,
        "solutions": [
            {"path": [list(cell) for cell in path], "cost": list(map(float, cost))}
            for path, cost in result.solutions
        ],
        "topsis_selections": selected_paths(result.solutions),
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _draw_map(ax, cost_map, result):
    ax.imshow(cost_map.nav_density, cmap="viridis", origin="upper", vmin=0, vmax=1)
    blocked = np.ma.masked_where(~cost_map.occupancy, cost_map.occupancy)
    ax.imshow(blocked, cmap="Reds", origin="upper", alpha=0.7)
    colors = ("#ffffff", "#ff7f0e", "#d62728", "#17becf", "#bcbd22")
    for index, (path, _) in enumerate(result.solutions):
        xy = np.asarray(path)
        ax.plot(xy[:, 0], xy[:, 1], color=colors[index % len(colors)], alpha=0.75, linewidth=1.5)
    ax.scatter(*cost_map.start, c="lime", s=36, edgecolors="black", label="start", zorder=3)
    ax.scatter(*cost_map.goal, c="magenta", s=36, edgecolors="black", label="goal", zorder=3)
    ax.set_title(f"{result.method}\n{len(result.solutions)} route(s), {result.runtime_ms:.1f} ms")
    ax.set_aspect("equal")


def _visibility_scale(results):
    """Return one non-negative f3 colour scale for every method in a case."""
    values = [cost[2] for result in results for _, cost in result.solutions]
    # A constant zero field otherwise makes Matplotlib expand the scale around
    # zero and display misleading negative tick labels.
    return 0.0, max(max(values, default=0.0), 1.0)


def write_scenario_figure(path, case, results):
    """Three route overlays plus a common f1/f2 objective view coloured by f3."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, result in zip(axes.flat[:3], results):
        _draw_map(ax, case.cost_map, result)
    ax = axes.flat[3]
    markers = ("o", "s", "^")
    vmin, vmax = _visibility_scale(results)
    norm = Normalize(vmin=vmin, vmax=vmax)
    for marker, result in zip(markers, results):
        if not result.solutions:
            continue
        costs = np.asarray([cost for _, cost in result.solutions])
        ax.scatter(costs[:, 0], costs[:, 1], c=costs[:, 2], marker=marker,
                   s=55, label=result.method, cmap="magma", norm=norm)
    ax.set_xlabel("f1 flight time [s]")
    ax.set_ylabel("f2 navigation deficit [s]")
    ax.set_title("Objective-space comparison (colour: f3 visibility deficit)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)
    else:
        ax.text(0.5, 0.5, "No feasible route", transform=ax.transAxes,
                ha="center", va="center")
    figure.colorbar(ScalarMappable(norm=norm, cmap="magma"), ax=ax, label="f3 [s]")
    figure.suptitle(f"Benchmark scenario: {case.name}")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_summary_figure(path, rows):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    feasible = [row for row in rows if row["feasible"]]
    methods = sorted({row["method"] for row in rows})
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for method in methods:
        subset = [row for row in feasible if row["method"] == method]
        axes[0].plot([row["scenario"] for row in subset], [row["runtime_ms"] for row in subset],
                     marker="o", label=method)
        axes[1].plot([row["scenario"] for row in subset], [row["normalized_hypervolume"] for row in subset],
                     marker="o", label=method)
        axes[2].plot([row["scenario"] for row in subset], [row["union_front_recall"] for row in subset],
                     marker="o", label=method)
    axes[0].set_title("Planning time")
    axes[0].set_ylabel("ms")
    axes[1].set_title("Normalized Pareto hypervolume")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_title("Union-front recall")
    axes[2].set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)

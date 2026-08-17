"""Visualize the characteristics CSV produced by a completed planner run.

This module reads ``route_characteristics.csv`` and never invokes EMOA* or
recalculates route characteristics.

Example::

    python -m scripts.moa.visualize_results \
        --csv plots/pulyny/route_characteristics.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REQUIRED_COLUMNS = {
    "route_id", "topsis_selected", "topsis_rank", "topsis_closeness",
    "horizontal_length_m", "flight_time_s", "average_ground_speed_mps",
    "navigation_deficit_s", "average_navigation_quality",
    "visibility_deficit_s", "average_visibility", "energy_used_wh",
    "minimum_altitude_msl_m", "maximum_altitude_msl_m",
}


def load_characteristics_csv(path):
    """Load already-calculated characteristics into plotting arrays."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError("characteristics CSV is missing columns: " +
                             ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError("characteristics CSV contains no routes")
    rows.sort(key=lambda row: int(row["route_id"]))
    route_ids = np.asarray([int(row["route_id"]) for row in rows], dtype=int)
    if not np.array_equal(route_ids, np.arange(len(rows))):
        raise ValueError("route_id values must be consecutive and start at zero")

    def values(column):
        return np.asarray([float(row[column]) for row in rows], dtype=float)

    selected = np.asarray(
        [row["topsis_selected"].strip().lower() in {"true", "1", "yes"}
         for row in rows], dtype=bool)
    if selected.sum() != 1:
        raise ValueError("characteristics CSV must contain exactly one TOPSIS-selected route")
    ranks = np.asarray([int(row["topsis_rank"]) for row in rows], dtype=int)
    if sorted(ranks.tolist()) != list(range(1, len(rows) + 1)):
        raise ValueError("topsis_rank values must form a complete 1..N ranking")

    flight_time = values("flight_time_s")
    nav_deficit = values("navigation_deficit_s")
    visibility_deficit = values("visibility_deficit_s")
    return {
        "costs": np.column_stack((flight_time, nav_deficit, visibility_deficit)),
        "topsis_order": np.argsort(ranks, kind="stable"),
        "topsis_closeness": values("topsis_closeness"),
        "topsis_best": np.asarray(int(np.flatnonzero(selected)[0])),
        "horizontal_length_m": values("horizontal_length_m"),
        "flight_time_s": flight_time,
        "average_ground_speed_mps": values("average_ground_speed_mps"),
        "average_navigation_quality": values("average_navigation_quality"),
        "average_visibility": values("average_visibility"),
        "energy_used_wh": values("energy_used_wh"),
        "minimum_altitude_msl_m": values("minimum_altitude_msl_m"),
        "maximum_altitude_msl_m": values("maximum_altitude_msl_m"),
    }


def _setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _mark_selected(axis, x, y, route_id, label=True):
    axis.scatter([x], [y], marker="*", s=250, color="black", edgecolor="white",
                 linewidth=0.9, zorder=5, label="TOPSIS selected" if label else None)
    axis.annotate(f"P{route_id}", (x, y), xytext=(5, 6),
                  textcoords="offset points", fontsize=8, fontweight="bold")


def plot_characteristics(results, output_path, top_routes=15, dpi=150):
    """Create a six-panel characteristics dashboard from loaded results."""
    plt = _setup_plotting()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    costs = np.asarray(results["costs"], dtype=float)
    route_ids = np.arange(len(costs))
    best = int(results["topsis_best"])
    closeness = np.asarray(results["topsis_closeness"], dtype=float)
    order = np.asarray(results["topsis_order"], dtype=int)
    lengths_km = np.asarray(results["horizontal_length_m"], dtype=float) / 1000.0
    times_min = np.asarray(results["flight_time_s"], dtype=float) / 60.0
    nav_quality = np.asarray(results["average_navigation_quality"], dtype=float)
    visibility = np.asarray(results["average_visibility"], dtype=float)
    energy = np.asarray(results["energy_used_wh"], dtype=float)
    min_altitude = np.asarray(results["minimum_altitude_msl_m"], dtype=float)
    max_altitude = np.asarray(results["maximum_altitude_msl_m"], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(
        f"Pareto-route characteristics ({len(costs)} routes; selected P{best})",
        fontsize=14, fontweight="bold")
    point_style = dict(c=closeness, cmap="viridis", vmin=0, vmax=1,
                       s=38, edgecolors="0.25", linewidths=0.35, alpha=0.85)

    # Objective-space trade-off. Marker area encodes f3 while colour encodes TOPSIS.
    axis = axes[0, 0]
    f3_span = np.ptp(costs[:, 2])
    f3_scaled = ((costs[:, 2] - costs[:, 2].min()) / f3_span
                 if f3_span > 0 else np.zeros(len(costs)))
    scatter = axis.scatter(costs[:, 0], costs[:, 1],
                           s=35 + 100 * f3_scaled, **{k: v for k, v in point_style.items()
                                                      if k != "s"})
    _mark_selected(axis, costs[best, 0], costs[best, 1], best)
    axis.set(xlabel="f1: flight time (s)",
             ylabel="f2: navigation deficit (s)",
             title="Objective trade-off\n(marker size = f3 visibility deficit)")
    fig.colorbar(scatter, ax=axis, label="TOPSIS closeness")
    axis.legend(fontsize=8)

    # Physical length and wind-aware duration.
    axis = axes[0, 1]
    axis.scatter(lengths_km, times_min, **point_style)
    _mark_selected(axis, lengths_km[best], times_min[best], best)
    axis.set(xlabel="horizontal route length (km)", ylabel="flight time (min)",
             title="Route length and duration")
    axis.legend(fontsize=8)

    # These are time-weighted route-average qualities reconstructed from f2/f3.
    axis = axes[0, 2]
    axis.scatter(nav_quality, visibility, **point_style)
    _mark_selected(axis, nav_quality[best], visibility[best], best)
    axis.set(xlabel="average navigation quality", ylabel="average visibility",
             title="Environmental quality (higher is better)", xlim=(-0.02, 1.02),
             ylim=(-0.02, 1.02))
    axis.legend(fontsize=8)

    # Constant-power energy is proportional to time; fall back to speed if absent.
    axis = axes[1, 0]
    if np.isfinite(energy).any():
        valid = np.isfinite(energy)
        axis.scatter(times_min[valid], energy[valid],
                     c=closeness[valid], cmap="viridis", vmin=0, vmax=1,
                     s=38, edgecolors="0.25", linewidths=0.35, alpha=0.85)
        _mark_selected(axis, times_min[best], energy[best], best)
        axis.set(xlabel="flight time (min)", ylabel="energy used (Wh)",
                 title="Energy demand")
    else:
        speeds = np.asarray(results["average_ground_speed_mps"], dtype=float)
        axis.scatter(lengths_km, speeds, **point_style)
        _mark_selected(axis, lengths_km[best], speeds[best], best)
        axis.set(xlabel="horizontal route length (km)",
                 ylabel="average ground speed (m/s)",
                 title="Ground speed (energy model unavailable)")
    axis.legend(fontsize=8)

    # Altitude envelopes are shown in TOPSIS order for an interpretable x axis.
    axis = axes[1, 1]
    ranked_min = min_altitude[order]
    ranked_max = max_altitude[order]
    rank_x = np.arange(1, len(order) + 1)
    axis.vlines(rank_x, ranked_min, ranked_max, color="0.65", linewidth=0.8,
                alpha=0.65)
    axis.scatter(rank_x, ranked_min, s=12, color="tab:blue", label="minimum")
    axis.scatter(rank_x, ranked_max, s=12, color="tab:red", label="maximum")
    axis.vlines(1, ranked_min[0], ranked_max[0], color="black", linewidth=3,
                label=f"selected P{best}")
    axis.set(xlabel="TOPSIS rank", ylabel="altitude MSL (m)",
             title="Route altitude envelope")
    axis.legend(fontsize=8)

    # A limited ranking panel stays readable even when the Pareto front is large.
    axis = axes[1, 2]
    shown = order[:min(max(1, int(top_routes)), len(order))]
    y = np.arange(len(shown))
    colors = ["black" if route_id == best else "tab:green" for route_id in shown]
    axis.barh(y, closeness[shown], color=colors, alpha=0.8)
    axis.set_yticks(y, [f"P{route_id}" for route_id in shown])
    axis.invert_yaxis()
    axis.set(xlabel="TOPSIS closeness", xlim=(0, 1),
             title=f"TOPSIS ranking (top {len(shown)})")
    for row, route_id in enumerate(shown):
        axis.text(min(closeness[route_id] + 0.01, 0.96), row,
                  f"{closeness[route_id]:.3f}", va="center", fontsize=7)

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize an existing route-characteristics CSV without rerunning EMOA*")
    parser.add_argument("--csv", required=True,
                        help="route_characteristics.csv produced during planning")
    parser.add_argument("--output",
                        help="output PNG (default: route_characteristics.png beside CSV)")
    parser.add_argument("--top-routes", type=int, default=15,
                        help="number of routes shown in the TOPSIS ranking panel")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if args.top_routes < 1:
        parser.error("--top-routes must be positive")
    if args.dpi < 1:
        parser.error("--dpi must be positive")

    csv_path = Path(args.csv)
    output_path = (Path(args.output) if args.output else
                   csv_path.with_name("route_characteristics.png"))
    results = load_characteristics_csv(csv_path)
    plot_characteristics(results, output_path, args.top_routes, args.dpi)
    best = int(results["topsis_best"])
    print(f"wrote {output_path}")
    print(f"routes={len(results['costs'])}; TOPSIS selected=P{best}; "
          f"length={results['horizontal_length_m'][best] / 1000.0:.3f} km; "
          f"time={results['flight_time_s'][best] / 60.0:.2f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

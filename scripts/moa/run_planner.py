"""CLI driver for the MOA* pre-flight planner.

Run as a module:

    python -m scripts.moa.run_planner --scenario lake
    python -m scripts.moa.run_planner --map layers.npz --start 0,13 --goal 18,0

Computes the exact Pareto front, prints the non-dominated cost vectors and
selects one path with TOPSIS. The map NPZ must contain layers compatible with
:class:`CostMap` (keys ``dem``, ``nav_density``, ``visibility``, ``wind_field``,
``occupancy`` and optional ``resolution_m``, ``start``, ``goal``, ``v_air``,
``v_max``).  Constraint fields such as ``z_max_m``, battery parameters, and
``landing_sites`` are also accepted when present.
"""

import argparse
import json
import sys

import numpy as np

from .edge_costs import CostMap
from .moa_star import EmoaStarLateBS
from .topsis import topsis
from . import synthetic

SCENARIOS = {
    "lake": synthetic.lake_map,
    "nfz": synthetic.nfz_map,
    "tailwind": synthetic.tailwind_map,
    "walled": synthetic.walled_map,
    "terrain": synthetic.terrain_map,
    "realistic": synthetic.realistic_map,
}


def _parse_cell(text):
    x, y = text.split(",")
    return (int(x), int(y))


def load_npz(path, start=None, goal=None):
    data = np.load(path, allow_pickle=True)
    kwargs = {key: data[key] for key in
              ("dem", "nav_density", "visibility", "wind_field", "occupancy")}
    for key in ("resolution_m", "v_air", "v_max", "v_air_min_mps",
                "v_air_max_mps", "z_max_m", "cruise_altitude_agl_m",
                "battery_energy_wh", "energy_reserve_wh", "cruise_power_w",
                "min_turn_radius_m"):
        if key in data:
            kwargs[key] = float(data[key])
    if "landing_sites" in data:
        kwargs["landing_sites"] = data["landing_sites"].astype(bool)
    if start is not None:
        kwargs["start"] = start
    elif "start" in data:
        kwargs["start"] = tuple(int(v) for v in data["start"])
    if goal is not None:
        kwargs["goal"] = goal
    elif "goal" in data:
        kwargs["goal"] = tuple(int(v) for v in data["goal"])
    return CostMap(**kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="UAV pre-flight MOA* planner")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS),
                        help="built-in synthetic scenario")
    parser.add_argument("--map", help="NPZ file with CostMap-compatible layers")
    parser.add_argument("--start", type=_parse_cell, help="x,y cell index")
    parser.add_argument("--goal", type=_parse_cell, help="x,y cell index")
    parser.add_argument("--v-air", type=float, help="airspeed (m/s)")
    parser.add_argument("--v-max", type=float, help="max wind (m/s)")
    parser.add_argument("--weights", default="0.5,0.3,0.2",
                        help="TOPSIS objective weights (comma-separated)")
    args = parser.parse_args(argv)

    if args.scenario:
        cost_map = SCENARIOS[args.scenario]()
    elif args.map:
        cost_map = load_npz(args.map, args.start, args.goal)
    else:
        parser.error("provide --scenario or --map")

    solver = EmoaStarLateBS(
        cost_map,
        v_air=args.v_air if args.v_air is not None else cost_map.v_air,
        v_max=args.v_max if args.v_max is not None else cost_map.v_max,
    )
    solutions = solver.solve()

    if not solutions:
        print("No feasible path found.")
        return 1

    weights = np.array([float(w) for w in args.weights.split(",")])
    costs = np.array([c for _, c in solutions])
    order, closeness = topsis(costs, weights)
    best = order[0]

    print(f"Grid: {cost_map.shape[0]}x{cost_map.shape[1]} cells, "
          f"resolution {cost_map.resolution_m} m")
    print(f"start={cost_map.start} goal={cost_map.goal} "
          f"v_air={cost_map.v_air} v_max={cost_map.v_max}")
    print(f"Pareto front: {len(solutions)} non-dominated paths "
          f"(expanded {solver.n_expanded} labels, generated {solver.n_generated})")
    print(f"{'i':>2} {'f1_time':>10} {'f2_nav_def':>12} {'f3_vis_def':>12} "
          f"{'steps':>6} {'C_i':>6} {'best':>5}")
    for rank, idx in enumerate(order):
        path, c = solutions[idx]
        print(f"{idx:>2} {c[0]:>10.3f} {c[1]:>12.3f} {c[2]:>12.3f} "
              f"{len(path):>6} {closeness[idx]:>6.3f} {'*' if idx == best else '':>5}")
    print("\nSelected path (TOPSIS, C = {:.3f}):".format(closeness[best]))
    print(json.dumps([list(p) for p in solutions[best][0]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

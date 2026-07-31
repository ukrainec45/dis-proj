"""Visual demonstrations of the EMOA*-Late-BS planner.

Modes
-----
trace : annotated step-by-step transcript of the search on a small map.
        No plotting dependencies required.
map   : static figure -- routes on the grid, Pareto-front scatter,
        and the backward-Dijkstra heuristic. Requires matplotlib.
anim  : GIF of the label-expansion sequence (markers light up in the
        order labels are expanded, final frames overlay the Pareto routes).
        Requires matplotlib + pillow.

Run as a module:

    python -m scripts.moa.visualize --mode trace
    python -m scripts.moa.visualize --mode trace --limit 60
    python -m scripts.moa.visualize --mode map    --save-dir plots
    python -m scripts.moa.visualize --mode anim   --save-dir plots
    python -m scripts.moa.visualize --mode map    --scenario lake
"""

import argparse
import os
import sys

import numpy as np

from .domination import non_dominated_indices
from .heuristics import compute_heuristics
from .moa_star import EmoaStarLateBS
from .topsis import topsis
from . import synthetic

SCENARIOS = {
    "demo": synthetic.demo_map,
    "lake": synthetic.lake_map,
    "terrain": synthetic.terrain_map,
    "realistic": synthetic.realistic_map,
}


# ----------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------
def _fmt(v, nd=2):
    return "(" + ", ".join(f"{x:.{nd}f}" for x in v) + ")"


def _cell(v):
    return f"({v[0]},{v[1]})"


def _fmt_trunc(v):
    return "[" + ", ".join(f"{x:.2f}" for x in v) + "]"


def _representative_indices(costs, best, limit=8):
    """Return every route for a small front, key routes for a large one."""
    if len(costs) <= limit:
        return list(range(len(costs)))
    return sorted({int(np.argmin(costs[:, 0])), int(np.argmin(costs[:, 1])),
                   int(np.argmin(costs[:, 2])), int(best)})


def dedupe_front(sols, ndigits=6):
    """Drop float-noise duplicates from the returned front (for display only)."""
    costs = np.array([c for _, c in sols])
    key = np.round(costs, ndigits)
    keep = non_dominated_indices(key)
    out = [(p, c) for i, (p, c) in enumerate(sols) if i in keep]
    out.sort(key=lambda item: item[1])
    return out


def _low_nav_components(cost_map):
    """8-connected components of cells with ``nav_density < 0.5`` (pure numpy).

    Returns ``(labels, n, biggest)``: ``labels`` is the component id per cell
    (0 = not low), ``n`` the number of components, ``biggest`` the 1-based id
    of the largest one (0 if there are none). Used so only the main feature-poor
    region (the lake) is labelled ``~``/``lake``, while separate poor-terrain
    patches keep their dark colour but are not mislabelled.
    """
    from collections import deque

    mask = cost_map.nav_density < 0.5
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=int)
    offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               if (dx, dy) != (0, 0)]
    cur = 0
    for sy in range(H):
        for sx in range(W):
            if mask[sy, sx] and labels[sy, sx] == 0:
                cur += 1
                labels[sy, sx] = cur
                q = deque([(sx, sy)])
                while q:
                    x, y = q.popleft()
                    for dx, dy in offsets:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < W and 0 <= ny < H and mask[ny, nx] \
                                and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            q.append((nx, ny))
    if cur == 0:
        return labels, 0, 0
    sizes = np.bincount(labels.ravel())[1:]
    return labels, cur, int(np.argmax(sizes)) + 1


# ----------------------------------------------------------------------------
# ASCII map rendering
# ----------------------------------------------------------------------------
def ascii_map(cost_map, path=None, marker=None):
    """Render the grid to ASCII. ``path`` is a waypoint list that is overlaid
    with ``marker`` (a single character); S/G/~/o/f/X are fixed symbols."""
    H, W = cost_map.shape
    labels, _, lake_label = _low_nav_components(cost_map)
    rows = []
    for y in range(H):
        line = []
        for x in range(W):
            cell = (x, y)
            if path is not None and cell in path:
                if cell == cost_map.start:
                    line.append("S")
                elif cell == cost_map.goal:
                    line.append("G")
                else:
                    line.append(marker)
            elif cell == cost_map.start:
                line.append("S")
            elif cell == cost_map.goal:
                line.append("G")
            elif cost_map.is_occupied(cell):
                line.append("X")
            elif cost_map.nav_density[y, x] < 0.5:
                line.append("~" if lake_label and labels[y, x] == lake_label
                            else "o")
            elif cost_map.visibility[y, x] < 0.6:
                line.append("f")
            else:
                line.append(".")
        rows.append("".join(line))
    return "\n".join(rows)


def _print_map_header(cost_map):
    print(f"MAP  {cost_map.shape[0]}x{cost_map.shape[1]} cells, "
          f"resolution {cost_map.resolution_m:.0f} m")
    print(f"start = {cost_map.start}   goal = {cost_map.goal}   "
          f"v_air = {cost_map.v_air} m/s   v_max = {cost_map.v_max} m/s")
    print("legend:  . good navigation quality   ~ low navigation-quality area   "
          "o poor terrain   f low-visibility zone   X NFZ   "
          "S start   G goal")
    if cost_map.z_max_m is not None:
        print(f"constraints: z_max={cost_map.z_max_m:g} m, "
              f"turn radius >= {cost_map.min_turn_radius_m:g} m"
              if cost_map.min_turn_radius_m is not None else
              f"constraints: z_max={cost_map.z_max_m:g} m")


# ----------------------------------------------------------------------------
# trace mode
# ----------------------------------------------------------------------------
def trace(cost_map, limit=30, expand_report=25):
    """Run the solver with recording and print an annotated transcript."""
    solver = EmoaStarLateBS(cost_map, record=True)
    solutions = solver.solve()
    events = solver.events
    front = dedupe_front(solutions)

    _print_map_header(cost_map)
    print("\nGrid (row 0 at the top):")
    print(ascii_map(cost_map))

    # heuristic preview: exact flight-time-to-goal from the backward Dijkstra
    h = compute_heuristics(cost_map, solver.v_air, solver.v_max)
    print("\nBackward-Dijkstra heuristic h1(v) = exact flight time (s) "
          "to the goal:\n(admissible + consistent, so EMOA* may prune safely)")
    for y in range(cost_map.shape[0]):
        print("   " + " ".join(f"{h[y, x, 0]:6.1f}"
                               if np.isfinite(h[y, x, 0]) else "    inf"
                               for x in range(cost_map.shape[1])))

    print(f"\nSearch transcript: {len(events)} events recorded, showing the "
          f"first {limit} pops (plus every goal decision).")
    print("   pop  = label taken off OPEN      f = g + h  (h = heuristic)\n"
          "   EXPAND  = kept + its vertex front is updated\n"
          "   PRUNE front/sol = discarded by late dominance check\n"
          "   gen / gen_pruned = a successor pushed onto / rejected from OPEN\n"
          "   GOAL n = complete path found; (k->...)  evicted solutions removed")

    pops_seen = 0
    adds_seen = 0
    expanded_since_report = 0
    detail_children = True
    for k, ev in enumerate(events, start=1):
        kind = ev["kind"]
        if kind == "add_sol":
            adds_seen += 1
            label = f"GOAL {len(ev['prev_sols'])} -> {ev['n_sols']}"
            if ev["evicted"]:
                label += "  evicted: " + " ".join(_fmt(g) for g in ev["evicted"])
            print(f"  {k:>4} {label:<34} {_fmt(ev['g'])}   ({ev['n_sols']} sols)")
            continue
        if kind != "pop":
            continue
        pops_seen += 1
        if ev["pruned"]:
            print(f"  {k:>4} pop {_cell(ev['vertex']):<8} f={_fmt(ev['f']):<22}"
                  f" PRUNE ({ev['reason']})   OPEN {ev['n_open']}")
            continue
        if ev["vertex"] == solver.goal:
            continue  # handled by the add_sol event
        print(f"  {k:>4} pop {_cell(ev['vertex']):<8} f={_fmt(ev['f']):<22}"
              f" EXPAND  front[{_cell(ev['vertex'])}]={ev['front_size']}   "
              f"OPEN {ev['n_open']}")
        expanded_since_report += 1
        if expanded_since_report >= expand_report:
            expanded_since_report = 0
            print(f"        .... {pops_seen} labels popped so far, "
                  f"{len(ev['prev_sols'])} solutions in hand ....")
        if detail_children:
            # the flat stream interleaves this pop's gen events before the next pop
            while k < len(events) and events[k - 1]["kind"] in ("gen", "gen_pruned"):
                c = events[k - 1]
                if c["kind"] == "gen":
                    print(f"        gen {_cell(c['vertex']):<8} f={_fmt(c['f']):<22}"
                          f" push (parent {_cell(c['parent'])})")
                else:
                    print(f"        gen {_cell(c['vertex']):<8} f={_fmt(c['f']):<22}"
                          f" DISCARD ({c['reason']})")
                k += 1
            if pops_seen >= 4:
                detail_children = False
        if pops_seen >= limit:
            break

    n_kept = sum(1 for e in events if e["kind"] == "add_sol")
    print(f"\n... remaining events suppressed "
          f"({adds_seen} of {n_kept} goal decisions shown above, "
          f"{len(events)} events total, "
          f"{solver.n_expanded} expansions, {solver.n_generated} generations).")

    print(f"\nFINAL PARETO FRONT ({len(front)} non-dominated paths):")
    print(f"   {'i':<3} {'f1_time':>8} {'f2_nav':>8} {'f3_vis':>8} {'steps':>6}")
    for i, (path, c) in enumerate(front):
        print(f"   {i:<3} {c[0]:>8.2f} {c[1]:>8.2f} {c[2]:>8.2f} {len(path):>6}")
        print(ascii_map(cost_map, path=path, marker=str(i)))
        print()

    order, closeness = topsis(np.array([c for _, c in front]),
                              np.array([0.5, 0.3, 0.2]))
    print(f"TOPSIS (w = 0.5/0.3/0.2) picks path {order[0]} "
          f"(C = {closeness[order[0]]:.3f}).")


# ----------------------------------------------------------------------------
# matplotlib modes
# ----------------------------------------------------------------------------
def _setup_plotting():
    import matplotlib
    matplotlib.use("Agg")  # headless-safe (saves to files, no GUI)
    import matplotlib.pyplot as plt
    return plt


def map_plot(cost_map, save_dir, weights=(0.5, 0.3, 0.2)):
    plt = _setup_plotting()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, Patch
    solver = EmoaStarLateBS(cost_map)
    solutions = dedupe_front(solver.solve())
    if not solutions:
        print("No feasible path found; nothing to plot.")
        return 1
    h = compute_heuristics(cost_map, solver.v_air, solver.v_max)
    H, W = cost_map.shape

    costs = np.array([c for _, c in solutions])
    order, closeness = topsis(costs, np.array(weights))
    best = int(order[0])
    shown = _representative_indices(costs, best)
    route_colors = {idx: plt.cm.tab10(pos % 10) for pos, idx in enumerate(shown)}

    fig = plt.figure(figsize=(19, 7.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.05, 0.9], wspace=0.40)
    fig.subplots_adjust(bottom=0.31)

    # --- left: map with Pareto routes --------------------------------
    ax = fig.add_subplot(gs[0])
    nav = np.ma.masked_array(cost_map.nav_density,
                             mask=cost_map.occupancy)
    im = ax.imshow(nav, cmap="viridis", vmin=0, vmax=1,
                   extent=[-0.5, W - 0.5, H - 0.5, -0.5], aspect="equal")
    low_visibility = cost_map.visibility < 0.6
    for y, x in np.argwhere(low_visibility):
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                               facecolor=(0.35, 0.6, 1.0, 0.3),
                               edgecolor="tab:blue", hatch="//", lw=0.9))
    altitude_blocked = (cost_map.z_max_m is not None and
                        cost_map.dem + cost_map.cruise_altitude_agl_m > cost_map.z_max_m)
    if isinstance(altitude_blocked, np.ndarray):
        for y, x in np.argwhere(altitude_blocked & ~cost_map.occupancy):
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                                   facecolor=(0.9, 0.2, 0.2, 0.22),
                                   edgecolor="firebrick", hatch="xx", lw=0.7))
    ax.set_title("Pareto-optimal routes on the grid\n"
                 "(background: navigation density, colorbar below)", fontsize=10)
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.grid(True, color="0.7", lw=0.4)
    ax.plot([cost_map.start[0]], [cost_map.start[1]], "o", color="white",
            mec="black", ms=9, zorder=5)
    ax.plot([cost_map.goal[0]], [cost_map.goal[1]], "*", color="red",
            ms=16, zorder=5)
    if cost_map.landing_sites is not None:
        ly, lx = np.where(cost_map.landing_sites)
        ax.plot(lx, ly, "^", color="limegreen", mec="black", ms=8, zorder=5)

    # route roles (which objective each route optimises) + TOPSIS pick
    roles = {int(np.argmin(costs[:, 0])): "fastest",
             int(np.argmin(costs[:, 1])): "best navigation",
             int(np.argmin(costs[:, 2])): "best visibility"}
    handles = []
    for i in shown:
        path, c = solutions[i]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        if i == best:
            ax.plot(xs, ys, color="black", lw=5, alpha=0.22, zorder=3.4)
        ax.plot(xs, ys, color=route_colors[i], lw=(3.4 if i == best else 2.4),
                alpha=0.95, zorder=4)
        ax.annotate(f"P{i}", (xs[1], ys[1]), color=route_colors[i], fontsize=9,
                    fontweight="bold", xytext=(3, 3), textcoords="offset points")
        pick = "  * TOPSIS choice" if i == best else ""
        handles.append(Line2D([], [], color=route_colors[i], lw=2.6,
                              label=f"P{i}  {roles.get(i, 'representative'):<17} f=({c[0]:.2f}, "
                                    f"{c[1]:.2f}, {c[2]:.2f}){pick}"))
    if np.any(cost_map.nav_density < 0.5):
        handles.append(Patch(facecolor=plt.cm.viridis(0.0), edgecolor="none",
                             label="low navigation-quality area  (nav_density < 0.5)"))
    if low_visibility.any():
        handles.append(Patch(facecolor=(0.35, 0.6, 1.0, 0.3),
                             edgecolor="tab:blue", hatch="//",
                             label="low-visibility zone  (visibility < 0.6)"))
    if cost_map.occupancy.any():
        handles.append(Patch(facecolor="white", edgecolor="0.45",
                             label="NFZ  (no-fly zone)"))
    if isinstance(altitude_blocked, np.ndarray) and altitude_blocked.any():
        handles.append(Patch(facecolor=(0.9, 0.2, 0.2, 0.22), edgecolor="firebrick",
                             hatch="xx", label="above z_max (blocked)"))
    if cost_map.landing_sites is not None:
        handles.append(Line2D([], [], marker="^", color="limegreen", mfc="limegreen",
                              mec="black", ms=8, linestyle="None", label="emergency landing site"))
    handles.append(Line2D([], [], marker="o", color="white", mfc="white",
                          mec="black", ms=8, linestyle="None", label="start"))
    handles.append(Line2D([], [], marker="*", color="red", ms=14,
                          linestyle="None", label="goal"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=1, fontsize=7.5, framealpha=0.92, handletextpad=0.5,
              columnspacing=0.8,
              title=("P_i = Pareto-optimal route; f = (time, nav-deficit, vis-deficit)"
                     if len(shown) == len(solutions) else
                     f"{len(solutions)} Pareto routes; key routes shown"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="nav_density (0 = featureless, 1 = rich)")
    ax.set_xlabel("column x"); ax.set_ylabel("row y")

    # --- middle: Pareto front scatter --------------------------------
    ax2 = fig.add_subplot(gs[1])
    sizes = 90 + 70 * costs[:, 2] / max(costs[:, 2].max(), 1e-9)
    ax2.scatter(costs[:, 0], costs[:, 1], s=sizes, c="0.75",
                edgecolors="0.35", zorder=2, label="Pareto route")
    for i in shown:
        c = costs[i]
        ax2.scatter([c[0]], [c[1]], s=sizes[i], c=[route_colors[i]],
                    edgecolors="black", zorder=3)
        ax2.annotate(f"P{i}  ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})",
                     (c[0], c[1]), fontsize=8,
                     xytext=(4, 6), textcoords="offset points")
    ax2.plot([costs[best, 0]], [costs[best, 1]], "*", color="black", ms=18,
             zorder=4)
    ax2.annotate("TOPSIS best", (costs[best, 0], costs[best, 1]),
                 fontsize=8, xytext=(4, -14), textcoords="offset points")
    ax2.set_xlabel("f1 = flight time (s)")
    ax2.set_ylabel("f2 = navigation-deficit time (s)")
    ax2.set_title("Pareto front in objective space\n"
                  "(P_i = route of the same color; marker size = f3, "
                  "* = TOPSIS pick)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    ax2.annotate("", xy=(costs[0, 0] - 0.4, costs[-1, 1]),
                 xytext=(costs[0, 0] - 0.4, costs[0, 1]),
                 arrowprops=dict(arrowstyle="->", color="0.3"))
    ax2.text(costs[0, 0] - 0.42, (costs[0, 1] + costs[-1, 1]) / 2,
             "trade-off", rotation=90, fontsize=8, color="0.3", va="center")

    # --- right: heuristic cost-to-goal --------------------------------
    ax3 = fig.add_subplot(gs[2])
    hdisp = np.ma.masked_array(h[:, :, 0], mask=cost_map.occupancy)
    im = ax3.imshow(hdisp, cmap="magma",
                    extent=[-0.5, W - 0.5, H - 0.5, -0.5], aspect="equal")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="h1 (s)")
    ax3.plot([cost_map.start[0]], [cost_map.start[1]], "o", color="white",
             mec="black", ms=9, zorder=5)
    ax3.plot([cost_map.goal[0]], [cost_map.goal[1]], "*", color="red", ms=16,
             zorder=5)
    ax3.set_title("Heuristic h1 = exact time-to-goal\n(backward Dijkstra)",
                  fontsize=10)
    ax3.set_xticks(range(W)); ax3.set_yticks(range(H))
    ax3.set_xlabel("column x"); ax3.set_ylabel("row y")

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"moa_{cost_map.shape[0]}x{cost_map.shape[1]}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}  ({len(solutions)} Pareto paths, "
          f"{solver.n_expanded} expansions / {solver.n_generated} generations)")
    return 0


def anim(cost_map, save_dir, frame_step=3, fps=8):
    """GIF of the label-expansion sequence overlaid on the grid."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    solver = EmoaStarLateBS(cost_map, record=True)
    solutions = dedupe_front(solver.solve())
    if not solutions:
        print("No feasible path found; nothing to animate.")
        return 1
    H, W = cost_map.shape

    expands = [e for e in solver.events if e["kind"] == "expand"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(expands)))
    path_colors = plt.cm.tab10(np.arange(len(solutions)))

    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.imshow(np.ma.masked_array(cost_map.nav_density, mask=cost_map.occupancy),
              cmap="Greys_r", vmin=0, vmax=1,
              extent=[-0.5, W - 0.5, H - 0.5, -0.5], aspect="equal", alpha=0.7)
    ax.plot([cost_map.start[0]], [cost_map.start[1]], "o", color="white",
            mec="black", ms=9, zorder=6)
    ax.plot([cost_map.goal[0]], [cost_map.goal[1]], "*", color="red", ms=16,
            zorder=6)
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.grid(True, color="0.8", lw=0.4)

    drawn, lines = [], []
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                   fontsize=10, family="monospace",
                   bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    def frame(idx):
        n = (idx + 1) * frame_step
        shown = expands[:n]
        while len(drawn) < len(shown):
            v = shown[len(drawn)]["vertex"]
            drawn.append(ax.plot([v[0]], [v[1]], "o", color=colors[len(drawn)],
                                 ms=7, alpha=0.55, zorder=3)[0])
        for i, (path_, _) in enumerate(solutions):
            xs = [c[0] for c in path_]
            ys = [c[1] for c in path_]
            if len(lines) <= i:
                lines.append(ax.plot([], [], color=path_colors[i], lw=2.4,
                                     zorder=4)[0])
            shown_pts = min(len(xs), max(0, n - len(expands) + i * 3 + 1))
            lines[i].set_data(xs[:shown_pts], ys[:shown_pts])
        text.set_text(f"expanded {len(shown)} / {len(expands)}\n"
                      f"open labels {shown[-1]['n_open']}\n"
                      f"solutions {shown[-1]['n_sols']}")
        return drawn + lines + [text]

    n_frames = max(1, int(np.ceil(len(expands) / frame_step)) + 4)
    anim_fig = FuncAnimation(fig, frame, frames=n_frames, blit=False)
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"moa_{cost_map.shape[0]}x{cost_map.shape[1]}.gif")
    anim_fig.save(out, writer=PillowWriter(fps=fps))
    print(f"wrote {out}  ({len(expands)} expansions in {n_frames} frames, "
          f"{len(solutions)} Pareto paths)")
    plt.close(fig)
    return 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="EMOA*-Late-BS visualisation (trace / map / anim)")
    parser.add_argument("--mode", choices=("trace", "map", "anim"), required=True)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="demo")
    parser.add_argument("--save-dir", default="plots")
    parser.add_argument("--limit", type=int, default=30,
                        help="trace: number of pops to print in detail")
    parser.add_argument("--frame-step", type=int, default=3,
                        help="anim: labels per frame")
    parser.add_argument("--weights", default="0.5,0.3,0.2",
                        help="TOPSIS objective weights (comma-separated)")
    args = parser.parse_args(argv)

    cost_map = SCENARIOS[args.scenario]()
    weights = tuple(float(w) for w in args.weights.split(","))

    if args.mode == "trace":
        trace(cost_map, limit=args.limit)
        return 0
    if args.mode == "map":
        return map_plot(cost_map, args.save_dir, weights)
    return anim(cost_map, args.save_dir, frame_step=args.frame_step)


if __name__ == "__main__":
    sys.exit(main())

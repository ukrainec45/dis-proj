"""Consistent vector heuristic for EMOA*-Late-BS via backward Dijkstra.

For each objective, the exact single-objective cost-to-goal (backward shortest
paths) is computed with the same edge-cost function the forward search uses.
Component-wise this yields an admissible and consistent heuristic vector
h(v) = (h1(v), h2(v), h3(v)) with h(goal) = 0, which the dimensionality
reduction in ``moa_star.py`` relies on.
"""

import heapq

import numpy as np

from .edge_costs import NEIGHBOR_OFFSETS, M_OBJECTIVES, edge_objectives


def _backward_dijkstra(cost_map, v_air, v_max, objective):
    """Minimum cost of objective ``objective`` from every free cell to the goal."""
    goal = cost_map.goal
    H, W = cost_map.dem.shape
    dist = np.full((H, W), np.inf)

    gx, gy = goal[0], goal[1]
    dist[gy, gx] = 0.0
    heap = [(0.0, gx, gy)]
    while heap:
        d, x, y = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dx, dy in NEIGHBOR_OFFSETS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if cost_map.occupancy[ny, nx]:
                continue
            c = edge_objectives((nx, ny), (x, y), cost_map, v_air, v_max)
            if c is None:
                continue
            nd = d + c[objective]
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(heap, (nd, nx, ny))
    return dist


def compute_heuristics(cost_map, v_air, v_max):
    """Return an [H, W, M] lower-bound heuristic array.

    ``h[y, x, j]`` is the exact backward cost of objective ``j`` from cell
    ``(x, y)`` to the goal (``np.inf`` where unreachable).
    """
    hs = [_backward_dijkstra(cost_map, v_air, v_max, j) for j in range(M_OBJECTIVES)]
    return np.stack(hs, axis=-1)


def compute_time_to_landing(cost_map, v_air, v_max):
    """Minimum flight time from each cell to any configured landing site.

    The result is used only for the optional battery-reserve constraint.  It is
    calculated on reversed directed edges, just like the planner heuristic.
    """
    if cost_map.landing_sites is None:
        return None
    H, W = cost_map.dem.shape
    dist = np.full((H, W), np.inf)
    heap = []
    for y, x in np.argwhere(cost_map.landing_sites & ~cost_map.occupancy):
        dist[y, x] = 0.0
        heapq.heappush(heap, (0.0, int(x), int(y)))
    while heap:
        d, x, y = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dx, dy in NEIGHBOR_OFFSETS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            c = edge_objectives((nx, ny), (x, y), cost_map, v_air, v_max)
            if c is None:
                continue
            nd = d + c[0]
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(heap, (nd, nx, ny))
    return dist

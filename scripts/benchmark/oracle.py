"""Small-map exhaustive oracle for validating the EMOA* Pareto front."""

import numpy as np

from scripts.moa.domination import non_dominated_indices
from scripts.moa.edge_costs import NEIGHBOR_OFFSETS, edge_objectives, turn_is_feasible
from scripts.moa.moa_star import EmoaStarLateBS
from scripts.moa import synthetic

from .metrics import exact_front_recall


def brute_force_front(cost_map):
    """Enumerate simple paths; intended only for tiny validation maps."""
    found = []

    def visit(node, previous, visited, cost):
        if node == cost_map.goal:
            found.append(tuple(cost))
            return
        x, y = node
        for dx, dy in NEIGHBOR_OFFSETS:
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < cost_map.shape[1] and 0 <= nxt[1] < cost_map.shape[0]):
                continue
            if nxt in visited or cost_map.is_occupied(nxt):
                continue
            if previous is not None and not turn_is_feasible(previous, node, nxt, cost_map):
                continue
            edge = edge_objectives(node, nxt, cost_map, cost_map.v_air, cost_map.v_max)
            if edge is not None:
                visit(nxt, node, visited | {nxt}, [a + b for a, b in zip(cost, edge)])

    visit(cost_map.start, None, {cost_map.start}, [0.0, 0.0, 0.0])
    if not found:
        return []
    arr = np.asarray(found)
    return [tuple(arr[index]) for index in non_dominated_indices(arr)]


def validate_oracle_suite(seeds=range(10)):
    """Compare EMOA* against exhaustive fronts for seeded 4×3 maps."""
    rows = []
    for seed in seeds:
        cost_map = synthetic.brute_map(seed=seed)
        expected = brute_force_front(cost_map)
        actual = [cost for _, cost in EmoaStarLateBS(cost_map).solve()]
        rows.append({
            "seed": seed,
            "oracle_front_size": len(expected),
            "emoa_front_size": len(actual),
            "oracle_recall": exact_front_recall(actual, expected),
            "exact_match": exact_front_recall(actual, expected) == 1.0 and
                           exact_front_recall(expected, actual) == 1.0,
        })
    return rows

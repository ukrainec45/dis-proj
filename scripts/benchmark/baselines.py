"""Planner adapters used by the benchmark runner."""

from time import perf_counter

import numpy as np

from scripts.moa.domination import dominates, non_dominated_indices
from scripts.moa.moa_star import EmoaStarLateBS

from .models import PlannerResult
from .scalar_astar import ScalarAStar


QUARTER_SIMPLEX_WEIGHTS = tuple(
    (i / 4.0, j / 4.0, k / 4.0)
    for i in range(5) for j in range(5) for k in range(5)
    if i + j + k == 4
)


def _run_scalar(cost_map, method, weights):
    solver = ScalarAStar(cost_map, weights)
    started = perf_counter()
    solutions = solver.solve()
    return PlannerResult(method, solutions, (perf_counter() - started) * 1000.0,
                         solver.n_expanded, solver.n_generated, None,
                         {"weights": list(weights)})


def run_emoa(cost_map):
    solver = EmoaStarLateBS(cost_map)
    started = perf_counter()
    solutions = solver.solve()
    return PlannerResult("emoa_star", solutions, (perf_counter() - started) * 1000.0,
                         solver.n_expanded, solver.n_generated)


def run_time_only_astar(cost_map):
    return _run_scalar(cost_map, "time_only_astar", (1.0, 0.0, 0.0))


def _unique_non_dominated(solutions):
    if not solutions:
        return []
    unique = []
    for path, cost in solutions:
        if not any(np.allclose(cost, previous, rtol=0.0, atol=1e-9)
                   for _, previous in unique):
            unique.append((path, cost))
    costs = [cost for _, cost in unique]
    return [unique[i] for i in non_dominated_indices(costs)]


def run_repeated_weighted_astar(cost_map, weights=QUARTER_SIMPLEX_WEIGHTS):
    started = perf_counter()
    all_solutions = []
    total_expanded = total_generated = 0
    feasible_runs = 0
    for weight in weights:
        run = _run_scalar(cost_map, "weighted_astar", weight)
        all_solutions.extend(run.solutions)
        total_expanded += run.n_expanded or 0
        total_generated += run.n_generated or 0
        feasible_runs += int(run.feasible)
    return PlannerResult(
        "repeated_weighted_astar", _unique_non_dominated(all_solutions),
        (perf_counter() - started) * 1000.0, total_expanded, total_generated, None,
        {"weights": [list(weight) for weight in weights], "feasible_weight_runs": feasible_runs},
    )


METHODS = (run_emoa, run_time_only_astar, run_repeated_weighted_astar)

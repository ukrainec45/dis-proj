"""Pareto-front metrics with no dependency beyond NumPy."""

import itertools

import numpy as np

from scripts.moa.domination import non_dominated_indices


EPS = 1e-9


def cost_array(solutions):
    return np.asarray([cost for _, cost in solutions], dtype=float).reshape((-1, 3)) \
        if solutions else np.empty((0, 3), dtype=float)


def coverage(front_a, front_b):
    """C(A, B): fraction of B's vectors weakly dominated by a vector in A."""
    a, b = cost_array(front_a), cost_array(front_b)
    if len(b) == 0:
        return None
    if len(a) == 0:
        return 0.0
    return float(np.mean([np.any(np.all(a <= vector + EPS, axis=1)) for vector in b]))


def union_front(results):
    """Cost vectors non-dominated across all feasible method results."""
    vectors = [cost for result in results for _, cost in result.solutions]
    if not vectors:
        return np.empty((0, 3), dtype=float)
    arr = np.asarray(vectors, dtype=float)
    return arr[non_dominated_indices(arr)]


def reference_point(front):
    """A conservative, stored reference point for minimisation hypervolume."""
    front = np.asarray(front, dtype=float)
    if front.size == 0:
        return None
    return np.maximum(1.05 * front.max(axis=0), 1e-6)


def hypervolume(front, reference):
    """Exact union volume of 3-D minimisation boxes [cost, reference]."""
    points = np.asarray(front, dtype=float)
    if len(points) == 0 or reference is None:
        return 0.0
    points = points[np.all(points < np.asarray(reference) - EPS, axis=1)]
    if len(points) == 0:
        return 0.0
    points = points[non_dominated_indices(points)]
    axes = [np.unique(np.r_[points[:, dimension], reference[dimension]])
            for dimension in range(3)]
    volume = 0.0
    for indices in itertools.product(*[range(len(axis) - 1) for axis in axes]):
        lower = np.array([axes[d][indices[d]] for d in range(3)])
        if np.any(np.all(points <= lower + EPS, axis=1)):
            upper = np.array([axes[d][indices[d] + 1] for d in range(3)])
            volume += float(np.prod(upper - lower))
    return volume


def scenario_metrics(results):
    """Per-method quality metrics computed against a common scenario reference."""
    shared_front = union_front(results)
    reference = reference_point(shared_front)
    by_method = {}
    for result in results:
        by_method[result.method] = {
            "hypervolume": hypervolume(cost_array(result.solutions), reference),
            "coverage": {
                other.method: coverage(result.solutions, other.solutions)
                for other in results if other.method != result.method
            },
        }
    return shared_front, reference, by_method

"""Pareto dominance helpers for multi-objective cost vectors (all minimised)."""

import numpy as np


def weakly_dominates(a, b):
    """True if a is no worse than b on every objective."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= b))


def dominates(a, b):
    """True if a weakly dominates b and is strictly better on at least one objective."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_indices(cost_vectors):
    """Indices of cost vectors that are mutually non-dominated."""
    arr = np.asarray(cost_vectors, dtype=float)
    keep = []
    for i in range(len(arr)):
        if any(dominates(arr[j], arr[i]) for j in range(len(arr)) if j != i):
            continue
        keep.append(i)
    return keep

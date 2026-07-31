"""TOPSIS multi-criteria decision making over the Pareto front.

Selects a single path from the non-dominated front using the closeness
coefficient C_i = d_i^- / (d_i^+ + d_i^-).
"""

import numpy as np


def topsis(cost_matrix, weights, minimize=None):
    """Rank rows of ``cost_matrix`` (paths) by TOPSIS closeness.

    Parameters
    ----------
    cost_matrix : (n_paths, M) array of objective values (all minimised here).
    weights : (M,) array of objective weights (need not sum to 1).
    minimize : (M,) bool array, default all True.

    Returns
    -------
    order : (n_paths,) int array, indices sorted by descending closeness.
    closeness : (n_paths,) float array of C_i in [0, 1].
    """
    D = np.asarray(cost_matrix, dtype=float)
    n, m = D.shape
    if minimize is None:
        minimize = np.ones(m, dtype=bool)
    minimize = np.asarray(minimize, dtype=bool)
    w = np.asarray(weights, dtype=float)

    denom = np.sqrt(np.sum(D ** 2, axis=0))
    denom = np.where(denom > 0, denom, 1.0)
    Dhat = D / denom
    V = Dhat * w

    ideal = np.where(minimize, V.min(axis=0), V.max(axis=0))
    anti = np.where(minimize, V.max(axis=0), V.min(axis=0))

    d_pos = np.sqrt(np.sum((V - ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((V - anti) ** 2, axis=1))
    total = d_pos + d_neg
    closeness = np.where(total > 0, d_neg / np.where(total > 0, total, 1.0), 0.5)

    order = np.argsort(-closeness, kind="stable")
    return order, closeness

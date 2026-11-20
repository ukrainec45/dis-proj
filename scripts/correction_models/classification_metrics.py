"""Proper multiclass scores, event reliability and scene-cluster uncertainty."""
import numpy as np

from .contracts import probabilities, derived
from .models import labels


def reliability(p, outcome, bins=5):
    result = []
    for i in range(bins):
        mask = (p >= i/bins) & ((p <= 1) if i == bins-1 else (p < (i+1)/bins))
        result.append(dict(lower=i/bins, upper=(i+1)/bins, count=int(mask.sum()),
                           predicted=float(p[mask].mean()) if mask.any() else None,
                           observed=float(outcome[mask].mean()) if mask.any() else None))
    return result


def evaluate(p, y):
    p, y = probabilities(p), labels(y)
    if len(p) != len(y):
        raise ValueError('Prediction/label mismatch')
    onehot = np.eye(8)[y]
    accepted = y > 0
    accepted_mass = p[:, 1:].sum(axis=1)
    mask = accepted & (accepted_mass > 0)
    q = p[mask, 1:]/accepted_mass[mask, None]
    target = np.eye(7)[y[mask]-1]
    rps = float(np.square(np.cumsum(q, axis=1)[:, :-1]-np.cumsum(target, axis=1)[:, :-1]).mean()) if mask.any() else None
    d = derived(p)
    events = dict(accept=accepted, improve=y >= 5, worsen=(y >= 1) & (y <= 3), severe_worsen=y == 1)
    return dict(sample_count=len(y), log_loss=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
                brier_score=float(np.square(p-onehot).sum(axis=1).mean()),
                conditional_accepted_rps=rps, rps_evaluable_count=int(mask.sum()),
                rps_zero_accepted_mass_count=int((accepted & (accepted_mass == 0)).sum()),
                class_counts=np.bincount(y, minlength=8).tolist(),
                events={k: dict(brier_score=float(np.square(d[k]-v).mean()),
                                predicted=float(d[k].mean()), observed=float(v.mean()),
                                reliability=reliability(d[k], v)) for k, v in events.items()})


def grouped_evaluation(p, y, groups):
    p, y = probabilities(p), labels(y)
    groups = np.asarray(groups)
    if groups.shape != y.shape:
        raise ValueError('Group count mismatch')
    return {str(g): evaluate(p[groups == g], y[groups == g]) for g in sorted(set(groups))}


def paired_cluster_interval(candidate, baseline, y, clusters, repeats=1000, seed=42):
    """Positive gain = lower log loss. Whole spatial groups resampled together.

    Two clusters do not justify a population claim; explicitly mark intervals
    exploratory below ten independent clusters rather than counting frames.
    """
    candidate, baseline, y = probabilities(candidate), probabilities(baseline), labels(y)
    clusters = np.asarray(clusters)
    if len(candidate) != len(y) or baseline.shape != candidate.shape or clusters.shape != y.shape:
        raise ValueError('Paired interval membership mismatch')
    unique = sorted(set(clusters))
    if len(unique) < 2:
        return dict(cluster_count=len(unique), interval=None, exploratory=True)
    gains = np.log(np.clip(candidate[np.arange(len(y)), y], 1e-12, 1))-np.log(np.clip(baseline[np.arange(len(y)), y], 1e-12, 1))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        indices = np.concatenate([np.flatnonzero(clusters == g) for g in rng.choice(unique, len(unique), replace=True)])
        draws.append(float(gains[indices].mean()))
    return dict(cluster_count=len(unique), log_loss_gain=float(gains.mean()),
                interval=np.quantile(draws, [.025, .975]).tolist(), exploratory=len(unique) < 10)


def quantile_interval(p, level=.5):
    """Accepted delta quantile identified only to a bin; None = infinite tail."""
    if not 0 < level < 1:
        raise ValueError('Quantile must be in (0,1)')
    edges = [None, -20, -5, -1, 1, 5, 20, None]
    result = []
    for row in probabilities(p):
        mass = row[1:].sum()
        if mass == 0:
            result.append(None)
            continue
        i = min(int(np.searchsorted(np.cumsum(row[1:]/mass), level)), 6)
        result.append(dict(lower_m=edges[i], upper_m=edges[i+1]))
    return result

"""Exact empirical CRPS/energy scores; unresolved tail mass is never fabricated."""
import numpy as np


def weighted_quantile(values, weights, probability):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if not 0 <= probability <= 1 or weights.sum() <= 0:
        raise ValueError('Valid quantile and positive mass required')
    order = np.argsort(values, kind='stable')
    index = min(np.searchsorted(np.cumsum(weights[order])/weights.sum(), probability), len(order)-1)
    return float(values[order[index]])


def radial_crps(values, weights, observed):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    weights = weights/weights.sum()
    order = np.argsort(values)
    v, w = values[order], weights[order]
    # Half of E|X-X'|, in O(n log n).
    pair_half = np.sum(w*(v*(np.cumsum(w)-w)-(np.cumsum(w*v)-w*v)))
    return float(np.sum(weights*np.abs(values-observed))-pair_half)


def energy_score(vectors, weights, observed):
    vectors, weights = np.asarray(vectors, float), np.asarray(weights, float)
    mask = weights > 0
    vectors, weights = vectors[mask], weights[mask]/weights[mask].sum()
    pair = np.linalg.norm(vectors[:, None]-vectors[None, :], axis=2)
    return float(weights@np.linalg.norm(vectors-observed, axis=1)-.5*(weights@pair@weights))


def evaluate_distribution(joint, outcomes):
    bank = joint['bank']
    classes = np.asarray(bank['class_index'])
    vectors = np.asarray(bank['error_after_xy_m'], float)
    norms = np.linalg.norm(vectors, axis=1)
    pair = np.linalg.norm(vectors[:, None]-vectors[None, :], axis=2)
    results = {}
    for mode in ('full_attempt', 'accepted_only'):
        crps, energy, cover50, cover90, unknown = [], [], [], [], []
        conditional, complete, normalized = [], [], []
        for i, outcome in enumerate(outcomes):
            if mode == 'accepted_only' and not outcome['accepted']:
                continue
            w = joint['weights'][i].copy()
            unresolved = float(joint['unresolved_mass'][i])
            if mode == 'accepted_only':
                w[classes == 0] = 0
                acceptance = float(joint['probabilities'][i, 1:].sum())
                unresolved_accepted = float(joint['probabilities'][i,
                    [c for c in joint['missing_classes'] if c > 0]].sum())
                unresolved = unresolved_accepted/acceptance if acceptance > 0 else 1.
            unknown.append(unresolved)
            if w.sum() <= 0:
                continue
            w /= w.sum()
            observed = np.asarray(outcome['error_after_xy_m'])
            radius = float(np.linalg.norm(observed))
            conditional.append((radial_crps(norms, w, radius),
                float(w@np.linalg.norm(vectors-observed, axis=1)),
                bool(radius <= weighted_quantile(norms, w, .5)),
                bool(radius <= weighted_quantile(norms, w, .9))))
            normalized.append(w)
            complete.append(unresolved == 0.)
        if normalized:
            weights = np.asarray(normalized)
            pair_half = .5*np.einsum('ij,ij->i', weights@pair, weights)
            conditional = [(c, e-half, q50, q90) for (c, e, q50, q90), half in zip(conditional, pair_half)]
            for values, is_complete in zip(conditional, complete):
                if is_complete:
                    c, e, q50, q90 = values
                    crps.append(c)
                    energy.append(e)
                    cover50.append(q50)
                    cover90.append(q90)
        resolved = dict(count=len(conditional),
            crps_m=float(np.mean([v[0] for v in conditional])) if conditional else None,
            energy_score_m=float(np.mean([v[1] for v in conditional])) if conditional else None,
            upper_50_coverage=float(np.mean([v[2] for v in conditional])) if conditional else None,
            upper_90_coverage=float(np.mean([v[3] for v in conditional])) if conditional else None,
            interpretation='Diagnostic after conditioning on resolved empirical mass; not a score of the complete outcome law')
        results[mode] = dict(count=len(unknown), scored_count=len(crps),
            unresolved_count=int(sum(v > 0 for v in unknown)),
            mean_unresolved_mass=float(np.mean(unknown)) if unknown else None,
            crps_m=float(np.mean(crps)) if crps else None,
            energy_score_m=float(np.mean(energy)) if energy else None,
            upper_50_coverage=float(np.mean(cover50)) if cover50 else None,
            upper_90_coverage=float(np.mean(cover90)) if cover90 else None,
            resolved_component_diagnostic=resolved,
            coverage_note='One-sided radial bounds; no claim outside the empirical training support')
    return results

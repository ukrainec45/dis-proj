"""One-correction budget on observed candidate sets, not counterfactual flights."""
from collections import defaultdict

import numpy as np

from .contracts import probabilities, derived


def compare_opportunities(rows, candidate_p, geometry_p, navigation_scores=None):
    """Rows are aligned final-test attempts. opportunity_group must be preregistered.

    Keep scenario, sigma and trial separate; do not treat synthetic independent
    priors on successive frames as a trajectory. Missing measured nav scores
    are reported as unavailable, never replaced with a made-up texture proxy.
    """
    p, geo = probabilities(candidate_p), probabilities(geometry_p)
    if len(rows) != len(p) or geo.shape != p.shape:
        raise ValueError('Opportunity prediction row mismatch')
    if any(r['split'] != 'test' or r['features'] is None or r['label']['technical_error'] for r in rows):
        raise ValueError('Only supported, nontechnical final-test rows are allowed')
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if not r.get('opportunity_group'):
            raise ValueError('Pre-register opportunity_group before viewing outcomes')
        groups[(r['opportunity_group'], r['condition'], r['sigma_xy_m'], r['trial'])].append(i)
    scores = {'candidate': derived(p)['utility'], 'geometry': derived(geo)['utility']}
    if navigation_scores is not None:
        nav = np.asarray(navigation_scores, float)
        if nav.shape != (len(rows),) or not np.isfinite(nav).all():
            raise ValueError('Aligned finite map-only navigation scores required')
        scores['navigation_metric'] = nav
    empirical = np.array([1 if r['label']['class_index'] >= 5 else
                          -1 if 1 <= r['label']['class_index'] <= 3 else 0 for r in rows])
    records = []
    for group, indices in sorted(groups.items()):
        if len(indices) < 2:
            continue
        oracle = int(max(empirical[indices]))
        for method, values in scores.items():
            chosen = min(indices, key=lambda i: (-values[i], rows[i]['attempt_id']))
            label = rows[chosen]['label']
            records.append(dict(group=list(group), method=method, attempt_id=rows[chosen]['attempt_id'],
                utility=int(empirical[chosen]), delta_m=label['delta_m'],
                error_after_m=label['error_after_m'], regret=oracle-int(empirical[chosen]),
                severe_worsen=label['class_index'] == 1))
        records.append(dict(group=list(group), method='random_expected',
            utility=float(empirical[indices].mean()),
            delta_m=float(np.mean([rows[i]['label']['delta_m'] for i in indices])),
            regret=float(oracle-empirical[indices].mean()),
            severe_worsen=float(np.mean([rows[i]['label']['class_index'] == 1 for i in indices]))))
    if not records:
        raise ValueError('Need at least two observed opportunities per preregistered group')
    summary = {}
    for method in sorted({r['method'] for r in records}):
        selected = [r for r in records if r['method'] == method]
        summary[method] = dict(groups=len(selected), mean_utility=float(np.mean([r['utility'] for r in selected])),
            mean_delta_m=float(np.mean([r['delta_m'] for r in selected])),
            mean_regret=float(np.mean([r['regret'] for r in selected])),
            severe_worsen_rate=float(np.mean([r['severe_worsen'] for r in selected])))
    return dict(summary=summary, selections=records, navigation_control_available=navigation_scores is not None,
                route_arrival_or_safety_validated=False)


def evaluate_opportunities(run_dir, dataset_path, evaluated_dir, output, navigation_scores=None):
    """Join frozen held-out probabilities by ID; never assume JSON row order."""
    from pathlib import Path
    from .artifacts import load_verified, save_verified, assert_frozen_environment
    from .dataset import load_dataset, dataset_id, usable
    run = load_verified(Path(run_dir)/'frozen_run.json')
    assert_frozen_environment(run)
    data = load_dataset(dataset_path)
    if dataset_id(data) != run['dataset_id']:
        raise ValueError('Opportunity dataset differs from frozen run')
    predictions = load_verified(Path(evaluated_dir)/'predictions.json')
    rows = usable(data['rows'], 'test')
    def align(key):
        saved = predictions[key]
        by_id = {r['attempt_id']: r['probabilities'] for r in saved}
        if len(by_id) != len(saved) or set(by_id) != {r['attempt_id'] for r in rows}:
            raise ValueError('Opportunity prediction membership mismatch')
        return [by_id[r['attempt_id']] for r in rows]
    nav = None
    if navigation_scores is not None:
        nav = [navigation_scores[r['attempt_id']] for r in rows]
    result = compare_opportunities(rows, align(run['selected']), align(run['geometry_control']), nav)
    result.update(selected_before_test=run['selected'], geometry_control=run['geometry_control'])
    save_verified(output, result)
    return result

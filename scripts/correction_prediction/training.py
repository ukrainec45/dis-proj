"""Working-scene development and a separate immutable final-test entry point."""
from pathlib import Path

import numpy as np

from .artifacts import manifest, digest, load_verified, save_verified, write_once, assert_frozen_environment
from .contracts import PROTOCOL_ID, CLASS_NAMES
from .dataset import load_dataset, validate_rows, usable, dataset_id
from .features import ALL_FEATURES
from .models import CANDIDATES, FrozenModel, fit_model, fit_temperature, temperature_scale
from .metrics import evaluate, grouped_evaluation, paired_cluster_interval, quantile_interval


def arrays(rows):
    return [r['features'] for r in rows], np.array([r['label']['class_index'] for r in rows], dtype=int)


def aligned_predictions(path, rows, identity):
    saved = load_verified(path)
    if saved['dataset_id'] != identity:
        raise ValueError('External prediction dataset differs')
    by_id = {r['attempt_id']: r['probabilities'] for r in saved['predictions']}
    if len(by_id) != len(saved['predictions']) or not {r['attempt_id'] for r in rows} <= set(by_id):
        raise ValueError('External predictions missing/duplicated; do not drop failed calls')
    from .contracts import probabilities
    return probabilities([by_id[r['attempt_id']] for r in rows]), saved


def run_training(dataset_path, output, *, jev_predictions=None):
    """No test scores are computed here. Pilot-only datasets are rejected."""
    data = load_dataset(dataset_path)
    validate_rows(data['rows'], require_main=True)
    if data['provenance']['stage'] != 'main' or not data['provenance'].get('main_design'):
        raise ValueError('Agree and record the main campaign design before training')
    identity = dataset_id(data)
    splits = {k: usable(data['rows'], k) for k in ('train', 'validation', 'calibration')}
    if any(not r for r in splits.values()):
        raise ValueError('Each working split needs supported, nontechnical examples')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new training output directory')
    xtrain, ytrain = arrays(splits['train'])
    xval, yval = arrays(splits['validation'])
    xcal, ycal = arrays(splits['calibration'])
    results, frozen, val_predictions = {}, {}, {}
    for name in CANDIDATES:
        model = fit_model(name, xtrain, ytrain)
        model.artifact['dataset_id'] = identity
        pval, pcal = model.predict(xval, False), model.predict(xcal, False)
        model.artifact['temperature'] = fit_temperature(pcal, ycal)
        frozen[name] = model
        for variant, p in [('raw', pval), ('calibrated', model.predict(xval))]:
            key = name+':'+variant
            results[key] = evaluate(p, yval)
            results[key]['outside_training_ranges'] = int(model.support(xval).sum())
            val_predictions[key] = p
    external = {}
    for name, path in (jev_predictions or {}).items():
        if name not in ('jev_numeric', 'jev_map'):
            raise ValueError('Unknown external candidate')
        combined, saved = aligned_predictions(path, splits['validation']+splits['calibration'], identity)
        if saved['include_test']:
            raise ValueError('Working prediction file must not contain final-test predictions')
        if saved['with_image'] != (name == 'jev_map'):
            raise ValueError('Jev variant mismatch')
        pval, pcal = combined[:len(yval)], combined[len(yval):]
        t = fit_temperature(pcal, ycal)
        bounds = np.array([[r[k] for k in ALL_FEATURES] for r in xtrain])
        external[name] = dict(dataset_id=identity, protocol_id=PROTOCOL_ID,
            class_names=list(CLASS_NAMES), with_image=saved['with_image'], model=saved['model'],
            prompt_version=saved['prompt_version'], temperature=t, features=list(ALL_FEATURES),
            minimum=bounds.min(axis=0).tolist(), maximum=bounds.max(axis=0).tolist(),
            localizer_settings=data['provenance']['localizer_settings'])
        for variant, p in [('raw', pval), ('calibrated', temperature_scale(pval, t))]:
            key = name+':'+variant
            results[key] = evaluate(p, yval)
            val_predictions[key] = p
    # Stable insertion order is the declared complexity tie-breaker.
    selected = min(results, key=lambda k: results[k]['log_loss'])
    frozen_manifest = manifest(dataset_id=identity, selected=selected,
                               geometry_control=min((k for k in results if '_geometry:' in k),
                                                    key=lambda k: results[k]['log_loss']),
                               candidate_order=list(results), evaluated_final_test=False,
                               external_candidates=list(external),
                               model_ids={**{k: v.model_id for k, v in frozen.items()},
                                          **{k: digest(v) for k, v in external.items()}},
                               map_ids=[digest(v) for v in data['provenance']['references'].values()],
                               localizer_settings=data['provenance']['localizer_settings'],
                               excluded_counts={s: dict(total=sum(r['split'] == s for r in data['rows']),
                                   usable=len(usable(data['rows'], s))) for s in ('train', 'validation', 'calibration', 'test')})
    # Reserve the directory only after all fits and external checks have succeeded.
    output.mkdir(parents=True)
    for name, model in frozen.items():
        model.save(output/'models'/(name+'.json'))
    for name, a in external.items():
        save_verified(output/'models'/(name+'.json'), a)
    predictions = {name: [dict(attempt_id=r['attempt_id'], probabilities=p.tolist())
                         for r, p in zip(splits['validation'], values)]
                   for name, values in val_predictions.items()}
    save_verified(output/'validation_predictions.json', predictions)
    save_verified(output/'frozen_run.json', frozen_manifest)
    write_once(output/'summary.json', dict(selected=selected, validation=results,
        validation_passed=False, final_test_evaluated=False,
        warnings=['Research only; probabilities are not a route-arrival or safety model.']))
    from .reporting import write_score_report, bin_sensitivity
    write_score_report(output, results, title='Working validation', selected=selected)
    write_once(output/'bin_sensitivity.json', {k: bin_sensitivity(p, splits['validation'])
                                             for k, p in val_predictions.items()})
    return dict(selected=selected, validation=results)


def evaluate_frozen(run_dir, dataset_path, output, *, jev_predictions=None):
    """Explicit one-shot held-out evaluation. Never refits or rewrites a model."""
    run = load_verified(Path(run_dir)/'frozen_run.json')
    assert_frozen_environment(run)
    data = load_dataset(dataset_path)
    if dataset_id(data) != run['dataset_id']:
        raise ValueError('Final-test dataset differs from frozen development run')
    rows = usable(data['rows'], 'test')
    if not rows:
        raise ValueError('No supported final-test examples')
    x, y = arrays(rows)
    all_p, results = {}, {}
    for name in CANDIDATES:
        model = FrozenModel.load(Path(run_dir)/'models'/(name+'.json'))
        if model.model_id != run['model_ids'][name]:
            raise ValueError('Model changed after freeze')
        for variant, calibrated in [('raw', False), ('calibrated', True)]:
            all_p[name+':'+variant] = model.predict(x, calibrated)
    for name in run['external_candidates']:
        if name not in (jev_predictions or {}):
            raise ValueError('Missing frozen Jev final-test predictions')
        p, saved = aligned_predictions(jev_predictions[name], rows, run['dataset_id'])
        a = load_verified(Path(run_dir)/'models'/(name+'.json'))
        if digest(a) != run['model_ids'][name]:
            raise ValueError('Jev calibration changed after freeze')
        if not saved['include_test'] or saved['model'] != a['model'] or saved['prompt_version'] != a['prompt_version'] or saved['with_image'] != a['with_image']:
            raise ValueError('Jev final-test protocol differs from frozen candidate')
        all_p[name+':raw'] = p
        all_p[name+':calibrated'] = temperature_scale(p, a['temperature'])
    for name, p in all_p.items():
        results[name] = dict(overall=evaluate(p, y),
            by_scene=grouped_evaluation(p, y, [r['scene_id'] for r in rows]),
            by_condition=grouped_evaluation(p, y, [r['condition'] for r in rows]),
            by_sigma=grouped_evaluation(p, y, [str(r['sigma_xy_m']) for r in rows]),
            paired_gain_vs_constant=paired_cluster_interval(p, all_p['constant:raw'], y,
                                                          [r['spatial_group'] for r in rows]))
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new final-test output directory')
    output.mkdir(parents=True)
    save_verified(output/'predictions.json', {k: [dict(attempt_id=r['attempt_id'], probabilities=p.tolist(),
        accepted_median_bin=quantile_interval([p])[0]) for r, p in zip(rows, values)] for k, values in all_p.items()})
    write_once(output/'summary.json', dict(selected_before_test=run['selected'], results=results,
        final_test_evaluated=True, validation_passed=False,
        total_test_attempts=sum(r['split'] == 'test' for r in data['rows']), usable_test_attempts=len(rows),
        reason='Readiness needs scientific review of prediction AND practical utility; no automatic certification.'))
    from .reporting import write_score_report, bin_sensitivity
    write_score_report(output, results, title='Frozen final test', selected=run['selected'], final_test=True)
    write_once(output/'bin_sensitivity.json', {k: bin_sensitivity(p, rows) for k, p in all_p.items()})
    return results

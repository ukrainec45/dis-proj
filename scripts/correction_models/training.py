"""Freeze development choices before one-shot held-territory evaluation."""
from pathlib import Path

import numpy as np

from .artifacts import digest, manifest, save_verified, load_verified, write_once, assert_frozen_environment
from .contracts import PROTOCOL_ID
from .dataset import load_dataset, usable, model_inputs
from .models import CANDIDATES, fit_model, FrozenModel, fit_temperature
from .classification_metrics import evaluate, grouped_evaluation, paired_cluster_interval
from .distribution_metrics import evaluate_distribution


def _report(output, summary, title):
    lines = [f'# {title}', '', 'Research only. Delivery probability and Jetson latency are not validated.', '',
             '| Candidate | Log loss | Brier | Outside marginal ranges |', '|---|---:|---:|---:|']
    for name, score in summary['classification'].items():
        lines.append(f"| {name} | {score['log_loss']:.6f} | {score['brier_score']:.6f} | {score['outside_training_ranges']} |")
    lines += ['', 'Selected on validation: '+summary['selected'], '',
        'Distribution scores, unresolved tail mass, calibration counts and grouped metrics are in summary.json.',
        'Four held-out dataset scenes provide a limited generalization test; rare harms remain uncertain.',
        'Geographic non-overlap is not independently verified; global coordinates are sanitized.']
    path = Path(output)/'report.md'
    with path.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')


def _group_reports(p, y, rows):
    return {key: grouped_evaluation(p, y, [r[key] for r in rows])
            for key in ('scene_id', 'condition', 'sigma_xy_m')}


def run_training(dataset_path, output):
    data = load_dataset(dataset_path)
    root, output = Path(dataset_path).parent, Path(output)
    if output.exists():
        raise FileExistsError('Use a new training directory; frozen runs are immutable')
    if data['provenance']['stage'] != 'main' or data['provenance']['protocol_id'] != PROTOCOL_ID:
        raise ValueError('Main protocol required')
    splits = {s: usable(data['rows'], s) for s in ('train', 'validation', 'calibration')}
    if any(not v for v in splits.values()):
        raise ValueError('Every development split needs supported examples')
    cache = {}
    inputs = {s: model_inputs(rows, root, cache) for s, rows in splits.items()}
    labels = {s: np.asarray([r['label']['class_index'] for r in rows], int) for s, rows in splits.items()}
    outcomes = [r['label'] for r in splits['train']]
    models, scores, predictions = {}, {}, {}
    for name in CANDIDATES:
        model = fit_model(name, inputs['train'], outcomes)
        model.artifact.update(dataset_id=digest(data), encoder=data['provenance']['encoder'],
                              localizer_settings=data['provenance']['localizer_settings'])
        pcal = model.predict(inputs['calibration'], calibrated=False)
        model.artifact['temperature'] = fit_temperature(pcal, labels['calibration'])
        models[name] = model
        for variant, calibrated in (('raw', False), ('calibrated', True)):
            key = name+':'+variant
            p = model.predict(inputs['validation'], calibrated=calibrated)
            predictions[key] = p
            scores[key] = dict(evaluate(p, labels['validation']),
                               outside_training_ranges=int(model.support(inputs['validation']).sum()))
    # Stable order breaks exact ties towards simpler models and raw probabilities.
    selected = min(scores, key=lambda k: scores[k]['log_loss'])
    controls = {'constant': 'constant:raw'}
    for representation in ('geometry', 'map', 'camp'):
        controls[representation] = min((k for k in scores if '_'+representation in k and '_no_visibility' not in k),
                                      key=lambda k: scores[k]['log_loss'])
    distribution_keys = list(dict.fromkeys([selected, *controls.values()]))
    distribution = {}
    for key in distribution_keys:
        name, variant = key.split(':')
        distribution[key] = evaluate_distribution(models[name].predict_joint(inputs['validation'], variant == 'calibrated'),
                                                  [r['label'] for r in splits['validation']])
    frozen = manifest(dataset_id=digest(data), selected=selected, controls=controls,
        evaluation_scope='held_out_dataset_scenes', geographic_overlap_verified=False,
        candidates=list(models), distribution_candidates=distribution_keys,
        model_ids={k: m.model_id for k, m in models.items()},
        map_ids=[digest(v) for v in data['provenance']['references'].values()],
        encoder=data['provenance']['encoder'], localizer_settings=data['provenance']['localizer_settings'],
        evaluated_final_test=False)
    summary = dict(selected=selected, classification=scores, distribution=distribution,
        grouped={k: _group_reports(predictions[k], labels['validation'], splits['validation']) for k in distribution_keys},
        split_counts={s: dict(total=sum(r['split'] == s for r in data['rows']), supported=len(usable(data['rows'], s)))
                      for s in ('train', 'validation', 'calibration', 'test')},
        research_only=True, validation_passed=False, final_test_evaluated=False,
        geographic_overlap_verified=False, evaluation_scope='held_out_dataset_scenes',
        warnings=['Missing training classes represent unresolved outcome mass, not zero risk.',
                  'Improved prediction does not imply improved localization or delivery.'])
    from .reporting import localizer_report
    summary['localizer_outcomes'] = localizer_report([
        r for r in data['rows'] if r['split'] == 'validation' and not r['label']['technical_error']])
    output.mkdir(parents=True)
    for name, model in models.items():
        model.save(output/'models'/(name+'.json'))
    save_verified(output/'frozen_run.json', frozen)
    save_verified(output/'validation_predictions.json', {k: v.tolist() for k, v in predictions.items()})
    write_once(output/'summary.json', summary)
    _report(output, summary, 'Development validation')
    from .reporting import write_figures
    write_figures(summary, output)
    return summary


def evaluate_frozen(run_dir, dataset_path, output):
    run_dir, output = Path(run_dir), Path(output)
    run = load_verified(run_dir/'frozen_run.json')
    assert_frozen_environment(run)
    if output.exists() or (run_dir/'final_test_receipt.json').exists():
        raise FileExistsError('Final test already reserved/evaluated; do not tune on its results')
    data = load_dataset(dataset_path)
    if digest(data) != run['dataset_id']:
        raise ValueError('Dataset differs from frozen development run')
    rows = usable(data['rows'], 'test')
    if not rows:
        raise ValueError('No supported final test examples')
    inputs = model_inputs(rows, Path(dataset_path).parent)
    y = np.asarray([r['label']['class_index'] for r in rows], int)
    models = {name: FrozenModel.load(run_dir/'models'/(name+'.json')) for name in run['candidates']}
    if any(m.model_id != run['model_ids'][name] for name, m in models.items()):
        raise ValueError('Frozen model changed')
    # Reserve ON THE RUN, so choosing a different output cannot re-open final test.
    save_verified(run_dir/'final_test_receipt.json', dict(dataset_id=run['dataset_id'], output=str(output.resolve()),
                                                       selected=run['selected']))
    output.mkdir(parents=True)
    scores, predictions, distribution = {}, {}, {}
    for name, model in models.items():
        for variant, calibrated in (('raw', False), ('calibrated', True)):
            key = name+':'+variant
            joint = model.predict_joint(inputs, calibrated)
            p = joint['probabilities']
            predictions[key] = p
            scores[key] = dict(evaluate(p, y), outside_training_ranges=int(model.support(inputs).sum()))
            if key in run['distribution_candidates']:
                distribution[key] = evaluate_distribution(joint, [r['label'] for r in rows])
    comparisons = {key: paired_cluster_interval(predictions[key], predictions['constant:raw'], y,
        [r['spatial_group'] for r in rows]) for key in run['distribution_candidates']}
    summary = dict(selected=run['selected'], classification=scores, distribution=distribution,
        grouped={k: _group_reports(predictions[k], y, rows) for k in run['distribution_candidates']},
        comparison_to_constant=comparisons, supported_count=len(rows),
        excluded_count=sum(r['split'] == 'test' for r in data['rows'])-len(rows),
        final_test_evaluated=True, research_only=True, validation_passed=False,
        flight_validation_passed=False, held_out_test_scenes=len({r['scene_id'] for r in rows}),
        geographic_overlap_verified=False, evaluation_scope='held_out_dataset_scenes')
    from .reporting import localizer_report
    summary['localizer_outcomes'] = localizer_report([
        r for r in data['rows'] if r['split'] == 'test' and not r['label']['technical_error']])
    save_verified(output/'predictions.json', dict(attempt_ids=[r['attempt_id'] for r in rows],
                                               predictions={k: v.tolist() for k, v in predictions.items()}))
    write_once(output/'summary.json', summary)
    _report(output, summary, 'Frozen held-out-scene test')
    from .reporting import write_figures
    write_figures(summary, output)
    return summary

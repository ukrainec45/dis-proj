"""Exploratory pilot baselines, separate from the gated main training protocol.

Whole spatial groups are held out. Repetitions of a frame are never split.
No Jev calls, image matching, GPU work, temperature fitting or final test.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import manifest, save_verified, write_once
from .contracts import CLASS_NAMES
from .dataset import load_dataset, dataset_id
from .metrics import evaluate, grouped_evaluation, paired_cluster_interval, quantile_interval
from .models import CANDIDATES, fit_model
from .reporting import bin_sensitivity, write_score_report


def pilot_folds(rows):
    """Hold out a scene or a combined group of overlapping scenes as a unit."""
    scenes, frames = defaultdict(set), defaultdict(set)
    for row in rows:
        scenes[row['scene_id']].add(row['spatial_group'])
        frames[(row['scene_id'], row['sample_id'])].add(row['spatial_group'])
    if any(len(v) != 1 for v in list(scenes.values())+list(frames.values())):
        raise ValueError('Keep each scene and all frame repetitions in one spatial group')
    groups = sorted({r['spatial_group'] for r in rows})
    if len(groups) < 2:
        raise ValueError('Need at least two spatial groups for exploratory held-out checks')
    folds = []
    for group in groups:
        train = [i for i, r in enumerate(rows) if r['spatial_group'] != group]
        held = [i for i, r in enumerate(rows) if r['spatial_group'] == group]
        folds.append((group, train, held))
    return folds


def run_pilot_baselines(dataset_path, output):
    """Fit fixed baselines; save OOF checks and all-data exploratory refits.

    This entry point does not turn pilot rows into main/train/test rows and
    never creates a frozen_run.json accepted by the planning model loader.
    """
    data = load_dataset(dataset_path)
    if data['provenance'].get('stage') != 'pilot' or any(r['split'] != 'pilot' for r in data['rows']):
        raise ValueError('This entry point accepts original pilot rows only; use train for the main protocol')
    rows = [r for r in data['rows'] if r['features'] is not None and not r['label']['technical_error']]
    if not rows:
        raise ValueError('No supported, nontechnical pilot attempts; diagnose the campaign first')
    rows.sort(key=lambda r: r['attempt_id'])
    folds = pilot_folds(rows)
    output = Path(output).resolve()
    source = Path(dataset_path).resolve()
    if output == source.parent or output in source.parents or output.is_relative_to(source.parent):
        raise ValueError('Use a sibling output directory, separate from the source pilot campaign')
    if output.exists():
        raise FileExistsError('Use a new pilot-baseline output directory; existing results are immutable')
    identity = dataset_id(data)
    features = [r['features'] for r in rows]
    labels = np.array([r['label']['class_index'] for r in rows], dtype=int)
    # Models only see this feature allowlist; row IDs, GT, errors and scenes are not features.
    predictions = {name: np.full((len(rows), 8), np.nan) for name in CANDIDATES}
    fold_results, fold_models = [], {}
    for fold_number, (group, train_indices, held_indices) in enumerate(folds):
        xtrain = [features[i] for i in train_indices]
        xheld = [features[i] for i in held_indices]
        ytrain, yheld = labels[train_indices], labels[held_indices]
        fold_id = f'fold-{fold_number+1:02d}'
        print(f'{fold_id}: held out {group}; train={len(train_indices)}, held={len(held_indices)}', flush=True)
        record = dict(fold_id=fold_id, held_out_group=group,
            train_scenes=sorted({rows[i]['scene_id'] for i in train_indices}),
            held_out_scenes=sorted({rows[i]['scene_id'] for i in held_indices}),
            train_attempt_ids=[rows[i]['attempt_id'] for i in train_indices],
            held_out_attempt_ids=[rows[i]['attempt_id'] for i in held_indices],
            class_counts_train=np.bincount(ytrain, minlength=8).tolist(), models={})
        fold_models[fold_id] = {}
        for name in CANDIDATES:
            model = fit_model(name, xtrain, ytrain)
            model.artifact.update(dataset_id=identity, exploratory_pilot=True,
                                  validation_passed=False, fold_id=fold_id,
                                  held_out_group=group, training_groups=[g for g, _, _ in folds if g != group])
            p = model.predict(xheld, calibrated=False)
            predictions[name][held_indices] = p
            fold_models[fold_id][name] = model
            record['models'][name] = dict(metrics=evaluate(p, yheld),
                missing_training_classes=[CLASS_NAMES[i] for i in model.artifact['missing_classes']],
                outside_training_ranges=int(model.support(xheld).sum()))
        fold_results.append(record)
    evaluations = {}
    groups = [r['spatial_group'] for r in rows]
    for name, p in predictions.items():
        by_group = grouped_evaluation(p, labels, groups)
        evaluations[name] = dict(overall=evaluate(p, labels), by_spatial_group=by_group,
            by_scene=grouped_evaluation(p, labels, [r['scene_id'] for r in rows]),
            by_condition=grouped_evaluation(p, labels, [r['condition'] for r in rows]),
            by_sigma=grouped_evaluation(p, labels, [str(r['sigma_xy_m']) for r in rows]),
            scene_macro_log_loss=float(np.mean([s['log_loss'] for s in by_group.values()])),
            paired_gain_vs_constant=paired_cluster_interval(p, predictions['constant'], labels, groups))
    # Refit every fixed candidate on the full pilot, never present this as an independent evaluation.
    refits = {}
    print('Refitting all five fixed candidates on the full pilot; research use only.', flush=True)
    for name in CANDIDATES:
        model = fit_model(name, features, labels)
        model.artifact.update(dataset_id=identity, exploratory_pilot=True, validation_passed=False,
            final_test_evaluated=False, training_unique_frames=len({(r['scene_id'], r['sample_id']) for r in rows}))
        refits[name] = model
    counts = np.bincount(labels, minlength=8)
    warnings = [
        'Exploratory pilot only: dependent repetitions and very few scenes.',
        'Spatial independence between different scene IDs has not been established by this diagnostic.',
        'Probabilities are uncalibrated. No model was selected on an independent test.',
        'The all-data refits were not evaluated on new data and are not ready for planner use.']
    harm_frames = {(r['scene_id'], r['sample_id']) for r in rows if 1 <= r['label']['class_index'] <= 3}
    warnings.append(f'Harm evidence: {int(counts[1:4].sum())} worsening attempts in {len(harm_frames)} '
                    'distinct frames; dependent repetitions do not establish rare-event reliability.')
    missing = [CLASS_NAMES[i] for i in np.flatnonzero(counts == 0)]
    if missing:
        warnings.append('Classes absent from the entire usable pilot: '+', '.join(missing))
    summary = dict(exploratory_pilot=True, validation_passed=False, final_test_evaluated=False,
        calibration_fitted=False, model_selected=False, jev_evaluated=False,
        total_attempts=len(data['rows']), usable_attempts=len(rows),
        excluded_technical=sum(r['label']['technical_error'] for r in data['rows']),
        excluded_unsupported=sum(r['features'] is None for r in data['rows']),
        unique_frames=len({(r['scene_id'], r['sample_id']) for r in rows}),
        spatial_groups=len(folds), outcomes=dict(zip(CLASS_NAMES, counts.tolist())),
        evaluations=evaluations, warnings=warnings)
    # Writes happen only after data checks and fits succeed. Never overwrite the input campaign.
    output.mkdir(parents=True)
    for fold_id, models in fold_models.items():
        for name, model in models.items():
            model.save(output/'fold_models'/fold_id/(name+'.json'))
    for name, model in refits.items():
        model.save(output/'exploratory_models'/(name+'.json'))
    save_verified(output/'run.json', manifest(dataset_id=identity, method='leave-one-spatial-group-out',
        source_dataset=str(source), exploratory_pilot=True, calibration_fitted=False,
        final_test_evaluated=False, model_ids={k: m.model_id for k, m in refits.items()}))
    save_verified(output/'folds.json', fold_results)
    save_verified(output/'oof_predictions.json', {name: [dict(attempt_id=r['attempt_id'],
        scene_id=r['scene_id'], sample_id=r['sample_id'], spatial_group=r['spatial_group'],
        probabilities=p.tolist(), accepted_median_bin=quantile_interval([p])[0])
        for r, p in zip(rows, values)] for name, values in predictions.items()})
    write_once(output/'summary.json', summary)
    write_once(output/'bin_sensitivity.json', {k: bin_sensitivity(p, rows) for k, p in predictions.items()})
    write_score_report(output, evaluations, title='Exploratory pilot baselines: held-out spatial groups',
        selected='none', selection_note='No model selected. No calibration or independent final test.',
        scope_note='OOF checks on a small, already inspected pilot. All-data refits are research-only.')
    # Keep the statistical limitations visible in the human-readable report as well.
    with (output/'report.md').open('a', encoding='utf-8') as stream:
        stream.write('\n'+'\n'.join('- '+w for w in warnings)+'\n')
    return summary

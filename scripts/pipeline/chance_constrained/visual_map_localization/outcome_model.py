"""First fixed-domain correct-fix probability models.

Real retained tables are fitted in hosted Colab. Local verification uses only
synthetic tables. This is not a planner transition model or calibrated R model.
"""
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

GEOMETRY = ('log_agl_m', 'map_coverage', 'log_search_to_footprint_ratio',
            'log_projection_condition', 'log_footprint_area_m2',
            'log_query_bbox_pixels', 'log_reference_crop_pixels')
MAP_INPUTS = GEOMETRY + ('log_reference_keypoint_density_mpx',)
MODEL_INPUTS = {'constant': (), 'geometry': GEOMETRY, 'geometry_and_map_features': MAP_INPUTS}
L2_STRENGTH = 1.0  # Fixed before validation: summed logistic loss + .5 * ||beta||^2.


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def _csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_tables(folder):
    """Validate the frozen predictor/label tables without reading source images."""
    folder = Path(folder)
    spec = _read(folder / 'feature_spec.json')
    if spec['target_threshold_m'] != 3 or spec['primary_target'] != 'accepted_and_horizontal_error_le_3m':
        raise ValueError('This protocol requires the predefined accepted-within-3m target')
    for key, name in [('features', 'features.csv'), ('labels', 'labels.csv')]:
        if _sha(folder / name) != spec['sha256'][key]:
            raise ValueError(f'Dataset checksum mismatch: {name}')
    for name, expected in MODEL_INPUTS.items():
        if tuple(spec['model_inputs'][name]) != expected:
            raise ValueError(f'Predictor allowlist changed: {name}')
    features, labels = _read_csv(folder / 'features.csv'), _read_csv(folder / 'labels.csv')
    by_id = {r['sample']: r for r in labels}
    if len(by_id) != len(labels) or len({r['sample'] for r in features}) != len(features):
        raise ValueError('Duplicate sample identifiers')
    if {r['sample'] for r in features} != set(by_id):
        raise ValueError('Feature/label sample membership differs')
    rows, outcomes = [], []
    for feature in features:
        label = by_id[feature['sample']]
        scene = feature['scene_id']
        expected_split = {'Scene_05': 'train', 'Scene_07': 'validation'}.get(scene)
        if (expected_split is None or feature['split'] != expected_split
                or label['split'] != expected_split or label['scene_id'] != scene
                or PurePosixPath(feature['sample']).parent.name != scene):
            raise ValueError('Only Scene_05 train and Scene_07 validation are allowed; no holdout rows')
        accepted, target = int(label['accepted']), int(label['target_success_3m'])
        error = float(label['error_m']) if label['error_m'] else None
        if accepted not in (0, 1) or target not in (0, 1):
            raise ValueError('Invalid binary label')
        if (accepted and (error is None or not np.isfinite(error) or error < 0)) or (not accepted and error is not None):
            raise ValueError('Acceptance/error label mismatch')
        if target != int(bool(accepted) and error <= 3):
            raise ValueError('Target does not agree with accepted XY error')
        expected_outcome = 'reject' if not accepted else 'accepted_within_3m' if target else 'accepted_over_3m'
        if label['outcome_3m'] != expected_outcome:
            raise ValueError('Outcome label mismatch')
        row = dict(sample=feature['sample'], scene_id=scene, split=expected_split)
        row.update({key: float(feature[key]) for key in MAP_INPUTS})
        if not all(np.isfinite(row[key]) for key in MAP_INPUTS):
            raise ValueError('Non-finite predictor')
        rows.append(row)
        outcomes.append(target)
    y = np.asarray(outcomes, dtype=int)
    train = np.array([r['split'] == 'train' for r in rows])
    if not train.any() or train.all() or len(np.unique(y[train])) != 2:
        raise ValueError('Both splits and two training classes are required')
    domain = spec.get('experiment_domain')
    if not domain or domain.get('prior_covariance_xy_m2') != [[400, 0], [0, 400]]:
        raise ValueError('Explicit sigma-20m experiment_domain metadata is required')
    return spec, rows, y, train


def _matrix(rows, names):
    X = np.array([[row[key] for key in names] for row in rows], dtype=float)
    if X.shape != (len(rows), len(names)) or not np.isfinite(X).all():
        raise ValueError('Expected a finite predictor matrix')
    return X


def fit_binary(rows, y, names):
    """Convex L2 logistic regression; training-only standardization."""
    y = np.asarray(y, dtype=float)
    if y.shape != (len(rows),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Two binary classes are required for fitting')
    X = _matrix(rows, names)
    base = dict(schema_version=1, feature_names=list(names), training_count=len(y),
                training_positive_count=int(y.sum()), l2_strength=L2_STRENGTH,
                probability_event='matcher accepted and horizontal error <= 3 m', target_threshold_m=3.0)
    prior = float(y.mean())
    intercept = float(np.log(prior/(1-prior)))
    if not names:
        return dict(base, model_type='constant', intercept=intercept, coefficients=[],
                    mean=[], scale=[], feature_min=[], feature_max=[], optimizer='empirical_training_frequency')
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale < 1e-12] = 1
    Z = (X-mean)/scale
    def objective(theta):
        scores = theta[0] + Z @ theta[1:]
        loss = float(np.sum(np.logaddexp(0, scores)-y*scores) + .5*L2_STRENGTH*np.dot(theta[1:], theta[1:]))
        residual = expit(scores)-y
        gradient = np.r_[residual.sum(), Z.T @ residual + L2_STRENGTH*theta[1:]]
        return loss, gradient
    initial = np.r_[intercept, np.zeros(len(names))]
    optimum = minimize(objective, initial, method='L-BFGS-B', jac=True,
                       options=dict(maxiter=2000, ftol=1e-12, gtol=1e-8))
    if not optimum.success or not np.isfinite(optimum.x).all():
        raise RuntimeError(f'Logistic optimization failed: {optimum.message}')
    return dict(base, model_type='l2_logistic', intercept=float(optimum.x[0]),
                coefficients=optimum.x[1:].tolist(), mean=mean.tolist(), scale=scale.tolist(),
                feature_min=X.min(axis=0).tolist(), feature_max=X.max(axis=0).tolist(),
                optimizer='scipy_L-BFGS-B', optimizer_iterations=int(optimum.nit),
                objective=float(optimum.fun))


def predict_model(model, rows):
    """Return point probabilities and a coarse per-feature range warning."""
    names = model['feature_names']
    if tuple(names) not in MODEL_INPUTS.values():
        raise ValueError('Unknown predictor list')
    X = _matrix(rows, names)
    if not names:
        return np.full(len(rows), expit(model['intercept'])), np.zeros(len(rows), dtype=bool)
    scale, mean, beta = (np.asarray(model[k], dtype=float) for k in ('scale', 'mean', 'coefficients'))
    if any(v.shape != (len(names),) for v in (scale, mean, beta)) or np.any(scale <= 0):
        raise ValueError('Invalid model dimensions or scale')
    p = expit(float(model['intercept']) + ((X-mean)/scale) @ beta)
    outside = ((X < np.asarray(model['feature_min'])-1e-12) |
               (X > np.asarray(model['feature_max'])+1e-12)).any(axis=1)
    return p, outside



def load_frozen_model(output, name=None):
    """Load a checksummed JSON model and retain its experimental domain metadata."""
    output = Path(output)
    manifest = _read(output / 'manifest.json')
    name = name or manifest['selected_by_validation']
    if name not in MODEL_INPUTS:
        raise ValueError('Unknown frozen model name')
    relative = 'frozen_models/' + name + '.json'
    path = output / relative
    if _sha(path) != manifest['artifact_sha256'][relative]:
        raise ValueError('Frozen model checksum mismatch')
    model = _read(path)
    if tuple(model['feature_names']) != MODEL_INPUTS[name] or model['target_threshold_m'] != 3:
        raise ValueError('Frozen model contract mismatch')
    model['experiment_domain'] = manifest['experiment_domain']
    model['deployment_status'] = manifest['deployment_status']
    return model

def probability_metrics(y, probabilities):
    y, p = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    if not np.isin(y, [0, 1]).all() or y.shape != p.shape or not len(y) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probability evaluation input')
    clipped = np.clip(p, 1e-15, 1-1e-15)
    positive, negative = int(y.sum()), int(len(y)-y.sum())
    auc = ((rankdata(p)[y == 1].sum()-positive*(positive+1)/2)/(positive*negative)
           if positive and negative else None)
    bins = []
    membership = np.minimum((p*5).astype(int), 4)
    for i in range(5):
        mask = membership == i
        bins.append(dict(bin_lower=i/5, bin_upper=(i+1)/5, count=int(mask.sum()),
                         mean_probability=float(p[mask].mean()) if mask.any() else None,
                         observed_success=float(y[mask].mean()) if mask.any() else None))
    return dict(sample_count=len(y), positive_count=positive, observed_success_rate=float(y.mean()),
                log_loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
                brier_score=float(np.mean((p-y)**2)), roc_auc=None if auc is None else float(auc),
                mean_probability=float(p.mean()), calibration_bins=bins)


def run_training(dataset_dir, output):
    """Train on Scene_05, evaluate Scene_07, freeze refits; never read Scene_21."""
    dataset_dir, output = Path(dataset_dir).resolve(), Path(output).resolve()
    if output.is_relative_to(dataset_dir) or dataset_dir.is_relative_to(output):
        raise ValueError('Training outputs and dataset must be disjoint')
    spec, rows, y, train_mask = load_tables(dataset_dir)
    identity = dict(schema_version=1, protocol='correct_fix_3m_logistic_v1',
                    dataset_sha256={name: _sha(dataset_dir / name) for name in ('features.csv', 'labels.csv', 'feature_spec.json')},
                    source_sha256=_sha(Path(__file__)), numpy=np.__version__, scipy=scipy.__version__,
                    l2_strength=L2_STRENGTH, model_inputs={k:list(v) for k,v in MODEL_INPUTS.items()},
                    primary_selection_metric='Scene_07_log_loss', train_scene='Scene_05', validation_scene='Scene_07',
                    final_test_scene='Scene_21', target_threshold_m=3.0)
    if (output / 'run.json').exists():
        if _read(output / 'run.json') != identity:
            raise ValueError('Training identity changed; use a new output directory')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty model output directory')
    output.mkdir(parents=True, exist_ok=True)
    _write(output / 'run.json', identity)
    train_rows = [r for r, mask in zip(rows, train_mask) if mask]
    validation_rows = [r for r, mask in zip(rows, train_mask) if not mask]
    train_y, validation_y = y[train_mask], y[~train_mask]
    evaluations, predictions, calibration, fitted = {}, [], [], {}
    for name, names in MODEL_INPUTS.items():
        model = fit_binary(train_rows, train_y, names)
        fitted[name] = model
        p, outside = predict_model(model, validation_rows)
        result = probability_metrics(validation_y, p)
        result['outside_training_feature_ranges_count'] = int(outside.sum())
        evaluations[name] = result
        _write(output / 'validation_models' / (name+'.json'), model)
        for row, target, probability, flag in zip(validation_rows, validation_y, p, outside):
            predictions.append(dict(sample=row['sample'], scene_id=row['scene_id'], model=name,
                                    target_success_3m=int(target), probability=float(probability), extrapolated=bool(flag)))
        calibration.extend(dict(model=name, **row) for row in result['calibration_bins'])
    # Fixed candidate order breaks equal-loss ties in favour of simpler models.
    selected = min(MODEL_INPUTS, key=lambda name:evaluations[name]['log_loss'])
    final = {}
    for name, names in MODEL_INPUTS.items():
        final[name] = fit_binary(rows, y, names)
        _write(output / 'frozen_models' / (name+'.json'), final[name])
    _csv(output / 'validation_predictions.csv', predictions,
         ['sample','scene_id','model','target_success_3m','probability','extrapolated'])
    _csv(output / 'calibration.csv', calibration,
         ['model','bin_lower','bin_upper','count','mean_probability','observed_success'])
    summary = dict(training_status='working_scene_evaluated_models_frozen', evaluations=evaluations,
                   selected_by_validation=selected, final_refit_count=len(rows),
                   final_test_evaluated=False, validation_passed=False, target_platform_calibrated=False,
                   trained_flight_risk_model=False,
                   map_feature_validation_log_loss_gain=evaluations['geometry']['log_loss']-evaluations['geometry_and_map_features']['log_loss'])
    _write(output / 'summary.json', summary)
    artifact_paths = list((output / 'validation_models').glob('*.json')) + list((output / 'frozen_models').glob('*.json'))
    model_manifest = dict(identity, selected_by_validation=selected, experiment_domain=spec['experiment_domain'],
                          feature_meanings=spec['input_meanings'], final_test_evaluated=False,
                          validation_passed=False, deployment_status='research_only',
                          supported_domain_warning='Feature min/max are coarse diagnostics, not calibrated support bounds.',
                          limitations=spec['limitations'] + [
                              'p_success_3m combines rejection and inaccurate acceptance in its complement.',
                              'Do not equate this probability to route arrival probability.',
                              'No model of VIO drift, corrected residual transitions, false-fix dynamics, or calibrated R.',
                              'Validation selects candidates; it is not an independent final test.',
                              'No IID frame bootstrap or broad cross-scene generalization claim.'],
                          artifact_sha256={str(p.relative_to(output)).replace('\\','/'):_sha(p) for p in artifact_paths})
    _write(output / 'manifest.json', model_manifest)
    _draw_calibration(evaluations, output)
    lines = ['# Correct-fix probability model training', '',
             'Target: matcher accepts a horizontal fix with dataset-relative error <=3 m.',
             'Training: Scene_05. Validation: Scene_07. Scene_21 has not been read or matched.',
             'Fixed L2 strength 1.0, summed logistic loss, unpenalized intercept, training-only standardization.',
             'Lower log loss and Brier score are better. AUC measures ranking, not calibration.', '',
             '| Model | Validation log loss | Brier score | ROC AUC | Out-of-range validation rows |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, result in evaluations.items():
        auc = 'unavailable' if result['roc_auc'] is None else f"{result['roc_auc']:.6f}"
        lines.append(f"| {name} | {result['log_loss']:.6f} | {result['brier_score']:.6f} | {auc} | {result['outside_training_feature_ranges_count']} |")
    lines += ['', f'Selected by validation log loss: {selected}. All three fixed candidates are refitted on both working scenes and frozen for final comparison.',
              f"Observed map-feature log-loss gain over geometry: {summary['map_feature_validation_log_loss_gain']:.6f}. This is a point comparison on one validation scene, not a statistical significance claim.", '',
              'Inspect calibration.png and calibration.csv: predicted probabilities should agree with observed frequencies. No automatic calibration-success certificate is inferred from small bins.',
              'Feature extraction uses supplied camera geometry and the reference map; actual query feature counts and all post-match diagnostics are excluded.',
              'Covariance is fixed at sigma=20 m per XY axis. Predicting a growing uncertainty along a route requires additional training conditions.',
              'This model can become an input to correction-opportunity evaluation after independent testing and state-transition integration. It is not connected to the planner and does not certify route safety.',
              'Final Scene_21 evaluation remains pending. Do not tune candidates using that scene.',
              'Model exports use JSON coefficients and standardization parameters, without pickle or a scikit-learn dependency.']
    (output / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def _draw_calibration(evaluations, output, title="Scene_07 validation calibration"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Ideal agreement')
    for name, result in evaluations.items():
        bins = [r for r in result['calibration_bins'] if r['count']]
        ax.plot([r['mean_probability'] for r in bins], [r['observed_success'] for r in bins], 'o-', label=name)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability', ylabel='Observed success frequency',
           title=title)
    ax.legend(fontsize=8)
    fig.savefig(Path(output) / 'calibration.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_reverse_diagnostic(dataset_dir, output):
    """Fit Scene_07 and evaluate Scene_05 in hosted Colab; no final refits.

    Source split labels retain their original provenance. They are reversed
    only in this diagnostic's in-memory masks and explicit run identity.
    This is development evidence after the forward results were observed.
    """
    dataset_dir, output = Path(dataset_dir).resolve(), Path(output).resolve()
    if output.is_relative_to(dataset_dir) or dataset_dir.is_relative_to(output):
        raise ValueError('Diagnostic outputs and dataset must be disjoint')
    spec, rows, y, original_train = load_tables(dataset_dir)
    train_mask = ~original_train
    if len(np.unique(y[train_mask])) != 2 or len(np.unique(y[~train_mask])) != 2:
        raise ValueError('Both working scenes must contain both outcome classes')
    identity = dict(schema_version=1, protocol='correct_fix_3m_reverse_diagnostic_v1',
                    dataset_sha256={name: _sha(dataset_dir/name) for name in ('features.csv', 'labels.csv', 'feature_spec.json')},
                    source_sha256=_sha(Path(__file__)), numpy=np.__version__, scipy=scipy.__version__,
                    l2_strength=L2_STRENGTH, model_inputs={k:list(v) for k,v in MODEL_INPUTS.items()},
                    train_scene='Scene_07', evaluation_scene='Scene_05', final_test_scene='Scene_21',
                    target_threshold_m=3.0, independent_final_test=False,
                    source_split_labels='Original forward split retained unchanged; masks reversed in memory')
    if (output/'run.json').exists():
        if _read(output/'run.json') != identity:
            raise ValueError('Diagnostic identity changed; use a new output directory')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty diagnostic output directory')
    output.mkdir(parents=True, exist_ok=True)
    _write(output/'run.json', identity)
    training = [r for r,t in zip(rows,train_mask) if t]
    evaluation = [r for r,t in zip(rows,train_mask) if not t]
    evaluations, predictions, calibration = {}, [], []
    for name, names in MODEL_INPUTS.items():
        model = fit_binary(training, y[train_mask], names)
        _write(output/'evaluation_models'/(name+'.json'), model)
        probabilities, outside = predict_model(model, evaluation)
        metrics = probability_metrics(y[~train_mask], probabilities)
        metrics['outside_training_feature_ranges_count'] = int(outside.sum())
        evaluations[name] = metrics
        for row, target, probability, flag in zip(evaluation,y[~train_mask],probabilities,outside):
            predictions.append(dict(sample=row['sample'],scene_id=row['scene_id'],model=name,
                                    target_success_3m=int(target),probability=float(probability),extrapolated=bool(flag)))
        calibration.extend(dict(model=name,**b) for b in metrics['calibration_bins'])
    _csv(output/'evaluation_predictions.csv',predictions,
         ['sample','scene_id','model','target_success_3m','probability','extrapolated'])
    _csv(output/'calibration.csv',calibration,
         ['model','bin_lower','bin_upper','count','mean_probability','observed_success'])
    summary = dict(training_status='reverse_working_scene_diagnostic_completed',evaluations=evaluations,
                   training_count=len(training),training_positive_count=int(y[train_mask].sum()),
                   evaluation_count=len(evaluation),evaluation_positive_count=int(y[~train_mask].sum()),
                   final_test_evaluated=False,validation_passed=False,independent_final_test=False,
                   selected_by_validation=None,final_refit_performed=False,
                   map_feature_evaluation_log_loss_gain=evaluations['geometry']['log_loss']-evaluations['geometry_and_map_features']['log_loss'])
    _write(output/'summary.json',summary)
    artifacts=list((output/'evaluation_models').glob('*.json'))
    manifest=dict(identity,experiment_domain=spec['experiment_domain'],deployment_status='diagnostic_only',
                  final_test_evaluated=False,validation_passed=False,
                  limitations=spec['limitations']+[
                      'Reverse direction was proposed after inspecting forward results; not a fresh independent test.',
                      'All candidates are reported; no deployment selection or pooled refit is performed.',
                      'Scene_21 outcomes are not read. Source tables and forward model outputs are not changed.'],
                  artifact_sha256={p.relative_to(output).as_posix():_sha(p) for p in artifacts})
    _write(output/'manifest.json',manifest)
    _draw_calibration(evaluations,output,title='Scene_05 reverse diagnostic calibration')
    lines=['# Reverse working-scene diagnostic','',
           'Training: Scene_07. Evaluation: Scene_05. Target: accepted fix with horizontal error <=3 m.',
           'The original source split labels are unchanged; the run identity records the reversed direction.',
           'Same predictors, L2 strength, target and optimizer as the forward experiment. Scaling uses Scene_07 only.',
           'This diagnostic was proposed after viewing forward results; neither working scene is an untouched final test.',
           'No candidate selection, pooled refit, planner integration or Scene_21 evaluation is performed.','',
           '| Model | Mean probability | Observed success | Log loss | Brier | AUC | Outside training ranges |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,m in evaluations.items():
        lines.append(f"| {name} | {m['mean_probability']:.6f} | {m['observed_success_rate']:.6f} | {m['log_loss']:.6f} | {m['brier_score']:.6f} | {m['roc_auc']:.6f} | {m['outside_training_feature_ranges_count']} |")
    lines += ['', 'Lower log loss and Brier are better. AUC measures ordering, not probability calibration.',
              'Compare both directions separately; nearby frames are correlated and scene sizes differ.',
              'A direction-dependent result can diagnose poor transfer but does not identify a single causal mechanism.',
              'The constant has no features, so zero range warnings do not demonstrate support.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary

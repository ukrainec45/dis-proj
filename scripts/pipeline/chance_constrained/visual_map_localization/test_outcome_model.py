"""Synthetic-only checks for outcome probability fitting and export."""
import csv
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from . import outcome_model as om


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def refresh_hashes(folder):
    p = folder / 'feature_spec.json'; s = json.loads(p.read_text())
    for key, name in [('features','features.csv'), ('labels','labels.csv')]:
        s['sha256'][key] = hashlib.sha256((folder/name).read_bytes()).hexdigest()
    p.write_text(json.dumps(s))


@pytest.fixture
def tables(tmp_path):
    folder = tmp_path / 'dataset'; folder.mkdir()
    rng = np.random.default_rng(914)
    x, labels = [], []
    for scene, split, count in [('Scene_05','train',240),('Scene_07','validation',120)]:
        for i in range(count):
            row = dict(sample=f'{scene}/L{scene[-2:]}_{i:04d}.npz', scene_id=scene, split=split)
            row.update({k:float(rng.normal()) for k in om.MAP_INPUTS})
            row['log_agl_m'] += 4.4 if split=='train' else 4.6
            probability = expit(3*row['log_reference_keypoint_density_mpx'])
            target = int(rng.random() < probability)
            accepted = int(target or i%2)
            error = (1.0 if target else 8.0) if accepted else ''
            label = dict(sample=row['sample'],scene_id=scene,split=split,accepted=accepted,
                         error_m=error,target_success_3m=target,
                         outcome_3m='reject' if not accepted else 'accepted_within_3m' if target else 'accepted_over_3m')
            x.append(row); labels.append(label)
    write_csv(folder/'features.csv',x);write_csv(folder/'labels.csv',labels)
    spec = dict(target_threshold_m=3,primary_target='accepted_and_horizontal_error_le_3m',
                model_inputs={k:list(v) for k,v in om.MODEL_INPUTS.items()},sha256={},input_meanings={},
                experiment_domain=dict(prior_covariance_xy_m2=[[400,0],[0,400]]),limitations=[])
    (folder/'feature_spec.json').write_text(json.dumps(spec))
    refresh_hashes(folder)
    return folder


def test_probability_scores_and_ties():
    result = om.probability_metrics([0,1],[.5,.5])
    assert result['log_loss'] == pytest.approx(np.log(2))
    assert result['brier_score'] == .25 and result['roc_auc'] == .5
    assert om.probability_metrics([0,1],[.1,.9])['roc_auc'] == 1
    assert om.probability_metrics([1,1],[.8,1])['roc_auc'] is None
    with pytest.raises(ValueError): om.probability_metrics([0,2],[.2,.8])


def test_training_only_scaling_predictive_gain_and_frozen_export(tables, tmp_path):
    out = tmp_path/'models'
    original = (tables/'features.csv').read_bytes()
    summary = om.run_training(tables,out)
    assert summary['selected_by_validation']=='geometry_and_map_features'
    assert summary['map_feature_validation_log_loss_gain'] > .1
    assert not summary['validation_passed'] and not summary['final_test_evaluated']
    _,rows,y,train = om.load_tables(tables)
    train_model = json.loads((out/'validation_models/geometry.json').read_text())
    np.testing.assert_allclose(train_model['mean'],om._matrix([r for r,t in zip(rows,train) if t],om.GEOMETRY).mean(axis=0))
    assert train_model['training_count']==240
    frozen = om.load_frozen_model(out,'geometry')
    assert frozen['training_count']==360 and frozen['deployment_status']=='research_only'
    np.testing.assert_allclose(frozen['mean'],om._matrix(rows,om.GEOMETRY).mean(axis=0))
    p, outside = om.predict_model(frozen,rows)
    assert p.shape==(360,) and np.isfinite(p).all() and not outside.any()
    serialized = json.loads(json.dumps(frozen))
    np.testing.assert_allclose(om.predict_model(serialized,rows)[0],p,rtol=0,atol=0)
    assert original == (tables/'features.csv').read_bytes()
    assert (out/'calibration.png').exists()
    assert om.run_training(tables,out)['selected_by_validation']==summary['selected_by_validation']
    path = out/'frozen_models/geometry.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='checksum'):om.load_frozen_model(out,'geometry')


def test_checksum_leakage_holdout_and_bad_targets_are_rejected(tables):
    original = (tables/'features.csv').read_bytes()
    (tables/'features.csv').write_bytes(original+b'\n')
    with pytest.raises(ValueError,match='checksum'):om.load_tables(tables)
    (tables/'features.csv').write_bytes(original)
    p=tables/'feature_spec.json';s=json.loads(p.read_text())
    s['model_inputs']['geometry'].append('inliers');p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='allowlist'):om.load_tables(tables)
    s['model_inputs']['geometry'].pop();p.write_text(json.dumps(s))
    rows=om._read_csv(tables/'features.csv')
    rows[0]['scene_id']='Scene_21'
    write_csv(tables/'features.csv',rows);refresh_hashes(tables)
    with pytest.raises(ValueError,match='holdout'):om.load_tables(tables)
    (tables/'features.csv').write_bytes(original);refresh_hashes(tables)
    labels=om._read_csv(tables/'labels.csv')
    labels[0]['target_success_3m']=1-int(labels[0]['target_success_3m'])
    write_csv(tables/'labels.csv',labels);refresh_hashes(tables)
    with pytest.raises(ValueError,match='Target'):om.load_tables(tables)


def test_constant_training_frequency_and_extrapolation(tables):
    _,rows,y,train=om.load_tables(tables)
    model=om.fit_binary(rows,y,())
    p,_=om.predict_model(model,rows)
    np.testing.assert_allclose(p,y.mean())
    model=om.fit_binary(rows,y,om.GEOMETRY)
    shifted=dict(rows[0]);shifted['log_agl_m']=100
    p,flag=om.predict_model(model,[shifted])
    assert flag[0] and 0<=p[0]<=1
    with pytest.raises(ValueError,match='Two binary'):om.fit_binary(rows,np.ones(len(rows)),om.GEOMETRY)


def test_training_notebook_code_english_and_valid():
    import ast,re
    path=Path(__file__).resolve().parents[4]/'notebooks/vml_outcome_training.ipynb'
    notebook=json.loads(path.read_text(encoding='utf-8'))
    for c in notebook['cells']:
        if c['cell_type']=='code':
            source=''.join(c['source']);ast.parse(source)
            assert not re.search(r'[\u0400-\u04ff]',source)
            # Executed Colab outputs are retained as research evidence.


def test_reverse_direction_uses_only_scene07_for_fit_and_preserves_sources(tables, tmp_path):
    originals={p.name:p.read_bytes() for p in tables.iterdir()}
    forward=tmp_path/'forward';forward.mkdir()
    (forward/'run.json').write_text('{"protocol":"original-forward"}')
    with pytest.raises(ValueError,match='identity changed'):
        om.run_reverse_diagnostic(tables,forward)
    assert (forward/'run.json').read_text()=='{"protocol":"original-forward"}'
    out=tmp_path/'reverse'
    summary=om.run_reverse_diagnostic(tables,out)
    assert summary['training_count']==120 and summary['evaluation_count']==240
    assert summary['selected_by_validation'] is None and not summary['final_refit_performed']
    assert not summary['independent_final_test'] and not summary['final_test_evaluated']
    _,rows,y,original_train=om.load_tables(tables)
    training=[r for r,t in zip(rows,original_train) if not t]
    evaluation=[r for r,t in zip(rows,original_train) if t]
    model=om._read(out/'evaluation_models/geometry.json')
    assert model['training_count']==120 and model['training_positive_count']==int(y[~original_train].sum())
    np.testing.assert_allclose(model['mean'],om._matrix(training,om.GEOMETRY).mean(0))
    constant=om._read(out/'evaluation_models/constant.json')
    np.testing.assert_allclose(om.predict_model(constant,evaluation)[0],y[~original_train].mean())
    predictions=om._read_csv(out/'evaluation_predictions.csv')
    assert len(predictions)==720 and {r['scene_id'] for r in predictions}=={'Scene_05'}
    manifest=om._read(out/'manifest.json')
    for relative,expected in manifest['artifact_sha256'].items():
        assert om._sha(out/relative)==expected
    assert manifest['deployment_status']=='diagnostic_only'
    assert not (out/'frozen_models').exists()
    assert originals=={p.name:p.read_bytes() for p in tables.iterdir()}
    assert 'Scene_07' in (out/'report.md').read_text()
    assert om.run_reverse_diagnostic(tables,out)==summary


def test_reverse_notebook_has_hosted_guard_and_english_valid_code():
    import ast,re
    path=Path(__file__).resolve().parents[4]/'notebooks/vml_outcome_reverse.ipynb'
    notebook=json.loads(path.read_text(encoding='utf-8'))
    code=''
    for c in notebook['cells']:
        if c['cell_type']=='code':
            source=''.join(c['source']);ast.parse(source);code+=source
            assert not re.search(r'[\u0400-\u04ff]',source)
    assert 'COLAB_RELEASE_TAG' in code and 'run_reverse_diagnostic' in code
    assert 'correct-fix-reverse-v1' in code

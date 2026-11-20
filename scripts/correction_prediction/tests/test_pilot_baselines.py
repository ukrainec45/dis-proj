import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.correction_prediction.artifacts import load_verified, read
from scripts.correction_prediction.contracts import label_outcome
from scripts.correction_prediction.dataset import save_dataset, load_dataset
from scripts.correction_prediction.features import ALL_FEATURES
from scripts.correction_prediction.models import CANDIDATES, FrozenModel
from scripts.correction_prediction.pilot_baselines import pilot_folds, run_pilot_baselines


def synthetic_pilot():
    rng = np.random.default_rng(42)
    rows = []
    deltas = [0, -30, -10, -2, 0, 2, 10, 30]
    for scene in ('Scene_05', 'Scene_08', 'Scene_11', 'Scene_21'):
        for j, delta in enumerate(deltas):
            f = {k: float(v) for k, v in zip(ALL_FEATURES, rng.normal(size=len(ALL_FEATURES)))}
            f['sigma_major_m'] = float(j)
            rows.append(dict(attempt_id=f'{scene}-{j}', scene_id=scene,
                sample_id=f'frame-{j//4}', spatial_group=scene, split='pilot', condition='clear',
                sigma_xy_m=20., trial=j % 3, features=f,
                label=label_outcome([50., 0, 0], [50.-delta, 0, 0], [0., 0, 0], j != 0)))
    return rows


def test_folds_keep_entire_scenes_and_frame_repetitions_together():
    rows = synthetic_pilot()
    folds = pilot_folds(rows)
    held_all = []
    for group, train, held in folds:
        train_frames = {(rows[i]['scene_id'], rows[i]['sample_id']) for i in train}
        held_frames = {(rows[i]['scene_id'], rows[i]['sample_id']) for i in held}
        assert not train_frames & held_frames
        assert all(rows[i]['spatial_group'] != group for i in train)
        assert all(rows[i]['spatial_group'] == group for i in held)
        held_all.extend(held)
    assert sorted(held_all) == list(range(len(rows))) and len(folds) == 4
    rows[0]['spatial_group'] = 'other'
    with pytest.raises(ValueError, match='scene'):
        pilot_folds(rows)


def test_pilot_training_produces_only_group_held_out_predictions_and_research_refits(tmp_path):
    rows = synthetic_pilot()
    dataset = tmp_path/'pilot/dataset.json'
    save_dataset(dataset, rows, dict(stage='pilot'))
    before = dataset.read_bytes()
    output = tmp_path/'baselines'
    summary = run_pilot_baselines(dataset, output)
    assert dataset.read_bytes() == before
    assert summary['usable_attempts'] == 32 and summary['unique_frames'] == 8
    assert not summary['calibration_fitted'] and not summary['model_selected']
    assert not summary['final_test_evaluated'] and not summary['validation_passed']
    assert not (output/'frozen_run.json').exists()
    folds = load_verified(output/'folds.json')
    source = load_dataset(dataset)
    by_id = {r['attempt_id']: r for r in source['rows']}
    oof = load_verified(output/'oof_predictions.json')
    for name in CANDIDATES:
        assert {r['attempt_id'] for r in oof[name]} == set(by_id)
        refit = FrozenModel.load(output/'exploratory_models'/(name+'.json'))
        assert refit.artifact['exploratory_pilot'] and refit.artifact['training_count'] == 32
        for fold in folds:
            model = FrozenModel.load(output/'fold_models'/fold['fold_id']/(name+'.json'))
            assert model.artifact['training_count'] == 24
            held = [by_id[k]['features'] for k in fold['held_out_attempt_ids']]
            expected = model.predict(held, calibrated=False)
            saved = {r['attempt_id']: r['probabilities'] for r in oof[name]}
            np.testing.assert_allclose([saved[k] for k in fold['held_out_attempt_ids']], expected)
    assert (output/'calibration.png').is_file()
    assert 'No model selected' in (output/'report.md').read_text()
    assert summary['evaluations']['constant']['paired_gain_vs_constant']['log_loss_gain'] == 0
    with pytest.raises(FileExistsError):
        run_pilot_baselines(dataset, output)


def test_exploratory_entry_point_rejects_main_and_failed_pilots(tmp_path):
    rows = synthetic_pilot()
    main = tmp_path/'main/dataset.json'
    save_dataset(main, rows, dict(stage='main'))
    with pytest.raises(ValueError, match='pilot rows only'):
        run_pilot_baselines(main, tmp_path/'bad-main')
    for row in rows:
        row['label'] = dict(technical_error=True, class_index=None, delta_m=None)
    failed = tmp_path/'failed/dataset.json'
    save_dataset(failed, rows, dict(stage='pilot'))
    with pytest.raises(ValueError, match='diagnose'):
        run_pilot_baselines(failed, tmp_path/'bad-failed')


def test_cpu_notebook_has_no_localization_or_jev_execution():
    root = Path(__file__).resolve().parents[1]
    notebook = read(root/'notebooks/baseline_models.ipynb')
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    assert 'run_pilot_baselines(DATASET, BASELINE_OUTPUT)' in code
    assert 'run_pilot(' not in code and 'initialize_backend(' not in code
    assert 'run_jev_predictions(' not in code and 'import torch' not in code
    assert 'pilot-v2-cublas/dataset.json' in code

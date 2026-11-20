from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from scripts.correction_prediction.artifacts import save_verified, load_verified
from scripts.correction_prediction.diagnostics import diagnose_campaign
from scripts.correction_prediction.campaign import pilot_config, run_campaign


def test_diagnostics_groups_actual_exceptions_without_dataset(tmp_path):
    for i, message in enumerate(['missing setting', 'missing setting', 'GPU unavailable']):
        save_verified(tmp_path/'records'/f'{i}.json', dict(scene_id='scene', sample_id=str(i),
            condition='clear', label={'technical_error': True},
            localization=dict(exception_type='RuntimeError', exception=message, traceback='trace '+message)))
    report = diagnose_campaign(tmp_path)
    assert report['saved_attempts'] == report['technical_errors'] == 3
    assert report['errors'][0]['count'] == 2 and report['errors'][0]['message'] == 'missing setting'
    assert report['examples'][0]['traceback'] == 'trace missing setting'
    assert not (tmp_path/'dataset.json').exists()


def test_pilot_saves_first_failure_then_stops_and_refuses_to_repeat(tmp_path):
    from scripts.correction_prediction import campaign
    config = pilot_config()
    backend = SimpleNamespace(config=config, identity={'synthetic': True})
    selection = [dict(scene_id=s, sample_id=str(j), path=str(tmp_path/'source'/s/f'{j}.npz'),
        reference_json=str(tmp_path/'source'/s/'reference.json'), split='pilot', spatial_group=s)
        for s in config['scenes'] for j in range(2)]
    def sample_reader(path, *args):
        path = Path(path)
        return dict(scene_id=path.parent.name, sample_id=path.stem, source_identity={'synthetic': True},
            xyz_enu=np.array([0., 0., 80.]), rotation=np.diag([1., -1., -1.]),
            K=np.array([[20., 0., 16.], [0., 20., 16.], [0., 0., 1.]]),
            dist=np.zeros(5), rgb=np.zeros((32, 32, 3), 'uint8'))
    from scripts.visibility_localization.data import Reference
    reference = Reference('synthetic', np.zeros((1000, 1000, 3), 'uint8'),
        np.zeros((1000, 1000)), np.ones(2), np.array([-500., -500.]),
        np.ones(2), np.array([-500., -500.]), 0, {'synthetic': True})
    calls = []
    def broken(*args):
        calls.append(1)
        raise RuntimeError('deterministic synthetic failure')
    output = tmp_path/'output'
    kwargs = dict(sample_reader=sample_reader, reference_reader=lambda *args: reference,
                  reference_fingerprint=lambda *args: {'synthetic': True}, localizer=broken)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='Pilot stopped.*deterministic synthetic failure'):
            run_campaign(selection, backend, output, config, **kwargs)
    assert len(calls) == 1
    records = list((output/'records').glob('*.json'))
    assert len(records) == 1 and load_verified(records[0])['label']['technical_error'] is True
    assert not (output/'dataset.json').exists()
    assert diagnose_campaign(output)['examples'][0]['traceback'].endswith('RuntimeError: deterministic synthetic failure\n')

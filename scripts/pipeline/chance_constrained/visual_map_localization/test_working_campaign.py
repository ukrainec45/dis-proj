"""Software verification of the no-coverage-gate campaign; synthetic data only."""
import json
import shutil
from pathlib import Path

import pytest

from . import audit, working_campaign
from .test_anyvisloc_smoke import config, dataset
from .test_audit import baseline


@pytest.fixture
def frozen(dataset, baseline, config, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'draw_examples', lambda *args: None)
    out = tmp_path / 'audit'
    audit.audit_scene(dataset, dataset / 'Scene_05/L01_reference.json', out, config, baseline)
    shutil.copytree(dataset / 'Scene_05', dataset / 'Scene_21')
    audit.audit_scene(dataset, dataset / 'Scene_21/L01_reference.json', out, config, baseline)
    audit.finalize_audit(out)
    return out


def fake_matcher(calls, fail_at=None):
    def match(*args, coverage_policy):
        assert coverage_policy == 'available_crop'
        calls.append(1)
        if fail_at and len(calls) == fail_at:
            raise RuntimeError('simulated interruption')
        cfg, reference, prior = args[-1], args[-2], args[5]
        # Use the real geometry/crop computation for the supplied synthetic inputs.
        from .planar import rectify, search_bounds
        _, _, offsets, _, _ = rectify(*args[:5], cfg['rectified_gsd_m'],
                                      cfg['max_image_pixels'], cfg['maximum_projection_condition'])
        coverage = search_bounds(reference, offsets, prior, args[6], args[7])[-1]
        return dict(status='accepted', reason='accepted_xy_uncalibrated', z=[200, -200],
                    diagnostics=dict(map_coverage=coverage, coverage_policy=coverage_policy), runtime_s=.01)
    return match


def test_full_working_selection_resume_and_counterfactual(dataset, frozen, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(working_campaign, 'localize', fake_matcher(calls))
    output = tmp_path / 'campaign'
    frozen_bytes = (frozen / 'manifest.json').read_bytes()
    summary = working_campaign.run_working_scene(dataset, frozen, output, 'Scene_05')
    # Fixture has 20 geometry candidates: four satisfy strict coverage, sixteen do not.
    assert len(calls) == 20
    assert summary['status'] == 'completed'
    assert summary['expected_trial_count'] == summary['completed_trial_count'] == 20
    assert summary['available_crop']['accepted_count'] == 20
    assert summary['legacy_95pct_counterfactual']['accepted_count'] == 4
    assert summary['available_crop']['accepted_error_over_m']['1'] == 16
    assert summary['available_crop']['acc_at_m']['1'] == .2
    assert summary['scenes']['Scene_05']['newly_accepted_count'] == 16
    assert (frozen / 'manifest.json').read_bytes() == frozen_bytes
    working_campaign.run_working_scene(dataset, frozen, output, 'Scene_05')
    assert len(calls) == 20
    assert not (output / 'scenes/Scene_21').exists()
    envelope = next((output / 'scenes/Scene_05/trials').glob('*.json'))
    data = audit.read_json(envelope)
    data['trial']['error_m'] = 0
    audit.write_json(envelope, data)
    with pytest.raises(ValueError, match='checksum'):
        working_campaign.summarize_campaign(frozen, output)


def test_interruption_keeps_verified_trials_and_reports_incomplete(dataset, frozen, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(working_campaign, 'localize', fake_matcher(calls, fail_at=3))
    output = tmp_path / 'campaign'
    with pytest.raises(RuntimeError, match='interruption'):
        working_campaign.run_working_scene(dataset, frozen, output, 'Scene_05')
    partial = working_campaign.summarize_campaign(frozen, output)
    assert partial['status'] == 'incomplete' and partial['completed_trial_count'] == 2
    recovered = []
    monkeypatch.setattr(working_campaign, 'localize', fake_matcher(recovered))
    complete = working_campaign.run_working_scene(dataset, frozen, output, 'Scene_05')
    assert len(recovered) == 18 and complete['status'] == 'completed'


def test_holdout_and_modified_sources_are_rejected_before_matching(dataset, frozen, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Matching must not start')
    monkeypatch.setattr(working_campaign, 'localize', forbidden)
    output = tmp_path / 'campaign'
    with pytest.raises(ValueError, match='working scenes'):
        working_campaign.run_working_scene(dataset, frozen, output, 'Scene_21')
    assert not output.exists()
    path = dataset / 'Scene_05/map.png'
    path.write_bytes(path.read_bytes()+b'changed')  # Still decodable, source checksum differs.
    with pytest.raises(ValueError, match='Audited source changed'):
        working_campaign.run_working_scene(dataset, frozen, output, 'Scene_05')
    assert not output.exists()


def test_available_crop_is_a_real_localizer_mode(dataset, config):
    import numpy as np
    from .anyvisloc import load_reference, load_sample
    from .smoke_localizer import localize
    ref = load_reference(dataset / 'Scene_01/L01_reference.json', 'aerial')
    s = load_sample(dataset / 'Scene_01/L01_0001.npz', pose_world_frame='enu')
    args = (s['image'], s['K'], s['dist'], s['rotation'], 60, [390, -200], np.eye(2)*400, .99, ref, config)
    assert localize(*args)['reason'] == 'insufficient_map_coverage'
    result = localize(*args, coverage_policy='available_crop')
    assert result['reason'] != 'insufficient_map_coverage'
    assert result['diagnostics']['coverage_policy'] == 'available_crop'


def test_working_notebook_code_is_english_and_valid_python():
    import ast, re
    path = Path(__file__).resolve().parents[4] / 'notebooks/vml_anyvisloc_working.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            # Retain executed Colab outputs as research evidence.
            source = ''.join(cell['source'])
            assert not re.search(r'[\u0400-\u04ff]',source)
            ast.parse(source)


def test_two_working_scenes_report_pending_then_complete(dataset, frozen, config, tmp_path, monkeypatch):
    # Construct a fresh three-scene audit; the already-frozen fixture remains untouched.
    shutil.copytree(dataset / 'Scene_05', dataset / 'Scene_07')
    combined = tmp_path / 'three-scene-audit'
    audit.audit_dataset(dataset, combined, config, tmp_path / 'baseline')
    manifest = audit.finalize_audit(combined)
    assert manifest['roles'] == {'Scene_05': 'working', 'Scene_07': 'working', 'Scene_21': 'holdout'}
    calls = []
    monkeypatch.setattr(working_campaign, 'localize', fake_matcher(calls))
    output = tmp_path / 'campaign'
    first = working_campaign.run_working_scene(dataset, combined, output, 'Scene_05')
    assert first['status'] == 'incomplete'
    assert first['completed_trial_count'] == 20 and first['expected_trial_count'] == 40
    assert first['scenes']['Scene_07']['status'] == 'pending'
    second = working_campaign.run_working_scene(dataset, combined, output, 'Scene_07')
    assert second['status'] == 'completed'
    assert second['completed_trial_count'] == second['expected_trial_count'] == 40
    assert len(calls) == 40
    assert len(list((output / 'scenes/Scene_05/trials').glob('*.json'))) == 20
    assert len(list((output / 'scenes/Scene_07/trials').glob('*.json'))) == 20
    assert not (output / 'scenes/Scene_21').exists()

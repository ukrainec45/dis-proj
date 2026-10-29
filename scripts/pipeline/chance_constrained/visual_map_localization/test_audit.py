"""Synthetic data only. Tests never access Drive, download scenes or run notebooks."""
import copy
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from . import audit, anyvisloc_smoke
from .anyvisloc import load_reference, load_sample
from .planar import rectify, search_bounds, search_crop
from .smoke_localizer import localize
from .test_anyvisloc_smoke import config, dataset  # Reuse synthetic camera/map fixtures.


@pytest.fixture
def baseline(dataset, config, tmp_path, monkeypatch):
    # Give fixture the identity of the explicitly allowed control scene.
    (dataset / 'Scene_01').rename(dataset / 'Scene_05')
    # The last 16 camera positions genuinely require crops past the west edge.
    for path in sorted((dataset / 'Scene_05').glob('L01_*.npz')):
        if int(path.stem[-4:]) >= 5:
            with np.load(path) as data:
                arrays = dict(data)
            arrays['xyz'] = arrays['xyz'].copy()
            arrays['pose_c2w'] = arrays['pose_c2w'].copy()
            arrays['xyz'][0] = arrays['pose_c2w'][0, 3] = 5
            np.savez(path, **arrays)
    calls = []
    def recorded(image, K, dist, rotation, agl, prior, cov, mass, reference, cfg):
        _, _, offsets, _, _ = rectify(image, K, dist, rotation, agl, .5, 16000000, 100)
        coverage = search_bounds(reference, offsets, prior, cov, mass)[-1]
        accepted = len(calls) % 20 < 4
        assert (coverage >= cfg['minimum_map_coverage']) == accepted
        calls.append(1)
        return dict(status='accepted' if accepted else 'reject',
                    reason='accepted_xy_uncalibrated' if accepted else 'insufficient_map_coverage',
                    z=list(prior) if accepted else None,
                    diagnostics=dict(map_coverage=coverage))
    folder = tmp_path / 'baseline'
    with monkeypatch.context() as patch:
        patch.setattr(anyvisloc_smoke, 'localize', recorded)
        anyvisloc_smoke.run_smoke(dataset, folder, config)
    return folder


def test_geometry_matches_crop_and_separates_prior(dataset, config):
    ref = load_reference(dataset / 'Scene_01/L01_reference.json', 'aerial')
    sample = load_sample(dataset / 'Scene_01/L01_0001.npz', pose_world_frame='enu')
    full = audit.geometry(ref, sample, 60, sample['xyz'][:2], config)
    assert full['footprint_coverage'] == pytest.approx(1)
    assert full['gt_search_coverage'] == full['prior_search_coverage'] == 1
    partial = audit.geometry(ref, sample, 60, [390, -200], config)
    assert partial['footprint_coverage'] == 1
    assert partial['prior_search_coverage'] < partial['gt_search_coverage']
    assert 'prior_shift_reduces_coverage' in partial['factors']
    _, _, offsets, _, _ = rectify(sample['image'], sample['K'], sample['dist'], sample['rotation'], 60, .5, 1000000, 100)
    _, _, coverage = search_crop(ref, offsets, [390, -200], np.eye(2)*400, .99, 1000000)
    assert coverage == partial['prior_search_coverage']
    outside = audit.geometry(ref, sample, 60, [2000, -200], config)
    assert outside['prior_search_coverage'] == 0
    assert not outside['geometry_supported']
    assert 'map_outside' in outside['factors']
    sample['xyz'][:2] = [5, -5]
    clipped = audit.geometry(ref, sample, 60, [5, -5], config)
    assert 0 < clipped['footprint_coverage'] < 1
    assert 'footprint_outside_map' in clipped['factors']


def test_coverage_policy_is_explicit_and_strict_default(dataset, config):
    ref = load_reference(dataset / 'Scene_01/L01_reference.json', 'aerial')
    sample = load_sample(dataset / 'Scene_01/L01_0001.npz', pose_world_frame='enu')
    args = (sample['image'], sample['K'], sample['dist'], sample['rotation'], 60,
            [390, -200], np.eye(2)*400, .99, ref, config)
    before = copy.deepcopy(config)
    assert localize(*args)['reason'] == 'insufficient_map_coverage'
    result = localize(*args, coverage_policy='audit_clipped')
    assert result['reason'] != 'insufficient_map_coverage'
    assert result['diagnostics']['coverage_policy'] == 'audit_clipped'
    assert config == before
    assert localize(*args)['reason'] == 'insufficient_map_coverage'
    with pytest.raises(ValueError, match='policy'):
        localize(*args, coverage_policy='typo')


def test_full_audit_continues_freezes_split_and_never_matches(dataset, baseline, config, tmp_path, monkeypatch):
    working = dataset / 'Scene_05'
    new = dataset / 'Scene_06'
    shutil.copytree(working, new)
    # A broken extra frame must not prevent subsequent good frames being visited.
    (working / 'L01_0000.npz').write_bytes(b'broken npz')
    def forbidden(*args, **kwargs):
        pytest.fail('Inventory must not run matching')
    monkeypatch.setattr(audit, 'localize', forbidden)
    output = tmp_path / 'audit'
    result = audit.audit_scene(dataset, working / 'L01_reference.json', output, config, baseline)
    assert result['frame_count'] == 21
    rows = audit.read_json(output / 'scenes/Scene_05/frames.json')
    assert rows[0]['error'] and rows[-1]['geometry_supported']
    assert sum(r.get('baseline_verified', False) for r in rows) == 20
    assert not result['fully_verified']
    other = audit.audit_scene(dataset, new / 'L01_reference.json', output, config, baseline)
    assert other['fully_verified']
    manifest = audit.finalize_audit(output)
    assert manifest['roles'] == {'Scene_05': 'working', 'Scene_06': 'holdout'}
    assert manifest == audit.finalize_audit(output)
    with pytest.raises(ValueError, match='frozen'):
        audit.audit_scene(dataset, new / 'L01_reference.json', output, config, baseline)
    assert (output / 'report.md').exists()
    assert (output / 'scenes.csv').exists()


def test_shared_map_failure_and_missing_reference_are_scene_errors(dataset, baseline, config, tmp_path):
    scene = dataset / 'Scene_05'
    (scene / 'map.png').unlink()
    result = audit.audit_scene(dataset, scene / 'L01_reference.json', tmp_path / 'audit', config, baseline)
    assert result['shared_errors'] and result['status'] == 'incomplete'
    assert result['frame_count'] == 21
    rows = audit.read_json(tmp_path / 'audit/scenes/Scene_05/frames.json')
    assert not any(r.get('baseline_verified') for r in rows)
    assert not audit.finalize_audit(tmp_path / 'audit')['reserve_formed']


def test_baseline_corruption_and_source_changes_are_detected(dataset, baseline, config, tmp_path):
    changed = copy.deepcopy(config)
    changed['seed'] += 1
    with pytest.raises(ValueError, match='configuration'):
        audit.load_baseline(baseline, changed)
    scene = dataset / 'Scene_05'
    path = scene / 'L01_0001.npz'
    with np.load(path) as data:
        arrays = dict(data)
    arrays['image'] = arrays['image'].copy()
    arrays['image'][0, 0] = 0
    np.savez(path, **arrays)
    audit.audit_scene(dataset, scene / 'L01_reference.json', tmp_path / 'audit', config, baseline)
    rows = audit.read_json(tmp_path / 'audit/scenes/Scene_05/frames.json')
    row = next(r for r in rows if r['sample'].endswith('0001.npz'))
    assert not row['baseline_verified']
    assert 'hash_mismatch' in row['error']
    aggregate = next(baseline.glob('shards/**/aerial/sift/smoke/trials.json'))
    aggregate.write_text('[]')
    with pytest.raises(ValueError, match='checksum'):
        audit.load_baseline(baseline, config)


def test_control_replays_exact_16_with_resume_and_tamper_guard(dataset, baseline, config, tmp_path, monkeypatch):
    out = tmp_path / 'audit'
    audit.audit_scene(dataset, dataset / 'Scene_05/L01_reference.json', out, config, baseline)
    audit.finalize_audit(out)
    baseline_hash = audit.digest(baseline / 'manifest.json')
    calls = []
    def control(*args, coverage_policy):
        assert coverage_policy == 'audit_clipped'
        calls.append(1)
        return dict(status='accepted', reason='accepted_xy_uncalibrated', z=[5, -200],
                    diagnostics=dict(coverage_policy=coverage_policy))
    monkeypatch.setattr(audit, 'localize', control)
    results = audit.run_control(dataset, baseline, out, tmp_path / 'control', config)
    assert len(results) == len(calls) == 16
    assert all(r['error_m'] == 0 for r in results)
    audit.run_control(dataset, baseline, out, tmp_path / 'control', config)
    assert len(calls) == 16
    assert audit.digest(baseline / 'manifest.json') == baseline_hash
    frame_path = out / 'scenes/Scene_05/frames.json'
    frame_path.write_text('[]')
    with pytest.raises(ValueError, match='frame audit changed'):
        audit.run_control(dataset, baseline, out, tmp_path / 'control', config)


def test_control_does_not_run_for_holdout(dataset, baseline, config, tmp_path, monkeypatch):
    out = tmp_path / 'audit'
    audit.audit_scene(dataset, dataset / 'Scene_05/L01_reference.json', out, config, baseline)
    manifest = audit.finalize_audit(out)
    manifest['roles']['Scene_05'] = 'holdout'
    audit.write_json(out / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='working Scene_05'):
        audit.run_control(dataset, baseline, out, tmp_path / 'control', config)


def test_archive_inventory_staging_budget_and_traversal(tmp_path):
    folder = tmp_path / 'archives'
    folder.mkdir()
    path = folder / 'Scene_02.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('Scene_02/L02_reference.json', '{}')
    reports = tmp_path / 'reports'
    inventory = audit.inventory_archives(folder, reports, budget_bytes=1)
    assert not inventory['within_archive_budget']
    assert inventory['archives'][0]['scene_ids'] == ['Scene_02']
    receipt = audit.stage_archive(path, tmp_path / 'staged', reports, audit.digest(path), reserve_bytes=0)
    assert receipt['unpacked_bytes'] == 2 and path.exists()
    reused = audit.stage_archive(path, tmp_path / 'staged', reports, audit.digest(path))
    assert reused['status'] == 'reused' and reused['extracted_files'] == 0
    bad = folder / 'bad.zip'
    with zipfile.ZipFile(bad, 'w') as archive:
        archive.writestr('../escape.txt', 'no')
    with pytest.raises(ValueError, match='Unsafe'):
        audit.stage_archive(bad, tmp_path / 'unsafe', reports, audit.digest(bad))
    assert not (tmp_path / 'escape.txt').exists()


def test_split_uses_hash_ranking_and_ceil_not_scene_order(dataset, baseline, config, tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(audit, 'draw_examples', lambda *args: None)
    # Four new scenes -> two holdouts. All data/geometry checks remain real.
    names = ['Scene_09', 'Scene_06', 'Scene_08', 'Scene_07']
    out = tmp_path / 'audit'
    for name in names:
        target = dataset / name
        shutil.copytree(dataset / 'Scene_05', target)
        audit.audit_scene(dataset, target / 'L01_reference.json', out, config, baseline)
    manifest = audit.finalize_audit(out)
    expected = sorted(names, key=lambda x: hashlib.sha256(f'42:{x}'.encode()).hexdigest())[:2]
    assert sorted(k for k, v in manifest['roles'].items() if v == 'holdout') == sorted(expected)


def test_missing_reference_is_discovered_and_source_data_remain_intact(dataset, baseline, config, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'draw_examples', lambda *args: None)
    scene = dataset / 'Scene_05'
    (scene / 'L01_reference.json').unlink()
    reports = audit.audit_dataset(dataset, tmp_path / 'audit', config, baseline)
    assert len(reports) == 1
    assert reports[0]['status'] == 'incomplete'
    assert reports[0]['frame_count'] == 21
    assert len(list(scene.glob('L01_*.npz'))) == 21
    manifest = audit.finalize_audit(tmp_path / 'audit')
    assert not manifest['reserve_formed']
    text = (tmp_path / 'audit/report.md').read_text(encoding='utf-8')
    # Original coverage rejects still appear individually, even without geometry.
    assert text.count('insufficient_map_coverage') == 16
    assert 'baseline_shared_sources_unverified' in text


def test_notebook_code_is_valid_python_and_english():
    import ast
    path = Path(__file__).resolve().parents[4] / 'notebooks/vml_anyvisloc_audit.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    import re
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Saved Colab outputs are research evidence and may be retained.
            source = ''.join(cell['source'])
            assert not re.search(r'[\u0400-\u04ff]', source)
            ast.parse(source)



def test_staging_recovers_missing_files_and_refuses_modified_files(tmp_path, monkeypatch):
    from types import SimpleNamespace
    folder = tmp_path / 'archives'
    folder.mkdir()
    path = folder / 'Scene_05.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('Scene_05/a.bin', b'1234')
        archive.writestr('Scene_05/b.bin', b'5678')
    dest, receipts = tmp_path / 'runtime', tmp_path / 'receipts'
    checksum = audit.digest(path)
    audit.stage_archive(path, dest, receipts, checksum, reserve_bytes=0)
    first = dest / 'Scene_05/a.bin'
    second = dest / 'Scene_05/b.bin'
    original_mtime = first.stat().st_mtime_ns
    second.unlink()
    resumed = audit.stage_archive(path, dest, receipts, checksum, reserve_bytes=0)
    assert resumed['status'] == 'resumed'
    assert resumed['extracted_files'] == resumed['verified_existing_files'] == 1
    assert second.read_bytes() == b'5678'
    assert first.stat().st_mtime_ns == original_mtime
    # Reuse needs no additional extraction space, even below the default reserve.
    monkeypatch.setattr(audit.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    assert audit.stage_archive(path, dest, receipts, checksum)['status'] == 'reused'
    first.write_bytes(b'xxxx')  # Same size, different CRC.
    with pytest.raises(ValueError, match='checksum mismatch'):
        audit.stage_archive(path, dest, receipts, checksum)
    assert first.read_bytes() == b'xxxx'
    assert second.read_bytes() == b'5678'


def test_finalize_rebuilds_missing_reports_without_changing_frozen_manifest(dataset, baseline, config, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'draw_examples', lambda *args: None)
    out = tmp_path / 'audit'
    audit.audit_scene(dataset, dataset / 'Scene_05/L01_reference.json', out, config, baseline)
    original = audit.finalize_audit(out)
    manifest_bytes = (out / 'manifest.json').read_bytes()
    scene_hash = audit.digest(out / 'scenes/Scene_05/scene.json')
    frames_hash = audit.digest(out / 'scenes/Scene_05/frames.json')
    (out / 'report.md').unlink()
    (out / 'scenes.csv').unlink()
    def forbidden(*args, **kwargs):
        pytest.fail('Report repair must not rerun geometry or matching')
    monkeypatch.setattr(audit, 'geometry', forbidden)
    monkeypatch.setattr(audit, 'localize', forbidden)
    assert audit.finalize_audit(out) == original
    assert (out / 'manifest.json').read_bytes() == manifest_bytes
    assert audit.digest(out / 'scenes/Scene_05/scene.json') == scene_hash
    assert audit.digest(out / 'scenes/Scene_05/frames.json') == frames_hash
    assert (out / 'scenes.csv').exists()
    assert 'AnyVisLoc audit' in (out / 'report.md').read_text()
    # Frozen evidence corruption must not be turned into a new report.
    (out / 'scenes/Scene_05/frames.json').write_text('[]')
    report_bytes = (out / 'report.md').read_bytes()
    with pytest.raises(ValueError, match='Frozen audit evidence changed'):
        audit.finalize_audit(out)
    assert (out / 'report.md').read_bytes() == report_bytes

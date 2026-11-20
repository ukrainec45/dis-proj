import copy
import json
from io import BytesIO
import zipfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from scripts.correction_models.artifacts import save_verified, load_verified, write_once
from scripts.correction_models.selection import (audit_groups, validate_selection, farthest_frames,
    read_metadata, inventory, prepare_selection, scene_policy, selection_readiness)
from .conftest import fake_reader


def test_audit_is_not_inferred_from_distinct_scene_names(selection):
    audit = copy.deepcopy(selection['spatial_overlap_audit'])
    audit['reviewed'] = False
    with pytest.raises(ValueError, match='Geographic overlap audit'):
        audit_groups(audit, list(audit['scenes']))


def test_overlapping_polygons_cannot_be_split_into_independent_groups(selection):
    audit = copy.deepcopy(selection['spatial_overlap_audit'])
    polygon = dict(type='Polygon', coordinates=[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]])
    for scene in ('Scene_01', 'Scene_02'):
        audit['scenes'][scene]['footprint_wgs84'] = polygon
    with pytest.raises(ValueError, match='Overlapping'):
        audit_groups(audit, ['Scene_01', 'Scene_02'])


def test_frozen_selection_counts_and_leakage(selection):
    validate_selection(selection)
    bad = copy.deepcopy(selection)
    bad['rows'][0]['split'] = 'test'
    with pytest.raises(ValueError):
        validate_selection(bad)
    bad = copy.deepcopy(selection)
    bad['rows'][-1]['scene_id'] = 'Scene_21'
    with pytest.raises(ValueError, match='Previously reviewed'):
        validate_selection(bad)


def test_farthest_sampling_is_reproducible_and_separates_positions():
    rows = [dict(scene_id='scene', sample_id=str(i), xyz_enu=[i, 0, 0]) for i in range(30)]
    a = farthest_frames(rows)
    assert a == farthest_frames(list(reversed(rows)))
    assert len({r['xyz_enu'][0] for r in a}) == 10
    assert min(r['xyz_enu'][0] for r in a) == 0
    assert max(r['xyz_enu'][0] for r in a) == 29


def test_inventory_reads_metadata_without_image(tmp_path):
    directory = tmp_path/'Scene_01'
    directory.mkdir()
    (directory/'L01_reference.json').write_text('{}')
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 80]
    path = directory/'L01_0001.npz'
    # Deliberately no image member: metadata inventory must not load an image.
    np.savez(path, xyz=pose[:3, 3], pose_c2w=pose, euler_deg=np.zeros(3))
    assert read_metadata(path)['nadir_deg'] == 0
    report = inventory(tmp_path, tmp_path/'inventory.json')
    assert report['frames'] == 1 and report['eligible'] == 1


def test_metadata_rejects_huge_npy_shape_before_numpy_load(tmp_path, monkeypatch):
    stream = BytesIO()
    np.lib.format.write_array_header_1_0(stream, dict(
        descr='<f8', fortran_order=False, shape=(10**12,)))
    path = tmp_path/'bad.npz'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('xyz.npy', stream.getvalue())
    def forbidden(*args, **kwargs):
        raise AssertionError('Must reject oversized shape before allocating')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='Invalid metadata shape'):
        read_metadata(path)


def test_metadata_rejects_large_member_before_decompression(tmp_path):
    path = tmp_path/'bad.npz'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('xyz.npy', b'0'*(64*1024+1))
    with pytest.raises(ValueError, match='Oversized metadata member'):
        read_metadata(path)


def test_inventory_resumes_completed_scenes_after_memory_error(tmp_path):
    for scene in ('Scene_01', 'Scene_02'):
        directory = tmp_path/scene
        directory.mkdir()
        (directory/'L01_reference.json').write_text('{}')
        pose = np.eye(4)
        pose[:3, 3] = [1, 2, 80]
        np.savez(directory/'L01_0001.npz', xyz=pose[:3, 3], pose_c2w=pose, euler_deg=np.zeros(3))
    output = tmp_path/'inventory.json'
    def interrupted(path, *args):
        if path.parent.name == 'Scene_02':
            raise MemoryError('simulated allocation failure')
        return read_metadata(path, *args)
    with pytest.raises(MemoryError):
        inventory(tmp_path, output, reader=interrupted, progress=None)
    assert not output.exists()
    seen, messages = [], []
    def resumed(path, *args):
        seen.append(path.parent.name)
        return read_metadata(path, *args)
    report = inventory(tmp_path, output, reader=resumed, progress=messages.append)
    assert report['frames'] == 2 and seen == ['Scene_02']
    assert any('Scene_01: reused' in message for message in messages)
    assert any('RSS=' in message and 'available=' in message for message in messages)


def test_low_ram_stops_before_reading_frame(tmp_path, monkeypatch):
    import psutil
    directory = tmp_path/'Scene_01'
    directory.mkdir()
    (directory/'L01_reference.json').write_text('{}')
    (directory/'L01_0001.npz').write_bytes(b'not read')
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=128*1024**2))
    def forbidden(*args):
        raise AssertionError('Reader must not start with insufficient RAM')
    with pytest.raises(MemoryError, match='less than 512'):
        inventory(tmp_path, tmp_path/'inventory.json', reader=forbidden, progress=None)


def test_prepare_exact_160_main_selection(tmp_path, selection, monkeypatch):
    import scripts.correction_models.selection as module
    monkeypatch.setattr(module, 'reference_identity', lambda *args: {'fixture': 'map'})
    path, audit, output = tmp_path/'inventory.json', tmp_path/'audit.json', tmp_path/'selected.json'
    save_verified(path, dict(rows=selection['rows'], errors=[]))
    write_once(audit, selection['spatial_overlap_audit'])
    report = prepare_selection(path, audit, output, sample_reader=fake_reader)
    assert report['frames'] == 160 and report['expected_attempts'] == 1920
    result = load_verified(output)
    assert not set(result['split_scenes']['test']) & {'Scene_05', 'Scene_07', 'Scene_08', 'Scene_11', 'Scene_21'}
    validate_selection(result)


def test_insufficient_independent_territories_fail_before_collection(tmp_path, selection):
    path, audit = tmp_path/'inventory.json', tmp_path/'audit.json'
    save_verified(path, dict(rows=selection['rows'][:150], errors=[]))
    write_once(audit, selection['spatial_overlap_audit'])
    with pytest.raises(ValueError, match='Need 16'):
        prepare_selection(path, audit, tmp_path/'out.json', sample_reader=fake_reader)


def test_legacy_blank_audit_does_not_block_scene_disjoint_selection(tmp_path, selection, monkeypatch):
    import scripts.correction_models.selection as module
    monkeypatch.setattr(module, 'reference_identity', lambda *args: {'fixture': 'map'})
    inventory_path, audit_path, output = tmp_path/'inventory.json', tmp_path/'audit.json', tmp_path/'selection.json'
    save_verified(inventory_path, dict(rows=selection['rows'], errors=[]))
    old = dict(reviewed=False, method='', evidence='', scenes={
        s: dict(spatial_group=s, evidence='', footprint_wgs84=None)
        for s in selection['spatial_overlap_audit']['scenes']})
    write_once(audit_path, old)
    original = audit_path.read_bytes()
    prepare_selection(inventory_path, audit_path, output, sample_reader=fake_reader)
    result = load_verified(output)
    policy = result['spatial_overlap_audit']
    assert policy['reviewed'] is False and policy['geographic_overlap_verified'] is False
    assert policy['evaluation_scope'] == 'held_out_dataset_scenes'
    assert audit_path.read_bytes() == original
    validate_selection(result)


def test_missing_audit_can_use_official_scene_groups(tmp_path, selection, monkeypatch):
    import scripts.correction_models.selection as module
    monkeypatch.setattr(module, 'reference_identity', lambda *args: {'fixture': 'map'})
    source, output = tmp_path/'inventory.json', tmp_path/'selection.json'
    save_verified(source, dict(rows=selection['rows'], errors=[]))
    assert prepare_selection(source, tmp_path/'absent.json', output, sample_reader=fake_reader)['frames'] == 160
    assert load_verified(output)['source_audit_sha256'] is None


def test_readiness_explains_scene_with_four_eligible_frames_before_source_reads(tmp_path, selection):
    rows = copy.deepcopy(selection['rows'])
    for row in rows:
        if row['scene_id'] == 'Scene_12' and int(row['sample_id'].split('_')[-1]) >= 4:
            row['eligible'] = False
    data = dict(rows=rows, errors=[])
    status = selection_readiness(data)
    assert status['eligible_scene_count'] == 15 and status['minimum_additional_scenes'] == 1
    assert status['insufficient_scenes']['Scene_12']['eligible_frames'] == 4
    path = tmp_path/'inventory.json'
    save_verified(path, data)
    with pytest.raises(ValueError, match=r'Scene_12.*eligible_frames.*4'):
        prepare_selection(path, tmp_path/'missing.json', tmp_path/'selection.json')


def test_scene_policy_cannot_claim_verified_geography():
    policy = scene_policy(['Scene_01'])
    policy['geographic_overlap_verified'] = True
    with pytest.raises(ValueError, match='unverified'):
        audit_groups(policy, ['Scene_01'])


def test_exact_map_duplicates_cannot_cross_scene_splits(tmp_path, selection, monkeypatch):
    import scripts.correction_models.selection as module
    for row in selection['rows']:
        Path(row['reference_json']).write_text(json.dumps(dict(modes=dict(aerial=dict(map_path='map.png')))))
    def fingerprints(path, mode):
        scene = Path(path).parent.name
        return {'map.png': 'duplicate' if scene in ('Scene_01', 'Scene_02') else scene}
    monkeypatch.setattr(module, 'reference_identity', fingerprints)
    source = tmp_path/'inventory.json'
    save_verified(source, dict(rows=selection['rows'], errors=[]))
    with pytest.raises(ValueError, match='Need 16 eligible scene groups'):
        prepare_selection(source, None, tmp_path/'selection.json', sample_reader=fake_reader)


def test_old_selection_protocol_is_rejected(selection):
    selection['protocol_id'] = 'correction-models-v2'
    with pytest.raises(ValueError, match='protocol changed'):
        validate_selection(selection)


def test_notebook_reuses_completed_inventory_without_rescanning(tmp_path, selection, monkeypatch, capsys):
    import nbformat
    import scripts.correction_models.selection as module
    import scripts.correction_models.archives as archives
    source, audit = tmp_path/'inventory.json', tmp_path/'audit.json'
    save_verified(source, dict(rows=selection['rows'], errors=[]))
    write_once(audit, dict(reviewed=False, method='', evidence='', scenes={}))
    original = audit.read_bytes()
    monkeypatch.setattr(archives, 'inventory_snapshot_paths', lambda *args: (source, audit))
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed inventory must not reopen any NPZ')
    monkeypatch.setattr(module, 'inventory', forbidden)
    notebook = nbformat.read(Path(__file__).parents[1]/'notebooks/01_collect_colab.ipynb', as_version=4)
    cell = next(c.source for c in notebook.cells if c.cell_type == 'code'
                and 'Selection readiness:' in c.source)
    exec(compile(cell, '<inventory-cell>', 'exec'), dict(DATASET_ROOT='unused', DATA=tmp_path))
    assert 'Cached inventory: 160 frames' in capsys.readouterr().out
    assert audit.read_bytes() == original


def test_selection_cell_loads_frozen_selection_without_prior_cell_imports(tmp_path, selection):
    import nbformat
    path = tmp_path/'selection.json'
    save_verified(path, selection)
    notebook = nbformat.read(Path(__file__).parents[1]/'notebooks/01_collect_colab.ipynb', as_version=4)
    cell = next(c.source for c in notebook.cells if c.cell_type == 'code'
                and "print('Using frozen selection:'" in c.source)
    namespace = dict(SELECTION=path)
    exec(compile(cell, '<selection-cell>', 'exec'), namespace)
    assert namespace['frozen_selection'] == selection

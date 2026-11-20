from pathlib import Path
import zipfile

import pytest

from scripts.correction_models.archives import unpack_archives, inventory_snapshot_paths


def scene_zip(path, scene, *, flat=False, prefix=''):
    path.parent.mkdir(parents=True, exist_ok=True)
    number = scene.split('_')[1]
    parent = '' if flat else prefix+scene+'/'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(parent+f'L{number}_reference.json', '{}')
        archive.writestr(parent+f'L{number}_0001.npz', b'synthetic fixture, not training data')


def test_finds_four_archived_scenes_recursively_and_reports_deficit(tmp_path):
    root, dataset = tmp_path/'archives', tmp_path/'dataset'
    for i in range(1, 5):
        scene_zip(root/'nested'/f'Scene_{i:02d}.zip', f'Scene_{i:02d}', prefix='AnyVisLoc/')
    result = unpack_archives(root, dataset, progress=None)
    assert result['available_scene_count'] == 4
    assert result['minimum_additional_scene_count'] == 12
    assert result['newly_extracted_archives'] == 4
    assert len(list(root.rglob('*.zip'))) == 4  # Source archives retained.
    assert len(list(dataset.rglob('L??_reference.json'))) == 4


def test_flat_archive_is_extracted_into_named_scene(tmp_path):
    root, dataset = tmp_path/'archives', tmp_path/'dataset'
    scene_zip(root/'Scene_05.zip', 'Scene_05', flat=True)
    result = unpack_archives(root, dataset, progress=None)
    assert result['scenes'] == ['Scene_05']
    assert (dataset/'Scene_05/L05_reference.json').is_file()


def test_completed_extraction_is_not_repeated(tmp_path, monkeypatch):
    import scripts.correction_models.archives as module
    root, dataset = tmp_path/'archives', tmp_path/'dataset'
    scene_zip(root/'Scene_01.zip', 'Scene_01')
    unpack_archives(root, dataset, progress=None)
    def forbidden(*args):
        pytest.fail('Unchanged completed archive was extracted again')
    monkeypatch.setattr(module, 'stage_archive', forbidden)
    result = unpack_archives(root, dataset, progress=None)
    assert result['reused_archives'] == 1 and result['newly_extracted_archives'] == 0


def test_unsafe_zip_is_rejected_before_any_member_is_written(tmp_path):
    root, dataset = tmp_path/'archives', tmp_path/'dataset'
    root.mkdir()
    with zipfile.ZipFile(root/'Scene_01.zip', 'w') as archive:
        archive.writestr('Scene_01/L01_reference.json', '{}')
        archive.writestr('../escape.txt', 'unsafe')
    with pytest.raises(ValueError, match='Unsafe'):
        unpack_archives(root, dataset, progress=None)
    assert not (tmp_path/'escape.txt').exists()
    assert not (dataset/'Scene_01/L01_reference.json').exists()


def test_partial_extraction_resumes_and_conflicting_files_are_preserved(tmp_path):
    root, dataset = tmp_path/'archives', tmp_path/'dataset'
    scene_zip(root/'Scene_01.zip', 'Scene_01')
    target = dataset/'Scene_01/L01_reference.json'
    target.parent.mkdir(parents=True)
    target.write_text('{}')
    assert unpack_archives(root, dataset, progress=None)['newly_extracted_archives'] == 1
    # Without a completion receipt, the extractor checks existing files by CRC.
    other = tmp_path/'other'
    conflicting = other/'Scene_01/L01_reference.json'
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text('different source data')
    with pytest.raises(ValueError, match='Existing staged file changed'):
        unpack_archives(root, other, progress=None)
    assert conflicting.read_text() == 'different source data'


def test_newly_added_scene_gets_new_inventory_and_audit_paths(tmp_path):
    root, dataset = tmp_path/'archives', tmp_path/'dataset'
    scene_zip(root/'Scene_01.zip', 'Scene_01')
    unpack_archives(root, dataset, progress=None)
    original = inventory_snapshot_paths(dataset, tmp_path/'data')
    assert original == inventory_snapshot_paths(dataset, tmp_path/'data')
    scene_zip(root/'Scene_02.zip', 'Scene_02')
    unpack_archives(root, dataset, progress=None)
    assert original != inventory_snapshot_paths(dataset, tmp_path/'data')


def test_already_extracted_scenes_work_without_zips(tmp_path):
    root, dataset = tmp_path/'archives', tmp_path/'dataset'
    root.mkdir()
    ref = dataset/'Scene_01/L01_reference.json'
    ref.parent.mkdir(parents=True)
    ref.write_text('{}')
    result = unpack_archives(root, dataset, progress=None)
    assert result['available_scene_count'] == 1 and result['discovered_archives'] == 0

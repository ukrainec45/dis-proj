"""Exercise notebook transfer orchestration with fake URLs and no network calls."""
from pathlib import Path

import nbformat

from .test_archives import scene_zip


def transfer_cell():
    path = Path(__file__).parents[1]/'notebooks/01_collect_colab.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return next(c.source for c in notebook.cells if c.cell_type == 'code' and 'DOWNLOAD_MISSING_SCENES = False' in c.source)


def test_default_does_not_make_network_requests(tmp_path, monkeypatch):
    import scripts.visibility_localization.colab as shared
    import scripts.correction_models.download_access as access
    import scripts.correction_models.downloads as transfers
    def forbidden(*args):
        raise AssertionError('Default cell must not make a network request')
    monkeypatch.setattr(shared, 'official_scene_links', forbidden)
    monkeypatch.setattr(access, 'official_scene_links', forbidden)
    monkeypatch.setattr(shared, 'download_archive', forbidden)
    monkeypatch.setattr(transfers, 'download_archive', forbidden)
    exec(compile(transfer_cell(), '<download-cell>', 'exec'),
         dict(ARCHIVE_ROOT=tmp_path, DATASET_ROOT=tmp_path/'extracted'))


def test_transfer_skips_present_scenes_and_refreshes_each_download_url(tmp_path, monkeypatch):
    import scripts.visibility_localization.colab as shared
    import scripts.correction_models.download_access as access
    import scripts.correction_models.downloads as transfers
    present = {'Scene_05', 'Scene_08', 'Scene_11', 'Scene_21'}
    for scene in present:
        scene_zip(tmp_path/(scene+'.zip'), scene)
    catalog = {f'Scene_{i:02d}': {'url': f'https://example.invalid/stale/{i}'} for i in range(1, 22)}
    refreshed, downloads = [], []
    def links(scenes):
        if scenes is None:
            return catalog
        refreshed.append(scenes[0])
        return {scenes[0]: {'url': 'https://example.invalid/fresh/'+scenes[0]}}
    monkeypatch.setattr(shared, 'official_scene_links', links)
    monkeypatch.setattr(access, 'official_scene_links', links)
    def capture(url, path, *, url_provider):
        downloads.append((url_provider(), path))
    monkeypatch.setattr(transfers, 'download_archive', capture)
    cell = transfer_cell().replace('DOWNLOAD_MISSING_SCENES = False', 'DOWNLOAD_MISSING_SCENES = True')
    exec(compile(cell, '<download-cell>', 'exec'), dict(ARCHIVE_ROOT=tmp_path, DATASET_ROOT=tmp_path/'extracted'))
    assert len(downloads) == 12
    assert len(refreshed) == 12
    assert not present.intersection(refreshed)
    assert all('/fresh/' in url and path.parent == tmp_path for url, path in downloads)

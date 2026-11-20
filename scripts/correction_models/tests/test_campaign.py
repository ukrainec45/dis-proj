from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from scripts.correction_models.artifacts import save_verified, load_verified
from scripts.correction_models.config import main_config
from scripts.correction_models.campaign import run_collection
from scripts.correction_models.embeddings import cached_embedding, CachedCampEncoder
from .conftest import fake_reader, FakeEncoder


def test_paired_collection_bounded_resume_and_embedding_reuse(tmp_path, selection, reference):
    path = tmp_path/'selection.json'
    save_verified(path, selection)
    backend = SimpleNamespace(config=main_config(), identity={'synthetic': True})
    calls = []
    encoder = FakeEncoder()

    def localizer(backend, image, K, dist, ref, prior, mode, seed):
        calls.append((prior.xyz.copy(), image.copy(), seed))
        return dict(accepted=False, status='inference_complete', pred_xyz_enu=None)

    kwargs = dict(sample_reader=fake_reader, reference_reader=lambda *args: reference,
                  reference_fingerprint=lambda *args: {'fixture': 'map'}, localizer=localizer, encoder=encoder)
    out = tmp_path/'out'
    first = run_collection(path, backend, out, max_new_attempts=1, **kwargs)
    assert first['completed_attempts'] == 1 and not first['complete']
    second = run_collection(path, backend, out, max_new_attempts=2, **kwargs)
    assert second['completed_attempts'] == 3 and len(calls) == 3
    assert encoder.calls == 1  # Same prior-centred map, three visibility conditions.
    np.testing.assert_array_equal(calls[0][0], calls[2][0])
    assert calls[0][2] == calls[2][2]
    assert not np.array_equal(calls[0][1], calls[1][1])
    rows = [load_verified(p) for p in (out/'records').glob('*.json')]
    assert len({r['prior_seed'] for r in rows}) == 1
    assert all('query' in r and 'feature_profile' in r and 'localization_profile' in r for r in rows)
    assert not (out/'dataset.json').exists()


def test_first_technical_error_saved_and_never_becomes_rejection(tmp_path, selection, reference):
    path = tmp_path/'selection.json'
    save_verified(path, selection)
    backend = SimpleNamespace(config=main_config(), identity={})
    def broken(*args):
        raise RuntimeError('synthetic GPU failure')
    kwargs = dict(sample_reader=fake_reader, reference_reader=lambda *args: reference,
                  reference_fingerprint=lambda *args: {'fixture': 'map'}, localizer=broken, encoder=FakeEncoder())
    with pytest.raises(RuntimeError, match='synthetic GPU failure'):
        run_collection(path, backend, tmp_path/'out', max_new_attempts=20, **kwargs)
    records = list((tmp_path/'out/records').glob('*.json'))
    assert len(records) == 1
    row = load_verified(records[0])
    assert row['label']['technical_error'] and row['label']['class_index'] is None
    assert 'Traceback' in row['localization']['traceback']
    with pytest.raises(RuntimeError, match='synthetic GPU failure'):
        run_collection(path, backend, tmp_path/'out', max_new_attempts=20, **kwargs)
    assert len(list((tmp_path/'out/records').glob('*.json'))) == 1


def test_changed_sources_and_cache_are_rejected(tmp_path, selection, reference):
    path = tmp_path/'selection.json'
    save_verified(path, selection)
    backend = SimpleNamespace(config=main_config(), identity={})
    kwargs = dict(sample_reader=fake_reader, reference_reader=lambda *args: reference,
        reference_fingerprint=lambda *args: {'fixture': 'map'}, encoder=FakeEncoder(),
        localizer=lambda *args: dict(accepted=False, status='inference_complete', pred_xyz_enu=None))
    run_collection(path, backend, tmp_path/'out', max_new_attempts=1, **kwargs)
    asset = next((tmp_path/'out/embeddings').glob('*.json'))
    asset.write_text('{}')
    with pytest.raises(ValueError, match='asset changed'):
        run_collection(path, backend, tmp_path/'out', max_new_attempts=1, **kwargs)


def test_invalid_accepted_coordinates_are_saved_as_technical_error(tmp_path, selection, reference):
    path = tmp_path/'selection.json'
    save_verified(path, selection)
    backend = SimpleNamespace(config=main_config(), identity={})
    with pytest.raises(RuntimeError, match='Collection stopped'):
        run_collection(path, backend, tmp_path/'out', max_new_attempts=1, sample_reader=fake_reader,
            reference_reader=lambda *args: reference, reference_fingerprint=lambda *args: {'fixture': 'map'},
            encoder=FakeEncoder(), localizer=lambda *args: dict(accepted=True, pred_xyz_enu=None))
    row = load_verified(next((tmp_path/'out/records').glob('*.json')))
    assert row['label']['technical_error'] and row['label']['class_index'] is None


def test_exact_embedding_cache_is_usable_without_torch(tmp_path):
    encoder = FakeEncoder()
    patch = np.zeros((256, 256, 3), np.uint8)
    expected, meta = cached_embedding(encoder, patch, tmp_path)
    actual = CachedCampEncoder(tmp_path, encoder.identity)(patch)
    np.testing.assert_array_equal(actual, expected)
    cached_embedding(encoder, patch, tmp_path)
    assert encoder.calls == 1
    with pytest.raises(ValueError, match='not_cached'):
        CachedCampEncoder(tmp_path, encoder.identity)(patch+1)


def test_reference_fingerprint_cache_avoids_rehashing_unchanged_maps(tmp_path, monkeypatch):
    import scripts.correction_models.fingerprints as module
    root = tmp_path/'reference'
    root.mkdir()
    (root/'map.bin').write_bytes(b'synthetic map')
    (root/'dsm.bin').write_bytes(b'synthetic dsm')
    ref = root/'L01_reference.json'
    ref.write_text(json.dumps(dict(modes=dict(aerial=dict(map_path='map.bin', dsm_path='dsm.bin')))))
    original, calls = module.file_hash, []
    def counted(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(module, 'file_hash', counted)
    first = module.cached_reference_identity(ref, 'aerial', tmp_path/'cache')
    assert len(calls) == 3
    assert module.cached_reference_identity(ref, 'aerial', tmp_path/'cache') == first
    assert len(calls) == 3
    (root/'map.bin').write_bytes(b'a different synthetic map')
    assert module.cached_reference_identity(ref, 'aerial', tmp_path/'cache') != first
    assert len(calls) == 6

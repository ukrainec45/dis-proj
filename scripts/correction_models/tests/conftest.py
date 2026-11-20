"""Entirely synthetic fixtures; never empirical localization evidence."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.correction_models.artifacts import save_verified, digest, manifest
from scripts.correction_models.config import main_config
from scripts.correction_models.contracts import PROTOCOL_ID, CONDITIONS, label_outcome
from scripts.correction_models.features import ALL_FEATURES
from scripts.visibility_localization.core import file_hash
from scripts.visibility_localization.data import Reference


@pytest.fixture
def reference():
    return Reference('synthetic', np.random.default_rng(1).integers(0, 255, (1200, 1200, 3), dtype=np.uint8),
        np.zeros((1200, 1200)), np.ones(2), np.zeros(2), np.ones(2), np.zeros(2), 0., {'fixture': 'map'})


@pytest.fixture
def selection(tmp_path):
    rows, splits = [], {'train': [], 'validation': [], 'calibration': [], 'test': []}
    audit = dict(reviewed=True, method='synthetic test polygons', evidence='NOT REAL GEOGRAPHY', scenes={})
    for i in range(1, 17):
        scene = f'Scene_{i:02d}'
        split = 'train' if i <= 8 else 'validation' if i <= 10 else 'calibration' if i <= 12 else 'test'
        splits[split].append(scene)
        audit['scenes'][scene] = dict(spatial_group=scene, evidence='synthetic test fixture', footprint_wgs84=None)
        directory = tmp_path/'sources'/scene
        directory.mkdir(parents=True)
        ref = directory/'L01_reference.json'
        ref.write_text('{}')
        for j in range(10):
            path = directory/f'L01_{j:04d}.npz'
            path.write_bytes(b'synthetic input')
            rows.append(dict(scene_id=scene, sample_id=path.stem, path=str(path), reference_json=str(ref),
                xyz_enu=[500.+j, -500., 80.], nadir_deg=0., eligible=True, split=split, spatial_group=scene,
                source_identity={path.name: file_hash(path)}))
    return dict(protocol_id=PROTOCOL_ID, rows=rows, split_scenes=splits, references={s: {'fixture': 'map'} for s in audit['scenes']},
                spatial_overlap_audit=audit, expected_attempts=1920)


def fake_reader(path, *args):
    path = Path(path)
    j = int(path.stem.split('_')[-1])
    return dict(scene_id=path.parent.name, sample_id=path.stem, rgb=np.zeros((32, 32, 3), np.uint8),
        xyz_enu=np.array([500.+j, -500., 80.]), K=np.array([[20., 0, 16], [0, 20, 16], [0, 0, 1]]),
        dist=np.zeros(5), rotation=np.diag([1., -1., -1.]), nadir_deg=0.,
        source_identity={path.name: file_hash(path)})


class FakeEncoder:
    identity = {'version': 'synthetic-test-only'}

    def __init__(self):
        self.calls = 0

    def __call__(self, patch):
        self.calls += 1
        return np.array([patch.mean()/255, patch.std()/255, 1.])


def small_inputs(n=64):
    inputs, outcomes = [], []
    for i in range(n):
        cls = i % 8
        # Clear boundaries: reject; -21,-10,-2,0,+2,+10,+30 metres.
        delta = [0., -21., -10., -2., 0., 2., 10., 30.][cls]
        outcomes.append(label_outcome([200, 0, 0], [200-delta, 0, 0], [0, 0, 0], cls != 0))
        features = dict.fromkeys(ALL_FEATURES, 0.)
        features.update(sigma_major_m=float(5+i % 4), sigma_minor_m=5., agl_m=80.,
                        gray_std=cls/7, fog_extreme=float(i % 3 == 1), smoke_extreme=float(i % 3 == 2))
        inputs.append(dict(features=features, camp=[cls/7, i/n, 1.]))
    return inputs, outcomes


@pytest.fixture
def complete_dataset(tmp_path, selection):
    root = tmp_path/'campaign'
    rows = []
    source_inputs, source_outcomes = small_inputs(1920)
    index = 0
    for item in selection['rows']:
        for sigma in (5., 20., 50., 100.):
            for condition in CONDITIONS:
                value = source_inputs[index]
                features = dict(value['features'], sigma_major_m=sigma, sigma_minor_m=sigma,
                                gray_std=((index//3) % 8)/7,
                                fog_extreme=float(condition == 'fog_extreme'), smoke_extreme=float(condition == 'smoke_extreme'))
                vector = [features['gray_std'], sigma/100, 1.]
                key = digest(vector)
                path = root/'embeddings'/(key+'.json')
                if not path.exists():
                    save_verified(path, dict(vector=vector, encoder={'version': 'synthetic-test-only'}, cache_key=key))
                rows.append(dict(attempt_id=str(index), campaign_id='synthetic', scene_id=item['scene_id'],
                    sample_id=item['sample_id'], spatial_group=item['spatial_group'], split=item['split'],
                    condition=condition, sigma_xy_m=sigma, prior_seed=42, prior_metadata={'synthetic': True},
                    features=features, label=source_outcomes[index], map_context=None,
                    embedding=dict(path='embeddings/'+path.name, sha256=file_hash(path), dimension=3)))
                index += 1
    provenance = manifest(stage='main', references=selection['references'],
                          encoder={'version': 'synthetic-test-only'}, localizer_settings=main_config())
    path = root/'dataset.json'
    save_verified(path, dict(protocol_id=PROTOCOL_ID, provenance=provenance, rows=rows))
    return path

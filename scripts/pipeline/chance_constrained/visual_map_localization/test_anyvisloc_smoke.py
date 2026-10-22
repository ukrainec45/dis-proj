"""Synthetic software checks only: no AnyVisLoc data or notebook execution."""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from .anyvisloc import dataset_to_enu, load_reference, load_sample
from .anyvisloc_smoke import checked_trial, run_smoke
from .planar import covariance_extent, ground_offsets, rectify, search_crop
from .smoke_localizer import localize


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[4] / 'configs/chance_constrained/vml/anyvisloc_smoke.v1.json'
    result = json.loads(path.read_text(encoding='utf-8-sig'))
    result['pose_world_frame'] = 'enu'  # Existing fixture uses explicit ENU camera poses.
    return result


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'dataset'
    scene = root / 'Scene_01'
    scene.mkdir(parents=True)
    rng = np.random.default_rng(14)
    reference = rng.integers(0, 256, (800, 800, 3), dtype=np.uint8)
    for _ in range(100):
        cv2.circle(reference, tuple(map(int, rng.integers(10, 790, 2))),
                   int(rng.integers(3, 10)), (255, 255, 255), -1)
    mode = dict(map_path='map.png', dsm_path='dsm.npy', map_resolution=[.5, .5],
                dsm_resolution=[.5, .5], map_origin_local=[0, 0], dsm_origin_local=[0, 0])
    (scene / 'L01_reference.json').write_text(json.dumps(dict(modes=dict(aerial=mode, satellite=mode))))
    cv2.imwrite(str(scene / 'map.png'), cv2.cvtColor(reference, cv2.COLOR_RGB2BGR))
    np.save(scene / 'dsm.npy', np.zeros((800, 800)))
    pose = np.diag([1., -1., -1., 1.])
    pose[:3, 3] = [200, -200, 60]
    for i in range(21):
        altitude = 10 if i == 0 else 60
        p = pose.copy()
        p[2, 3] = altitude
        np.savez(scene / f'L01_{i:04d}.npz', image=reference[320:480, 320:480],
                 K=np.array([[120., 0, 80], [0, 120, 80], [0, 0, 1]]),
                 dist=np.zeros(5), pose_c2w=p, xyz=[200, 200, altitude])
    return root


def test_adapter_frames_and_relative_dsm_origin(dataset):
    np.testing.assert_array_equal(dataset_to_enu([1, 2, 3]), [1, -2, 3])
    ref_path = dataset / 'Scene_01/L01_reference.json'
    reference = load_reference(ref_path, 'aerial')
    assert reference.transform * (20, 30) == (10, -15)
    sample = load_sample(dataset / 'Scene_01/L01_0001.npz', pose_world_frame='enu')
    np.testing.assert_array_equal(sample['xyz'], [200, -200, 60])
    with pytest.raises(ValueError, match='translation'):
        load_sample(dataset / 'Scene_01/L01_0001.npz', pose_world_frame='dataset')
    raw = json.loads(ref_path.read_text())
    raw['modes']['aerial']['map_origin_local'] = [10, 20]
    raw['modes']['aerial']['dsm_origin_local'] = [2, 3]
    ref_path.write_text(json.dumps(raw))
    ref = load_reference(ref_path, 'aerial')
    assert ref.dsm_transform * (0, 0) == (12, -23)
    with pytest.raises(ValueError, match='dsm_outside'):
        ref.elevation([-100, 100])


def test_covariance_crop_is_footprint_plus_ellipse(dataset):
    mass = 1-np.exp(-2)
    np.testing.assert_allclose(covariance_extent([[4, 1], [1, 9]], mass), [4, 6])
    for invalid in ([[1, 2], [2, 1]], [[1, 2], [0, 1]]):
        with pytest.raises(ValueError):
            covariance_extent(invalid, .99)
    with pytest.raises(ValueError):
        covariance_extent(np.eye(2), 1)
    ref = load_reference(dataset / 'Scene_01/L01_reference.json', 'aerial')
    footprint = np.array([[-10, -10], [10, 10]])
    crop, transform, coverage = search_crop(ref, footprint, [120, -120],
                                           np.diag([4, 9]), mass, 1000000)
    assert coverage == 1
    assert transform * (0, 0) == pytest.approx((106, -104))
    assert crop.shape[:2] == (65, 57)


def test_planar_nadir_oblique_and_horizon(config):
    K = np.array([[100., 0, 40], [0, 100, 40], [0, 0, 1]])
    rotation = np.diag([1., -1., -1.])
    pixels = np.array([[40., 40.], [50, 40]])
    np.testing.assert_allclose(ground_offsets(pixels, K, np.zeros(5), rotation, 60),
                               [[0, 0], [6, 0]], atol=1e-12)
    image = np.full((80, 80, 3), 127, np.uint8)
    _, _, footprint, anchor, condition = rectify(image, K, np.zeros(5), rotation,
                                                 60, .6, 100000, 100)
    np.testing.assert_allclose(anchor, [40, 40], atol=1e-5)
    assert condition == pytest.approx(1)
    with pytest.raises(ValueError, match='horizon'):
        ground_offsets(pixels, K, np.zeros(5), np.eye(3), 60)
    # Tilted planar rectification is geometric, without an angle cutoff.
    tilt, _ = cv2.Rodrigues(np.array([.4, 0., 0.]))
    _, _, _, _, condition = rectify(image, K, np.zeros(5), tilt @ rotation,
                                     60, .6, 100000, 100)
    assert np.isfinite(condition)


def test_smoke_end_to_end_and_resume(dataset, tmp_path, config, monkeypatch):
    output = tmp_path / 'output'
    summary = run_smoke(dataset, output, config)
    assert summary['aerial']['trial_count'] == 20
    assert summary['satellite']['accepted_count'] == 20
    manifest = json.loads((output / 'manifest.json').read_text())
    assert not manifest['R_calibrated']
    assert manifest['validation_passed'] == dict(aerial=False, satellite=False)
    trials = list((output / 'shards').rglob('L01_*.json'))
    assert len(trials) == 40
    row = checked_trial(trials[0])
    result = row['result']
    assert result['component_mask'] == [True, True, False]
    assert np.linalg.eigvalsh(result['R']).min() > 0
    assert np.linalg.norm(row['residual_xy_m']) < 1
    assert row['outcome_component'] is None
    from . import anyvisloc_smoke
    def forbidden(*args, **kwargs):
        pytest.fail('Completed trial was executed again')
    monkeypatch.setattr(anyvisloc_smoke, 'localize', forbidden)
    assert run_smoke(dataset, output, config) == summary
    changed = copy.deepcopy(config)
    changed['seed'] += 1
    with pytest.raises(ValueError, match='changed'):
        run_smoke(dataset, output, changed)
    envelope = json.loads(trials[0].read_text())
    envelope['trial']['agl_m'] = 99
    trials[0].write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='checksum'):
        run_smoke(dataset, output, config)


def test_uniform_image_rejects(dataset, config):
    sample = load_sample(dataset / 'Scene_01/L01_0001.npz', pose_world_frame='enu')
    reference = load_reference(dataset / 'Scene_01/L01_reference.json', 'aerial')
    result = localize(np.full_like(sample['image'], 127), sample['K'], sample['dist'],
                      sample['rotation'], 60, [120, -120], np.eye(2)*400,
                      config['search_probability_mass'], reference, config)
    assert result['status'] == 'reject'
    assert result['z'] is None and result['R'] is None


def test_resume_after_interruption(dataset, tmp_path, config, monkeypatch):
    from . import anyvisloc_smoke
    original = anyvisloc_smoke.localize
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError('simulated disconnect')
        return original(*args, **kwargs)
    monkeypatch.setattr(anyvisloc_smoke, 'localize', interrupted)
    output = tmp_path / 'interrupted'
    with pytest.raises(RuntimeError, match='disconnect'):
        run_smoke(dataset, output, config)
    assert len(list((output / 'shards').rglob('L01_*.json'))) == 2
    recovered_calls = []
    def recovered(*args, **kwargs):
        recovered_calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(anyvisloc_smoke, 'localize', recovered)
    result = run_smoke(dataset, output, config)
    assert len(recovered_calls) == 38
    assert result['satellite']['trial_count'] == 20
    aerial = {checked_trial(p)['sample']: checked_trial(p)['prior_mean_xy']
              for p in (output / 'shards').rglob('aerial/sift/smoke/L01_*.json')}
    satellite = {checked_trial(p)['sample']: checked_trial(p)['prior_mean_xy']
                 for p in (output / 'shards').rglob('satellite/sift/smoke/L01_*.json')}
    assert aerial == satellite and len(aerial) == 20


def test_release_attitude_conversion_nadir_and_pitch():
    from .anyvisloc import anyvisloc_attitude_to_enu
    nadir = anyvisloc_attitude_to_enu(np.eye(3), [0, 0, 0])
    np.testing.assert_allclose(nadir, np.diag([1., -1., -1.]))
    K = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1.]])
    pitch, yaw = np.radians([20., 70.])
    cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    raw = np.array([[cy*cp, -sy, cy*sp], [sy*cp, cy, sy*sp], [-sp, 0, cp]])
    rotation = anyvisloc_attitude_to_enu(raw, [0, 20, 70])
    assert np.linalg.det(rotation) == pytest.approx(1)
    offset = ground_offsets([[50., 50.]], K, np.zeros(5), rotation, 60)[0]
    # Upstream zero-roll footprint: East=h*tan(p)*sin(yaw), North=h*tan(p)*cos(yaw).
    np.testing.assert_allclose(offset, 60*np.tan(pitch)*np.array([sy, cy]), atol=1e-10)
    with pytest.raises(ValueError, match='Euler attitude'):
        anyvisloc_attitude_to_enu(raw, [0, 0, 0])


def test_release_pose_translation_and_loader(dataset):
    path = dataset / 'Scene_01/L01_0001.npz'
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    values['pose_c2w'] = np.eye(4)
    values['pose_c2w'][:3, 3] = values['xyz']
    values['euler_deg'] = np.zeros(3)
    np.savez(path, **values)
    sample = load_sample(path, pose_world_frame='anyvisloc_ned_attitude')
    np.testing.assert_allclose(sample['xyz'], [200, -200, 60])
    np.testing.assert_allclose(sample['rotation'], np.diag([1., -1., -1.]))
    values['pose_c2w'][0, 3] += 1
    np.savez(path, **values)
    with pytest.raises(ValueError, match='translation'):
        load_sample(path, pose_world_frame='anyvisloc_ned_attitude')


def test_search_crop_limits_matching_grid_and_preserves_pixel_centres(dataset):
    reference = load_reference(dataset / 'Scene_01/L01_reference.json', 'aerial')
    footprint = np.array([[-100., -100.], [100., 100.]])
    arguments = (reference, footprint, np.array([200., -200.]), np.zeros((2, 2)), .99, 20000)
    with pytest.raises(ValueError, match='search_size_unsupported'):
        search_crop(*arguments)
    crop, transform, coverage = search_crop(*arguments, matching_gsd_m=2.)
    assert crop.shape[:2] == (101, 101)
    assert coverage == 1
    assert transform * (50, 50) == pytest.approx((200., -200.))
    assert 0 < transform.a <= 2 and -2 <= transform.e < 0
    # Pixel-centre-aware transform covers exactly the native pixel-edge bounds.
    assert transform * (-.5, -.5) == pytest.approx((99.75, -99.75))
    assert transform * (100.5, 100.5) == pytest.approx((300.25, -300.25))
    with pytest.raises(ValueError, match='search_size_unsupported'):
        search_crop(*arguments, matching_gsd_m=.1)

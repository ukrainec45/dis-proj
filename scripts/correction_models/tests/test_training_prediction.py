import copy
from dataclasses import replace

import numpy as np
import pytest

from scripts.correction_models.artifacts import load_verified, read, save_verified, digest
from scripts.correction_models.contracts import CorrectionQuery, CLASS_NAMES
from scripts.correction_models.dataset import load_dataset, validate_rows
from scripts.correction_models.features import extract_features
from scripts.correction_models.models import fit_model
from scripts.correction_models.prediction import Predictor
from .conftest import small_inputs


def test_main_dataset_complete_pairing_and_no_leakage(complete_dataset):
    data = load_dataset(complete_dataset)
    assert len(data['rows']) == 1920
    bad = copy.deepcopy(data['rows'])
    bad[1]['prior_seed'] = 999
    with pytest.raises(ValueError, match='share the same prior'):
        validate_rows(bad, require_complete=True)
    bad = copy.deepcopy(data['rows'])
    bad[0]['label']['error_after_xy_m'] = [999, 0]
    with pytest.raises(ValueError, match='vector/norm'):
        validate_rows(bad, require_complete=True)


def test_training_freeze_final_test_and_json_export(complete_dataset, tmp_path, monkeypatch):
    import scripts.correction_models.training as training
    import scripts.correction_models.models as models
    keys = ('constant', 'forest_geometry', 'forest_map', 'forest_camp', 'forest_map_no_visibility')
    monkeypatch.setattr(training, 'CANDIDATES', {k: models.CANDIDATES[k] for k in keys})
    fit = models.fit_model
    monkeypatch.setattr(training, 'fit_model', lambda *a, **k: fit(*a, **k, forest_parameters={'n_estimators': 3}))
    run = tmp_path/'run'
    summary = training.run_training(complete_dataset, run)
    assert summary['final_test_evaluated'] is False
    assert summary['split_counts']['train']['total'] == 960
    assert set(summary['grouped'][summary['selected']]['scene_id']) == {'Scene_09', 'Scene_10'}
    assert (run/'calibration.png').is_file() and (run/'scores.csv').is_file()
    frozen_before = (run/'frozen_run.json').read_bytes()
    final = training.evaluate_frozen(run, complete_dataset, tmp_path/'final')
    assert final['selected'] == summary['selected']
    assert final['held_out_test_scenes'] == 4
    assert final['geographic_overlap_verified'] is False
    assert final['supported_count'] == 480
    assert all(r['exploratory'] for r in final['comparison_to_constant'].values())
    assert (run/'frozen_run.json').read_bytes() == frozen_before
    with pytest.raises(FileExistsError, match='Final test'):
        training.evaluate_frozen(run, complete_dataset, tmp_path/'different-final')
    with pytest.raises(FileExistsError):
        training.run_training(complete_dataset, run)


def test_predictor_uses_only_map_query_and_declares_rejection_semantics(reference):
    query = CorrectionQuery(position=[500, -500, 80], crs='LOCAL_ENU', camera_to_enu=np.diag([1., -1., -1.]),
        K=[[20, 0, 16], [0, 20, 16], [0, 0, 1]], dist=np.zeros(5), image_shape=(32, 32),
        covariance_xy=np.eye(2)*25, yaw_sigma_deg=5, tilt_sigma_deg=2, z_sigma_m=5,
        scenario='clear', map_id=digest(reference.identity))
    features, _, _ = extract_features(query, reference)
    x, outcomes = small_inputs()
    x = [dict(features=features, camp=[]) for _ in outcomes]
    model = fit_model('forest_geometry', x, outcomes, forest_parameters={'n_estimators': 3})
    model.artifact['localizer_settings'] = {'synthetic': True}
    predictor = Predictor(model, reference, audited_map_ids=[query.map_id])
    answer = predictor.predict_correction(query)
    assert set(answer['probabilities']) == set(CLASS_NAMES)
    assert answer['delivery_probability'] is None and answer['flight_validation_passed'] is False
    assert answer['rejection']['action'] == 'retain incoming position estimate'
    dist = answer['accepted_error_distribution']
    assert np.isclose(sum(dist['weights'])+answer['rejection']['probability'], 1)
    assert predictor.predict_correction(replace(query, map_id='other'))['probabilities'] is None
    assert Predictor(model, reference).predict_correction(query)['reason'] == 'unverified_map_domain'
    anisotropic = replace(query, covariance_xy=np.diag([25., 36.]))
    assert predictor.predict_correction(anisotropic)['reason'] == 'anisotropic_prior_not_validated'


def test_metric_raster_reads_bounded_window(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from scripts.correction_models.raster import RasterMap
    image, dem = tmp_path/'map.tif', tmp_path/'dem.tif'
    transform = from_origin(0, 1000, .5, .5)
    with rasterio.open(image, 'w', driver='GTiff', height=2000, width=2000, count=3,
                       dtype='uint8', crs='EPSG:32635', transform=transform) as dst:
        dst.write(np.random.default_rng(1).integers(1, 255, (3, 2000, 2000), dtype=np.uint8))
    with rasterio.open(dem, 'w', driver='GTiff', height=2000, width=2000, count=1,
                       dtype='float32', crs='EPSG:32635', transform=transform) as dst:
        dst.write(np.zeros((1, 2000, 2000), np.float32))
    reference = RasterMap(image, dem, max_window_side=256)
    query = CorrectionQuery(position=[500, 500, 80], crs='EPSG:32635', camera_to_enu=np.diag([1., -1., -1.]),
        K=[[20, 0, 16], [0, 20, 16], [0, 0, 1]], dist=np.zeros(5), image_shape=(32, 32),
        covariance_xy=np.eye(2)*10000, yaw_sigma_deg=5, tilt_sigma_deg=2, z_sigma_m=5,
        scenario='clear', map_id=reference.map_id)
    window = reference.window(query)
    assert max(window.bgr.shape[:2]) <= 256
    features, patch, _ = extract_features(query, window)
    assert features['map_gsd_x_m'] == .5
    assert patch.shape == (256, 256, 3)

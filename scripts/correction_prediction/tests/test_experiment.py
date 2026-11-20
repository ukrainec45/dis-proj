"""Synthetic correctness checks; no network, GPU, empirical results or secrets."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.correction_prediction.artifacts import digest, load_verified, save_verified, read
from scripts.correction_prediction.contracts import CorrectionQuery, CLASS_NAMES, label_outcome, probabilities, derived
from scripts.correction_prediction.features import ALL_FEATURES, extract_features
from scripts.correction_prediction.models import fit_model, FrozenModel, fit_temperature, temperature_scale
from scripts.correction_prediction.metrics import evaluate, quantile_interval, paired_cluster_interval
from scripts.correction_prediction.dataset import validate_rows, save_dataset, load_dataset
from scripts.correction_prediction.jev import build_request, parse_response, JevClient
from scripts.correction_prediction.campaign import pilot_config, run_campaign, localizer_settings
from scripts.correction_prediction.planning import timed_route_queries, Predictor, RasterReference, saved_route_queries
from scripts.correction_prediction.training import run_training, evaluate_frozen
from scripts.correction_prediction.opportunities import compare_opportunities


@pytest.mark.parametrize('delta,expected', [(-21, 1), (-20, 2), (-5, 3), (-1, 4),
                                         (0, 4), (1, 5), (5, 6), (20, 7), (1000, 7)])
def test_boundary_labels(delta, expected):
    label = label_outcome([2000, 0, 0], [2000-delta, 0, 0], [0, 0, 0], True)
    assert label['class_index'] == expected
    assert label['delta_m'] == delta


def test_rejection_ignores_diagnostic_pose_and_technical_is_not_rejection():
    r = label_outcome([3, 4, 0], [0, 0, 0], [0, 0, 0], False)
    assert r['class_index'] == 0 and r['delta_m'] == 0 and r['error_after_m'] == 5
    assert label_outcome([0, 0, 0], None, [0, 0, 0], False, technical_error=True)['class_index'] is None
    with pytest.raises(ValueError):
        label_outcome([0, 0, 0], None, [0, 0, 0], True)


@pytest.fixture
def query():
    return CorrectionQuery(position=[500, -500, 80], crs='LOCAL_ENU',
        camera_to_enu=np.diag([1, -1, -1]), K=[[20, 0, 16], [0, 20, 16], [0, 0, 1]],
        dist=np.zeros(5), image_shape=(32, 32), covariance_xy=np.eye(2)*25,
        yaw_sigma_deg=5, tilt_sigma_deg=2, z_sigma_m=5, scenario='clear', map_id='map')


@pytest.fixture
def reference():
    from scripts.visibility_localization.data import Reference
    rng = np.random.default_rng(1)
    return Reference('fixture', rng.integers(0, 255, (1000, 1000, 3), dtype=np.uint8),
        np.zeros((1000, 1000)), np.ones(2), np.zeros(2), np.ones(2), np.zeros(2), 0, {'map': 'fixture'})


def test_query_validation_and_map_features_are_gt_free(query, reference):
    f, image, _ = extract_features(query, reference)
    assert set(f) == set(ALL_FEATURES) and image.shape == (256, 256, 3)
    assert f['map_coverage'] > .99 and f['entropy_bits'] > 0
    q2 = replace(query, scenario='fog_extreme')
    f2, image2, _ = extract_features(q2, reference)
    assert f2['fog_extreme'] == 1
    np.testing.assert_array_equal(image, image2)
    for key in set(ALL_FEATURES)-{'fog_extreme'}:
        assert f[key] == f2[key]
    with pytest.raises(ValueError, match='metric'):
        replace(query, crs='EPSG:4326')
    with pytest.raises(ValueError, match='positive definite'):
        replace(query, covariance_xy=[[-1, 0], [0, 1]])
    with pytest.raises(ValueError, match='unsupported_scenario'):
        extract_features(replace(query, scenario='new_weather'), reference)
    with pytest.raises(ValueError, match='no_map_coverage'):
        extract_features(replace(query, position=[10000, 10000, 80]), reference)


def test_probability_validation_and_event_sums():
    p = np.ones((1, 8))/8
    assert derived(p)['accept'][0] == 7/8
    assert derived(p)['improve'][0] == 3/8
    for bad in [np.zeros((1, 8)), np.ones((1, 7))/7, [[np.nan]*8]]:
        with pytest.raises(ValueError):
            probabilities(bad)


def test_proper_scores_quantiles_and_cluster_limits():
    p = np.eye(8)
    result = evaluate(p, np.arange(8))
    assert result['log_loss'] == 0 and result['brier_score'] == 0 and result['conditional_accepted_rps'] == 0
    assert quantile_interval([p[7]])[0] == dict(lower_m=20, upper_m=None)
    assert quantile_interval([p[0]]) == [None]
    assert paired_cluster_interval(p, p, np.arange(8), ['same']*8)['interval'] is None
    assert paired_cluster_interval(p, p, np.arange(8), ['a']*4+['b']*4)['interval'] == [0, 0]


def numeric_rows(n=80):
    rng = np.random.default_rng(42)
    x = [{k: float(v) for k, v in zip(ALL_FEATURES, rng.normal(size=len(ALL_FEATURES)))} for _ in range(n)]
    y = np.arange(n) % 8
    for i in range(n):
        x[i]['sigma_major_m'] = float(y[i])
    return x, y


@pytest.mark.parametrize('name', ['constant', 'logistic_geometry', 'logistic_map', 'forest_geometry', 'forest_map'])
def test_model_serialization_and_calibration(name, tmp_path):
    x, y = numeric_rows()
    model = fit_model(name, x, y)
    p = model.predict(x, False)
    assert p.shape == (80, 8)
    model.artifact['temperature'] = fit_temperature(p, y)
    model.save(tmp_path/'model.json')
    loaded = FrozenModel.load(tmp_path/'model.json')
    np.testing.assert_allclose(loaded.predict(x), model.predict(x))
    assert not loaded.support(x).any()
    if model.artifact['features']:
        q = dict(x[0], **{model.artifact['features'][0]: 1e9})
        assert loaded.support([q])[0]
    with pytest.raises(FileExistsError):
        model.save(tmp_path/'model.json')


def test_binary_and_missing_class_models():
    x, _ = numeric_rows(20)
    for y in [np.arange(20) % 2, np.zeros(20, dtype=int)]:
        m = fit_model('logistic_map', x, y)
        probabilities(m.predict(x))
        assert m.artifact['missing_classes']


def test_temperature_does_not_worsen_calibration_objective():
    y = np.arange(80) % 8
    p = np.ones((80, 8))*.01
    p[:, 0] = .93
    t = fit_temperature(p, y)
    assert evaluate(temperature_scale(p, t), y)['log_loss'] < evaluate(p, y)['log_loss']


def make_dataset():
    features, labels = numeric_rows(128)
    rows = []
    deltas = [0, -30, -10, -2, 0, 2, 10, 30]
    for i, (f, cls) in enumerate(zip(features, labels)):
        split = ['train', 'validation', 'calibration', 'test'][i//32]
        delta = deltas[cls]
        label = label_outcome([50, 0, 0], [50-delta, 0, 0], [0, 0, 0], bool(cls))
        rows.append(dict(attempt_id=str(i), features=f, label=label, split=split,
            scene_id='fresh_'+split, sample_id='frame_'+str(i), spatial_group='area_'+split,
            opportunity_group='episode_'+split, condition='clear', sigma_xy_m=20., trial=0,
            map_context=None))
    return rows


def test_spatial_leakage_and_reviewed_test_rejected():
    rows = make_dataset()
    validate_rows(rows, require_main=True)
    rows[0]['spatial_group'] = rows[-1]['spatial_group']
    with pytest.raises(ValueError, match='leakage'):
        validate_rows(rows)
    rows = make_dataset()
    rows[-1]['scene_id'] = 'Scene_21'
    with pytest.raises(ValueError, match='reviewed'):
        validate_rows(rows)


def test_development_freeze_and_independent_evaluation(tmp_path):
    rows = make_dataset()
    provenance = dict(stage='main', main_design={'approved_attempts': len(rows)},
                      references={'fixture': {'sha': 'map'}}, localizer_settings=localizer_settings(pilot_config()))
    dataset = tmp_path/'dataset.json'
    save_dataset(dataset, rows, provenance)
    run = tmp_path/'train'
    result = run_training(dataset, run)
    assert result['selected']
    assert read(run/'summary.json')['final_test_evaluated'] is False
    frozen_before = (run/'frozen_run.json').read_bytes()
    report = evaluate_frozen(run, dataset, tmp_path/'test')
    assert 'constant:raw' in report
    from scripts.correction_prediction.opportunities import evaluate_opportunities
    practical = evaluate_opportunities(run, dataset, tmp_path/'test', tmp_path/'opportunities.json')
    assert 'candidate' in practical['summary']
    assert frozen_before == (run/'frozen_run.json').read_bytes()
    assert read(tmp_path/'test/summary.json')['validation_passed'] is False
    corrupted = json.loads(dataset.read_text())
    corrupted['payload']['rows'][0]['features']['gray_std'] = 12345
    dataset.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match='Checksum'):
        load_dataset(dataset)


def test_jev_requests_are_preflight_only_and_budgeted(tmp_path):
    f = numeric_rows(1)[0][0]
    settings = localizer_settings(pilot_config())
    body = build_request(f, settings)
    response = dict(model=body['model'], answers={'outcome': dict(type='choice', choice='reject',
                        probabilities=dict(zip(CLASS_NAMES, [1., 0, 0, 0, 0, 0, 0, 0])))})
    calls = []
    def transport(request, key, images):
        calls.append(request)
        return response
    with pytest.raises(RuntimeError, match='network disabled'):
        JevClient(tmp_path).predict(body)
    client = JevClient(tmp_path, max_requests=1, transport=transport)
    assert client.predict(body)[0] == 1
    assert client.predict(body)[0] == 1 and len(calls) == 1
    other = build_request(dict(f, agl_m=f['agl_m']+1), settings)
    with pytest.raises(RuntimeError, match='budget exhausted'):
        client.predict(other)
    with pytest.raises(ValueError, match='allowlist'):
        build_request(dict(f, ground_truth_x=123), settings)
    with pytest.raises(ValueError, match='Pin'):
        build_request(f, settings, 'jev-latest')
    with pytest.raises(ValueError, match='another model'):
        parse_response(dict(response, model='changed'), body['model'])
    request_image = build_request(f, settings, map_rgb=np.zeros((256, 256, 3), 'uint8'))
    assert 'assets' in request_image and 'assets' not in body


def test_jev_failure_consumes_budget_and_is_not_a_label(tmp_path):
    def fail(*args):
        raise OSError('offline')
    client = JevClient(tmp_path, max_requests=1, transport=fail)
    body = build_request(numeric_rows(1)[0][0], localizer_settings(pilot_config()))
    with pytest.raises(OSError):
        client.predict(body)
    assert not list(tmp_path.glob('[!r]*.json'))
    with pytest.raises(RuntimeError, match='budget exhausted'):
        client.predict(body)


def test_timed_route_sampling_is_independent_of_cell_subdivision():
    factory = lambda xy, t: (xy.tolist(), t)
    a = timed_route_queries([[0, 0], [100, 0]], [10], factory, 2)
    b = timed_route_queries([[0, 0], [50, 0], [100, 0]], [5, 5], factory, 2)
    assert a == b and len(a) == 6
    assert timed_route_queries([[0, 0]], [], factory, 2) == [([0., 0.], 0.)]


def test_predictor_refuses_unknown_domains(query, reference):
    query = replace(query, map_id=digest(reference.identity))
    f, _, _ = extract_features(query, reference)
    model = fit_model('constant', [f], np.array([0]))
    predictor = Predictor(model, reference, allowed_map_ids=[query.map_id])
    assert predictor.predict_correction(replace(query, map_id='new'))['probabilities'] is None
    assert predictor.predict_correction(query)['probabilities']['reject'] > 0


def test_observed_opportunity_comparison():
    rows = [r for r in make_dataset() if r['split'] == 'test']
    p = np.eye(8)[[r['label']['class_index'] for r in rows]]
    result = compare_opportunities(rows, p, np.ones_like(p)/8)
    assert result['summary']['candidate']['mean_regret'] == 0
    assert result['route_arrival_or_safety_validated'] is False


def test_288_attempt_campaign_resume(tmp_path, query, reference):
    config = pilot_config()
    backend = SimpleNamespace(config=config, identity={'fake': True})
    selection = [dict(scene_id=s, sample_id=f'frame_{j}', path=str(tmp_path/'sources'/s/f'{j}.npz'),
                      reference_json=str(tmp_path/'sources'/s/'ref.json'), split='pilot', spatial_group=s)
                 for s in config['scenes'] for j in range(2)]
    def sample_reader(path, *args):
        from pathlib import Path
        p = Path(path)
        return dict(scene_id=p.parent.name, sample_id='frame_'+p.stem,
                    xyz_enu=query.position, rotation=query.camera_to_enu,
                    K=query.K, dist=query.dist, rgb=np.zeros((32, 32, 3), 'uint8'), source_identity={'fake': 'sha'})
    calls = []
    def localizer(*args):
        calls.append(1)
        return dict(accepted=False, pred_xyz_enu=None, status='inference_complete')
    output = tmp_path/'output'
    kwargs = dict(sample_reader=sample_reader, reference_reader=lambda *args: reference,
                  reference_fingerprint=lambda *args: {'fake': 'map'}, localizer=localizer)
    report = run_campaign(selection, backend, output, config, **kwargs)
    assert report['completed_attempts'] == 288 and len(calls) == 288
    assert run_campaign(selection, backend, output, config, **kwargs) == report
    assert len(calls) == 288
    with pytest.raises(ValueError, match='Main campaign'):
        run_campaign(selection, backend, tmp_path/'other', config, stage='main', **kwargs)


def test_geotiff_adapter_and_saved_route(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    transform = from_origin(500000, 5600000, 1, 1)
    for name, count, dtype, value in [('rgb', 3, 'uint8', 100), ('dem', 1, 'float32', 10)]:
        with rasterio.open(tmp_path/(name+'.tif'), 'w', driver='GTiff', width=32, height=32,
                           count=count, dtype=dtype, crs='EPSG:32635', transform=transform) as dst:
            dst.write(np.full((count, 32, 32), value, dtype=dtype))
    ref = RasterReference(tmp_path/'rgb.tif', tmp_path/'dem.tif')
    xy = ref.pixels_to_enu([[0, 0]])
    np.testing.assert_allclose(ref.enu_to_pixels(xy), [[0, 0]])
    assert ref.plane_at(xy[0]) == 10
    from scripts.moa.edge_costs import CostMap
    from scripts.moa.route_results import save_route_results
    cm = CostMap(dem=np.zeros((2, 2)), nav_density=np.ones((2, 2)), visibility=np.ones((2, 2)),
                 wind_field=np.zeros((2, 2, 2)), occupancy=np.zeros((2, 2), dtype=bool))
    route_file = tmp_path/'route.npz'
    save_route_results(route_file, cm, [([(0, 0), (1, 0)], [1, 0, 0])])
    queries = saved_route_queries(route_file, 0, cm, lambda xy, t: (xy.tolist(), t), .1,
                                 grid_to_metric=lambda cell: np.array(cell)*10)
    assert len(queries) > 1


def test_jev_working_freeze_test_flow_and_freeze_integrity(tmp_path):
    from scripts.correction_prediction.jev_campaign import run_jev_predictions
    rows = make_dataset()
    provenance = dict(stage='main', main_design={'approved_attempts': len(rows)},
                      references={'fixture': {'sha': 'map'}}, localizer_settings=localizer_settings(pilot_config()))
    dataset = tmp_path/'dataset.json'
    save_dataset(dataset, rows, provenance)
    calls = []
    def transport(body, key, images):
        calls.append(body)
        assert 'gt_xyz' not in json.dumps(body) and not images
        return dict(model=body['model'], answers={'outcome': dict(type='choice', choice='reject',
                    probabilities={k: 1/8 for k in CLASS_NAMES})})
    working = tmp_path/'working_jev.json'
    run_jev_predictions(dataset, working, max_requests=64, transport=transport)
    assert len(calls) == 64
    run = tmp_path/'run'
    run_training(dataset, run, jev_predictions={'jev_numeric': working})
    test_predictions = tmp_path/'test_jev.json'
    run_jev_predictions(dataset, test_predictions, max_requests=96, transport=transport, frozen_run=run)
    assert len(calls) == 96
    results = evaluate_frozen(run, dataset, tmp_path/'evaluated', jev_predictions={'jev_numeric': test_predictions})
    assert 'jev_numeric:calibrated' in results
    path = run/'models/constant.json'
    payload = load_verified(path)
    payload['temperature'] = 100
    # A newly valid outer checksum must still fail the immutable run binding.
    path.write_text(json.dumps({'sha256': digest(payload), 'payload': payload}))
    with pytest.raises(ValueError, match='changed after freeze'):
        evaluate_frozen(run, dataset, tmp_path/'bad', jev_predictions={'jev_numeric': test_predictions})


def test_notebook_and_pilot_config_are_consistent():
    import ast
    from pathlib import Path
    package = Path(__file__).resolve().parents[1]
    notebook = read(package/'notebooks/pilot.ipynb')
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            # The researcher can save executed cells; do not erase their evidence
            # or require a particular value of their execution switches.
            assert cell.get('execution_count') is None or isinstance(cell['execution_count'], int)
    text = '\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'JEV_MAX_REQUESTS = 0' in text
    assert 'CAMP_CHECKPOINT is not None' not in text
    assert 'initialize_backend(RUNTIME_ROOT, CAMP_CHECKPOINT)' in text
    configuration = read(package/'configs/pilot.json')
    assert configuration['sigma_xy_m'] == pilot_config()['sigma_xy_m']
    assert configuration['expected_attempts'] == configuration['frames']*len(configuration['sigma_xy_m'])*len(configuration['conditions'])*configuration['trials']


def test_sensitivity_has_no_loss_for_perfect_forecasts():
    from scripts.correction_prediction.reporting import bin_sensitivity
    rows = make_dataset()
    p = np.eye(8)[[r['label']['class_index'] for r in rows]]
    assert all(v['log_loss'] == 0 and v['brier_score'] == 0 for v in bin_sensitivity(p, rows).values())


def test_learning_curve_does_not_use_test_outcomes(tmp_path):
    from scripts.correction_prediction.learning import learning_curve
    rows = make_dataset()
    provenance = dict(stage='main', main_design={'approved_attempts': len(rows)},
                      references={'fixture': {'sha': 'map'}}, localizer_settings=localizer_settings(pilot_config()))
    first = tmp_path/'first.json'
    save_dataset(first, rows, provenance)
    a = learning_curve(first, tmp_path/'curve_a.json')
    for row in rows:
        if row['split'] == 'test':
            row['label'] = label_outcome([50, 0, 0], [500, 0, 0], [0, 0, 0], True)
    second = tmp_path/'second.json'
    save_dataset(second, rows, provenance)
    b = learning_curve(second, tmp_path/'curve_b.json')
    assert a['points'] == b['points']
    assert a['final_test_evaluated'] is False and a['data_sufficiency_established'] is False


def test_bad_jev_reply_is_cached_for_audit(tmp_path):
    body = build_request(numeric_rows(1)[0][0], localizer_settings(pilot_config()))
    calls = []
    def wrong(*args):
        calls.append(1)
        return dict(model='unrequested-version', answers={})
    client = JevClient(tmp_path, max_requests=1, transport=wrong)
    for _ in range(2):
        with pytest.raises(ValueError, match='another model'):
            client.predict(body)
    assert len(calls) == 1


def test_artifact_accepts_numpy_diagnostics_but_rejects_nonfinite(tmp_path):
    save_verified(tmp_path/'record.json', dict(height=np.int64(128), xy=np.array([1., 2.])))
    assert load_verified(tmp_path/'record.json') == dict(height=128, xy=[1., 2.])
    with pytest.raises(ValueError):
        digest(dict(value=float('nan')))


def test_camp_download_is_cached_and_overrides_are_explicit(tmp_path):
    from scripts.correction_prediction.colab import ensure_camp_checkpoint, CAMP_FILENAME
    from scripts.visibility_localization.colab import CAMP_URL
    calls = []
    def downloader(url, output, **kwargs):
        from pathlib import Path
        assert url == CAMP_URL and kwargs == dict(fuzzy=True, quiet=False)
        calls.append(url)
        Path(output).write_bytes(b'PK\x03\x04fixture')
        return output
    path = ensure_camp_checkpoint(cache_dir=tmp_path, downloader=downloader)
    assert path == tmp_path/CAMP_FILENAME and len(calls) == 1
    assert ensure_camp_checkpoint(cache_dir=tmp_path, downloader=downloader) == path
    assert ensure_camp_checkpoint(path, cache_dir=tmp_path, downloader=downloader) == path
    assert len(calls) == 1 and not list(tmp_path.glob('*.download'))
    with pytest.raises(FileNotFoundError, match='not found'):
        ensure_camp_checkpoint(tmp_path/'typo.pth', cache_dir=tmp_path, downloader=downloader)


@pytest.mark.parametrize('failure', ['html', 'empty', 'network'])
def test_failed_camp_download_never_publishes_cache(tmp_path, failure):
    from scripts.correction_prediction.colab import ensure_camp_checkpoint, CAMP_FILENAME
    def downloader(url, output, **kwargs):
        from pathlib import Path
        if failure == 'network':
            raise OSError('Network unavailable')
        Path(output).write_bytes(b'<html>Download limit</html>' if failure == 'html' else b'')
        return output
    with pytest.raises(RuntimeError, match='Could not download CAMP'):
        ensure_camp_checkpoint(cache_dir=tmp_path, downloader=downloader)
    assert not (tmp_path/CAMP_FILENAME).exists() and not list(tmp_path.glob('*.download'))


def test_camp_none_resolves_before_backend_initialization(tmp_path, monkeypatch):
    from pathlib import Path
    from contextlib import contextmanager
    from scripts.correction_prediction import colab
    from scripts.visibility_localization import colab as upstream_colab, backend as backend_module
    checkpoint = tmp_path/'existing.pth'
    checkpoint.write_bytes(b'PK\x03\x04fixture')
    observed = []
    def resolve(value, **kwargs):
        assert value is None
        observed.append('resolve')
        return checkpoint
    @contextmanager
    def complete(source, file):
        assert file == checkpoint
        yield
    def backend_factory(source, config, file):
        assert file == checkpoint and observed == ['resolve']
        return SimpleNamespace()
    monkeypatch.setattr(colab, 'ensure_camp_checkpoint', resolve)
    monkeypatch.setattr(colab, 'verify_cuda_matmul', lambda: None)
    monkeypatch.setattr(colab, 'camp_from_complete_checkpoint', complete)
    monkeypatch.setattr(upstream_colab, 'checkout_upstream', lambda path: path)
    monkeypatch.setattr(backend_module, 'AnyVisLocBackend', backend_factory)
    result = colab.initialize_backend(tmp_path/'runtime')
    assert result.scratch.is_dir()

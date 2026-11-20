"""Resumable paired localization campaign; real GPU work is explicitly invoked."""
import csv
import time
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts.visibility_localization.core import defaults, make_prior, degrade, stable_seed, file_hash
from scripts.visibility_localization.quick_visibility import visibility_config, pilot_order
from scripts.visibility_localization.data import read_sample, load_reference, reference_identity
from scripts.visibility_localization.backend import localize

from .artifacts import digest, manifest, read, save_verified, load_verified, write_once
from .contracts import CorrectionQuery, CONDITIONS, label_outcome
from .dataset import save_dataset
from .features import extract_features


def pilot_config():
    config = visibility_config(defaults())
    config.update(reference_mode='aerial', matching_method='Roma', map_alignment='prior_yaw',
                  sigma_xy_m=[5., 20., 50., 100.], trials=3, sample_count=8,
                  scenes=['Scene_05', 'Scene_08', 'Scene_11', 'Scene_21'])
    return config


def fixed_pilot_selection(csv_path):
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as f:
        selection = list(csv.DictReader(f))
    ids = [(r['scene_id'], r['sample_id']) for r in selection]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate source frames')
    scenes = pilot_config()['scenes']
    if set(r['scene_id'] for r in selection) != set(scenes):
        raise ValueError('Use the previous quick visibility selection, with its four scenes')
    selected = pilot_order(selection, scenes, 42)[:8]
    return [dict(r, split='pilot', spatial_group=r['scene_id']) for r in selected]


def query_from_prior(prior, sample, map_id):
    return CorrectionQuery(position=prior.xyz, crs='LOCAL_ENU', camera_to_enu=prior.camera_to_enu,
        K=sample['K'], dist=sample['dist'], image_shape=tuple(sample['rgb'].shape[:2]),
        covariance_xy=np.eye(2)*prior.sigma_xy_m**2, yaw_sigma_deg=prior.yaw_sigma_deg,
        tilt_sigma_deg=prior.tilt_sigma_deg, z_sigma_m=prior.z_sigma_m,
        scenario='clear', map_id=map_id)


def localizer_settings(config):
    keys = ('top_k', 'min_matches', 'min_inliers', 'min_inlier_ratio', 'max_median_reprojection_px',
            'pnp_reprojection_px', 'map_alignment', 'reference_mode', 'match_resize_ratio')
    return {k: config[k] for k in keys}


def run_pilot(selection_csv, backend, output):
    return run_campaign(fixed_pilot_selection(selection_csv), backend, output, pilot_config(),
                        stage='pilot', source_selection_sha256=file_hash(selection_csv))


def _halt_on_technical_error(row, record_path, backend):
    if hasattr(backend, 'release_reference'):
        backend.release_reference()
    error = row['localization']
    raise RuntimeError(
        f"Pilot stopped after a technical error: {error.get('exception_type', 'unknown')}: "
        f"{error.get('exception', 'message unavailable')}. Saved record: {record_path}. "
        'Inspect this exception before starting another campaign; this is not a localization rejection.\n'
        + (error.get('traceback') or 'No Python traceback was supplied by the localizer.'))


def run_campaign(selection, backend, output, config, *, stage='pilot',
                 source_selection_sha256=None, main_design=None,
                 stop_on_technical_error=None,
                 sample_reader=read_sample, reference_reader=load_reference,
                 reference_fingerprint=reference_identity, localizer=localize):
    """A main design must explicitly document the agreed split/budget and overlap audit.

    Injected readers/localizer are for synthetic integration tests, never used to
    generate reported empirical results. Runtime timings are measurements only.
    """
    if stage not in ('pilot', 'main'):
        raise ValueError('Unknown campaign stage')
    if stop_on_technical_error is None:
        stop_on_technical_error = stage == 'pilot'
    if type(stop_on_technical_error) is not bool:
        raise ValueError('stop_on_technical_error must be boolean')
    if not selection or len({(r['scene_id'], r['sample_id']) for r in selection}) != len(selection):
        raise ValueError('Nonempty unique frame selection required')
    if config != backend.config:
        raise ValueError('Backend configuration differs from experiment configuration')
    if config['matching_method'] != 'Roma' or config['map_alignment'] != 'prior_yaw':
        raise ValueError('Protocol requires RoMa and prior_yaw')
    expected = len(selection)*len(config['sigma_xy_m'])*config['trials']*len(CONDITIONS)
    if stage == 'pilot' and (config != pilot_config() or len(selection) != 8 or expected != 288):
        raise ValueError('Pilot must be the frozen 8 x 4 x 3 x 3 design')
    if stage == 'main':
        if not main_design or main_design.get('approved_attempts') != expected or not main_design.get('spatial_overlap_audit'):
            raise ValueError('Main campaign needs an agreed exact budget and spatial overlap audit')
        from .dataset import SPLITS, REVIEWED
        if any(r.get('split') not in SPLITS-{'pilot'} or not r.get('spatial_group') or not r.get('opportunity_group') for r in selection):
            raise ValueError('Explicit main split/group assignments required')
        for field in ('scene_id', 'spatial_group'):
            for value in {r[field] for r in selection}:
                if len({r['split'] for r in selection if r[field] == value}) != 1:
                    raise ValueError('Spatial/scene split leakage')
        if any(r['split'] == 'test' and r['scene_id'] in REVIEWED for r in selection):
            raise ValueError('Reviewed test scene')
    output = Path(output).resolve()
    for r in selection:
        for field in ('path', 'reference_json'):
            source = Path(r[field]).resolve().parent
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError('Keep experiment outputs separate from source scenes')
    identities = []
    refs = {}
    for r in selection:
        s = sample_reader(r['path'], config['pose_convention'], config['max_image_side'])
        if (s['scene_id'], s['sample_id']) != (r['scene_id'], r['sample_id']):
            raise ValueError('Frame metadata/selection mismatch')
        identities.append(dict(scene_id=s['scene_id'], sample_id=s['sample_id'], source=s['source_identity'],
                               split=r['split'], spatial_group=r['spatial_group'],
                               opportunity_group=r.get('opportunity_group')))
        if r['scene_id'] not in refs:
            refs[r['scene_id']] = reference_fingerprint(r['reference_json'], config['reference_mode'])
    frozen = manifest(config=config, backend=backend.identity, selection=identities, references=refs,
                      stage=stage, expected_attempts=expected, main_design=main_design,
                      stop_on_technical_error=stop_on_technical_error,
                      source_selection_sha256=source_selection_sha256,
                      synthetic_prior=True, synthetic_visibility=True, localizer_settings=localizer_settings(config))
    identity = digest(frozen)
    config_path = output/'campaign.json'
    if config_path.exists():
        if load_verified(config_path) != frozen:
            raise ValueError('Campaign changed; use a new output directory')
    else:
        save_verified(config_path, frozen)
    rows = []
    reference, scene = None, None
    for selected in sorted(selection, key=lambda r: (r['scene_id'], r['sample_id'])):
        sample = sample_reader(selected['path'], config['pose_convention'], config['max_image_side'])
        if scene != selected['scene_id']:
            if hasattr(backend, 'release_reference'):
                backend.release_reference()
            reference = None
            reference = reference_reader(selected['reference_json'], config['reference_mode'])
            scene = selected['scene_id']
        for sigma in config['sigma_xy_m']:
            for trial in range(config['trials']):
                pair = [scene, sample['sample_id'], sigma, trial]
                prior_seed = stable_seed(config['seed'], *pair, 'prior')
                prior, prior_meta = make_prior(sample['xyz_enu'], sample['rotation'], sigma,
                                               config['profiles']['main'], prior_seed)
                for condition in CONDITIONS:
                    attempt_id = digest([identity, pair, condition])
                    path = output/'records'/(attempt_id+'.json')
                    if path.exists():
                        row = load_verified(path)
                        if row['attempt_id'] != attempt_id:
                            raise ValueError('Attempt identity mismatch')
                        if row.get('map_context') and file_hash(output/row['map_context']) != row['map_context_sha256']:
                            raise ValueError('Map context changed')
                        rows.append(row)
                        if stop_on_technical_error and row['label']['technical_error']:
                            _halt_on_technical_error(row, path, backend)
                        continue
                    query = query_from_prior(prior, sample, digest(refs[scene]))
                    from dataclasses import replace
                    query = replace(query, scenario=condition)
                    row = dict(attempt_id=attempt_id, campaign_id=identity, scene_id=scene,
                               sample_id=sample['sample_id'], spatial_group=selected['spatial_group'],
                               split=selected['split'], condition=condition, sigma_xy_m=sigma, trial=trial,
                               opportunity_group=selected.get('opportunity_group'),
                               prior_seed=prior_seed, prior_metadata=prior_meta, features=None,
                               map_context=None, unsupported_reason=None)
                    try:
                        features, patch, metadata = extract_features(query, reference)
                        image_path = output/'maps'/(attempt_id+'.png')
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(image_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)):
                            raise OSError('Could not save map context')
                        row.update(features=features, feature_metadata=metadata,
                                   map_context=str(image_path.relative_to(output)),
                                   map_context_sha256=file_hash(image_path))
                    except ValueError as exc:
                        row['unsupported_reason'] = str(exc)
                    smoke_seed = stable_seed(config['seed'], scene, sample['sample_id'], trial, 'smoke')
                    image, _, _ = degrade(sample['rgb'], condition, config, smoke_seed)
                    started = time.perf_counter()
                    try:
                        prediction = localizer(backend, image, sample['K'], sample['dist'], reference,
                                               prior, 'prior', stable_seed(prior_seed, 'inference'))
                    except Exception as exc:
                        prediction = dict(status='technical_error', accepted=False,
                                          pred_xyz_enu=None, exception_type=type(exc).__name__,
                                          exception=str(exc), traceback=traceback.format_exc())
                    technical = prediction.get('status') == 'technical_error'
                    row.update(label=label_outcome(prior.xyz, prediction.get('pred_xyz_enu'), sample['xyz_enu'],
                                                  prediction['accepted'], technical_error=technical),
                               localization=prediction, elapsed_s=time.perf_counter()-started,
                               smoke_seed=smoke_seed)
                    save_verified(path, row)
                    rows.append(row)
                    if stop_on_technical_error and technical:
                        _halt_on_technical_error(row, path, backend)
    if hasattr(backend, 'release_reference'):
        backend.release_reference()
    dataset_path = output/'dataset.json'
    provenance = dict(campaign_id=identity, stage=stage, localizer_settings=localizer_settings(config),
                      references=refs, main_design=main_design)
    if not dataset_path.exists():
        save_dataset(dataset_path, rows, provenance)
    else:
        saved = load_verified(dataset_path)
        if saved['rows'] != rows:
            raise ValueError('Existing dataset changed')
    working = [r for r in rows if r['split'] != 'test']
    summary = dict(expected_attempts=expected, completed_attempts=len(rows),
                   technical_errors=sum(r['label']['technical_error'] for r in rows),
                   unsupported_queries=sum(r['features'] is None for r in rows),
                   outcomes=dict(Counter(r['label'].get('class_name', 'technical_error') for r in working)),
                   outcomes_exclude_final_test=True,
                   measured_localization_s=sum(r['elapsed_s'] for r in rows),
                   main_campaign_approved=False, validation_passed=False)
    if not (output/'summary.json').exists():
        write_once(output/'summary.json', summary)
    return summary

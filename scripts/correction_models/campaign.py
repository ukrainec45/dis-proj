"""Main 1920-attempt paired campaign; collection and profiling share each inference."""
from pathlib import Path
from dataclasses import asdict
import time
import traceback

import cv2
import numpy as np

from scripts.visibility_localization.core import make_prior, degrade, stable_seed, file_hash
from scripts.visibility_localization.data import read_sample, load_reference, reference_identity
from scripts.visibility_localization.backend import localize
from .artifacts import digest, manifest, save_verified, load_verified
from .config import main_config, APPROVED_ATTEMPTS
from .contracts import CorrectionQuery, CONDITIONS, label_outcome
from .features import extract_features
from .selection import validate_selection
from .embeddings import CampEncoder, cached_embedding
from .profiling import Profile


def localizer_settings(config):
    # Full config is frozen: search extent, acceptance, noise and degradation all matter.
    return dict(config)


def query_from_prior(prior, sample, map_id, condition):
    return CorrectionQuery(position=prior.xyz, crs='LOCAL_ENU', camera_to_enu=prior.camera_to_enu,
        K=sample['K'], dist=sample['dist'], image_shape=sample['rgb'].shape[:2],
        covariance_xy=np.eye(2)*prior.sigma_xy_m**2, yaw_sigma_deg=prior.yaw_sigma_deg,
        tilt_sigma_deg=prior.tilt_sigma_deg, z_sigma_m=prior.z_sigma_m,
        scenario=condition, map_id=map_id)


def _halt(row, path):
    error = row['localization']
    raise RuntimeError(f"Collection stopped at {path}: {error.get('exception_type')}: "
                       f"{error.get('exception')}\n{error.get('traceback', '')}")


def run_collection(selection_path, backend, output, *, max_new_attempts=120, max_seconds=None,
                   sample_reader=read_sample, reference_reader=load_reference,
                   reference_fingerprint=reference_identity, localizer=localize, encoder=None, progress=print):
    selection = load_verified(selection_path)
    validate_selection(selection)
    config = main_config()
    if backend.config != config:
        raise ValueError('Backend differs from frozen main config')
    from .environment import require_collection_cuda
    require_collection_cuda(backend)
    if type(max_new_attempts) is not int or max_new_attempts <= 0:
        raise ValueError('Positive max_new_attempts required')
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('Positive wall-time budget required')
    output = Path(output).resolve()
    for item in selection['rows']:
        for field in ('path', 'reference_json'):
            source = Path(item[field]).resolve().parent
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError('Keep outputs separate from source scenes')
    if reference_fingerprint is reference_identity:
        from .fingerprints import cached_reference_identity
        # Share the source-validation cache created during selection so a new
        # result directory does not trigger another full read of every map/DSM.
        reference_fingerprint = lambda path, mode: cached_reference_identity(
            path, mode, Path(selection_path).parent/'reference_fingerprints')
    refs = {s: reference_fingerprint(next(r['reference_json'] for r in selection['rows'] if r['scene_id'] == s), 'aerial')
            for s in selection['split_scenes']['train'] + selection['split_scenes']['validation'] +
            selection['split_scenes']['calibration'] + selection['split_scenes']['test']}
    if refs != selection['references']:
        raise ValueError('Reference files changed after selection')
    # Check source files even when all attempts are cached.
    for item in selection['rows']:
        s = sample_reader(item['path'], config['pose_convention'], config['max_image_side'])
        if (s['scene_id'], s['sample_id'], s['source_identity']) != (item['scene_id'], item['sample_id'], item['source_identity']):
            raise ValueError('Frame identity changed after inventory')
    encoder = encoder or CampEncoder(backend)
    frozen = manifest(stage='main', config=config, backend=backend.identity,
        selection_sha256=digest(selection), references=refs, encoder=encoder.identity,
        split_policy=selection['spatial_overlap_audit'],
        expected_attempts=APPROVED_ATTEMPTS, localizer_settings=localizer_settings(config),
        synthetic_prior=True, synthetic_visibility=True)
    identity = digest(frozen)
    frozen_path = output/'campaign.json'
    if frozen_path.exists():
        if load_verified(frozen_path) != frozen:
            raise ValueError('Campaign changed: use a new output directory')
    else:
        save_verified(frozen_path, frozen)
    recorded = getattr(backend, '_profiled_outputs', set())
    if getattr(backend, 'initialization_profile', None) is not None and str(output) not in recorded:
        import uuid
        save_verified(output/'sessions'/(uuid.uuid4().hex+'.json'),
                      dict(initialization_profile=backend.initialization_profile,
                           includes='model construction only; excludes checkpoint download and upstream checkout'))
        backend._profiled_outputs = recorded | {str(output)}
    rows, new_count = [], 0
    started = time.perf_counter()
    reference, scene = None, None
    torch = getattr(backend, 'torch', None)
    try:
        for item in sorted(selection['rows'], key=lambda r: (r['scene_id'], r['sample_id'])):
            sample = None
            for sigma in config['sigma_xy_m']:
                prior_seed = stable_seed(config['seed'], item['scene_id'], item['sample_id'], sigma, 'prior')
                for condition in CONDITIONS:
                    attempt_id = digest([identity, item['scene_id'], item['sample_id'], sigma, condition])
                    path = output/'records'/(attempt_id+'.json')
                    if path.exists():
                        row = load_verified(path)
                        if row['attempt_id'] != attempt_id or row['campaign_id'] != identity:
                            raise ValueError('Cached attempt identity mismatch')
                        if row['label']['technical_error']:
                            _halt(row, path)
                        verify_assets(row, output)
                        rows.append(row)
                        continue
                    if new_count >= max_new_attempts or (max_seconds is not None and time.perf_counter()-started >= max_seconds):
                        return _finish(rows, output, frozen, selection, complete=False, new_count=new_count)
                    if sample is None:
                        sample = sample_reader(item['path'], config['pose_convention'], config['max_image_side'])
                    if scene != item['scene_id']:
                        if hasattr(backend, 'release_reference'):
                            backend.release_reference()
                        reference = None
                        with Profile(torch) as loading:
                            reference = reference_reader(item['reference_json'], 'aerial')
                        scene = item['scene_id']
                        reference_load_profile = loading.result
                    else:
                        reference_load_profile = None
                    prior, prior_meta = make_prior(sample['xyz_enu'], sample['rotation'], sigma,
                                                   config['profiles']['main'], prior_seed)
                    query = query_from_prior(prior, sample, digest(refs[scene]), condition)
                    row = dict(attempt_id=attempt_id, campaign_id=identity, scene_id=scene,
                        sample_id=item['sample_id'], spatial_group=item['spatial_group'], split=item['split'],
                        condition=condition, sigma_xy_m=sigma, prior_seed=prior_seed, prior_metadata=prior_meta,
                        query=asdict(query),
                        features=None, embedding=None, map_context=None, unsupported_reason=None,
                        reference_load_profile=reference_load_profile)
                    smoke_seed = stable_seed(config['seed'], scene, item['sample_id'], 'smoke')
                    try:
                        with Profile(torch) as feature_profile:
                            try:
                                features, patch, metadata = extract_features(query, reference)
                            except ValueError as exc:
                                row['unsupported_reason'] = str(exc)
                            else:
                                patch_id = digest(dict(map=digest(refs[scene]), pixels=file_hash_bytes(patch)))
                                image_path = output/'maps'/(patch_id+'.png')
                                if not image_path.exists():
                                    image_path.parent.mkdir(parents=True, exist_ok=True)
                                    if not cv2.imwrite(str(image_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)):
                                        raise OSError('Failed to save map context')
                                vector, embedding_meta = cached_embedding(encoder, patch, output/'embeddings')
                                row.update(features=features, feature_metadata=metadata,
                                    map_context=str(image_path.relative_to(output)), map_context_sha256=file_hash(image_path),
                                    embedding=dict(path='embeddings/'+embedding_meta['cache_key']+'.json',
                                        sha256=file_hash(output/'embeddings'/(embedding_meta['cache_key']+'.json')),
                                        dimension=len(vector), **embedding_meta))
                        row['feature_profile'] = feature_profile.result
                        image, _, _ = degrade(sample['rgb'], condition, config, smoke_seed)
                        with Profile(torch) as localization_profile:
                            prediction = localizer(backend, image, sample['K'], sample['dist'], reference,
                                prior, 'prior', stable_seed(prior_seed, 'inference'))
                        row['localization_profile'] = localization_profile.result
                    except Exception as exc:
                        prediction = dict(status='technical_error', accepted=False, pred_xyz_enu=None,
                            exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
                    technical = prediction.get('status') == 'technical_error'
                    try:
                        label = label_outcome(prior.xyz, prediction.get('pred_xyz_enu'), sample['xyz_enu'],
                                              prediction['accepted'], technical_error=technical)
                    except Exception as exc:
                        prediction = dict(status='technical_error', accepted=False, pred_xyz_enu=None,
                            exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
                        technical = True
                        label = label_outcome(prior.xyz, None, sample['xyz_enu'], False, technical_error=True)
                    row.update(label=label, localization=prediction, smoke_seed=smoke_seed)
                    save_verified(path, row)
                    rows.append(row)
                    new_count += 1
                    if progress is not None:
                        progress(f'[{len(rows)}/{APPROVED_ATTEMPTS}] {scene}/{item["sample_id"]} '
                                 f'sigma={sigma:g} {condition}; '
                                 f'{"technical error saved" if technical else "inference saved"}; test outcomes not shown')
                    if technical:
                        _halt(row, path)
        return _finish(rows, output, frozen, selection, complete=True, new_count=new_count)
    finally:
        if hasattr(backend, 'release_reference'):
            backend.release_reference()


def file_hash_bytes(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_assets(row, root):
    root = Path(root).resolve()
    for relative, expected in ((row.get('map_context'), row.get('map_context_sha256')),
            ((row.get('embedding') or {}).get('path'), (row.get('embedding') or {}).get('sha256'))):
        if relative is not None:
            path = (root/relative).resolve()
            if not path.is_relative_to(root) or file_hash(path) != expected:
                raise ValueError('Cached map/embedding asset changed or outside campaign')


def _finish(rows, output, frozen, selection, *, complete, new_count):
    # Partial checkpoints are the per-attempt records. Dataset is published only when complete.
    if complete:
        from .dataset import validate_rows
        validate_rows(rows, require_complete=True)
        payload = dict(protocol_id=frozen['protocol_id'], rows=rows, provenance=frozen,
                       selection_sha256=digest(selection))
        path = output/'dataset.json'
        if path.exists():
            if load_verified(path) != payload:
                raise ValueError('Completed dataset differs')
        else:
            save_verified(path, payload)
    # Count all cached records, not only the prefix visited in a bounded session.
    total = len(list((output/'records').glob('*.json')))
    return dict(complete=complete, completed_attempts=total, new_attempts=new_count,
        expected_attempts=APPROVED_ATTEMPTS, dataset=str(output/'dataset.json') if complete else None,
        final_test_outcomes_hidden=True, research_only=True, flight_validation_passed=False)

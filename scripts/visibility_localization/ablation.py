"""Controlled AnyVisLoc comparisons. GT crops are diagnostics, never deployment inference."""
from dataclasses import replace
import gc
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

from .backend import AnyVisLocBackend, bind_upstream, localize
from .campaign import environment, read_json, write_json
from .core import UPSTREAM_COMMIT, Prior, array_hash, digest, file_hash, make_prior, score_result, stable_seed
from .data import load_reference, read_sample, relative_file
from .models import ROMA_WEIGHTS as WEIGHTS, download_roma_weights


def prepare_upstream(source):
    source = Path(source).resolve()
    actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise ValueError('Unexpected upstream commit')
    # Ignore CRLF conversion only; reject substantive tracked Python changes.
    result = subprocess.run(['git','-C',str(source),'diff','--no-ext-diff','--ignore-cr-at-eol',
                             '--exit-code','HEAD','--','*.py'],capture_output=True,text=True)
    if result.returncode != 0:
        raise ValueError('Upstream Python source differs from the pinned commit; use a fresh checkout')
    sys.path.insert(0, str(source))
    avl = importlib.import_module('avl_utils')
    runner = importlib.import_module('run_avl_match_loc')
    for module in (avl, runner):
        if Path(module.__file__).resolve().parent != source:
            raise ValueError('Another upstream checkout is already imported; restart the session')
    return avl, runner


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(1)


def author_options(runner, source, dataset_root, output, reference_mode, pose_priori='yp'):
    previous = sys.argv
    try:
        sys.argv = ['run_avl_match_loc.py', '--dataset_root', str(dataset_root),
                    '--yaml', str(Path(source)/'config_selectable_matchers.yaml'),
                    '--reference_mode', reference_mode, '--pose_priori', pose_priori,
                    '--save_dir', str(output), '--save_match_arrays', '--visualize']
        options = runner.get_parse()
    finally:
        sys.argv = previous
    options._combo_dir = str(output)
    options.skip_existing = False
    return options


def crop_transform(info):
    """Full map pixel -> author rotated/final-cropped patch pixel."""
    transform = np.asarray(info['rotation_matrix'], dtype=np.float64).copy()
    transform[:,2] -= transform[:,:2]@np.asarray(info['pre_crop'][:2])
    transform[:,2] -= np.asarray(info['final_crop'][:2])
    return transform


def crop_points_to_local(points, info, ref_resize):
    """Preserve author float32 resize/inverse-rotation stages before DSM lookup.

    Do not collapse the affine or vectorize the scalar arithmetic: NumPy scalar
    promotion and rounding at DSM cell boundaries are part of the reference path.
    """
    points = np.asarray(points, np.float32).reshape(-1, 2)/max(float(ref_resize), 1e-12)
    inverse = cv2.invertAffineTransform(np.asarray(info['rotation_matrix'], np.float32))
    rx, ry = info['final_crop'][:2]
    x0, y0 = info['pre_crop'][:2]
    resolution = np.asarray(info['map_resolution'], np.float32).reshape(-1)
    origin = np.asarray(info['map_origin_local'], np.float32).reshape(-1)
    xs, ys = [], []
    for point in points:
        x, y = float(rx)+float(point[0]), float(ry)+float(point[1])
        px = inverse[0,0]*x + inverse[0,1]*y + inverse[0,2]
        py = inverse[1,0]*x + inverse[1,1]*y + inverse[1,2]
        col, row = float(x0)+float(px), float(y0)+float(py)
        xs.append(col*float(resolution[0])+float(origin[0]))
        ys.append(row*float(resolution[1])+float(origin[1]))
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def load_pair(dataset_root, selection, avl):
    from avl_data import AnyVisLocNPZDataset
    dataset = AnyVisLocNPZDataset(str(dataset_root), scenes=[selection['scene_id']])
    matches = [i for i,p in enumerate(dataset.samples) if Path(p).resolve() == Path(selection['path']).resolve()]
    if len(matches) != 1:
        raise ValueError('Selected NPZ not found unambiguously by the author loader')
    sample = dataset[matches[0]]
    ours = read_sample(selection['path'], max_side=1_000_000)
    bgr = avl.tensor_rgb_to_bgr_uint8(sample['image'])
    # Record discrepancies rather than silently comparing different camera inputs.
    if not np.array_equal(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ours['rgb']):
        raise ValueError('Author and project loaders decode different image pixels')
    k = sample['K'].detach().cpu().numpy().astype(np.float32)
    dist = sample['dist'].detach().cpu().numpy().astype(np.float32)
    if not np.allclose(k, ours['K'], atol=1e-5, rtol=0):
        raise ValueError('Author/project camera calibration mismatch')
    ours.update(K=k, dist=dist)
    return sample, ours, bgr


def observe_matcher(matcher, observations, replay=None):
    def observed(image, patch, *args, **kwargs):
        observations['query_hash'] = array_hash(image)
        observations['patch_hash'] = array_hash(patch)
        observations['query_shape'] = list(image.shape)
        observations['patch_shape'] = list(patch.shape)
        observations['query_image'] = image.copy()
        observations['patch_image'] = patch.copy()
        if replay is None:
            points = matcher(image, patch, *args, **kwargs)
        else:
            points = replay
        observations['query_points'] = np.asarray(points[0], np.float32).reshape(-1,2)
        observations['patch_points'] = np.asarray(points[1], np.float32).reshape(-1,2)
        return points
    return observed


def run_author(sample, methods, options, runner, seed):
    observation = {}
    matcher = observe_matcher(runner.run_pixel_match_anyvisloc, observation)
    solver = runner.estimate_drone_pose_anyvisloc
    def observed_solver(points, k, x, y, z, **kwargs):
        observation['world'] = np.column_stack([x, -np.asarray(y), z]).astype(np.float32)
        observation['pixels'] = np.asarray(points, np.float32)
        cv2.setRNGSeed(seed)
        return solver(points, k, x, y, z, **kwargs)
    run = bind_upstream(runner.process_one_sample, run_pixel_match_anyvisloc=matcher,
                        estimate_drone_pose_anyvisloc=observed_solver)
    seed_all(seed)
    result = run(sample, methods, methods['matching_method'], options)
    return result, observation


def run_adapter_crop(sample, ours, reference, methods, options, runner, avl, config, output, seed,
                     replay=None):
    true_pos = runner.make_true_pos(sample)
    view = avl.build_anyvisloc_reference_view(sample, options.reference_mode)
    bgr = avl.tensor_rgb_to_bgr_uint8(sample['image'])
    patch, info = runner.crop_gt_reference_patch(view['map'], true_pos, view, options)
    # Deliberately share AUTHOR crop/resize. The variable under test is our adapter.
    _, _, fine_scale, ref_resize, _ = runner.resize_for_matching(bgr, patch, true_pos, view['map_resolution'], options)
    observations = {}
    namespace = SimpleNamespace(**vars(avl))
    namespace.run_pixel_match_anyvisloc = observe_matcher(avl.run_pixel_match_anyvisloc, observations, replay)
    engine = AnyVisLocBackend.__new__(AnyVisLocBackend)
    engine.avl, engine.methods, engine.config = namespace, methods, config
    engine.options = SimpleNamespace(**vars(options))
    engine.options.strategy = 'Top1'
    # Nonidentity transform is required even for p: the patch has an XY offset.
    engine.options.pose_priori = 'yp'
    engine.scratch = Path(output)
    engine.scratch.mkdir(parents=True, exist_ok=True)
    engine.upstream_config = {}
    import torch
    engine.synchronize = torch.cuda.synchronize
    candidate = dict(row=0, col=0, height=patch.shape[0], width=patch.shape[1], rank=1,
                     score=1., fine_scale=fine_scale)
    patch_reference = replace(reference, bgr=patch)
    seed_all(seed)
    result = engine.match_candidate(ours['rgb'], ours['K'], ours['dist'], patch_reference,
                                    candidate, seed, capture=True, map_transform=crop_transform(info),
                                    map_points_to_local=lambda points: crop_points_to_local(points, info, ref_resize))
    result = score_result(result, ours['xyz_enu'], config['success_threshold_m'])
    result['crop_info'] = info
    inputs = result.get('pnp_inputs', {})
    for name in ('world', 'pixels'):
        if name in inputs:
            observations[name] = np.asarray(inputs[name], np.float32)
    return result, observations


def compare_controls(author, adapted, first, second, tolerance_m=.01):
    def same(name, atol=1e-3):
        if name not in first or name not in second:
            return name not in first and name not in second
        a,b = np.asarray(first[name]), np.asarray(second[name])
        return a.shape == b.shape and bool(np.allclose(a,b,atol=atol,rtol=0))
    author_xyz = None if author.get('pred_x') is None else np.array([author['pred_x'], -author['pred_y'], author['pred_z']])
    adapted_xyz = adapted.get('pred_xyz_enu')
    comparable = author_xyz is not None and adapted_xyz is not None
    delta = float(np.linalg.norm(author_xyz-np.asarray(adapted_xyz))) if comparable else None
    checks = dict(same_query=first.get('query_hash') == second.get('query_hash'),
                  same_patch=first.get('patch_hash') == second.get('patch_hash'),
                  same_query_matches=same('query_points'), same_patch_matches=same('patch_points'),
                  same_pnp_pixels=same('pixels'), same_pnp_world=same('world'),
                  same_inliers=author['num_inliers'] == adapted['inliers'],
                  same_pose=(delta <= tolerance_m) if comparable else (author_xyz is None and adapted_xyz is None),
                  pose_delta_m=delta, pnp_comparable=comparable)
    for name in ('world','pixels'):
        a, b = first.get(name), second.get(name)
        if a is not None and b is not None and np.shape(a) == np.shape(b) and np.size(a):
            difference = np.abs(np.asarray(a)-np.asarray(b))
            checks[f'max_{name}_delta'] = float(difference.max())
            if name == 'world':
                checks['max_world_xy_delta_m'] = float(difference[:,:2].max())
                checks['max_world_z_delta_m'] = float(difference[:,2].max())
    checks['passed'] = all(checks[k] for k in checks if k.startswith('same_'))
    checks['pose_verified'] = checks['passed'] and comparable
    return checks


def compact(stage, scene, sample_id, mode, result, author=False):
    error = result.get('pred_error_m') if author else result.get('error_xy_m')
    valid = bool(result.get('pnp_success')) if author else bool(result.get('valid_pnp'))
    return dict(stage=stage, scene_id=scene, sample_id=sample_id, reference_mode=mode,
                error_xy_m=error, pnp_valid=valid, correct_10m=bool(valid and error is not None and error<=10.),
                raw_matches=result.get('num_matches') if author else result.get('raw_matches'),
                inliers=result.get('num_inliers') if author else result.get('inliers'),
                accepted=None if author else result.get('accepted'),
                status='author_pnp' if author else result.get('status'),
                rejection_reason=result.get('rejection_reason',''))


def initialize_manifest(output, source, selection, config, modes, weight_hashes):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = set()
    for selected in selection:
        reference_path = Path(selected['reference_json'])
        paths.update([Path(selected['path']), reference_path])
        metadata = json.loads(reference_path.read_text(encoding='utf-8-sig'))
        for mode in modes:
            for field in ('map_path','dsm_path'):
                paths.add(relative_file(reference_path.parent, metadata['modes'][mode][field]))
        with np.load(selected['path'], allow_pickle=False) as data:
            if data['image'].size == 0:
                paths.add(relative_file(Path(selected['path']).parent, str(data['image_relpath'].item())))
    manifest = dict(upstream_commit=UPSTREAM_COMMIT, config=config, reference_modes=modes,
                    selection=selection, environment=environment(),
                    source_sha256={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},
                    upstream_sha256={name:file_hash(Path(source)/name) for name in
                                     ['avl_utils.py','avl_data.py','run_avl_match_loc.py','config_selectable_matchers.yaml']},
                    dataset_sha256={str(p):file_hash(p) for p in sorted(paths)}, weight_sha256=weight_hashes,
                    gt_controls=['A_author_Roma','B_adapter_replay','B_adapter_live','C_SP_LG','C_SP_LG_p','D0_search_exact_pose'],
                    deployment_inference=['D1_search_noisy_pose'], images='clear_only_native_size')
    manifest['experiment_id'] = digest(manifest)
    path = output/'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Experiment identity changed; choose a new CONTROL_RUN_TAG')
    write_json(path, manifest)
    return manifest


def record(output, key, row, result, observation=None):
    root = Path(output)/'records'/key
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'result.json', result)
    write_json(root/'row.json', row)
    if observation is not None:
        arrays = {k:v for k,v in observation.items() if isinstance(v,np.ndarray) and not k.endswith('_image')}
        for name in ('query_image','patch_image'):
            if name in observation and not cv2.imwrite(str(root/(name+'.png')), observation[name]):
                raise OSError(f'Cannot save matcher input: {root/name}')
        np.savez_compressed(root/'correspondences.npz', **arrays)
        write_json(root/'input_hashes.json', {k:v for k,v in observation.items() if not isinstance(v,np.ndarray)})


def crop_controls(source, dataset_root, selection, config, modes, output, methods, *, roma):
    avl, runner = prepare_upstream(source)
    rows, parity_rows = [], []
    for selected in selection:
        sample, ours, _ = load_pair(dataset_root, selected, avl)
        scene, sample_id = selected['scene_id'], selected['sample_id']
        seed = stable_seed(config['seed'], scene, sample_id, 'crop_control')
        for mode in modes:
            reference = load_reference(selected['reference_json'], mode)
            prefix = f'{mode}-{scene}-{sample_id}'
            options = author_options(runner, source, dataset_root, Path(output)/'author'/prefix, mode)
            options.resize_ratio = config['match_resize_ratio']
            if roma:
                author, observation = run_author(sample, methods, options, runner, seed)
                row = compact('A_author_Roma',scene,sample_id,mode,author,author=True)
                record(output,prefix+'-A',row,author,observation); rows.append(row)
                for stage, replay in [('B_adapter_replay',(observation['query_points'],observation['patch_points'])),
                                      ('B_adapter_live',None)]:
                    adapted, inputs = run_adapter_crop(sample,ours,reference,methods,options,runner,avl,config,
                                                       Path(output)/'adapter'/prefix/stage,seed,replay)
                    row = compact(stage,scene,sample_id,mode,adapted)
                    record(output,prefix+'-'+stage,row,adapted,inputs); rows.append(row)
                    check = dict(scene_id=scene,sample_id=sample_id,reference_mode=mode,stage=stage,
                                 **compare_controls(author,adapted,observation,inputs))
                    parity_rows.append(check)
                    write_json(Path(output)/'parity'/f'{prefix}-{stage}.json',check)
            else:
                for stage, pose_mode in [('C_SP_LG','yp'),('C_SP_LG_p','p')]:
                    options.pose_priori = pose_mode
                    adapted, inputs = run_adapter_crop(sample,ours,reference,methods,options,runner,avl,config,
                                                       Path(output)/'adapter'/prefix/stage,seed)
                    row = compact(stage,scene,sample_id,mode,adapted)
                    if stage == 'C_SP_LG':
                        baseline_inputs = read_json(Path(output)/'records'/(prefix+'-A')/'input_hashes.json')
                        if any(inputs[k] != baseline_inputs[k] for k in ('query_hash','patch_hash')):
                            raise ValueError('C_SP_LG changed the author matcher images; comparison is confounded')
                        row['same_images_as_A'] = True
                    record(output,prefix+'-'+stage,row,adapted,inputs); rows.append(row)
            pd.DataFrame(rows).to_csv(Path(output)/('roma_controls.csv' if roma else 'sp_controls.csv'),index=False)
            if roma:
                pd.DataFrame(parity_rows).to_csv(Path(output)/'adapter_parity.csv',index=False)
            avl._REF_ARRAY_CACHE.clear()
            gc.collect()
    return pd.DataFrame(rows), pd.DataFrame(parity_rows)


def search_controls(backend, source, dataset_root, selection, config, modes, output):
    avl, _ = prepare_upstream(source)
    rows = []
    for selected in selection:
        sample, ours, _ = load_pair(dataset_root, selected, avl)
        scene,sample_id = selected['scene_id'],selected['sample_id']
        seed = stable_seed(config['seed'],scene,sample_id,'search_control')
        sigma = config['sigma_xy_m'][0]
        exact = Prior(ours['xyz_enu'].copy(),ours['rotation'].copy(),sigma,0.,0.,0.)
        noisy, noise = make_prior(ours['xyz_enu'],ours['rotation'],sigma,config['profiles']['main'],seed)
        for mode in modes:
            reference = load_reference(selected['reference_json'],mode)
            for stage,prior in [('D0_search_exact_pose',exact),('D1_search_noisy_pose',noisy)]:
                # Inference takes only image/calibration/map/prior. GT is evaluated afterwards.
                prediction = localize(backend,ours['rgb'],ours['K'],ours['dist'],reference,prior,'prior',seed)
                result = score_result(prediction,ours['xyz_enu'],config['success_threshold_m'])
                result.update(prior_xyz_enu=prior.xyz.tolist(),prior_rotation=prior.camera_to_enu.tolist(),
                              prior_metadata=noise if stage.startswith('D1') else {'exact_pose_control':True})
                row = compact(stage,scene,sample_id,mode,result)
                record(output,f'{mode}-{scene}-{sample_id}-{stage}',row,result)
                rows.append(row)
                pd.DataFrame(rows).to_csv(Path(output)/'search_controls.csv',index=False)
    return pd.DataFrame(rows)


def summarize_controls(output):
    output = Path(output)
    rows = [read_json(p) for p in sorted((output/'records').glob('*/row.json'))]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No completed controls')
    frame.to_csv(output/'all_controls.csv',index=False)
    summary = frame.groupby(['reference_mode','stage'],sort=False).agg(
        n=('sample_id','size'), n_pnp=('pnp_valid','sum'), n_correct_10m=('correct_10m','sum'),
        median_error_m=('error_xy_m','median'), median_inliers=('inliers','median')).reset_index()
    summary['correct_fraction'] = summary.n_correct_10m/summary.n
    summary.to_csv(output/'summary.csv',index=False)
    return frame,summary

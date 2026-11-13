"""Short paired visibility stress test of the pinned author's full search path.

Author pose metadata supplies geometry/yaw, as in run_avl.py. This is a
controlled visibility test, not GPS-denied deployment performance.
"""
import gc
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .backend import bind_upstream
from .campaign import environment, log_resources, read_json, write_json
from .core import degrade, digest, file_hash, stable_seed
from .data import reference_identity


CONDITIONS = ('clear', 'fog_strong', 'fog_extreme', 'smoke_strong', 'smoke_extreme')


def visibility_config(base):
    import copy
    config = copy.deepcopy(base)
    config['fog'] = {'fog_strong': .3, 'fog_extreme': .1}
    config['smoke'] = {
        'smoke_strong': dict(coverage=.7, transmission=.2, softness=.08, color_rgb=[.5,.48,.45]),
        'smoke_extreme': dict(coverage=.9, transmission=.05, softness=.08, color_rgb=[.5,.48,.45])}
    return config


def pilot_order(selection, scenes, seed=42):
    """Two frames per scene first; selection never uses localization outcomes."""
    groups = {scene: sorted([r for r in selection if r['scene_id']==scene],
                           key=lambda r: stable_seed(seed,scene,r['sample_id'],'visibility_pilot'))
              for scene in scenes}
    if not groups or min(map(len,groups.values())) < 2:
        raise ValueError('At least two selected frames per scene are required')
    if sum(map(len,groups.values())) != len(selection):
        raise ValueError('Unexpected scene in selection')
    return [groups[s][i] for i in range(max(map(len,groups.values())))
            for s in scenes if i < len(groups[s])]


def author_options(source, dataset_root, output):
    import importlib
    runner = importlib.import_module('run_avl')
    if Path(runner.__file__).resolve().parent != Path(source).resolve():
        raise ValueError('Another upstream checkout is imported; restart the session')
    previous = sys.argv
    try:
        sys.argv = ['run_avl.py','--dataset_root',str(dataset_root),
                    '--reference_mode','aerial','--pose_priori','yp','--strategy','Topn_opt']
        options = runner.get_parse()
    finally:
        sys.argv = previous
    options.visualize = False
    options.disable_ref_feature_cache = True
    options._combo_dir = str(output)
    return options


def run_frame(backend, sample, rgb, options, config, seed, scratch, gallery):
    """Original retrieval, matching, PnP and final selection; instrumented only.

    gallery lives for one frame. The map embedding is reused; query features,
    ranking, RoMa and PnP are recomputed for every condition.
    """
    avl = backend.avl
    backend.seed(seed)
    cv2.setRNGSeed(int(seed))
    true_pos = avl.anyvisloc_sample_to_truepos(sample)
    view = avl.build_anyvisloc_reference_view(sample, 'aerial')
    if 'map' not in gallery:
        image, transform = avl.dumpRotateImage(view['map'], true_pos['yaw'])
        gallery.update(map=image, transform=transform)
    image, transform = gallery['map'], gallery['transform']
    query = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cache_hit = 'features' in gallery
    def extract(size, cfg, methods, mids, block_size, ref, uav, img_transform, model):
        if 'features' not in gallery:
            gf,qf = avl.extract_features(size,cfg,methods,mids,block_size,ref,uav,img_transform,model)
            gallery['features'] = gf.detach().cpu()
        else:
            _,qf = avl.extract_features(size,cfg,methods,mids[:1],block_size,ref,uav,img_transform,model)
            gf = gallery['features'].to(qf.device)
        return gf,qf
    retrieval = bind_upstream(avl.retrieval_all_anyvisloc, extract_features=extract)
    start = time.perf_counter()
    order,rows,cols,pde,ph,pw,fine_scale,_ = retrieval(image,query,true_pos,
        view['map_resolution'],view['map_origin_local'],transform,str(scratch),options,
        backend.upstream_config,backend.methods)
    backend.synchronize()
    retrieval_s = time.perf_counter()-start
    observations = {}
    active = {}
    def observed_matcher(*args,**kwargs):
        # The author passes /topN_ransac.jpg as the fifth positional argument.
        rank = int(Path(args[4]).name.split('_')[0].removeprefix('top'))-1
        points = avl.run_pixel_match_anyvisloc(*args,**kwargs)
        active['rank'] = rank
        observations[rank] = dict(raw_matches=len(points[0]),matches=0,inlier_ratio=0.)
        return points
    class ObserveCV:
        def __getattr__(self,name): return getattr(cv2,name)
        def solvePnPRansac(self,*args,**kwargs):
            result = cv2.solvePnPRansac(*args,**kwargs)
            obs = observations[active['rank']]
            obs['matches'] = len(args[0])
            ok,rvec,tvec,idx = result
            if ok and idx is not None and len(idx):
                indices = idx.reshape(-1)
                obs['inlier_ratio'] = len(indices)/len(args[0])
                projected = cv2.projectPoints(args[0],rvec,tvec,args[2],args[3])[0].reshape(-1,2)
                residual = np.linalg.norm(projected-np.asarray(args[1]).reshape(-1,2),axis=1)
                obs['reprojection_median_px'] = float(np.median(residual[indices]))
            return result
    solver = bind_upstream(avl.estimate_drone_pose_anyvisloc,cv2=ObserveCV())
    matching = bind_upstream(avl.Match2Pos_all_anyvisloc,run_pixel_match_anyvisloc=observed_matcher,
                             estimate_drone_pose_anyvisloc=solver)
    K = sample['K'].detach().cpu().numpy().astype(np.float32)
    dist = sample['dist'].detach().cpu().numpy().astype(np.float32)
    positions,inliers,match_times,pnp_times = matching(options,backend.upstream_config,query,fine_scale,K,
        image,view['dsm'],rows,cols,ph,pw,str(scratch),backend.methods,transform,
        view['map_resolution'],view['map_origin_local'],view['dsm_resolution'],view['dsm_origin_local'],
        dist=dist,truePos=None)
    backend.synchronize()
    position,error,errors,ok,best = avl.pos2error_anyvisloc(true_pos,positions,inliers)
    selected = observations.get(best,{})
    valid = bool(ok and error is not None and np.isfinite(error))
    result = dict(valid_pnp=valid,error_xy_m=float(error) if valid else None,
        status='pose' if valid else 'pnp_failed',pred_location=position,
        inliers=int(inliers[best]) if best is not None else 0, selected_rank=best+1 if best is not None else None,
        matches=selected.get('matches',0),raw_matches=selected.get('raw_matches',0),
        inlier_ratio=selected.get('inlier_ratio',0.),reprojection_median_px=selected.get('reprojection_median_px'),
        retrieval_s=retrieval_s,matching_s=float(sum(match_times)),pnp_s=float(sum(pnp_times)),
        total_s=time.perf_counter()-start,gallery_cache_hit=cache_hit,
        retrieval_top5_pde=[float(x) for x in pde[:5]],
        # Author PDE is normalized center distance, not our polygon-overlap recall.
        retrieval_top5_near_center=any(float(x)<=.5 for x in pde[:5]),
        candidate_inliers=[int(x) for x in inliers],candidate_errors_m=errors)
    for threshold in (3,5,10):
        result[f'accurate_{threshold}m'] = bool(valid and error<=threshold)
    return result


def run_campaign(backend, source, dataset_root, selection, base_config, output, *, full=False):
    """Default 8 frames x 5 conditions; full=True resumes toward 24 x 5."""
    from .ablation import load_pair, prepare_upstream
    source,output = Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    avl,_ = prepare_upstream(source)
    if backend.methods['matching_method'] != 'Roma':
        raise ValueError('This experiment requires Roma')
    config = visibility_config(base_config)
    ordered = pilot_order(selection,config['scenes'],config['seed'])
    if len(ordered)!=24 or len(config['scenes'])!=4:
        raise ValueError('Expected the existing 24-frame, four-scene pool')
    options = author_options(source,dataset_root,output/'runtime')
    # Clear any retained production alignment before author reference loading.
    backend.release_reference()
    manifest = dict(schema_version=1,experiment='author_visibility_quick_v1',
        source_hashes={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')},
        backend=backend.identity,environment=environment(),conditions=list(CONDITIONS),visibility_config=config,
        author_options=vars(options),selection=ordered,
        inputs=[dict(scene_id=r['scene_id'],sample_id=r['sample_id'],sample_sha256=file_hash(r['path'])) for r in ordered],
        references={r['scene_id']:reference_identity(r['reference_json'],'aerial') for r in ordered},
        author_pose_metadata_used_for_geometry=True,custom_acceptance_gates=False,
        gallery_cache='one_frame_cpu_features_only',query_size='original_author_loader')
    identity = digest(manifest)
    manifest['experiment_id'] = identity
    path = output/'experiment_config.json'
    if path.exists() and read_json(path)!=manifest:
        raise ValueError('Experiment identity changed; use a new RUN_TAG')
    write_json(path,manifest)
    chosen = ordered if full else ordered[:8]
    target_ids={(r['scene_id'],r['sample_id']) for r in chosen}
    expected=len(chosen)*len(CONDITIONS)
    def saved():
        rows=[]
        for path in (output/'records').glob('*.json'):
            envelope=read_json(path)
            row=envelope['result']
            if digest(row)!=envelope['sha256'] or row['experiment_id']!=identity or path.stem!=row['attempt_id']:
                raise ValueError(f'Corrupt or foreign record: {path}')
            if (row['scene_id'],row['sample_id']) in target_ids:
                rows.append(row)
        frame=pd.DataFrame(rows)
        frame.to_csv(output/'per_sample_results.csv',index=False)
        write_json(output/'progress.json',dict(expected=expected,completed=len(rows),
                   complete=len(rows)==expected,full=full,technical_errors=sum(r['status']=='technical_error' for r in rows)))
        return frame
    try:
        for frame_index,row in enumerate(chosen):
            pair_id=digest([row['scene_id'],row['sample_id']])[:24]
            keys={condition:digest([identity,pair_id,condition]) for condition in CONDITIONS}
            if all((output/'records'/f'{key}.json').is_file() for key in keys.values()):
                saved()  # Validate before reuse.
                continue
            for name in ('_REF_ARRAY_CACHE','_REF_JSON_CACHE'):
                getattr(avl,name,{}).clear()
            gc.collect()
            log_resources(output,'author_frame_begin',backend,scene_id=row['scene_id'],sample_id=row['sample_id'])
            sample,ours,_=load_pair(dataset_root,row,avl)
            gallery={}
            try:
                for condition in CONDITIONS:
                    key=keys[condition]
                    destination=output/'records'/f'{key}.json'
                    if destination.exists():
                        continue
                    seed=stable_seed(config['seed'],pair_id,'author_visibility')
                    smoke_seed=stable_seed(config['seed'],pair_id,'smoke')
                    rgb,transmission,visibility=degrade(ours['rgb'],condition,config,smoke_seed)
                    start=time.perf_counter()
                    record=dict(experiment_id=identity,attempt_id=key,pair_id=pair_id,condition=condition,
                        scene_id=row['scene_id'],sample_id=row['sample_id'],seed=seed,smoke_seed=smoke_seed,**visibility)
                    try:
                        record.update(run_frame(backend,sample,rgb,options,config,seed,output/'runtime',gallery))
                    except Exception as exc:
                        import traceback
                        record.update(status='technical_error',valid_pnp=False,error_xy_m=None,
                                      exception=str(exc),traceback=traceback.format_exc(),total_s=time.perf_counter()-start)
                    write_json(destination,dict(result=record,sha256=digest(record)))
                    saved()
                    print(f"{frame_index+1}/{len(chosen)} {row['scene_id']}/{row['sample_id']} {condition}: "
                          f"{record['status']}, error={record.get('error_xy_m')}, inliers={record.get('inliers')}",flush=True)
                    if record['status']=='technical_error':
                        raise RuntimeError(f'Technical failure saved at {destination}; inspect before continuing')
            finally:
                gallery.clear()
                del sample,ours
                for name in ('_REF_ARRAY_CACHE','_REF_JSON_CACHE'):
                    getattr(avl,name,{}).clear()
                gc.collect()
            current=saved()
            if frame_index==0:
                print('Estimated inference minutes for this stage:',round(current.total_s.sum()*len(chosen)/60,1),flush=True)
    finally:
        frame=saved()
    return frame


def summarize(frame, output):
    """Pair by frame, count PnP/technical failures in all-attempt accuracy."""
    output=Path(output)
    if frame.empty or frame.duplicated(['pair_id','condition']).any():
        raise ValueError('Expected unique nonempty paired observations')
    clear=frame.loc[frame.condition.eq('clear')]
    paired=frame.merge(clear,on='pair_id',suffixes=('','_clear'),validate='many_to_one')
    rows=[]
    for condition in CONDITIONS:
        group=paired.loc[paired.condition.eq(condition)].copy()
        if group.empty: continue
        valid=group.valid_pnp.fillna(False).astype(bool) & group.error_xy_m.notna()
        accurate=valid & group.error_xy_m.le(10)
        baseline=group.valid_pnp_clear.fillna(False).astype(bool) & group.error_xy_m_clear.le(10)
        available=group.status.ne('technical_error') & group.status_clear.ne('technical_error')
        group['delta_inliers']=pd.to_numeric(group.get('inliers'),errors='coerce')-pd.to_numeric(group.get('inliers_clear'),errors='coerce')
        group['delta_error_m']=group.error_xy_m-group.error_xy_m_clear
        values=group.loc[available,'delta_inliers'].dropna().to_numpy()
        ci=[None,None]
        if len(values):
            bootstrap=np.random.default_rng(42).choice(values,(2000,len(values)),replace=True).mean(axis=1)
            ci=np.quantile(bootstrap,[.025,.975]).tolist()
        rows.append(dict(condition=condition,n=len(group),n_technical=int(group.status.eq('technical_error').sum()),
            n_pnp=int(valid.sum()),accurate_3m=int((valid & group.error_xy_m.le(3)).sum()),
            accurate_5m=int((valid & group.error_xy_m.le(5)).sum()),accurate_10m=int(accurate.sum()),
            accuracy_10m=float(accurate.mean()),lost_vs_clear=int((baseline & ~accurate).sum()),
            gained_vs_clear=int((~baseline & accurate).sum()),
            median_pose_error_m=float(group.loc[valid,'error_xy_m'].median()) if valid.any() else None,
            mean_delta_inliers=float(values.mean()) if len(values) else None,
            delta_inliers_ci_low=ci[0],delta_inliers_ci_high=ci[1],
            median_paired_delta_error_m=float(group.delta_error_m.median()) if group.delta_error_m.notna().any() else None))
    overview=pd.DataFrame(rows)
    paired.to_csv(output/'paired_vs_clear.csv',index=False)
    overview.to_csv(output/'visibility_overview.csv',index=False)
    return overview,paired

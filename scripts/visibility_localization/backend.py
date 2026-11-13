"""Use the original retrieval and matching/PnP functions called by run_avl.py.

Only prior-based filtering, observation generation and metric collection are ours.
bind_upstream keeps the original function bodies and isolates instrumentation.
"""
from collections import OrderedDict
import importlib
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import cv2
import numpy as np
from shapely.geometry import Polygon, Point

from .core import UPSTREAM_COMMIT, digest, footprint, file_hash, array_hash


def geographic_tiles(reference, mids, ph, pw, center, radius, mode):
    if mode not in ("prior","global"): raise ValueError(mode)
    tiles,seen = [],set()
    for row,col in mids:
        r,co = int(row-ph/2),int(col-pw/2)
        if (r,co) in seen: continue
        seen.add((r,co))
        polygon = reference.pixels_to_enu([[co-.5,r-.5],[co+pw-.5,r-.5],
                                          [co+pw-.5,r+ph-.5],[co-.5,r+ph-.5]])
        if mode == "prior" and Polygon(polygon).distance(Point(center)) > radius: continue
        tiles.append(dict(row=r,col=co,height=ph,width=pw,polygon_enu=polygon.tolist()))
    return tiles


def estimate_pose(image_points, world_points, K, dist, config, seed):
    """OpenCV upstream P3P solver, with pose-quality instrumentation."""
    image_points = np.asarray(image_points, np.float64).reshape(-1,2)
    world_points = np.asarray(world_points, np.float64).reshape(-1,3)
    cv2.setRNGSeed(int(seed))
    if len(world_points) < config["min_matches"]:
        return dict(valid_pnp=False, accepted=False, rejection_reason="insufficient_matches",
                    matches=len(world_points), inliers=0, inlier_ratio=0.)
    ok, rvec, tvec, indices = cv2.solvePnPRansac(world_points, image_points,
        np.asarray(K,float), np.asarray(dist,float), flags=cv2.SOLVEPNP_P3P,
        reprojectionError=config["pnp_reprojection_px"],
        iterationsCount=config["pnp_iterations"], confidence=config["pnp_confidence"])
    if not ok or indices is None or not len(indices):
        return dict(valid_pnp=False, accepted=False, rejection_reason="pnp_failed",
                    matches=len(world_points), inliers=0, inlier_ratio=0.)
    R = cv2.Rodrigues(rvec)[0]
    xyz = (-R.T@tvec).reshape(3)
    projected = cv2.projectPoints(world_points, rvec, tvec, K, dist)[0].reshape(-1,2)
    residual = np.linalg.norm(projected-image_points, axis=1)
    idx = indices.reshape(-1)
    n = len(idx)
    ratio = n/len(world_points)
    median = float(np.median(residual[idx]))
    depths = (world_points@R.T+tvec.reshape(3))[:,2]
    positive = float(np.mean(depths > 0))
    reason = ("nonfinite_pose" if not np.isfinite(xyz).all() else
              "insufficient_inliers" if n < config["min_inliers"] else
              "low_inlier_ratio" if ratio < config["min_inlier_ratio"] else
              "high_reprojection" if median > config["max_median_reprojection_px"] else
              "negative_depth" if positive < config["min_positive_depth_fraction"] else "")
    return dict(valid_pnp=True, accepted=not reason, rejection_reason=reason,
                pred_xyz_enu=xyz.tolist(), camera_to_enu=R.T.tolist(), matches=len(world_points),
                inliers=n, inlier_ratio=ratio, reprojection_median_px=median,
                positive_depth_fraction=positive, inlier_indices=idx.tolist())


def bind_upstream(function, **overrides):
    """Use the same function code with an isolated globals dictionary."""
    from types import FunctionType
    scope = dict(function.__globals__)
    scope.update(overrides)
    bound = FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


class NoCandidates(Exception):
    pass


def retrieval_geometry(reference, prior, K, dist, image_shape, config):
    """Use only the noisy prior and reference DSM; never a sample/true pose."""
    heights, valid = reference.heights(np.asarray(prior.xyz[:2]).reshape(1, 2))
    plane_z = float(heights[0]) if valid[0] else reference.plane_z
    metrics = dict(retrieval_plane_z_m=plane_z,
                   retrieval_plane_source='local_dsm' if valid[0] else 'scene_median',
                   prior_height_above_plane_m=float(prior.xyz[2]-plane_z),
                   geometry_fallback=False)
    side = min(image_shape[:2])
    try:
        visible, center = footprint(prior.xyz, prior.camera_to_enu, K, dist,
                                    image_shape, plane_z, center_crop=True)
        resolution = max(visible.bounds[2]-visible.bounds[0],
                         visible.bounds[3]-visible.bounds[1])/side
        return center, [resolution], metrics
    except ValueError as exc:
        if str(exc) != 'unsupported_prior_geometry':
            raise
        # Invalid height/rays must not invent a pose or use the evaluation GT.
        # Search explicit physical scales around the noisy XY position instead.
        sizes = config['fallback_patch_sizes_m']
        metrics.update(geometry_fallback=True, geometry_fallback_reason=str(exc),
                       fallback_patch_sizes_m=list(sizes))
        return np.asarray(prior.xyz[:2]), [float(s)/(side*config['patch_scale'])
                                         for s in sizes], metrics


def retrieval_stride(reference, view_m, max_pixels):
    """Bound the retrieval raster crop without shrinking its ground coverage."""
    if max_pixels < 4:
        raise ValueError('max_patch_pixels must be at least 4')
    requested = view_m / reference.resolution
    clipped = np.minimum(requested, np.asarray(reference.bgr.shape[:2])[::-1]-2)
    stride = max(1, math.ceil(math.sqrt(float(np.prod(clipped))/max_pixels)))
    while True:
        shape = np.ceil(np.asarray(reference.bgr.shape[:2])/stride).astype(int)
        if min(shape) < 10:
            raise ValueError('Retrieval raster too small for the upstream grid; check scale and pixel budget')
        patch = 2*np.ceil(requested[::-1]/(2*stride)).astype(int)
        limits = 2*((shape-2)//2)
        patch = np.minimum(patch, limits)
        if int(np.prod(patch)) <= max_pixels:
            return stride
        stride += 1


class AnyVisLocBackend:
    """The retrieval_all_anyvisloc -> Match2Pos_all_anyvisloc path from run_avl.py."""
    def __init__(self, source, config, checkpoint):
        import torch
        import yaml
        import shutil
        from types import SimpleNamespace
        self.torch, self.config = torch, dict(config)
        matching_method = config.get('matching_method', 'SP_LG')
        if matching_method not in ('SP_LG', 'Roma'):
            raise ValueError('Unknown matching method; choose SP_LG or Roma')
        self.source = Path(source).resolve()
        actual = subprocess.check_output(['git','-C',str(self.source),'rev-parse','HEAD'],text=True).strip()
        if actual != UPSTREAM_COMMIT: raise ValueError('Unexpected AnyVisLoc commit')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
        self.seed(config['seed'])
        sys.path.insert(0,str(self.source))
        self.avl = importlib.import_module('avl_utils')
        self.upstream_config = yaml.safe_load((self.source/'config_selectable_matchers.yaml').read_text())
        self.upstream_config.update(DEVICE='cuda',RETRIEVAL_METHODS=['CAMP'],MATCHING_METHODS=[matching_method],
            RETRIEVAL_TOPN=config['top_k'],RETRIEVAL_COVER=config['gallery_overlap']*100,
            RETRIEVAL_SIZE=config['retrieval_size'],BATCH_SIZE=config['retrieval_batch'])
        expected = self.source/'Retrieval_Models/CAMP/weights/weights_0.9446_for_U1652.pth'
        expected.parent.mkdir(parents=True,exist_ok=True)
        if Path(checkpoint).resolve() != expected.resolve():
            if expected.exists() and file_hash(expected) != file_hash(checkpoint):
                raise ValueError('Different CAMP checkpoint already in upstream weights directory')
            if not expected.exists(): shutil.copy2(checkpoint,expected)
        self.options = SimpleNamespace(device='cuda',pose_priori='p',visualize=False,strategy='Top1',
            disable_ref_feature_cache=True,resize_ratio=config['match_resize_ratio'],patch_scale=config['patch_scale'],
            PnP_method='P3P',pnp_reproj_error=config['pnp_reprojection_px'],
            pnp_iterations=config['pnp_iterations'],pnp_confidence=config['pnp_confidence'],
            match_keypoints=config['match_keypoints'],
            selectable_code_dir=str(self.source/'Matching_Models/Sparse_matchers'),
            selectable_module='selectable_sparse_matcher')
        self.methods = self.avl.retrieval_init({'retrieval_method':'CAMP'},self.upstream_config)
        roma_hashes = None
        if matching_method == 'Roma':
            from .models import download_roma_weights
            roma_hashes = download_roma_weights(self.source)
        self.methods['matching_method'] = matching_method
        self.methods = self.avl.matching_init(self.methods,self.options,self.upstream_config)
        self.cache = OrderedDict()
        self.scratch = self.source.parent/'visibility_runtime'
        self.scratch.mkdir(exist_ok=True)
        self.identity = dict(upstream_commit=actual,camp_checkpoint_sha256=file_hash(expected),
            camp_state_sha256=self._state_hash(self.methods['retrieval_model']),
            retrieval_method='CAMP',matching_method=matching_method,pnp_method='upstream_P3P',
            runner='run_avl.py',retrieval_function='avl_utils.retrieval_all_anyvisloc',
            matching_function='avl_utils.Match2Pos_all_anyvisloc',
            pnp_function='avl_utils.estimate_drone_pose_anyvisloc',
            source_sha256={name:file_hash(self.source/name) for name in [
                'avl_utils.py','avl_data.py','run_avl.py','run_avl_match_loc.py','config_selectable_matchers.yaml']})
        if matching_method == 'Roma':
            self.identity.update(roma_state_sha256=self._state_hash(self.methods['matching_model']),
                                 roma_checkpoint_sha256=roma_hashes)
        else:
            self.identity.update(
                superpoint_state_sha256=self._state_hash(self.methods['matching_model']['extractor']),
                lightglue_state_sha256=self._state_hash(self.methods['matching_model']['matcher']))

    def _state_hash(self,model):
        import hashlib
        h=hashlib.sha256()
        for name,tensor in sorted(model.state_dict().items()):
            h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def synchronize(self): self.torch.cuda.synchronize()

    def release_reference(self):
        """Drop scene-sized rasters before loading/rotating their replacements."""
        self._aligned_reference = None
        self._aligned_key = None
        if hasattr(self, 'cache'):
            self.cache.clear()

    def prepare_reference(self, reference, prior):
        if self.config.get('map_alignment', 'none') == 'none':
            return reference, dict(map_alignment='none', map_yaw_deg=0.)
        from .alignment import align_reference, prior_yaw_deg
        # Conditions of the same frame share a prior. Retain just one full map.
        key = digest(dict(reference=reference.identity, yaw=prior_yaw_deg(prior.camera_to_enu)))
        if getattr(self, '_aligned_key', None) != key:
            self.release_reference()
            self._aligned_reference = align_reference(reference, prior, self.avl.dumpRotateImage)
            self._aligned_key = key
        aligned = self._aligned_reference
        return aligned, dict(map_alignment='prior_yaw', map_yaw_deg=aligned.yaw_deg,
                             map_yaw_source='noisy_camera_prior',
                             map_transform=aligned.map_transform.tolist())

    def seed(self,seed):
        random.seed(seed); np.random.seed(seed)
        self.torch.manual_seed(seed); self.torch.cuda.manual_seed_all(seed)
        self.torch.backends.cudnn.benchmark=False
        self.torch.backends.cudnn.deterministic=True
        self.torch.backends.cuda.enable_flash_sdp(False)
        self.torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.torch.backends.cuda.enable_math_sdp(True)
        self.torch.use_deterministic_algorithms(True)
        cv2.setNumThreads(1)

    def retrieve(self,rgb,K,dist,reference,prior,mode):
        start = time.perf_counter()
        center, resolutions, geometry = retrieval_geometry(reference, prior, K, dist,
                                                            rgb.shape, self.config)
        candidates, scale_metrics = [], []
        for resolution in resolutions:
            tiles, metrics = self._retrieve_scale(rgb, K, dist, reference, prior, mode,
                                                  center, resolution)
            candidates.extend(tiles)
            scale_metrics.append(metrics)
        # CAMP descriptors share the same query and normalization across scales.
        candidates.sort(key=lambda c: c['score'], reverse=True)
        candidates = candidates[:self.config['top_k']]
        for rank, candidate in enumerate(candidates, 1):
            candidate['rank'] = rank
        metrics = dict(geometry, search_center_enu=np.asarray(center).tolist(),
                       search_radius_m=self.config['search_sigma']*prior.sigma_xy_m,
                       retrieval_scales=scale_metrics,
                       retrieval_s=time.perf_counter()-start,
                       gallery_encoding_s=None, retrieval_online_s=None,
                       reference_cache_hit=all(m['reference_cache_hit'] for m in scale_metrics))
        for key in ['gallery_count', 'searched_count', 'feature_extraction_s']:
            metrics[key] = sum(m.get(key, 0) for m in scale_metrics)
        if not candidates:
            metrics['rejection_reason'] = scale_metrics[0].get('rejection_reason', 'retrieval_empty')
        return candidates, metrics

    def _retrieve_scale(self,rgb,K,dist,reference,prior,mode,center,ground_resolution):
        start=time.perf_counter()
        c=self.config
        side=min(rgb.shape[:2])
        view_m=side*ground_resolution*c['patch_scale']
        stride=retrieval_stride(reference,view_m,c['max_patch_pixels'])
        # A view of the same map; full-resolution reference and DSM remain intact.
        retrieval_image=reference.bgr[::stride,::stride]
        retrieval_resolution=reference.resolution*stride
        radius=c['search_sigma']*prior.sigma_xy_m
        metrics=dict(search_radius_m=radius,search_center_enu=np.asarray(center).tolist(),reference_cache_hit=False,
                     retrieval_stride=stride,requested_patch_size_m=float(view_m),
                     ground_resolution_m_per_px=float(ground_resolution))
        selected=[]
        encoded={}

        def prior_grid(image,block_size,step_size):
            ph,pw=block_size
            if ph*pw > c['max_patch_pixels']: raise RuntimeError('Retrieval pixel budget exceeded')
            estimated=math.ceil(image.shape[0]/step_size[0])*math.ceil(image.shape[1]/step_size[1])
            if estimated > c['max_gallery_tiles']: raise RuntimeError('Gallery resource limit exceeded')
            mids=self.avl.compute_block_mid_wo_black(image,block_size,step_size).reshape(-1,2)
            # Keep all candidate coordinates in the ORIGINAL map pixels for PnP.
            selected.extend(geographic_tiles(reference,mids*stride,ph*stride,pw*stride,center,radius,mode))
            metrics.update(gallery_count=len(mids),searched_count=len(selected))
            if not selected: raise NoCandidates('retrieval_empty')
            return np.array([[(t['row']+t['height']/2)/stride,
                              (t['col']+t['width']/2)/stride] for t in selected])

        def cached_extract(size,config,methods,mids,block_size,image,query,transform,model):
            key=digest(dict(reference=reference.identity,tiles=selected,size=size))
            begin=time.perf_counter()
            if key in self.cache:
                gallery=self.cache.pop(key); self.cache[key]=gallery
                # Original extractor also obtains qf. One dummy gallery tile keeps its API unchanged.
                _,qf=self.avl.extract_features(size,config,methods,mids[:1],block_size,image,query,transform,model)
                gf=gallery.to(qf.device)
                metrics['reference_cache_hit']=True
            else:
                gf,qf=self.avl.extract_features(size,config,methods,mids,block_size,image,query,transform,model)
                self.synchronize()
                self.cache[key]=gf.detach().cpu()
                while len(self.cache)>c['gallery_cache_entries']: self.cache.popitem(last=False)
            self.synchronize()
            metrics['feature_extraction_s']=time.perf_counter()-begin
            encoded['scores']=(gf@qf.reshape(-1)).detach().cpu().numpy().reshape(-1)
            return gf,qf

        center_pixel = (reference.enu_to_pixels(center)/stride if hasattr(reference, 'map_transform') else
                        (center*np.array([1.,-1.])-reference.origin)/retrieval_resolution)
        retrieval=bind_upstream(self.avl.retrieval_all_anyvisloc,
            compute_block_mid_wo_black=prior_grid,extract_features=cached_extract,
            view_center_anyvisloc=lambda *args,**kwargs:tuple(center_pixel),
            resolution_size_anyvisloc=lambda *args:(ground_resolution,np.array([side,side])))
        # Only noisy measurements and camera calibration. No sample/GT enters the author function.
        data=dict(x=float(prior.xyz[0]),y=float(-prior.xyz[1]),z=float(prior.xyz[2]),
            rel_alt=float(prior.xyz[2]-reference.plane_z),width=rgb.shape[1],height=rgb.shape[0],K=K)
        try:
            order,rows,cols,_,ph,pw,finescale,_=retrieval(retrieval_image,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),
                data,retrieval_resolution,reference.origin,np.array([[1,0,0],[0,1,0]],np.float32),
                str(self.scratch),self.options,self.upstream_config,self.methods)
        except NoCandidates as exc:
            return [],dict(metrics,retrieval_s=time.perf_counter()-start,rejection_reason=str(exc))
        candidates=[]
        for rank,i in enumerate(order[:c['top_k']]):
            candidates.append(dict(selected[int(i)],rank=rank+1,score=float(encoded['scores'][int(i)]),fine_scale=float(finescale)*stride))
        self.synchronize()
        metrics['retrieval_s']=time.perf_counter()-start
        # Upstream extracts gallery and query together. Do not invent separate timings.
        metrics['gallery_encoding_s']=None
        metrics['retrieval_online_s']=None
        return candidates,metrics

    def match_candidate(self,rgb,K,dist,reference,candidate,seed,capture=False,*,map_transform=None,
                        map_points_to_local=None):
        state={}
        avl=self.avl
        class ObserveCV:
            def __getattr__(self,name): return getattr(cv2,name)
            def solvePnPRansac(self,*args,**kwargs):
                cv2.setRNGSeed(seed)
                result=cv2.solvePnPRansac(*args,**kwargs)
                state.update(world=np.asarray(args[0]),pixels=np.asarray(args[1]),K=np.asarray(args[2]),
                             dist=np.asarray(args[3]),solver_result=result)
                return result
        original_solver=bind_upstream(avl.estimate_drone_pose_anyvisloc,cv2=ObserveCV())
        def observed_matching(*args,**kwargs):
            points=avl.run_pixel_match_anyvisloc(*args,**kwargs)
            state['raw_points']=points
            return points
        overrides = dict(estimate_drone_pose_anyvisloc=original_solver,
                         run_pixel_match_anyvisloc=observed_matching)
        if map_points_to_local is not None:
            def sample_fixed_crop(dsm, xs, ys, resolution, origin):
                # Fixed-crop controls retain the author's staged rounding before
                # DSM indexing. A collapsed inverse affine is not numerically equal.
                exact_x, exact_y = map_points_to_local(state['raw_points'][1])
                if np.shape(exact_x) != np.shape(xs) or np.shape(exact_y) != np.shape(ys):
                    raise ValueError('Fixed-crop point conversion changed correspondence count')
                xs[...] = exact_x
                ys[...] = exact_y
                return avl.sample_dsm_local(dsm, xs, ys, resolution, origin)
            overrides['sample_dsm_local'] = sample_fixed_crop
        matching=bind_upstream(avl.Match2Pos_all_anyvisloc, **overrides)
        r,co,ph,pw=[candidate[k] for k in ('row','col','height','width')]
        options = self.options
        if hasattr(reference, 'map_transform'):
            from types import SimpleNamespace
            options = SimpleNamespace(**vars(self.options))
            options.pose_priori = 'yp'
            if map_transform is None:
                map_transform = reference.map_transform
        positions,inliers,match_times,pnp_times=matching(options,self.upstream_config,
            cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),candidate['fine_scale'],K,reference.bgr,reference.dsm,
            [r],[co],ph,pw,str(self.scratch),self.methods,
            np.array([[1,0,0],[0,1,0]],np.float32) if map_transform is None else map_transform,
            reference.resolution,reference.origin,reference.dsm_resolution,reference.dsm_origin,
            dist=dist,truePos=None)
        self.synchronize()
        raw=state.get('raw_points',(np.empty((0,2)),np.empty((0,2))))
        result=dict(valid_pnp=False,accepted=False,rejection_reason='insufficient_matches',
            matches=len(state.get('world',[])),raw_matches=len(raw[0]),inliers=int(inliers[0]),inlier_ratio=0.,
            matching_s=float(sum(match_times)),pnp_s=float(sum(pnp_times)),rank=candidate['rank'],score=candidate['score'])
        if 'solver_result' in state:
            ok,rvec,tvec,indices=state['solver_result']
            if not ok or indices is None:
                result['rejection_reason']='pnp_failed'
            else:
                R=cv2.Rodrigues(rvec)[0]
                projected=cv2.projectPoints(state['world'],rvec,tvec,state['K'],state['dist'])[0].reshape(-1,2)
                residual=np.linalg.norm(projected-state['pixels'],axis=1)
                idx=indices.reshape(-1)
                median=float(np.median(residual[idx]))
                ratio=len(idx)/len(state['world'])
                positive=float(np.mean((state['world']@R.T+tvec.reshape(3))[:,2]>0))
                p=positions[0]
                xyz=np.array([p['X'],-p['Y'],p['Z']])
                c=self.config
                reason=('nonfinite_pose' if not np.isfinite(xyz).all() else
                    'insufficient_matches' if len(state['world'])<c['min_matches'] else
                    'insufficient_inliers' if len(idx)<c['min_inliers'] else
                    'low_inlier_ratio' if ratio<c['min_inlier_ratio'] else
                    'high_reprojection' if median>c['max_median_reprojection_px'] else
                    'negative_depth' if positive<c['min_positive_depth_fraction'] else '')
                result.update(valid_pnp=True,accepted=not reason,rejection_reason=reason,pred_xyz_enu=xyz.tolist(),
                    camera_to_enu=R.T.tolist(),inlier_ratio=ratio,reprojection_median_px=median,
                    positive_depth_fraction=positive,inlier_indices=idx.tolist())
        if capture:
            scale=self.config['match_resize_ratio']
            result['visualization']=dict(query_pts=(np.asarray(raw[0])/scale).tolist(),
                map_pts=(np.asarray(raw[1])*candidate['fine_scale']/scale).tolist())
            result['pnp_inputs'] = {name: state[name].tolist() for name in ('world','pixels','K','dist')
                                    if name in state}
        return result


def localize(backend, rgb, K, dist, reference, prior, mode, seed, capture=False):
    """GT-free public inference interface. No sample object or true pose accepted."""
    backend.seed(seed)
    backend.synchronize()
    start = time.perf_counter()
    alignment = dict(map_alignment='none', map_yaw_deg=0.)
    if hasattr(backend, 'prepare_reference'):
        reference, alignment = backend.prepare_reference(reference, prior)
    candidates,metrics = backend.retrieve(rgb,K,dist,reference,prior,mode)
    metrics.update(alignment)
    details = []
    for i,candidate in enumerate(candidates):
        details.append(backend.match_candidate(rgb,K,dist,reference,candidate,seed+i,capture))
    accepted = [d for d in details if d["accepted"]]
    pool = accepted or details
    # Ranking uses only geometry and retrieval score, never evaluator coordinates.
    best = max(pool,key=lambda d:(d["inliers"],-d.get("reprojection_median_px",float("inf")),d["score"])) if pool else {}
    backend.synchronize()
    return dict(metrics,accepted=bool(accepted),status="inference_complete",
        rejection_reason="" if accepted else best.get("rejection_reason",metrics.get("rejection_reason","retrieval_empty")),
        pred_xyz_enu=best.get("pred_xyz_enu"),valid_pnp=best.get("valid_pnp",False),
        raw_matches=best.get("raw_matches",0),matches=best.get("matches",0),inliers=best.get("inliers",0),
        inlier_ratio=best.get("inlier_ratio",0.),reprojection_median_px=best.get("reprojection_median_px"),
        selected_rank=best.get("rank"),candidates=candidates,candidate_results=details,
        returned_count=len(candidates),top_scores=[c["score"] for c in candidates],
        matching_s=sum(d["matching_s"] for d in details),pnp_s=sum(d["pnp_s"] for d in details),
        total_s=time.perf_counter()-start)

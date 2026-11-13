"""Read-only CPU compatibility check; synthetic geometry, not real measurements."""
import ast
from pathlib import Path
import subprocess
import time
from collections import OrderedDict
from types import SimpleNamespace
from PIL import Image

import cv2
import numpy as np

from .backend import estimate_pose,AnyVisLocBackend
from .core import defaults,UPSTREAM_COMMIT,Prior
from .data import Reference


def verify(source, *, ablation=False, quick_visibility=False):
    root = Path(source)
    commit = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if commit != UPSTREAM_COMMIT: raise ValueError("Unexpected upstream commit")
    tree = ast.parse((root/"avl_utils.py").read_text(encoding="utf-8"))
    names = {"rotvector2rot","rot_to_euler","estimate_drone_pose_anyvisloc","_pnp_flag",
             "as_vec2","dsm_to_2d","sample_dsm_local","ensure_dir","Match2Pos_all_anyvisloc",
             "retrieval_all_anyvisloc","crop_center_from_bgr","_even_ceil","compute_block_mid_wo_black",
             "dumpRotateImage","pos2error_anyvisloc","view_center_anyvisloc",
             "resolution_size_anyvisloc","_apply_affine_xy"}
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(nodes) != len(names): raise ValueError("Upstream interface changed")
    import math
    env = dict(np=np,cv2=cv2,Path=Path,time=time,math=math,Image=Image)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"pinned_upstream_pose","exec"),env)
    rng = np.random.default_rng(123)
    points = rng.uniform([-30,-30,-5],[30,30,5],(100,3))
    R = np.diag([1.,-1.,-1.])
    xyz = np.array([0.,0.,100.])
    K = np.array([[500.,0.,160.],[0.,500.,120.],[0.,0.,1.]])
    pixels = cv2.projectPoints(points,cv2.Rodrigues(R)[0],-R@xyz,K,np.zeros(5))[0].reshape(-1,2)
    pixels += rng.normal(0,.1,pixels.shape)
    ours = estimate_pose(pixels,points,K,np.zeros(5),defaults(),42)
    cv2.setRNGSeed(42)
    original = env["estimate_drone_pose_anyvisloc"](pixels,K,points[:,0],-points[:,1],points[:,2],
        dist_coeffs=np.zeros(5),pnp_method="P3P",reprojection_error=8.,iterations_count=2000,confidence=.999)
    original_xyz = np.array([original[0]["X"],-original[0]["Y"],original[0]["Z"]])
    np.testing.assert_allclose(ours["pred_xyz_enu"],original_xyz,atol=1e-4,rtol=0)
    assert ours["inliers"] == original[2]
    mids = env['compute_block_mid_wo_black'](np.full((400,400,3),100,np.uint8),[80,80],[40,40])
    assert mids.shape[1] == 2 and len(mids) > 0
    for filename in ["avl_data.py","run_avl.py","run_avl_match_loc.py","config_selectable_matchers.yaml"]:
        if not (root/filename).is_file(): raise FileNotFoundError(filename)
    # Exercise the ORIGINAL retrieval and matching bodies without neural weights.
    # Only model features/matches are test fixtures; no resulting scores are scientific data.
    class TensorFixture:
        def __init__(self,array): self.array=np.asarray(array); self.device="cpu"
        def __matmul__(self,other): return TensorFixture(self.array@other.array)
        def reshape(self,*shape): return TensorFixture(self.array.reshape(*shape))
        def unsqueeze(self,axis): return TensorFixture(np.expand_dims(self.array,axis))
        def squeeze(self): return TensorFixture(self.array.squeeze())
        def detach(self): return self
        def cpu(self): return self
        def to(self,device): return self
        def numpy(self): return self.array
    def feature_fixture(size,config,methods,mids,*args):
        return TensorFixture(np.column_stack([np.arange(len(mids)),np.ones(len(mids))])),TensorFixture([1.,0.])
    avl=SimpleNamespace(**{name:env[name] for name in names})
    avl.extract_features=feature_fixture
    engine=AnyVisLocBackend.__new__(AnyVisLocBackend)
    engine.avl=avl
    engine.config=defaults()
    engine.synchronize=lambda:None
    engine.cache=OrderedDict()
    import tempfile
    with tempfile.TemporaryDirectory(prefix="visibility-upstream-") as temporary:
        engine.scratch=Path(temporary)
        engine.methods=dict(retrieval_cover=50,retrieval_model=None,img_transform=lambda im:np.asarray(im))
        engine.upstream_config=dict(RETRIEVAL_SIZE=384)
        engine.options=SimpleNamespace(pose_priori="p",visualize=False,strategy="Top1",
            disable_ref_feature_cache=True,patch_scale=1.5,resize_ratio=.4,
            PnP_method="P3P",pnp_reproj_error=8.,pnp_iterations=2000,pnp_confidence=.999)
        reference=Reference("synthetic",np.full((400,400,3),100,np.uint8),np.zeros((400,400)),
            np.ones(2),np.array([-200.,-200.]),np.ones(2),np.array([-200.,-200.]),0.,{"synthetic":True})
        prior=Prior(xyz,R,20.,5.,2.,5.)
        query=np.full((240,320,3),100,np.uint8)
        candidates,metrics=engine.retrieve(query,K,np.zeros(5),reference,prior,"prior")
        assert len(candidates)==5 and metrics["searched_count"] <= metrics["gallery_count"]
        candidate=dict(row=160,col=160,height=80,width=80,rank=1,score=1.,fine_scale=1.)
        world=np.column_stack([rng.uniform(-20,20,(100,2)),np.zeros(100)])
        query_points=cv2.projectPoints(world,cv2.Rodrigues(R)[0],-R@xyz,K,np.zeros(5))[0].reshape(-1,2)
        map_points=(world[:,:2]*[1,-1]-reference.origin-[160,160])*.4
        avl.run_pixel_match_anyvisloc=lambda *a,**kw:(query_points*.4,map_points)
        result=engine.match_candidate(query,K,np.zeros(5),reference,candidate,42)
        assert result["valid_pnp"] and result["accepted"]
        np.testing.assert_allclose(result["pred_xyz_enu"],xyz,atol=.01,rtol=0)
        # Production yaw path: original rotation -> retrieval -> inverse map
        # transform -> DSM -> original P3P. Matches are synthetic fixtures only.
        from scipy.spatial.transform import Rotation
        original_map_hash = __import__('hashlib').sha256(reference.bgr.tobytes()).hexdigest()
        engine.config['map_alignment'] = 'prior_yaw'
        for yaw in [-137., 0., 37., 179.]:
            camera = Rotation.from_euler('z', -yaw, degrees=True).as_matrix() @ R
            rotated_prior = Prior(xyz, camera, 20., 5., 2., 5.)
            aligned, alignment = engine.prepare_reference(reference, rotated_prior)
            assert np.isclose(alignment['map_yaw_deg'], yaw)
            aligned_tiles, _ = engine.retrieve(query,K,np.zeros(5),aligned,rotated_prior,'prior')
            assert len(aligned_tiles) == 5
            full_map_points = world[:,:2]*[1,-1]-reference.origin
            rotated_points = cv2.transform(full_map_points[None,:,:],aligned.map_transform)[0]
            tile = dict(row=0,col=0,height=aligned.bgr.shape[0],width=aligned.bgr.shape[1],
                        rank=1,score=1.,fine_scale=1.)
            avl.run_pixel_match_anyvisloc=lambda *a,**kw:(query_points*.4,rotated_points*.4)
            aligned_result=engine.match_candidate(query,K,np.zeros(5),aligned,tile,42,capture=True)
            assert aligned_result['accepted']
            np.testing.assert_allclose(aligned_result['pnp_inputs']['world'],world,atol=1e-4,rtol=0)
            np.testing.assert_allclose(aligned_result['pred_xyz_enu'],xyz,atol=.01,rtol=0)
            assert engine.options.pose_priori == 'p'  # No persistent options mutation.
        assert __import__('hashlib').sha256(reference.bgr.tobytes()).hexdigest() == original_map_hash
        engine.config['map_alignment'] = 'none'
        avl.run_pixel_match_anyvisloc=lambda *a,**kw:(query_points*.4,map_points)
        if ablation:
            # Fixed rotated GT crop used ONLY by the separate ablation notebook.
            from .ablation import crop_transform, crop_points_to_local
            from dataclasses import replace
            transform = crop_transform(dict(rotation_matrix=cv2.getRotationMatrix2D((40.,40.),27.,1.),
                                            pre_crop=[160,160,240,240],final_crop=[0,0,80,80]))
            full_map_points = world[:,:2]*[1,-1]-reference.origin
            patch_points = cv2.transform(full_map_points[None,:,:],transform)[0]
            avl.run_pixel_match_anyvisloc=lambda *a,**kw:(query_points*.4,patch_points*.4)
            engine.options.pose_priori='yp'
            rotated_result=engine.match_candidate(query,K,np.zeros(5),replace(reference,bgr=reference.bgr[:80,:80]),
                dict(row=0,col=0,height=80,width=80,rank=1,score=1.,fine_scale=1.),42,map_transform=transform)
            assert rotated_result['accepted']
            np.testing.assert_allclose(rotated_result['pred_xyz_enu'],xyz,atol=.01,rtol=0)
            # Compare to ACTUAL author conversion, including float32 scalar
            # rounding and large offsets, before nearest-neighbor DSM sampling.
            runner_tree=ast.parse((root/'run_avl_match_loc.py').read_text(encoding='utf-8'))
            conversion_names={'ref_patch_pts_to_local','pixel_to_local'}
            conversion_nodes=[n for n in runner_tree.body
                              if isinstance(n,ast.FunctionDef) and n.name in conversion_names]
            assert len(conversion_nodes)==len(conversion_names)
            conversion_env=dict(np=np,cv2=cv2)
            exec(compile(ast.Module(body=conversion_nodes,type_ignores=[]),'author_crop_conversion','exec'),conversion_env)
            crop_info=dict(rotation_matrix=cv2.getRotationMatrix2D((380.,410.),37.123,1.),
                           pre_crop=[27183,19547,27943,20367],final_crop=[71,49,651,589],
                           map_resolution=[.073245,.073245],map_origin_local=[-713.2481,-483.137])
            raw_points=rng.uniform(0,600,(3000,2)).astype(np.float32)
            ref_resize=.38123456789
            author_xy=conversion_env['ref_patch_pts_to_local'](raw_points/ref_resize,crop_info)
            converted=crop_points_to_local(raw_points,crop_info,ref_resize)
            np.testing.assert_array_equal(converted[0],author_xy[0])
            np.testing.assert_array_equal(converted[1],author_xy[1])
            exact_result=engine.match_candidate(query,K,np.zeros(5),replace(reference,bgr=reference.bgr[:80,:80]),
                dict(row=0,col=0,height=80,width=80,rank=1,score=1.,fine_scale=1.),42,map_transform=transform,
                map_points_to_local=lambda points:(world[:,0],-world[:,1]))
            assert exact_result['accepted']
            np.testing.assert_allclose(exact_result['pred_xyz_enu'],xyz,atol=.01,rtol=0)
            engine.options.pose_priori='p'
            avl.run_pixel_match_anyvisloc=lambda *a,**kw:(query_points*.4,map_points)
        # Invalid prior height now exercises the ORIGINAL retrieval at all fallback
        # scales, including decimation, without injecting evaluation coordinates.
        engine.config['max_patch_pixels'] = 1000
        below_plane = Prior(np.array([0.,0.,-10.]), R, 20., 5., 2., 5.)
        fallback_candidates, fallback_metrics = engine.retrieve(query,K,np.zeros(5),reference,below_plane,'prior')
        assert fallback_metrics['geometry_fallback'] and len(fallback_candidates) == 5
        assert len(fallback_metrics['retrieval_scales']) == 3
        assert all(m['retrieval_stride'] > 1 for m in fallback_metrics['retrieval_scales'])
        for tile in fallback_candidates:
            assert tile['row'] >= 0 and tile['col'] >= 0
            assert tile['row']+tile['height'] <= reference.bgr.shape[0]
            assert tile['col']+tile['width'] <= reference.bgr.shape[1]
        fallback_match = engine.match_candidate(query,K,np.zeros(5),reference,fallback_candidates[0],42)
        assert fallback_match['raw_matches'] == len(query_points)
        assert fallback_match['matches'] > 0
        if quick_visibility:
            from .quick_visibility import run_frame
            original_retrieval=avl.retrieval_all_anyvisloc
            original_matcher=avl.run_pixel_match_anyvisloc
            geometry={}
            def record_retrieval(*args):
                # Retain the original body and the quick experiment's cache hook.
                from .backend import bind_upstream
                result=bind_upstream(original_retrieval,extract_features=extract_features)(*args)
                geometry.update(rows=result[1],cols=result[2],scale=result[6])
                return result
            avl.retrieval_all_anyvisloc=record_retrieval
            avl.anyvisloc_sample_to_truepos=lambda sample:dict(x=0.,y=0.,z=100.,rel_alt=100.,
                yaw=0.,pitch=0.,width=320,height=240,K=K)
            avl.build_anyvisloc_reference_view=lambda sample,mode:dict(map=reference.bgr,dsm=reference.dsm,
                map_resolution=reference.resolution,map_origin_local=reference.origin,
                dsm_resolution=reference.dsm_resolution,dsm_origin_local=reference.dsm_origin)
            def known_correspondences(*args,**kwargs):
                rank=int(Path(args[4]).name.split('_')[0].removeprefix('top'))-1
                offset=[geometry['cols'][rank],geometry['rows'][rank]]
                original_pixels=world[:,:2]*[1,-1]-reference.origin
                return query_points*.4,(original_pixels-offset)*.4/geometry['scale']
            avl.run_pixel_match_anyvisloc=known_correspondences
            engine.config['max_patch_pixels']=4000000
            engine.seed=lambda seed:None
            options=SimpleNamespace(**vars(engine.options))
            options.pose_priori='yp'
            options.strategy='Topn_opt'
            options.patch_scale=1.
            engine.methods['retrieval_topn']=5
            sample=dict(K=TensorFixture(K),dist=TensorFixture(np.zeros(5)))
            gallery={}
            for brightness in [100,50]:
                measured=run_frame(engine,sample,np.full_like(query,brightness),options,{},42,engine.scratch,gallery)
                assert measured['valid_pnp'] and measured['inliers']==len(world)
                assert measured['matches']==len(world) and measured['error_xy_m'] < .01
            assert measured['gallery_cache_hit']
            avl.retrieval_all_anyvisloc=original_retrieval
            avl.run_pixel_match_anyvisloc=original_matcher
    return dict(synthetic_check_only=True,commit=commit,original_pnp_agrees_at_1e_4_m=True,
                original_grid_executed=True,original_retrieval_executed=True,
                original_matching_pnp_executed=True, fallback_multiscale_retrieval_executed=True,
                retrieval_decimation_executed=True, prior_yaw_retrieval_matching_pnp_verified=True,
                fixed_crop_conversion_verified=ablation,quick_author_visibility_verified=quick_visibility)

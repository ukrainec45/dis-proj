"""Synthetic checks only; these do not validate neural inference on AnyVisLoc."""
import copy
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation
from shapely.geometry import box

from .core import (CONDITIONS,defaults,degrade,make_prior,nadir_angle,footprint,
                   retrieval_truth,score_result,digest,array_hash)
from .backend import (estimate_pose,localize,geographic_tiles,bind_upstream,
                      retrieval_geometry,retrieval_stride,AnyVisLocBackend)
from .data import read_sample,load_reference
from .campaign import run_experiment,read_json
from .analysis import summarize,plot_results
from .colab import stage_archive

NADIR = np.diag([1.,-1.,-1.])
K = np.array([[500.,0,160],[0,500,120],[0,0,1.]])


def test_adapter_keeps_original_globals_unchanged():
    from types import SimpleNamespace
    def original(values=(1,2)):
        return np.sum(values)
    adapted=bind_upstream(original,np=SimpleNamespace(sum=lambda values:100))
    assert original()==3 and adapted()==100
    assert original.__globals__['np'] is np
    assert adapted.__code__ is original.__code__
    assert adapted.__globals__ is not original.__globals__


@pytest.mark.parametrize("condition",CONDITIONS)
def test_observation_models(condition):
    rgb = np.random.default_rng(1).integers(0,256,(240,320,3),dtype=np.uint8)
    original = rgb.copy()
    a,t,_ = degrade(rgb,condition,defaults(),123)
    b,t2,_ = degrade(rgb,condition,defaults(),123)
    assert np.array_equal(a,b) and np.array_equal(t,t2)
    assert a.shape == rgb.shape and a.dtype == np.uint8 and 0 <= a.min() <= a.max() <= 255
    assert np.array_equal(rgb,original)
    if condition == "clear": assert np.array_equal(a,rgb) and np.all(t == 1)
    if condition.startswith("smoke"): assert t.std() > .05


def test_prior_distribution_and_rotations():
    profile = defaults()["profiles"]["main"]
    errors = np.array([make_prior(np.array([0,0,100.]),NADIR,20,profile,i)[1]["prior_error_enu_m"] for i in range(5000)])
    assert np.allclose(errors.std(axis=0),[20,20,5],rtol=.04)
    assert np.max(np.abs(errors.mean(axis=0))) < 1
    p,meta = make_prior(np.array([1,2,100.]),NADIR,20,profile,42)
    p2,meta2 = make_prior(np.array([1,2,100.]),NADIR,20,profile,42)
    assert meta == meta2 and np.array_equal(p.xyz,p2.xyz)
    assert np.allclose(p.camera_to_enu.T@p.camera_to_enu,np.eye(3))
    assert np.isclose(np.linalg.det(p.camera_to_enu),1.)
    assert nadir_angle(NADIR) == 0
    assert np.isclose(nadir_angle(Rotation.from_euler("x",15,degrees=True).as_matrix()@NADIR),15)


def test_overlap_and_footprint():
    f,center = footprint(np.array([0,0,100.]),NADIR,K,np.zeros(5),(240,320,3),0,center_crop=True)
    assert np.allclose(center,[0,0],atol=.2)
    result = retrieval_truth([dict(polygon_enu=list(box(0,-24,24,24).exterior.coords)[:-1])],box(-24,-24,24,24),[0,0])
    assert result["recall_overlap_50"] and not result["recall_overlap_75"]
    with pytest.raises(ValueError,match="unsupported_prior_geometry"):
        footprint(np.array([0,0,-1.]),NADIR,K,np.zeros(5),(240,320),0)


def test_geographic_search_uses_only_prior_and_keeps_global_comparison(tmp_path):
    selected = fixture_scene(tmp_path)[0]
    ref = load_reference(selected["reference_json"])
    mids = np.array([[50,50],[200,200],[350,350],[350,350]])
    near_origin = geographic_tiles(ref,mids,40,40,[0,0],10,"prior")
    far_prior = geographic_tiles(ref,mids,40,40,[150,-150],10,"prior")
    global_tiles = geographic_tiles(ref,mids,40,40,[100000,100000],10,"global")
    empty = geographic_tiles(ref,mids,40,40,[100000,100000],10,"prior")
    assert len(near_origin) == 1 and near_origin[0]["row"] == 180
    assert len(far_prior) == 1 and far_prior[0]["row"] == 330
    assert len(global_tiles) == 3 and empty == []


def test_local_terrain_replaces_misleading_scene_plane(tmp_path):
    selected = fixture_scene(tmp_path)[0]
    ref = load_reference(selected['reference_json'])
    ref.plane_z = 200.  # Remote high terrain would reject a camera at Z=100.
    prior, _ = make_prior(np.array([0.,0.,100.]), NADIR, 20,
                          dict(yaw_deg=0.,tilt_deg=0.,z_m=0.), 42)
    center, scales, metrics = retrieval_geometry(ref, prior, K, np.zeros(5), (240,320,3), defaults())
    assert metrics['retrieval_plane_z_m'] == 0.
    assert metrics['retrieval_plane_source'] == 'local_dsm'
    assert not metrics['geometry_fallback'] and len(scales) == 1


def test_dsm_median_handles_readonly_selections_without_modifying_source(tmp_path, monkeypatch):
    from . import data
    selected=fixture_scene(tmp_path)[0]
    class ReadonlySelection(np.ndarray):
        def __getitem__(self,key):
            result=super().__getitem__(key)
            if isinstance(key,np.ndarray) and isinstance(result,np.ndarray):
                result.flags.writeable=False
            return result
    source=np.array([[4.,np.nan],[2.,8.]],np.float32).view(ReadonlySelection)
    source.flags.writeable=False
    before=source.tobytes()
    assert not source[np.isfinite(source)].flags.writeable
    monkeypatch.setattr(data.np,'load',lambda *a,**kw:source)
    reference=data.load_reference(selected['reference_json'])
    assert reference.plane_z==4.
    assert source.tobytes()==before and not source.flags.writeable


@pytest.mark.parametrize('invalid_height', [True, False])
def test_invalid_prior_geometry_searches_explicit_scales(tmp_path, invalid_height):
    selected = fixture_scene(tmp_path)[0]
    ref = load_reference(selected['reference_json'])
    prior, _ = make_prior(np.array([0.,0.,-10. if invalid_height else 100.]),
                          NADIR if invalid_height else np.eye(3), 20,
                          dict(yaw_deg=0.,tilt_deg=0.,z_m=0.), 42)
    center, scales, metrics = retrieval_geometry(ref, prior, K, np.zeros(5), (240,320,3), defaults())
    assert metrics['geometry_fallback']
    np.testing.assert_array_equal(center, prior.xyz[:2])
    np.testing.assert_allclose(np.array(scales)*240*defaults()['patch_scale'], [64.,128.,256.])


def test_large_patch_is_resampled_without_changing_match_scale(tmp_path):
    from collections import OrderedDict
    from types import SimpleNamespace
    selected = fixture_scene(tmp_path)[0]
    ref = load_reference(selected['reference_json'])
    config = defaults()
    config['max_patch_pixels'] = 10000
    stride = retrieval_stride(ref, 240., config['max_patch_pixels'])
    assert stride > 1
    # Geometry fixture calls the same replacement hooks as pinned AVL; no neural inference.
    def retrieval(image, query, data, resolution, origin, affine, scratch, options, config, methods):
        ground, size = resolution_size_anyvisloc(data, options)
        patch = (2*np.ceil(size[0]*ground*options.patch_scale/resolution[::-1]/2)).astype(int)
        mids = compute_block_mid_wo_black(image, patch, patch//2)
        extract_features(384, config, methods, mids, patch, image, query, None, None)
        return np.arange(len(mids)), [], [], [], *patch, ground/resolution.mean(), 0.
    class Feature:
        device = 'cpu'
        def __init__(self, x): self.x = np.asarray(x)
        def __matmul__(self, other): return Feature(self.x@other.x)
        def reshape(self, *shape): return Feature(self.x.reshape(*shape))
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.x
    def extract(*args):
        return Feature(np.ones((len(args[3]), 2))), Feature([1., 0.])
    engine = AnyVisLocBackend.__new__(AnyVisLocBackend)
    engine.config, engine.cache = config, OrderedDict()
    engine.synchronize = lambda: None
    engine.scratch, engine.upstream_config, engine.methods = tmp_path, {}, {}
    engine.options = SimpleNamespace(patch_scale=config['patch_scale'])
    engine.avl = SimpleNamespace(retrieval_all_anyvisloc=retrieval, extract_features=extract,
                                compute_block_mid_wo_black=lambda image, patch, step: np.array([[50.,50.]]))
    prior, _ = make_prior(np.array([0.,0.,100.]), NADIR, 20, config['profiles']['main'], 42)
    before = array_hash(ref.bgr)
    tiles, metrics = engine._retrieve_scale(np.zeros((240,320,3),np.uint8), K, np.zeros(5),
                                           ref, prior, 'global', np.array([0.,0.]), 2/3)
    assert len(tiles) == 1 and metrics['retrieval_stride'] == stride
    tile = tiles[0]
    assert tile['height'] == tile['width'] == 240
    assert tile['fine_scale'] == pytest.approx(2/3)
    assert (tile['height']//stride)*(tile['width']//stride) <= config['max_patch_pixels']
    assert array_hash(ref.bgr) == before


def test_choose_candidate_by_inliers_not_evaluation_error(tmp_path):
    selected = fixture_scene(tmp_path)[0]
    sample,ref = read_sample(selected["path"]),load_reference(selected["reference_json"])
    class TwoCandidates(SyntheticBackend):
        def retrieve(self,*args):
            return [dict(rank=1,score=.9),dict(rank=2,score=.8)],{}
        def match_candidate(self,rgb,K,dist,reference,candidate,seed,capture=False):
            return dict(accepted=True,valid_pnp=True,pred_xyz_enu=[1000.,0.,100.] if candidate["rank"] == 1 else [0.,0.,100.],
                inliers=30 if candidate["rank"] == 1 else 15,matches=40,raw_matches=40,inlier_ratio=.75,
                reprojection_median_px=.2,matching_s=.01,pnp_s=.01,**candidate)
    prior,_ = make_prior(sample["xyz_enu"],sample["rotation"],20,defaults()["profiles"]["main"],42)
    prediction = localize(TwoCandidates(defaults()),sample["rgb"],K,np.zeros(5),ref,prior,"prior",42)
    assert prediction["selected_rank"] == 1
    assert score_result(prediction,sample["xyz_enu"])["status"] == "false_accept"


def test_pnp_with_outliers():
    rng = np.random.default_rng(18)
    xyz = np.array([20.,-10.,100.])
    points = np.column_stack([rng.uniform(-25,60,100),rng.uniform(-50,25,100),rng.uniform(-5,10,100)])
    rvec = cv2.Rodrigues(NADIR)[0]
    pixels = cv2.projectPoints(points,rvec,-NADIR@xyz,K,np.zeros(5))[0].reshape(-1,2)
    pixels += rng.normal(0,.15,pixels.shape)
    pixels[:20] = rng.uniform([0,0],[320,240],(20,2))
    result = estimate_pose(pixels,points,K,np.zeros(5),defaults(),42)
    assert result["accepted"] and result["inliers"] >= 75
    assert np.linalg.norm(np.asarray(result["pred_xyz_enu"])-xyz) < .5
    assert result["reprojection_median_px"] < 1
    failed = estimate_pose(pixels[:3],points[:3],K,np.zeros(5),defaults(),42)
    assert not failed["valid_pnp"] and not failed["accepted"]


def fixture_scene(tmp_path):
    root = tmp_path/"data"/"Scene_05"
    root.mkdir(parents=True)
    cv2.imwrite(str(root/"satellite_map.png"),np.full((400,400,3),150,np.uint8))
    np.save(root/"satellite_dsm.npy",np.zeros((400,400),np.float32))
    (root/"L05_reference.json").write_text(json.dumps(dict(modes=dict(satellite=dict(
        map_path="satellite_map.png",dsm_path="satellite_dsm.npy",map_resolution=[1,1],
        dsm_resolution=[1,1],map_origin_local=[-200,-200])))))
    pose = np.eye(4)
    pose[:3,3] = [0,0,100]
    np.savez(root/"L05_0001.npz",image=np.full((240,320,3),100,np.uint8),K=K,dist=np.zeros(5),
             pose_c2w=pose,xyz=[0,0,100],euler_deg=[0,0,0])
    return [dict(scene_id="Scene_05",sample_id="L05_0001",path=str(root/"L05_0001.npz"),
                 reference_json=str(root/"L05_reference.json"))]


class SyntheticBackend:
    """Orchestration fixture, deliberately not a neural/localization substitute."""
    def __init__(self,config,technical=False):
        self.config,self.technical,self.calls,self.priors = config,technical,0,[]
        self.identity = dict(test_fixture=True)
    def seed(self,seed): pass
    def synchronize(self): pass
    def retrieve(self,rgb,K,dist,reference,prior,mode):
        self.calls += 1
        self.priors.append(prior.xyz.copy())
        if self.technical: raise RuntimeError("synthetic technical failure")
        # No GT access: ranking and mock positions fixed by fixture inputs.
        return [dict(rank=1,score=.9,polygon_enu=[[-30,-30],[30,-30],[30,30],[-30,30]])],dict(retrieval_s=.01)
    def match_candidate(self,*args):
        return dict(accepted=True,valid_pnp=True,pred_xyz_enu=[0.,0.,100.],inliers=15,
                    matches=20,raw_matches=20,inlier_ratio=.75,reprojection_median_px=.2,
                    matching_s=.01,pnp_s=.01,rank=1,score=.9,rejection_reason="")


def test_inference_gt_independence_and_reference_immutability(tmp_path):
    selected = fixture_scene(tmp_path)[0]
    sample = read_sample(selected["path"])
    reference = load_reference(selected["reference_json"])
    before = array_hash(reference.bgr)
    prior,_ = make_prior(sample["xyz_enu"],sample["rotation"],20,defaults()["profiles"]["main"],42)
    backend = SyntheticBackend(defaults())
    a = localize(backend,sample["rgb"],K,np.zeros(5),reference,prior,"prior",42)
    # Separate evaluator gets different GT; every inference input stays fixed.
    one = score_result(a,[0,0,100])
    two = score_result(a,[1000,1000,100])
    b = localize(backend,sample["rgb"],K,np.zeros(5),reference,prior,"prior",42)
    # Wall time is intentionally excluded from the equality comparison.
    assert {k:v for k,v in a.items() if k != "total_s"} == {k:v for k,v in b.items() if k != "total_s"}
    assert one["status"] == "correct" and two["status"] == "false_accept"
    assert array_hash(reference.bgr) == before


def test_paired_resume_and_identity(tmp_path):
    selected = fixture_scene(tmp_path)
    c = defaults()
    c.update(sigma_xy_m=[20],trials=1)
    backend = SyntheticBackend(c)
    output = tmp_path/"results"
    frame = run_experiment(backend,selected,c,output)
    assert len(frame) == 6 and backend.calls == 6
    assert all(np.array_equal(backend.priors[0],p) for p in backend.priors)
    assert len(set(frame.prior_seed)) == 1
    run_experiment(backend,selected,c,output)
    assert backend.calls == 6
    summary,paired = summarize(frame,output)
    assert np.all(summary.correct_fraction == 1.) and np.all(paired.delta_correct == 0)
    plot_results(frame,output)
    manifest = read_json(output/"experiment_config.json")
    assert manifest["ground_truth_used_in_inference"] is False
    c["seed"] = 99
    with pytest.raises(ValueError,match="identity changed"):
        run_experiment(SyntheticBackend(c),selected,c,output)


def test_technical_errors_and_missing_error_are_not_zero(tmp_path):
    c = defaults()
    c.update(sigma_xy_m=[20],trials=1)
    output = tmp_path/"results"
    frame = run_experiment(SyntheticBackend(c,technical=True),fixture_scene(tmp_path),c,output)
    assert len(frame) == 6 and frame.status.eq("technical_error").all()
    assert frame.error_xy_m.isna().all()
    summary,paired = summarize(frame,output)
    assert summary.n_algorithm_trials.eq(0).all() and summary.correct_fraction.isna().all()
    assert not paired.pair_valid.any()


def test_denominators_include_rejections_and_false_accepts(tmp_path):
    rows = []
    for i,(status,error) in enumerate([("correct",1.),("rejected",None),("false_accept",50.),("technical_error",None)]):
        rows.append(dict(scene_id="synthetic",condition="clear",sigma_xy_m=20,profile="main",retrieval_mode="prior",
            reference_mode="satellite",experiment_id="test",pair_id=str(i),status=status,error_xy_m=error,
            accepted=status in ("correct","false_accept"),inliers=10))
    summary,_ = summarize(pd.DataFrame(rows),tmp_path)
    row = summary.iloc[0]
    assert row.n_attempted == 4 and row.n_algorithm_trials == 3
    assert row.correct_fraction == pytest.approx(1/3) and row.false_accept_fraction == pytest.approx(1/3)
    assert row.error_mean_m == 25.5


def test_archive_traversal_and_changed_file(tmp_path):
    archive = tmp_path/"bad.zip"
    with zipfile.ZipFile(archive,"w") as f: f.writestr("../escape.txt","bad")
    with pytest.raises(ValueError,match="Unsafe"): stage_archive(archive,tmp_path/"staged")
    archive = tmp_path/"good.zip"
    with zipfile.ZipFile(archive,"w") as f: f.writestr("Scene_05/data.txt","original")
    stage_archive(archive,tmp_path/"staged")
    stage_archive(archive,tmp_path/"staged")
    (tmp_path/"staged"/"Scene_05"/"data.txt").write_text("changed")
    with pytest.raises(ValueError,match="changed"): stage_archive(archive,tmp_path/"staged")

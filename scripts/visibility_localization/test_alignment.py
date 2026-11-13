"""CPU geometry/record tests for measured-yaw alignment, without neural inference."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .alignment import align_reference, prior_yaw_deg
from .backend import AnyVisLocBackend, geographic_tiles
from .campaign import run_experiment
from .core import CONDITIONS, Prior, array_hash, defaults, validate_config
from .data import load_reference
from .test_experiment import fixture_scene, SyntheticBackend, NADIR


@pytest.mark.parametrize('yaw', [-179., -90., 0., 37., 179.])
def test_camera_prior_recovers_author_yaw_with_tilt(yaw):
    raw = Rotation.from_euler('ZYX', [yaw, 8., -6.], degrees=True).as_matrix()
    ned_to_enu = np.array([[0,1,0],[1,0,0],[0,0,-1.]])
    camera_to_frd = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    assert prior_yaw_deg(ned_to_enu @ raw @ camera_to_frd) == pytest.approx(yaw)


def test_rotated_map_preserves_world_polygons_and_original_dsm(tmp_path):
    ref = load_reference(fixture_scene(tmp_path)[0]['reference_json'])
    before = array_hash(ref.bgr)
    ref.dsm = np.arange(400*400).reshape(400,400)
    prior = Prior(np.array([0.,0.,100.]), Rotation.from_euler('z',-37,degrees=True).as_matrix()@NADIR,
                  20.,5.,2.,5.)
    transform = cv2.getRotationMatrix2D((200,200),37,1.)
    aligned = align_reference(ref,prior,lambda image,yaw:(cv2.warpAffine(image,transform,(400,400)),transform))
    pixels = np.array([[50.,70.],[200.,200.],[300.,250.]])
    rotated = cv2.transform(pixels[None],transform)[0]
    np.testing.assert_allclose(aligned.pixels_to_enu(rotated),ref.pixels_to_enu(pixels),atol=1e-10)
    for pixel, enu in zip(rotated,ref.pixels_to_enu(pixels)):
        np.testing.assert_allclose(aligned.enu_to_pixels(enu),pixel,atol=1e-10)
    np.testing.assert_array_equal(aligned.heights([[0.,0.]])[0],ref.heights([[0.,0.]])[0])
    assert aligned.dsm is ref.dsm and array_hash(ref.bgr) == before
    # Local filtering measures distance after undoing rotation, not raster pixels.
    midpoint=aligned.enu_to_pixels([0.,0.])[::-1].reshape(1,2)
    assert len(geographic_tiles(aligned,midpoint,40,40,[0.,0.],1.,'prior')) == 1
    assert not geographic_tiles(aligned,midpoint,40,40,[1000.,1000.],1.,'prior')


def test_alignment_is_cached_by_map_and_prior_yaw_and_disabled_by_default(tmp_path):
    ref = load_reference(fixture_scene(tmp_path)[0]['reference_json'])
    engine=AnyVisLocBackend.__new__(AnyVisLocBackend)
    engine.config=defaults()
    calls=[]
    def rotate(image,yaw):
        calls.append(yaw)
        matrix=cv2.getRotationMatrix2D((200,200),yaw,1.)
        return cv2.warpAffine(image,matrix,(400,400)),matrix
    engine.avl=SimpleNamespace(dumpRotateImage=rotate)
    prior=Prior(np.array([0.,0.,100.]),NADIR,20.,5.,2.,5.)
    assert engine.prepare_reference(ref,prior)[0] is ref and not calls
    engine.config['map_alignment']='prior_yaw'
    first,meta=engine.prepare_reference(ref,prior)
    assert engine.prepare_reference(ref,prior)[0] is first and len(calls)==1
    changed=Prior(prior.xyz,Rotation.from_euler('z',-15,degrees=True).as_matrix()@NADIR,20.,5.,2.,5.)
    second,_=engine.prepare_reference(ref,changed)
    assert second is not first and len(calls)==2 and first.identity!=second.identity
    assert meta['map_yaw_source']=='noisy_camera_prior'
    engine.config['map_alignment']='bad'
    with pytest.raises(ValueError,match='map alignment'):
        validate_config(engine.config)


def test_all_visibility_conditions_use_same_noisy_yaw_and_keep_pairing(tmp_path):
    selected=fixture_scene(tmp_path)
    config=defaults()
    config.update(sample_count=1,sigma_xy_m=[20.],trials=1,map_alignment='prior_yaw')
    class Backend(SyntheticBackend):
        prepare_reference=AnyVisLocBackend.prepare_reference
        release_reference=AnyVisLocBackend.release_reference
    engine=Backend(config)
    engine.avl=SimpleNamespace(dumpRotateImage=lambda image,yaw:
        (image.copy(),cv2.getRotationMatrix2D((200,200),yaw,1.)))
    aligned=run_experiment(engine,selected,config,tmp_path/'aligned')
    assert len(aligned)==len(CONDITIONS) and set(aligned.map_alignment)=={'prior_yaw'}
    assert aligned.map_yaw_deg.nunique()==1 and abs(aligned.map_yaw_deg.iloc[0])>0
    assert set(aligned.map_yaw_source)=={'noisy_camera_prior'}
    baseline_config=dict(config,map_alignment='none')
    baseline=run_experiment(SyntheticBackend(baseline_config),selected,baseline_config,tmp_path/'baseline')
    for key in ['pair_id','prior_seed','smoke_seed','inference_seed','prior_xyz_enu']:
        if key in aligned:
            assert aligned[key].tolist()==baseline[key].tolist()


def test_old_rotated_raster_is_released_before_allocating_next(tmp_path):
    import weakref
    ref=load_reference(fixture_scene(tmp_path)[0]['reference_json'])
    engine=AnyVisLocBackend.__new__(AnyVisLocBackend)
    engine.config=dict(defaults(),map_alignment='prior_yaw')
    old=[]
    def rotate(image,yaw):
        if old:
            assert old[0]() is None, 'Old rotated raster still alive during replacement'
        return image.copy(),cv2.getRotationMatrix2D((200,200),yaw,1.)
    engine.avl=SimpleNamespace(dumpRotateImage=rotate)
    prior=Prior(np.array([0.,0.,100.]),NADIR,20.,5.,2.,5.)
    engine.prepare_reference(ref,prior)
    old.append(weakref.ref(engine._aligned_reference.bgr))
    changed=Prior(prior.xyz,Rotation.from_euler('z',-15,degrees=True).as_matrix()@NADIR,20.,5.,2.,5.)
    engine.prepare_reference(ref,changed)
    engine.release_reference()
    assert engine._aligned_reference is None and engine._aligned_key is None


@pytest.mark.parametrize('array', [np.zeros((0,2)),np.arange(200).reshape(10,20),
                                  np.arange(200).reshape(10,20)[:,::2]])
def test_streamed_array_hash_matches_existing_identity(array):
    import hashlib
    contiguous=np.ascontiguousarray(array)
    legacy=hashlib.sha256(str((contiguous.shape,contiguous.dtype)).encode()+contiguous.tobytes()).hexdigest()
    assert array_hash(array)==legacy


def test_map_hash_does_not_allocate_map_sized_byte_copies():
    import tracemalloc
    image=np.zeros((1024,2048,3),np.uint8)
    tracemalloc.start()
    try:
        array_hash(image)
        _,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < image.nbytes//2

"""CPU tests for control comparisons and rotated crop coordinate conversion."""
import cv2
import numpy as np

from .ablation import crop_transform, crop_points_to_local, compare_controls, observe_matcher, compact


def test_rotated_crop_transform_roundtrip_with_offsets():
    rotation=cv2.getRotationMatrix2D((200.,180.),37.,1.)
    info=dict(rotation_matrix=rotation,pre_crop=[127,283,527,643],final_crop=[41,23,301,283])
    transform=crop_transform(info)
    points=np.array([[200.,400.],[350.,450.],[420.,530.]])
    expected=cv2.transform((points-[127,283])[None,:,:],rotation)[0]-[41,23]
    actual=cv2.transform(points[None,:,:],transform)[0]
    np.testing.assert_allclose(actual,expected,atol=1e-10)
    restored=cv2.transform(actual[None,:,:],cv2.invertAffineTransform(transform))[0]
    np.testing.assert_allclose(restored,points,atol=1e-10)


def test_parity_rejects_pose_mismatch_even_when_error_magnitudes_match():
    first=dict(query_hash='q',patch_hash='p',query_points=np.ones((8,2)),patch_points=np.ones((8,2)),
               pixels=np.ones((8,2)),world=np.ones((8,3)))
    author=dict(pred_x=10.,pred_y=0.,pred_z=100.,num_inliers=8)
    adapted=dict(pred_xyz_enu=[-10.,0.,100.],inliers=8)
    check=compare_controls(author,adapted,first,first)
    assert not check['passed'] and check['pose_delta_m']==20.
    adapted['pred_xyz_enu']=[10.,0.,100.]
    assert compare_controls(author,adapted,first,first)['pose_verified']
    second=dict(first,world=np.zeros((8,3)))
    assert not compare_controls(author,adapted,first,second)['passed']


def test_two_failed_pnp_runs_do_not_prove_pose_parity():
    inputs=dict(query_hash='q',patch_hash='p')
    check=compare_controls(dict(pred_x=None,num_inliers=0),dict(pred_xyz_enu=None,inliers=0),inputs,inputs)
    assert check['passed'] and not check['pose_verified']


def test_replay_retains_input_hashes_without_calling_neural_model():
    def forbidden(*args,**kwargs): raise AssertionError('Replay called a model')
    points=(np.ones((8,2)),np.full((8,2),2.))
    first={}
    callback=observe_matcher(forbidden,first,replay=points)
    callback(np.zeros((12,13,3),np.uint8),np.zeros((15,16,3),np.uint8))
    assert first['query_shape']==[12,13,3] and first['patch_shape']==[15,16,3]
    np.testing.assert_array_equal(first['patch_points'],points[1])


def test_author_pnp_success_is_separate_from_accuracy_and_adapter_acceptance():
    row=compact('A','s','i','satellite',dict(pnp_success=True,pred_error_m=30.,num_inliers=50),author=True)
    assert row['pnp_valid'] and not row['correct_10m'] and row['accepted'] is None


def test_fixed_crop_points_restore_offset_and_resolution_before_dsm_indexing():
    info=dict(rotation_matrix=np.array([[0.,-1.,20.],[1.,0.,0.]],np.float32),
              pre_crop=[100,200,140,240],final_crop=[2,4,12,14],
              map_resolution=[2.,3.],map_origin_local=[-100.,-200.])
    # Raw [1,2] at half scale -> patch [2,4], rotated [4,8], pre-crop [8,16].
    x,y=crop_points_to_local(np.array([[1.,2.]],np.float32),info,.5)
    np.testing.assert_array_equal(x,np.array([116.],np.float32))
    np.testing.assert_array_equal(y,np.array([448.],np.float32))

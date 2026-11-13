"""CPU checks for the short author's visibility experiment; no neural scores."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .core import defaults, degrade
from .quick_visibility import CONDITIONS, pilot_order, run_frame, summarize, visibility_config


def test_pilot_is_balanced_and_extends_without_reordering():
    selection=[dict(scene_id=f'Scene_{s}',sample_id=f'frame_{i}') for s in range(4) for i in range(6)]
    order=pilot_order(selection,[f'Scene_{s}' for s in range(4)])
    assert order==pilot_order(selection[::-1],[f'Scene_{s}' for s in range(4)])
    assert pd.Series([r['scene_id'] for r in order[:8]]).value_counts().eq(2).all()
    assert len({(r['scene_id'],r['sample_id']) for r in order})==24
    assert len(order[:8])*len(CONDITIONS)==40 and len(order)*len(CONDITIONS)==120


def test_stress_levels_and_paired_smoke_are_deterministic():
    config=visibility_config(defaults())
    image=np.full((64,64,3),200,np.uint8)
    outputs={c:degrade(image,c,config,42) for c in CONDITIONS}
    assert outputs['fog_extreme'][1].mean()<outputs['fog_strong'][1].mean()
    assert outputs['smoke_extreme'][1].mean()<outputs['smoke_strong'][1].mean()
    np.testing.assert_array_equal(outputs['clear'][0],image)
    for condition in CONDITIONS:
        np.testing.assert_array_equal(outputs[condition][0],degrade(image,condition,config,42)[0])


def test_summary_counts_failed_pnp_and_pairs_each_frame(tmp_path):
    rows=[]
    for i in range(2):
        for condition in CONDITIONS:
            bad=i==0 and condition!='clear'
            rows.append(dict(pair_id=str(i),condition=condition,status='pnp_failed' if bad else 'pose',
                             valid_pnp=not bad,error_xy_m=None if bad else 1.,inliers=0 if bad else 100))
    overview,paired=summarize(pd.DataFrame(rows),tmp_path)
    assert overview.loc[overview.condition.eq('clear'),'accuracy_10m'].iloc[0]==1.
    assert overview.loc[overview.condition.ne('clear'),'accuracy_10m'].eq(.5).all()
    assert overview.loc[overview.condition.ne('clear'),'lost_vs_clear'].eq(1).all()
    assert overview.loc[overview.condition.ne('clear'),'mean_delta_inliers'].eq(-50).all()
    assert len(paired)==10


def test_author_path_recomputes_query_caches_gallery_and_uses_no_acceptance_gate(tmp_path):
    from .test_experiment import K
    class Tensor:
        def __init__(self, x): self.x=np.asarray(x); self.device='cpu'
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.x
        def to(self,device): return self
    extraction=[]
    def extract(size,cfg,methods,mids,block_size,ref,query,transform,model):
        extraction.append((len(mids),int(query[0,0,0])))
        return Tensor(np.ones((len(mids),2))),Tensor([1.,0.])
    def retrieval(image,query,true_pos,res,origin,transform,scratch,options,cfg,methods):
        gf,qf=extract_features(384,cfg,methods,np.array([[20,20],[40,40]]),[20,20],image,query,None,None)
        return [0,1],[0,20],[0,20],[.1,.8],20,20,1.,0.
    def matching(options,cfg,query,scale,k,ref,dsm,rows,cols,ph,pw,scratch,methods,*args,**kwargs):
        assert kwargs['truePos'] is None and options.strategy=='Topn_opt'
        for rank in [1,2]:
            run_pixel_match_anyvisloc(query,ref,methods,scratch,f'/top{rank}_ransac.jpg')
        # Deliberately fewer than our production min_inliers. Author still returns a pose.
        return [{'X':1.,'Y':0.,'Z':100.},{'X':50.,'Y':0.,'Z':100.}],[2,1],[0.,0.],[0.,0.]
    def choose(true_pos,positions,inliers):
        i=int(np.argmax(inliers))
        return positions[i],positions[i]['X'],[p['X'] for p in positions],True,i
    avl=SimpleNamespace(anyvisloc_sample_to_truepos=lambda sample:dict(x=0.,y=0.,z=100.,yaw=0.),
        build_anyvisloc_reference_view=lambda sample,mode:dict(map=np.zeros((80,80,3),np.uint8),
            dsm=np.zeros((80,80)),map_resolution=[1,1],map_origin_local=[0,0],
            dsm_resolution=[1,1],dsm_origin_local=[0,0]),
        dumpRotateImage=lambda image,yaw:(image,np.array([[1.,0.,0.],[0.,1.,0.]])),
        extract_features=extract,retrieval_all_anyvisloc=retrieval,Match2Pos_all_anyvisloc=matching,
        run_pixel_match_anyvisloc=lambda *a,**kw:(np.zeros((10,2)),np.zeros((10,2))),
        estimate_drone_pose_anyvisloc=lambda *a,**kw:None,pos2error_anyvisloc=choose)
    backend=SimpleNamespace(avl=avl,seed=lambda seed:None,synchronize=lambda:None,
                            upstream_config={},methods={})
    sample=dict(K=Tensor(K),dist=Tensor(np.zeros(5)))
    gallery={}
    for brightness in [100,50]:
        result=run_frame(backend,sample,np.full((80,80,3),brightness,np.uint8),
                         SimpleNamespace(strategy='Topn_opt'),{},42,tmp_path,gallery)
        assert result['valid_pnp'] and result['accurate_10m'] and result['inliers']==2
        assert result['selected_rank']==1 and result['raw_matches']==10
    assert extraction==[(2,100),(1,50)]
    assert result['gallery_cache_hit']

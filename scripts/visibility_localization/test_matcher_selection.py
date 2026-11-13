"""CPU integration checks for model dispatch, metadata, and the fixed 18-attempt run."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from .backend import AnyVisLocBackend
from .campaign import run_experiment, read_json
from .core import CONDITIONS, UPSTREAM_COMMIT, defaults, validate_config
from .data import select_diagnostic_samples
from .models import ROMA_WEIGHTS, download_roma_weights
from .test_experiment import fixture_scene, SyntheticBackend


@pytest.mark.parametrize('method', ['Roma', 'SP_LG', None])
def test_backend_initializes_actual_matcher_and_matching_identity(tmp_path, monkeypatch, method):
    source = tmp_path/'upstream'
    source.mkdir()
    for name in ['avl_utils.py','avl_data.py','run_avl.py','run_avl_match_loc.py']:
        (source/name).write_text('# fixture\n')
    (source/'config_selectable_matchers.yaml').write_text('DEVICE: cuda\nMATCHING_METHODS: [Roma]\n')
    checkpoint = tmp_path/'camp.pth'
    checkpoint.write_bytes(b'camp-fixture')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True)))
    monkeypatch.setitem(sys.modules, 'yaml', SimpleNamespace(safe_load=lambda text:
                        {'DEVICE':'cuda','MATCHING_METHODS':['Roma']}))
    monkeypatch.setattr(AnyVisLocBackend, 'seed', lambda self, seed:None)
    monkeypatch.setattr(AnyVisLocBackend, '_state_hash', lambda self, model:model.tag)
    from . import backend, models
    monkeypatch.setattr(backend.subprocess, 'check_output', lambda *a,**kw:UPSTREAM_COMMIT)
    downloads=[]
    def download(path):
        downloads.append(path)
        return {name:'checkpoint-hash' for name in ROMA_WEIGHTS}
    monkeypatch.setattr(models, 'download_roma_weights', download)
    calls=[]
    def retrieval(methods, config):
        calls.append(('retrieval',copy.deepcopy(config)))
        return dict(methods,retrieval_model=SimpleNamespace(tag='camp-state'))
    def matching(methods, options, config):
        calls.append(('matching',methods['matching_method'],options.pose_priori))
        model = (SimpleNamespace(tag='roma-state') if methods['matching_method']=='Roma' else
                 dict(extractor=SimpleNamespace(tag='sp-state'),matcher=SimpleNamespace(tag='lg-state')))
        return dict(methods,matching_model=model)
    monkeypatch.setitem(sys.modules,'avl_utils',SimpleNamespace(retrieval_init=retrieval,matching_init=matching))
    config=defaults()
    if method is None:
        config.pop('matching_method')
    else:
        config['matching_method']=method
    engine=AnyVisLocBackend(source,config,checkpoint)
    expected=method or 'SP_LG'
    assert calls[0][1]['MATCHING_METHODS']==[expected]
    assert calls[1]==('matching',expected,'p')
    assert engine.identity['matching_method']==expected
    assert engine.identity['camp_state_sha256']=='camp-state'
    if expected=='Roma':
        assert downloads==[source.resolve()]
        assert engine.identity['roma_state_sha256']=='roma-state'
        assert set(engine.identity['roma_checkpoint_sha256'])==set(ROMA_WEIGHTS)
        assert 'superpoint_state_sha256' not in engine.identity
        assert 'lightglue_state_sha256' not in engine.identity
    else:
        assert not downloads and 'roma_state_sha256' not in engine.identity
        assert engine.identity['superpoint_state_sha256']=='sp-state'
        assert engine.identity['lightglue_state_sha256']=='lg-state'


def test_roma_checkpoint_download_reuses_existing_official_files(tmp_path, monkeypatch):
    calls=[]
    def download(url,destination,progress):
        calls.append(url)
        Path(destination).write_bytes(url.encode())
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(hub=SimpleNamespace(download_url_to_file=download)))
    first=download_roma_weights(tmp_path)
    assert set(calls)==set(ROMA_WEIGHTS.values())
    assert download_roma_weights(tmp_path)==first and len(calls)==2
    assert all((tmp_path/name).is_file() for name in first)


def test_roma_download_retries_504_and_reuses_drive_cache(tmp_path, monkeypatch):
    from urllib.error import HTTPError
    from . import models
    calls=[]
    waits=[]
    def download(url,destination,progress):
        calls.append(url)
        if len(calls)==1:
            Path(destination).write_bytes(b'partial')
            raise HTTPError(url,504,'Gateway Time-out',None,None)
        Path(destination).write_bytes(url.encode())
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(hub=SimpleNamespace(download_url_to_file=download)))
    monkeypatch.setattr(models.time,'sleep',waits.append)
    cache=tmp_path/'drive-cache'
    first=download_roma_weights(tmp_path/'session1',cache_dir=cache)
    assert len(calls)==3 and waits==[2]
    second=download_roma_weights(tmp_path/'session2',cache_dir=cache)
    assert first==second and len(calls)==3
    assert not list(tmp_path.rglob('*.download'))


@pytest.mark.parametrize('code,expected_calls',[(504,4),(403,1)])
def test_roma_failed_download_never_publishes_partial_weights(tmp_path, monkeypatch, code, expected_calls):
    from urllib.error import HTTPError
    from . import models
    calls=[]
    def download(url,destination,progress):
        calls.append(url)
        Path(destination).write_bytes(b'incomplete')
        raise HTTPError(url,code,'failure',None,None)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(hub=SimpleNamespace(download_url_to_file=download)))
    monkeypatch.setattr(models.time,'sleep',lambda delay:None)
    with pytest.raises(RuntimeError,match='Cannot download roma_outdoor.pth'):
        download_roma_weights(tmp_path/'source',cache_dir=tmp_path/'cache')
    assert len(calls)==expected_calls
    assert not list(tmp_path.rglob('*.pth')) and not list(tmp_path.rglob('*.download'))


def test_fixed_selection_is_ordered_and_never_substitutes_missing_frames():
    selection=[dict(scene_id='Scene_08',sample_id='L08_0049'),dict(scene_id='Scene_05',sample_id='L05_0969')]
    wanted=[('Scene_05','L05_0969'),('Scene_08','L08_0049')]
    assert select_diagnostic_samples(selection,wanted)==selection[::-1]
    with pytest.raises(ValueError,match='missing'):
        select_diagnostic_samples(selection,wanted+[('Scene_11','L11_0545')])


def test_aerial_roma_records_exactly_18_attempts_with_truthful_metadata(tmp_path):
    identities=[('Scene_05','L05_0969'),('Scene_08','L08_0049'),('Scene_11','L11_0545')]
    selection=[]
    for index,(scene,sample_id) in enumerate(identities):
        item=fixture_scene(tmp_path/str(index))[0]
        old_scene=Path(item['path']).parent
        new_scene=old_scene.with_name(scene)
        if new_scene!=old_scene:
            old_scene.rename(new_scene)
        npz=new_scene/Path(item['path']).name
        npz.rename(new_scene/(sample_id+'.npz'))
        reference=new_scene/Path(item['reference_json']).name
        meta=read_json(reference)
        meta['modes']['aerial']=dict(meta['modes']['satellite'])
        reference.write_text(json.dumps(meta))
        selection.append(dict(scene_id=scene,sample_id=sample_id,path=str(new_scene/(sample_id+'.npz')),
                              reference_json=str(reference)))
    config=defaults()
    config.update(reference_mode='aerial',matching_method='Roma',sample_count=3,sigma_xy_m=[20.],trials=1)
    backend=SyntheticBackend(config)
    backend.identity['matching_method']='Roma'
    selected=select_diagnostic_samples(selection,identities)
    frame=run_experiment(backend,selected,config,tmp_path/'output')
    assert len(frame)==backend.calls==18
    assert set(frame.matching_method)=={'Roma'} and set(frame.reference_mode)=={'aerial'}
    assert frame.groupby(['scene_id','sample_id']).size().eq(len(CONDITIONS)).all()
    assert frame.groupby('condition').size().eq(3).all()
    for envelope in (tmp_path/'output'/'records').glob('*.json'):
        assert read_json(envelope)['result']['matching_method']=='Roma'


def test_unknown_or_mismatched_matcher_stops_before_inference(tmp_path):
    config=defaults()
    config['matching_method']='unknown'
    with pytest.raises(ValueError,match='matching method'):
        validate_config(config)
    config['matching_method']='Roma'
    backend=SyntheticBackend(config)
    backend.identity['matching_method']='SP_LG'
    with pytest.raises(ValueError,match='matcher/config mismatch'):
        run_experiment(backend,[],config,tmp_path/'output')

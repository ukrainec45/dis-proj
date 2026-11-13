"""Checks for larger frame coverage and accuracy/acceptance denominators."""
import pandas as pd
import pytest

from .analysis import summarize_batch
from .core import CONDITIONS
from .data import validate_balanced_selection


def test_24_unique_frames_cover_four_scenes_and_144_conditions():
    scenes=['Scene_05','Scene_08','Scene_11','Scene_21']
    selection=[dict(scene_id=scene,sample_id=f'frame_{i}') for scene in scenes for i in range(6)]
    coverage=validate_balanced_selection(selection,scenes,24)
    assert coverage.selected_frames.tolist()==[6,6,6,6]
    attempts=[(r['scene_id'],r['sample_id'],condition) for r in selection for condition in CONDITIONS]
    assert len(attempts)==len(set(attempts))==144
    with pytest.raises(ValueError,match='24 unique'):
        validate_balanced_selection(selection[:-1],scenes,24)
    duplicated=selection[:-1]+[selection[0]]
    with pytest.raises(ValueError,match='24 unique'):
        validate_balanced_selection(duplicated,scenes,24)
    unbalanced=[dict(r) for r in selection]
    unbalanced[-1].update(scene_id='Scene_05',sample_id='extra')
    with pytest.raises(ValueError,match='6 frames per scene'):
        validate_balanced_selection(unbalanced,scenes,24)


def test_selector_returns_reproducible_six_frames_per_scene(tmp_path, monkeypatch):
    from . import data
    from .core import defaults
    scenes=['Scene_05','Scene_08','Scene_11','Scene_21']
    for scene in scenes:
        directory=tmp_path/'dataset'/scene
        directory.mkdir(parents=True)
        prefix='L'+scene[-2:]
        (directory/(prefix+'_reference.json')).write_text('{}')
        for index in range(10):
            (directory/(prefix+f'_{index:04d}.npz')).touch()
    monkeypatch.setattr(data,'read_sample',lambda *args,**kwargs:{'nadir_deg':0.})
    config=defaults()
    config.update(scenes=scenes,sample_count=24)
    first,_=data.select_samples(tmp_path/'dataset',config,tmp_path/'selection-1')
    second,_=data.select_samples(tmp_path/'dataset',config,tmp_path/'selection-2')
    assert first==second
    assert data.validate_balanced_selection(first,scenes,24).selected_frames.eq(6).all()


def test_batch_overview_keeps_accurate_rejections_and_technical_failures_visible(tmp_path):
    rows=[]
    for i,(status,error,valid) in enumerate([
        ('correct',1.,True),('rejected',2.,True),('false_accept',30.,True),
        ('rejected',None,False),('technical_error',None,False)]):
        rows.append(dict(scene_id='Scene_05',sample_id=str(i),reference_mode='aerial',
                         matching_method='Roma',sigma_xy_m=20.,profile='main',retrieval_mode='prior',
                         condition='clear',status=status,error_xy_m=error,valid_pnp=valid,
                         accepted=status in ('correct','false_accept'),
                         rejection_reason='high_reprojection' if status=='rejected' else ''))
    overview,reasons=summarize_batch(pd.DataFrame(rows),tmp_path)
    row=overview.iloc[0]
    assert row.n_attempted==row.n_frames==5
    assert row.n_technical==1 and row.n_pnp==3
    assert row.n_within_threshold==2 and row.n_rejected_within_threshold==1
    assert row.n_correct==row.n_false_accept==1 and row.n_rejected==2
    assert row.correct_fraction_all==.2 and row.within_threshold_fraction_all==.4
    assert row.median_accepted_error_m==15.5
    assert reasons.at[0,'attempts']==2

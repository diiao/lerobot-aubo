from tests.bamboo_sorting.paths import REPO_ROOT
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_shadow(monkeypatch):
    directory=REPO_ROOT/'examples/phone_to_auboi10'
    monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('target_shadow_test',directory/'run_joint_target_shadow.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_plan_and_missing_onsite_flag_never_connect(monkeypatch,tmp_path):
    m=load_shadow(monkeypatch)
    def fail(*a,**kw):raise AssertionError('hardware/model accessed')
    monkeypatch.setattr(m,'ReadOnlyStation',fail);monkeypatch.setattr(m,'InferencePipe',fail)
    assert m.main([])==0
    with pytest.raises(SystemExit):m.main(['--stage','capture','--output',str(tmp_path/'capture')])


def test_capture_only_reads_and_always_closes(monkeypatch,tmp_path):
    m=load_shadow(monkeypatch);calls=[]
    class Station:
        def connect(self):calls.append('connect_read_only')
        def observe(self,**kw):
            assert kw==dict(require_stationary=True,max_camera_age_ms=100)
            return [0]*7,{k:np.zeros((480,640,3),np.uint8) for k in ('global_rgb','grasp_rgb')},[10,10,10]
        def close(self):calls.append('close')
    monkeypatch.setattr(m,'ReadOnlyStation',Station);monkeypatch.setattr(m.time,'perf_counter',lambda:10.01)
    out=tmp_path/'capture'
    assert m.main(['--stage','capture','--onsite-confirmed','--output',str(out)])==0
    assert calls==['connect_read_only','close']
    snapshot=json.loads((out/'snapshot.json').read_text())
    assert not snapshot['robot_motion_sent'] and not snapshot['gripper_command_sent']
    assert snapshot['snapshot_id'] in (out/'review.html').read_text()


def test_mask_review_requires_snapshot_identity_and_nonoverlap(monkeypatch):
    m=load_shadow(monkeypatch)
    labels=dict(schema='joint_target_shadow_selection',snapshot_id='id',human_reviewed=True,
                targets=[dict(target_id='A',polygons=[[[10,10],[20,10],[20,20]]]),
                         dict(target_id='B',polygons=[[[40,40],[50,40],[50,50]]])])
    assert len(m.reviewed_masks(dict(snapshot_id='id'),labels))==2
    with pytest.raises(ValueError,match='bound'):m.reviewed_masks(dict(snapshot_id='other'),labels)
    labels['targets'][1]['polygons']=labels['targets'][0]['polygons']
    with pytest.raises(ValueError,match='overlap'):m.reviewed_masks(dict(snapshot_id='id'),labels)


@pytest.mark.parametrize('actual_mode,status,accepted',[
    ('paired_foreground','separated',True),
    (None,'matched',False),
    ('highest_score','matched',False),
    ('paired_foreground','matched',False),
])
@pytest.mark.parametrize('requested_mode', ['paired_foreground', 'multi_foreground'])
def test_paired_pipe_rejects_old_service_fallback(monkeypatch,tmp_path,actual_mode,status,accepted,requested_mode):
    load_shadow(monkeypatch)
    if actual_mode == 'paired_foreground':
        actual_mode = requested_mode
    import run_joint_trial as trial
    from lerobot.bamboo_sorting.policy.joint_target_policy import target_policy_contract_record
    from lerobot.bamboo_sorting.runtime.joint_target_runtime import encode_mask
    class Process:
        def __init__(self,*a,**kw):self.stdin=io.StringIO()
        def wait(self,timeout):pass
    monkeypatch.setattr(trial.subprocess,'Popen',Process)
    metadata=dict(frame_id='frame',target_id='B',phase='approach')
    mask=np.zeros((480,640),bool);mask[200:300,400:420]=True
    chunk=[[0.]*7 for _ in range(50)]
    ready=dict(ready=True,checkpoint_sha256='hash',contract=target_policy_contract_record(),
               action_chunk_available=True,target_live=True,segmentation_checkpoint='seg')
    response=dict(id=0,checkpoint_sha256='hash',shape=[50,7],action=chunk[0],raw_action=chunk[0],
        action_chunk=chunk,raw_action_chunk=chunk,target_input=metadata,
        target_evidence=dict(**metadata,seed_snapshot_id='s',human_reviewed=False,
            source='same_frame_segmentation',mask_png=encode_mask(mask),association={'status':status},
            target_selection_mode=actual_mode))
    messages=iter([ready,response]);monkeypatch.setattr(trial.InferencePipe,'receive',lambda self,timeout:next(messages))
    pipe=trial.InferencePipe(tmp_path,target_conditioned=True,expected_sha256='hash',return_action_chunk=True,
        live_target_seed=dict(snapshot_id='s',target_selection_mode=requested_mode),segmentation_checkpoint='seg')
    try:
        images={k:np.zeros((480,640,3),np.uint8) for k in ('global_rgb','grasp_rgb')}
        if accepted:
            assert pipe.predict(0,[0]*7,images,target_input=metadata)['target_evidence']['association']['status']=='separated'
        else:
            with pytest.raises(ValueError):pipe.predict(0,[0]*7,images,target_input=metadata)
    finally:pipe.close()


@pytest.mark.parametrize('overlap,status',[(None,'matched'),(0,'matched'),(85,'matched'),(437,'matched'),(85,'lost')])
def test_target_pipe_negotiates_and_sends_contract(monkeypatch,tmp_path,overlap,status):
    m=load_shadow(monkeypatch)
    import run_joint_trial as trial
    from lerobot.bamboo_sorting.policy.joint_target_policy import target_policy_contract_record
    contract=target_policy_contract_record();launch=[]
    class Process:
        def __init__(self,command,**kw):launch.extend(command);self.stdin=io.StringIO()
        def wait(self,timeout):pass
    monkeypatch.setattr(trial.subprocess,'Popen',Process)
    chunk=[[0.]*7 for _ in range(50)]
    ready=dict(ready=True,checkpoint_sha256='hash',contract=contract,action_chunk_available=True)
    response=dict(id=0,checkpoint_sha256='hash',shape=[50,7],action=chunk[0],raw_action=chunk[0],
                  action_chunk=chunk,raw_action_chunk=chunk)
    metadata=dict(frame_id='f',target_id='a',phase='approach')
    options={}
    if overlap is not None:
        from lerobot.bamboo_sorting.runtime.joint_target_runtime import encode_mask
        mask=np.zeros((480,640),bool);mask[200:300,400:420]=True
        options=dict(live_target_seed={'snapshot_id':'snapshot'},segmentation_checkpoint='segmenter')
        ready.update(target_live=True,segmentation_checkpoint='segmenter')
        response.update(target_input=metadata,target_evidence=dict(**metadata,seed_snapshot_id='snapshot',
            human_reviewed=False,source='same_frame_segmentation',mask_png=encode_mask(mask),
            association={'status':status},other_reviewed_target_overlap_px=overlap))
    messages=iter([ready,response])
    monkeypatch.setattr(trial.InferencePipe,'receive',lambda self,timeout:next(messages))
    pipe=trial.InferencePipe(tmp_path,target_conditioned=True,expected_sha256='hash',return_action_chunk=True,**options)
    try:
        assert '--target-conditioned' in launch[-1]
        with pytest.raises(ValueError,match='metadata'):
            pipe.predict(0,[0]*7,dict(global_rgb=np.zeros((480,640,3),np.uint8)))
        keys=('global_rgb','grasp_rgb') if overlap is not None else contract['image_keys']
        images={k:np.zeros((480,640,3),np.uint8) for k in keys}
        if status!='matched':
            with pytest.raises(ValueError,match='fresh target association'):
                pipe.predict(0,[0]*7,images,codec='rgb_zlib',target_input=metadata)
            return
        result=pipe.predict(0,[0]*7,images,codec='rgb_zlib',target_input=metadata)
        if overlap is not None:assert result['target_evidence']['other_reviewed_target_overlap_px']==overlap
        packet=json.loads(pipe.process.stdin.getvalue())
        assert packet['target_contract']==contract and packet['target_input']['target_id']=='a'
    finally:pipe.close()

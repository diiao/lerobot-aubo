import json
import math
import numpy as np
from pathlib import Path
import pytest
from lerobot.bamboo_sorting.joint_smooth import *
from lerobot.bamboo_sorting.joint_trial import START_DEG


def fk(q):return [.2+q[0]*.01,-.5,.3]


def test_trajectory_bounded_with_target_reversal_and_convergence():
    t=SmoothTrajectory([0]*6);last_v=[0]*6
    for i in range(400):
        target=[2 if i<100 else -1]*6
        before=t.q.copy();q,tcp=t.advance(target,fk)
        assert max(abs(v) for v in t.v)<=MAX_SPEED
        assert max(abs(a-b)/DT for a,b in zip(t.v,last_v))<=MAX_ACCEL+1e-8
        assert math.dist(fk(before),tcp)<=MAX_TCP_SPEED*DT+1e-7
        last_v=t.v.copy()
    assert t.q==pytest.approx([-1]*6,abs=.001)


def test_prediction_distance_separate_from_controller_limits():
    state=[-57.65431566565926,-.6645925301704693,117.42801944969176,29.636383572763556,90.66313814689934,-182.15161343663397,0]
    action=[-56.643585205078125,-.3033311367034912,117.9039306640625,29.727054595947266,90.61083221435547,-181.3079376220703,0]
    args=dict(state=state,tcp=[.18,-.65,.16],target_tcp=[.1912,-.65,.16],lower=[-360]*6,upper=[360]*6,timestamps=[1,1,1],now=1.2)
    assert validate_prediction(action,**args)==action
    assert validate_prediction(action,**{**args,'now':1.362})==action
    with pytest.raises(ValueError,match='expired'):validate_prediction(action,**{**args,'now':1.401})
    with pytest.raises(ValueError,match='workspace'):validate_prediction(action,**{**args,'target_tcp':[.19,-.65,-.001]})
    farther=[state[0]+10,*action[1:]]
    assert validate_prediction(farther,**{**args,'target_tcp':[.3,-.65,.16]})==farther
    with pytest.raises(ValueError,match='controller joint limits'):
        validate_prediction([361,*action[1:]],**args)


def test_both_orders_explicit_prediction_age_budget():
    state=[*START_DEG,0]
    args=dict(action=state,state=state,tcp=[.2,-.5,.3],target_tcp=[.2,-.5,.3],
              lower=[-360]*6,upper=[360]*6,timestamps=[1,1,1],now=1.44)
    with pytest.raises(ValueError,match='limit=400 ms'):validate_prediction(**args)
    assert validate_prediction(**args,max_prediction_age=.5)==state
    assert validate_prediction(**{**args,'now':1.499},max_prediction_age=.5)==state
    with pytest.raises(ValueError,match='limit=500 ms'):
        validate_prediction(**{**args,'now':1.501},max_prediction_age=.5)
    assert validate_prediction(**{**args,'now':1.5133},max_prediction_age=.75)==state
    with pytest.raises(ValueError,match='limit=750 ms'):
        validate_prediction(**{**args,'now':1.751},max_prediction_age=.75)
    assert validate_prediction(**{**args,'now':1.7527},max_prediction_age=1.5)==state
    assert validate_prediction(**{**args,'now':2.499},max_prediction_age=1.5)==state
    with pytest.raises(ValueError,match='limit=1500 ms'):
        validate_prediction(**{**args,'now':2.501},max_prediction_age=1.5)


def chunk_response(close_at=None):
    chunk=[[*START_DEG,0.] for _ in range(50)]
    for i,row in enumerate(chunk):row[5]+=i*.1
    if close_at is not None:chunk[close_at][-1]=100.
    return {'shape':[50,7],'action':chunk[0].copy(),'raw_action':chunk[0].copy(),
        'action_chunk':[r.copy() for r in chunk],'raw_action_chunk':[r.copy() for r in chunk]}


def test_camera_pause_tolerance_preserves_skew_and_total_age_bounds():
    state=[*START_DEG,0]
    args=dict(action=state,state=state,tcp=[.2,-.5,.3],target_tcp=[.2,-.5,.3],
              lower=[-360]*6,upper=[360]*6,timestamps=[1.,1.2527,1.26],now=1.41,max_prediction_age=.75)
    with pytest.raises(ValueError,match='camera skew'):validate_prediction(**args)
    assert validate_prediction(**args,max_camera_skew=.3)==state
    with pytest.raises(ValueError,match='camera skew'):
        validate_prediction(**{**args,'timestamps':[1.,1.301,1.31]},max_camera_skew=.3)
    with pytest.raises(ValueError,match='expired'):
        validate_prediction(**{**args,'now':1.751},max_camera_skew=.3)


def test_age_aligned_selection_keeps_joint_and_closure_row_together():
    chunk=validated_action_chunk(chunk_response(close_at=3))
    action,selection=select_approach_action(chunk,[10.,10.01,10.02],10.22)
    assert selection['nominal_index']==5 and selection['selected_index']==3
    assert selection['closure_boundary'] and action==chunk[3]
    assert action[-1]==100 and action[5]==pytest.approx(START_DEG[5]+.3)
    # Ordinary approach uses actual observation age, not time since receipt.
    chunk=validated_action_chunk(chunk_response())
    assert select_approach_action(chunk,[10.,10.01,10.02],10.22)[0]==chunk[5]
    with pytest.raises(ValueError,match='clock'):
        select_approach_action(chunk,[10.,10.01,10.3],10.22)
    with pytest.raises(ValueError,match='exhausted'):
        select_approach_action(chunk,[10.]*3,12.1)
    # A later selected row must still pass the existing target validation.
    chunk[5][0]=361.
    selected,_=select_approach_action(chunk,[10.]*3,10.22)
    with pytest.raises(ValueError,match='controller joint limits'):
        validate_prediction(selected,[*START_DEG,0],[.2,-.5,.3],[.2,-.5,.3],[-360]*6,[360]*6,[10.]*3,10.22)


@pytest.mark.parametrize('corruption',['missing','nan','decoded','first'])
def test_inconsistent_full_chunks_are_rejected(corruption):
    response=chunk_response()
    if corruption=='missing':response.pop('raw_action_chunk')
    elif corruption=='nan':response['raw_action_chunk'][12][3]=float('nan')
    elif corruption=='decoded':response['action_chunk'][4][-1]=100.
    else:response['action'][5]+=1.
    with pytest.raises(ValueError):validated_action_chunk(response)


def test_transport_age_selection_preserves_release_row_and_existing_checks():
    chunk=[[*START_DEG,100.] for _ in range(50)]
    for i,row in enumerate(chunk):row[0]+=.01*i;row[5]+=.02*i
    selected,info=select_transport_action(chunk,[10.,10.01,10.02],10.42)
    assert selected==chunk[10] and info['selected_index']==10 and not info['release_boundary']
    chunk[3][-1]=0.
    # A later re-close or invalid goal must not skip the first release.
    chunk[4][0]=361.
    selected,info=select_transport_action(chunk,[10.]*3,10.42)
    assert selected==chunk[3] and info['selected_index']==3 and info['release_boundary']
    assert selected[0]==START_DEG[0]+.03 and selected[5]==START_DEG[5]+.06
    args=dict(state=[*START_DEG,100],tcp=[.2,-.5,.3],target_tcp=[.2,-.5,.3],
              lower=[-360]*6,upper=[360]*6,timestamps=[10.]*3,now=10.42,max_prediction_age=.5)
    assert validate_prediction(selected,**args)==selected
    with pytest.raises(ValueError,match='expired'):
        validate_prediction(selected,**{**args,'now':10.6})
    chunk[3][0]=361.
    with pytest.raises(ValueError,match='controller joint limits'):
        validate_prediction(select_transport_action(chunk,[10.]*3,10.42)[0],**args)
    with pytest.raises(ValueError,match='exhausted'):select_transport_action(chunk,[10.]*3,12.1)
    with pytest.raises(ValueError,match='clock'):select_transport_action(chunk,[11.]*3,10.)


def load_entry(monkeypatch):
    import importlib.util
    root=Path(__file__).resolve().parents[2]/'examples/phone_to_auboi10'
    monkeypatch.syspath_prepend(str(root))
    spec=importlib.util.spec_from_file_location('smooth_entry',root/'run_joint_smooth.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_age_aligned_approach_plan_is_explicit_and_hardware_free(monkeypatch,capsys):
    m=load_entry(monkeypatch)
    def fail(*a,**kw):raise AssertionError('hardware touched')
    monkeypatch.setattr(m,'ReadOnlyStation',fail);monkeypatch.setattr(m,'InferencePipe',fail)
    with pytest.raises(SystemExit):m.main(['--approach-age-aligned'])
    assert m.main(['--model','mixed-both-orders','--approach-age-aligned'])==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['suction_writes'] and plan['requested_suction_cycles']==2
    assert plan['action_selection']=='observation_age_while_open_first_step_while_closed'
    assert plan['pending_suction_policy']=='continue_inference_while_open'
    assert plan['max_prediction_age_s']==.5 and plan['watchdog_s']==.6
    assert plan['joint_speed_deg_s']==5 and plan['tcp_speed_m_s']==.05


def test_target_plan_requires_seed_and_one_cycle_without_hardware(monkeypatch,capsys):
    m=load_entry(monkeypatch)
    import joint_target_live_setup
    def forbidden(*args,**kwargs):raise AssertionError('hardware/model accessed')
    monkeypatch.setattr(m,'ReadOnlyStation',forbidden);monkeypatch.setattr(m,'InferencePipe',forbidden)
    monkeypatch.setattr(joint_target_live_setup,'prepare_live_setup',lambda *a:(
        dict(snapshot_id='seed'),dict(expected_sha256='target-model',segmentation_checkpoint='segmentation')))
    with pytest.raises(SystemExit):m.main(['--model','target-pair'])
    args=['--model','target-pair','--target-snapshot','snapshot','--target-selections','labels','--target-id','A']
    with pytest.raises(SystemExit):m.main([*args,'--cycles','2'])
    assert m.main(args)==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['target_id']=='A' and plan['requested_suction_cycles']==1
    assert plan['action_selection']=='observation_age_while_open_first_step_while_closed'
    assert plan['max_seconds']==120 and plan['max_prediction_age_s']==.5
    assert plan['watchdog_s']==.6 and plan['joint_speed_deg_s']==5 and plan['tcp_speed_m_s']==.05
    assert m.main([*args,'--latency-tolerant'])==0
    relaxed=json.loads(capsys.readouterr().out)
    assert relaxed['max_prediction_age_s']==1.5 and relaxed['watchdog_s']==1.5
    assert relaxed['inference_timeout_s']==5.
    changed={key for key in plan if plan[key]!=relaxed[key]}
    assert changed=={'latency_tolerant','max_prediction_age_s','watchdog_s','inference_timeout_s'}


def test_paired_target_plan_selects_isolated_service_without_hardware(monkeypatch,capsys,tmp_path):
    m=load_entry(monkeypatch)
    import joint_target_live_setup
    from lerobot.bamboo_sorting.joint_target_runtime import decode_mask
    def forbidden(*args,**kwargs):raise AssertionError('hardware/model accessed')
    monkeypatch.setattr(m,'ReadOnlyStation',forbidden);monkeypatch.setattr(m,'InferencePipe',forbidden)
    (tmp_path/'snapshot.json').write_text(json.dumps(dict(schema='joint_target_shadow_snapshot',snapshot_id='s',execution_eligible=False)))
    labels=tmp_path/'labels.json'
    labels.write_text(json.dumps(dict(schema='joint_target_shadow_selection',snapshot_id='s',human_reviewed=True,
        targets=[dict(target_id='A',polygons=[[[490,200],[502,200],[502,400],[490,400]]]),
                 dict(target_id='B',polygons=[[[400,200],[412,200],[412,400],[400,400]]])])))
    with pytest.raises(SystemExit):m.main(['--paired-target-separation'])
    with pytest.raises(SystemExit):m.main(['--transport-age-aligned'])
    args=['--model','target-pair','--target-snapshot',str(tmp_path),'--target-selections',str(labels),
          '--target-id','B','--latency-tolerant','--paired-target-separation']
    assert m.main(args)==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['target_selection_mode']=='paired_foreground' and plan['target_id']=='B'
    assert plan['max_prediction_age_s']==1.5 and plan['watchdog_s']==1.5
    assert plan['requested_suction_cycles']==1 and plan['max_seconds']==120
    assert plan['transport_age_aligned'] is False
    assert m.main([*args,'--transport-age-aligned'])==0
    transport_plan=json.loads(capsys.readouterr().out)
    assert transport_plan['transport_age_aligned'] is True
    assert transport_plan['action_selection']=='observation_age_open_and_transport'
    assert {k:v for k,v in transport_plan.items() if k not in ('action_selection','transport_age_aligned')}=={
        k:v for k,v in plan.items() if k not in ('action_selection','transport_age_aligned')}
    seed,options=joint_target_live_setup.prepare_live_setup(tmp_path,labels,'B',paired_separation=True)
    targets={t['target_id']:decode_mask(t['mask_png']) for t in seed['paired_targets']}
    np.testing.assert_array_equal(decode_mask(seed['mask_png']),targets['B'])
    np.testing.assert_array_equal(decode_mask(seed['excluded_mask_png']),targets['A'])
    assert '/paired_foreground/' in options['inference_script']
    assert options['pythonpath'].split(':')[0].endswith('/paired_foreground/src')
    default_seed,default_options=joint_target_live_setup.prepare_live_setup(tmp_path,labels,'B')
    assert 'paired_targets' not in default_seed
    assert '/paired_foreground/' not in default_options['inference_script']


@pytest.mark.parametrize('count', [1, 2, 3])
@pytest.mark.parametrize('model', ['target-pair', 'target-stack'])
def test_multi_target_plan_uses_explicit_service_for_remaining_pile(monkeypatch, capsys, tmp_path, count, model):
    m = load_entry(monkeypatch)
    import joint_target_live_setup
    from lerobot.bamboo_sorting.joint_target_runtime import FreshTargetInputs
    def forbidden(*a, **kw): raise AssertionError('hardware/model accessed')
    monkeypatch.setattr(m, 'ReadOnlyStation', forbidden)
    monkeypatch.setattr(m, 'InferencePipe', forbidden)
    (tmp_path/'snapshot.json').write_text(json.dumps(dict(schema='joint_target_shadow_snapshot',
                                                         snapshot_id='stack', execution_eligible=False)))
    targets = [dict(target_id=f's-{i+1:03}', polygons=[[[100+i*100,100], [112+i*100,100],
                                                     [112+i*100,400], [100+i*100,400]]]) for i in range(count)]
    labels = tmp_path/'labels.json'
    labels.write_text(json.dumps(dict(schema='joint_target_shadow_selection', snapshot_id='stack',
                                     human_reviewed=True, targets=targets)))
    seed, options = joint_target_live_setup.prepare_live_setup(tmp_path, labels, 's-001', multi_separation=True, model=model)
    if model == 'target-stack':
        assert options['checkpoint'] == joint_target_live_setup.STACK_CHECKPOINT
        assert options['expected_sha256'] == joint_target_live_setup.STACK_MODEL_SHA
        assert options['inference_script'] == joint_target_live_setup.STACK_SERVICE_ROOT + '/joint_inference_stdio.py'
        assert options['remote_root'] == joint_target_live_setup.STACK_REMOTE_ROOT
    else:
        assert '/multi_foreground/' in options['inference_script']
    FreshTargetInputs(seed, forbidden)  # No segmenter call at construction, including last remaining strip.
    args = ['--model', model, '--target-snapshot', str(tmp_path), '--target-selections', str(labels),
            '--target-id', 's-001']
    if model == 'target-pair':
        args.append('--multi-target-separation')
    assert m.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['target_selection_mode'] == 'multi_foreground' and plan['requested_suction_cycles'] == 1
    assert plan['checkpoint'] == options['checkpoint']
    if model == 'target-stack':
        old_args = [*args]
        old_args[1] = 'target-pair'
        assert m.main([*old_args, '--multi-target-separation']) == 0
        old_plan = json.loads(capsys.readouterr().out)
        assert {k for k in plan if plan[k] != old_plan.get(k)} == {
            'model', 'checkpoint', 'checkpoint_sha256', 'inference_script',
            'empty_target_reobservations_per_target','empty_target_policy'}
        with pytest.raises(SystemExit):
            m.main([*args, '--cycles', '3'])
    assert m.main([*args, '--latency-tolerant', '--max-seconds', '600']) == 0
    tolerant = json.loads(capsys.readouterr().out)
    expected_changes = {'latency_tolerant': True, 'max_prediction_age_s': 1.5,
                        'watchdog_s': 1.5, 'inference_timeout_s': 5., 'max_seconds': 600.}
    if model == 'target-stack':
        expected_changes.update(max_camera_age_ms=400, max_camera_skew_s=.3,
                                max_control_gap_s=.7, max_state_age_s=.25)
    assert {k: v for k, v in tolerant.items() if v != plan[k]} == expected_changes
    if model == 'target-stack':
        sequence = [t['target_id'] for t in targets]
        sequence_args = args[:-2] + ['--target-sequence', *sequence]
        assert m.main(sequence_args) == 0
        sequence_plan = json.loads(capsys.readouterr().out)
        assert sequence_plan['requested_suction_cycles'] == count
        assert sequence_plan['automatic_home'] == (count > 1)
        assert sequence_plan['remaining_targets_by_pick'] == [sequence[i:] for i in range(count)]
        for i, target in enumerate(sequence):
            filtered, _ = joint_target_live_setup.prepare_live_setup(
                tmp_path, labels, target, multi_separation=True, model=model, remaining_target_ids=sequence[i:])
            FreshTargetInputs(filtered, forbidden)
            assert [t['target_id'] for t in filtered['visible_targets']] == sequence[i:]
        assert json.loads(labels.read_text())['targets'] == targets
        with pytest.raises(SystemExit):m.main(args[:-2] + ['--target-sequence', sequence[0], sequence[0]])
        with pytest.raises(ValueError):m.main(args[:-2] + ['--target-sequence', 'missing'])
    with pytest.raises(SystemExit):
        m.main([*args, '--paired-target-separation'])


@pytest.mark.parametrize('model,expected_cycles',[('legacy-single',1),('mixed-double',2),('mixed-both-orders',2)])
def test_plan_does_not_connect(monkeypatch,capsys,model,expected_cycles):
    m=load_entry(monkeypatch)
    def fail(*a,**k):raise AssertionError('device touched')
    monkeypatch.setattr(m,'ReadOnlyStation',fail);monkeypatch.setattr(m,'InferencePipe',fail)
    assert m.main(['--model',model])==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['requested_suction_cycles']==expected_cycles
    assert plan['max_seconds']==(360 if expected_cycles==2 else 120)
    assert plan['servo_command_time_s']==(.08 if expected_cycles==2 else DT)
    assert plan['max_camera_age_ms']==(250 if expected_cycles==2 else 100)
    assert plan['max_prediction_age_s']==(.5 if model=='mixed-both-orders' else .4)
    assert plan['watchdog_s']==.6
    assert plan['inference_timeout_s']==2.
    assert plan['checkpoint_sha256']=={'legacy-single':m.MODEL_SHA,
        'mixed-double':m.MIXED_MODEL_SHA,'mixed-both-orders':m.BOTH_ORDERS_MODEL_SHA}[model]
    assert plan['startup_release_if_unset']=='operator_confirmed'


def test_latency_tolerance_changes_only_input_and_prediction_timing(monkeypatch,capsys):
    m=load_entry(monkeypatch)
    def fail(*a,**kw):raise AssertionError('hardware touched')
    monkeypatch.setattr(m,'ReadOnlyStation',fail);monkeypatch.setattr(m,'InferencePipe',fail)
    argv=['--model','mixed-both-orders','--approach-age-aligned','--cycles','1']
    assert m.main(argv)==0
    standard=json.loads(capsys.readouterr().out)
    assert m.main(argv+['--latency-tolerant'])==0
    tolerant=json.loads(capsys.readouterr().out)
    assert {k:(standard[k],v) for k,v in tolerant.items() if standard[k]!=v}=={
        'latency_tolerant':(False,True),'inference_timeout_s':(2.,5.),
        'max_prediction_age_s':(.5,.75),'watchdog_s':(.6,1.),
        'max_camera_age_ms':(250,400),'max_camera_skew_s':(.1,.3)}
    assert m.main(argv+['--latency-tolerant','--inference-timeout-s','3'])==0
    assert json.loads(capsys.readouterr().out)['inference_timeout_s']==3.
    with pytest.raises(SystemExit):m.main(['--latency-tolerant'])


def test_explicit_cycle_count_preserves_mixed_control_limits(monkeypatch,capsys):
    m=load_entry(monkeypatch)
    def fail(*a,**kw):raise AssertionError('device touched')
    monkeypatch.setattr(m,'ReadOnlyStation',fail);monkeypatch.setattr(m,'InferencePipe',fail)
    for invalid in ('0','-1','1.5'):
        with pytest.raises(SystemExit):m.main(['--cycles',invalid])
    for invalid in ('0','-1','nan','inf'):
        with pytest.raises(SystemExit):m.main(['--inference-timeout-s',invalid])
        with pytest.raises(SystemExit):m.main(['--max-seconds',invalid])
    assert m.main(['--model','mixed-both-orders','--approach-age-aligned','--cycles','1','--inference-timeout-s','5'])==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['requested_suction_cycles']==1
    assert plan['max_seconds']==360 and plan['servo_command_time_s']==.08
    assert plan['max_camera_age_ms']==250 and plan['max_prediction_age_s']==.5
    assert plan['inference_timeout_s']==5 and plan['watchdog_s']==.6


@pytest.mark.parametrize('answer,release_ok,expected_return',[
    ('q',True,0),('',True,0),('',False,1),
])
def test_unset_initial_suction_requires_confirmed_release(tmp_path,monkeypatch,answer,release_ok,expected_return):
    from lerobot.robots.aubo_i10 import aubo_i10
    m=load_entry(monkeypatch);events=[]
    monkeypatch.setattr(m.sys.stdin,'isatty',lambda:True)
    monkeypatch.setattr(m,'read_task',lambda:'Pick one strip and place it in the collection area.')
    replies=iter([answer,'q'])
    monkeypatch.setattr('builtins.input',lambda prompt:next(replies))
    class Station:
        def __init__(self):
            self.iface=self;self.state=self;self.algorithm=self;self.io_readback=self
            self.pins=(False,False)
            station[0]=self
        def connect(self):pass
        def close(self):events.append('station_closed')
        def current(self,**kw):
            if self.pins==(False,False):raise ValueError('unknown suction output pair: (False, False)')
            return [*START_DEG,0],[.2,-.5,.3],1.
        def getMotionControl(self):return self
        def isServoModeEnabled(self):return False
        def isSteady(self):return True
        def forwardKinematics(self,q):return [.2,-.5,.3,0,0,0],0
    class Pipe:
        def __init__(self,*args,**kwargs):pass
        def close(self):events.append('pipe_closed')
        def predict(self,*args):raise AssertionError('inference before start')
    class Suction:
        last_gripper_command_trace=None
        def __init__(self,*args):pass
        def suction_release(self):
            events.append('release')
            self.last_gripper_command_trace={'do_api_success':release_ok}
            if release_ok:station[0].pins=(False,True)
            return release_ok
    station=[None]
    monkeypatch.setattr(m,'ReadOnlyStation',Station)
    monkeypatch.setattr(m,'InferencePipe',Pipe)
    monkeypatch.setattr(aubo_i10,'AuboI10Robot',Suction)
    out=tmp_path/'run'
    assert m.main(['--execute','--model','mixed-double','--output',str(out)])==expected_return
    result=json.loads((out/'complete.json').read_text())
    assert result['sent_frames']==result['predictions']==0
    assert events.count('release')==(0 if answer else 1)
    assert events[-2:]==['station_closed','pipe_closed']
    if answer:
        assert result['reason']=='cancelled' and result['failure'] is None
        assert not (out/'final_state.json').exists()
    elif release_ok:
        assert result['reason']=='cancelled' and result['failure'] is None
        assert (out/'final_state.json').exists()
    else:
        assert 'initial suction release failed' in result['failure']
    if not answer:
        trace=json.loads((out/'trace.jsonl').read_text().strip())
        assert trace['initial_suction_release']['do_api_success']==release_ok


@pytest.mark.parametrize('model,expected_cycles,inference_error,slow_fk,approach',[
    ('legacy-single',1,False,False,False),('legacy-single',1,True,False,False),('legacy-single',1,'hung',False,False),
    ('mixed-double',2,False,False,False),('mixed-double',2,True,False,False),('mixed-double',2,'hung',False,False),
    ('mixed-double',2,False,True,False),
    ('mixed-both-orders',2,False,False,False),('mixed-both-orders',2,True,False,False),
    ('mixed-both-orders',2,False,True,False),
    ('mixed-both-orders',2,False,False,2),('mixed-both-orders',2,False,False,7),
    ('mixed-both-orders',2,True,False,7),('mixed-both-orders',2,'hung',False,7),
    ('mixed-both-orders',2,False,True,7),
    ('mixed-both-orders',2,False,False,'replan'),
    ('mixed-both-orders',2,False,False,'replan_at_receipt'),
    ('mixed-both-orders',2,False,False,'discard'),
    ('mixed-both-orders',2,'pending_failure',False,2),
    ('mixed-both-orders',1,False,False,False),
    ('mixed-both-orders',1,False,False,7),
    ('mixed-both-orders',1,True,False,7),
    ('legacy-single',2,False,False,False),
    ('mixed-both-orders',1,False,False,'latency'),
    ('mixed-both-orders',1,'hung',False,'latency'),
    ('mixed-both-orders',1,False,False,'camera_latency'),
    ('target-pair',1,False,False,7),
    ('target-pair',1,True,False,7),
    ('target-pair',1,'hung',False,7),
    ('target-pair',1,False,False,'latency'),
    ('target-pair',1,'hung',False,'latency'),
    ('target-pair',1,False,False,'serial_latency'),
    ('target-pair',1,False,False,'serial_slow'),
    ('target-pair',1,False,False,'transport'),
    ('target-pair',1,'transport_hung',False,'transport'),
    ('target-stack',1,False,True,'serial_slow'),
    ('target-stack',1,'hung',False,'latency'),
    ('target-stack',1,False,False,'camera_latency'),
    ('target-stack',3,False,False,'sequence'),
    ('target-stack',3,False,False,'sequence_motion_preview'),
    ('target-stack',3,'home_failure',False,'sequence'),
    ('target-stack',3,'next_failure',False,'sequence'),
    *[('target-stack',3,case,False,'sequence') for case in
      ('empty_once','empty_persistent','empty_interrupt','empty_bad_recovery','empty_stop_failure')],
])
def test_persistent_servo_and_stop_on_inference_error(tmp_path,monkeypatch,inference_error,slow_fk,model,expected_cycles,approach):
    import numpy as np
    from lerobot.robots.aubo_i10 import aubo_i10
    m=load_entry(monkeypatch);clock=[10.];events=[];state=[*START_DEG,0];enabled=[False]
    target_metadata=[]
    motion_preview=approach=='sequence_motion_preview'
    if motion_preview:approach='sequence'
    motion_calls=[];preview_packets=[]
    if motion_preview:
        import cv2
        from lerobot.bamboo_sorting import joint_target_motion,joint_target_preview
        from lerobot.bamboo_sorting.joint_target_runtime import encode_mask
        mask=np.zeros((480,640),bool);mask[100:200,200:210]=True
        mask_png=encode_mask(mask)
        (tmp_path/'snapshot').mkdir()
        cv2.imwrite(str(tmp_path/'snapshot/global_rgb.png'),np.zeros((480,640,3),np.uint8))
        class MotionReferences:
            def __init__(self,rgb,masks):events.append(('motion_init',tuple(masks)))
            def update(self,rgb,targets,*,frame_id,required):
                motion_calls.append(tuple(targets))
                return dict(frame_id=frame_id,masks_png={t:mask_png for t in targets})
        class Preview:
            url='http://127.0.0.1:1234/'
            def publish(self,**packet):preview_packets.append(packet)
            def close(self):events.append(('preview_closed',True))
        monkeypatch.setattr(joint_target_motion,'TargetMotionReferences',MotionReferences)
        monkeypatch.setattr(joint_target_preview,'LiveTargetPreview',Preview)
    sequence_test=approach=='sequence'
    recovery_case=str(inference_error).startswith('empty_')
    sequence_failure=inference_error in ('home_failure','next_failure') or recovery_case
    empty_requests=[];b_requests=[];recovery_starts=[]
    original_trajectory=m.SmoothTrajectory
    def trajectory(q):
        if empty_requests:recovery_starts.append(list(q))
        return original_trajectory(q)
    monkeypatch.setattr(m,'SmoothTrajectory',trajectory)
    original_save=m.save_target_failure
    def save_failure(*args):
        events.append(('save_failure',args[1]))
        if inference_error!='empty_stop_failure':assert not enabled[0]
        return original_save(*args)
    monkeypatch.setattr(m,'save_target_failure',save_failure)
    worker_targets=[];homes=[]
    def fake_home(motion,read_state,**kw):
        assert not enabled[0] and read_state()[-1]==0
        homes.append(True)
        if inference_error=='home_failure':raise RuntimeError('test homing failure')
        state[:6]=START_DEG;clock[0]+=2.
        return {'moved':True,'final_state':state.copy()}
    monkeypatch.setattr(m,'home_to_start',fake_home)
    if model in ('target-pair','target-stack'):
        import joint_target_live_setup
        monkeypatch.setattr(joint_target_live_setup,'prepare_live_setup',lambda *a,**kw:(
            dict(snapshot_id='seed',target_id=a[2],**({'visible_targets':[dict(target_id=t,mask_png=mask_png) for t in kw['remaining_target_ids']]} if motion_preview else {})),dict(live_target_seed=dict(target_id=a[2],remaining=kw.get('remaining_target_ids')),expected_sha256='target-model',target_conditioned=True,
                return_action_chunk=True,segmentation_checkpoint='segmentation')))
    replan_test=approach in ('replan','replan_at_receipt')
    transport_test=approach in ('transport','sequence')
    transport_error=inference_error=='transport_hung'
    if transport_test or sequence_test:approach=7
    replan_at_receipt=approach=='replan_at_receipt'
    discard_test=approach=='discard'
    camera_latency_test=approach=='camera_latency'
    serial_slow_test=approach=='serial_slow'
    serial_latency_test=approach in ('serial_latency','serial_slow')
    latency_test=approach in ('latency','camera_latency','serial_latency','serial_slow') or sequence_test
    if latency_test:approach=7
    if replan_test or discard_test:approach=7
    if replan_at_receipt:approach=2
    phase_calls={};transport_responses=[0]
    delayed=[False];state_reads=[0];requests=[]
    def sleep(s):clock[0]+=s
    monkeypatch.setattr(m.time,'perf_counter',lambda:clock[0]);monkeypatch.setattr(m.time,'sleep',sleep)
    monkeypatch.setattr(m.sys.stdin,'isatty',lambda:True)
    monkeypatch.setattr(m,'read_task',lambda:'Pick one strip and place it in the collection area.')
    monkeypatch.setattr('builtins.input',lambda _: '')
    class Station:
        lower=[-360]*6;upper=[360]*6
        cameras={}
        def __init__(self):self.iface=self;self.state=self;self.algorithm=self;self.io_readback=self
        def connect(self):pass
        def close(self):pass
        def current(self,**kw):
            state_reads[0]+=1
            return state.copy(),[.2,-.5,.3],clock[0]
        def observe(self,**kw):
            assert kw['max_camera_age_ms']==(400 if latency_test and model!='target-pair' else (100 if model=='legacy-single' else 250))
            if 'state_snapshot' in kw:
                assert kw['state_snapshot'][0][-1]==state[-1]
            times=[clock[0]-.3,clock[0]-.02,clock[0]] if camera_latency_test else [clock[0]-.050327,clock[0],clock[0]]
            images={name:np.full((480,640,3),len(requests)%256,np.uint8) for name in ('global_rgb','grasp_rgb')} if recovery_case or motion_preview else {}
            return state.copy(),images,times
        def getMotionControl(self):return self
        def isSteady(self):return True
        def isServoModeEnabled(self):return enabled[0]
        def setServoMode(self,value):
            events.append(('mode',value))
            if not value and inference_error=='empty_stop_failure' and len(empty_requests)==1:
                empty_requests.append('stop_failed')
                return -1
            enabled[0]=value
            if not value and recovery_case and empty_requests and state[-1]==0:
                state[1]+=.07  # Simulate measured settling beyond the last sent position.
            return 0
        def servoJoint(self,q,*args):
            assert args[2]==(DT if model=='legacy-single' else .08)
            events.append(('send',clock[0]));state[:6]=[math.degrees(x) for x in q];return 0
        def forwardKinematics(self,q):
            if slow_fk and not delayed[0] and len([e for e in events if e[0]=='send'])>=3:
                delayed[0]=True;clock[0]+=.4 if model=='target-stack' else .11
            return [.2,-.5,.3,0,0,0],0
    class Pipe:
        def __init__(self,*a,**kw):
            self.target_id=kw.get('live_target_seed',{}).get('target_id')
            if sequence_test:
                assert not enabled[0]
                worker_targets.append(self.target_id)
                assert kw['live_target_seed']['remaining']==['A','B','C'][len(worker_targets)-1:]
                clock[0]+=2.
            assert kw['request_timeout_s']==(5. if latency_test else 2.)
            if model=='mixed-double':
                assert kw['expected_sha256']==m.MIXED_MODEL_SHA
                assert kw['checkpoint']==m.MIXED_CHECKPOINT
                assert kw['inference_script']==m.MIXED_INFERENCE_SCRIPT
            elif model=='mixed-both-orders':
                assert kw['expected_sha256']==m.BOTH_ORDERS_MODEL_SHA
                assert kw['checkpoint']==m.BOTH_ORDERS_CHECKPOINT
                assert kw['remote_root']==m.BOTH_ORDERS_REMOTE_ROOT
                assert kw['inference_script']==(m.BOTH_ORDERS_APPROACH_SCRIPT if approach else m.BOTH_ORDERS_INFERENCE_SCRIPT)
                assert kw.get('return_action_chunk',False)==bool(approach)
            elif model in ('target-pair','target-stack'):
                assert kw['target_conditioned'] and kw['expected_sha256']=='target-model'
                assert kw['return_action_chunk']
            else:
                assert 'expected_sha256' not in kw
        def close(self):pass
        def predict(self,index,*a,**kw):
            if model in ('target-pair','target-stack'):
                assert kw['codec']=='rgb_zlib'
                target_metadata.append(dict(kw['target_input']))
                assert kw['target_input']['target_id']==self.target_id
                if motion_preview:
                    if kw['target_input']['phase']=='approach':
                        assert kw['target_motion']['frame_id']==kw['target_input']['frame_id']
                        assert list(kw['target_motion']['masks_png'])==['A','B','C'][['A','B','C'].index(self.target_id):]
                    else:assert 'target_motion' not in kw
            if recovery_case and self.target_id=='B':
                b_requests.append(index)
                if len(b_requests)==2 or (len(b_requests)>2 and inference_error=='empty_persistent'):
                    from lerobot.bamboo_sorting.joint_target_runtime import encode_mask
                    assert state[-1]==0
                    empty_requests.append(index);events.append(('empty',index))
                    raise m.TargetForegroundMissing(dict(**kw['target_input'],
                        segmentation_instances_png=encode_mask(np.zeros((480,640),bool),allow_empty=True)))
                if len(b_requests)>2 and inference_error=='empty_interrupt':raise KeyboardInterrupt()
                if len(b_requests)>2 and inference_error=='empty_bad_recovery':raise RuntimeError('recovery model failure')
            if sequence_test and inference_error=='next_failure' and self.target_id=='B':raise RuntimeError('next target inference failed')
            if inference_error and not sequence_failure and index and not transport_error:raise RuntimeError('inference failed')
            if approach:
                chunk=[[*START_DEG,0.] for _ in range(50)]
                if not inference_error or sequence_failure or inference_error=='pending_failure' or transport_error:
                    phase=sum(kind=='suction' for kind,value in events)
                    phase_calls[phase]=phase_calls.get(phase,0)+1
                    if a[0][-1]==100:
                        if model in ('target-pair','target-stack') and kw['target_input']['phase']=='transport':transport_responses[0]+=1
                        # Closed-gripper phase still uses the original first
                        # step: move before release, then repeat the next pick.
                        for row in chunk:row[0]+=.4;row[5]+=.4
                        if transport_test:
                            for i,row in enumerate(chunk):
                                row[0]=START_DEG[0]+(.02*i if i<12 else .4)
                                row[-1]=0. if i==12 else 100.
                                if i>12:row[0]=361.  # Must never skip the release row.
                            if transport_error:
                                for row in chunk:row[0]=START_DEG[0]+.2;row[-1]=100.
                    elif state[-1]==100:
                        # An in-flight open-state prediction must be discarded
                        # after IO, before its obsolete joint/IO target is used.
                        for row in chunk:row[0]=361.
                    else:
                        # Later rows reopen and move farther: never skip over
                        # the first closure even when nominal age exceeds it.
                        for i in range(approach,50):
                            chunk[i][5]+=.4 if i==approach else 3.
                            chunk[i][6]=100. if i==approach else 0.
                        if phase_calls[phase]>1:
                            # Fresh pre-IO prediction confirms the same target.
                            chunk=[[*START_DEG,100.] for _ in range(50)]
                            for row in chunk:row[5]+=.4
                        if discard_test and phase==2:
                            # Already at the next goal: IO completes before
                            # the in-flight result returns, so discard it.
                            chunk=[[*START_DEG,0.] for _ in range(50)]
                            for i,row in enumerate(chunk):
                                row[0]+=.4;row[5]+=.4
                                if i>=approach:row[-1]=100.
                        if replan_test and phase==2:
                            if phase_calls[phase]==1:
                                # A not-yet-reached second closure target.
                                for row in chunk:row[3]+=.4
                            elif phase_calls[phase]==2:
                                # New observation says keep approaching OPEN.
                                # Do not close at the superseded target.
                                chunk=[[*START_DEG,0.] for _ in range(50)]
                                for row in chunk:row[3]+=.2;row[5]+=.4
                            else:
                                chunk=[[*START_DEG,100.] for _ in range(50)]
                                for row in chunk:row[3]+=.2;row[5]+=.4
                if serial_slow_test and index==0:
                    # Keep approaching while the second 550-ms request runs;
                    # the old 750-ms age limit stopped this exact waiting path.
                    chunk=[[*START_DEG,0.] for _ in range(50)]
                return {'action':chunk[0], 'raw_action':chunk[0], 'shape':[50,7],
                    'action_chunk':chunk,'raw_action_chunk':chunk,
                    **({'target_evidence':dict(kw['target_input'],mask_png=None)} if motion_preview else {}),
                    **({'target_input':dict(kw['target_input'])} if model in ('target-pair','target-stack') else {})}
            # The fake policy keeps proposing cycles; only the runner stops it.
            suction=0 if inference_error else (100 if index%2==0 else 0)
            return {'action':[*START_DEG,suction]}
    class Future:
        def __init__(self,fn,args):
            self.fn=fn;self.args=args
            self.ready=clock[0]+(.55 if serial_slow_test else .34 if serial_latency_test else .54 if latency_test and not camera_latency_test and args[0]==0 else .14)
            if transport_test and args[1][-1]==100:self.ready=clock[0]+.34
        def done(self):
            if inference_error=='hung' and self.args[0]>0:return False
            if transport_error and self.args[1][-1]==100 and transport_responses[0]>=1:return False
            return clock[0]>=self.ready
        def result(self):return self.fn(*self.args)
    class Pool:
        def __init__(self,**kw):pass
        def submit(self,fn,*args):
            assert args[0] not in requests  # Discarding must not overwrite saved input snapshots.
            requests.append(args[0])
            return Future(fn,args)
        def shutdown(self,**kw):pass
    class Suction:
        last_gripper_command_trace=None
        def __init__(self,*args):pass
        def _control_suction_based_on_gripper(self,x):
            if approach:
                assert abs(state[5]-START_DEG[5]-.4)<.15
                if x==0:assert abs(state[0]-START_DEG[0]-.4)<.15
                if replan_test and x==100 and any(kind=='suction' and value==0 for kind,value in events):
                    assert phase_calls[2]>=3  # No IO for the superseded closure.
                    assert state[3]==pytest.approx(START_DEG[3]+.2,abs=.15)
            state[-1]=x;self.last_gripper_command_trace={'target':x};events.append(('suction',x))
    class Recorder:
        def __init__(self,*args):pass
        def start(self):events.append(('video_start',True))
        def close(self):
            assert not enabled[0]
            events.append(('video_close',True));return {'complete':True}
    monkeypatch.setattr(m,'ReadOnlyStation',Station);monkeypatch.setattr(m,'InferencePipe',Pipe)
    monkeypatch.setattr(m,'ThreadPoolExecutor',Pool);monkeypatch.setattr(aubo_i10,'AuboI10Robot',Suction)
    monkeypatch.setattr(m,'TrialVideoRecorder',Recorder)
    out=tmp_path/'run'
    argv=['--execute','--model',model,'--output',str(out)]
    if model=='target-stack':argv+=['--max-seconds','600']
    if model in ('target-pair','target-stack'):
        argv+=['--target-snapshot','snapshot','--target-selections','labels']
        argv+=['--target-sequence','A','B','C'] if sequence_test else ['--target-id','A']
    if expected_cycles!=(1 if model=='legacy-single' else 2):argv+=['--cycles',str(expected_cycles)]
    if approach:argv.append('--approach-age-aligned')
    if latency_test:argv.append('--latency-tolerant')
    if transport_test:argv.append('--transport-age-aligned')
    if motion_preview:
        argv[argv.index('--target-snapshot')+1]=str(tmp_path/'snapshot')
        argv.extend(['--track-target-motion','--target-preview'])
    result=m.main(argv)
    if motion_preview:
        assert [x for x in events if x[0]=='motion_init']==[('motion_init',('A','B','C'))]
        assert all(x in motion_calls for x in [('A','B','C'),('B','C'),('C',),()])
        assert preview_packets and ('preview_closed',True) in events
    if recovery_case:
        complete=json.loads((out/'complete.json').read_text())
        rows=list(map(json.loads,(out/'trace.jsonl').read_text().splitlines()))
        commands=list(map(json.loads,(out/'commands.jsonl').read_text().splitlines()))
        misses=[r['target_foreground_missing'] for r in rows if 'target_foreground_missing' in r]
        recovered=[r['target_recovered'] for r in rows if 'target_recovered' in r]
        assert not enabled[0]
        assert len(misses)==(4 if inference_error=='empty_persistent' else 1)
        first=misses[0]['index']
        # Failure occurs with a pending close; it must never commit that IO.
        before=[c for c in commands if c['target_id']=='B' and c['prediction_index']<first]
        assert before and before[-1]['suction_target_frozen'] and before[-1]['target'][-1]==100
        assert all(c['prediction_index'] not in empty_requests for c in commands)
        for i,event in enumerate(events):
            if event[0]=='empty':
                tail=events[i+1:]
                next_enable=next((j for j,e in enumerate(tail) if e==('mode',True)),len(tail))
                assert not any(e[0] in ('send','suction') for e in tail[:next_enable])
        for miss in misses:
            folder=out/miss['evidence']
            saved=json.loads((folder/'failure.json').read_text())
            assert saved['index']==miss['index'] and saved['target_input']['target_id']=='B'
            assert (folder/'global_rgb.png').exists() and (folder/'grasp_rgb.png').exists()
        if inference_error=='empty_once':
            assert result==0 and complete['completed_target_ids']==['A','B','C']
            assert len(recovered)==1 and recovered[0]['index']>first
            assert recovered[0]['trajectory_velocity']==[0.]*6
            assert recovery_starts[0]==recovered[0]['state'][:6]
            assert all(c['prediction_index']>first for c in commands if c['time']>=next(
                r['target_reobserve']['stopped_at'] for r in rows if 'target_reobserve' in r) and c['target_id']=='B')
            assert worker_targets==['A','B','C'] and len(homes)==2
        else:
            assert not recovered and complete['completed_target_ids']==['A']
            assert worker_targets==['A','B'] and len(homes)==1
            assert [v for k,v in events if k=='suction']==[100,0]
            assert result==(0 if inference_error=='empty_interrupt' else 1)
            if inference_error=='empty_interrupt':assert complete['reason']=='operator_interrupt'
            if inference_error=='empty_bad_recovery':assert 'recovery model failure' in complete['failure']
        return
    assert result==int(bool(inference_error))
    assert [e for e in events if e[0]=='mode']==[('mode',True),('mode',False)]*(3 if sequence_test and not inference_error else 1)
    assert len([e for e in events if e[0]=='send'])>1 and not enabled[0]
    r=json.loads((out/'complete.json').read_text())
    assert r['suction_cycle_commanded']==(not inference_error)
    assert r['requested_suction_cycles']==expected_cycles
    assert r['completed_suction_cycles']==(expected_cycles if not inference_error else 1 if sequence_failure else 0)
    assert [x for kind,x in events if kind=='suction']==([100,0]*expected_cycles if not inference_error else [100,0] if sequence_failure else [100] if transport_error else [])
    if approach and not inference_error:
        assert r['reason']==('model_suction_cycle_completed' if expected_cycles==1 else 'model_suction_cycles_completed')
        assert r['approach_age_aligned'] and r['predictions']>=2*expected_cycles
        records=[json.loads(s) for s in (out/'commands.jsonl').read_text().splitlines()]
        frozen=[c for c in records if c['suction_target_frozen'] and c['target'][-1]==100]
        assert frozen and all(c['action_selection']['selected_index'] in (approach,0) for c in frozen)
        accepted={r['index']:r['prediction'] for r in map(json.loads,(out/'trace.jsonl').read_text().splitlines())
                  if 'prediction' in r}
        assert all(c['target']==accepted[c['prediction_index']]['action_chunk'][c['action_selection']['selected_index']]
                   for c in frozen)
        if transport_test:
            closed=[c for c in records if c['state'][-1]==100 and c['action_selection'] is not None]
            assert closed and any(c['action_selection']['selected_index']>0 for c in closed)
            assert all(c['target']==accepted[c['prediction_index']]['action_chunk'][c['action_selection']['selected_index']]
                       and accepted[c['prediction_index']]['target_input']['phase']=='transport' for c in closed)
            releases=[c for c in closed if c['suction_target_frozen']]
            assert releases and all(c['action_selection']['selected_index']==12 and c['target'][-1]==0 for c in releases)
            # Between predictions the selected step advances; it is not fixed at receipt.
            assert any(len({c['action_selection']['selected_index'] for c in closed if c['prediction_index']==i})>1
                       for i in {c['prediction_index'] for c in closed})
            assert r['transport_age_aligned'] is True
        else:
            assert any(c['state'][-1]==100 and c['action_selection'] is None for c in records)
        assert json.loads((out/'final_state.json').read_text())['state'][6]==0
        if discard_test:
            assert 'observation_precedes_suction_change' in (out/'trace.jsonl').read_text()
        if replan_test:
            changes=[json.loads(s)['pending_suction_replanned'] for s in (out/'trace.jsonl').read_text().splitlines()
                     if 'pending_suction_replanned' in json.loads(s)]
            assert any(r['previous_target'][-1]==100 and r['updated_target'][-1]==0 for r in changes)
    control_events=[e for e in events if e[0] not in ('motion_init','preview_closed')]
    assert r['video']['complete'] and control_events[0][0]=='video_start' and control_events[-1][0]=='video_close'
    trace=(out/'trace.jsonl').read_text()
    if latency_test:
        first=next(json.loads(s) for s in trace.splitlines() if 'prediction' in json.loads(s))
        if camera_latency_test:
            assert .25 < first['sensor_times'][1]-first['sensor_times'][0] < .3
            assert .4 < first['received_at']-min(first['sensor_times']) < .75
        elif serial_slow_test:
            assert .55 < first['received_at']-min(first['sensor_times']) < .75
            assert any(c.get('action_selection') and .75<c['action_selection']['observation_age_s']<1.5
                       for c in records)
        elif serial_latency_test:assert .3 < first['received_at']-min(first['sensor_times']) < .5
        else:assert .5 < first['received_at']-min(first['sensor_times']) < .75
        if inference_error=='hung':
            expected='no fresh prediction within 1500 ms' if model in ('target-pair','target-stack') else 'limit=750 ms'
            assert expected in r['failure']
    assert 'observation_skipped' not in trace
    assert ('state_refresh_before_servo' in trace)==slow_fk
    if model in ('target-pair','target-stack'):
        rows=[json.loads(line) for line in trace.splitlines()]
        accepted=[row['target_phase_command'] for row in rows if 'target_phase_command' in row]
        if sequence_test:
            accepted_predictions={row['index']:row['prediction'] for row in rows if 'prediction' in row}
            assert len(homes)==(2 if not inference_error else 1)
            assert worker_targets==(['A','B','C'] if not inference_error else ['A'] if inference_error=='home_failure' else ['A','B'])
            assert r['completed_target_ids']==(['A','B','C'] if not inference_error else ['A'])
            for target in r['completed_target_ids']:
                phase_rows=[x for x in accepted if x['target_id']==target]
                assert len(phase_rows)==2 and phase_rows[0]['first_accepted_close_s']==phase_rows[1]['first_accepted_close_s']
                assert next(meta for meta in target_metadata if meta['target_id']==target)['phase']=='approach'
            closes=[x['first_accepted_close_s'] for x in accepted[::2]]
            assert closes==sorted(set(closes))
            if not inference_error:
                for c in records:
                    assert c['target_id']==accepted_predictions[c['prediction_index']]['target_input']['target_id']
        elif not inference_error:
            assert len(accepted)==2
            assert accepted[0]['first_accepted_close_s']==accepted[1]['first_accepted_close_s']
            # A short synthetic cycle may finish from a pre-close camera frame
            # paired with a post-close state. Phase follows the image timestamp,
            # not the gripper state or an assumption that transport must occur.
            first_close=accepted[0]['first_accepted_close_s']
            for row in rows:
                if 'prediction' in row:
                    meta=next(meta for meta in target_metadata if meta['frame_id']==f'seed/live/{row["index"]}')
                    assert meta['phase']==('transport' if row['sensor_times'][0]>first_close else 'approach')
        elif transport_error:
            assert len(accepted)==1
            assert 'expired prediction' in r['failure']
            assert json.loads((out/'final_state.json').read_text())['state'][6]==100
            commands=[json.loads(line) for line in (out/'commands.jsonl').read_text().splitlines()]
            assert any(c['state'][-1]==100 and c['action_selection'] and c['action_selection']['selected_index']>0 for c in commands)
        else:
            assert not accepted and all(meta['phase']=='approach' for meta in target_metadata)
    if slow_fk:assert delayed[0] and state_reads[0]>len([e for e in events if e[0]=='send'])
    if model=='target-stack':
        assert json.loads((out/'plan.json').read_text())['max_seconds']==600
        if slow_fk:
            sent_times=[v for k,v in events if k=='send']
            assert any(.35 < b-a < .7 for a,b in zip(sent_times,sent_times[1:]))


def test_j6_goal_jump_is_smoothed_and_servo_tracking_still_checked():
    state=[-20.52568286943511,-.26836802649967184,122.88070629304367,34.61300456793464,89.44394229201521,-147.68657562608755,0]
    action=[-19.525897979736328,-.56625896692276,122.88737487792969,34.91891860961914,89.41212463378906,-144.48699951171875,0]
    assert validate_prediction(action,state,[.507,-.409,.12],[.52,-.4,.12],[-360]*6,[360]*6,[1]*3,1.2)==action
    t=SmoothTrajectory(state[:6]);previous_v=t.v.copy()
    for _ in range(200):
        old=t.q.copy();q,_=t.advance(action[:6],lambda q:[.5,-.4,.12])
        assert max(abs(a-b) for a,b in zip(q,old))<=MAX_SPEED*DT+1e-8
        assert max(abs(a-b) for a,b in zip(t.v,previous_v))<=MAX_ACCEL*DT+1e-8
        previous_v=t.v.copy()
    assert t.q==pytest.approx(action[:6],abs=.001)
    # A physically stuck robot must still stop even though its model goal is valid.
    with pytest.raises(ValueError,match='J6 joint target'):
        require_joint_target(t.q,state[:6],[-360]*6,[360]*6,MAX_TRACKING_ERROR_DEG)


def test_servo_disable_waits_for_controller_without_resending():
    from lerobot.bamboo_sorting.joint_trial import SingleStepSession
    class Motion:
        def __init__(self):self.reads=0
        def isServoModeEnabled(self):self.reads+=1;return self.reads<25
    sleeps=[];m=Motion()
    SingleStepSession(m,sleep=sleeps.append)._wait_servo(False,timeout_s=1.)
    assert sum(sleeps)==pytest.approx(.12)
    class Stuck:
        def isServoModeEnabled(self):return True
    sleeps=[]
    with pytest.raises(RuntimeError,match='remains enabled'):
        SingleStepSession(Stuck(),sleep=sleeps.append)._wait_servo(False,timeout_s=1.)
    assert sum(sleeps)==pytest.approx(1.)


@pytest.mark.parametrize('skew,accepted',[(.050327,True),(.099,True),(.101,False)])
def test_camera_skew_relaxed_to_100ms(skew,accepted):
    args=([*START_DEG,0],[*START_DEG,0],[.2,-.5,.3],[.2,-.5,.3],[-360]*6,[360]*6,[1,1+skew,1+skew],1.2)
    if accepted:
        assert validate_prediction(*args)==[*START_DEG,0]
    else:
        with pytest.raises(ValueError,match='camera skew'):validate_prediction(*args)


def test_tcp_limit_and_j6_deceleration_are_satisfied_together():
    # Last commanded q/v and target from joint_smooth_20260922_235754_796241.
    # A local linear kinematics surrogate reproduces the conflict without RPC;
    # it is not a claim of exact AUBO FK replay.
    q0=[-23.806631537483575,2.03388843016352,123.0585963053864,32.518577394666586,89.54535485981886,-211.53457096427564]
    v0=[4.041209964756092,-.6345783592048441,.15510540447439303,.7445300663751566,-.13183160933018875,-3.0555219335025994]
    target=[-22.648202896118164,1.8356952667236328,123.06416320800781,32.72412872314453,89.51194763183594,-211.74574279785156]
    def coupled_fk(q):return [.47+.012*(q[0]-q0[0]),-.43+.001*(q[5]-q0[5]),.12]
    t=SmoothTrajectory(q0);t.v=v0.copy()
    q,tcp=t.advance(target,coupled_fk)
    assert math.dist(coupled_fk(q0),tcp)<=MAX_TCP_SPEED*DT+1e-7
    assert max(abs(a-b) for a,b in zip(t.v,v0))<=MAX_ACCEL*DT+1e-8
    assert max(abs(v) for v in t.v)<=MAX_SPEED
    assert q!=q0


def test_nonlinear_multiaxis_tracking_limits_and_fk_call_budget():
    import random
    rng=random.Random(23);calls=[0]
    def coupled_fk(q):
        calls[0]+=1
        return [.3+.013*q[0]+.002*math.sin(q[1]),-.5+.006*q[1]+.001*q[5],.3+.004*q[2]]
    t=SmoothTrajectory([0]*6)
    for tick in range(800):
        if tick%7==0:target=[rng.uniform(-8,8) for _ in range(6)]
        previous_q=t.q.copy();previous_v=t.v.copy();before=calls[0]
        q,tcp=t.advance(target,coupled_fk)
        assert calls[0]-before<=9  # Bounded RPC load if FK comes from the controller.
        assert math.dist(coupled_fk(previous_q),tcp)<=MAX_TCP_SPEED*DT+1e-7
        assert max(abs(a-b) for a,b in zip(t.v,previous_v))<=MAX_ACCEL*DT+1e-8
        assert max(abs(v) for v in t.v)<=MAX_SPEED


def test_unusable_braking_candidate_rejected_without_mutating_trajectory():
    t=SmoothTrajectory([0]*6);t.v=[2.]*6
    old_q=t.q.copy();old_v=t.v.copy()
    def abrupt_fk(q):return [.2 if q==old_q else .3,-.5,.3]
    with pytest.raises(ValueError):t.advance([10]*6,abrupt_fk)
    assert t.q==old_q and t.v==old_v


def test_failure_inputs_saved_losslessly_and_existing_evidence_not_overwritten(tmp_path,monkeypatch):
    import cv2
    from lerobot.bamboo_sorting.joint_target_runtime import encode_mask
    m=load_entry(monkeypatch)
    images={k:np.random.default_rng(i).integers(0,256,(480,640,3),dtype=np.uint8)
            for i,k in enumerate(('global_rgb','grasp_rgb'))}
    meta=dict(target_id='B',phase='approach',frame_id='live/42')
    evidence=dict(**meta,segmentation_instances_png=encode_mask(np.zeros((480,640),bool),allow_empty=True),
        segmentation_segments=[],visible_masks_png={'B':None,'C':None})
    exc=m.TargetForegroundMissing(evidence)
    folder=tmp_path/m.save_target_failure(tmp_path,42,[*START_DEG,0],images,[1.,2.,3.],meta,exc)
    for key,value in images.items():
        np.testing.assert_array_equal(cv2.cvtColor(cv2.imread(str(folder/f'{key}.png')),cv2.COLOR_BGR2RGB),value)
    saved=json.loads((folder/'failure.json').read_text())
    assert saved['target_evidence']==evidence and saved['sensor_times']==[1.,2.,3.]
    before=(folder/'global_rgb.png').read_bytes()
    with pytest.raises(FileExistsError):m.save_target_failure(tmp_path,42,[],{},[],meta,exc)
    assert (folder/'global_rgb.png').read_bytes()==before

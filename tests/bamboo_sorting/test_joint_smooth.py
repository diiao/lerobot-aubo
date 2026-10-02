import json
import math
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


def chunk_response(close_at=None):
    chunk=[[*START_DEG,0.] for _ in range(50)]
    for i,row in enumerate(chunk):row[5]+=i*.1
    if close_at is not None:chunk[close_at][-1]=100.
    return {'shape':[50,7],'action':chunk[0].copy(),'raw_action':chunk[0].copy(),
            'action_chunk':[r.copy() for r in chunk],'raw_action_chunk':[r.copy() for r in chunk]}


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
    assert plan['checkpoint_sha256']=={'legacy-single':m.MODEL_SHA,
        'mixed-double':m.MIXED_MODEL_SHA,'mixed-both-orders':m.BOTH_ORDERS_MODEL_SHA}[model]
    assert plan['startup_release_if_unset']=='operator_confirmed'


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
])
def test_persistent_servo_and_stop_on_inference_error(tmp_path,monkeypatch,inference_error,slow_fk,model,expected_cycles,approach):
    import numpy as np
    from lerobot.robots.aubo_i10 import aubo_i10
    m=load_entry(monkeypatch);clock=[10.];events=[];state=[*START_DEG,0];enabled=[False]
    replan_test=approach in ('replan','replan_at_receipt')
    replan_at_receipt=approach=='replan_at_receipt'
    discard_test=approach=='discard'
    if replan_test or discard_test:approach=7
    if replan_at_receipt:approach=2
    phase_calls={}
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
            assert kw['max_camera_age_ms']==(250 if expected_cycles==2 else 100)
            if 'state_snapshot' in kw:
                assert kw['state_snapshot'][0][-1]==state[-1]
            return state.copy(),{},[clock[0]-.050327,clock[0],clock[0]]
        def getMotionControl(self):return self
        def isSteady(self):return True
        def isServoModeEnabled(self):return enabled[0]
        def setServoMode(self,value):events.append(('mode',value));enabled[0]=value;return 0
        def servoJoint(self,q,*args):
            assert args[2]==(.08 if expected_cycles==2 else DT)
            events.append(('send',clock[0]));state[:6]=[math.degrees(x) for x in q];return 0
        def forwardKinematics(self,q):
            if slow_fk and not delayed[0] and len([e for e in events if e[0]=='send'])>=3:
                delayed[0]=True;clock[0]+=.11
            return [.2,-.5,.3,0,0,0],0
    class Pipe:
        def __init__(self,*a,**kw):
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
            else:
                assert 'expected_sha256' not in kw
        def close(self):pass
        def predict(self,index,*a):
            if inference_error and index:raise RuntimeError('inference failed')
            if approach:
                chunk=[[*START_DEG,0.] for _ in range(50)]
                if not inference_error or inference_error=='pending_failure':
                    phase=sum(kind=='suction' for kind,value in events)
                    phase_calls[phase]=phase_calls.get(phase,0)+1
                    if a[0][-1]==100:
                        # Closed-gripper phase still uses the original first
                        # step: move before release, then repeat the next pick.
                        for row in chunk:row[0]+=.4;row[5]+=.4
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
                return {'action':chunk[0], 'raw_action':chunk[0], 'shape':[50,7],
                    'action_chunk':chunk,'raw_action_chunk':chunk}
            suction=0 if inference_error else (100 if index in ((0,2) if expected_cycles==2 else (0,)) else 0)
            return {'action':[*START_DEG,suction]}
    class Future:
        def __init__(self,fn,args):self.fn=fn;self.args=args;self.ready=clock[0]+.14
        def done(self):
            if inference_error=='hung' and self.args[0]>0:return False
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
    if approach:argv.append('--approach-age-aligned')
    assert m.main(argv)==int(bool(inference_error))
    assert [e for e in events if e[0]=='mode']==[('mode',True),('mode',False)]
    assert len([e for e in events if e[0]=='send'])>1 and not enabled[0]
    r=json.loads((out/'complete.json').read_text())
    assert r['suction_cycle_commanded']==(not inference_error)
    assert r['requested_suction_cycles']==expected_cycles
    assert r['completed_suction_cycles']==(expected_cycles if not inference_error else 0)
    assert [x for kind,x in events if kind=='suction']==([100,0]*expected_cycles if not inference_error else [])
    if approach and not inference_error:
        assert r['reason']=='model_suction_cycles_completed' and r['approach_age_aligned']
        assert r['predictions']>=4
        records=[json.loads(s) for s in (out/'commands.jsonl').read_text().splitlines()]
        frozen=[c for c in records if c['suction_target_frozen'] and c['target'][-1]==100]
        assert frozen and all(c['action_selection']['selected_index'] in (approach,0) for c in frozen)
        accepted={r['index']:r['prediction'] for r in map(json.loads,(out/'trace.jsonl').read_text().splitlines())
                  if 'prediction' in r}
        assert all(c['target']==accepted[c['prediction_index']]['action_chunk'][c['action_selection']['selected_index']]
                   for c in frozen)
        assert any(c['state'][-1]==100 and c['action_selection'] is None for c in records)
        assert json.loads((out/'final_state.json').read_text())['state'][6]==0
        if discard_test:
            assert 'observation_precedes_suction_change' in (out/'trace.jsonl').read_text()
        if replan_test:
            changes=[json.loads(s)['pending_suction_replanned'] for s in (out/'trace.jsonl').read_text().splitlines()
                     if 'pending_suction_replanned' in json.loads(s)]
            assert any(r['previous_target'][-1]==100 and r['updated_target'][-1]==0 for r in changes)
    assert r['video']['complete'] and events[0][0]=='video_start' and events[-1][0]=='video_close'
    trace=(out/'trace.jsonl').read_text()
    assert 'observation_skipped' not in trace
    assert ('state_refresh_before_servo' in trace)==slow_fk
    if slow_fk:assert delayed[0] and state_reads[0]>len([e for e in events if e[0]=='send'])


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

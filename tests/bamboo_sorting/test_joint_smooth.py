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
    with pytest.raises(ValueError,match='expired'):validate_prediction(action,**{**args,'now':1.4})
    with pytest.raises(ValueError,match='workspace'):validate_prediction(action,**{**args,'target_tcp':[.19,-.65,-.001]})
    farther=[state[0]+10,*action[1:]]
    assert validate_prediction(farther,**{**args,'target_tcp':[.3,-.65,.16]})==farther
    with pytest.raises(ValueError,match='controller joint limits'):
        validate_prediction([361,*action[1:]],**args)


def load_entry(monkeypatch):
    import importlib.util
    root=Path(__file__).resolve().parents[2]/'examples/phone_to_auboi10'
    monkeypatch.syspath_prepend(str(root))
    spec=importlib.util.spec_from_file_location('smooth_entry',root/'run_joint_smooth.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_plan_does_not_connect(monkeypatch):
    m=load_entry(monkeypatch)
    def fail(*a,**k):raise AssertionError('device touched')
    monkeypatch.setattr(m,'ReadOnlyStation',fail);monkeypatch.setattr(m,'InferencePipe',fail)
    assert m.main([])==0


@pytest.mark.parametrize('inference_error',[False,True,'hung'])
def test_persistent_servo_and_stop_on_inference_error(tmp_path,monkeypatch,inference_error):
    import numpy as np
    from lerobot.robots.aubo_i10 import aubo_i10
    m=load_entry(monkeypatch);clock=[10.];events=[];state=[*START_DEG,0];enabled=[False]
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
        def current(self,**kw):return state.copy(),[.2,-.5,.3],clock[0]
        def observe(self,**kw):
            if 'state_snapshot' in kw:
                assert kw['state_snapshot'][0][-1]==state[-1]
            return state.copy(),{},[clock[0]-.050327,clock[0],clock[0]]
        def getMotionControl(self):return self
        def isSteady(self):return True
        def isServoModeEnabled(self):return enabled[0]
        def setServoMode(self,value):events.append(('mode',value));enabled[0]=value;return 0
        def servoJoint(self,q,*args):events.append(('send',clock[0]));state[:6]=[math.degrees(x) for x in q];return 0
        def forwardKinematics(self,q):return [.2,-.5,.3,0,0,0],0
    class Pipe:
        def __init__(self,*a,**kw):pass
        def close(self):pass
        def predict(self,index,*a):
            if inference_error and index:raise RuntimeError('inference failed')
            return {'action':[*START_DEG,0 if inference_error or index else 100]}
    class Future:
        def __init__(self,fn,args):self.fn=fn;self.args=args;self.ready=clock[0]+.14
        def done(self):
            if inference_error=='hung' and self.args[0]>0:return False
            return clock[0]>=self.ready
        def result(self):return self.fn(*self.args)
    class Pool:
        def __init__(self,**kw):pass
        def submit(self,fn,*args):return Future(fn,args)
        def shutdown(self,**kw):pass
    class Suction:
        last_gripper_command_trace=None
        def __init__(self,*args):pass
        def _control_suction_based_on_gripper(self,x):state[-1]=x;self.last_gripper_command_trace={'target':x}
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
    assert m.main(['--execute','--output',str(out)])==int(bool(inference_error))
    assert [e for e in events if e[0]=='mode']==[('mode',True),('mode',False)]
    assert len([e for e in events if e[0]=='send'])>1 and not enabled[0]
    r=json.loads((out/'complete.json').read_text())
    assert r['suction_cycle_commanded']==(not inference_error)
    assert r['video']['complete'] and events[0][0]=='video_start' and events[-1][0]=='video_close'
    assert 'observation_skipped' not in (out/'trace.jsonl').read_text()


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

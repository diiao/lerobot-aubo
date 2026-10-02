import importlib.util
import math
from pathlib import Path

import pytest

from lerobot.bamboo_sorting.joint_trial import START_DEG, SingleStepSession, TrialLimits, check_step, home_to_start, stationary_step_limits


def inputs():
    return dict(action=[START_DEG[0]+.1,*START_DEG[1:],0],current=[*START_DEG,0],
        lower=[-360]*6,upper=[360]*6,current_tcp=[.4,-.4,.4],target_tcp=[.401,-.4,.4],
        sensor_times=[1,1,1],state_time=1.02,now=1.05)


def test_accept_preserves_action():
    args=inputs();result=check_step(**args)
    assert result['passed'] and result['action']==args['action']


@pytest.mark.parametrize('key,value,reason',[
    ('now',1.101,'expired_observation'),('state_time',.9,'expired_current_state'),
    ('sensor_times',[.97,1.03,1.03],'camera_skew'),('sensor_times',[2,2,2],'invalid_clock'),
    ('action',[*START_DEG,100],'single_step_requires_suction_off_and_unchanged'),
    ('action',[START_DEG[0]+360,*START_DEG[1:],0],'tracking distance'),
    ('target_tcp',[.41,-.4,.4],'tcp_command_step_or_speed'),
    ('current',[START_DEG[0]+1,*START_DEG[1:],0],'not_at_start'),
    ('upper',[-70,360,360,360,360,360],'outside controller joint limits'),
    ('action',[*START_DEG,50],'invalid_suction'),
])
def test_reject(key,value,reason):
    args=inputs();args[key]=value;result=check_step(**args)
    assert not result['passed'] and any(reason in s for s in result['reasons'])


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True])
def test_bad_numeric(value):
    args=inputs();args['action'][0]=value
    with pytest.raises(ValueError):check_step(**args)


class Motion:
    def __init__(self,ret=0):self.calls=[];self.enabled=False;self.ret=ret
    def isServoModeEnabled(self):return self.enabled
    def setServoMode(self,value):self.calls.append(('servo',value));self.enabled=value;return 0
    def servoJoint(self,*args):self.calls.append(('send',args));return self.ret


def test_single_send_units_and_stop():
    m=Motion();gate=check_step(**inputs());session=SingleStepSession(m,sleep=lambda s:None)
    session.execute(gate,revalidate=lambda:gate)
    assert len([c for c in m.calls if c[0]=='send'])==1
    assert m.calls[1][1][0]==pytest.approx([math.radians(v) for v in gate['action'][:6]])
    assert m.calls[-1]==('servo',False) and not m.enabled
    with pytest.raises(RuntimeError,match='consumed'):session.execute(gate,revalidate=lambda:gate)


def test_async_servo_transition_waits_without_resending():
    class AsyncMotion(Motion):
        pending=None
        reads=0
        def setServoMode(self,value):
            self.calls.append(('servo',value));self.pending=value;self.reads=0;return 0
        def isServoModeEnabled(self):
            self.reads+=1
            if self.pending is not None and self.reads>=3:
                self.enabled=self.pending;self.pending=None
            return self.enabled
    m=AsyncMotion();gate=check_step(**inputs());sleeps=[]
    session=SingleStepSession(m,sleep=sleeps.append)
    session.execute(gate,revalidate=lambda:gate)
    assert [c[0] for c in m.calls]==['servo','send','servo']
    assert sleeps==[.005,.005,.04,.005,.005]
    assert session.command_attempted and session.command_accepted and session.servo_disabled


def test_servo_enable_timeout_never_sends():
    class StuckMotion(Motion):
        def setServoMode(self,value):self.calls.append(('servo',value));return 0
    m=StuckMotion();gate=check_step(**inputs());sleeps=[]
    session=SingleStepSession(m,sleep=sleeps.append)
    with pytest.raises(RuntimeError,match='enable not confirmed'):
        session.execute(gate,revalidate=lambda:gate)
    assert m.calls==[('servo',True),('servo',False)] and sleeps==[.005]*5
    assert not session.command_attempted and not session.command_accepted and session.servo_disabled


def test_servo_disable_timeout_records_accepted_command():
    class StuckEnabled(Motion):
        def setServoMode(self,value):
            self.calls.append(('servo',value))
            if value:self.enabled=True
            return 0
    m=StuckEnabled();gate=check_step(**inputs())
    session=SingleStepSession(m,sleep=lambda _:None)
    with pytest.raises(RuntimeError,match='remains enabled'):
        session.execute(gate,revalidate=lambda:gate)
    assert session.command_attempted and session.command_accepted and not session.servo_disabled
    assert sum(c[0]=='send' for c in m.calls)==1


@pytest.mark.parametrize('ret',[2,-13,-1,True,None])
def test_failed_send_stops_never_retries(ret):
    m=Motion(ret);gate=check_step(**inputs())
    with pytest.raises(RuntimeError):SingleStepSession(m,sleep=lambda s:None).execute(gate,revalidate=lambda:gate)
    assert sum(c[0]=='send' for c in m.calls)==1
    assert m.calls==[('servo',True),m.calls[1],('servo',False)]


def test_rejected_never_arms():
    m=Motion()
    with pytest.raises(ValueError):SingleStepSession(m).execute({'passed':False},revalidate=lambda:None)
    assert not m.calls


def test_expires_during_arm_no_send():
    m=Motion();gate=check_step(**inputs())
    args=inputs();args['now']=1.2
    with pytest.raises(ValueError):SingleStepSession(m).execute(gate,revalidate=lambda:check_step(**args))
    assert m.calls==[('servo',True),('servo',False)]


def test_existing_owner_not_touched():
    m=Motion();m.enabled=True;gate=check_step(**inputs())
    with pytest.raises(RuntimeError,match='owner'):SingleStepSession(m).execute(gate,revalidate=lambda:gate)
    assert not m.calls


def test_interrupted_after_send_stops():
    def interrupt(_):raise KeyboardInterrupt()
    m=Motion();gate=check_step(**inputs())
    with pytest.raises(KeyboardInterrupt):SingleStepSession(m,sleep=interrupt).execute(gate,revalidate=lambda:gate)
    assert m.calls[-1]==('servo',False)


def load_entry():
    path=Path(__file__).resolve().parents[2]/'examples/phone_to_auboi10/run_joint_trial.py'
    spec=importlib.util.spec_from_file_location('joint_trial_entry',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_plan_never_connects(monkeypatch):
    module=load_entry()
    def fail(*args):raise AssertionError('hardware or inference access in plan')
    monkeypatch.setattr(module,'ReadOnlyStation',fail);monkeypatch.setattr(module,'InferencePipe',fail)
    assert module.main([])==0


@pytest.mark.parametrize('requested,available,bad_chunk',[
    (False,False,False),(True,False,False),(True,True,False),(True,True,True),
])
def test_full_chunk_pipe_negotiation_and_validation(tmp_path,monkeypatch,requested,available,bad_chunk):
    import io
    module=load_entry();launch=[]
    class Process:
        def __init__(self,args,**kwargs):
            launch.extend(args);self.stdin=io.StringIO();self.waited=False
        def wait(self,timeout):self.waited=True
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    ready={'ready':True,'checkpoint_sha256':module.MODEL_SHA,
        'contract':module.joint_contract_record(),'action_chunk_available':available}
    chunk=[[*START_DEG,0.] for _ in range(50)]
    response={'id':3,'checkpoint_sha256':module.MODEL_SHA,'shape':[50,7],
        'action':chunk[0],'raw_action':chunk[0]}
    if available:
        response.update(action_chunk=[row.copy() for row in chunk],raw_action_chunk=chunk)
        if bad_chunk:response['action_chunk'][9][-1]=100.
    responses=iter([ready,response])
    monkeypatch.setattr(module.InferencePipe,'receive',lambda self,timeout:next(responses))
    if requested and not available:
        with pytest.raises(ValueError,match='does not provide full actions'):
            module.InferencePipe(tmp_path,return_action_chunk=requested)
        return
    pipe=module.InferencePipe(tmp_path,return_action_chunk=requested)
    assert ('--return-action-chunk' in launch[-1])==requested
    try:
        if bad_chunk:
            with pytest.raises(ValueError,match='inconsistent decoded'):
                pipe.predict(3,[*START_DEG,0],{})
        else:assert pipe.predict(3,[*START_DEG,0],{})['action']==chunk[0]
    finally:pipe.close()
    assert pipe.process.waited and pipe.log.closed


def test_failed_shadow_never_gets_motion(tmp_path,monkeypatch):
    import json
    import numpy as np
    module=load_entry();closed=[]
    class Station:
        lower=[-360]*6;upper=[360]*6
        @property
        def iface(self):raise AssertionError('shadow accessed actuator interface')
        def connect(self):pass
        def observe(self):return [*START_DEG,0],{'global_rgb':np.zeros((480,640,3),dtype=np.uint8)},[1,1,1]
        def gate(self,*args):return {'passed':False,'reasons':['expired_observation']}
        def close(self):closed.append('station')
    class Pipe:
        def __init__(self,*args):pass
        def predict(self,*args):return {'action':[*START_DEG,0],'inference_s':.2}
        def close(self):closed.append('pipe')
    monkeypatch.setattr(module,'ReadOnlyStation',Station);monkeypatch.setattr(module,'InferencePipe',Pipe)
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    output=tmp_path/'run'
    assert module.main(['--stage','shadow','--output',str(output),'--onsite-confirmed'])==1
    assert closed==['station','pipe']
    result=json.loads((output/'complete.json').read_text())
    assert result['samples']==12 and not result['passed']


def test_single_step_requires_shadow_before_connect(tmp_path,monkeypatch):
    module=load_entry()
    def fail(*args):raise AssertionError('hardware touched')
    monkeypatch.setattr(module,'ReadOnlyStation',fail)
    with pytest.raises(SystemExit):module.main(['--stage','single-step','--output',str(tmp_path/'run'),'--onsite-confirmed'])


@pytest.mark.parametrize('codec', ['rgb_zlib', 'jpeg95'])
def test_image_transport_preserves_rgb_roles(codec):
    import numpy as np
    entry = load_entry()
    path = Path(__file__).resolve().parents[2]/'examples/phone_to_auboi10/joint_inference_stdio.py'
    spec = importlib.util.spec_from_file_location('joint_server', path)
    server = importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
    red = np.full((480,640,3), [255,0,0], dtype=np.uint8)
    blue = np.full((480,640,3), [0,0,255], dtype=np.uint8)
    inputs = {'global_rgb':red, 'grasp_rgb':blue}
    decoded = server.decode_images(entry.encode_images(inputs, codec), codec)
    for name, expected in inputs.items():
        actual = decoded[f'observation.images.{name}']
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert np.max(np.abs(actual.astype(int)-expected.astype(int))) <= (1 if codec == 'jpeg95' else 0)


def stationary_inputs():
    args=inputs()
    args.update(captured_at=1.02, observed_state=list(args['current']), now=1.2, state_time=1.19)
    return args


def test_stationary_timing_requires_unchanged_state():
    result=check_step(**stationary_inputs())
    assert result['passed'] and result['timing_mode']=='stationary_single_step'
    assert result['observation_age_s']==pytest.approx(.2)


@pytest.mark.parametrize('changes,reason',[
    ({'sensor_times':[.8,1,1]},'expired_at_capture'),
    ({'now':1.271,'state_time':1.26},'stationary_prediction_timeout'),
    ({'captured_at':.99},'invalid_capture_clock'),
    ({'observed_state':[START_DEG[0]+.03,*START_DEG[1:],0]},'state_changed_during_prediction'),
    ({'observed_state':[*START_DEG,100]},'state_changed_during_prediction'),
    ({'action':[START_DEG[0]+.3,*START_DEG[1:],0]},'tracking distance'),
])
def test_stationary_mode_still_rejects(changes,reason):
    args=stationary_inputs();args.update(changes)
    result=check_step(**args)
    assert not result['passed'] and any(reason in s for s in result['reasons'])


class HomeMotion(Motion):
    fraction=.25
    def getSpeedFraction(self):return self.fraction
    def setSpeedFraction(self,value):self.calls.append(('fraction',value));self.fraction=value;return 0
    def moveJoint(self,*args):self.calls.append(('move',args));return self.ret
    def stopJoint(self,*args):self.calls.append(('stop',args));return 0


def test_home_fixed_target_no_io_or_servo():
    m=HomeMotion()
    states=iter([[START_DEG[0]+1,*START_DEG[1:],0],[*START_DEG,0]])
    result=home_to_start(m,lambda:next(states),lower=[-360]*6,upper=[360]*6,is_steady=lambda:True)
    assert result['moved'] and len(m.calls)==3 and m.fraction==.25
    assert m.calls[0]==('fraction',.5) and m.calls[-1]==('fraction',.25)
    assert m.calls[1][1][0]==pytest.approx([math.radians(q) for q in START_DEG])
    assert m.calls[1][1][1:3]==pytest.approx([math.radians(80),math.radians(60)])


@pytest.mark.parametrize('state',[[361,*START_DEG[1:],0],[*START_DEG,100]])
def test_home_rejects_outside_joint_limits_or_holding(state):
    m=HomeMotion()
    with pytest.raises(ValueError):home_to_start(m,lambda:state,lower=[-360]*6,upper=[360]*6,is_steady=lambda:True)
    assert not m.calls


@pytest.mark.parametrize('distance',[45.,200.])
def test_home_from_work_area_uses_full_trajectory_and_distance_timeout(distance):
    m=HomeMotion();initial=[START_DEG[0]+distance,*START_DEG[1:],0]
    states=iter([initial,[*START_DEG,0]])
    result=home_to_start(m,lambda:next(states),lower=[-360]*6,upper=[360]*6,is_steady=lambda:True)
    assert result['moved'] and result['initial_max_distance_deg']==pytest.approx(distance)
    assert result['timeout_s']==max(30.,distance/30.+15.)
    assert [c[0] for c in m.calls]==['fraction','move','fraction'] and m.fraction==.25


def test_home_error_stops_no_retry():
    m=HomeMotion(-13)
    with pytest.raises(RuntimeError):home_to_start(m,lambda:[START_DEG[0]+1,*START_DEG[1:],0],
        lower=[-360]*6,upper=[360]*6,is_steady=lambda:True)
    assert [c[0] for c in m.calls]==['fraction','move','stop','fraction'] and m.fraction==.25


def test_home_timeout_stops():
    m=HomeMotion();clock=iter([0.,31.])
    with pytest.raises(TimeoutError):home_to_start(m,lambda:[START_DEG[0]+1,*START_DEG[1:],0],
        lower=[-360]*6,upper=[360]*6,is_steady=lambda:True,clock=lambda:next(clock))
    assert [c[0] for c in m.calls]==['fraction','move','stop','fraction'] and m.fraction==.25


def test_home_entry_no_policy_or_cameras(tmp_path,monkeypatch):
    module=load_entry();events=[]
    monkeypatch.chdir(tmp_path)
    class State:
        def isSteady(self):return True
    class Station:
        lower=[-360]*6;upper=[360]*6;state=State()
        def __init__(self):self.iface=self;self.m=HomeMotion()
        def connect(self,*,include_cameras):
            assert not include_cameras;events.append('connect')
        def getMotionControl(self):return self.m
        def current(self,**kwargs):return [*START_DEG,0],None,None
        def close(self):events.append('close')
    def fail(*args):raise AssertionError('homing loaded inference')
    monkeypatch.setattr(module,'ReadOnlyStation',Station)
    monkeypatch.setattr(module,'InferencePipe',fail)
    assert module.main(['--stage','home','--onsite-confirmed'])==0
    assert events==['connect','close']
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('pins,release_ok,servo_owned,expected_release,expected_home,release_option',[
    ((False,False),True,False,True,True,'--release-if-unset'),
    ((False,False),False,False,True,False,'--release-if-unset'),
    ((False,True),True,False,False,True,'--release-if-unset'),
    ((True,False),True,False,False,False,'--release-if-unset'),
    ((True,True),True,False,False,False,'--release-if-unset'),
    ((False,False),True,True,False,False,'--release-if-unset'),
    ((True,False),True,False,True,True,'--release-before-home'),
    ((True,False),False,False,True,False,'--release-before-home'),
    ((False,False),True,False,True,True,'--release-before-home'),
    ((True,False),True,True,False,False,'--release-before-home'),
    ((False,True),True,False,False,True,None),
    ((True,False),True,False,True,True,None),
    ((False,False),True,False,True,True,None),
    ((True,True),True,False,True,True,None),
    ((True,False),False,False,True,False,None),
])
@pytest.mark.parametrize('write_logs',[False,True])
def test_home_release_if_unset_before_motion(tmp_path,monkeypatch,pins,release_ok,
                                           servo_owned,expected_release,expected_home,release_option,write_logs):
    import json
    from lerobot.robots.aubo_i10 import aubo_i10
    module=load_entry();events=[];outputs=[pins]
    monkeypatch.chdir(tmp_path)
    class Station:
        lower=[-360]*6;upper=[360]*6
        def __init__(self):self.iface=self;self.state=self;self.io_readback=self
        def connect(self,*,include_cameras):assert not include_cameras
        def getMotionControl(self):return self
        def isServoModeEnabled(self):return servo_owned
        def isSteady(self):return True
        def current(self,**kwargs):
            if outputs[0]==(False,True):suction=0
            elif outputs[0]==(True,False):suction=100
            else:raise ValueError(f'unknown suction output pair: {outputs[0]}')
            return [*START_DEG,suction],None,None
        def close(self):events.append('close')
    class Suction:
        last_gripper_command_trace=None
        def __init__(self,*args):pass
        def suction_release(self):
            events.append('release')
            self.last_gripper_command_trace={'do_api_success':release_ok}
            if release_ok:outputs[0]=(False,True)
            return release_ok
    original_home=module.home_to_start
    def home(*args,**kwargs):
        result=original_home(*args,**kwargs);events.append('home');return result
    def fail(*args,**kwargs):raise AssertionError('home loaded inference')
    monkeypatch.setattr(module,'ReadOnlyStation',Station)
    monkeypatch.setattr(module,'InferencePipe',fail)
    monkeypatch.setattr(module,'home_to_start',home)
    monkeypatch.setattr(aubo_i10,'AuboI10Robot',Suction)
    def no_prompt(*args):raise AssertionError('home must not request interactive confirmation')
    monkeypatch.setattr(module.sys.stdin,'isatty',lambda:False)
    monkeypatch.setattr('builtins.input',no_prompt)
    out=tmp_path/'home'
    args=['--stage','home','--onsite-confirmed']
    if release_option:args.append(release_option)
    if write_logs:args.extend(['--output',str(out)])
    assert module.main(args)==(0 if expected_home else 1)
    assert ('release' in events)==expected_release
    assert ('home' in events)==expected_home
    if expected_release and expected_home:assert events.index('release')<events.index('home')
    if write_logs:
        assert (out/'initial_suction_release.json').exists()==expected_release
        result=json.loads((out/'complete.json').read_text())
        assert result['passed']==expected_home
    else:assert list(tmp_path.iterdir())==[]
    assert events[-1]=='close'


def test_static_step_duration_preserves_velocity_limits_and_start_envelope():
    limits=stationary_step_limits()
    assert limits.max_joint_speed_deg_s==TrialLimits().max_joint_speed_deg_s
    assert limits.max_tcp_speed_m_s==TrialLimits().max_tcp_speed_m_s
    args=stationary_inputs()
    args.update(limits=limits, action=[START_DEG[0]+.4,*START_DEG[1:],0],target_tcp=[.404,-.4,.4])
    gate=check_step(**args)
    assert gate['passed']
    sleeps=[];m=Motion()
    SingleStepSession(m,limits=limits,sleep=sleeps.append).execute(gate,revalidate=lambda:gate)
    assert m.calls[1][1][3]==.2 and sleeps==[.2]
    args['action'][0]=START_DEG[0]+.6
    assert 'target_outside_start_envelope' in check_step(**args)['reasons']


def test_wait_until_stationary_and_timeout():
    from lerobot.bamboo_sorting.joint_trial import wait_until_stationary
    now=[0.];reads=[]
    def read():reads.append(len(reads));return reads[-1]
    def sleep(dt):now[0]+=dt
    assert wait_until_stationary(read,lambda:len(reads)==3,clock=lambda:now[0],sleep=sleep)==2
    with pytest.raises(TimeoutError,match='settle'):
        wait_until_stationary(read,lambda:False,clock=lambda:now[0],sleep=sleep)
    assert 1 <= now[0] < 1.1


@pytest.mark.parametrize('reject_at,expected_sends',[(None,3),(1,1)])
def test_short_loop_new_observations_and_stop_on_rejection(tmp_path,monkeypatch,reject_at,expected_sends):
    from lerobot.bamboo_sorting.joint_trial import short_loop_limits
    import hashlib,json,time
    from dataclasses import asdict
    import numpy as np
    module=load_entry();m=Motion();observed=[];predicted=[]
    class Station:
        lower=[-360]*6;upper=[360]*6
        def __init__(self):self.iface=self;self.state=self;self.q=[*START_DEG,0]
        def connect(self):pass
        def close(self):pass
        def isSteady(self):return not m.enabled
        def getMotionControl(self):return m
        def current(self,**kwargs):
            if sum(c[0]=='send' for c in m.calls):
                self.q[0]=START_DEG[0]+.05*sum(c[0]=='send' for c in m.calls)
            return self.q.copy(),[0,0,0],time.perf_counter()
        def observe(self):
            assert not m.enabled
            state=self.current()[0];observed.append(state.copy())
            return state,{'global_rgb':np.zeros((480,640,3),dtype=np.uint8)},[time.perf_counter()]*3
        def gate(self,action,*args,**kwargs):
            passed=len(predicted)-1!=reject_at
            return {'passed':passed,'action':action,'reasons':[] if passed else ['test_rejection']}
    class Pipe:
        def __init__(self,*args):pass
        def close(self):pass
        def predict(self,index,state,images):
            predicted.append((index,state.copy()))
            return {'action':[state[0]+.05,*state[1:]],'inference_s':.01}
    monkeypatch.setattr(module,'ReadOnlyStation',Station);monkeypatch.setattr(module,'InferencePipe',Pipe)
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    shadow=tmp_path/'shadow.json'
    shadow.write_text(json.dumps({'stage':'short-shadow','completed':True,'gate_pass_count':12,'samples':12,
        'checkpoint_sha256':module.MODEL_SHA,'camera_set_sha256':hashlib.sha256(module.CAMERA_SET.read_bytes()).hexdigest(),
        'transport_codec':'jpeg95','fast_matmul':False,'timing_mode':'stationary_single_step',
        'limits':asdict(short_loop_limits()),'completed_at_unix':time.time()}))
    out=tmp_path/'loop'
    status=module.main(['--stage','short-loop','--stationary-step','--onsite-confirmed','--shadow-report',str(shadow),'--output',str(out)])
    assert status==(0 if reject_at is None else 1)
    assert sum(c[0]=='send' for c in m.calls)==expected_sends and not m.enabled
    assert len(observed)==(3 if reject_at is None else 2)
    assert observed[1][0]==pytest.approx(START_DEG[0]+.05)
    report=json.loads((out/'complete.json').read_text())
    assert report['completed']==(reject_at is None)
    for index in range(expected_sends):
        assert json.loads((out/f'step_{index:02d}.json').read_text())['servo_disabled']


def test_short_loop_separate_envelope_preserves_step_limits():
    from dataclasses import asdict
    from lerobot.bamboo_sorting.joint_trial import short_loop_limits
    short=short_loop_limits();single=stationary_step_limits()
    expected=asdict(single);expected['start_tolerance_deg']=1.5
    assert asdict(short)==expected
    args=stationary_inputs()
    args['current'][0]+=.4;args['observed_state'][0]+=.4;args['action'][0]+=.6
    assert check_step(**args,limits=short)['passed']
    assert not check_step(**args,limits=single)['passed']
    args['action'][0]=START_DEG[0]+1.51
    assert 'target_outside_start_envelope' in check_step(**args,limits=short)['reasons']


def test_full_episode_retains_workspace_and_step_checks():
    args=stationary_inputs()
    args['current'][0]+=10;args['observed_state'][0]+=10;args['action'][0]+=10
    args['action'][-1]=100
    limits=stationary_step_limits()
    assert check_step(**args,limits=limits,full_episode=True)['passed']
    assert not check_step(**args,limits=limits)['passed']
    args['target_tcp'][2]=-.001
    assert 'outside_capture_workspace' in check_step(**args,limits=limits,full_episode=True)['reasons']
    args['action'][0]+=2
    assert any('tracking distance' in r for r in check_step(**args,limits=limits,full_episode=True)['reasons'])


def test_manual_live_episode_model_suction_and_exit(tmp_path,monkeypatch):
    import json,time
    import numpy as np
    directory=Path(__file__).resolve().parents[2]/'examples/phone_to_auboi10'
    monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('joint_live_entry',directory/'run_joint_live.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from lerobot.robots.aubo_i10 import aubo_i10
    m=Motion();pins={'value':0};writes=[];seen=[]
    class Station:
        def __init__(self):self.iface=self;self.state=self;self.io_readback=pins
        def connect(self):pass
        def close(self):pass
        def isSteady(self):return not m.enabled
        def getMotionControl(self):return m
        def current(self,**kwargs):return [*START_DEG,pins['value']],[.4,-.4,.4],time.perf_counter()
        def observe(self):return self.current()[0],{'global_rgb':np.zeros((480,640,3),dtype=np.uint8)},[time.perf_counter()]*3
        def gate(self,action,*args,**kwargs):
            assert kwargs['full_episode'] is True
            return {'passed':True,'action':action,'reasons':[]}
    class Pipe:
        def __init__(self,*args,task):assert task==module.JOINT_TASK
        def close(self):pass
        def predict(self,index,state,images):
            seen.append(state[-1]);return {'action':[*START_DEG,100 if index==0 else 0]}
    class Suction:
        last_gripper_command_trace=None
        def __init__(self,*args):pass
        def _control_suction_based_on_gripper(self,value):
            writes.append(value);pins['value']=value;self.last_gripper_command_trace={'target':value}
    monkeypatch.setattr(module,'ReadOnlyStation',Station);monkeypatch.setattr(module,'InferencePipe',Pipe)
    monkeypatch.setattr(module,'SingleStepSession',lambda motion,limits:SingleStepSession(motion,limits=limits,sleep=lambda _:None))
    monkeypatch.setattr(aubo_i10,'AuboI10Robot',Suction)
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    monkeypatch.setattr(module.sys.stdin,'isatty',lambda:True)
    answers=iter(['', 'different task', module.JOINT_TASK, ''])
    monkeypatch.setattr('builtins.input',lambda _: next(answers))
    out=tmp_path/'live'
    assert module.main(['--execute','--output',str(out)])==0
    assert writes==[100,0] and seen==[0,100]
    assert sum(c[0]=='send' for c in m.calls)==2 and not m.enabled
    result=json.loads((out/'complete.json').read_text())
    assert result['suction_cycle_commanded'] and result['physical_grasp_success'] is None
    assert json.loads((out/'plan.json').read_text())['task']==module.JOINT_TASK
    assert all(json.loads(line)['task']==module.JOINT_TASK for line in (out/'trace.jsonl').read_text().splitlines())


def test_current_timestamp_starts_at_joint_read_and_no_unused_speed_rpc(monkeypatch):
    from types import SimpleNamespace
    m=load_entry();clock=[10.];calls=[]
    monkeypatch.setattr(m.time,'perf_counter',lambda:clock[0])
    class State:
        def isPowerOn(self):return True
        def isWithinSafetyLimits(self):return True
        def isCollisionOccurred(self):return False
        def getSafetyModeType(self):clock[0]+=.2;return SimpleNamespace(name='Normal')
        def getJointPositions(self):calls.append(clock[0]);return [math.radians(x) for x in START_DEG]
        def getJointSpeeds(self):raise AssertionError('unneeded velocity RPC')
        def getTcpPose(self):return [.2,-.5,.3,0,0,0]
    s=m.ReadOnlyStation();s.state=State()
    s.io_readback=SimpleNamespace(getStandardDigitalOutput=lambda pin:pin==3)
    q,tcp,stamp=s.current(require_stationary=False)
    assert stamp==calls[0]==10.2 and q==pytest.approx([*START_DEG,0])


@pytest.mark.parametrize('max_camera_age_ms',[100,250])
def test_observe_reuses_state_without_refreshing_its_timestamp(max_camera_age_ms):
    import numpy as np
    from types import SimpleNamespace
    m=load_entry();s=m.ReadOnlyStation();ages=[]
    def read(**kw):
        ages.append(kw['max_age_ms'])
        return np.zeros((480,640,3),dtype=np.uint8),10.02
    s.cameras={name:SimpleNamespace(read_latest_with_timestamp=read) for name in ('global_rgb','grasp_rgb')}
    def fail(**kwargs):raise AssertionError('duplicate robot RPC')
    s.current=fail
    snapshot=([*START_DEG,0],[.2,-.5,.3],10.)
    state,images,stamps=s.observe(require_stationary=False,state_snapshot=snapshot,
                                  max_camera_age_ms=max_camera_age_ms)
    assert state==snapshot[0] and stamps==[10.02,10.02,10.]
    assert ages==[max_camera_age_ms,max_camera_age_ms]

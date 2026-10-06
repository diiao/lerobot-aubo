#!/usr/bin/env python
"""Plan, read-only prediction, or ONE bounded joint command; never full grasp.

No hardware/model imports in plan mode. Shadow obtains state/config/algorithm
and IO-readback interfaces, calls no motion API and no digital-output setter.
Only explicit home/single-step modes may obtain MotionControl. Homing ensures
suction is released before moving; already released outputs are left unchanged.
No motion retries. Homing never loads a policy or opens cameras.
"""
import argparse
import base64
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import selectors
import shlex
import subprocess
import sys
import time
import zlib

from lerobot.bamboo_sorting.aubo_joint_contract import finite_vector, joint_contract_record
from lerobot.bamboo_sorting.joint_smooth import validated_action_chunk
from lerobot.bamboo_sorting.joint_trial import TrialLimits, check_step, SingleStepSession, home_to_start, stationary_step_limits, short_loop_limits, wait_until_stationary

REMOTE_ROOT='/home/rentao/program/lerobot-aubo-smolvla-joint-legacy-v2-20260922'
REMOTE_PYTHON='/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv/bin/python'
MODEL_SHA='95a596e2a98dbe60623f801b2d12a6028132ad321a0bcb2ac7ae334ef5613aad'
REPO=Path(__file__).resolve().parents[2]
CAMERA_SET=REPO/'configs/aubo_i10/CameraSetV2.json'
RPC_TIMEOUT_MS=300


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def encode_images(images, codec):
    """Keep RGB order explicit; legacy lossless encoding remains for comparison."""
    import cv2
    encoded = {}
    for name, image in images.items():
        if image.shape != (480, 640, 3) or str(image.dtype) != 'uint8':
            raise ValueError('expected RGB uint8 [480,640,3]')
        if codec == 'rgb_zlib':
            payload = zlib.compress(image.tobytes(), 1)
        elif codec == 'jpeg95':
            ok, payload = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError('JPEG encode failed')
        else:
            raise ValueError('unsupported image codec')
        encoded[name] = base64.b64encode(payload).decode()
    return encoded


class InferencePipe:
    def __init__(self, output, *, fast_matmul=False, bf16=False, cuda_graph=False, task=None,
                 remote_root=REMOTE_ROOT, checkpoint=None, expected_sha256=MODEL_SHA,
                 inference_script=None,return_action_chunk=False,request_timeout_s=2.):
        if not math.isfinite(request_timeout_s) or request_timeout_s<=0:
            raise ValueError('inference request timeout must be finite and positive')
        self.request_timeout_s=request_timeout_s
        self.log=(output/'inference_stderr.log').open('w')
        self.expected_sha256=expected_sha256
        self.return_action_chunk=return_action_chunk
        deployment = 'joint_live_bf16_20260922' if bf16 else 'joint_live_matmul_20260922'
        script = 'joint_inference_stdio.py'
        if cuda_graph:
            deployment, script = 'joint_live_graph_20260922', 'graph_inference_stdio.py'
        if task is not None:
            deployment, script = 'joint_live_manual_task_20260922', 'joint_inference_stdio.py'
        script_path=inference_script or f'{remote_root}/artifacts/{deployment}/{script}'
        checkpoint_path=checkpoint or f'{remote_root}/outputs/joint_legacy_v2_run01/final'
        args=['env',f'PYTHONPATH={remote_root}/src','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
            'HF_HUB_DISABLE_TELEMETRY=1','WANDB_MODE=disabled',REMOTE_PYTHON,'-u',
            str(script_path), '--checkpoint',str(checkpoint_path),
            '--expected-sha256',self.expected_sha256]
        if fast_matmul:
            args.append('--fast-matmul')
        if bf16:
            args.append('--bf16')
        if return_action_chunk:
            args.append('--return-action-chunk')
        if task is not None:
            args.extend(['--task',task])
        self.process=subprocess.Popen(['ssh','-T','gpu',shlex.join(args)],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        try:
            ready=self.receive(120)
            if ready.get('ready') is not True or ready.get('checkpoint_sha256')!=self.expected_sha256 or ready.get('contract')!=joint_contract_record():
                raise ValueError('inference identity mismatch')
            if bool(ready.get('bf16',False)) != bf16:
                raise ValueError('inference precision mismatch')
            if task is not None and ready.get('task') != task:
                raise ValueError('inference task mismatch')
            if return_action_chunk and ready.get('action_chunk_available') is not True:
                raise ValueError('inference worker does not provide full actions')
        except BaseException:
            self.close()
            raise

    def receive(self,timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout,selectors.EVENT_READ)
            if not selector.select(timeout):raise TimeoutError('inference pipe timeout')
        line=self.process.stdout.readline(100000)
        if not line:raise RuntimeError('inference exited; inspect inference_stderr.log')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        return result

    def predict(self, index, state, images, *, codec='jpeg95', seed=None):
        start = time.perf_counter()
        packet={'id':index,'state':state,'codec':codec,'images':encode_images(images, codec)}
        if seed is not None:
            packet['seed'] = seed  # Offline paired comparison only; live requests omit it.
        encoded_at = time.perf_counter()
        wire = json.dumps(packet)+'\n'
        self.process.stdin.write(wire);self.process.stdin.flush()
        result=self.receive(self.request_timeout_s)
        if result.get('id')!=index or result.get('checkpoint_sha256')!=self.expected_sha256 or result.get('shape')!=[50,7]:
            raise ValueError('invalid prediction identity or shape')
        if self.return_action_chunk:
            # Validate the full wire response once, before any consumer uses it.
            result['action_chunk']=validated_action_chunk(result)
        else:
            finite_vector(result['raw_action'],7,'raw model action')
            finite_vector(result['action'],7,'decoded model action')
        result['transport'] = {'codec':codec, 'request_bytes':len(wire.encode()),
            'encode_s':encoded_at-start, 'roundtrip_s':time.perf_counter()-start}
        return result

    def close(self):
        # EOF ends only our stdio worker. Never pkill/kill a pre-existing task.
        if self.process.stdin and not self.process.stdin.closed:self.process.stdin.close()
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.log.close()


class ReadOnlyStation:
    def __init__(self):
        self.rpc=None;self.cameras={};self.previous={}

    def connect(self, *, include_cameras=True):
        import pyaubo_sdk
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.bamboo_sorting.c0_smoke_capture import frozen_c0_smoke_camera_mapping
        mapping=frozen_c0_smoke_camera_mapping()
        for name,p in (mapping.items() if include_cameras else []):
            cam=OpenCVCamera(OpenCVCameraConfig(index_or_path=p['device'],width=640,height=480,
                fps=30 if name=='global_rgb' else 25,fourcc='MJPG',warmup_s=3))
            self.cameras[name]=cam;cam.connect(warmup=True);cam.async_read_with_timestamp()
        self.rpc=pyaubo_sdk.RpcClient();self.rpc.setRequestTimeout(RPC_TIMEOUT_MS)
        self.rpc.connect('192.168.31.200',30004)
        if not self.rpc.hasConnected():raise ConnectionError('AUBO not connected')
        self.rpc.login('aubo','123456')
        if not self.rpc.hasLogined():raise ConnectionError('AUBO login failed')
        names=self.rpc.getRobotNames()
        if len(names)!=1:raise ValueError('ambiguous robot identity')
        self.iface=self.rpc.getRobotInterface(names[0])
        self.state=self.iface.getRobotState()
        self.config=self.iface.getRobotConfig()
        self.algorithm=self.iface.getRobotAlgorithm()
        # SDK exposes reads and writes on the same object; only getters below.
        self.io_readback=self.iface.getIoControl()
        self.lower=[math.degrees(v) for v in finite_vector(self.config.getJointMinPositions(),6,'lower')]
        self.upper=[math.degrees(v) for v in finite_vector(self.config.getJointMaxPositions(),6,'upper')]

    def current(self, *, require_stationary=True):
        if not self.state.isPowerOn() or not self.state.isWithinSafetyLimits() or self.state.isCollisionOccurred():
            raise RuntimeError('robot is not in a safe powered state')
        safety=self.state.getSafetyModeType()
        if getattr(safety,'name',str(safety).split('.')[-1])!='Normal':raise RuntimeError(f'safety mode {safety}')
        ts=time.perf_counter()  # Timestamp the joint read, not earlier safety RPCs.
        q=[math.degrees(v) for v in finite_vector(self.state.getJointPositions(),6,'SDK joints')]
        if require_stationary:
            speeds=finite_vector(self.state.getJointSpeeds(),6,'SDK speeds')
            if max(abs(v) for v in speeds)>math.radians(.1):raise RuntimeError('robot moving: stationary trial required')
        tcp=finite_vector(self.state.getTcpPose(),6,'SDK TCP')
        pins=(self.io_readback.getStandardDigitalOutput(2),self.io_readback.getStandardDigitalOutput(3))
        if pins==(True,False):suction=100.
        elif pins==(False,True):suction=0.
        else:raise ValueError(f'unknown suction output pair: {pins}')
        return q+[suction],tcp[:3],ts

    def observe(self, *, require_stationary=True, state_snapshot=None, max_camera_age_ms=100):
        if state_snapshot is not None and require_stationary:
            raise ValueError('state reuse only applies to moving observations')
        images={};times=[]
        for name in ('global_rgb','grasp_rgb'):
            image,stamp=self.cameras[name].read_latest_with_timestamp(max_age_ms=max_camera_age_ms)
            if stamp<=self.previous.get(name,-1):raise ValueError(f'repeated camera frame: {name}')
            if image.shape!=(480,640,3) or str(image.dtype)!='uint8':raise ValueError('camera schema mismatch')
            images[name]=image.copy();times.append(stamp);self.previous[name]=stamp
        state,tcp,stamp=state_snapshot if state_snapshot is not None else self.current(require_stationary=require_stationary)
        times.append(stamp)
        return state,images,times

    def gate(self,action,sensor_times, *, captured_at=None, observed_state=None, limits=TrialLimits(), full_episode=False):
        state,tcp,stamp=self.current()
        current_fk,ret=self.algorithm.forwardKinematics([math.radians(v) for v in state[:6]])
        if ret!=0 or math.dist(finite_vector(current_fk,6,'current FK')[:3],tcp)>.001:
            raise ValueError('FK/current TCP mismatch; tool frame not verified')
        target_fk,ret=self.algorithm.forwardKinematics([math.radians(v) for v in action[:6]])
        if ret!=0:raise ValueError('target FK failed')
        target_fk=finite_vector(target_fk,6,'target FK')
        return check_step(action,current=state,lower=self.lower,upper=self.upper,current_tcp=tcp,
            target_tcp=target_fk[:3],sensor_times=sensor_times,state_time=stamp,now=time.perf_counter(),
            captured_at=captured_at,observed_state=observed_state,limits=limits,full_episode=full_episode)

    def close(self):
        errors=[]
        for cam in self.cameras.values():
            try:
                if cam.is_connected:cam.disconnect()
            except Exception as exc:errors.append(str(exc))
        if self.rpc:
            try:
                if self.rpc.hasLogined():self.rpc.logout()
                self.rpc.disconnect()
            except Exception as exc:errors.append(str(exc))
        if errors:raise RuntimeError('cleanup: '+'; '.join(errors))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('plan','home','shadow','short-shadow','single-step','short-loop'),default='plan')
    p.add_argument('--output',type=Path)
    p.add_argument('--shadow-report',type=Path)
    p.add_argument('--onsite-confirmed',action='store_true')
    release=p.add_mutually_exclusive_group()
    release.add_argument('--release-if-unset',action='store_true',
                   help='Home only: send release if both suction outputs are low, without an input prompt')
    release.add_argument('--release-before-home',action='store_true',
                   help='Home default: ensure suction is released before homing, without an input prompt')
    p.add_argument('--stationary-step',action='store_true',
                   help='Explicit static-scene, unchanged-state single-step timing; not continuous execution')
    args=p.parse_args(argv)
    if args.stage=='home' and not args.release_if_unset:
        args.release_before_home=True
    release_requested=args.release_if_unset or args.release_before_home
    if release_requested and args.stage!='home':
        p.error('suction release options are only valid with --stage home')
    if args.stage in ('short-loop','short-shadow') and not args.stationary_step:
        p.error('short-loop requires --stationary-step (stop between observations)')
    expected_samples=12 if args.stage in ('shadow','short-shadow') else 3 if args.stage=='short-loop' else 1
    limits=short_loop_limits() if args.stage in ('short-loop','short-shadow') else stationary_step_limits() if args.stationary_step else TrialLimits()
    camera_sha=hashlib.sha256(CAMERA_SET.read_bytes()).hexdigest()
    plan={'stage':args.stage,'checkpoint_sha256':MODEL_SHA,'camera_set_sha256':camera_sha,
          'transport_codec':'jpeg95','fast_matmul':False,
          'timing_mode':'stationary_single_step' if args.stationary_step else 'continuous',
          'limits':asdict(limits),'contract':joint_contract_record(),
          'suction_writes':release_requested,'release_if_unset':args.release_if_unset,
          'release_before_home':args.release_before_home,
          'automatic_homing':False,'continuous_execution':False}
    plan['explicit_homing'] = args.stage=='home'
    plan['stop_between_steps'] = args.stage=='short-loop'
    plan['max_steps'] = expected_samples
    if args.stage=='plan':print(json.dumps(plan,indent=2));return 0
    if args.output is None and args.stage!='home':p.error('fresh --output required')
    if args.output is not None and args.output.exists():p.error('fresh --output required')
    if not args.onsite_confirmed:p.error('onsite confirmation required for live observation or motion')
    if args.stage in ('single-step','short-loop'):
        if not args.shadow_report:p.error('--shadow-report required')
        report=json.loads(args.shadow_report.read_text())
        eligible = report.get('passed') is True
        if args.stationary_step:
            eligible = report.get('completed') is True and report.get('gate_pass_count',0)>0
        required_preview='short-shadow' if args.stage=='short-loop' else 'shadow'
        if report.get('stage')!=required_preview or not eligible or report.get('samples')!=12:
            p.error('complete passing shadow report required')
        if report.get('checkpoint_sha256')!=MODEL_SHA or report.get('camera_set_sha256')!=camera_sha:
            p.error('shadow identity mismatch')
        if report.get('transport_codec') != 'jpeg95' or report.get('fast_matmul') is not False:
            p.error('shadow inference settings mismatch')
        if report.get('timing_mode') != plan['timing_mode']:
            p.error('shadow timing mode mismatch')
        if report.get('limits') != plan['limits']:
            p.error('shadow limits mismatch')
        age=time.time()-report.get('completed_at_unix',0)
        if not 0<=age<=120:p.error('shadow evidence expired (120 seconds)')
    if args.output is not None:
        args.output.mkdir(parents=True,exist_ok=False);write(args.output/'plan.json',plan)
    station=ReadOnlyStation();pipe=None;records=[];passed=False;failure=None
    try:
        if args.stage=='home':
            station.connect(include_cameras=False)
            if release_requested:
                if station.iface.getMotionControl().isServoModeEnabled():
                    raise RuntimeError('existing servo owner; refusing homing')
                should_release=False
                try:
                    state,_,_=station.current()
                    should_release=args.release_before_home and state[-1]!=0
                except ValueError as exc:
                    allowed={'unknown suction output pair: (False, False)'}
                    if args.release_before_home:allowed.add('unknown suction output pair: (True, True)')
                    if str(exc) not in allowed:raise
                    should_release=True
                if should_release:
                    print('发送夹爪释放指令，核对读回后归位。',flush=True)
                    from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot
                    from lerobot.robots.aubo_i10.config_aubo_i10 import AuboI10Config
                    suction=AuboI10Robot(AuboI10Config());suction.io_control=station.io_readback
                    try:
                        if not suction.suction_release():raise RuntimeError('initial suction release failed')
                    finally:
                        if args.output is not None:
                            write(args.output/'initial_suction_release.json',suction.last_gripper_command_trace)
                    state,_,_=station.current()
                    if state[-1]!=0:raise ValueError('release state not confirmed; refusing homing')
            def home_progress(event):
                if args.output is not None:
                    with (args.output/'home_progress.jsonl').open('a') as f:
                        f.write(json.dumps(event,allow_nan=False)+'\n')
            receipt=home_to_start(station.iface.getMotionControl(),
                lambda:station.current(require_stationary=False)[0],lower=station.lower,upper=station.upper,
                is_steady=station.state.isSteady,progress=home_progress)
            if args.output is not None:write(args.output/'home.json',receipt)
            passed=True
        else:
            pipe=InferencePipe(args.output)  # Load/warm GPU BEFORE connecting cameras/robot.
            station.connect()
            write(args.output/'controller_limits.json',{'lower':station.lower,'upper':station.upper})
            for index in range(expected_samples):
                if args.stage=='short-loop':
                    wait_until_stationary(lambda:station.current(require_stationary=False),station.state.isSteady)
                    settled_at=time.perf_counter()
                    time.sleep(.05)  # Both cameras must have a post-settle frame.
                start=time.perf_counter();state,images,timestamps=station.observe()
                if args.stage=='short-loop' and min(timestamps[:2]) < settled_at:
                    raise ValueError('camera frame predates settled state')
                captured_at=time.perf_counter()
                result=pipe.predict(index,state,images)
                timing={'captured_at':captured_at,'observed_state':state,'limits':limits} if args.stationary_step else {}
                gate=station.gate(result['action'],timestamps,**timing)
                record={'index':index,'state':state,'sensor_times':timestamps,'prediction':result,
                    'captured_at':captured_at,
                    'gate':gate,'total_s':time.perf_counter()-start}
                records.append(record)
                with (args.output/'trace.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                if args.stage in ('single-step','short-loop'):
                    if not gate['passed']:raise ValueError(f'pre-send rejection: {gate["reasons"]}')
                    session=SingleStepSession(station.iface.getMotionControl(),limits=limits)
                    try:
                        session.execute(gate,revalidate=lambda:station.gate(result['action'],timestamps,**timing))
                    finally:
                        receipt_name=f'step_{index:02d}.json' if args.stage=='short-loop' else 'single_step.json'
                        write(args.output/receipt_name,{'command_attempted':session.command_attempted,
                            'sent':session.command_accepted,'servo_disabled':session.servo_disabled,'suction_writes':False})
                    final_state,final_tcp,read_at=wait_until_stationary(
                        lambda:station.current(require_stationary=False),station.state.isSteady)
                    feedback_name=f'post_step_{index:02d}.json' if args.stage=='short-loop' else 'post_step_state.json'
                    write(args.output/feedback_name,{'before_state':state,'final_state':final_state,
                        'target':result['action'],'final_tcp':final_tcp,'read_at':read_at,
                        'joint_change_deg':[b-a for a,b in zip(state[:6],final_state[:6])],
                        'is_steady':station.state.isSteady()})
                if index==0 or args.stage=='short-loop':
                    import cv2
                    for name,image in images.items():
                        image_name=f'{index:02d}_{name}.png' if args.stage=='short-loop' else f'{name}.png'
                        if not cv2.imwrite(str(args.output/image_name),cv2.cvtColor(image,cv2.COLOR_RGB2BGR)):
                            raise RuntimeError('preview save failed')
                print(json.dumps({'index':index,'gate':gate,'inference_s':result['inference_s']}),flush=True)
                time.sleep(.05)
            passed=all(r['gate']['passed'] for r in records)
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}'
        print(failure,file=sys.stderr,flush=True)
    finally:
        for close in (station.close,pipe.close if pipe else lambda:None):
            try:close()
            except BaseException as exc:
                failure=f'{failure or ""}; cleanup {type(exc).__name__}: {exc}';passed=False
        summary={**plan,'passed':passed and failure is None,'samples':len(records),'failure':failure,
                 'completed':failure is None and (args.stage=='home' and passed or len(records)==expected_samples),
                 'gate_pass_count':sum(r['gate']['passed'] for r in records),
                 'completed_at_unix':time.time(),'physical_success_verified':False}
        if args.output is not None:write(args.output/'complete.json',summary)
        if args.stage=='home':print('归位完成' if summary['passed'] else '归位失败',flush=True)
    return 0 if summary['passed'] else 1

if __name__=='__main__':
    import signal
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    raise SystemExit(main())

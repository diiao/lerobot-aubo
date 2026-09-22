#!/usr/bin/env python
"""Manual 25-Hz joint servo with asynchronous policy inference; no auto start."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

from run_joint_live import read_task
from run_joint_trial import CAMERA_SET,MODEL_SHA,RPC_TIMEOUT_MS,InferencePipe,ReadOnlyStation,write
from lerobot.bamboo_sorting.aubo_joint_contract import finite_vector,require_joint_target
from lerobot.bamboo_sorting.joint_trial import START_DEG,SingleStepSession,wait_until_stationary
from lerobot.bamboo_sorting.joint_video import TrialVideoRecorder
from lerobot.bamboo_sorting.joint_smooth import (DT,MAX_SPEED,MAX_ACCEL,MAX_TRACKING_ERROR_DEG,
    MAX_TCP_SPEED,MAX_WITHOUT_PREDICTION,MAX_PREDICTION_AGE,MAX_CAMERA_SKEW,SmoothTrajectory,validate_prediction,workspace)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    plan={'mode':'asynchronous_inference_continuous_servo','checkpoint_sha256':MODEL_SHA,
          'camera_set_sha256':hashlib.sha256(CAMERA_SET.read_bytes()).hexdigest(),'control_hz':1/DT,
          'joint_speed_deg_s':MAX_SPEED,'joint_acceleration_deg_s2':MAX_ACCEL,'tcp_speed_m_s':MAX_TCP_SPEED,
          'max_prediction_delta_deg':None,'max_prediction_delta_m':None,
          'max_servo_tracking_error_deg':MAX_TRACKING_ERROR_DEG,
          'max_prediction_age_s':MAX_PREDICTION_AGE,'watchdog_s':MAX_WITHOUT_PREDICTION,
          'max_camera_skew_s':MAX_CAMERA_SKEW,
          'rpc_timeout_ms':RPC_TIMEOUT_MS,'max_control_gap_s':.35,
          'suction_writes':True,'max_seconds':120,'automatic_home':False,'automatic_release':False,
          'servo_disable_confirmation_timeout_s':1.0,'video_recording':{'enabled':True,'fps':25,'directory':'videos'}}
    if not args.execute:print(json.dumps(plan,indent=2));return 0
    if not sys.stdin.isatty():p.error('interactive terminal required')
    try:task=read_task()
    except (EOFError,KeyboardInterrupt):return 0
    if task is None:return 0
    plan['task']=task
    output=args.output or Path('artifacts')/('joint_smooth_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False);write(output/'plan.json',plan)
    station=ReadOnlyStation();pipe=None;pool=None;servo=None;owned=False;failure=None
    reason='not_started';sent=0;predictions=0;seen_on=False;cycle=False
    recorder=None;video_result=None
    trace=(output/'trace.jsonl').open('w',buffering=1)
    commands=(output/'commands.jsonl').open('w',buffering=1)
    def log(f,value):f.write(json.dumps(value,allow_nan=False)+'\n')
    try:
        print('加载模型、连接相机；尚未发送运动。',flush=True)
        pipe=InferencePipe(output,task=task);station.connect()
        state,tcp,_=station.current()
        if state[-1]!=0 or max(abs(a-b) for a,b in zip(state[:6],START_DEG))>1.5:
            raise ValueError('请先归位并关闭吸盘，再启动')
        motion=station.iface.getMotionControl();servo=SingleStepSession(motion)
        if motion.isServoModeEnabled():raise RuntimeError('existing servo owner')
        from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot
        from lerobot.robots.aubo_i10.config_aubo_i10 import AuboI10Config
        suction=AuboI10Robot(AuboI10Config());suction.io_control=station.io_readback
        def fk(q):
            pose,ret=station.algorithm.forwardKinematics([math.radians(x) for x in q])
            if ret!=0:raise ValueError(f'FK failed: {ret}')
            return list(finite_vector(pose,6,'FK')[:3])
        if math.dist(fk(state[:6]),tcp)>.001:raise ValueError('FK/current TCP mismatch')
        print(f'记录：{output.resolve()}\n持续伺服；Ctrl+C停止，停止不自动释放吸盘。',flush=True)
        if input('现场无人、急停在手边。按回车开始，输入q退出：').strip():
            reason='cancelled';return 0
        recorder=TrialVideoRecorder(station.cameras,output);recorder.start()
        print(f'双相机录像已启动：{output.resolve()}/videos/',flush=True)
        pool=ThreadPoolExecutor(max_workers=1)
        def infer(index,state,images):
            result=pipe.predict(index,state,images)
            if index%5==0:
                import cv2
                for name,img in images.items():
                    if not cv2.imwrite(str(output/f'{index:04d}_{name}.jpg'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR)):
                        raise RuntimeError('snapshot save failed')
            return result
        future=None;goal=None;pending_suction=False;last_observation=None;trajectory=None
        deadline=time.perf_counter()+120;next_tick=time.perf_counter();last_tick=next_tick
        prediction_deadline=None
        reason='time_limit';held_suction=0.
        while time.perf_counter()<deadline:
            now=time.perf_counter()
            if now<next_tick:time.sleep(next_tick-now)
            now=time.perf_counter()
            if owned and now-last_tick>.35:raise TimeoutError('servo loop delayed over 350 ms')
            last_tick=now;next_tick=now+DT
            state,tcp,stamp=station.current(require_stationary=False)
            if state[-1]!=held_suction:raise ValueError('external suction state change')
            workspace(tcp)
            if future is not None and future.done():
                result=future.result();future=None
                log(trace,{'index':predictions,'task':task,'observed_state':observed_state,
                    'sensor_times':sensor_times,'prediction':result,'received_at':time.perf_counter()})
                accepted=validate_prediction(result['action'],state,tcp,fk(result['action'][:6]),
                    station.lower,station.upper,sensor_times,time.perf_counter())
                predictions+=1;goal=accepted;last_observation=min(sensor_times)
                prediction_deadline=last_observation+MAX_WITHOUT_PREDICTION
                pending_suction=goal[-1]!=held_suction
                if not owned:
                    trajectory=SmoothTrajectory(state[:6]);owned=True
                    servo._require_zero(motion.setServoMode(True),'enable servo');servo._wait_servo(True)
            if goal is not None:
                if not pending_suction and time.perf_counter()>prediction_deadline:
                    raise TimeoutError('no fresh prediction within 500 ms')
                if pending_suction and time.perf_counter()-last_observation>1.5:
                    raise TimeoutError('suction transition target not reached within 1.5 seconds')
                q,target_tcp=trajectory.advance(goal[:6],fk)
                require_joint_target(q,state[:6],station.lower,station.upper,MAX_TRACKING_ERROR_DEG)
                if time.perf_counter()-stamp>.1:raise TimeoutError('current state expired')
                servo._require_zero(motion.servoJoint([math.radians(x) for x in q],math.radians(MAX_ACCEL),
                    math.radians(MAX_SPEED),DT,0.,200.),'servoJoint')
                sent+=1
                log(commands,{'time':time.perf_counter(),'state':state,'command':q,'velocity':trajectory.v,
                    'target':goal,'tcp':tcp,'command_tcp':target_tcp})
                if pending_suction and max(abs(a-b) for a,b in zip(goal[:6],state[:6]))<.15 and max(abs(v) for v in trajectory.v)<.3:
                    # Goal is held while switching IO; no averaging of discrete suction.
                    suction.is_suction_on=held_suction==100
                    try:suction._control_suction_based_on_gripper(goal[-1])
                    finally:log(trace,{'suction':suction.last_gripper_command_trace})
                    held_suction=goal[-1];pending_suction=False;seen_on=seen_on or held_suction==100
                    cycle=seen_on and held_suction==0
                    if cycle:reason='model_suction_cycle_completed';break
                    # IO readback can take 175 ms; do not catch up missed ticks.
                    last_tick=time.perf_counter();next_tick=last_tick+DT
                    # A separate deadline allows a post-IO observation; original
                    # sensor timestamps remain unchanged in prediction records.
                    prediction_deadline=last_tick+MAX_WITHOUT_PREDICTION
                    # IO changed after the cycle's read: the next observation
                    # must contain the new suction state, not the old snapshot.
                    state,tcp,stamp=station.current(require_stationary=False)
                if sent%25==0:print(f'已发送 {sent} 帧，预测 {predictions} 次，吸盘 {held_suction:.0f}',flush=True)
            if future is None and not pending_suction:
                observed_state,images,sensor_times=station.observe(require_stationary=False,state_snapshot=(state,tcp,stamp))
                if time.perf_counter()-min(sensor_times)>.1:raise ValueError('stale input at capture')
                future=pool.submit(infer,predictions,observed_state,images)
        if not owned:reason='no_motion'
    except KeyboardInterrupt:reason='operator_interrupt'
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}';reason='error';print(failure,file=sys.stderr,flush=True)
        (output/'error_traceback.txt').write_text(traceback.format_exc())
    finally:
        if owned:
            try:
                servo._require_zero(servo.motion.setServoMode(False),'disable servo');servo._wait_servo(False,timeout_s=1.)
            except BaseException as exc:failure=f'{failure or ""}; stop: {exc}'
        if getattr(station,'iface',None) is not None:
            try:
                q,tcp,stamp=wait_until_stationary(lambda:station.current(require_stationary=False),station.state.isSteady)
                write(output/'final_state.json',{'state':q,'tcp':tcp,'is_steady':station.state.isSteady(),
                    'servo_enabled':station.iface.getMotionControl().isServoModeEnabled()})
            except BaseException as exc:failure=f'{failure or ""}; final state: {exc}'
        if recorder is not None:
            try:
                video_result=recorder.close()
                if not video_result['complete']:print(f'录像不完整：{video_result["errors"]}',file=sys.stderr,flush=True)
            except BaseException as exc:
                video_result={'complete':False,'errors':[str(exc)]}
                print(f'录像收尾失败：{exc}',file=sys.stderr,flush=True)
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        for close in (station.close,pipe.close if pipe else lambda:None):
            try:close()
            except BaseException as exc:failure=f'{failure or ""}; cleanup: {exc}'
        trace.close();commands.close()
        write(output/'complete.json',{'reason':reason,'sent_frames':sent,'predictions':predictions,
            'failure':failure,'suction_cycle_commanded':cycle,'physical_grasp_success':None,'video':video_result})
        print(f'结束：{reason}，记录：{output.resolve()}',flush=True)
    return 1 if failure else 0


if __name__=='__main__':
    def terminate(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    raise SystemExit(main())

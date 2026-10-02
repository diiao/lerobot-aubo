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
    MAX_TCP_SPEED,MAX_WITHOUT_PREDICTION,MAX_PREDICTION_AGE,MAX_CAMERA_SKEW,SmoothTrajectory,validate_prediction,workspace,
    validated_action_chunk,select_approach_action)

MIXED_REMOTE_ROOT='/home/rentao/program/lerobot-aubo-smolvla-joint-mixed-20260927'
MIXED_MODEL_SHA='79eb23606f6aa4e77484dd4bbf16a8cd273bb6a17ccfc9668ef44a4e2ed22ac7'
MIXED_CHECKPOINT=f'{MIXED_REMOTE_ROOT}/outputs/joint_mixed_single60_double50_run01/final'
MIXED_INFERENCE_SCRIPT=f'{MIXED_REMOTE_ROOT}/artifacts/aubo_joint_two_strip_top45_pilot_20260927/joint_inference_stdio.py'
BOTH_ORDERS_REMOTE_ROOT='/home/rentao/program/lerobot-aubo-smolvla-joint-both-orders-20260928'
BOTH_ORDERS_MODEL_SHA='c24b61b8d902d395c88890a3243879507d75333d5deb0bc99c13572ecddbfc3b'
BOTH_ORDERS_CHECKPOINT=f'{BOTH_ORDERS_REMOTE_ROOT}/outputs/joint_mixed_single60_top45_50_top90_50_30k_run01/final'
BOTH_ORDERS_INFERENCE_SCRIPT=f'{BOTH_ORDERS_REMOTE_ROOT}/artifacts/aubo_joint_both_orders_20260928/joint_inference_stdio.py'
BOTH_ORDERS_APPROACH_SCRIPT=f'{BOTH_ORDERS_REMOTE_ROOT}/artifacts/joint_age_aligned_approach_20260928/joint_inference_stdio.py'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--model',choices=('legacy-single','mixed-double','mixed-both-orders'),default='legacy-single')
    p.add_argument('--approach-age-aligned',action='store_true',
                   help='Experimental mixed-both-orders approach: age-align open-gripper targets; retain full pick/place cycles')
    args=p.parse_args(argv)
    approach=args.approach_age_aligned
    if approach and args.model!='mixed-both-orders':p.error('--approach-age-aligned requires --model mixed-both-orders')
    mixed=args.model in ('mixed-double','mixed-both-orders')
    expected_cycles=2 if mixed else 1
    max_seconds=360 if mixed else 120
    servo_command_time=.08 if mixed else DT
    max_camera_age_ms=250 if mixed else 100
    max_prediction_age=.5 if args.model=='mixed-both-orders' else MAX_PREDICTION_AGE
    model_sha=MIXED_MODEL_SHA if mixed else MODEL_SHA
    pipe_options=({'remote_root':MIXED_REMOTE_ROOT,'checkpoint':MIXED_CHECKPOINT,
                   'expected_sha256':MIXED_MODEL_SHA,'inference_script':MIXED_INFERENCE_SCRIPT}
                  if mixed else {})
    if args.model=='mixed-both-orders':
        model_sha=BOTH_ORDERS_MODEL_SHA
        pipe_options={'remote_root':BOTH_ORDERS_REMOTE_ROOT,'checkpoint':BOTH_ORDERS_CHECKPOINT,
                      'expected_sha256':model_sha,'inference_script':BOTH_ORDERS_INFERENCE_SCRIPT}
        if approach:pipe_options.update(inference_script=BOTH_ORDERS_APPROACH_SCRIPT,return_action_chunk=True)
    plan={'mode':'asynchronous_inference_continuous_servo','model':args.model,
          'action_selection':'observation_age_while_open_first_step_while_closed' if approach else 'first_step',
          'pending_suction_policy':'continue_inference_while_open' if approach else 'accept_fresh_inflight_before_io',
          'checkpoint_sha256':model_sha,'requested_suction_cycles':expected_cycles,
          'camera_set_sha256':hashlib.sha256(CAMERA_SET.read_bytes()).hexdigest(),'control_hz':1/DT,
          'joint_speed_deg_s':MAX_SPEED,'joint_acceleration_deg_s2':MAX_ACCEL,'tcp_speed_m_s':MAX_TCP_SPEED,
          'max_prediction_delta_deg':None,'max_prediction_delta_m':None,
          'max_servo_tracking_error_deg':MAX_TRACKING_ERROR_DEG,
          'max_prediction_age_s':max_prediction_age,'watchdog_s':MAX_WITHOUT_PREDICTION,
          'max_camera_skew_s':MAX_CAMERA_SKEW,
          'max_camera_age_ms':max_camera_age_ms,
          'rpc_timeout_ms':RPC_TIMEOUT_MS,'max_control_gap_s':.35,
          'max_state_age_s':.1,'stale_state_refreshes_per_step':1 if mixed else 0,
          'servo_command_time_s':servo_command_time,
          'suction_writes':True,'max_seconds':max_seconds,'automatic_home':False,'automatic_release':False,
          'startup_release_if_unset':'operator_confirmed',
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
    initial_state_known=False
    reason='not_started';sent=0;predictions=0;completed_cycles=0;cycle=False
    recorder=None;video_result=None
    trace=(output/'trace.jsonl').open('w',buffering=1)
    commands=(output/'commands.jsonl').open('w',buffering=1)
    def log(f,value):f.write(json.dumps(value,allow_nan=False)+'\n')
    try:
        print('加载模型、连接相机；尚未发送运动。',flush=True)
        pipe=InferencePipe(output,task=task,**pipe_options);station.connect()
        motion=station.iface.getMotionControl()
        if motion.isServoModeEnabled():raise RuntimeError('existing servo owner')
        from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot
        from lerobot.robots.aubo_i10.config_aubo_i10 import AuboI10Config
        suction=AuboI10Robot(AuboI10Config());suction.io_control=station.io_readback
        try:
            state,tcp,_=station.current()
        except ValueError as exc:
            if str(exc)!='unknown suction output pair: (False, False)':raise
            print('DO2/DO3 均为低电平，当前吸盘指令状态未知。',flush=True)
            if input('确认吸盘未持物。按回车仅发送释放指令并核对输出，输入 q 退出：').strip():
                reason='cancelled';return 0
            try:
                if not suction.suction_release():raise RuntimeError('initial suction release failed')
            finally:log(trace,{'initial_suction_release':suction.last_gripper_command_trace})
            state,tcp,_=station.current()
        initial_state_known=True
        if state[-1]!=0 or max(abs(a-b) for a,b in zip(state[:6],START_DEG))>1.5:
            raise ValueError('请先归位并关闭吸盘，再启动')
        servo=SingleStepSession(motion)
        def fk(q):
            pose,ret=station.algorithm.forwardKinematics([math.radians(x) for x in q])
            if ret!=0:raise ValueError(f'FK failed: {ret}')
            return list(finite_vector(pose,6,'FK')[:3])
        if math.dist(fk(state[:6]),tcp)>.001:raise ValueError('FK/current TCP mismatch')
        print(f'记录：{output.resolve()}\n持续伺服；Ctrl+C停止，停止不自动释放吸盘。',flush=True)
        if input('确认工作区无人、操作者在场且急停在手边。按回车开始，输入q退出：').strip():
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
        active_chunk=None;active_times=None;selection=None;prediction_id=None
        discard_pending_result=False;request_index=0;inflight_index=None
        last_servo_sent_at=None
        deadline=time.perf_counter()+max_seconds;next_tick=time.perf_counter();last_tick=next_tick
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
            if future is not None and future.done() and discard_pending_result:
                result=future.result();future=None;discard_pending_result=False
                log(trace,{'index':inflight_index,'discarded_prediction':result,'reason':'observation_precedes_suction_change',
                    'observed_state':observed_state,'sensor_times':sensor_times})
            if future is not None and future.done():
                result=future.result();future=None
                previous_pending_target=goal.copy() if approach and pending_suction else None
                log(trace,{'index':inflight_index,'task':task,'observed_state':observed_state,
                    'sensor_times':sensor_times,'prediction':result,'received_at':time.perf_counter()})
                candidate=result['action']
                if approach:
                    active_chunk=validated_action_chunk(result);active_times=list(sensor_times)
                    selection=None
                    if held_suction==0:
                        candidate,selection=select_approach_action(active_chunk,active_times,time.perf_counter())
                    prediction_id=inflight_index
                accepted=validate_prediction(candidate,state,tcp,fk(candidate[:6]),
                    station.lower,station.upper,sensor_times,time.perf_counter(),
                    max_prediction_age=max_prediction_age)
                predictions+=1;goal=accepted;last_observation=min(sensor_times)
                if previous_pending_target is not None:
                    log(trace,{'pending_suction_replanned':{'previous_target':previous_pending_target,
                        'updated_target':goal,'prediction_index':prediction_id}})
                prediction_deadline=last_observation+MAX_WITHOUT_PREDICTION
                pending_suction=goal[-1]!=held_suction
                if not owned:
                    trajectory=SmoothTrajectory(state[:6]);owned=True
                    servo._require_zero(motion.setServoMode(True),'enable servo');servo._wait_servo(True)
            if goal is not None:
                if not pending_suction and time.perf_counter()>prediction_deadline:
                    raise TimeoutError(f'no fresh prediction within {MAX_WITHOUT_PREDICTION*1000:.0f} ms')
                if pending_suction and time.perf_counter()-last_observation>1.5:
                    raise TimeoutError('suction transition target not reached within 1.5 seconds')
                if approach and active_chunk is not None and held_suction==0 and not pending_suction:
                    candidate,new_selection=select_approach_action(active_chunk,active_times,time.perf_counter())
                    if new_selection['selected_index']!=selection['selected_index']:
                        goal=validate_prediction(candidate,state,tcp,fk(candidate[:6]),
                            station.lower,station.upper,active_times,time.perf_counter(),
                            max_prediction_age=max_prediction_age)
                    selection=new_selection
                    pending_suction=goal[-1]!=held_suction
                # Hold this coherent closure row while awaiting arrival, but
                # let an in-flight fresh observation replan BEFORE IO changes.
                q,target_tcp=trajectory.advance(goal[:6],fk)
                require_joint_target(q,state[:6],station.lower,station.upper,MAX_TRACKING_ERROR_DEG)
                state_age=time.perf_counter()-stamp
                if state_age>.1:
                    if not mixed:raise TimeoutError('current state expired')
                    state,tcp,stamp=station.current(require_stationary=False)
                    if state[-1]!=held_suction:raise ValueError('external suction state change')
                    workspace(tcp)
                    require_joint_target(q,state[:6],station.lower,station.upper,MAX_TRACKING_ERROR_DEG)
                    if time.perf_counter()-stamp>.1:raise TimeoutError('current state expired after refresh')
                    log(trace,{'state_refresh_before_servo':{'previous_age_s':state_age}})
                command_started_at=time.perf_counter()
                if mixed and last_servo_sent_at is not None and command_started_at-last_servo_sent_at>.35:
                    raise TimeoutError('servo command gap exceeded 350 ms')
                servo._require_zero(motion.servoJoint([math.radians(x) for x in q],math.radians(MAX_ACCEL),
                    math.radians(MAX_SPEED),servo_command_time,0.,200.),'servoJoint')
                last_servo_sent_at=command_started_at
                sent+=1
                command_record={'time':time.perf_counter(),'state':state,'command':q,'velocity':trajectory.v,
                    'target':goal,'tcp':tcp,'command_tcp':target_tcp}
                if approach:command_record.update(prediction_index=prediction_id,action_selection=selection,
                    suction_target_frozen=pending_suction)
                log(commands,command_record)
                if pending_suction and max(abs(a-b) for a,b in zip(goal[:6],state[:6]))<.15 and max(abs(v) for v in trajectory.v)<.3:
                    # Goal is held while switching IO; no averaging of discrete suction.
                    suction.is_suction_on=held_suction==100
                    try:suction._control_suction_based_on_gripper(goal[-1])
                    finally:log(trace,{'suction':suction.last_gripper_command_trace})
                    previous_suction=held_suction
                    held_suction=goal[-1];pending_suction=False
                    if approach:
                        discard_pending_result=future is not None
                        active_chunk=None;active_times=None;selection=None
                    if previous_suction==100 and held_suction==0:
                        completed_cycles+=1
                        if completed_cycles==expected_cycles:
                            cycle=True
                            reason='model_suction_cycle_completed' if not mixed else 'model_suction_cycles_completed'
                            break
                    # IO readback can take 175 ms; do not catch up missed ticks.
                    last_tick=time.perf_counter();next_tick=last_tick+DT
                    # A separate deadline allows a post-IO observation; original
                    # sensor timestamps remain unchanged in prediction records.
                    prediction_deadline=last_tick+MAX_WITHOUT_PREDICTION
                    # IO changed after the cycle's read: the next observation
                    # must contain the new suction state, not the old snapshot.
                    state,tcp,stamp=station.current(require_stationary=False)
                if sent%25==0:print(f'已发送 {sent} 帧，预测 {predictions} 次，吸盘 {held_suction:.0f}',flush=True)
            # The gripper is still open while approaching a closure goal.
            # Keep observing/replanning instead of treating that goal as final.
            if future is None and (not pending_suction or (approach and held_suction==0)):
                observed_state,images,sensor_times=station.observe(require_stationary=False,
                    state_snapshot=(state,tcp,stamp),max_camera_age_ms=max_camera_age_ms)
                if time.perf_counter()-min(sensor_times)>max_camera_age_ms/1000:
                    raise ValueError('stale input at capture')
                inflight_index=request_index;request_index+=1
                future=pool.submit(infer,inflight_index,observed_state,images)
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
        if initial_state_known and getattr(station,'iface',None) is not None:
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
            'approach_age_aligned':approach,
            'requested_suction_cycles':expected_cycles,'completed_suction_cycles':completed_cycles,
            'failure':failure,'suction_cycle_commanded':cycle,'physical_grasp_success':None,'video':video_result})
        print(f'结束：{reason}，记录：{output.resolve()}',flush=True)
    return 1 if failure else 0


if __name__=='__main__':
    def terminate(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    raise SystemExit(main())

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

from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_FIELDS,finite_vector,require_joint_target
from lerobot.bamboo_sorting.policy.joint_target_policy import TargetInputSession,is_transport
from lerobot.bamboo_sorting.runtime.joint_trial import START_DEG,SingleStepSession,wait_until_stationary,home_to_start
from lerobot.bamboo_sorting.runtime.joint_video import TrialVideoRecorder
from lerobot.bamboo_sorting.runtime.joint_target_runtime import TargetForegroundMissing
from lerobot.bamboo_sorting.runtime.joint_smooth import (DT,MAX_SPEED,MAX_ACCEL,MAX_TRACKING_ERROR_DEG,
    MAX_TCP_SPEED,MAX_WITHOUT_PREDICTION,MAX_PREDICTION_AGE,MAX_CAMERA_SKEW,SmoothTrajectory,validate_prediction,workspace,
    select_approach_action,select_transport_action)
from lerobot.bamboo_sorting.model_profiles import (
    BOTH_ORDERS_APPROACH_SCRIPT, BOTH_ORDERS_CHECKPOINT, BOTH_ORDERS_INFERENCE_SCRIPT,
    BOTH_ORDERS_MODEL_SHA, BOTH_ORDERS_REMOTE_ROOT, LEGACY_MODEL_SHA as MODEL_SHA,
    MIXED_CHECKPOINT, MIXED_INFERENCE_SCRIPT, MIXED_MODEL_SHA, MIXED_REMOTE_ROOT,
)
from lerobot.bamboo_sorting.runtime.inference import InferencePipe
from lerobot.bamboo_sorting.runtime.station import CAMERA_SET, RPC_TIMEOUT_MS, ReadOnlyStation, write
from lerobot.bamboo_sorting.runtime.task_prompt import read_task

__all__ = ['BOTH_ORDERS_MODEL_SHA', 'CAMERA_SET', 'DT', 'InferencePipe', 'MIXED_MODEL_SHA',
           'MODEL_SHA', 'ReadOnlyStation', 'main']


MAX_TARGET_REOBSERVATIONS = 3  # Per target, including separate intermittent misses.


def save_target_failure(output, index, state, images, sensor_times, metadata, exc):
    """Lossless request inputs and same-frame segmentation; never replace a run."""
    import base64
    import cv2
    folder=output/'target_failures'/f'{index:06d}'
    folder.mkdir(parents=True,exist_ok=False)
    write(folder/'failure.json',dict(index=index,observed_state=state,sensor_times=sensor_times,
        target_input=metadata,error=str(exc),error_code=exc.code,target_evidence=exc.evidence))
    for name,img in images.items():
        if not cv2.imwrite(str(folder/f'{name}.png'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR)):
            raise RuntimeError('failed input image save failed')
    (folder/'segmentation_instances.png').write_bytes(base64.b64decode(
        exc.evidence['segmentation_instances_png'],validate=True))
    return str(folder.relative_to(output))



def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--model',choices=('legacy-single','mixed-double','mixed-both-orders','target-pair','target-stack'),default='legacy-single')
    p.add_argument('--target-snapshot',type=Path,help='Reviewed seed snapshot; only its target selection is reused')
    p.add_argument('--target-selections',type=Path,help='Human-reviewed visible target selection JSON bound to the snapshot')
    p.add_argument('--target-id',help='Explicit target identity, e.g. s-001')
    p.add_argument('--target-sequence',nargs='+',help='target-stack: ordered remaining targets; automatically home between picks')
    p.add_argument('--target-preview',action='store_true',help='Read-only localhost view of same-request RGB and predicted target outline')
    p.add_argument('--track-target-motion',action='store_true',help='Experimental target-stack RGB motion references with current-foreground checks')
    p.add_argument('--paired-target-separation',action='store_true',
                   help='Explicit target-pair trial: separate current foreground using both reviewed target references')
    p.add_argument('--multi-target-separation',action='store_true',
                   help='Use current instances and all reviewed visible seeds for one stack pick')
    p.add_argument('--transport-age-aligned',action='store_true',
                   help='Target model trial: age-align transport actions without skipping the first release')
    p.add_argument('--cycles',type=int,help='Stop after this many close/open command cycles; defaults to the model setting')
    p.add_argument('--inference-timeout-s',type=float,
                   help='Wait for a model response; does not relax observation age or servo watchdog limits')
    p.add_argument('--max-seconds',type=float,help='Maximum run duration, or per-target duration with --target-sequence; completion stops earlier')
    p.add_argument('--latency-tolerant',action='store_true',
                   help='Target models: 1.5 s prediction age/watchdog; target-stack also: 400 ms camera age, 300 ms skew, 700 ms control gap, 250 ms state age; mixed-both-orders: 750 ms / 1 s and 400 ms / 300 ms; all: 5 s response wait')
    p.add_argument('--approach-age-aligned',action='store_true',
                   help='Experimental mixed-both-orders approach: age-align open-gripper targets; retain full pick/place cycles')
    args=p.parse_args(argv)
    target_mode=args.model in ('target-pair','target-stack')
    if args.target_preview and not target_mode:p.error('--target-preview requires a target model')
    if args.track_target_motion and args.model!='target-stack':p.error('--track-target-motion requires target-stack')
    sequence=args.target_sequence
    if sequence:
        if args.model!='target-stack' or args.target_id:
            p.error('--target-sequence requires target-stack and replaces --target-id')
        if len(set(sequence))!=len(sequence):p.error('target sequence must contain distinct identities')
        if args.cycles not in (None,len(sequence)):p.error('--cycles must match the target sequence length')
        args.target_id=sequence[0]
    if args.model=='target-stack':
        if args.paired_target_separation:p.error('target-stack uses multi-target separation')
        args.multi_target_separation=True
    if args.multi_target_separation and (not target_mode or args.paired_target_separation):
        p.error('--multi-target-separation requires a target model and excludes --paired-target-separation')
    if args.paired_target_separation and not target_mode:
        p.error('--paired-target-separation requires --model target-pair')
    if args.transport_age_aligned and not target_mode:
        p.error('--transport-age-aligned requires a target model')
    if args.cycles is not None and args.cycles<1:p.error('--cycles must be a positive integer')
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds<=0):
        p.error('--max-seconds must be finite and positive')
    inference_timeout_s=args.inference_timeout_s if args.inference_timeout_s is not None else (5. if args.latency_tolerant else 2.)
    if not math.isfinite(inference_timeout_s) or inference_timeout_s<=0:
        p.error('--inference-timeout-s must be finite and positive')
    if args.latency_tolerant and not (args.model=='mixed-both-orders' or target_mode):
        p.error('--latency-tolerant requires mixed-both-orders or a target model')
    approach=args.approach_age_aligned or target_mode
    if approach and not (args.model=='mixed-both-orders' or target_mode):p.error('--approach-age-aligned requires mixed-both-orders or a target model')
    mixed=args.model in ('mixed-double','mixed-both-orders') or target_mode
    return args,sequence,target_mode,approach,mixed,inference_timeout_s,p


def resolve_targets(args, sequence, target_mode, parser):
    """Load reviewed seeds and model options; no hardware or model access."""
    target_session=None;target_seed=None;target_options=None;target_setups=None
    if target_mode:
        if not all((args.target_snapshot,args.target_selections,args.target_id)):
            parser.error('target models require --target-snapshot, --target-selections and --target-id')
        if not sequence and args.cycles not in (None,1):parser.error('target models run exactly one selected-object cycle')
        from joint_target_live_setup import prepare_live_setup
        setup_options={'paired_separation':True} if args.paired_target_separation else {}
        if args.multi_target_separation:
            setup_options={'multi_separation':True}
        if args.model=='target-stack':
            setup_options['model']='target-stack'
        if args.track_target_motion:
            setup_options['motion_tracking']=True
        if sequence:
            target_setups=[prepare_live_setup(args.target_snapshot,args.target_selections,target,
                remaining_target_ids=sequence[index:],**setup_options) for index,target in enumerate(sequence)]
            target_seed,target_options=target_setups[0]
        else:
            target_seed,target_options=prepare_live_setup(args.target_snapshot,args.target_selections,args.target_id,**setup_options)
        target_session=TargetInputSession(args.target_id)
    elif any((args.target_snapshot,args.target_selections,args.target_id)):
        parser.error('target selection requires a target model')
    return target_session,target_seed,target_options,target_setups


def resolve_config(args, sequence, target_mode, approach, mixed, inference_timeout_s, target_options):
    """Derive timing/limit settings and remote worker options from the mode."""
    expected_cycles=len(sequence) if sequence else (args.cycles if args.cycles is not None else (2 if mixed and not target_mode else 1))
    max_seconds=args.max_seconds if args.max_seconds is not None else (360 if mixed and not target_mode else 120)
    servo_command_time=.08 if mixed else DT
    max_camera_age_ms=250 if mixed else 100
    max_prediction_age=.5 if args.model=='mixed-both-orders' or target_mode else MAX_PREDICTION_AGE
    max_without_prediction=MAX_WITHOUT_PREDICTION
    max_camera_skew=MAX_CAMERA_SKEW
    max_control_gap=.35
    max_state_age=.1
    if args.latency_tolerant:
        max_prediction_age=1.5 if target_mode else .75
        max_without_prediction=1.5 if target_mode else 1.
        if not target_mode or args.model=='target-stack':
            max_camera_age_ms=400;max_camera_skew=.3
        if args.model=='target-stack':
            max_control_gap=.7;max_state_age=.25
    model_sha=MIXED_MODEL_SHA if mixed else MODEL_SHA
    pipe_options=({'remote_root':MIXED_REMOTE_ROOT,'checkpoint':MIXED_CHECKPOINT,
                   'expected_sha256':MIXED_MODEL_SHA,'inference_script':MIXED_INFERENCE_SCRIPT}
                  if mixed else {})
    if args.model=='mixed-both-orders':
        model_sha=BOTH_ORDERS_MODEL_SHA
        pipe_options={'remote_root':BOTH_ORDERS_REMOTE_ROOT,'checkpoint':BOTH_ORDERS_CHECKPOINT,
                      'expected_sha256':model_sha,'inference_script':BOTH_ORDERS_INFERENCE_SCRIPT}
        if approach:pipe_options.update(inference_script=BOTH_ORDERS_APPROACH_SCRIPT,return_action_chunk=True)
    if target_mode:
        pipe_options=target_options
        model_sha=target_options['expected_sha256']
    pipe_options['request_timeout_s']=inference_timeout_s
    return dict(expected_cycles=expected_cycles,max_seconds=max_seconds,servo_command_time=servo_command_time,
        max_camera_age_ms=max_camera_age_ms,max_prediction_age=max_prediction_age,
        max_without_prediction=max_without_prediction,max_camera_skew=max_camera_skew,
        max_control_gap=max_control_gap,max_state_age=max_state_age,mixed=mixed,
        model_sha=model_sha,pipe_options=pipe_options,inference_timeout_s=inference_timeout_s,
        approach=approach)


def build_plan(args, sequence, target_mode, target_seed, target_options, config):
    """Describe the requested run without connecting hardware or a model."""
    expected_cycles=config['expected_cycles'];max_seconds=config['max_seconds']
    plan={'mode':'asynchronous_inference_continuous_servo','model':args.model,
          'action_selection':('observation_age_open_and_transport' if args.transport_age_aligned else
                              'observation_age_while_open_first_step_while_closed' if config['approach'] else 'first_step'),
          'transport_age_aligned':args.transport_age_aligned,
          'pending_suction_policy':'continue_inference_while_open' if config['approach'] else 'accept_fresh_inflight_before_io',
          'checkpoint_sha256':config['model_sha'],'requested_suction_cycles':expected_cycles,
          'camera_set_sha256':hashlib.sha256(CAMERA_SET.read_bytes()).hexdigest(),'control_hz':1/DT,
          'joint_speed_deg_s':MAX_SPEED,'joint_acceleration_deg_s2':MAX_ACCEL,'tcp_speed_m_s':MAX_TCP_SPEED,
          'max_prediction_delta_deg':None,'max_prediction_delta_m':None,
          'max_servo_tracking_error_deg':MAX_TRACKING_ERROR_DEG,
          'latency_tolerant':args.latency_tolerant,
          'max_prediction_age_s':config['max_prediction_age'],'watchdog_s':config['max_without_prediction'],
          'max_camera_skew_s':config['max_camera_skew'],
          'max_camera_age_ms':config['max_camera_age_ms'],
          'rpc_timeout_ms':RPC_TIMEOUT_MS,'inference_timeout_s':config['inference_timeout_s'],
          'max_control_gap_s':config['max_control_gap'],
          'max_state_age_s':config['max_state_age'],'stale_state_refreshes_per_step':1 if config['mixed'] else 0,
          'servo_command_time_s':config['servo_command_time'],
          'suction_writes':True,'max_seconds':max_seconds,'automatic_home':False,'automatic_release':False,
          'startup_release_if_unset':'operator_confirmed',
          'servo_disable_confirmation_timeout_s':1.0,'video_recording':{'enabled':True,'fps':25,'directory':'videos'}}
    if target_mode:
        plan.update(target_id=args.target_id,seed_snapshot_id=target_seed['snapshot_id'],
            checkpoint=target_options.get('checkpoint'),inference_script=target_options.get('inference_script'),
            target_selection_mode=target_seed.get('target_selection_mode', 'highest_score'),
            target_mask_source=('current instances with visible seeds; merged instances flagged; stop if target has no pixels'
                                if args.multi_target_separation else
                                'fresh foreground separated by both reviewed targets; stop if selected target has no pixels'
                                if args.paired_target_separation else 'fresh segmentation plus reviewed-seed association; stop on failure'),
            target_phase_source='first accepted close command; never prediction',
            segmentation_checkpoint=target_options['segmentation_checkpoint'])
    if sequence:
        plan.update(target_sequence=sequence,automatic_home=len(sequence)>1,
                    homing_between_targets_only=True,max_seconds_scope='per_target',
                    inference_session_per_target=True,
                    remaining_targets_by_pick=[sequence[i:] for i in range(len(sequence))])
    if args.model=='target-stack':
        plan.update(empty_target_reobservations_per_target=MAX_TARGET_REOBSERVATIONS,
            empty_target_policy='stop servo, clear actions, preserve gripper; fresh observations only')
    plan.update(target_preview=args.target_preview,track_target_motion=args.track_target_motion)
    return plan


def main(argv=None):
    args,sequence,target_mode,approach,mixed,inference_timeout_s,parser=parse_args(argv)
    target_session,target_seed,target_options,target_setups=resolve_targets(args,sequence,target_mode,parser)
    config=resolve_config(args,sequence,target_mode,approach,mixed,inference_timeout_s,target_options)
    plan=build_plan(args,sequence,target_mode,target_seed,target_options,config)
    if not args.execute:print(json.dumps(plan,indent=2));return 0
    if not sys.stdin.isatty():parser.error('interactive terminal required')
    return execute_run(args,sequence,target_mode,target_seed,target_options,target_setups,target_session,plan,config)


def execute_run(args,sequence,target_mode,target_seed,target_options,target_setups,target_session,plan,config):
    """Single-target execution, sequence scheduling, target switching and cleanup."""
    expected_cycles=config['expected_cycles'];max_seconds=config['max_seconds']
    servo_command_time=config['servo_command_time'];max_camera_age_ms=config['max_camera_age_ms']
    max_prediction_age=config['max_prediction_age'];max_without_prediction=config['max_without_prediction']
    max_camera_skew=config['max_camera_skew'];max_control_gap=config['max_control_gap']
    max_state_age=config['max_state_age'];mixed=config['mixed'];pipe_options=config['pipe_options']
    inference_timeout_s=config['inference_timeout_s'];approach=config['approach']
    try:task=read_task()
    except (EOFError,KeyboardInterrupt):return 0
    if task is None:return 0
    plan['task']=task
    output=args.output or Path('artifacts')/('joint_smooth_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False);write(output/'plan.json',plan)
    station=ReadOnlyStation();pipe=None;pool=None;servo=None;owned=False;failure=None
    initial_state_known=False
    reason='not_started';sent=0;predictions=0;completed_cycles=0;cycle=False
    current_target_id=args.target_id
    recorder=None;video_result=None
    preview=None;motion_tracker=None
    trace=(output/'trace.jsonl').open('w',buffering=1)
    commands=(output/'commands.jsonl').open('w',buffering=1)
    def log(f,value):f.write(json.dumps(value,allow_nan=False)+'\n')
    try:
        if args.target_preview:
            from lerobot.bamboo_sorting.runtime.joint_target_preview import LiveTargetPreview
            preview=LiveTargetPreview()
            print(f'目标预测预览（浏览器打开）：{preview.url}',flush=True)
            write(output/'preview.json',dict(url=preview.url,read_only=True))
        if args.track_target_motion:
            import cv2
            from lerobot.bamboo_sorting.perception.joint_target_motion import TargetMotionReferences
            from lerobot.bamboo_sorting.runtime.joint_target_runtime import decode_mask
            initial_rgb=cv2.cvtColor(cv2.imread(str(args.target_snapshot/'global_rgb.png')),cv2.COLOR_BGR2RGB)
            motion_tracker=TargetMotionReferences(initial_rgb,
                {t['target_id']:decode_mask(t['mask_png']) for t in target_seed['visible_targets']})
        if target_mode:
            if sequence:
                print(f'目标顺序：{" → ".join(sequence)}；每次释放后自动归位再抓下一根，最后一根结束后停止。',flush=True)
            else:
                print(f'本轮目标：{args.target_id}；一次夹紧→释放后结束。请保持圈选时的竹条摆放和全局相机位置。',flush=True)
            if args.paired_target_separation:
                print('本轮启用双目标前景分离试验；仅使用当前分割像素，无所选目标像素时停止。',flush=True)
            if args.multi_target_separation:
                print('本轮启用多实例可见种子分离；目标为空时停伺服，target-stack每根最多3次新观测重试，其他错误停止。',flush=True)
            if args.track_target_motion:
                print('实验性目标运动跟随：RGB运动更新参考，再核对当前分割；身份或前景支持不足时停止。',flush=True)
            if args.transport_age_aligned:
                print('本轮搬运按观察年龄取步；保留首次释放边界，到达目标后才松爪。',flush=True)
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
        def infer(index,state,images,metadata=None):
            extra={}
            if motion_tracker is not None:
                if sequence:
                    offset=sequence.index(metadata['target_id'])
                    tracked=sequence[offset+(metadata['phase']=='transport'):]
                else:
                    tracked=[t['target_id'] for t in target_seed['visible_targets']
                        if metadata['phase']=='approach' or t['target_id']!=metadata['target_id']]
                # Preserve remaining-target motion across transport and new
                # inference sessions; never track a carried strip as a scene seed.
                motion=motion_tracker.update(images['global_rgb'],tracked,
                    frame_id=metadata['frame_id'],required=False)
                log(trace,{'target_motion':{'index':index,'target_input':metadata,'reference':motion}})
                if metadata['phase']=='approach':extra['target_motion']=motion
            result=(pipe.predict(index,state,images,codec='rgb_zlib',target_input=metadata,**extra)
                    if target_mode else pipe.predict(index,state,images))
            if index%5==0:
                import cv2
                for name,img in images.items():
                    if not cv2.imwrite(str(output/f'{index:04d}_{name}.jpg'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR)):
                        raise RuntimeError('snapshot save failed')
            return result
        future=None;goal=None;pending_suction=False;last_observation=None;trajectory=None
        active_chunk=None;active_times=None;selection=None;prediction_id=None;active_transport=False
        discard_pending_result=False;request_index=0;inflight_index=None
        last_servo_sent_at=None
        target_misses=0;recovery_after=None
        deadline=time.perf_counter()+max_seconds;next_tick=time.perf_counter();last_tick=next_tick
        prediction_deadline=None
        reason='time_limit';held_suction=0.
        while time.perf_counter()<deadline:
            now=time.perf_counter()
            if now<next_tick:time.sleep(next_tick-now)
            now=time.perf_counter()
            if owned and now-last_tick>max_control_gap:raise TimeoutError(f'servo loop delayed over {max_control_gap*1000:.0f} ms')
            last_tick=now;next_tick=now+DT
            state,tcp,stamp=station.current(require_stationary=False)
            if state[-1]!=held_suction:raise ValueError('external suction state change')
            workspace(tcp)
            future_ready=future is not None and future.done()
            if future_ready:
                try:
                    result=future.result()
                except TargetForegroundMissing as exc:
                    future=None
                    # The single worker has completed: there is no remaining
                    # in-flight action to drain or reuse. Stop before disk IO.
                    goal=None;pending_suction=False;trajectory=None;last_observation=None
                    active_chunk=None;active_times=None;selection=None;prediction_id=None;active_transport=False
                    prediction_deadline=None;last_servo_sent_at=None;discard_pending_result=False
                    try:
                        if owned:
                            servo._require_zero(motion.setServoMode(False),'disable servo on empty target')
                            servo._wait_servo(False,timeout_s=1.);owned=False
                        wait_until_stationary(lambda:station.current(require_stationary=False),station.state.isSteady)
                    finally:
                        evidence_path=save_target_failure(output,inflight_index,observed_state,
                            images,sensor_times,metadata,exc)
                        log(trace,{'target_foreground_missing':{'index':inflight_index,
                            'target_id':current_target_id,'evidence':evidence_path,'held_suction':held_suction}})
                        if preview is not None:
                            preview.publish(images=images,evidence=exc.evidence,index=inflight_index,
                                sensor_times=sensor_times,status='target missing; robot stopped')
                    target_misses+=1
                    if (args.model!='target-stack' or held_suction!=0
                            or target_session.accepted_close_time is not None
                            or target_misses>MAX_TARGET_REOBSERVATIONS):
                        raise
                    recovery_after=time.perf_counter()
                    last_tick=recovery_after;next_tick=last_tick+DT
                    log(trace,{'target_reobserve':{'attempt':target_misses,'target_id':current_target_id,
                        'stopped_at':recovery_after,'held_suction':held_suction}})
                    continue
            if future_ready and discard_pending_result:
                future=None;discard_pending_result=False
                log(trace,{'index':inflight_index,'discarded_prediction':result,'reason':'observation_precedes_suction_change',
                    'observed_state':observed_state,'sensor_times':sensor_times})
            if future is not None and future_ready:
                future=None
                previous_pending_target=goal.copy() if approach and pending_suction else None
                log(trace,{'index':inflight_index,'task':task,'observed_state':observed_state,
                    'sensor_times':sensor_times,'prediction':result,'received_at':time.perf_counter()})
                candidate=result['action']
                if approach:
                    # InferencePipe already checked and normalized the full chunk.
                    active_chunk=result['action_chunk'];active_times=list(sensor_times)
                    active_transport=(args.transport_age_aligned and
                                      result.get('target_input',{}).get('phase')=='transport')
                    selection=None
                    if held_suction==0:
                        candidate,selection=select_approach_action(active_chunk,active_times,time.perf_counter())
                    elif active_transport:
                        candidate,selection=select_transport_action(active_chunk,active_times,time.perf_counter())
                    prediction_id=inflight_index
                accepted=validate_prediction(candidate,state,tcp,fk(candidate[:6]),
                    station.lower,station.upper,sensor_times,time.perf_counter(),
                    max_prediction_age=max_prediction_age,max_camera_skew=max_camera_skew)
                predictions+=1;goal=accepted;last_observation=min(sensor_times)
                if preview is not None:
                    preview.publish(images=images,evidence=result['target_evidence'],index=inflight_index,
                        sensor_times=sensor_times)
                if previous_pending_target is not None:
                    log(trace,{'pending_suction_replanned':{'previous_target':previous_pending_target,
                        'updated_target':goal,'prediction_index':prediction_id}})
                prediction_deadline=last_observation+max_without_prediction
                pending_suction=goal[-1]!=held_suction
                if not owned:
                    trajectory=SmoothTrajectory(state[:6]);owned=True
                    servo._require_zero(motion.setServoMode(True),'enable servo');servo._wait_servo(True)
                    if recovery_after is not None:
                        log(trace,{'target_recovered':{'index':inflight_index,'target_id':current_target_id,
                            'state':state,'sensor_times':sensor_times,'trajectory_velocity':trajectory.v.copy()}})
                        recovery_after=None
            if goal is not None:
                if not pending_suction and time.perf_counter()>prediction_deadline:
                    raise TimeoutError(f'no fresh prediction within {max_without_prediction*1000:.0f} ms')
                if pending_suction and time.perf_counter()-last_observation>1.5:
                    raise TimeoutError('suction transition target not reached within 1.5 seconds')
                if approach and active_chunk is not None and (held_suction==0 or active_transport) and not pending_suction:
                    select_action=select_approach_action if held_suction==0 else select_transport_action
                    candidate,new_selection=select_action(active_chunk,active_times,time.perf_counter())
                    if new_selection['selected_index']!=selection['selected_index']:
                        goal=validate_prediction(candidate,state,tcp,fk(candidate[:6]),
                            station.lower,station.upper,active_times,time.perf_counter(),
                            max_prediction_age=max_prediction_age,max_camera_skew=max_camera_skew)
                    selection=new_selection
                    pending_suction=goal[-1]!=held_suction
                # Hold this coherent IO-transition row while awaiting arrival, but
                # let an in-flight fresh observation replan BEFORE IO changes.
                q,target_tcp=trajectory.advance(goal[:6],fk)
                require_joint_target(q,state[:6],station.lower,station.upper,MAX_TRACKING_ERROR_DEG)
                state_age=time.perf_counter()-stamp
                if state_age>max_state_age:
                    if not mixed:raise TimeoutError('current state expired')
                    state,tcp,stamp=station.current(require_stationary=False)
                    if state[-1]!=held_suction:raise ValueError('external suction state change')
                    workspace(tcp)
                    require_joint_target(q,state[:6],station.lower,station.upper,MAX_TRACKING_ERROR_DEG)
                    if time.perf_counter()-stamp>max_state_age:raise TimeoutError('current state expired after refresh')
                    log(trace,{'state_refresh_before_servo':{'previous_age_s':state_age}})
                command_started_at=time.perf_counter()
                if mixed and last_servo_sent_at is not None and command_started_at-last_servo_sent_at>max_control_gap:
                    raise TimeoutError(f'servo command gap exceeded {max_control_gap*1000:.0f} ms')
                servo._require_zero(motion.servoJoint([math.radians(x) for x in q],math.radians(MAX_ACCEL),
                    math.radians(MAX_SPEED),servo_command_time,0.,200.),'servoJoint')
                last_servo_sent_at=command_started_at
                sent+=1
                command_record={'time':time.perf_counter(),'state':state,'command':q,'velocity':trajectory.v,
                    'target':goal,'tcp':tcp,'command_tcp':target_tcp}
                if sequence:command_record.update(target_id=current_target_id,pick_index=completed_cycles)
                if approach:command_record.update(prediction_index=prediction_id,action_selection=selection,
                    suction_target_frozen=pending_suction)
                log(commands,command_record)
                if pending_suction and max(abs(a-b) for a,b in zip(goal[:6],state[:6]))<.15 and max(abs(v) for v in trajectory.v)<.3:
                    # Goal is held while switching IO; no averaging of discrete suction.
                    suction.is_suction_on=held_suction==100
                    try:suction._control_suction_based_on_gripper(goal[-1])
                    finally:log(trace,{'suction':suction.last_gripper_command_trace})
                    if target_mode:
                        accepted_at=time.perf_counter()
                        target_session.accept_command(dict(zip(JOINT_FIELDS,goal,strict=True)),
                            accepted_monotonic_s=accepted_at)
                        log(trace,{'target_phase_command':{'action':goal,'accepted_monotonic_s':accepted_at,
                            'first_accepted_close_s':target_session.accepted_close_time,'target_id':current_target_id}})
                        if preview is not None:
                            preview.publish(target_id=current_target_id,phase='transport',
                                status='gripper command accepted; waiting for next observation',jpeg=None)
                    previous_suction=held_suction
                    held_suction=goal[-1];pending_suction=False
                    if approach:
                        discard_pending_result=future is not None
                        active_chunk=None;active_times=None;selection=None;active_transport=False
                    if previous_suction==100 and held_suction==0:
                        completed_cycles+=1
                        if sequence:
                            log(trace,{'target_cycle_completed':{'target_id':current_target_id,
                                'completed_cycles':completed_cycles,'physical_grasp_success':None}})
                        if completed_cycles==expected_cycles:
                            cycle=True
                            reason='model_suction_cycle_completed' if expected_cycles==1 else 'model_suction_cycles_completed'
                            break
                        if sequence:
                            # End the old servo and worker before changing identity. No old
                            # chunk or in-flight observation may enter the next pick.
                            servo._require_zero(motion.setServoMode(False),'disable servo between targets')
                            servo._wait_servo(False,timeout_s=1.);owned=False
                            wait_until_stationary(lambda:station.current(require_stationary=False),station.state.isSteady)
                            if future is not None:
                                discarded=future.result();future=None
                                log(trace,{'index':inflight_index,'discarded_prediction':discarded,
                                           'reason':'previous_target_completed','target_id':current_target_id})
                            pipe.close();pipe=None
                            print(f'{current_target_id} 指令周期完成；自动归位，随后抓 {sequence[completed_cycles]}。',flush=True)
                            receipt=home_to_start(motion,lambda:station.current(require_stationary=False)[0],
                                lower=station.lower,upper=station.upper,is_steady=station.state.isSteady,
                                progress=lambda event:log(trace,{'between_target_home':event}))
                            log(trace,{'between_target_home_complete':receipt})
                            current_target_id=sequence[completed_cycles]
                            if preview is not None:
                                preview.publish(target_id=current_target_id,phase='approach',
                                    status='loading next target; waiting for prediction',jpeg=None)
                            target_seed,next_options=target_setups[completed_cycles]
                            target_session=TargetInputSession(current_target_id)
                            worker_output=output/f'target_{completed_cycles+1:02d}'
                            worker_output.mkdir()
                            pipe=InferencePipe(worker_output,task=task,**{**next_options,'request_timeout_s':inference_timeout_s})
                            goal=None;pending_suction=False;last_observation=None;trajectory=None
                            active_chunk=None;active_times=None;selection=None;prediction_id=None;active_transport=False
                            discard_pending_result=False;inflight_index=None;prediction_deadline=None
                            last_servo_sent_at=None
                            target_misses=0;recovery_after=None
                            next_tick=time.perf_counter();last_tick=next_tick;deadline=next_tick+max_seconds
                            log(trace,{'target_started':{'target_id':current_target_id,
                                'remaining_targets':sequence[completed_cycles:],'accepted_close_time':None}})
                            continue
                    # IO readback can take 175 ms; do not catch up missed ticks.
                    last_tick=time.perf_counter();next_tick=last_tick+DT
                    # A separate deadline allows a post-IO observation; original
                    # sensor timestamps remain unchanged in prediction records.
                    prediction_deadline=last_tick+max_without_prediction
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
                if recovery_after is not None and min(sensor_times)<=recovery_after:
                    if time.perf_counter()-recovery_after>max_without_prediction:
                        raise TimeoutError('no post-stop observation for target recovery')
                    continue
                inflight_index=request_index;request_index+=1
                if target_mode:
                    metadata=dict(target_id=current_target_id,frame_id=f'{target_seed["snapshot_id"]}/live/{inflight_index}',
                        phase='transport' if is_transport(sensor_times[0],target_session.accepted_close_time) else 'approach')
                    future=pool.submit(infer,inflight_index,observed_state,images,metadata)
                else:
                    future=pool.submit(infer,inflight_index,observed_state,images)
        if sent==0:reason='no_motion'
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
        if preview is not None:preview.close()
        for close in (station.close,pipe.close if pipe else lambda:None):
            try:close()
            except BaseException as exc:failure=f'{failure or ""}; cleanup: {exc}'
        trace.close();commands.close()
        write(output/'complete.json',{'reason':reason,'sent_frames':sent,'predictions':predictions,
            'model':args.model,'target_id':args.target_id,
            **({'target_sequence':sequence,'current_target_id':current_target_id,
                'completed_target_ids':sequence[:completed_cycles]} if sequence else {}),
            'approach_age_aligned':approach,
            'transport_age_aligned':args.transport_age_aligned,
            'requested_suction_cycles':expected_cycles,'completed_suction_cycles':completed_cycles,
            'failure':failure,'suction_cycle_commanded':cycle,'physical_grasp_success':None,'video':video_result})
        print(f'结束：{reason}，记录：{output.resolve()}',flush=True)
    return 1 if failure else 0


if __name__=='__main__':
    def terminate(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    raise SystemExit(main())

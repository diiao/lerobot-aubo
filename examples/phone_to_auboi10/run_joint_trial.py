#!/usr/bin/env python
"""Plan, read-only prediction, or ONE bounded joint command; never full grasp.

No hardware/model imports in plan mode. Shadow obtains state/config/algorithm
and IO-readback interfaces, calls no motion API and no digital-output setter.
Only explicit home/single-step modes may obtain MotionControl. Homing ensures
suction is released before moving; already released outputs are left unchanged.
No motion retries. Homing never loads a policy or opens cameras.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess  # kept so existing callers can patch the entry's process module
import sys
import time

from lerobot.bamboo_sorting.aubo_joint_contract import joint_contract_record
from lerobot.bamboo_sorting.runtime.joint_trial import (
    TrialLimits, SingleStepSession, home_to_start, stationary_step_limits, short_loop_limits,
    wait_until_stationary,
)
from lerobot.bamboo_sorting.model_profiles import (
    LEGACY_MODEL_SHA as MODEL_SHA,
    LEGACY_REMOTE_PYTHON as REMOTE_PYTHON,
    LEGACY_REMOTE_ROOT as REMOTE_ROOT,
)
from lerobot.bamboo_sorting.runtime.inference import InferencePipe
from lerobot.bamboo_sorting.runtime.station import (
    CAMERA_SET, RPC_TIMEOUT_MS, ReadOnlyStation, encode_images, write,
)

__all__ = [
    'CAMERA_SET', 'InferencePipe', 'MODEL_SHA', 'REMOTE_PYTHON', 'REMOTE_ROOT',
    'RPC_TIMEOUT_MS', 'ReadOnlyStation', 'encode_images', 'main', 'write',
]


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

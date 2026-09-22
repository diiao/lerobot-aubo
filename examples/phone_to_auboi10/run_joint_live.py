#!/usr/bin/env python
"""Manual, supervised one-episode SmolVLA trial, including model suction commands.

Default prints the plan. --execute loads the model and cameras, then requires
Enter in the local terminal. Ctrl+C stops motion without releasing held objects.
Uses stop-observe-move operation, not continuous 25-Hz streaming.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

from run_joint_trial import CAMERA_SET, MODEL_SHA, InferencePipe, ReadOnlyStation, write
from lerobot.bamboo_sorting.joint_trial import START_DEG, SingleStepSession, stationary_step_limits, wait_until_stationary
from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_TASK


def read_task():
    print(f'请输入本轮任务指令（输入 q 退出）：\n{JOINT_TASK}',flush=True)
    while True:
        task=input('任务指令：').strip()
        if task.lower()=='q':return None
        if task==JOINT_TASK:return task
        print('当前模型仅训练了上面这句任务，请完整输入或粘贴；输入 q 退出。',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    limits=stationary_step_limits()
    plan={'checkpoint_sha256':MODEL_SHA,'camera_set_sha256':hashlib.sha256(CAMERA_SET.read_bytes()).hexdigest(),
          'mode':'manual_stop_observe_move_episode','suction_writes':True,'max_seconds':120,'max_steps':200,
          'limits':asdict(limits),'start_envelope_during_episode':False,
          'tcp_workspace_m':{'min':[-.8,-1.2,0.],'max':[1.,0.,.8]},
          'automatic_home':False,'release_on_exit':False,'continuous_25hz':False}
    if not args.execute:
        print(json.dumps(plan,indent=2));return 0
    if not sys.stdin.isatty():parser.error('--execute requires a local interactive terminal')
    try:
        task=read_task()
    except (KeyboardInterrupt,EOFError):
        print('\n已取消，未连接设备。');return 0
    if task is None:return 0
    plan['task']=task
    output=args.output or Path('artifacts')/('joint_live_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False);write(output/'plan.json',plan)
    station=ReadOnlyStation();pipe=None;sent=0;seen_on=False;cycle=False;failure=None;reason='not_started'
    try:
        print('正在加载模型和相机；此时不会移动。',flush=True)
        pipe=InferencePipe(output,task=task)
        station.connect()
        initial,_,_=station.current()
        if initial[-1]!=0 or max(abs(a-b) for a,b in zip(initial[:6],START_DEG))>1.5:
            raise ValueError('请先回到初始位置并确认吸盘关闭，再启动本轮')
        motion=station.iface.getMotionControl()
        if motion.isServoModeEnabled():raise RuntimeError('已有伺服控制者，停止接管')
        # Reuse the existing driver only for verified two-pin suction writes.
        # Do not connect it or invoke its fixed-J5/legacy motion methods.
        from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot
        from lerobot.robots.aubo_i10.config_aubo_i10 import AuboI10Config
        suction=AuboI10Robot(AuboI10Config())
        suction.io_control=station.io_readback
        suction.is_suction_on=False
        print(f'记录目录：{output.resolve()}\n最多120秒；模型控制运动和吸盘。Ctrl+C停止，停止时不自动释放。',flush=True)
        if input('确认现场无人、急停在手边。按回车开始，输入q退出：').strip():
            reason='operator_cancelled';return 0
        deadline=time.perf_counter()+120
        reason='step_limit'
        for index in range(200):
            if time.perf_counter()>=deadline:reason='time_limit';break
            wait_until_stationary(lambda:station.current(require_stationary=False),station.state.isSteady)
            settled_at=time.perf_counter();time.sleep(.05)
            state,images,timestamps=station.observe()
            if min(timestamps[:2])<settled_at:raise ValueError('camera frame predates settled state')
            captured_at=time.perf_counter()
            result=pipe.predict(index,state,images)
            timing={'captured_at':captured_at,'observed_state':state,'limits':limits,'full_episode':True}
            gate=station.gate(result['action'],timestamps,**timing)
            record={'index':index,'task':task,'state':state,'sensor_times':timestamps,'prediction':result,'gate':gate}
            with (output/'trace.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            if not gate['passed']:raise ValueError(f'action rejected: {gate["reasons"]}')
            if time.perf_counter()>=deadline:reason='time_limit';break
            session=SingleStepSession(motion,limits=limits)
            try:
                session.execute(gate,revalidate=lambda:station.gate(result['action'],timestamps,**timing))
            finally:
                write(output/f'step_{index:03d}.json',{'attempted':session.command_attempted,
                    'accepted':session.command_accepted,'servo_disabled':session.servo_disabled})
                sent+=int(session.command_accepted)
            final_state,final_tcp,_=wait_until_stationary(lambda:station.current(require_stationary=False),station.state.isSteady)
            if final_state[-1]!=state[-1]:raise ValueError('suction changed externally during motion')
            suction.is_suction_on=state[-1]==100
            try:
                suction._control_suction_based_on_gripper(result['action'][-1])
            finally:
                write(output/f'suction_{index:03d}.json',suction.last_gripper_command_trace)
            seen_on=seen_on or result['action'][-1]==100
            cycle=seen_on and result['action'][-1]==0
            write(output/f'feedback_{index:03d}.json',{'state':final_state,'tcp':final_tcp,
                'state_read_before_suction_write':True,'suction_target':result['action'][-1],'is_steady':station.state.isSteady()})
            import cv2
            for name,img in images.items():
                if not cv2.imwrite(str(output/f'{index:03d}_{name}.jpg'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR)):
                    raise RuntimeError('image save failed')
            print(f'步 {index+1}: TCP {[round(v,4) for v in final_tcp]}，吸盘 {result["action"][-1]:.0f}',flush=True)
            if cycle:reason='model_suction_cycle_completed';break
    except KeyboardInterrupt:
        reason='operator_interrupt'
        print('\n已收到停止，保留吸盘当前状态。',flush=True)
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}';reason='error'
        print(failure,file=sys.stderr,flush=True)
    finally:
        if getattr(station,'iface',None) is not None:
            try:
                q,tcp,stamp=wait_until_stationary(lambda:station.current(require_stationary=False),station.state.isSteady)
                write(output/'final_state.json',{'state':q,'tcp':tcp,'timestamp':stamp,
                    'is_steady':station.state.isSteady(),'servo_enabled':station.iface.getMotionControl().isServoModeEnabled()})
            except BaseException as exc:
                failure=f'{failure or ""}; final readback: {exc}'
        for close in (station.close,pipe.close if pipe else lambda:None):
            try:close()
            except BaseException as exc:failure=f'{failure or ""}; cleanup: {exc}'
        write(output/'complete.json',{'reason':reason,'sent_steps':sent,'suction_cycle_commanded':cycle,
            'failure':failure,'physical_grasp_success':None,'automatic_release':False})
        print(f'本轮结束：{reason}；发送 {sent} 步。记录：{output.resolve()}',flush=True)
    return 1 if failure else 0


if __name__=='__main__':
    def terminate(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    raise SystemExit(main())

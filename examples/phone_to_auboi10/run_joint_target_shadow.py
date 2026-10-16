#!/usr/bin/env python
"""One stationary observation and explicit target A/B predictions; NO motion or IO writes.

Default plan does not connect hardware or a model. Capture reads one stationary
snapshot only. Predict consumes saved images and two operator-reviewed masks,
so review time never makes a stale observation eligible for robot execution.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from run_joint_trial import CAMERA_SET, InferencePipe, ReadOnlyStation, write


REMOTE_ROOT='/home/rentao/program/lerobot-aubo-joint-target-pair'
MODEL_SHA='93fd4a6835a98e4d7a0b04bc6aafd51b7d07508e57dccd19e698472968ca6e0a'
CHECKPOINT=REMOTE_ROOT+'/outputs/joint_target_pair_smolvla/final'
INFERENCE_SCRIPT=REMOTE_ROOT+'/artifacts/joint_target_shadow_service/joint_inference_stdio.py'
PYTHONPATH=REMOTE_ROOT+'/src:/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/deps'


def render_review(output, snapshot):
    template=Path(__file__).with_name('joint_target_shadow_review.html').read_text()
    # The template consumes only a generated UUID. No pasted arbitrary HTML.
    (output/'review.html').write_text(template.replace('__SNAPSHOT_ID__',snapshot['snapshot_id']))


def capture(args):
    import cv2
    from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_FIELDS, finite_vector, joint_command
    args.output.mkdir(parents=True,exist_ok=False)
    station=ReadOnlyStation()
    try:
        station.connect()
        state,images,times=station.observe(require_stationary=True,max_camera_age_ms=100)
        state=list(finite_vector(state,7,'snapshot state'))
        joint_command(dict(zip(JOINT_FIELDS,state,strict=True)))
        if state[6] != 0:
            raise ValueError('shadow A/B capture requires an already open gripper; no release is sent')
        now=time.perf_counter()
        if (len(times)!=3 or any(not math.isfinite(t) or t>now or now-t>.25 for t in times)
                or abs(times[0]-times[1])>.1):
            raise ValueError('snapshot timestamps outside existing freshness/skew limits')
        snapshot=dict(schema='joint_target_shadow_snapshot',snapshot_id=uuid.uuid4().hex,
                      state=state,sensor_times=times,captured_monotonic_s=now,
                      camera_set_sha256=hashlib.sha256(CAMERA_SET.read_bytes()).hexdigest(),
                      robot_motion_sent=False,gripper_command_sent=False,execution_eligible=False,
                      purpose='stationary saved-observation A/B only')
        for name,rgb in images.items():
            if not cv2.imwrite(str(args.output/f'{name}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
                raise RuntimeError('failed to save snapshot RGB')
        write(args.output/'snapshot.json',snapshot)
        render_review(args.output,snapshot)
        print(f'静止观察已保存，没有发送运动或夹爪命令。\n打开：google-chrome --new-window "{(args.output/"review.html").resolve().as_uri()}"')
    finally:
        station.close()


def reviewed_masks(snapshot, labels, *, allow_multiple=False):
    import cv2
    import numpy as np
    if (labels.get('schema')!='joint_target_shadow_selection' or labels.get('snapshot_id')!=snapshot['snapshot_id']
            or labels.get('human_reviewed') is not True):
        raise ValueError('explicit user review bound to this snapshot required')
    targets=labels.get('targets',[])
    if allow_multiple:
        from lerobot.bamboo_sorting.joint_target_separation import visible_seed_masks
        return list(visible_seed_masks(targets).items())
    if len(targets)!=2 or len({t.get('target_id') for t in targets})!=2:
        raise ValueError('two distinct targets required')
    masks=[]
    for target in targets:
        if not isinstance(target.get('target_id'),str) or not target['target_id'].strip():
            raise ValueError('nonempty target identity required')
        mask=np.zeros((480,640),np.uint8)
        polygons=target.get('polygons',[])
        if not polygons:
            raise ValueError('visible polygons required; do not fill occluded regions')
        for polygon in polygons:
            points=np.asarray(polygon,dtype=float)
            if (points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all()
                    or np.any(points<0) or np.any(points[:,0]>639) or np.any(points[:,1]>479)):
                raise ValueError('polygon vertices must lie inside captured RGB')
            cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],1)
        if not mask.any():raise ValueError('empty visible target mask')
        masks.append(mask.astype(bool))
    if np.any(masks[0]&masks[1]):
        raise ValueError('visible target masks overlap; select only the visible region of each object')
    return [(target['target_id'],mask) for target,mask in zip(targets,masks)]


def predict(args):
    import cv2
    import numpy as np
    from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_TASK, JOINT_FIELDS, finite_vector, joint_command
    from lerobot.bamboo_sorting.joint_target_policy import TargetInputSession
    snapshot=json.loads((args.snapshot/'snapshot.json').read_text())
    if snapshot.get('schema')!='joint_target_shadow_snapshot' or snapshot.get('execution_eligible') is not False:
        raise ValueError('non-executable stationary snapshot required')
    state=finite_vector(snapshot['state'],7,'snapshot state')
    joint_command(dict(zip(JOINT_FIELDS,state,strict=True)))
    if snapshot['state'][6]!=0:raise ValueError('open-gripper approach snapshot required')
    labels=json.loads(args.selections.read_text())
    masks=reviewed_masks(snapshot,labels)
    frame={'observation.state':np.asarray(snapshot['state'],dtype=np.float32),'task':JOINT_TASK}
    for name in ('global_rgb','grasp_rgb'):
        bgr=cv2.imread(str(args.snapshot/f'{name}.png'))
        if bgr is None:raise ValueError('missing snapshot RGB')
        frame['observation.images.'+name]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'selection.json',labels)
    pipe=None;responses=[]
    try:
        pipe=InferencePipe(args.output,remote_root=REMOTE_ROOT,checkpoint=CHECKPOINT,
            expected_sha256=MODEL_SHA,inference_script=INFERENCE_SCRIPT,
            target_conditioned=True,pythonpath=PYTHONPATH,return_action_chunk=True,request_timeout_s=5.,task=JOINT_TASK)
        prepared=[]
        for target,mask in masks:
            annotation=dict(schema='aubo_joint_target_frame',frame_id=snapshot['snapshot_id'],
                            target_id=target,status='visible',reviewed=True)
            value=TargetInputSession(target).prepare(frame,frame_id=snapshot['snapshot_id'],
                global_timestamp=snapshot['sensor_times'][0],visible_mask=mask,annotation=annotation)
            images={name:value['observation.images.'+name] for name in ('target_global_rgb','grasp_rgb')}
            prepared.append((target,images,value['target_input']))
            cv2.imwrite(str(args.output/f'target_{len(prepared)}.png'),cv2.cvtColor(images['target_global_rgb'],cv2.COLOR_RGB2BGR))
        for seed in (42,43,44):
            for target,images,metadata in prepared:
                response=pipe.predict(len(responses),snapshot['state'],images,codec='rgb_zlib',seed=seed,target_input=metadata)
                responses.append(dict(target_id=target,seed=seed,**response))
        differences=[]
        for i in range(0,len(responses),2):
            a,b=(np.asarray(responses[j]['raw_action_chunk']) for j in (i,i+1))
            differences.append(float(np.abs(a[:,:6]-b[:,:6]).mean()))
        write(args.output/'result.json',dict(schema='joint_target_shadow_result',snapshot_id=snapshot['snapshot_id'],
            checkpoint_sha256=MODEL_SHA,responses=responses,paired_seed_mean_joint_changes_deg=differences,
            robot_motion_sent=False,gripper_command_sent=False,execution_eligible=False,
            physical_success_verified=False,scope='saved real observation sensitivity only; no action execution or success test'))
        print('影子预测完成；没有发送运动或夹爪命令。结果：',args.output/'result.json')
    finally:
        if pipe is not None:pipe.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('plan','capture','predict'),default='plan')
    parser.add_argument('--onsite-confirmed',action='store_true')
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--selections',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.stage=='plan':
        print(json.dumps(dict(mode='stationary_target_shadow_only',checkpoint=CHECKPOINT,checkpoint_sha256=MODEL_SHA,
            camera_set=str(CAMERA_SET),capture_requires_manual_onsite_launch=True,
            prediction_uses_saved_snapshot=True,robot_motion_supported=False,gripper_writes_supported=False,
            full_grasp_ready=False),indent=2));return 0
    if args.output is None or args.output.exists():parser.error('new --output directory required')
    if args.stage=='capture':
        if not args.onsite_confirmed:parser.error('capture requires --onsite-confirmed')
        capture(args)
    else:
        if args.snapshot is None or args.selections is None:parser.error('--snapshot and --selections required')
        predict(args)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

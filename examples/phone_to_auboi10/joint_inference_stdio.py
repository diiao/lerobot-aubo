#!/usr/bin/env python
"""Offline SmolVLA inference over an SSH stdio pipe; no hardware, no TCP listener."""
import argparse
import base64
import contextlib
import hashlib
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path


def decode_images(images, codec):
    import cv2
    import numpy as np
    result = {}
    for name in ('global_rgb', 'grasp_rgb'):
        payload = base64.b64decode(images[name], validate=True)
        if codec == 'rgb_zlib':
            dec = zlib.decompressobj()
            raw = dec.decompress(payload, 480*640*3+1)
            if len(raw) != 480*640*3 or not dec.eof:
                raise ValueError('image byte count mismatch')
            image = np.frombuffer(raw, dtype=np.uint8).reshape(480, 640, 3)
        elif codec == 'jpeg95':
            bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != (480, 640, 3):
                raise ValueError('JPEG image shape mismatch')
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError('unsupported image codec')
        result[f'observation.images.{name}'] = image
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--task',help='Explicit task entered by the onsite operator')
    p.add_argument('--fast-matmul',action='store_true',help='Use high float32 matmul precision; weights stay unchanged')
    p.add_argument('--bf16', action='store_true', help='Offline comparison of CUDA BF16 autocast')
    p.add_argument('--return-action-chunk',action='store_true',help='Include the full 50-step prediction')
    args=p.parse_args()
    # Reject shared GPU occupancy; never kill, wait for, or modify other jobs.
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy: raise RuntimeError(f'GPU occupied: {busy}')
    digest=hashlib.sha256()
    with (args.checkpoint/'model.safetensors').open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=args.expected_sha256:raise ValueError('checkpoint hash mismatch')
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import batch_to_transition,transition_to_batch,policy_action_to_transition,transition_to_policy_action
        from lerobot.bamboo_sorting.aubo_joint_contract import joint_contract_record,JOINT_TASK,JOINT_IMAGE_KEYS
        task = JOINT_TASK if args.task is None else args.task
        if task != JOINT_TASK:
            raise ValueError('checkpoint supports only the recorded training task')
        from lerobot.bamboo_sorting.smolvla_joint_adapter import SmolVLAJointOfflineAdapter
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('high' if args.fast_matmul else 'highest')
        if json.loads((args.checkpoint/'aubo_joint_contract.json').read_text())!=joint_contract_record():
            raise ValueError('checkpoint contract mismatch')
        cfg=PreTrainedConfig.from_pretrained(args.checkpoint,local_files_only=True)
        cfg.device='cuda'
        model=SmolVLAPolicy.from_pretrained(args.checkpoint,config=cfg,local_files_only=True,strict=True)
        pre=PolicyProcessorPipeline.from_pretrained(args.checkpoint,'policy_preprocessor.json',
            overrides={'device_processor':{'device':'cuda'}},to_transition=batch_to_transition,to_output=transition_to_batch)
        post=PolicyProcessorPipeline.from_pretrained(args.checkpoint,'policy_postprocessor.json',
            to_transition=policy_action_to_transition,to_output=transition_to_policy_action)
        adapter=SmolVLAJointOfflineAdapter(model,pre,post,contract=joint_contract_record())
        if args.bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unsupported on this GPU')
        def predict(frame):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                return adapter(frame)
        warm={'observation.state':np.array([-65.29,-5.88,113.77,31.07,90.88,-185.32,0],dtype=np.float32),
              'task':task,**{f'observation.images.{k}':np.zeros((480,640,3),dtype=np.uint8) for k in JOINT_IMAGE_KEYS}}
        for _ in range(2):predict(warm)
        model.reset()
    print(json.dumps({'ready':True,'checkpoint_sha256':digest.hexdigest(),'contract':joint_contract_record(),
                      'bf16':args.bf16,'task':task,'action_chunk_available':args.return_action_chunk}),flush=True)
    for line in sys.stdin:
        start=time.perf_counter()
        try:
            if len(line)>6000000:raise ValueError('request too large')
            req=json.loads(line)
            frame={'observation.state':np.asarray(req['state'],dtype=np.float32),'task':task}
            frame.update(decode_images(req['images'], req.get('codec', 'rgb_zlib')))
            if 'seed' in req:
                if type(req['seed']) is not int or not 0 <= req['seed'] < 2**32:
                    raise ValueError('invalid comparison seed')
                torch.manual_seed(req['seed'])
            decoded_at = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):decoded=predict(frame)
            response={'id':req['id'],'checkpoint_sha256':digest.hexdigest(),'action':decoded[0].tolist(),
                'raw_action':adapter.raw_actions[0].tolist(),'shape':list(decoded.shape),'inference_s':time.perf_counter()-start,
                'decode_s':decoded_at-start,'model_path_s':time.perf_counter()-decoded_at}
            if args.return_action_chunk:
                response.update(action_chunk=decoded.tolist(),raw_action_chunk=adapter.raw_actions.tolist())
            print(json.dumps(response,allow_nan=False),flush=True)
        except Exception as exc:
            print(json.dumps({'error':f'{type(exc).__name__}: {exc}'}),flush=True)
            return 1
    return 0

if __name__=='__main__':raise SystemExit(main())

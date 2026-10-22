"""SSH stdio inference pipe shared by trial, live and smooth entries.

The remote worker owns model loading. This module validates identity, shapes
and target evidence before any caller consumes a prediction.
"""

import json
import math
import selectors
import shlex
import subprocess
import time

from ..aubo_joint_contract import finite_vector, joint_contract_record
from .joint_smooth import validated_action_chunk
from .joint_target_runtime import TargetForegroundMissing
from ..mask_codec import decode_mask
from ..model_profiles import LEGACY_MODEL_SHA, LEGACY_REMOTE_PYTHON, LEGACY_REMOTE_ROOT

from .station import encode_images


class InferencePipe:
    def __init__(self, output, *, fast_matmul=False, bf16=False, cuda_graph=False, task=None,
                 remote_root=LEGACY_REMOTE_ROOT, checkpoint=None, expected_sha256=LEGACY_MODEL_SHA,
                 inference_script=None, return_action_chunk=False, request_timeout_s=2.,
                 target_conditioned=False, pythonpath=None, live_target_seed=None, segmentation_checkpoint=None):
        if (live_target_seed is None) != (segmentation_checkpoint is None):
            raise ValueError('live target seed and segmentation checkpoint must be provided together')
        if live_target_seed is not None and not target_conditioned:
            raise ValueError('live segmentation requires target-conditioned model')
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError('inference request timeout must be finite and positive')
        self.request_timeout_s = request_timeout_s
        self.log = (output / 'inference_stderr.log').open('w')
        self.expected_sha256 = expected_sha256
        self.return_action_chunk = return_action_chunk
        self.target_conditioned = target_conditioned
        self.live_target_seed = live_target_seed
        self.contract = joint_contract_record()
        if target_conditioned:
            from ..policy.joint_target_policy import target_policy_contract_record
            self.contract = target_policy_contract_record()
        deployment = 'joint_live_bf16_20260922' if bf16 else 'joint_live_matmul_20260922'
        script = 'joint_inference_stdio.py'
        if cuda_graph:
            deployment, script = 'joint_live_graph_20260922', 'graph_inference_stdio.py'
        if task is not None:
            deployment, script = 'joint_live_manual_task_20260922', 'joint_inference_stdio.py'
        script_path = inference_script or f'{remote_root}/artifacts/{deployment}/{script}'
        checkpoint_path = checkpoint or f'{remote_root}/outputs/joint_legacy_v2_run01/final'
        args = ['env', f'PYTHONPATH={pythonpath or remote_root + "/src"}', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                'HF_HUB_DISABLE_TELEMETRY=1', 'WANDB_MODE=disabled', LEGACY_REMOTE_PYTHON,
                '-u', str(script_path), '--checkpoint', str(checkpoint_path),
                '--expected-sha256', self.expected_sha256]
        if fast_matmul:
            args.append('--fast-matmul')
        if bf16:
            args.append('--bf16')
        if return_action_chunk:
            args.append('--return-action-chunk')
        if target_conditioned:
            args.append('--target-conditioned')
        if live_target_seed is not None:
            args.extend(['--target-segmentation-checkpoint', str(segmentation_checkpoint)])
        if task is not None:
            args.extend(['--task', task])
        self.process = subprocess.Popen(['ssh', '-T', 'gpu', shlex.join(args)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        try:
            ready = self.receive(120)
            if ready.get('ready') is not True or ready.get('checkpoint_sha256') != self.expected_sha256 or ready.get('contract') != self.contract:
                raise ValueError('inference identity mismatch')
            if bool(ready.get('bf16', False)) != bf16:
                raise ValueError('inference precision mismatch')
            if task is not None and ready.get('task') != task:
                raise ValueError('inference task mismatch')
            if return_action_chunk and ready.get('action_chunk_available') is not True:
                raise ValueError('inference worker does not provide full actions')
            if bool(ready.get('target_live', False)) != (live_target_seed is not None):
                raise ValueError('live target protocol mismatch')
            if live_target_seed is not None and ready.get('segmentation_checkpoint') != str(segmentation_checkpoint):
                raise ValueError('segmentation checkpoint mismatch')
            if live_target_seed and live_target_seed.get('motion_tracking') and ready.get('target_motion_available') is not True:
                raise ValueError('inference worker does not support target motion references')
        except BaseException:
            self.close()
            raise

    def receive(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError('inference pipe timeout')
        line = self.process.stdout.readline(100000)
        if not line:
            raise RuntimeError('inference exited; inspect inference_stderr.log')
        result = json.loads(line)
        if 'error' in result and result.get('error_code') != TargetForegroundMissing.code:
            raise RuntimeError(result['error'])
        return result

    def predict(self, index, state, images, *, codec='jpeg95', seed=None, target_input=None, target_motion=None):
        start = time.perf_counter()
        if self.target_conditioned:
            keys = ('global_rgb', 'grasp_rgb') if self.live_target_seed is not None else self.contract['image_keys']
            if (set(images) != set(keys) or not isinstance(target_input, dict)
                    or target_input.get('phase') not in ('approach', 'transport')
                    or not target_input.get('target_id') or not target_input.get('frame_id')):
                raise ValueError('prepared target images and explicit target metadata required')
        elif target_input is not None:
            raise ValueError('target metadata requires a target-conditioned checkpoint')
        packet = {'id': index, 'state': state, 'codec': codec, 'images': encode_images(images, codec)}
        if self.target_conditioned:
            packet.update(target_contract=self.contract, target_input=target_input)
        if self.live_target_seed is not None:
            packet['target_seed'] = self.live_target_seed
        if target_motion is not None:
            if (not self.live_target_seed or not self.live_target_seed.get('motion_tracking')
                    or target_input['phase'] != 'approach'
                    or target_motion.get('frame_id') != target_input['frame_id']):
                raise ValueError('unexpected or mismatched motion reference')
            packet['target_motion'] = target_motion
        if seed is not None:
            packet['seed'] = seed  # Offline paired comparison only; live requests omit it.
        encoded_at = time.perf_counter()
        wire = json.dumps(packet) + '\n'
        self.process.stdin.write(wire)
        self.process.stdin.flush()
        result = self.receive(self.request_timeout_s)
        if 'error' in result:
            evidence = result.get('target_evidence', {})
            if (result.get('error_code') != TargetForegroundMissing.code
                    or result.get('id') != index or result.get('checkpoint_sha256') != self.expected_sha256
                    or not self.live_target_seed or self.live_target_seed.get('target_selection_mode') != 'multi_foreground'
                    or result.get('target_input') != target_input or target_input.get('phase') != 'approach'
                    or evidence.get('target_id') != target_input['target_id']
                    or evidence.get('frame_id') != target_input['frame_id']
                    or evidence.get('phase') != 'approach'
                    or evidence.get('seed_snapshot_id') != self.live_target_seed['snapshot_id']
                    or evidence.get('target_selection_mode') != 'multi_foreground'
                    or evidence.get('source') != 'same_frame_segmentation'
                    or evidence.get('human_reviewed') is not False
                    or evidence.get('association', {}).get('status') != 'missing'
                    or decode_mask(evidence.get('mask_png'), allow_empty=True).any()):
                raise ValueError('invalid empty-target response identity or evidence')
            raise TargetForegroundMissing(evidence)
        if result.get('id') != index or result.get('checkpoint_sha256') != self.expected_sha256 or result.get('shape') != [50, 7]:
            raise ValueError('invalid prediction identity or shape')
        if self.live_target_seed is not None:
            evidence = result.get('target_evidence', {})
            if (result.get('target_input') != target_input or evidence.get('frame_id') != target_input['frame_id']
                    or evidence.get('target_id') != target_input['target_id'] or evidence.get('phase') != target_input['phase']
                    or evidence.get('seed_snapshot_id') != self.live_target_seed['snapshot_id']
                    or evidence.get('human_reviewed') is not False):
                raise ValueError('live target evidence mismatch')
            selection_mode = self.live_target_seed.get('target_selection_mode')
            separated = selection_mode in ('paired_foreground', 'multi_foreground')
            if separated and evidence.get('target_selection_mode') != selection_mode:
                raise ValueError('requested target separation was not used')
            if target_input['phase'] == 'approach':
                decode_mask(evidence.get('mask_png'))
                if self.live_target_seed.get('motion_tracking'):
                    motion = evidence.get('motion_reference', {})
                    if (motion.get('frame_id') != target_input['frame_id']
                            or motion.get('source') != 'rgb_optical_flow'
                            or motion.get('human_reviewed') is not False
                            or not motion.get('current_foreground_support')):
                        raise ValueError('motion tracking evidence missing')
                if (evidence.get('source') != 'same_frame_segmentation'
                        or evidence.get('association', {}).get('status') != ('separated' if separated else 'matched')):
                    raise ValueError('fresh target association required')
            elif evidence.get('source') != 'raw_rgb_after_accepted_close' or evidence.get('mask_png') is not None:
                raise ValueError('invalid transport evidence')
        if self.return_action_chunk:
            # Validate the full wire response once, before any consumer uses it.
            result['action_chunk'] = validated_action_chunk(result)
        else:
            finite_vector(result['raw_action'], 7, 'raw model action')
            finite_vector(result['action'], 7, 'decoded model action')
        result['transport'] = {'codec': codec, 'request_bytes': len(wire.encode()),
            'encode_s': encoded_at - start, 'roundtrip_s': time.perf_counter() - start}
        return result

    def close(self):
        # EOF ends only our stdio worker. Never pkill/kill a pre-existing task.
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.log.close()

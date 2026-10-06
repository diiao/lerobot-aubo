"""Shared target-conditioned data and numerical inference boundary; no hardware.

Raw captures stay legacy dual RGB. A reviewed sidecar binds each approach mask
to a dataset frame. Transport uses raw RGB after the first accepted close.
"""

import copy
import json
from pathlib import Path

import cv2
import numpy as np

from .aubo_joint_contract import GRIPPER_INDEX, joint_command
from .joint_target import TARGET_IMAGE_KEY, prepare_target_images, target_contract_record
from .smolvla_joint_adapter import SmolVLAJointOfflineAdapter


def target_policy_contract_record():
    record = target_contract_record()
    record['schema_version'] = 'AuboI10JointTargetApproach'
    record['target_conditioning'].update(
        phase_rule='outline until global RGB timestamp exceeds first accepted close timestamp; then raw RGB until reset',
        phase_source='accepted command history only; never current or future predicted actions',
        unavailable_target='reject missing approach mask; transport needs no mask',
    )
    return record


def is_transport(global_time, accepted_close_time):
    if not np.isfinite(global_time) or (accepted_close_time is not None and not np.isfinite(accepted_close_time)):
        raise ValueError('finite monotonic timestamps required')
    return accepted_close_time is not None and global_time > accepted_close_time


def prepare_policy_images(images, mask, annotation, *, frame_id, target_id, transport):
    if not transport:
        result = prepare_target_images(images, mask, annotation or {}, frame_id=frame_id, target_id=target_id)
    else:
        if not frame_id or not target_id:
            raise ValueError('frame and target identity required')
        for name in ('global_rgb', 'grasp_rgb'):
            image = images.get(name)
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (480, 640, 3):
                raise ValueError(f'invalid RGB image: {name}')
        result = {'images': {TARGET_IMAGE_KEY: images['global_rgb'].copy(),
                             'grasp_rgb': images['grasp_rgb'].copy()}, 'preview_only': False}
    result.update(contract=target_policy_contract_record(), target_input={
        'frame_id': frame_id, 'target_id': target_id, 'phase': 'transport' if transport else 'approach'})
    return result


class TargetInputSession:
    """One selected object per cycle. Call accept_command only AFTER SDK success.

    Not a controller. A new instance starts a new pick; reopening does not reset
    identity/phase, and prediction alone must never advance the phase.
    """
    def __init__(self, target_id):
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError('target_id required')
        self.target_id = target_id
        self.accepted_close_time = None

    def accept_command(self, action, *, accepted_monotonic_s):
        values = joint_command(action)
        if not np.isfinite(accepted_monotonic_s):
            raise ValueError('finite accepted command time required')
        if values['gripper_pos'] == 100 and self.accepted_close_time is None:
            self.accepted_close_time = float(accepted_monotonic_s)

    def prepare(self, frame, *, frame_id, global_timestamp, visible_mask=None, annotation=None):
        result = prepare_policy_images(
            {name: frame[f'observation.images.{name}'] for name in ('global_rgb', 'grasp_rgb')},
            visible_mask, annotation, frame_id=frame_id, target_id=self.target_id,
            transport=is_transport(global_timestamp, self.accepted_close_time))
        return {'observation.state': frame['observation.state'], 'task': frame['task'],
                'target_contract': result['contract'], 'target_input': result['target_input'],
                **{f'observation.images.{key}': value for key, value in result['images'].items()}}


class SmolVLAJointTargetOfflineAdapter(SmolVLAJointOfflineAdapter):
    image_keys = (TARGET_IMAGE_KEY, 'grasp_rgb')
    contract_record = staticmethod(target_policy_contract_record)

    def __call__(self, frame):
        self.raw_actions = None
        metadata = frame.get('target_input', {})
        if (frame.get('target_contract') != self.contract_record()
                or metadata.get('phase') not in ('approach', 'transport')
                or not metadata.get('frame_id') or not metadata.get('target_id')):
            raise ValueError('prepared target input and explicit phase contract required')
        return super().__call__(frame)


def read_visible_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.shape != (480, 640) or not np.isin(mask, [0, 1, 255]).all() or not mask.any():
        raise ValueError(f'nonempty binary visible mask required: {path}')
    return mask != 0


def prepare_target_source(record, annotation_path):
    """Bind a reviewed sidecar to previously audited accepted commands/frames.

    No propagation, re-inference or action relabeling. Preflight checks every
    approach frame; missing human-reviewed contours remain an explicit gap.
    """
    path = Path(annotation_path).resolve()
    spec = json.loads(path.read_text())
    if (spec.get('schema') != 'aubo_joint_target_annotations'
            or spec.get('source_dataset_root') != record['source_dataset_root']):
        raise ValueError('target sidecar schema/source dataset mismatch')
    session = json.loads((Path(record['evidence_root']) / 'session.json').read_text())
    saved = {a['episode_index']: a for a in session['attempts'] if a['status'] == 'saved_pending_finalize'}
    episodes = spec.get('episodes', [])
    annotations = {a['episode_index']: a for a in episodes}
    if len(annotations) != len(episodes):
        raise ValueError('duplicate target episode')
    prepared = []
    for ep in record['episodes']:
        capture, labels = saved[ep], annotations.get(ep, {})
        target = capture['metadata'].get('target_id')
        if (not target or labels.get('target_id') != target
                or labels.get('scene_id') != capture['metadata']['scene_id']
                or labels.get('reviewed') is not True):
            raise ValueError(f'episode {ep}: reviewed target must match capture metadata')
        frames = labels.get('frames', [])
        by_index = {a['frame_index']: a for a in frames}
        if len(by_index) != len(frames):
            raise ValueError('duplicate target frame')
        close_time = None
        timestamps = capture['sensor_and_suction_evidence']['timestamp_journal']['frames']
        needed = set()
        for row, clock in zip(capture['joint_frames'], timestamps, strict=True):
            index = row['frame_index']
            frame_id = f'{ep}/global_rgb/{index}'
            global_time = clock['timestamps']['global_rgb']['host_receive_monotonic_s']
            transport = is_transport(global_time, close_time)
            annotation = None
            if not transport:
                needed.add(index)
                annotation = copy.deepcopy(by_index.get(index, {}))
                if (annotation.get('schema') != 'aubo_joint_target_frame'
                        or annotation.get('frame_id') != frame_id
                        or annotation.get('target_id') != target
                        or annotation.get('status') != 'visible' or annotation.get('reviewed') is not True):
                    raise ValueError(f'episode {ep} frame {index}: reviewed same-frame mask required')
                annotation['mask_path'] = str((path.parent / annotation['mask_path']).resolve())
                read_visible_mask(annotation['mask_path'])
            prepared.append({'episode_index': ep, 'frame_index': index, 'frame_id': frame_id,
                             'target_id': target, 'transport': transport, 'annotation': annotation})
            # Current accepted action is applied AFTER preparing its observation.
            trace = row['joint_command_trace']
            if row['action'][GRIPPER_INDEX] == 100 and close_time is None:
                if trace['return_code'] != 0:
                    raise ValueError('close command was not accepted')
                close_time = trace['accepted_monotonic_s']
        if set(by_index) != needed:
            raise ValueError('target sidecar must contain exactly the approach frames')
    if len(prepared) != record['frames']:
        raise ValueError('target annotation frame count mismatch')
    return {**record, 'target_annotations': str(path), 'target_frames': prepared}


class JointTargetDataset:
    """Lazy view over original LeRobot samples; state/actions/padding untouched."""
    def __init__(self, dataset, frames):
        self.dataset = dataset
        self.frames = {(f['episode_index'], f['frame_index']): f for f in frames}
        if len(self.frames) != len(frames) or len(dataset) != len(frames):
            raise ValueError('target frame coverage mismatch')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch
        raw = self.dataset[index]
        spec = self.frames[(int(raw['episode_index']), int(raw['frame_index']))]
        images = {}
        for name in ('global_rgb', 'grasp_rgb'):
            tensor = raw[f'observation.images.{name}']
            if (tuple(tensor.shape) != (3, 480, 640) or not torch.isfinite(tensor).all()
                    or tensor.min() < 0 or tensor.max() > 1):
                raise ValueError('dataset RGB must be CHW float in [0,1]')
            images[name] = tensor.mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        annotation = spec['annotation']
        mask = None if annotation is None else read_visible_mask(annotation['mask_path'])
        result = prepare_policy_images(images, mask, annotation, frame_id=spec['frame_id'],
                                       target_id=spec['target_id'], transport=spec['transport'])
        output = dict(raw)
        original = output.pop('observation.images.global_rgb')
        # Preserve decoded pixels exactly outside the boundary and the entire wrist image.
        marked = original.clone()
        changed = np.any(result['images'][TARGET_IMAGE_KEY] != images['global_rgb'], axis=-1)
        marked[:, torch.from_numpy(changed)] = torch.tensor([1., 0., 1.], dtype=marked.dtype)[:, None]
        output[f'observation.images.{TARGET_IMAGE_KEY}'] = marked
        return output

"""Local reviewed-seed setup for one live target pick; no hardware access."""
import json
import numpy as np

from run_joint_target_shadow import REMOTE_ROOT, MODEL_SHA, CHECKPOINT, PYTHONPATH, reviewed_masks
from lerobot.bamboo_sorting.joint_target_runtime import encode_mask

LIVE_ROOT = REMOTE_ROOT + '/artifacts/joint_target_live_service'
SEGMENTATION_CHECKPOINT = '/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/outputs/short_strip_mask2former_baseline/best'
STACK_REMOTE_ROOT = '/home/rentao/program/lerobot-aubo-joint-stack'
STACK_CHECKPOINT = STACK_REMOTE_ROOT + '/outputs/joint_stack_target_smolvla/final'
STACK_MODEL_SHA = 'c45fdeb1d36f71749008357956b6f3ca90a042e7c8e049a99d307204363fb0e9'
STACK_SERVICE_ROOT = STACK_REMOTE_ROOT + '/artifacts/joint_stack_live_service'


def prepare_live_setup(snapshot_path, selection_path, target_id, *, paired_separation=False, multi_separation=False,
                       model='target-pair', remaining_target_ids=None, motion_tracking=False):
    if model not in ('target-pair', 'target-stack'):
        raise ValueError('unknown target model')
    if model == 'target-stack' and not multi_separation:
        raise ValueError('stack model requires visible seeds for all remaining strips')
    if motion_tracking and (model != 'target-stack' or not multi_separation):
        raise ValueError('motion tracking requires target-stack multi separation')
    if paired_separation and multi_separation:
        raise ValueError('choose one target separation mode')
    snapshot = json.loads((snapshot_path / 'snapshot.json').read_text())
    if snapshot.get('schema') != 'joint_target_shadow_snapshot' or snapshot.get('execution_eligible') is not False:
        raise ValueError('reviewed stationary seed snapshot required; old actions are never executed')
    labels = json.loads(selection_path.read_text())
    masks = dict(reviewed_masks(snapshot, labels, allow_multiple=multi_separation))
    if remaining_target_ids is not None:
        if (not multi_separation or not remaining_target_ids
                or len(set(remaining_target_ids)) != len(remaining_target_ids)
                or not set(remaining_target_ids) <= set(masks)):
            raise ValueError('distinct remaining targets from reviewed selection required')
        masks = {target: masks[target] for target in remaining_target_ids}
    if target_id not in masks:
        raise ValueError('target identity absent from reviewed selection')
    other = np.zeros((480, 640), bool)
    for target, mask in masks.items():
        if target != target_id:
            other |= mask
    seed = dict(schema='joint_target_live_seed', snapshot_id=snapshot['snapshot_id'],
                target_id=target_id, human_reviewed=True, mask_png=encode_mask(masks[target_id]),
                excluded_mask_png=encode_mask(other, allow_empty=multi_separation))
    if motion_tracking:
        seed['motion_tracking'] = 'rgb_optical_flow'
    service_root = LIVE_ROOT
    if paired_separation:
        seed.update(target_selection_mode='paired_foreground',
                    paired_targets=[dict(target_id=t, mask_png=encode_mask(masks[t])) for t in sorted(masks)])
        service_root += '/paired_foreground'
    if multi_separation:
        seed.update(target_selection_mode='multi_foreground',
                    visible_targets=[dict(target_id=t, mask_png=encode_mask(masks[t])) for t in sorted(masks)])
        service_root += '/multi_foreground'
    options = dict(remote_root=REMOTE_ROOT, checkpoint=CHECKPOINT, expected_sha256=MODEL_SHA,
                   inference_script=service_root + '/joint_inference_stdio.py',
                   pythonpath=service_root + '/src:' + PYTHONPATH,
                   target_conditioned=True, return_action_chunk=True,
                   live_target_seed=seed, segmentation_checkpoint=SEGMENTATION_CHECKPOINT)
    if model == 'target-stack':
        options.update(remote_root=STACK_REMOTE_ROOT, checkpoint=STACK_CHECKPOINT,
                       expected_sha256=STACK_MODEL_SHA,
                       inference_script=STACK_SERVICE_ROOT + '/joint_inference_stdio.py',
                       pythonpath=STACK_REMOTE_ROOT + '/src:/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/deps')
    return seed, options

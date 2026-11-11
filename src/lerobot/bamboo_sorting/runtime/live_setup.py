"""Local reviewed-seed setup for one live target pick; no hardware access."""

import json

import numpy as np

from ..model_profiles import (
    SEGMENTATION_CHECKPOINT,
    TARGET_PAIR_CHECKPOINT,
    TARGET_PAIR_LIVE_ROOT,
    TARGET_PAIR_MODEL_SHA,
    TARGET_PAIR_PYTHONPATH,
    TARGET_PAIR_REMOTE_ROOT,
    TARGET_STACK_CHECKPOINT,
    TARGET_STACK_MODEL_SHA,
    TARGET_STACK_PYTHONPATH,
    TARGET_STACK_REMOTE_ROOT,
    TARGET_STACK_SEGMENTATION_CHECKPOINT,
    TARGET_STACK_SERVICE_ROOT,
    TARGET_STACK_THIN_CHECKPOINT,
    TARGET_STACK_THIN_MODEL_SHA,
    TARGET_STACK_THIN_PYTHONPATH,
    TARGET_STACK_THIN_SERVICE_ROOT,
    TARGET_STACK_THREE_FOUR_CHECKPOINT,
    TARGET_STACK_THREE_FOUR_MODEL_SHA,
    TARGET_STACK_LOCAL_ALIGNMENT_CHECKPOINT,
    TARGET_STACK_LOCAL_ALIGNMENT_MODEL_SHA,
    TARGET_STACK_LOCAL_ALIGNMENT_SERVICE_ROOT,
    TARGET_STACK_LOCAL_ALIGNMENT_PYTHONPATH,
    TARGET_STACK_LOCAL_ALIGNMENT_70000_CHECKPOINT,
    TARGET_STACK_LOCAL_ALIGNMENT_70000_MODEL_SHA,
    TARGET_STACK_LOCAL_ALIGNMENT_70000_SERVICE_ROOT,
    TARGET_STACK_LOCAL_ALIGNMENT_70000_SEGMENTATION_SCORE_THRESHOLD,
    TARGET_STACK_LOCAL_ALIGNMENT_70000_PYTHONPATH,
    TARGET_STACK_OUTLINE_100000_CHECKPOINT,
    TARGET_STACK_OUTLINE_100000_MODEL_SHA,
    TARGET_STACK_OUTLINE_100000_SERVICE_ROOT,
    TARGET_STACK_OUTLINE_100000_SEGMENTATION_SCORE_THRESHOLD,
    TARGET_STACK_OUTLINE_100000_PYTHONPATH,
    TARGET_STACK_PI05_CHECKPOINT,
    TARGET_STACK_PI05_MODEL_SHA,
    TARGET_STACK_PI05_PYTHONPATH,
    TARGET_STACK_PI05_REMOTE_PYTHON,
    TARGET_STACK_PI05_REMOTE_ROOT,
    TARGET_STACK_PI05_SERVICE_ROOT,
    TARGET_STACK_PI05_FP32_CHECKPOINT,
    TARGET_STACK_PI05_FP32_MODEL_SHA,
    TARGET_STACK_PI05_FP32_INFERENCE_SCRIPT,
    TARGET_STACK_PI05_FP32_PYTHONPATH,
    TARGET_STACK_PI05_FP32_21000_CHECKPOINT,
    TARGET_STACK_PI05_FP32_21000_MODEL_SHA,
    TARGET_STACK_PI05_FP32_21000_INFERENCE_SCRIPT,
    TARGET_STACK_PI05_FP32_21000_PYTHONPATH,
)
from ..perception.target_seed import reviewed_masks
from ..mask_codec import encode_mask

LIVE_ROOT = TARGET_PAIR_LIVE_ROOT
STACK_REMOTE_ROOT = TARGET_STACK_REMOTE_ROOT
STACK_CHECKPOINT = TARGET_STACK_CHECKPOINT
STACK_MODEL_SHA = TARGET_STACK_MODEL_SHA
STACK_SERVICE_ROOT = TARGET_STACK_SERVICE_ROOT


def prepare_live_setup(snapshot_path, selection_path, target_id, *, paired_separation=False, multi_separation=False,
                       model='target-pair', remaining_target_ids=None, motion_tracking=False, outline_width_px=2,
                       stack_model='original'):
    if model not in ('target-pair', 'target-stack'):
        raise ValueError('unknown target model')
    if type(outline_width_px) is not int or outline_width_px not in (1, 2):
        raise ValueError('target outline width must be 1 or 2 pixels')
    if outline_width_px == 1 and model != 'target-stack':
        raise ValueError('thin outline checkpoint requires target-stack')
    if stack_model not in ('original', 'three-four', 'local-alignment', 'local-alignment-70000', 'target-outline-100000', 'pi05-29000', 'pi05-fp32-128', 'pi05-fp32-21000'):
        raise ValueError('unknown stack model')
    if stack_model != 'original' and (model != 'target-stack' or outline_width_px != 2):
        raise ValueError(f'{stack_model} checkpoint requires target-stack with 2px outline')
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
    options = dict(remote_root=TARGET_PAIR_REMOTE_ROOT, checkpoint=TARGET_PAIR_CHECKPOINT,
                   expected_sha256=TARGET_PAIR_MODEL_SHA,
                   inference_script=service_root + '/joint_inference_stdio.py',
                   pythonpath=service_root + '/src:' + TARGET_PAIR_PYTHONPATH,
                   target_conditioned=True, return_action_chunk=True,
                   live_target_seed=seed, segmentation_checkpoint=SEGMENTATION_CHECKPOINT)
    if model == 'target-stack':
        options.update(remote_root=TARGET_STACK_REMOTE_ROOT, checkpoint=TARGET_STACK_CHECKPOINT,
                       expected_sha256=TARGET_STACK_MODEL_SHA,
                       inference_script=TARGET_STACK_SERVICE_ROOT + '/joint_inference_stdio.py',
                       pythonpath=TARGET_STACK_PYTHONPATH,
                       segmentation_checkpoint=TARGET_STACK_SEGMENTATION_CHECKPOINT)
        if outline_width_px == 1:
            options.update(checkpoint=TARGET_STACK_THIN_CHECKPOINT, expected_sha256=TARGET_STACK_THIN_MODEL_SHA,
                           inference_script=TARGET_STACK_THIN_SERVICE_ROOT + '/joint_inference_stdio.py',
                           pythonpath=TARGET_STACK_THIN_PYTHONPATH, target_outline_width_px=1)
        if stack_model == 'three-four':
            options.update(checkpoint=TARGET_STACK_THREE_FOUR_CHECKPOINT,
                           expected_sha256=TARGET_STACK_THREE_FOUR_MODEL_SHA)
        if stack_model == 'local-alignment':
            options.update(checkpoint=TARGET_STACK_LOCAL_ALIGNMENT_CHECKPOINT,
                           expected_sha256=TARGET_STACK_LOCAL_ALIGNMENT_MODEL_SHA,
                           inference_script=TARGET_STACK_LOCAL_ALIGNMENT_SERVICE_ROOT + '/joint_inference_stdio.py',
                           pythonpath=TARGET_STACK_LOCAL_ALIGNMENT_PYTHONPATH)
        if stack_model == 'local-alignment-70000':
            seed['segmentation_score_threshold'] = TARGET_STACK_LOCAL_ALIGNMENT_70000_SEGMENTATION_SCORE_THRESHOLD
            options.update(checkpoint=TARGET_STACK_LOCAL_ALIGNMENT_70000_CHECKPOINT,
                           expected_sha256=TARGET_STACK_LOCAL_ALIGNMENT_70000_MODEL_SHA,
                           inference_script=TARGET_STACK_LOCAL_ALIGNMENT_70000_SERVICE_ROOT + '/joint_inference_stdio.py',
                           pythonpath=TARGET_STACK_LOCAL_ALIGNMENT_70000_PYTHONPATH)
        if stack_model == 'target-outline-100000':
            seed['segmentation_score_threshold'] = TARGET_STACK_OUTLINE_100000_SEGMENTATION_SCORE_THRESHOLD
            options.update(checkpoint=TARGET_STACK_OUTLINE_100000_CHECKPOINT,
                           expected_sha256=TARGET_STACK_OUTLINE_100000_MODEL_SHA,
                           inference_script=TARGET_STACK_OUTLINE_100000_SERVICE_ROOT + '/joint_inference_stdio.py',
                           pythonpath=TARGET_STACK_OUTLINE_100000_PYTHONPATH)
        if stack_model == 'pi05-29000':
            options.update(remote_root=TARGET_STACK_PI05_REMOTE_ROOT,
                           remote_python=TARGET_STACK_PI05_REMOTE_PYTHON,
                           checkpoint=TARGET_STACK_PI05_CHECKPOINT, expected_sha256=TARGET_STACK_PI05_MODEL_SHA,
                           inference_script=TARGET_STACK_PI05_SERVICE_ROOT + '/joint_inference_stdio.py',
                           pythonpath=TARGET_STACK_PI05_PYTHONPATH)
        if stack_model == 'pi05-fp32-128':
            from ..policy.pi05_joint_adapter import FP32_EXPERT_PRECISION
            options.update(remote_root=TARGET_STACK_PI05_REMOTE_ROOT,
                           remote_python=TARGET_STACK_PI05_REMOTE_PYTHON,
                           checkpoint=TARGET_STACK_PI05_FP32_CHECKPOINT,
                           expected_sha256=TARGET_STACK_PI05_FP32_MODEL_SHA,
                           inference_script=TARGET_STACK_PI05_FP32_INFERENCE_SCRIPT,
                           pythonpath=TARGET_STACK_PI05_FP32_PYTHONPATH,
                           expected_precision_contract=dict(FP32_EXPERT_PRECISION))
        if stack_model == 'pi05-fp32-21000':
            from ..policy.pi05_joint_adapter import FP32_EXPERT_PRECISION
            options.update(remote_root=TARGET_STACK_PI05_REMOTE_ROOT,
                           remote_python=TARGET_STACK_PI05_REMOTE_PYTHON,
                           checkpoint=TARGET_STACK_PI05_FP32_21000_CHECKPOINT,
                           expected_sha256=TARGET_STACK_PI05_FP32_21000_MODEL_SHA,
                           inference_script=TARGET_STACK_PI05_FP32_21000_INFERENCE_SCRIPT,
                           pythonpath=TARGET_STACK_PI05_FP32_21000_PYTHONPATH,
                           expected_precision_contract=dict(FP32_EXPERT_PRECISION))
    return seed, options

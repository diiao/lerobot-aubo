"""Checkpoint, service and remote-environment identities used by AUBO entries.

Plain constants only: no model, network or hardware access. Keep each profile
unchanged when a checkpoint moves; deployed services use the same values.
"""

LEGACY_REMOTE_ROOT = '/home/rentao/program/lerobot-aubo-smolvla-joint-legacy-v2-20260922'
LEGACY_REMOTE_PYTHON = '/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv/bin/python'
LEGACY_MODEL_SHA = '95a596e2a98dbe60623f801b2d12a6028132ad321a0bcb2ac7ae334ef5613aad'
LEGACY_DEFAULT_CHECKPOINT = LEGACY_REMOTE_ROOT + '/outputs/joint_legacy_v2_run01/final'

MIXED_REMOTE_ROOT = '/home/rentao/program/lerobot-aubo-smolvla-joint-mixed-20260927'
MIXED_MODEL_SHA = '79eb23606f6aa4e77484dd4bbf16a8cd273bb6a17ccfc9668ef44a4e2ed22ac7'
MIXED_CHECKPOINT = MIXED_REMOTE_ROOT + '/outputs/joint_mixed_single60_double50_run01/final'
MIXED_INFERENCE_SCRIPT = (
    MIXED_REMOTE_ROOT + '/artifacts/aubo_joint_two_strip_top45_pilot_20260927/joint_inference_stdio.py'
)

BOTH_ORDERS_REMOTE_ROOT = '/home/rentao/program/lerobot-aubo-smolvla-joint-both-orders-20260928'
BOTH_ORDERS_MODEL_SHA = 'c24b61b8d902d395c88890a3243879507d75333d5deb0bc99c13572ecddbfc3b'
BOTH_ORDERS_CHECKPOINT = (
    BOTH_ORDERS_REMOTE_ROOT + '/outputs/joint_mixed_single60_top45_50_top90_50_30k_run01/final'
)
BOTH_ORDERS_INFERENCE_SCRIPT = (
    BOTH_ORDERS_REMOTE_ROOT + '/artifacts/aubo_joint_both_orders_20260928/joint_inference_stdio.py'
)
BOTH_ORDERS_APPROACH_SCRIPT = (
    BOTH_ORDERS_REMOTE_ROOT + '/artifacts/joint_age_aligned_approach_20260928/joint_inference_stdio.py'
)

TARGET_PAIR_REMOTE_ROOT = '/home/rentao/program/lerobot-aubo-joint-target-pair'
TARGET_PAIR_MODEL_SHA = '93fd4a6835a98e4d7a0b04bc6aafd51b7d07508e57dccd19e698472968ca6e0a'
TARGET_PAIR_CHECKPOINT = TARGET_PAIR_REMOTE_ROOT + '/outputs/joint_target_pair_smolvla/final'
TARGET_PAIR_INFERENCE_SCRIPT = (
    TARGET_PAIR_REMOTE_ROOT + '/artifacts/joint_target_shadow_service/joint_inference_stdio.py'
)
TARGET_PAIR_PYTHONPATH = (
    TARGET_PAIR_REMOTE_ROOT + '/src:/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/deps'
)
TARGET_PAIR_LIVE_ROOT = TARGET_PAIR_REMOTE_ROOT + '/artifacts/joint_target_live_service'

SEGMENTATION_CHECKPOINT = (
    '/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/'
    'outputs/short_strip_mask2former_baseline/best'
)
TARGET_STACK_REMOTE_ROOT = '/home/rentao/program/lerobot-aubo-joint-stack'
TARGET_STACK_CHECKPOINT = TARGET_STACK_REMOTE_ROOT + '/outputs/joint_stack_target_smolvla/final'
TARGET_STACK_MODEL_SHA = 'c45fdeb1d36f71749008357956b6f3ca90a042e7c8e049a99d307204363fb0e9'
TARGET_STACK_SERVICE_ROOT = TARGET_STACK_REMOTE_ROOT + '/artifacts/joint_stack_live_service'
TARGET_STACK_PYTHONPATH = (
    TARGET_STACK_REMOTE_ROOT + '/src:/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/deps'
)

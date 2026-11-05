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
TARGET_STACK_SEGMENTATION_CHECKPOINT = (
    '/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/'
    'outputs/short_strip_mask2former_failure_finetuning/best'
)
TARGET_STACK_REMOTE_ROOT = '/home/rentao/program/lerobot-aubo-joint-stack'
TARGET_STACK_CHECKPOINT = TARGET_STACK_REMOTE_ROOT + '/outputs/joint_stack_target_smolvla/final'
TARGET_STACK_MODEL_SHA = 'c45fdeb1d36f71749008357956b6f3ca90a042e7c8e049a99d307204363fb0e9'
TARGET_STACK_THREE_FOUR_CHECKPOINT = TARGET_STACK_REMOTE_ROOT + '/outputs/joint_stack_three_four_target_smolvla/final'
TARGET_STACK_THREE_FOUR_MODEL_SHA = '80bde54859448a867da08bc377c0b5a83d421f60e2c2808b9afcd2bc771489a0'
TARGET_STACK_LOCAL_ALIGNMENT_CHECKPOINT = (
    TARGET_STACK_REMOTE_ROOT + '/outputs/joint_stack_local_alignment_retrained/checkpoints/best_step_060000'
)
TARGET_STACK_LOCAL_ALIGNMENT_MODEL_SHA = 'bd95ca61457094cbf72e009947aa9dd41b7312949b13fc96b64126fc7ac8d22e'
TARGET_STACK_LOCAL_ALIGNMENT_SERVICE_ROOT = TARGET_STACK_REMOTE_ROOT + '/artifacts/joint_stack_local_alignment_live_service'
TARGET_STACK_LOCAL_ALIGNMENT_PYTHONPATH = (
    TARGET_STACK_REMOTE_ROOT + '/artifacts/training/stack_local_alignment/source_snapshot/src'
    ':/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/deps'
)
TARGET_STACK_PI05_REMOTE_ROOT = '/home/rentao/program/lerobot-aubo-pi05'
TARGET_STACK_PI05_CHECKPOINT = TARGET_STACK_PI05_REMOTE_ROOT + '/outputs/pi05_joint_three_four/checkpoints/best_step_029000'
TARGET_STACK_PI05_MODEL_SHA = '03468b3f0e18229fd8b1d66cf6e815c414575e04f371c8c6969d9976d06127d6'
TARGET_STACK_PI05_REMOTE_PYTHON = TARGET_STACK_PI05_REMOTE_ROOT + '/.venv/bin/python'
TARGET_STACK_PI05_SERVICE_ROOT = TARGET_STACK_PI05_REMOTE_ROOT + '/artifacts/pi05_joint_live_service'
TARGET_STACK_PI05_PYTHONPATH = TARGET_STACK_PI05_SERVICE_ROOT + '/src'
TARGET_STACK_PI05_FP32_CHECKPOINT = TARGET_STACK_PI05_REMOTE_ROOT + '/artifacts/pi05_precision_repair/run/candidate'
TARGET_STACK_PI05_FP32_MODEL_SHA = '1e8fd4a3d4f86f2439ab8ead9da7fdf329aa0096e14b063893b9c7006bb8a830'
TARGET_STACK_PI05_FP32_SERVICE_ROOT = TARGET_STACK_PI05_REMOTE_ROOT + '/artifacts/pi05_precision_repair'
TARGET_STACK_PI05_FP32_INFERENCE_SCRIPT = (
    TARGET_STACK_PI05_FP32_SERVICE_ROOT + '/examples/phone_to_auboi10/joint_inference_stdio.py'
)
TARGET_STACK_PI05_FP32_PYTHONPATH = TARGET_STACK_PI05_FP32_SERVICE_ROOT + '/src'
TARGET_STACK_PI05_FP32_21000_CHECKPOINT = (
    TARGET_STACK_PI05_REMOTE_ROOT + '/outputs/pi05_joint_three_four_fp32_expert/checkpoints/best_step_021000'
)
TARGET_STACK_PI05_FP32_21000_MODEL_SHA = '7ada4146cd76303302c3b74134575ab43e082076bf81c37a0abad181a4c6dc4a'
TARGET_STACK_PI05_FP32_21000_SERVICE_ROOT = TARGET_STACK_PI05_REMOTE_ROOT + '/artifacts/pi05_fp32_21000_live_service'
TARGET_STACK_PI05_FP32_21000_INFERENCE_SCRIPT = TARGET_STACK_PI05_FP32_21000_SERVICE_ROOT + '/joint_inference_stdio.py'
TARGET_STACK_PI05_FP32_21000_PYTHONPATH = TARGET_STACK_PI05_REMOTE_ROOT + '/artifacts/pi05_fp32_retraining/src'
TARGET_STACK_SERVICE_ROOT = TARGET_STACK_REMOTE_ROOT + '/artifacts/joint_stack_live_service'
TARGET_STACK_PYTHONPATH = (
    TARGET_STACK_REMOTE_ROOT + '/src:/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/deps'
)
TARGET_STACK_THIN_CHECKPOINT = TARGET_STACK_REMOTE_ROOT + '/outputs/joint_stack_target_thin_outline_smolvla/final'
TARGET_STACK_THIN_MODEL_SHA = '992f69d28e043bc4c758e44d655a0605c113184441eeddc9983d347d5c3e378a'
TARGET_STACK_THIN_SERVICE_ROOT = TARGET_STACK_REMOTE_ROOT + '/artifacts/joint_stack_thin_outline_live_service'
TARGET_STACK_THIN_PYTHONPATH = (
    TARGET_STACK_THIN_SERVICE_ROOT + '/src:/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/deps'
)

# Same training run and joint contract as target-stack, but raw dual RGB only.
# The shared stack worker already supports the plain contract and full action
# chunk, so no separate remote service or segmentation dependency is needed.
BASELINE_STACK_CHECKPOINT = TARGET_STACK_REMOTE_ROOT + '/outputs/joint_stack_baseline_smolvla/final'
BASELINE_STACK_MODEL_SHA = '9c4a474c0702bddbe8c44c68b4306bbc3febad967474a6473275af5778151b14'
BASELINE_STACK_SERVICE_ROOT = TARGET_STACK_REMOTE_ROOT + '/artifacts/joint_stack_live_service'
BASELINE_STACK_INFERENCE_SCRIPT = BASELINE_STACK_SERVICE_ROOT + '/joint_inference_stdio.py'
BASELINE_STACK_PYTHONPATH = TARGET_STACK_REMOTE_ROOT + '/src'

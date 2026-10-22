"""Compatibility import for reviewed-seed live setup.

The implementation lives in the task package so entry scripts and runtime
modules no longer depend on the shadow command. Existing callers may keep
importing this module; new code should import runtime.live_setup directly.
"""

from lerobot.bamboo_sorting.model_profiles import (
    TARGET_PAIR_CHECKPOINT as CHECKPOINT,
    TARGET_PAIR_MODEL_SHA as MODEL_SHA,
    TARGET_PAIR_PYTHONPATH as PYTHONPATH,
    TARGET_PAIR_REMOTE_ROOT as REMOTE_ROOT,
    TARGET_STACK_CHECKPOINT as STACK_CHECKPOINT,
    TARGET_STACK_MODEL_SHA as STACK_MODEL_SHA,
    TARGET_STACK_REMOTE_ROOT as STACK_REMOTE_ROOT,
    TARGET_STACK_SERVICE_ROOT as STACK_SERVICE_ROOT,
)
from lerobot.bamboo_sorting.perception.target_seed import reviewed_masks
from lerobot.bamboo_sorting.runtime.live_setup import (
    LIVE_ROOT, SEGMENTATION_CHECKPOINT, prepare_live_setup,
)

__all__ = [
    'CHECKPOINT', 'LIVE_ROOT', 'MODEL_SHA', 'PYTHONPATH', 'REMOTE_ROOT',
    'SEGMENTATION_CHECKPOINT', 'STACK_CHECKPOINT', 'STACK_MODEL_SHA',
    'STACK_REMOTE_ROOT', 'STACK_SERVICE_ROOT', 'prepare_live_setup', 'reviewed_masks',
]

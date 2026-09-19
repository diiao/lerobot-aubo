from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from lerobot.bamboo_sorting.act_input_contract import (
    DROP_GRIPPER_STATE_VARIANT,
    FULL_STATE_VARIANT,
    FULL_ZERO_GRIPPER_STATE_VARIANT,
)
from lerobot.utils.constants import ACTION, OBS_STATE


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "phone_to_auboi10"
    / "train.py"
)
SPEC = importlib.util.spec_from_file_location("aubo_act_train_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


def _batch() -> dict[str, torch.Tensor]:
    actions = torch.zeros((2, 5, 8), dtype=torch.float32)
    actions[0, :, 7] = torch.tensor([100.0, 100.0, 0.0, 100.0, 0.0])
    actions[1, :, 7] = torch.tensor([0.0, 100.0, 100.0, 0.0, 100.0])
    state = torch.zeros((2, 13), dtype=torch.float32)
    state[1, 12] = 100.0
    return {
        OBS_STATE: state,
        ACTION: actions,
        "action_is_pad": torch.tensor(
            [[False, False, False, False, True], [False, False, False, False, False]]
        ),
    }


def test_build_weights_marks_on_and_off_transitions_only_in_gripper_dimension():
    weights = train.build_gripper_transition_loss_weights(
        _batch(),
        state_gripper_index=12,
        action_gripper_index=7,
        transition_weight=20.0,
    )

    assert weights is not None
    expected = torch.tensor(
        [[20.0, 1.0, 20.0, 20.0, 1.0], [20.0, 20.0, 1.0, 20.0, 20.0]]
    )
    torch.testing.assert_close(weights[..., 7], expected)
    torch.testing.assert_close(weights[..., :7], torch.ones_like(weights[..., :7]))


def test_build_weights_is_disabled_at_one():
    assert (
        train.build_gripper_transition_loss_weights(
            _batch(),
            state_gripper_index=12,
            action_gripper_index=7,
            transition_weight=1.0,
        )
        is None
    )


@pytest.mark.parametrize(
    "state_variant",
    [DROP_GRIPPER_STATE_VARIANT, FULL_ZERO_GRIPPER_STATE_VARIANT],
)
def test_keyframe_weighting_rejects_a_second_state_intervention(
    state_variant: str,
) -> None:
    with pytest.raises(ValueError, match="requires STATE_INPUT_VARIANT=full"):
        train.validate_gripper_transition_experiment(state_variant, 20.0)


def test_keyframe_weighting_accepts_full_13d_and_disabled_control() -> None:
    train.validate_gripper_transition_experiment(FULL_STATE_VARIANT, 20.0)
    train.validate_gripper_transition_experiment(FULL_ZERO_GRIPPER_STATE_VARIANT, 1.0)

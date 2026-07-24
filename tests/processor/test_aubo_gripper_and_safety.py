import numpy as np
import pytest

from lerobot.processor.converters import create_transition
from lerobot.robots.aubo_i10.robot_processor import (
    AuboEEBoundsAndSafety,
    AuboGripperVelocityToPosition,
)


def _run(step, action: dict, observation: dict) -> dict:
    step._current_transition = create_transition(observation=observation)
    return step.action(dict(action))


def test_latched_gripper_emits_persistent_suction_target():
    step = AuboGripperVelocityToPosition(latch=True)
    observation = {"gripper_pos": 0.0}

    assert _run(step, {"ee.gripper_vel": 0.0}, observation)["ee.gripper_pos"] == 0.0
    assert _run(step, {"ee.gripper_vel": 1.0}, observation)["ee.gripper_pos"] == 100.0
    assert _run(step, {"ee.gripper_vel": 0.0}, observation)["ee.gripper_pos"] == 100.0
    assert _run(step, {"ee.gripper_vel": -1.0}, observation)["ee.gripper_pos"] == 0.0
    assert _run(step, {"ee.gripper_vel": 0.0}, observation)["ee.gripper_pos"] == 0.0


def test_latched_gripper_reset_uses_observed_state():
    step = AuboGripperVelocityToPosition(latch=True)
    _run(step, {"ee.gripper_vel": 1.0}, {"gripper_pos": 0.0})
    step.reset()

    result = _run(step, {"ee.gripper_vel": 0.0}, {"gripper_pos": 0.0})
    assert result["ee.gripper_pos"] == 0.0


def test_legacy_gripper_mode_keeps_neutral_command():
    step = AuboGripperVelocityToPosition(latch=False)
    result = _run(step, {"ee.gripper_vel": 0.0}, {"gripper_pos": 0.0})
    assert result["ee.gripper_pos"] == 50.0


def test_safety_limiter_anchors_first_action_to_measured_pose():
    step = AuboEEBoundsAndSafety(max_ee_step_m=0.05)
    observation = {"ee.x": 0.0, "ee.y": -0.5, "ee.z": 0.2}
    action = {
        "ee.x": 0.5,
        "ee.y": -0.5,
        "ee.z": 0.2,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
    }

    result = _run(step, action, observation)
    displacement = np.linalg.norm(
        np.array([result["ee.x"], result["ee.y"], result["ee.z"]])
        - np.array([observation["ee.x"], observation["ee.y"], observation["ee.z"]])
    )
    assert displacement == pytest.approx(0.05)

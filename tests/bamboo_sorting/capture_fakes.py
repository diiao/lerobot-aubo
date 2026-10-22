"""Shared capture fakes and data builders for AUBO tests."""

import importlib.util
import math
from unittest.mock import Mock

import numpy as np

from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_NAMES
from lerobot.bamboo_sorting.capture.aubo_joint_capture import JointCaptureSession
from lerobot.robots.aubo_i10.joint_capture import AuboI10JointCaptureRobot
from tests.mocks.record_fakes import no_transition_trace
from tests.bamboo_sorting.paths import EXAMPLES


def entry():
    path = EXAMPLES / "record_joint.py"
    spec = importlib.util.spec_from_file_location("record_joint_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def robot():
    robot = object.__new__(AuboI10JointCaptureRobot)
    robot.robot_rpc_client = Mock()
    robot.robot_rpc_client.hasConnected.return_value = True
    robot.disable_servo_mode = Mock(return_value=True)
    robot.robot_interface = Mock()
    current = [0, 0, 0, 0, math.pi / 2, 0]
    robot.robot_interface.getRobotState().getJointPositions.return_value = current
    robot.robot_interface.getRobotAlgorithm().inverseKinematics.return_value = ([0.01, 0.02, 0, 0, math.radians(91), 0], 0)
    motion = robot.robot_interface.getMotionControl()
    motion.setSpeedFraction.return_value = 0
    motion.isServoModeEnabled.return_value = True
    motion.servoJoint.return_value = 0
    robot.is_servo_mode_enabled = True
    robot.joint_lower_deg = (-360,) * 6
    robot.joint_upper_deg = (360,) * 6
    robot.max_joint_tracking_deg = 2
    robot._joint_sequence = 0
    robot._previous_joint_target = None
    robot.last_joint_command_trace = None
    robot.fixed_axis5_deg = 90
    robot.servo_max_queue_retry = 1
    robot.servo_time, robot.servo_blend_radius = 0.04, 0
    robot.joint_acceleration, robot.joint_velocity = 1, 1
    robot._control_softpaws_based_on_gripper = lambda value: setattr(robot, "last_gripper_command_trace", no_transition_trace(False))
    return robot


def observation(image_shapes=None):
    shapes = image_shapes or {"global_rgb": (480, 640, 3), "grasp_rgb": (480, 640, 3)}
    return {**dict.fromkeys(JOINT_NAMES, 0.0), "J5": 90.0, "gripper_pos": 0.0,
            **dict.fromkeys(("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"), 0.0),
            **{name: np.zeros(shape, np.uint8) for name, shape in shapes.items()}}


def phone_target():
    return {**dict.fromkeys(("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"), 0.0),
            "ee_mode": "abs_j6yaw", "ee.j6_target": 0.015, "ee.gripper_pos": 0.0}


def session(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    return JointCaptureSession(dataset_root=tmp_path / "data", evidence_root=evidence, episode_count=60,
        prepare_episode=lambda i: {"scene_id": f"scene-{i}", "placement_reference": "line B", "split": "train"},
        outcome_provider=lambda i: "single_success")


def stats():
    # Deliberately unbalanced suction: normalized zero means 80, not off.
    return {key: {"mean": [0, 0, 0, 0, 0, 0, 80], "std": [1, 1, 1, 1, 1, 1, 40],
                  "min": [-1, -1, -1, -1, -1, -1, 0], "max": [1, 1, 1, 1, 1, 1, 100]}
            for key in ("observation.state", "action")}

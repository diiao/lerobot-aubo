"""Supervised phone capture with joint labels, separate from legacy C0.

The phone still uses Cartesian teleoperation. IK happens BEFORE labels are
formed. This driver records all six legacy IK joint targets plus suction.
J5 follows IK exactly as in the original phone entry; it is not fixed.
No SO101 scale, sign or offset mapping is applied to these AUBO joint values.
"""

import math
import time

from lerobot.bamboo_sorting.aubo_joint_contract import (
    JOINT_FIELDS, JOINT_NAMES, finite_vector, joint_command,
    expand_joint_command, joint_observation_features,
)
from .aubo_i10 import AuboI10Robot


class AuboI10JointCaptureRobot(AuboI10Robot):
    """Capture-only driver; this class is not a policy execution entry point."""

    def __init__(self, config):
        super().__init__(config)
        self.joint_lower_deg = None
        self.joint_upper_deg = None
        self.last_joint_command_trace = None
        self._joint_sequence = 0

    @property
    def observation_features(self):
        return joint_observation_features({name: (camera.height, camera.width, 3)
                                           for name, camera in self.cameras.items()})

    @property
    def action_features(self):
        return dict.fromkeys(JOINT_FIELDS, float)

    def connect(self, calibrate=True):
        super().connect(calibrate=calibrate)
        try:
            config = self.robot_interface.getRobotConfig()
            self.joint_lower_deg = tuple(math.degrees(v) for v in finite_vector(
                config.getJointMinPositions(), 6, "SDK joint lower limits"))
            self.joint_upper_deg = tuple(math.degrees(v) for v in finite_vector(
                config.getJointMaxPositions(), 6, "SDK joint upper limits"))
            if any(lo >= hi for lo, hi in zip(self.joint_lower_deg, self.joint_upper_deg, strict=True)):
                raise ValueError("invalid controller joint limits")
        except BaseException:
            self.disconnect()
            raise

    def reset_joint_capture(self):
        self.last_joint_command_trace = None

    def resolve_phone_joint_target(self, action, observation):
        """Resolve the existing abs_j6yaw phone mode exactly once, in radians.

        Read the current SDK joints as the IK seed, as the legacy sender does.
        Return degrees for the dataset/driver boundary; no motion or IO here.
        """
        pose_keys = ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz")
        if action.get("ee_mode") != "abs_j6yaw":
            raise ValueError("joint capture requires the established abs_j6yaw phone mode")
        pose = finite_vector([action[k] for k in pose_keys], 6, "phone pose")
        j6, suction = finite_vector([action["ee.j6_target"], action["ee.gripper_pos"]], 2, "phone controls")
        if suction not in (0.0, 100.0):
            raise ValueError("phone suction must be 0/100")
        if not self.is_connected or self.robot_interface is None:
            raise ConnectionError("joint capture robot is not connected")
        current_rad = finite_vector(self.robot_interface.getRobotState().getJointPositions(), 6, "SDK joints")
        result = self.robot_interface.getRobotAlgorithm().inverseKinematics(list(current_rad), list(pose))
        if len(result) != 2 or isinstance(result[1], bool) or result[1] != 0:
            raise ValueError("phone joint inverse kinematics failed")
        joints = list(finite_vector(result[0], 6, "IK joint target"))
        joints[5] = j6  # Preserve the existing phone J6-yaw control convention.
        target = [math.degrees(v) for v in joints]
        full = dict(zip(JOINT_NAMES, target, strict=True))
        return joint_command({k: suction if k == "gripper_pos" else full[k] for k in JOINT_FIELDS})

    def send_action(self, action):
        self.last_joint_command_trace = None
        self.last_gripper_command_trace = None
        command = joint_command(action)
        full = expand_joint_command(command)
        if not self.is_connected or self.robot_interface is None:
            raise ConnectionError("joint capture robot is not connected")
        super().send_action(full)
        # This is set only after both joint servo and suction handling succeeded.
        if self.last_joint_command_trace is None:
            raise RuntimeError("accepted joint servo command evidence is missing")
        return command

    def _send_joint_action_servo(self, action, motion):
        self._send_phone_joint_target([math.radians(action[k]) for k in JOINT_NAMES], motion)
        self._joint_sequence += 1
        self.last_joint_command_trace = {
            "sequence": self._joint_sequence,
            "joint_target_deg": [action[k] for k in JOINT_NAMES],
            "sdk_target_rad": [math.radians(action[k]) for k in JOINT_NAMES],
            "return_code": 0,
            "accepted_monotonic_s": time.perf_counter(),
        }


class PhoneToJointAction:
    """Wrap the existing phone processor; output exactly the recorded action."""

    def __init__(self, phone_processor, robot):
        self.phone_processor = phone_processor
        self.robot = robot

    def __call__(self, pair):
        _, observation = pair
        return self.robot.resolve_phone_joint_target(self.phone_processor(pair), observation)

    def reset(self):
        self.phone_processor.reset()
        self.robot.reset_joint_capture()

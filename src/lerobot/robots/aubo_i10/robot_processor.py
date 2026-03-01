#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    ObservationProcessorStep,
    TransitionKey,
)


@ProcessorStepRegistry.register("phone_ee_to_aubo_ee")
@dataclass
class PhoneEEToAuboEE(RobotActionProcessorStep):
    """
    Converts phone delta commands to absolute end-effector pose for AuboI10 robot.
    
    This processor takes the relative delta commands from the phone teleoperator
    and converts them to absolute end-effector poses by adding the delta to the
    current robot's TCP pose obtained from the observation.
    
    Unlike SO100 which uses URDF-based forward kinematics, this processor directly
    uses the robot's getTcpPose() output available in the observation.
    
    Attributes:
        end_effector_step_sizes: Scaling factors for delta commands (meters per unit).
        use_latched_reference: If True, latch reference on enable; if False, always use current pose.
        reference_ee_pose: Internal state storing the latched reference pose.
        _prev_enabled: Internal state to detect the rising edge of the enable signal.
        _command_when_disabled: Internal state to hold the last command while disabled.
    """
    
    end_effector_step_sizes: dict[str, float] = field(default_factory=lambda: {"x": 0.3, "y": 0.3, "z": 0.3})
    use_latched_reference: bool = True
    
    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        
        if observation is None:
            raise ValueError("Observation is required for computing absolute EE pose")
        
        current_ee_pose = np.array([
            float(observation.get("ee.x", 0.0)),
            float(observation.get("ee.y", 0.0)),
            float(observation.get("ee.z", 0.0)),
            float(observation.get("ee.wx", 0.0)),
            float(observation.get("ee.wy", 0.0)),
            float(observation.get("ee.wz", 0.0)),
        ])
        
        enabled = bool(action.pop("enabled"))
        tx = float(action.pop("target_x"))
        ty = float(action.pop("target_y"))
        tz = float(action.pop("target_z"))
        twx = float(action.pop("target_wx"))
        twy = float(action.pop("target_wy"))
        twz = float(action.pop("target_wz"))
        gripper_vel = float(action.pop("gripper_vel"))
        
        desired = None
        
        if enabled:
            ref = current_ee_pose
            if self.use_latched_reference:
                if not self._prev_enabled or self.reference_ee_pose is None:
                    self.reference_ee_pose = current_ee_pose.copy()
                ref = self.reference_ee_pose if self.reference_ee_pose is not None else current_ee_pose
            
            delta_p = np.array([
                tx * self.end_effector_step_sizes["x"],
                ty * self.end_effector_step_sizes["y"],
                tz * self.end_effector_step_sizes["z"],
            ], dtype=float)
            
            desired = ref.copy()
            desired[:3] = ref[:3] + delta_p
            desired[3] = ref[3] + twx * 0.1
            desired[4] = ref[4] + twy * 0.1
            desired[5] = ref[5] + twz * 0.1
            
            self._command_when_disabled = desired.copy()
        else:
            if self._command_when_disabled is None:
                self._command_when_disabled = current_ee_pose.copy()
            desired = self._command_when_disabled.copy()
        
        action["ee.x"] = float(desired[0])
        action["ee.y"] = float(desired[1])
        action["ee.z"] = float(desired[2])
        action["ee.wx"] = float(desired[3])
        action["ee.wy"] = float(desired[4])
        action["ee.wz"] = float(desired[5])
        action["ee.gripper_vel"] = gripper_vel
        
        self._prev_enabled = enabled
        return action

    def reset(self):
        self._prev_enabled = False
        self.reference_ee_pose = None
        self._command_when_disabled = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION].pop(feat, None)

        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_vel"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


@ProcessorStepRegistry.register("aubo_gripper_velocity_to_position")
@dataclass
class AuboGripperVelocityToPosition(RobotActionProcessorStep):
    """
    Converts gripper velocity command to gripper position for AuboI10.
    
    Integrates the velocity command over time to produce a position command,
    using the current gripper position from observation as starting point.
    
    Attributes:
        speed_factor: Scaling factor for velocity to position conversion.
        clip_min: Minimum gripper position (closed).
        clip_max: Maximum gripper position (open).
    """
    
    speed_factor: float = 20.0
    clip_min: float = 0.0
    clip_max: float = 100.0

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        
        if observation is None:
            raise ValueError("Observation is required for gripper position computation")
        
        gripper_vel = action.pop("ee.gripper_vel")
        current_gripper_pos = float(observation.get("gripper_pos", 50.0))
        
        delta = gripper_vel * self.speed_factor
        gripper_pos = float(np.clip(current_gripper_pos + delta, self.clip_min, self.clip_max))
        
        action["ee.gripper_pos"] = gripper_pos
        
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("ee.gripper_vel", None)
        features[PipelineFeatureType.ACTION]["ee.gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )
        return features


@ProcessorStepRegistry.register("aubo_ee_bounds_and_safety")
@dataclass
class AuboEEBoundsAndSafety(RobotActionProcessorStep):
    """
    Clips the end-effector pose to predefined bounds and checks for unsafe jumps.
    
    Ensures that the target end-effector pose remains within a safe operational workspace.
    
    Attributes:
        end_effector_bounds: Dictionary with "min" and "max" keys for position clipping.
        max_ee_step_m: Maximum allowed change in position (meters) between steps.
        _last_pos: Internal state storing the last commanded position.
    """
    
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {"min": [-0.8, -1.2, 0.0], "max": [0.8, 0.0, 0.8]}
    )
    max_ee_step_m: float = 0.05
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        x = action["ee.x"]
        y = action["ee.y"]
        z = action["ee.z"]
        wx = action["ee.wx"]
        wy = action["ee.wy"]
        wz = action["ee.wz"]

        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present"
            )

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)

        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)

        self._last_pos = pos

        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(twist[0])
        action["ee.wy"] = float(twist[1])
        action["ee.wz"] = float(twist[2])
        return action

    def reset(self):
        self._last_pos = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("aubo_ee_to_ee_delta")
@dataclass
class AuboEEToEEDelta(RobotActionProcessorStep):
    """
    Converts absolute end-effector pose to delta commands for robot control.
    
    This is used during replay/evaluation to convert the absolute pose from the policy
    to the delta format expected by the robot's send_action method.
    
    The robot's _send_ee_action_servo expects delta values that it adds to current pose.
    """
    
    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        
        if observation is None:
            raise ValueError("Observation is required for computing delta")
        
        current_ee_pose = np.array([
            float(observation.get("ee.x", 0.0)),
            float(observation.get("ee.y", 0.0)),
            float(observation.get("ee.z", 0.0)),
            float(observation.get("ee.wx", 0.0)),
            float(observation.get("ee.wy", 0.0)),
            float(observation.get("ee.wz", 0.0)),
        ])
        
        target_x = float(action.pop("ee.x"))
        target_y = float(action.pop("ee.y"))
        target_z = float(action.pop("ee.z"))
        target_wx = float(action.pop("ee.wx"))
        target_wy = float(action.pop("ee.wy"))
        target_wz = float(action.pop("ee.wz"))
        gripper_pos = action.pop("ee.gripper_pos", 50.0)
        
        action["ee.x"] = target_x - current_ee_pose[0]
        action["ee.y"] = target_y - current_ee_pose[1]
        action["ee.z"] = target_z - current_ee_pose[2]
        action["ee.wx"] = target_wx - current_ee_pose[3]
        action["ee.wy"] = target_wy - current_ee_pose[4]
        action["ee.wz"] = target_wz - current_ee_pose[5]
        action["ee.gripper_pos"] = gripper_pos
        
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

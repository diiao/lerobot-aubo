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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import RobotAction

from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep


@dataclass
@ProcessorStepRegistry.register(name="so101_to_aubo_processor")
class SO101ToAuboProcessorStep(RobotActionProcessorStep):
    """
    Processor step that transforms SO-101 leader arm actions to Aubo i10 robot actions.
    
    This processor handles:
    - Joint mapping from SO-101 (5 joints + gripper) to Aubo i10 (6 joints + gripper)
    - Direction inversion for certain joints
    - Offset calibration for joint positions
    - Fixed J5 axis configuration
    
    Mapping:
        - shoulder_pan.pos  → J1
        - shoulder_lift.pos → J2
        - elbow_flex.pos    → J3
        - wrist_flex.pos    → J4
        - J5 (fixed value)
        - wrist_roll.pos    → J6
        - gripper.pos       → gripper_pos
    """

    directions: list[float] = field(default_factory=lambda: [-1.0, -1.0, 1.0, -1.0, 1.0])
    offsets: list[float] = field(default_factory=lambda: [0.0, 0.0, 100.0, 90.0, 40.0])
    scale: float = 1.0
    fixed_j5_deg: float = 90.0

    leader_joint_keys: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
        ]
    )

    def action(self, action: RobotAction) -> RobotAction:
        """
        Transform SO-101 leader action to Aubo i10 action.
        
        Args:
            action: Dictionary containing SO-101 joint positions with keys like
                    'shoulder_pan.pos', 'shoulder_lift.pos', etc.
        
        Returns:
            Dictionary containing Aubo i10 joint positions with keys 'J1'-'J6' and 'gripper_pos'.
        """
        try:
            leader_joints_deg = [action[key] for key in self.leader_joint_keys]
        except KeyError as e:
            raise KeyError(
                f"Missing joint key in action: {e}. "
                f"Expected keys: {self.leader_joint_keys}, "
                f"Got keys: {list(action.keys())}"
            ) from e

        corrected = []
        for i in range(5):
            deg = leader_joints_deg[i]
            deg = deg * self.directions[i] + self.offsets[i]
            deg *= self.scale
            corrected.append(deg)

        aubo_action = {
            "J1": corrected[0],
            "J2": corrected[1],
            "J3": corrected[2],
            "J4": corrected[3],
            "J5": self.fixed_j5_deg,
            "J6": corrected[4],
        }

        gripper_pos = action.get("gripper.pos", 0)
        aubo_action["gripper_pos"] = gripper_pos

        return aubo_action

    def get_config(self) -> dict[str, Any]:
        return {
            "directions": self.directions,
            "offsets": self.offsets,
            "scale": self.scale,
            "fixed_j5_deg": self.fixed_j5_deg,
            "leader_joint_keys": self.leader_joint_keys,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transform feature descriptions from SO-101 format to Aubo i10 format.
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}
        
        for feature_type, feature_dict in features.items():
            if feature_type == PipelineFeatureType.ACTION:
                new_action_features = {}
                new_action_features["J1"] = PolicyFeature(dtype="float", shape=(1,))
                new_action_features["J2"] = PolicyFeature(dtype="float", shape=(1,))
                new_action_features["J3"] = PolicyFeature(dtype="float", shape=(1,))
                new_action_features["J4"] = PolicyFeature(dtype="float", shape=(1,))
                new_action_features["J5"] = PolicyFeature(dtype="float", shape=(1,))
                new_action_features["J6"] = PolicyFeature(dtype="float", shape=(1,))
                new_action_features["gripper_pos"] = PolicyFeature(dtype="float", shape=(1,))
                new_features[feature_type] = new_action_features
            else:
                new_features[feature_type] = feature_dict.copy()
        
        return new_features


@dataclass
@ProcessorStepRegistry.register(name="so100_to_aubo_processor")
class SO100ToAuboProcessorStep(SO101ToAuboProcessorStep):
    """
    Processor step for SO-100 leader to Aubo i10.
    Inherits from SO101ToAuboProcessorStep with same mapping logic.
    """
    pass

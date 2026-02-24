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

from typing import Any

from .converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from .core import RobotAction, RobotObservation
from .pipeline import IdentityProcessorStep, RobotProcessorPipeline
from .so_to_aubo_processor import SO101ToAuboProcessorStep, SO100ToAuboProcessorStep


def make_default_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor


def make_default_robot_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor


def make_default_robot_observation_processor() -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_default_processors():
    teleop_action_processor = make_default_teleop_action_processor()
    robot_action_processor = make_default_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor()
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)


def make_so101_to_aubo_processor(
    directions: list[float] | None = None,
    offsets: list[float] | None = None,
    scale: float = 1.0,
    fixed_j5_deg: float = 90.0,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """
    Create a processor pipeline that transforms SO-101 leader actions to Aubo i10 actions.
    
    Args:
        directions: Direction multipliers for each joint (default: [-1, -1, 1, -1, 1])
        offsets: Offset values in degrees for each joint (default: [0, 0, 100, 90, 40])
        scale: Global scale factor (default: 1.0)
        fixed_j5_deg: Fixed value for J5 axis in degrees (default: 90.0)
    
    Returns:
        A RobotProcessorPipeline that transforms SO-101 actions to Aubo i10 format.
    """
    step = SO101ToAuboProcessorStep(
        directions=directions if directions is not None else [-1.0, -1.0, 1.0, -1.0, 1.0],
        offsets=offsets if offsets is not None else [0.0, 0.0, 100.0, 90.0, 40.0],
        scale=scale,
        fixed_j5_deg=fixed_j5_deg,
    )
    
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[step],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
        name="so101_to_aubo_pipeline",
    )


def make_so100_to_aubo_processor(
    directions: list[float] | None = None,
    offsets: list[float] | None = None,
    scale: float = 1.0,
    fixed_j5_deg: float = 90.0,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """
    Create a processor pipeline that transforms SO-100 leader actions to Aubo i10 actions.
    
    Args:
        directions: Direction multipliers for each joint (default: [-1, -1, 1, -1, 1])
        offsets: Offset values in degrees for each joint (default: [0, 0, 100, 90, 40])
        scale: Global scale factor (default: 1.0)
        fixed_j5_deg: Fixed value for J5 axis in degrees (default: 90.0)
    
    Returns:
        A RobotProcessorPipeline that transforms SO-100 actions to Aubo i10 format.
    """
    step = SO100ToAuboProcessorStep(
        directions=directions if directions is not None else [-1.0, -1.0, 1.0, -1.0, 1.0],
        offsets=offsets if offsets is not None else [0.0, 0.0, 100.0, 90.0, 40.0],
        scale=scale,
        fixed_j5_deg=fixed_j5_deg,
    )
    
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[step],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
        name="so100_to_aubo_pipeline",
    )


def make_processors_for_teleop_robot_pair(
    teleop_type: str,
    robot_type: str,
    teleop_to_robot_config: dict[str, Any] | None = None,
) -> tuple[
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    RobotProcessorPipeline[RobotObservation, RobotObservation],
]:
    """
    Create appropriate processors based on teleoperator and robot type pair.
    
    This factory function automatically selects the correct processor pipeline
    based on the combination of teleoperator and robot types.
    
    Args:
        teleop_type: Type of teleoperator (e.g., "so101_leader", "so100_leader")
        robot_type: Type of robot (e.g., "aubo_i10", "so100_follower")
        teleop_to_robot_config: Optional configuration for the transformation processor
    
    Returns:
        A tuple of (teleop_action_processor, robot_action_processor, robot_observation_processor)
    """
    config = teleop_to_robot_config or {}
    
    if robot_type == "aubo_i10" and teleop_type in ["so101_leader", "so100_leader"]:
        if teleop_type == "so101_leader":
            teleop_action_processor = make_so101_to_aubo_processor(
                directions=config.get("directions"),
                offsets=config.get("offsets"),
                scale=config.get("scale", 1.0),
                fixed_j5_deg=config.get("fixed_j5_deg", 90.0),
            )
        else:
            teleop_action_processor = make_so100_to_aubo_processor(
                directions=config.get("directions"),
                offsets=config.get("offsets"),
                scale=config.get("scale", 1.0),
                fixed_j5_deg=config.get("fixed_j5_deg", 90.0),
            )
    else:
        teleop_action_processor = make_default_teleop_action_processor()
    
    robot_action_processor = make_default_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor()
    
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)

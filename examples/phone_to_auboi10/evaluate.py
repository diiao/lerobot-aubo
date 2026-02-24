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

import time

from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot, AuboI10Config
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

def main():
    # Initialize the robot and teleoperator
    robot_config = AuboI10Config()
    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)

    # Initialize the robot and teleoperator
    robot = AuboI10Robot(robot_config)
    teleop_device = Phone(teleop_config)

    # Build pipeline to convert phone action to robot joint action
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Init rerun viewer
    init_rerun(session_name="phone_auboi10_evaluate")

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting evaluation loop. Move your phone to teleoperate the robot...")
    print("Press Ctrl+C to stop evaluation.")

    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            robot_obs = robot.get_observation()

            # Get teleop action
            phone_obs = teleop_device.get_action()

            # Phone -> Robot action
            robot_action = phone_to_robot_joints_processor((phone_obs, robot_obs))

            # Convert phone action to Aubo joint commands
            joint_action = {
                "J1": robot_action.get("ee.x", 0.0) * 10.0,
                "J2": robot_action.get("ee.y", 0.0) * 10.0,
                "J3": robot_action.get("ee.z", 0.0) * 10.0,
                "J4": robot_action.get("ee.roll", 0.0) * 5.0,
                "J5": robot.fixed_axis5_deg,
                "J6": robot_action.get("ee.yaw", 0.0) * 5.0,
                "gripper_pos": robot_action.get("gripper", 0.0) * 100.0,
            }

            # Send action to robot
            _ = robot.send_action(joint_action)

            # Visualize
            log_rerun_data(observation=phone_obs, action=joint_action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("Stopping evaluation...")


if __name__ == "__main__":
    main()
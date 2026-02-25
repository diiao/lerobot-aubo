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
import numpy as np
import logging

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
from pathlib import Path
from datetime import datetime

FPS = 30
from lerobot.utils.utils import (
    # get_safe_torch_device,
    init_logging,
    # log_say,
)

def main():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir / f"record_{timestamp_str}.log"
    init_logging(log_file=log_file_path)
    # Initialize robot and teleoperator
    robot_config = AuboI10Config()
    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)  # 使用安卓手机

    # Initialize robot and teleoperator
    robot = AuboI10Robot(robot_config)
    teleop_device = Phone(teleop_config)

    # Build pipeline to convert phone action to robot action format
    phone_to_robot_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Init rerun viewer
    init_rerun(session_name="phone_auboi10_teleop")

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop in end-effector mode. Move your phone to teleoperate the robot...")
    print("The robot will use Aubo's moveLine interface for direct end-effector control.")
    
    # Safety: track last sent end-effector position for delta checking
    last_ee_pos = None
    max_ee_delta = 0.1  # Maximum allowed position change in meters
    
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        t_robot_obs_start = time.perf_counter()
        robot_obs = robot.get_observation()
        t_robot_obs_end = time.perf_counter()

        # Get teleop action
        t_phone_obs_start = time.perf_counter()
        phone_obs = teleop_device.get_action()
        t_phone_obs_end = time.perf_counter()

        # Phone -> EE pose (no IK conversion needed)
        t_process_start = time.perf_counter()
        robot_action = phone_to_robot_processor((phone_obs, robot_obs))
        t_process_end = time.perf_counter()

        # Convert target_* to ee.* format expected by AuboI10Robot
        t_convert_start = time.perf_counter()
        robot_action["ee.x"] = robot_action.pop("target_x", 0.0)
        robot_action["ee.y"] = robot_action.pop("target_y", 0.0)
        robot_action["ee.z"] = robot_action.pop("target_z", 0.0)
        robot_action["ee.wx"] = robot_action.pop("target_wx", 0.0)
        robot_action["ee.wy"] = robot_action.pop("target_wy", 0.0)
        robot_action["ee.wz"] = robot_action.pop("target_wz", 0.0)
        
        # Convert gripper_vel to gripper_pos
        gripper_vel = robot_action.pop("gripper_vel", 0.0)
        # Simple conversion: positive velocity opens gripper, negative closes it
        if gripper_vel > 0.5:
            robot_action["gripper_pos"] = 100.0  # Open
        elif gripper_vel < -0.5:
            robot_action["gripper_pos"] = 0.0    # Close
        else:
            robot_action["gripper_pos"] = 50.0    # Neutral

        # Convert all np.float64 values to Python float
        for key in robot_action:
            if isinstance(robot_action[key], (np.floating, np.integer)):
                robot_action[key] = float(robot_action[key])
        t_convert_end = time.perf_counter()

        logging.debug(f"Robot action: {robot_action}")
        
        # Safety check: verify position delta doesn't exceed threshold
        t_safety_start = time.perf_counter()
        # current_ee_pos = [robot_action["ee.x"], robot_action["ee.y"], robot_action["ee.z"]]
        # if last_ee_pos is not None:
        #     # Calculate Euclidean distance between current and last position
        #     delta = sum((c - l) ** 2 for c, l in zip(current_ee_pos, last_ee_pos)) ** 0.5
        #     if delta > max_ee_delta:
        #         logging.warning(f"Position delta {delta:.3f}m exceeds safety threshold {max_ee_delta}m, skipping this action")
        #         precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
        #         # Wait for user input before next iteration
        #         input("Press Enter to continue to next iteration...")
        #         continue
        t_safety_end = time.perf_counter()
        
        # Send action to robot
        # AuboI10Robot.send_action will detect ee.x/ee.y/ee.z/ee.wx/ee.wy/ee.wz
        # and use moveLine for direct end-effector control
        t_send_start = time.perf_counter()
        _ = robot.send_action(robot_action)
        t_send_end = time.perf_counter()
        


        # Visualize
        t_visualize_start = time.perf_counter()
        log_rerun_data(observation=phone_obs, action=robot_action)
        t_visualize_end = time.perf_counter()

        # Calculate and print timing information
        total_time = time.perf_counter() - t0
        robot_obs_time = t_robot_obs_end - t_robot_obs_start
        phone_obs_time = t_phone_obs_end - t_phone_obs_start
        process_time = t_process_end - t_process_start
        convert_time = t_convert_end - t_convert_start
        safety_time = t_safety_end - t_safety_start
        send_time = t_send_end - t_send_start
        visualize_time = t_visualize_end - t_visualize_start

        logging.debug(f"\nTiming breakdown (ms):")
        logging.debug(f"Total: {total_time*1000:.2f}")
        logging.debug(f"Robot observation: {robot_obs_time*1000:.2f}")
        logging.debug(f"Phone action: {phone_obs_time*1000:.2f}")
        logging.debug(f"Processing: {process_time*1000:.2f}")
        logging.debug(f"Conversion: {convert_time*1000:.2f}")
        logging.debug(f"Safety check: {safety_time*1000:.2f}")
        logging.debug(f"Send action: {send_time*1000:.2f}")
        logging.debug(f"Visualization: {visualize_time*1000:.2f}")

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
        
        # Wait for user input before next iteration
        # input("Press Enter to continue to next iteration...")


if __name__ == "__main__":
    main()
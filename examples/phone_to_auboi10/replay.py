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
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot, AuboI10Config
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

def main():
    # Initialize the robot
    robot_config = AuboI10Config()
    robot = AuboI10Robot(robot_config)

    # Connect to the robot
    robot.connect()

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    # Load dataset
    dataset = LeRobotDataset("./datasets/phone_auboi10")

    # Init rerun viewer
    init_rerun(session_name="phone_auboi10_replay")

    print("Starting replay loop...")
    print("Press Ctrl+C to stop replay.")

    try:
        for i, sample in enumerate(dataset):
            t0 = time.perf_counter()

            # Get action from dataset
            action = sample["action"]

            # Send action to robot
            _ = robot.send_action(action)

            # Visualize
            log_rerun_data(observation=sample["observation"], action=action)

            print(f"Replayed step {i}/{len(dataset)}")
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("Stopping replay...")


if __name__ == "__main__":
    main()
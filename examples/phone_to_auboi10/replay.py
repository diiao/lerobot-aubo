# !/usr/bin/env python

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
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot, AuboI10Config
from lerobot.robots.aubo_i10.robot_processor import AuboEEToEEDelta
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say, init_logging

EPISODE_IDX = 0
LOCAL_DATASET_PATH = "./datasets/phone_auboi10"


def main():
    # Initialize logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / "replay.log"
    init_logging(log_file=log_file_path)

    robot_config = AuboI10Config()

    robot = AuboI10Robot(robot_config)

    ee_to_delta_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            AuboEEToEEDelta(),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    dataset = LeRobotDataset(LOCAL_DATASET_PATH, episodes=[EPISODE_IDX])
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_IDX)
    actions = episode_frames.select_columns(ACTION)

    robot.connect()

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        print("Starting replay loop...")
        log_say(f"Replaying episode {EPISODE_IDX}")
        for idx in range(len(episode_frames)):
            t0 = time.perf_counter()

            ee_action = {
                name: float(actions[idx][ACTION][i])
                for i, name in enumerate(dataset.features[ACTION]["names"])
            }

            robot_obs = robot.get_observation()

            delta_action = ee_to_delta_processor((ee_action, robot_obs))

            _ = robot.send_action(delta_action)

            precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()

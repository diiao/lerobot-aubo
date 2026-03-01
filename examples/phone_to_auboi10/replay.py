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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot, AuboI10Config
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

EPISODE_IDX = 0
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"


def main():
    robot_config = AuboI10Config()

    robot = AuboI10Robot(robot_config)

    dataset = LeRobotDataset(HF_REPO_ID, episodes=[EPISODE_IDX])
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

            robot_action = {
                "ee.x": ee_action.get("ee.x", 0.0),
                "ee.y": ee_action.get("ee.y", 0.0),
                "ee.z": ee_action.get("ee.z", 0.0),
                "ee.wx": ee_action.get("ee.wx", 0.0),
                "ee.wy": ee_action.get("ee.wy", 0.0),
                "ee.wz": ee_action.get("ee.wz", 0.0),
                "gripper_pos": ee_action.get("ee.gripper_pos", ee_action.get("gripper_pos", 50.0)),
            }

            _ = robot.send_action(robot_action)

            precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()

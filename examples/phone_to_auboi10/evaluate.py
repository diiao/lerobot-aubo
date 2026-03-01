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

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_teleop_action_processor,
)
from lerobot.processor.converters import (
    observation_to_transition,
    transition_to_observation,
)
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot, AuboI10Config
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "My task description"
LOCAL_MODEL_PATH = "./models/phone_auboi10"
LOCAL_DATASET_PATH = "./datasets/phone_auboi10"


def main():
    robot_config = AuboI10Config()

    robot = AuboI10Robot(robot_config)

    policy = ACTPolicy.from_pretrained(LOCAL_MODEL_PATH)

    robot_joints_to_ee_pose_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    dataset = LeRobotDataset.create(
        repo_id=LOCAL_DATASET_PATH,
        fps=FPS,
        features=combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=robot_joints_to_ee_pose_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=make_default_teleop_action_processor(),
                initial_features=create_initial_features(
                    action={
                        f"ee.{k}": PolicyFeature(type=FeatureType.ACTION, shape=(1,))
                        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]
                    }
                ),
                use_videos=True,
            ),
        ),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=LOCAL_MODEL_PATH,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    robot.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="phone_auboi10_evaluate")

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        print("Starting evaluate loop...")
        episode_idx = 0
        for episode_idx in range(NUM_EPISODES):
            log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=make_default_teleop_action_processor(),
                robot_action_processor=None,
                robot_observation_processor=robot_joints_to_ee_pose_processor,
            )

            if not events["stop_recording"] and (
                (episode_idx < NUM_EPISODES - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    control_time_s=EPISODE_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                    teleop_action_processor=make_default_teleop_action_processor(),
                    robot_action_processor=None,
                    robot_observation_processor=robot_joints_to_ee_pose_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            episode_idx += 1
    finally:
        log_say("Stop recording")
        robot.disconnect()
        listener.stop()

        dataset.finalize()


if __name__ == "__main__":
    main()

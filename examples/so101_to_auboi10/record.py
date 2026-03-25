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

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot, AuboI10Config
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say, init_logging

NUM_EPISODES = 6
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "挑捡苹果并将其放入蓝色盒子中"
LOCAL_DATASET_PATH = "datasets/so101_auboi10"  # 请根据实际路径修改
RESUME = False  # 设为 True 则在已有数据集基础上继续录制，否则报错保护现有数据


def wait_for_key(events: dict, prompt: str = "按 → (右箭头键) 继续") -> bool:
    """
    等待用户按下右箭头键继续，或按 Esc 退出。
    返回 True 表示可以继续，False 表示用户按了 Esc 终止。
    """
    events["exit_early"] = False
    print("\n" + "=" * 50)
    print(prompt)
    print("按 Esc 终止整个录制")
    print("=" * 50)
    while not events["exit_early"] and not events["stop_recording"]:
        time.sleep(0.05)
    if events["stop_recording"]:
        return False
    events["exit_early"] = False
    return True


def main():
    # Initialize logging: 控制台 INFO，文件 DEBUG
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / "record.log"
    init_logging(log_file=log_file_path, console_level="INFO", file_level="DEBUG")

    # 相机配置
    camera_config = {
        "handeye": OpenCVCameraConfig(index_or_path=Path("/dev/video0"), width=640, height=480, fps=FPS),
        "fixed": OpenCVCameraConfig(index_or_path=Path("/dev/video2"), width=640, height=480, fps=FPS),
    }

    # 机器人配置
    robot_config = AuboI10Config(cameras=camera_config)

    # SO101遥操作器配置
    teleop_config = SOLeaderTeleopConfig(
        port="/dev/ttyACM0",  # 请根据实际端口修改
        use_degrees=True,
    )

    robot = AuboI10Robot(robot_config)
    so101 = SO101Leader(teleop_config)

    # SO101遥操作动作处理器 - 仅传递关节角度，不做任何转换
    # SO101输出: shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos
    # Aubo期望: shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos
    teleop_action_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[],  # 直接使用SO101的关节角度，无需转换
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # 机器人动作处理器 - 直接使用关节角度
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[],  # AuboI10Robot.send_action 会处理关节角度转换
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    

    # 机器人观测处理器
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    if RESUME:
        dataset = LeRobotDataset(repo_id=LOCAL_DATASET_PATH)
        dataset.start_image_writer(num_processes=0, num_threads=4)
        log_say(f"恢复录制：已有 {dataset.meta.total_episodes} 个 episodes，继续录制 {NUM_EPISODES} 个")
    else:
        dataset = LeRobotDataset.create(
            repo_id=LOCAL_DATASET_PATH,
            fps=FPS,
            features=combine_feature_dicts(
                aggregate_pipeline_dataset_features(
                    pipeline=teleop_action_processor,
                    initial_features=create_initial_features(action=so101.action_features),
                    use_videos=True,
                ),
                aggregate_pipeline_dataset_features(
                    pipeline=robot_observation_processor,
                    initial_features=create_initial_features(observation=robot.observation_features),
                    use_videos=True,
                ),
            ),
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )

    robot.connect()
    so101.connect()

    listener, events = init_keyboard_listener()

    try:
        if not robot.is_connected or not so101.is_connected:
            raise ValueError("机器人或遥操作器未连接！")

        print("\n" + "=" * 50)
        print("录制操作指南:")
        print("  → (右箭头): 开始/结束当前 episode")
        print("  ← (左箭头): 结束并重录当前 episode")
        print("  Esc:        终止整个录制")
        print("=" * 50)

        episode_idx = 0
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            # --- 等待用户按键开始录制 ---
            log_say(f"准备录制 episode {episode_idx + 1} / {NUM_EPISODES}")
            if not wait_for_key(events, f"按 → 开始录制 episode {episode_idx + 1} / {NUM_EPISODES}"):
                break

            # 每轮开始前重置处理器状态（清除上一轮的累积状态）
            teleop_action_processor.reset()
            robot_action_processor.reset()

            log_say(f"开始录制 episode {episode_idx + 1} / {NUM_EPISODES}")
            print("录制中... 按 → 结束本轮, 按 ← 重录, 按 Esc 终止")

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=so101,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=False,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            # --- 处理重录 ---
            if events["rerecord_episode"]:
                log_say("重新录制本轮 episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # --- 保存 episode ---
            dataset.save_episode()
            log_say(f"Episode {episode_idx + 1} 已保存")
            episode_idx += 1

            # 最后一轮不需要重置环境
            if episode_idx >= NUM_EPISODES or events["stop_recording"]:
                break

            # --- 重置环境阶段 ---
            log_say("请重置环境（可使用示教器移动机器人）")
            if not wait_for_key(events, "重置环境完成后，按 → 开始下一轮录制"):
                break

    finally:
        log_say("录制结束")
        robot.disconnect()
        so101.disconnect()
        if listener:
            listener.stop()

        dataset.finalize()
        print(f"\n数据集已保存至: {LOCAL_DATASET_PATH}")
        print(f"本次录制 {episode_idx} 个 episodes，数据集共计 {dataset.meta.total_episodes} 个 episodes")


if __name__ == "__main__":
    main()

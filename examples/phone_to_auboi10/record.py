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
from lerobot.robots.aubo_i10.robot_processor import (
    AuboEEBoundsAndSafety,
    AuboGripperVelocityToPosition,
    AuboLockVerticalYaw,
    PhoneEEToAuboEE,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say, init_logging

NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 40
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "抓取竹条"
LOCAL_DATASET_PATH = "./datasets/phone_auboi10_s3"

# 相机用稳定的 by-id 路径，避免重启/重插后 /dev/videoN 重新编号导致 handeye/fixed 错位。
# handeye = GENERAL WEBCAM（机械臂末端），fixed = USB2.0_CAM1（固定机位）。
# fourcc="MJPG" 必须，否则 USB2.0_CAM1(05a3:9230) 在 OpenCV 里读线程起不来 → 占位图。
HANDEYE_DEV = "/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH0319_20210712_v102-video-index0"
FIXED_DEV = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"


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

    camera_config = {
        "handeye": OpenCVCameraConfig(index_or_path=HANDEYE_DEV, width=640, height=480, fps=FPS, fourcc="MJPG"),
        "fixed": OpenCVCameraConfig(index_or_path=FIXED_DEV, width=640, height=480, fps=FPS, fourcc="MJPG"),
    }
    robot_config = AuboI10Config(cameras=camera_config)
    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)
    

    robot = AuboI10Robot(robot_config)
    phone = Phone(teleop_config)

    phone_to_robot_ee_pose_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            AuboLockVerticalYaw(
                yaw_velocity_mode=True,     # 操纵杆速度模式: 拨出去持续转, 回正停
                yaw_vel_gain=-0.035,        # 负号同旧 yaw_gain=-1.0 方向
                max_yaw_vel_deg_per_s=60.0,
                yaw_deadzone_deg=3.0,
                yaw_smoothing=0.0,
            ),
            PhoneEEToAuboEE(
                velocity_mode=True,
                end_effector_step_sizes={"x": 0.05, "y": 0.05, "z": 0.05},
                position_acceleration=4.0,
            ),
            AuboEEBoundsAndSafety(
                end_effector_bounds={"min": [-0.8, -1.2, 0.0], "max": [1.0, 0.0, 0.8]},
                max_ee_step_m=0.05,
            ),
            AuboGripperVelocityToPosition(),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # robot_action_processor：phone_to_robot_ee_pose_processor 已输出绝对位姿，此处为空通路
    ee_to_delta_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    dataset = LeRobotDataset.create(
        repo_id=LOCAL_DATASET_PATH,
        fps=FPS,
        features=combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=phone_to_robot_ee_pose_processor,
                initial_features=create_initial_features(action=phone.action_features),
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
    phone.connect()

    listener, events = init_keyboard_listener()

    try:
        if not robot.is_connected or not phone.is_connected:
            raise ValueError("Robot or teleop is not connected!")

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
            phone_to_robot_ee_pose_processor.reset()
            ee_to_delta_processor.reset()

            log_say(f"开始录制 episode {episode_idx + 1} / {NUM_EPISODES}")
            print("录制中... 按 → 结束本轮, 按 ← 重录, 按 Esc 终止")

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=phone,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=False,
                teleop_action_processor=phone_to_robot_ee_pose_processor,
                robot_action_processor=ee_to_delta_processor,
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
            # 关闭伺服模式，让示教器可以自由控制机器人
            robot.disable_servo_mode()
            log_say("请重置环境（可使用示教器移动机器人）")
            if not wait_for_key(events, "重置环境完成后，按 → 开始下一轮录制"):
                break

    finally:
        log_say("录制结束")
        robot.disconnect()
        phone.disconnect()
        if listener:
            listener.stop()

        dataset.finalize()
        print(f"\n数据集已保存至: {LOCAL_DATASET_PATH}")
        print(f"共录制 {episode_idx} 个 episodes")


if __name__ == "__main__":
    main()

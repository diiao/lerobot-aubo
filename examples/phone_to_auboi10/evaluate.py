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

import os
import time
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
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
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot, AuboI10Config
from lerobot.robots.aubo_i10.robot_processor import AuboEEBoundsAndSafety, AuboSetEEMode
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say, init_logging

NUM_EPISODES = 5
EXPECTED_CONTROL_FPS = 25
HANDEYE_CAPTURE_FPS = 30
FIXED_CAPTURE_FPS = 25
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "抓取竹条"
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "./models/bamboo_newview_act/best")
TRAINING_DATASET_PATH = os.environ.get("DATASET_PATH", "./datasets/bamboo_newview_full")
LOCAL_EVAL_DATASET_PATH = os.environ.get("EVAL_DATASET_PATH", "./datasets/bamboo_newview_eval_direct")


def wait_for_key(events: dict, prompt: str = "按 → (右箭头键) 继续") -> bool:
    """等待用户按下右箭头键继续，或按 Esc 退出。"""
    events["exit_early"] = False
    print("\n" + "=" * 50)
    print(prompt)
    print("按 Esc 终止")
    print("=" * 50)
    while not events["exit_early"] and not events["stop_recording"]:
        time.sleep(0.05)
    if events["stop_recording"]:
        return False
    events["exit_early"] = False
    return True


def main():
    # ------------------------------------------------------------------
    # 0. 日志
    # ------------------------------------------------------------------
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    init_logging(log_file=str(log_dir / "evaluate.log"))

    training_metadata = LeRobotDatasetMetadata(TRAINING_DATASET_PATH)
    control_fps = int(training_metadata.fps)
    if control_fps != EXPECTED_CONTROL_FPS:
        raise ValueError(
            f"新视角纯 ACT 数据应为 {EXPECTED_CONTROL_FPS} FPS，"
            f"但 {TRAINING_DATASET_PATH} 是 {control_fps} FPS"
        )

    # ------------------------------------------------------------------
    # 1. 配置机器人（必须包含相机，策略需要图像输入）
    # ------------------------------------------------------------------
    # 注意：评估在 GPU 机上运行时，相机经 usbip 从本地转发过来，/dev/videoN 节点号
    # 不一定还是 0/2。attach 后用 `v4l2-ctl --list-devices` 确认 "GENERAL WEBCAM"
    # (handeye，当前眼在手外相机) 和 "USB2.0_CAM1" (fixed) 各自落到哪个节点，按实际修改。
    # handeye/fixed 必须与训练时对应同一物理相机，否则图像特征对错位。
    camera_config = {
        "handeye": OpenCVCameraConfig(
            index_or_path="/dev/video0",
            width=640,
            height=480,
            fps=HANDEYE_CAPTURE_FPS,
            fourcc="MJPG",
            warmup_s=3,
        ),
        "fixed": OpenCVCameraConfig(
            index_or_path="/dev/video2",
            width=640,
            height=480,
            fps=FIXED_CAPTURE_FPS,
            fourcc="MJPG",
            warmup_s=3,
        ),
    }
    robot_config = AuboI10Config(cameras=camera_config, control_fps=control_fps)
    robot = AuboI10Robot(robot_config)

    # ------------------------------------------------------------------
    # 2. 加载训练好的 ACT 策略
    # ------------------------------------------------------------------
    policy = ACTPolicy.from_pretrained(LOCAL_MODEL_PATH)
    policy.eval()
    print(f"模型已加载: {LOCAL_MODEL_PATH}, device={policy.config.device}")

    # ------------------------------------------------------------------
    # 3. 构建处理器管线
    # ------------------------------------------------------------------
    # 训练时 record.py 用 abs_j6yaw 管线保存绝对 EE 位姿（含 ee.j6_target）。
    # 评估时策略输出同样的绝对位姿向量，直接下发即可，不需要转 delta；
    # 但策略输出向量不含 ee_mode（字符串不入数据集），用 AuboSetEEMode 补回去，
    # 让 send_action 走 _send_position_j6yaw（与训练一致）。
    ee_mode_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            AuboEEBoundsAndSafety(
                end_effector_bounds={"min": [-0.8, -1.2, 0.0], "max": [1.0, 0.0, 0.8]},
                max_ee_step_m=0.05,
            ),
            AuboSetEEMode(ee_mode="abs_j6yaw"),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # ------------------------------------------------------------------
    # 4. 使用训练数据集的统计量 + 模型配置创建预/后处理器
    # ------------------------------------------------------------------
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=LOCAL_MODEL_PATH,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # ------------------------------------------------------------------
    # 5. 创建评估数据集（复用训练数据集的 features 定义）
    # ------------------------------------------------------------------
    dataset = LeRobotDataset.create(
        repo_id=LOCAL_EVAL_DATASET_PATH,
        fps=control_fps,
        features=training_metadata.features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # ------------------------------------------------------------------
    # 6. 连接机器人并执行推理
    # ------------------------------------------------------------------
    robot.connect()
    listener, events = init_keyboard_listener()

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        print("\n" + "=" * 50)
        print("评估操作指南:")
        print("  → (右箭头): 开始/结束当前 episode")
        print("  ← (左箭头): 结束并重录当前 episode")
        print("  Esc:        终止整个评估")
        print("=" * 50)

        episode_idx = 0
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            # --- 等待用户按键开始 ---
            log_say(f"准备执行 episode {episode_idx + 1} / {NUM_EPISODES}")
            if not wait_for_key(events, f"按 → 开始执行 episode {episode_idx + 1} / {NUM_EPISODES}"):
                break

            # 重置处理器状态
            ee_mode_processor.reset()

            log_say(f"开始推理 episode {episode_idx + 1} / {NUM_EPISODES}")
            print("推理中... 按 → 结束本轮, 按 ← 重录, 按 Esc 终止")

            record_loop(
                robot=robot,
                events=events,
                fps=control_fps,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=False,
                teleop_action_processor=make_default_teleop_action_processor(),
                robot_action_processor=ee_mode_processor,
                robot_observation_processor=robot_observation_processor,
            )

            # --- 处理重录 ---
            if events["rerecord_episode"]:
                log_say("重新执行本轮 episode")
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
            robot.disable_servo_mode()
            log_say("请重置环境（可使用示教器移动机器人）")
            if not wait_for_key(events, "重置环境完成后，按 → 开始下一轮推理"):
                break

    finally:
        log_say("评估结束")
        robot.disconnect()
        if listener:
            listener.stop()

        dataset.finalize()
        print(f"\n评估数据已保存至: {LOCAL_EVAL_DATASET_PATH}")
        print(f"共执行 {episode_idx} 个 episodes")


if __name__ == "__main__":
    main()

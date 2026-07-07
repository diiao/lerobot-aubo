# !/usr/bin/env python

"""试跑推理：用 ACT checkpoint 在机器人上跑一段，无需键盘。

远程推理（GPU 机）跑此脚本：相机经 usbip 从本地转发，机器人走 LAN RPC。
倒计时自动开始，跑 DURATION_S 秒后退出，不保存数据集。
"""

import shutil
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
from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Config, AuboI10Robot
from lerobot.robots.aubo_i10.robot_processor import AuboSetEEMode
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.utils import init_logging

FPS = 30
DURATION_S = 30        # 单次推理时长
COUNTDOWN_S = 5        # 倒计时秒数
TASK_DESCRIPTION = "抓取苹果到蓝色的盒子里"

# 最终模型（训练循环后 save_pretrained 到根目录；checkpoint_5000/10000/15000 是中间快照）
LOCAL_MODEL_PATH = "./models/phone_auboi10"
TRAINING_DATASET_PATH = "./datasets/phone_auboi10"

# GPU 机上 usbip attach 后的相机节点（v4l2-ctl --list-devices 确认）：
#   /dev/video0 = GENERAL WEBCAM  -> handeye（与训练数据一致）
#   /dev/video2 = USB2.0_CAM1     -> fixed
HANDEYE_DEV = "/dev/video0"
FIXED_DEV = "/dev/video2"


def main():
    init_logging(console_level="INFO")
    # 清掉上轮残留的临时数据集目录（LeRobotDataset.create 不允许目录已存在）
    eval_trial_cache = Path.home() / ".cache/huggingface/lerobot/datasets/phone_auboi10_eval_trial"
    if eval_trial_cache.exists():
        shutil.rmtree(eval_trial_cache)
        print(f"已清理旧目录: {eval_trial_cache}")
    camera_config = {
        "handeye": OpenCVCameraConfig(index_or_path=HANDEYE_DEV, width=640, height=480, fps=FPS, fourcc="MJPG"),
        "fixed": OpenCVCameraConfig(index_or_path=FIXED_DEV, width=640, height=480, fps=FPS, fourcc="MJPG"),
    }
    robot_config = AuboI10Config(cameras=camera_config)
    robot = AuboI10Robot(robot_config)

    # 加载模型
    policy = ACTPolicy.from_pretrained(LOCAL_MODEL_PATH)
    policy.eval()
    print(f"模型已加载: {LOCAL_MODEL_PATH} | device={policy.config.device}")

    # 处理器：与训练一致（abs_j6yaw），策略输出向量不含 ee_mode，用 AuboSetEEMode 补回
    ee_mode_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[AuboSetEEMode(ee_mode="abs_j6yaw")],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # 用训练数据集的 features + stats 建预/后处理器（record_loop 需要 dataset.features 构帧）
    training_metadata = LeRobotDatasetMetadata(TRAINING_DATASET_PATH)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=LOCAL_MODEL_PATH,
        dataset_stats=training_metadata.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    # 临时数据集只为提供 features 给 record_loop，不保存 episode
    dataset = LeRobotDataset.create(
        repo_id="./datasets/phone_auboi10_eval_trial",
        fps=FPS,
        features=training_metadata.features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=2,
    )

    robot.connect()
    if not robot.is_connected:
        raise ValueError("Robot not connected!")

    print("\n" + "=" * 50)
    print(f"推理试跑：{DURATION_S}s | 任务: {TASK_DESCRIPTION}")
    print("把机器人和苹果摆到训练时的起始姿态。倒计时结束就开始。")
    print("=" * 50)
    for i in range(COUNTDOWN_S, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    print(">>> 开始推理（Ctrl+C 提前终止）")

    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
    try:
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=DURATION_S,
            single_task=TASK_DESCRIPTION,
            display_data=False,
            teleop_action_processor=make_default_teleop_action_processor(),
            robot_action_processor=ee_mode_processor,
            robot_observation_processor=robot_observation_processor,
        )
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
        print("推理结束")


if __name__ == "__main__":
    main()

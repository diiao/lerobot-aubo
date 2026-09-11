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

import logging
import math
import os
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

from aubo_start_poses import NORMAL_START_DEG, RECOVERY_START_DEG

NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "4"))
CONTROL_FPS = 25
HANDEYE_CAPTURE_FPS = 30  # 驱动只接受 30；硬件实测约 25 FPS
FIXED_CAPTURE_FPS = 25
EPISODE_TIME_SEC = 120
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "抓取竹条"
# 汇报用 ACT 单根角度批次。s01 已录的是 90°，角度以
# datasets/bamboo_act_report_manifest.json 为准，禁止用 sXX 序号反推。
# 必须传 root= 本地目录，否则会写到 ~/.cache/huggingface/lerobot/。
_DATASET_PATH = Path(os.environ.get("DATASET_PATH", "./datasets/bamboo_act_report_s14"))
LOCAL_DATASET_ROOT = _DATASET_PATH.resolve()
LOCAL_DATASET_REPO_ID = LOCAL_DATASET_ROOT.name
RECORD_START_MODE = os.environ.get("RECORD_START_MODE", "normal").strip().lower()
if RECORD_START_MODE not in {"normal", "recovery"}:
    raise ValueError("RECORD_START_MODE 只能是 'normal' 或 'recovery'")

# 相机用稳定的 by-id 路径，避免重启/重插后 /dev/videoN 重新编号导致 handeye/fixed 错位。
# handeye = GENERAL WEBCAM（眼在手外，当前重新调整的眼相机）。
# fixed = USB2.0_CAM1（另一视角；录制/训练/推理期间必须始终保持相同键名映射）。
# USB2.0_CAM1(05a3:9230) 使用 640x480 MJPG 25 FPS。YUYV 长测仍会触发
# USB 重新枚举，不能作为规避方案，并且会显著增加 USB2.0 带宽。
HANDEYE_DEV = "/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH0319_20210712_v102-video-index0"
FIXED_DEV = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"

# 两个起点均来自 aubo_start_poses.py：normal 与推理相同；recovery 是固定失败恢复状态。
NORMAL_MOVE_SPEED_FRACTION = 0.5
RECOVERY_MOVE_SPEED_FRACTION = 0.2


def wait_for_move_completion(motion, start_timeout_s: float = 5.0) -> bool:
    """等待 moveJoint 确实启动并结束，未启动时不继续执行下一段动作。"""
    deadline = time.monotonic() + start_timeout_s
    while motion.getExecId() == -1:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    while motion.getExecId() != -1:
        time.sleep(0.05)
    return True


def move_joint_and_wait(motion, target_deg, speed_fraction, speed_deg_s, accel_deg_s2) -> bool:
    """以明确速度移动到一个关节目标；只在运动完成后返回成功。"""
    target_rad = [math.radians(value) for value in target_deg]
    motion.setSpeedFraction(speed_fraction)
    motion.moveJoint(
        target_rad,
        math.radians(speed_deg_s),
        math.radians(accel_deg_s2),
        0,
        0,
    )
    return wait_for_move_completion(motion)


def return_to_start(robot):
    """按当前录制模式归位；normal 保持旧行为，recovery 经过标准位再低速到恢复位。"""
    logging.info("归位模式：%s", RECORD_START_MODE)
    motion = robot.robot_interface.getMotionControl()
    # 确保伺服关闭（moveJoint 需要在非伺服模式）
    if motion.isServoModeEnabled():
        motion.setServoMode(False)
        time.sleep(0.5)

    logging.info("归位：moveJoint 到标准起始位...")
    if not move_joint_and_wait(
        motion,
        NORMAL_START_DEG,
        NORMAL_MOVE_SPEED_FRACTION,
        speed_deg_s=80,
        accel_deg_s2=60,
    ):
        logging.error("未确认机械臂已到达标准起始位；本次归位失败")
        return False

    if RECORD_START_MODE == "recovery":
        logging.info("恢复模式：低速 moveJoint 到固定恢复起点...")
        if not move_joint_and_wait(
            motion,
            RECOVERY_START_DEG,
            RECOVERY_MOVE_SPEED_FRACTION,
            speed_deg_s=30,
            accel_deg_s2=30,
        ):
            logging.error("未确认机械臂已到达恢复起点；本次归位失败")
            return False

    # 通过机器人接口松吸盘，同时同步 is_suction_on 软件观测状态。
    try:
        if not robot.suction_release():
            logging.error("归位完成，但吸盘释放失败")
            return False
        else:
            logging.info("归位完成，吸盘已释放")
    except Exception as e:
        logging.error(f"吸盘释放失败: {e}")
        return False
    return True


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


def is_camera_read_failure(exc: Exception) -> bool:
    """只识别机器人观测层主动抛出的相机断流保护异常。"""
    message = str(exc)
    return isinstance(exc, RuntimeError) and "相机 " in message and "读取失败" in message


def reconnect_cameras(robot) -> bool:
    """重启所有学习相机，使两路后台取帧线程重新同步启动。"""
    for cam_name, cam in robot.cameras.items():
        try:
            if cam.is_connected or getattr(cam, "thread", None) is not None:
                cam.disconnect()
        except Exception:
            logging.warning("清理相机 %s 的旧连接失败", cam_name, exc_info=True)

    try:
        for cam_name, cam in robot.cameras.items():
            cam.connect(warmup=True)
            logging.info("相机 %s 重新连接成功", cam_name)
    except Exception:
        logging.error("相机重新连接失败", exc_info=True)
        # warmup 阶段也可能在失败相机上留下读取线程，因此回滚所有相机，
        # 让下一次按键重试从完全断开的状态开始。
        for cam_name, cam in robot.cameras.items():
            try:
                if cam.is_connected or getattr(cam, "thread", None) is not None:
                    cam.disconnect()
            except Exception:
                logging.warning("回滚相机 %s 重连失败", cam_name, exc_info=True)
        return False

    return True


def main():
    # Initialize logging: 控制台 INFO，文件 DEBUG
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / "record.log"
    init_logging(log_file=log_file_path, console_level="INFO", file_level="DEBUG")

    camera_config = {
        "handeye": OpenCVCameraConfig(
            index_or_path=HANDEYE_DEV,
            width=640,
            height=480,
            fps=HANDEYE_CAPTURE_FPS,
            fourcc="MJPG",
            warmup_s=3,
        ),
        "fixed": OpenCVCameraConfig(
            index_or_path=FIXED_DEV,
            width=640,
            height=480,
            fps=FIXED_CAPTURE_FPS,
            fourcc="MJPG",
            warmup_s=3,
        ),
    }
    robot_config = AuboI10Config(cameras=camera_config, control_fps=CONTROL_FPS)
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
            # 直接录制持续的目标吸盘状态 0/100，避免训练稀疏的按键脉冲。
            AuboGripperVelocityToPosition(latch=True),
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

    dataset = None
    listener = None
    episode_idx = 0
    try:
        # 把所有外部资源纳入 try/finally。任一连接或数据集创建步骤失败，
        # 都会释放已经打开的相机、机器人和手机连接。
        robot.connect()
        phone.connect()

        print(f"数据集将保存到: {LOCAL_DATASET_ROOT}")
        dataset = LeRobotDataset.create(
            repo_id=LOCAL_DATASET_REPO_ID,
            root=LOCAL_DATASET_ROOT,
            fps=CONTROL_FPS,
            features=combine_feature_dicts(
                aggregate_pipeline_dataset_features(
                    pipeline=phone_to_robot_ee_pose_processor,
                    initial_features=create_initial_features(action=phone.action_features),
                    use_videos=True,
                ),
                aggregate_pipeline_dataset_features(
                    pipeline=robot_observation_processor,
                    initial_features=create_initial_features(
                        observation=robot.observation_features
                    ),
                    use_videos=True,
                ),
            ),
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )

        listener, events = init_keyboard_listener()

        if not robot.is_connected or not phone.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print("\n" + "=" * 50)
        print("录制操作指南:")
        print("  → (右箭头): 开始/结束当前 episode")
        print("  ← (左箭头): 结束并重录当前 episode")
        if RECORD_START_MODE == "recovery":
            print("  r:          先回标准位，再低速到固定恢复起点（每条必做）")
        else:
            print("  r:          归位到统一正常起始位置")
        print("  Esc:        终止整个录制")
        print("=" * 50)

        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            # --- 等待用户按键开始录制（支持 r 归位）---
            log_say(f"准备录制 episode {episode_idx + 1} / {NUM_EPISODES}")
            robot.disable_servo_mode()  # 关伺服，让 r 归位能用 moveJoint
            print("\n" + "=" * 50)
            print(f"  按 -> 开始录制 episode {episode_idx + 1} / {NUM_EPISODES}")
            if RECORD_START_MODE == "recovery":
                print("  按 r 回标准位后低速到固定恢复起点（本模式必须先按 r）")
            else:
                print("  按 r 归位到统一正常起始位置")
            print("  按 Esc 终止")
            print("=" * 50)
            events["exit_early"] = False
            events["return_to_start"] = False
            # 恢复示教的有效性依赖固定起点，故每一条必须显式完成一次 r 归位。
            # normal 模式维持旧行为：操作者可选择 r 归位，或在已正确摆位时直接开始。
            episode_start_ready = RECORD_START_MODE == "normal"
            while not events["exit_early"] and not events["stop_recording"]:
                if events.get("return_to_start"):
                    events["return_to_start"] = False
                    episode_start_ready = return_to_start(robot)
                    if episode_start_ready:
                        if RECORD_START_MODE == "recovery":
                            print("已到固定恢复起点，摆好竹条后按 -> 开始录制")
                        else:
                            print("已归位到正常起点，摆好竹条后按 -> 开始录制")
                    else:
                        print("归位未完成；请检查机械臂后按 r 重试，不能开始本条录制")
                time.sleep(0.05)
            if events["stop_recording"]:
                break
            if not episode_start_ready:
                print("恢复模式必须先按 r 并确认归位成功，右箭头已忽略")
                events["exit_early"] = False
                continue
            events["exit_early"] = False

            # 每轮开始前重置处理器状态（清除上一轮的累积状态）
            phone_to_robot_ee_pose_processor.reset()
            ee_to_delta_processor.reset()

            log_say(f"开始录制 episode {episode_idx + 1} / {NUM_EPISODES}")
            print("录制中... 按 → 结束本轮, 按 ← 重录, 按 Esc 终止")

            try:
                record_loop(
                    robot=robot,
                    events=events,
                    fps=CONTROL_FPS,
                    teleop=phone,
                    dataset=dataset,
                    control_time_s=EPISODE_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=False,
                    teleop_action_processor=phone_to_robot_ee_pose_processor,
                    robot_action_processor=ee_to_delta_processor,
                    robot_observation_processor=robot_observation_processor,
                )
            except RuntimeError as exc:
                if not is_camera_read_failure(exc):
                    raise

                # 不在断流中途续接：先停机器人，再完整丢弃本轮，避免时间不连续的
                # 图像/动作进入同一个训练 episode。之前已保存的 episode 不受影响。
                robot.disable_servo_mode()
                dataset.clear_episode_buffer()
                events["rerecord_episode"] = False
                events["exit_early"] = False
                logging.error("相机断流，本轮已丢弃", exc_info=True)
                log_say(f"{exc}；当前 episode 已丢弃")

                while not events["stop_recording"]:
                    if not wait_for_key(
                        events,
                        "检查或重新插拔相机后，按 → 尝试重连；成功后将重录当前 episode",
                    ):
                        break
                    if reconnect_cameras(robot):
                        log_say("两台相机已重新连接，请归位后重录当前 episode")
                        break
                    print("相机重连仍失败；请继续检查设备，按 → 再试，或按 Esc 结束")

                if events["stop_recording"]:
                    break
                continue

            # Esc 会让 record_loop 提前返回；丢弃当前未完成片段，避免保存为训练 episode。
            if events["stop_recording"]:
                dataset.clear_episode_buffer()
                log_say("已丢弃未完成的 episode")
                break

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

            # 下一轮循环的“开始前等待”是唯一的归位/开始关口：按 r 归位、按右箭头开始。
            # 不再在这里重复等待一次，避免归位后需要按两次右箭头才能开始下一条。
            if episode_idx >= NUM_EPISODES or events["stop_recording"]:
                break

    except Exception:
        # 未被上面处理的异常也不能把半条 episode 留在数据集目录中。
        if (
            dataset is not None
            and dataset.episode_buffer is not None
            and dataset.episode_buffer.get("size", 0) > 0
        ):
            try:
                dataset.clear_episode_buffer()
                logging.info("异常退出前已清理未完成的 episode")
            except Exception:
                logging.error("异常退出时清理 episode 失败", exc_info=True)
        raise
    finally:
        log_say("录制结束")
        if listener:
            listener.stop()
        if phone.is_connected:
            phone.disconnect()
        if robot.is_connected:
            robot.disconnect()

        if dataset is not None:
            dataset.finalize()
            print(f"\n数据集已保存至: {LOCAL_DATASET_ROOT}")
            print(f"共录制 {episode_idx} 个 episodes")
        else:
            print("\n录制未开始，没有创建数据集")


if __name__ == "__main__":
    main()

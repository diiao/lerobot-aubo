#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""One-episode C0 smoke capture, pending formal final evidence binding.

This entry records exactly one teleoperated pick-and-place episode into a new
local dataset. It is not the formal ten-episode C0 batch, does not bind
C0EpisodeManifestV1, and does not grant training or policy execution.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from lerobot.bamboo_sorting.c0_smoke_capture import (
    C0_SMOKE_CAMERA_DEVICES,
    C0_SMOKE_DATASET_FPS,
    C0_SMOKE_IMAGE_KEYS,
    C0_SMOKE_NUM_EPISODES,
    C0_SMOKE_OPERATOR_CONFIRMATION,
    C0_SMOKE_STATUS,
    C0_SMOKE_TASK_TEXT,
    C0SmokeCaptureError,
    build_operator_checklist,
    c0_smoke_robot_observation_features,
    format_c0_smoke_postflight,
    frozen_c0_smoke_camera_mapping,
    maybe_run_c0_smoke_postflight,
    require_new_dataset_root,
    require_operator_confirmation,
    require_placement_reference,
    shutdown_c0_smoke_session,
)

EPISODE_TIME_SEC = 120
GLOBAL_RGB_CAPTURE_FPS = 30
GRASP_RGB_CAPTURE_FPS = 25


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=C0_SMOKE_STATUS)
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="New local dataset directory. Refused if it already exists.",
    )
    parser.add_argument(
        "--placement-reference",
        required=True,
        help="Non-empty physical scale mark or scale-line name. Never defaulted to 0°.",
    )
    return parser


def parse_c0_smoke_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.placement_reference = require_placement_reference(args.placement_reference)
    args.dataset_root = require_new_dataset_root(args.dataset_root)
    return args


def _load_record_helpers():
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import record as act_record

    act_record.RECORD_START_MODE = "normal"
    return act_record


def _confirm_or_abort(checklist: tuple[str, ...]) -> None:
    print("\n" + "=" * 60)
    for line in checklist:
        print(line)
    print("=" * 60)
    typed = input(f"Type {C0_SMOKE_OPERATOR_CONFIRMATION} to start: ")
    require_operator_confirmation(typed)


def main(argv: list[str] | None = None) -> int:
    args = parse_c0_smoke_args(argv)
    camera_mapping = frozen_c0_smoke_camera_mapping()
    if tuple(camera_mapping) != C0_SMOKE_IMAGE_KEYS:
        raise C0SmokeCaptureError("camera mapping keys must be global_rgb, grasp_rgb")
    if {key: mapping["device"] for key, mapping in camera_mapping.items()} != dict(
        C0_SMOKE_CAMERA_DEVICES
    ):
        raise C0SmokeCaptureError("camera devices do not match the frozen CameraSetV2 mapping")

    checklist = build_operator_checklist(
        placement_reference=args.placement_reference,
        dataset_root=args.dataset_root,
        camera_mapping=camera_mapping,
    )
    _confirm_or_abort(checklist)

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.datasets.utils import combine_feature_dicts
    from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
    from lerobot.processor.converters import (
        observation_to_transition,
        robot_action_observation_to_transition,
        transition_to_observation,
        transition_to_robot_action,
    )
    from lerobot.robots.aubo_i10.aubo_i10 import AuboI10Robot
    from lerobot.robots.aubo_i10.config_aubo_i10 import AuboI10Config
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
    from lerobot.utils.utils import init_logging, log_say

    act_record = _load_record_helpers()

    class C0SmokeAuboI10Robot(AuboI10Robot):
        @property
        def observation_features(self) -> dict[str, type | tuple]:
            return c0_smoke_robot_observation_features()

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    init_logging(log_file=log_dir / "record_c0_smoke.log", console_level="INFO", file_level="DEBUG")
    logging.info(C0_SMOKE_STATUS)
    logging.info("placement reference: %s", args.placement_reference)
    logging.info("dataset root: %s", args.dataset_root)

    camera_config = {
        "global_rgb": OpenCVCameraConfig(
            index_or_path=camera_mapping["global_rgb"]["device"],
            width=640,
            height=480,
            fps=GLOBAL_RGB_CAPTURE_FPS,
            fourcc="MJPG",
            warmup_s=3,
        ),
        "grasp_rgb": OpenCVCameraConfig(
            index_or_path=camera_mapping["grasp_rgb"]["device"],
            width=640,
            height=480,
            fps=GRASP_RGB_CAPTURE_FPS,
            fourcc="MJPG",
            warmup_s=3,
        ),
    }
    robot_config = AuboI10Config(cameras=camera_config, control_fps=C0_SMOKE_DATASET_FPS)
    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)
    robot = C0SmokeAuboI10Robot(robot_config)
    phone = Phone(teleop_config)

    phone_to_robot_ee_pose_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            AuboLockVerticalYaw(
                yaw_velocity_mode=True,
                yaw_vel_gain=-0.035,
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
            AuboGripperVelocityToPosition(latch=True),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
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
    shutdown_result = None
    try:
        robot.connect()
        phone.connect()

        print(f"数据集将保存到: {args.dataset_root}")
        print(C0_SMOKE_STATUS)
        dataset = LeRobotDataset.create(
            repo_id=args.dataset_root.name,
            root=args.dataset_root,
            fps=C0_SMOKE_DATASET_FPS,
            features=combine_feature_dicts(
                aggregate_pipeline_dataset_features(
                    pipeline=phone_to_robot_ee_pose_processor,
                    initial_features=create_initial_features(action=phone.action_features),
                    use_videos=True,
                ),
                aggregate_pipeline_dataset_features(
                    pipeline=robot_observation_processor,
                    initial_features=create_initial_features(
                        observation=c0_smoke_robot_observation_features()
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
        print("C0 smoke 录制操作指南:")
        print("  → (右箭头): 开始/结束当前 episode")
        print("  ← (左箭头): 结束并重录当前 episode")
        print("  r:          归位到统一正常起始位置")
        print("  Esc:        终止整个录制")
        print("=" * 50)

        while episode_idx < C0_SMOKE_NUM_EPISODES and not events["stop_recording"]:
            log_say(f"准备录制 episode {episode_idx + 1} / {C0_SMOKE_NUM_EPISODES}")
            robot.disable_servo_mode()
            print("\n" + "=" * 50)
            print(f"  按 -> 开始录制 episode {episode_idx + 1} / {C0_SMOKE_NUM_EPISODES}")
            print("  按 r 归位到统一正常起始位置")
            print("  按 Esc 终止")
            print("=" * 50)
            events["exit_early"] = False
            events["return_to_start"] = False
            episode_start_ready = True
            while not events["exit_early"] and not events["stop_recording"]:
                if events.get("return_to_start"):
                    events["return_to_start"] = False
                    episode_start_ready = act_record.return_to_start(robot)
                    if episode_start_ready:
                        print("已归位到正常起点，摆好木条后按 -> 开始录制")
                    else:
                        print("归位未完成；请检查机械臂后按 r 重试，不能开始本条录制")
                time.sleep(0.05)
            if events["stop_recording"]:
                break
            if not episode_start_ready:
                print("必须确认归位成功后才能开始；右箭头已忽略")
                events["exit_early"] = False
                continue
            events["exit_early"] = False

            phone_to_robot_ee_pose_processor.reset()
            ee_to_delta_processor.reset()
            log_say(f"开始录制 episode {episode_idx + 1} / {C0_SMOKE_NUM_EPISODES}")
            print("录制中... 按 → 结束本轮, 按 ← 重录, 按 Esc 终止")

            try:
                record_loop(
                    robot=robot,
                    events=events,
                    fps=C0_SMOKE_DATASET_FPS,
                    teleop=phone,
                    dataset=dataset,
                    control_time_s=EPISODE_TIME_SEC,
                    single_task=C0_SMOKE_TASK_TEXT,
                    display_data=False,
                    teleop_action_processor=phone_to_robot_ee_pose_processor,
                    robot_action_processor=ee_to_delta_processor,
                    robot_observation_processor=robot_observation_processor,
                )
            except RuntimeError as exc:
                if not act_record.is_camera_read_failure(exc):
                    raise
                robot.disable_servo_mode()
                dataset.clear_episode_buffer()
                events["rerecord_episode"] = False
                events["exit_early"] = False
                logging.error("相机断流，本轮已丢弃", exc_info=True)
                log_say(f"{exc}；当前 episode 已丢弃")
                while not events["stop_recording"]:
                    if not act_record.wait_for_key(
                        events,
                        "检查或重新插拔相机后，按 → 尝试重连；成功后将重录当前 episode",
                    ):
                        break
                    if act_record.reconnect_cameras(robot):
                        log_say("两台相机已重新连接，请归位后重录当前 episode")
                        break
                    print("相机重连仍失败；请继续检查设备，按 → 再试，或按 Esc 结束")
                if events["stop_recording"]:
                    break
                continue

            if events["stop_recording"]:
                dataset.clear_episode_buffer()
                log_say("已丢弃未完成的 episode")
                break

            if events["rerecord_episode"]:
                log_say("重新录制本轮 episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            log_say(f"Episode {episode_idx + 1} 已保存")
            episode_idx += 1
            break

    except Exception:
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
        shutdown_result = shutdown_c0_smoke_session(
            dataset=dataset,
            listener=listener,
            phone=phone,
            robot=robot,
        )
        try:
            log_say("录制结束")
        except Exception:
            logging.error("log_say during C0 smoke shutdown failed", exc_info=True)
        if dataset is not None:
            print(f"\n数据集目录: {args.dataset_root}")
            print(f"已保存 episodes: {episode_idx}")
            print(C0_SMOKE_STATUS)
        else:
            print("\n录制未开始，没有创建数据集")

    assert shutdown_result is not None
    report = maybe_run_c0_smoke_postflight(
        shutdown_result=shutdown_result,
        dataset=dataset,
        episode_idx=episode_idx,
        dataset_root=args.dataset_root,
    )
    print("\n" + format_c0_smoke_postflight(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

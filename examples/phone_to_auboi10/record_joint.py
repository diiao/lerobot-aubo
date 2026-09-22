#!/usr/bin/env python
"""AUBO i10 joint-space SmolVLA capture. Default: print plan, no hardware.

--record explicitly starts supervised hardware capture. Images and 7D joint
state/action are saved to a NEW dataset; historical C0 entries stay intact.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from lerobot.bamboo_sorting.aubo_joint_contract import (
    JOINT_FIELDS, JOINT_FPS, JOINT_IMAGE_KEYS, JOINT_TASK,
    joint_command, joint_contract_record, joint_dataset_features,
)
from lerobot.bamboo_sorting.aubo_joint_capture import JointCaptureSession, write_joint_json
from lerobot.bamboo_sorting.c0_smoke_capture import frozen_c0_smoke_camera_mapping


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--num-episodes", type=int, default=60)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--record", action="store_true", help="Connect cameras/phone/AUBO and enable supervised motion/IO")
    args = parser.parse_args(argv)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.evidence_root = args.evidence_root.expanduser().resolve()
    if args.num_episodes < 1:
        parser.error("num-episodes must be positive")
    if (args.dataset_root == args.evidence_root or args.dataset_root in args.evidence_root.parents
            or args.evidence_root in args.dataset_root.parents):
        parser.error("dataset and evidence roots must be separate, not nested")
    for path in (args.dataset_root, args.evidence_root):
        if path.exists():
            parser.error(f"refusing existing output path: {path}")
    return args


def learning_observation(raw):
    """TCP is available to phone teleop, but excluded from dataset/model state."""
    state = joint_command({key: raw[key] for key in JOINT_FIELDS})
    return {**state, **{key: raw[key] for key in JOINT_IMAGE_KEYS}}


class IdentityAction:
    def __call__(self, pair):
        return joint_command(pair[0])

    def reset(self):
        pass


def required_text(prompt):
    """Retry empty metadata without inventing or copying a scene identity."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("这一项不能为空，请输入内容后再按回车。")


def main(argv=None):
    args = parse_args(argv)
    cameras = frozen_c0_smoke_camera_mapping()
    plan = {"contract": joint_contract_record(), "dataset_root": str(args.dataset_root),
            "evidence_root": str(args.evidence_root), "num_episodes": args.num_episodes,
            "split": args.split, "camera_mapping": cameras,
            "teleoperation_profile": "legacy_abs_j6yaw",
            "mode": "supervised_capture" if args.record else "plan_only"}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.record:
        print("只显示方案；加 --record 才会连接设备并允许人工遥操作。")
        return 0

    # All hardware-capable imports are after the explicit --record boundary.
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor import RobotProcessorPipeline
    from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
    from lerobot.robots.aubo_i10.config_aubo_i10 import AuboI10Config
    from lerobot.robots.aubo_i10.joint_capture import AuboI10JointCaptureRobot, PhoneToJointAction
    from lerobot.robots.aubo_i10.robot_processor import (
        AuboEEBoundsAndSafety, AuboGripperVelocityToPosition, AuboLockVerticalYaw, PhoneEEToAuboEE,
    )
    from lerobot.scripts.lerobot_record import finalize_recorded_dataset, record_episode_sessions
    from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
    from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
    from lerobot.teleoperators.phone.teleop_phone import Phone
    from lerobot.utils.control_utils import init_keyboard_listener

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from record_c0_batch import read_fresh_c0_observation, _load_record_helpers
    helpers = _load_record_helpers()

    def return_to_joint_start(robot):
        if not robot.disable_servo_mode():
            return False
        # Reuse the complete legacy homing routine, including J5=90.88 degrees.
        return helpers.return_to_start(robot)

    class FreshJointRobot(AuboI10JointCaptureRobot):
        def __init__(self, config):
            super().__init__(config)
            self.previous_camera_timestamps = {}

        def get_observation(self):
            return read_fresh_c0_observation(
                read_observation=super().get_observation,
                read_timestamps=lambda: self.last_c0_sensor_timestamps,
                previous_camera_timestamps=self.previous_camera_timestamps,
            )

    camera_configs = {name: OpenCVCameraConfig(index_or_path=cameras[name]["device"],
                      width=640, height=480, fps=30 if name == "global_rgb" else 25,
                      fourcc="MJPG", warmup_s=3) for name in JOINT_IMAGE_KEYS}
    robot = FreshJointRobot(AuboI10Config(cameras=camera_configs, control_fps=JOINT_FPS))
    phone_config = PhoneConfig(phone_os=PhoneOS.ANDROID)
    phone = Phone(phone_config)
    phone_pipeline = RobotProcessorPipeline(
        steps=[MapPhoneActionToRobotAction(platform=phone_config.phone_os),
               AuboLockVerticalYaw(yaw_velocity_mode=True, yaw_vel_gain=-0.035,
                                   max_yaw_vel_deg_per_s=60.0, yaw_deadzone_deg=3.0, yaw_smoothing=0.0),
               PhoneEEToAuboEE(velocity_mode=True, end_effector_step_sizes={"x": 0.05, "y": 0.05, "z": 0.05},
                               position_acceleration=4.0),
               AuboEEBoundsAndSafety(end_effector_bounds={"min": [-0.8, -1.2, 0.0], "max": [1.0, 0.0, 0.8]},
                                     max_ee_step_m=0.05),
               AuboGripperVelocityToPosition(latch=True)],
        to_transition=robot_action_observation_to_transition, to_output=transition_to_robot_action,
    )
    teleop_processor = PhoneToJointAction(phone_pipeline, robot)
    dataset = listener = session = None
    args.evidence_root.mkdir(parents=True, exist_ok=False)
    write_joint_json(args.evidence_root / "plan.json", plan)
    try:
        robot.connect()
        phone.connect()
        dataset = LeRobotDataset.create(repo_id=args.dataset_root.name, root=args.dataset_root,
                    fps=JOINT_FPS, features=joint_dataset_features(), robot_type=robot.name,
                    use_videos=True, image_writer_threads=4)
        write_joint_json(args.dataset_root / "meta" / "aubo_joint_contract.json", joint_contract_record())
        write_joint_json(args.evidence_root / "controller_limits.json", {
            "lower_deg": list(robot.joint_lower_deg), "upper_deg": list(robot.joint_upper_deg),
            "teleoperation_profile": "legacy_abs_j6yaw"})
        events = dict(exit_early=False, return_to_start=False, stop_recording=False, rerecord_episode=False)

        def prepare_episode(index):
            nonlocal listener
            if not robot.disable_servo_mode():
                raise RuntimeError("cannot disable servo before scene preparation")
            # The terminal hotkey reader and input() must never consume stdin together.
            if listener is not None:
                listener.stop()
                listener = None
            print(f"\n准备第 {index + 1}/{args.num_episodes} 条；数据划分：{args.split}")
            scene = required_text("场景编号（同一摆放场景保持同一编号，不跨训练/验证/测试）：")
            placement = required_text("物理刻度线/摆放参考（例如尺线 B、方向 90°）：")
            listener, _ = init_keyboard_listener(events)
            print("按 r 归位，摆好竹条后按 → 开始；录制中 → 结束，← 重录，Esc 停止。")
            events["exit_early"] = events["return_to_start"] = False
            ready = False
            while not events["stop_recording"]:
                if events.get("return_to_start"):
                    events["return_to_start"] = False
                    ready = return_to_joint_start(robot)
                if events["exit_early"]:
                    events["exit_early"] = False
                    if ready:
                        break
                    print("请先按 r 完成归位。")
                time.sleep(0.05)
            # Ensure record_loop exits immediately if stopped during preparation.
            events["exit_early"] = bool(events["stop_recording"])
            teleop_processor.reset()
            return {"scene_id": scene, "placement_reference": placement, "split": args.split}

        def outcome_provider(index):
            nonlocal listener
            if not robot.disable_servo_mode():
                raise RuntimeError("cannot disable servo before outcome annotation")
            if listener is not None:
                listener.stop()
                listener = None
            choices = {"1": "single_success", "2": "empty", "3": "multi_pick", "4": "slip",
                       "5": "blocked", "6": "uncertain"}
            while True:
                choice = input(f"第 {index + 1} 条结果：1 单根成功，2 抓空，3 多抓，4 滑落，5 受阻，6 不确定：").strip()
                if choice in choices:
                    return choices[choice]

        session = JointCaptureSession(dataset_root=args.dataset_root, evidence_root=args.evidence_root,
                    episode_count=args.num_episodes, prepare_episode=prepare_episode, outcome_provider=outcome_provider)
        record_episode_sessions(robot=robot, teleop=phone, policy=None, preprocessor=None, postprocessor=None,
                    teleop_action_processor=teleop_processor, robot_action_processor=IdentityAction(),
                    robot_observation_processor=learning_observation, dataset=dataset, events=events,
                    fps=JOINT_FPS, num_episodes=args.num_episodes, episode_time_s=120, reset_time_s=0,
                    single_task=JOINT_TASK, play_sounds=False, display_data=False,
                    display_compressed_images=False, lifecycle=session)
    finally:
        original_error = sys.exc_info()[1]
        # Stop servo before potentially slow video finalization, including on failure.
        try:
            if not robot.disable_servo_mode():
                logging.error("cannot confirm servo disabled before joint dataset finalization")
        except Exception:
            logging.exception("joint capture servo shutdown failed before finalization")
        try:
            if dataset is not None:
                finalize_recorded_dataset(dataset, lifecycle=session,
                    recorded_episode_count=session.saved_count if session is not None else 0)
        except BaseException:
            if original_error is None:
                raise
            logging.exception("joint dataset finalization failed during capture failure")
        finally:
            # Attempt every cleanup even if one device fails to disconnect.
            for cleanup in (lambda: robot.disable_servo_mode(),
                            lambda: listener.stop() if listener else None,
                            lambda: phone.disconnect() if phone.is_connected else None,
                            robot.disconnect):
                try:
                    cleanup()
                except Exception:
                    logging.exception("joint capture cleanup failed")
    print(f"已保存 {session.saved_count} 条关节示范。请先运行 audit_joint_dataset.py 核对。")
    print("人工结果标签和控制器回读不等于物理成功证明；未启动训练或策略执行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

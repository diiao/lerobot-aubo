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

"""Separately authorized entry for ten NEW formal C0 capture candidates.

Importing this script is hardware-free.  ``main`` refuses existing output
roots, displays the complete plan/checklist, and requires the exact operator
confirmation before importing camera, AUBO, phone, or recording classes.

The earlier one-episode smoke dataset is preserved as excluded diagnostic
evidence.  It is not silently counted because it predates the required
per-sensor timestamp and final-evidence sidecars.

The entry saves raw dataset episodes plus timestamp/gripper/human-outcome
capture sidecars.  It deliberately does not claim final C0 manifest binding,
training authorization, inference authorization, or policy execution.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from lerobot.bamboo_sorting.c0_batch_capture import (
    C0_FORMAL_BATCH_STATUS,
    C0_FORMAL_EPISODE_COUNT,
    C0_FORMAL_TASK_TEXT,
    C0BatchCaptureError,
    C0BatchCaptureSessionV1,
    C0EpisodeCapturePlanV1,
    build_c0_batch_capture_plan,
    build_c0_batch_operator_checklist,
    operator_confirmation_for_plan,
    require_formal_operator_confirmation,
    require_new_c0_batch_paths,
)
from lerobot.bamboo_sorting.c0_smoke_capture import (
    C0_SMOKE_CAMERA_DEVICES,
    C0_SMOKE_DATASET_FPS,
    C0_SMOKE_IMAGE_KEYS,
    c0_smoke_robot_observation_features,
    frozen_c0_smoke_camera_mapping,
)

EPISODE_TIME_SEC = 120
GLOBAL_RGB_CAPTURE_FPS = 30
GRASP_RGB_CAPTURE_FPS = 25


def read_fresh_c0_observation(
    *,
    read_observation: Callable[[], dict[str, Any]],
    read_timestamps: Callable[[], object],
    previous_camera_timestamps: dict[str, float],
    timeout_s: float = 0.2,
    poll_interval_s: float = 0.002,
    monotonic: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Read until both formal cameras have a frame newer than the last saved read."""

    deadline = monotonic() + timeout_s
    stale_streams: list[str] = []
    while True:
        observation = read_observation()
        raw_timestamps = read_timestamps()
        current: dict[str, float] = {}
        stale_streams = []
        for stream in C0_SMOKE_IMAGE_KEYS:
            value = raw_timestamps.get(stream) if isinstance(raw_timestamps, Mapping) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                stale_streams.append(stream)
                continue
            current[stream] = float(value)
            previous = previous_camera_timestamps.get(stream)
            if previous is not None and current[stream] <= previous:
                stale_streams.append(stream)

        if not stale_streams:
            previous_camera_timestamps.update(current)
            return observation
        if monotonic() >= deadline:
            joined = ", ".join(stale_streams)
            raise C0BatchCaptureError(
                f"no fresh C0 camera frame within {timeout_s:.3f}s: {joined}"
            )
        sleep(poll_interval_s)


def recorded_count_for_finalize(
    lifecycle: C0BatchCaptureSessionV1 | None, fallback: int
) -> int:
    """Use durable capture sidecars when recording raised before returning a count."""

    return lifecycle.captured_episode_count if lifecycle is not None else fallback


def reset_c0_episode_processors(*processors: Any) -> None:
    """Clear position, orientation, yaw, gripper, and safety state between episodes."""

    for processor in processors:
        processor.reset()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=C0_FORMAL_BATCH_STATUS)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--placement-reference", required=True)
    parser.add_argument("--num-episodes", type=int, default=C0_FORMAL_EPISODE_COUNT)
    parser.add_argument("--formal-episode-offset", type=int, default=0)
    parser.add_argument(
        "--excluded-smoke-dataset-root",
        type=Path,
        default=Path("datasets/c0_smoke_single_strip_20260920T231125"),
        help="Read-only diagnostic smoke dataset; excluded from the ten formal episodes.",
    )
    return parser


def parse_c0_batch_args(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    plan = build_c0_batch_capture_plan(
        batch_id=args.batch_id,
        session_id=args.session_id,
        dataset_root=args.dataset_root,
        evidence_root=args.evidence_root,
        placement_reference=args.placement_reference,
        excluded_smoke_dataset_root=args.excluded_smoke_dataset_root,
        episode_count=args.num_episodes,
        formal_episode_offset=args.formal_episode_offset,
    )
    require_new_c0_batch_paths(plan)
    return args, plan


def _confirm_or_abort(checklist: tuple[str, ...], expected: str) -> None:
    print("\n" + "=" * 72)
    for line in checklist:
        print(line)
    print("=" * 72)
    typed = input(f"Type {expected} to start: ")
    require_formal_operator_confirmation(typed, expected)


def _load_record_helpers():
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import record as act_record

    act_record.RECORD_START_MODE = "normal"
    return act_record


def main(argv: list[str] | None = None) -> int:
    args, plan = parse_c0_batch_args(argv)
    camera_mapping = frozen_c0_smoke_camera_mapping()
    if tuple(camera_mapping) != C0_SMOKE_IMAGE_KEYS:
        raise C0BatchCaptureError("camera mapping keys must be global_rgb, grasp_rgb")
    if {key: mapping["device"] for key, mapping in camera_mapping.items()} != dict(
        C0_SMOKE_CAMERA_DEVICES
    ):
        raise C0BatchCaptureError(
            "camera devices do not match the frozen CameraSetV2 mapping"
        )
    _confirm_or_abort(
        build_c0_batch_operator_checklist(plan), operator_confirmation_for_plan(plan)
    )

    # Hardware-capable imports are intentionally delayed until after all path,
    # plan, camera-freeze, and explicit operator-confirmation gates pass.
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
    from lerobot.scripts.lerobot_record import (
        finalize_recorded_dataset,
        record_episode_sessions,
    )
    from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
    from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
    from lerobot.teleoperators.phone.teleop_phone import Phone
    from lerobot.utils.control_utils import init_keyboard_listener
    from lerobot.utils.utils import init_logging

    act_record = _load_record_helpers()

    class C0BatchAuboI10Robot(AuboI10Robot):
        def __init__(self, *robot_args: Any, **robot_kwargs: Any) -> None:
            super().__init__(*robot_args, **robot_kwargs)
            self._previous_c0_camera_timestamps: dict[str, float] = {}

        @property
        def observation_features(self) -> dict[str, type | tuple]:
            return c0_smoke_robot_observation_features()

        def get_observation(self) -> dict[str, Any]:
            return read_fresh_c0_observation(
                read_observation=super().get_observation,
                read_timestamps=lambda: self.last_c0_sensor_timestamps,
                previous_camera_timestamps=self._previous_c0_camera_timestamps,
            )

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    init_logging(
        log_file=log_dir / "record_c0_batch.log",
        console_level="INFO",
        file_level="DEBUG",
    )
    logging.info(C0_FORMAL_BATCH_STATUS)
    logging.info("batch plan: %s", plan.to_manifest_record())

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
    robot = C0BatchAuboI10Robot(robot_config)
    phone = Phone(teleop_config)

    teleop_processor = RobotProcessorPipeline[
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
    robot_action_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    dataset = None
    listener = None
    lifecycle = None
    recorded_episodes = 0
    Path(plan.evidence_root).mkdir(parents=True, exist_ok=False)
    try:
        robot.connect()
        phone.connect()
        dataset = LeRobotDataset.create(
            repo_id=Path(plan.dataset_root).name,
            root=Path(plan.dataset_root),
            fps=C0_SMOKE_DATASET_FPS,
            features=combine_feature_dicts(
                aggregate_pipeline_dataset_features(
                    pipeline=teleop_processor,
                    initial_features=create_initial_features(action=phone.action_features),
                    use_videos=True,
                ),
                aggregate_pipeline_dataset_features(
                    pipeline=observation_processor,
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
            raise C0BatchCaptureError("robot or phone teleoperator is not connected")

        def prepare_episode(episode: C0EpisodeCapturePlanV1) -> None:
            robot.disable_servo_mode()
            print("\n" + "=" * 72)
            print(f"Prepare episode {episode.episode_index + 1}/{plan.episode_count}")
            print(f"scene_id: {episode.scene_id}; split: {episode.split_name}")
            print(f"placement reference: {episode.placement_reference}")
            print("Press r to return to the frozen start pose (no Enter required).")
            print("After placing one strip on the named scale line, press right arrow to start.")
            print("During recording: right arrow finishes; left arrow discards and rerecords.")
            print("Press Esc to stop the whole session.")
            events["return_to_start"] = False
            start_pose_ready = False
            while not events["stop_recording"]:
                events["exit_early"] = False
                while not events["exit_early"] and not events["stop_recording"]:
                    if events.get("return_to_start"):
                        events["return_to_start"] = False
                        start_pose_ready = act_record.return_to_start(robot)
                        if start_pose_ready:
                            print(
                                "Start pose reached. Place the strip, then press right arrow to record."
                            )
                        else:
                            print("Return failed. Check the robot and press r to retry.")
                    time.sleep(0.05)
                if start_pose_ready or events["stop_recording"]:
                    break
                print("Right arrow ignored: press r and complete the return first.")
            events["exit_early"] = False
            if events["stop_recording"]:
                return
            # Returning with moveJoint changes the real robot pose.  The teleop
            # pipeline must forget the previous episode before servo commands
            # resume, otherwise its cached EE/J6 targets can pull toward the
            # previous episode's final pose on the first frame.
            reset_c0_episode_processors(teleop_processor, robot_action_processor)
            logging.info("C0 episode control processors reset before recording")
            print("Recording starts now. Press right arrow to finish; left arrow rerecords.")

        def human_outcome(episode: C0EpisodeCapturePlanV1) -> str:
            del episode
            return "uncertain"

        lifecycle = C0BatchCaptureSessionV1(
            plan=plan,
            prepare_episode=prepare_episode,
            human_outcome_provider=human_outcome,
        )
        recorded_episodes = record_episode_sessions(
            robot=robot,
            teleop=phone,
            policy=None,
            preprocessor=None,
            postprocessor=None,
            teleop_action_processor=teleop_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=observation_processor,
            dataset=dataset,
            events=events,
            fps=C0_SMOKE_DATASET_FPS,
            num_episodes=plan.episode_count,
            episode_time_s=EPISODE_TIME_SEC,
            reset_time_s=0,
            single_task=C0_FORMAL_TASK_TEXT,
            play_sounds=True,
            display_data=False,
            display_compressed_images=False,
            lifecycle=lifecycle,
        )
    finally:
        active_error = sys.exc_info()[1]
        try:
            if dataset is not None:
                finalize_recorded_dataset(
                    dataset,
                    lifecycle=lifecycle,
                    recorded_episode_count=recorded_count_for_finalize(
                        lifecycle, recorded_episodes
                    ),
                )
        except BaseException:
            if active_error is None:
                raise
            logging.error("dataset finalize failed while another error was active", exc_info=True)
        finally:
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    logging.error("keyboard listener stop failed", exc_info=True)
            try:
                robot.disable_servo_mode()
            except Exception:
                logging.error("disable servo during shutdown failed", exc_info=True)
            if phone.is_connected:
                phone.disconnect()
            if robot.is_connected:
                robot.disconnect()

    if lifecycle is None or not lifecycle.capture_session_complete:
        raise C0BatchCaptureError(
            "C0 capture shard is incomplete; final evidence is not claimed"
        )
    print(f"Captured {recorded_episodes} raw formal candidates.")
    print("Final manifest binding, VLM shadow audit, and training authorization remain pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

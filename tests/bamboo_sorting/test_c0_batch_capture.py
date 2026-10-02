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

"""Tests for timestamp journals and helpers still used by joint capture."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot.bamboo_sorting.c0_batch_capture import (
    C0BatchCaptureError,
    C0FormalFrameObserverV1,
    C0SensorTimestampJournalV1,
)
from lerobot.bamboo_sorting.c0_gripper_capture_journal import C0GripperPendingCaptureJournalV1
from lerobot.utils import control_utils

REPO_ROOT = Path(__file__).resolve().parents[2]

BATCH_SCRIPT = REPO_ROOT / "examples" / "phone_to_auboi10" / "record_c0_batch.py"

_BATCH_IMPORT_PROBE = r"""
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path

script = Path(%r)
spec = importlib.util.spec_from_file_location("record_c0_batch_under_test", script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

forbidden = [name for name in sys.modules if name.split(".")[0] in {"cv2", "pyaubo_sdk"}]
threads = [thread.name for thread in threading.enumerate() if thread is not threading.main_thread()]
sockets = []
for fd in os.listdir("/proc/self/fd"):
    path = f"/proc/self/fd/{fd}"
    if os.path.islink(path) and "socket" in os.readlink(path):
        sockets.append(fd)
print(json.dumps({"forbidden": forbidden, "threads": threads, "sockets": sockets}))
""" % str(BATCH_SCRIPT)


def _load_batch_script():
    spec = importlib.util.spec_from_file_location("record_c0_batch_under_test", BATCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hold_trace() -> dict[str, object]:
    return {
        "requested_transition": False,
        "target_suction_on": False,
        "requested_do": None,
        "do_writes": [],
        "do_write_attempt_count": 0,
        "do_api_success": None,
        "do_readback_supported": True,
        "do_readback": None,
        "do_readback_after_failure": None,
        "do_readback_error": None,
        "controller_output_state_known": False,
        "controller_output_matches_requested": None,
        "partial_write_possible": False,
        "commanded_state_before": False,
        "commanded_state_after": False,
        "error": None,
    }


class _Robot:
    def __init__(self, timestamps: dict[str, float]) -> None:
        self.last_c0_sensor_timestamps = timestamps
        self.last_gripper_command_trace = _hold_trace()


def _timestamps(value: float) -> dict[str, float]:
    return {
        "global_rgb": value,
        "grasp_rgb": value + 0.001,
        "robot_state": value + 0.002,
    }


def test_timestamp_journal_commits_exact_three_streams_per_saved_frame() -> None:
    journal = C0SensorTimestampJournalV1(episode_id="episode-00", episode_index=0)
    robot = _Robot(_timestamps(10.0))
    journal.before_send(robot=robot, candidate_frame_index=0)
    journal.on_send_success()
    journal.on_add_frame_success()

    robot.last_c0_sensor_timestamps = _timestamps(10.04)
    journal.before_send(robot=robot, candidate_frame_index=1)
    journal.on_send_success()
    journal.on_add_frame_success()

    record = journal.to_manifest_record()
    assert journal.frame_count == 2
    assert record["sensor_streams"] == ["global_rgb", "grasp_rgb", "robot_state"]
    assert record["frames"][0]["frame_index"] == 0
    assert set(record["frames"][0]["timestamps"]) == {
        "global_rgb",
        "grasp_rgb",
        "robot_state",
    }
    assert record["formal_final_evidence_bound"] is False
    assert record["training_authorized"] is False
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    "timestamps",
    [
        None,
        {"global_rgb": 1.0, "grasp_rgb": 1.0},
        {"global_rgb": 1.0, "grasp_rgb": 1.0, "robot_state": True},
        {"global_rgb": 1.0, "grasp_rgb": float("nan"), "robot_state": 1.0},
    ],
)
def test_timestamp_journal_rejects_missing_or_invalid_stream_evidence(
    timestamps: object,
) -> None:
    journal = C0SensorTimestampJournalV1(episode_id="episode-00", episode_index=0)
    robot = _Robot(_timestamps(1.0))
    robot.last_c0_sensor_timestamps = timestamps
    with pytest.raises(C0BatchCaptureError):
        journal.before_send(robot=robot, candidate_frame_index=0)


def test_timestamp_journal_rejects_duplicate_or_reversed_sensor_time() -> None:
    journal = C0SensorTimestampJournalV1(episode_id="episode-00", episode_index=0)
    robot = _Robot(_timestamps(10.0))
    journal.before_send(robot=robot, candidate_frame_index=0)
    journal.on_send_success()
    journal.on_add_frame_success()

    robot.last_c0_sensor_timestamps = _timestamps(10.0)
    with pytest.raises(C0BatchCaptureError, match="must increase"):
        journal.before_send(robot=robot, candidate_frame_index=1)


def test_timestamp_failure_never_commits_the_failed_frame() -> None:
    journal = C0SensorTimestampJournalV1(episode_id="episode-00", episode_index=0)
    robot = _Robot(_timestamps(10.0))
    journal.before_send(robot=robot, candidate_frame_index=0)
    journal.on_send_failure(error=RuntimeError("motion failed"))
    assert journal.frame_count == 0
    assert journal.status == "failed"
    assert "motion failed" in journal.to_manifest_record()["failure"]


def test_formal_observer_binds_timestamp_and_gripper_cycle() -> None:
    timestamp = C0SensorTimestampJournalV1(episode_id="episode-00", episode_index=0)
    gripper = C0GripperPendingCaptureJournalV1(
        episode_id="episode-00", episode_index=0, fps=25
    )
    observer = C0FormalFrameObserverV1(
        timestamp_journal=timestamp,
        gripper_journal=gripper,
    )
    robot = _Robot(_timestamps(10.0))
    kwargs = {
        "robot": robot,
        "dataset": object(),
        "action": {"ee.gripper_pos": 0.0},
        "robot_action_to_send": {"ee.gripper_pos": 0.0},
        "candidate_frame_index": 0,
    }
    observer.before_send(**kwargs)
    observer.on_send_success(sent_action={"ee.gripper_pos": 0.0}, **kwargs)
    observer.on_add_frame_success(**kwargs)

    record = observer.to_manifest_record()
    assert record["timestamp_journal"]["frame_count"] == 1
    assert len(record["gripper_journal"]["cycles"]) == 1
    assert record["dataset_episode_durably_saved"] is False
    assert record["policy_execution_authorized"] is False


def test_formal_observer_rejects_cross_episode_journals() -> None:
    timestamp = C0SensorTimestampJournalV1(episode_id="episode-00", episode_index=0)
    gripper = C0GripperPendingCaptureJournalV1(
        episode_id="episode-01", episode_index=0, fps=25
    )
    with pytest.raises(C0BatchCaptureError, match="episode_id"):
        C0FormalFrameObserverV1(
            timestamp_journal=timestamp,
            gripper_journal=gripper,
        )


def test_importing_formal_batch_entry_is_hardware_free() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _BATCH_IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    state = json.loads(result.stdout.strip().splitlines()[-1])
    assert state == {"forbidden": [], "threads": [], "sockets": []}


def test_formal_batch_waits_for_both_cameras_to_advance() -> None:
    script = _load_batch_script()
    timestamps = iter(
        [
            {"global_rgb": 1.0, "grasp_rgb": 2.0, "robot_state": 3.0},
            {"global_rgb": 1.04, "grasp_rgb": 2.04, "robot_state": 3.04},
        ]
    )
    observations = iter([{"attempt": 1}, {"attempt": 2}])
    current: dict[str, object] = {}

    def read_observation() -> dict[str, object]:
        current["timestamps"] = next(timestamps)
        return next(observations)

    previous = {"global_rgb": 1.0, "grasp_rgb": 2.0}
    result = script.read_fresh_c0_observation(
        read_observation=read_observation,
        read_timestamps=lambda: current["timestamps"],
        previous_camera_timestamps=previous,
        monotonic=iter([0.0, 0.01]).__next__,
        sleep=lambda _seconds: None,
    )

    assert result == {"attempt": 2}
    assert previous == {"global_rgb": 1.04, "grasp_rgb": 2.04}


def test_formal_batch_fails_closed_when_camera_does_not_advance() -> None:
    script = _load_batch_script()
    previous = {"global_rgb": 1.0, "grasp_rgb": 2.0}

    with pytest.raises(C0BatchCaptureError, match="global_rgb"):
        script.read_fresh_c0_observation(
            read_observation=lambda: {"attempt": 1},
            read_timestamps=lambda: {
                "global_rgb": 1.0,
                "grasp_rgb": 2.04,
                "robot_state": 3.0,
            },
            previous_camera_timestamps=previous,
            timeout_s=0.2,
            monotonic=iter([0.0, 0.2]).__next__,
            sleep=lambda _seconds: None,
        )

    assert previous == {"global_rgb": 1.0, "grasp_rgb": 2.0}


def test_keyboard_listener_can_reuse_session_events_without_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NonTTY:
        @staticmethod
        def isatty() -> bool:
            return False

    events = {
        "exit_early": True,
        "rerecord_episode": True,
        "stop_recording": True,
        "return_to_start": True,
    }
    monkeypatch.setattr(control_utils.sys, "stdin", _NonTTY())
    monkeypatch.setattr(control_utils, "is_headless", lambda: True)

    listener, bound_events = control_utils.init_keyboard_listener(events=events)
    try:
        assert bound_events is events
        assert events == {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "return_to_start": False,
        }
    finally:
        listener.stop()

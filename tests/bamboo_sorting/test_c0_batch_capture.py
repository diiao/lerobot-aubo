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

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from lerobot.bamboo_sorting.c0_batch_capture import (
    C0_BATCH_CAPTURE_PLAN_SCHEMA_VERSION,
    C0_FORMAL_EPISODE_COUNT,
    C0_FORMAL_OPERATOR_CONFIRMATION,
    C0_FORMAL_TASK_TEXT,
    C0_PILOT_VALIDATION_OPERATOR_CONFIRMATION,
    C0BatchCaptureError,
    C0BatchCapturePlanV1,
    C0BatchCaptureSessionV1,
    C0FormalFrameObserverV1,
    C0SensorTimestampJournalV1,
    build_c0_batch_capture_plan,
    build_c0_batch_operator_checklist,
    operator_confirmation_for_plan,
    require_formal_operator_confirmation,
    require_human_outcome,
    require_new_c0_batch_paths,
)
from lerobot.bamboo_sorting.c0_gripper_capture_journal import (
    C0GripperPendingCaptureJournalV1,
)
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


def _plan(tmp_path: Path, *, smoke: Path | None = None) -> C0BatchCapturePlanV1:
    return build_c0_batch_capture_plan(
        batch_id="c0-batch-20260921",
        session_id="c0-session-20260921",
        dataset_root=tmp_path / "formal-dataset",
        evidence_root=tmp_path / "formal-evidence",
        placement_reference="90-degree-scale-line",
        excluded_smoke_dataset_root=smoke,
    )


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


class _Dataset:
    def __init__(self) -> None:
        self.episode_buffer: dict[str, int] | None = None
        self.save_calls = 0

    def save_episode(self) -> None:
        self.save_calls += 1
        self.episode_buffer = None


def _timestamps(value: float) -> dict[str, float]:
    return {
        "global_rgb": value,
        "grasp_rgb": value + 0.001,
        "robot_state": value + 0.002,
    }


def test_plan_has_exactly_ten_new_unique_episodes_and_fixed_split(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    plan = _plan(tmp_path, smoke=smoke)
    record = plan.to_manifest_record()

    assert plan.schema_version == C0_BATCH_CAPTURE_PLAN_SCHEMA_VERSION
    assert plan.episode_count == C0_FORMAL_EPISODE_COUNT == 10
    assert [item.episode_index for item in plan.episodes] == list(range(10))
    assert len({item.episode_id for item in plan.episodes}) == 10
    assert len({item.scene_id for item in plan.episodes}) == 10
    assert [item.split_name for item in plan.episodes] == ["train"] * 8 + [
        "validation"
    ] * 2
    assert all(item.placement_reference == "90-degree-scale-line" for item in plan.episodes)
    assert all(item.to_manifest_record()["instruction_text"] == C0_FORMAL_TASK_TEXT for item in plan.episodes)
    assert record["excluded_smoke_dataset_root"] == str(smoke.resolve())
    assert "predates required per-sensor timestamp" in record["excluded_smoke_reason"]
    assert record["capture_started"] is False
    assert record["formal_final_evidence_bound"] is False
    assert record["training_authorized"] is False
    assert record["policy_execution_authorized"] is False
    assert record["hardware_access_performed_by_serialization"] is False
    json.dumps(record, allow_nan=False)


def test_plan_rejects_wrong_count_indexes_duplicates_split_or_placement(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    with pytest.raises(C0BatchCaptureError, match="1..10"):
        replace(plan, episodes=())

    episodes = list(plan.episodes)
    episodes[1] = replace(episodes[1], episode_index=0)
    with pytest.raises(C0BatchCaptureError, match="contiguous"):
        replace(plan, episodes=tuple(episodes))

    episodes = list(plan.episodes)
    episodes[1] = replace(episodes[1], episode_id=episodes[0].episode_id)
    with pytest.raises(C0BatchCaptureError, match="episode_id must be unique"):
        replace(plan, episodes=tuple(episodes))

    episodes = list(plan.episodes)
    episodes[8] = replace(episodes[8], split_name="train")
    with pytest.raises(C0BatchCaptureError, match="0..7 train"):
        replace(plan, episodes=tuple(episodes))

    episodes = list(plan.episodes)
    episodes[9] = replace(episodes[9], placement_reference="another-line")
    with pytest.raises(C0BatchCaptureError, match="must stay fixed"):
        replace(plan, episodes=tuple(episodes))


def test_new_paths_are_checked_without_creation(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    plan = _plan(tmp_path, smoke=smoke)
    require_new_c0_batch_paths(plan)
    assert not Path(plan.dataset_root).exists()
    assert not Path(plan.evidence_root).exists()

    Path(plan.dataset_root).mkdir()
    with pytest.raises(C0BatchCaptureError, match="dataset_root already exists"):
        require_new_c0_batch_paths(plan)


def test_checklist_and_operator_confirmation_are_explicit(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    text = "\n".join(build_c0_batch_operator_checklist(plan))
    assert "Capture shard episodes: exactly 10" in text
    assert "previous one-episode smoke" in text
    assert "new physical scene" in text
    assert C0_FORMAL_OPERATOR_CONFIRMATION in text
    with pytest.raises(C0BatchCaptureError, match="operator confirmation"):
        require_formal_operator_confirmation("YES")
    require_formal_operator_confirmation(C0_FORMAL_OPERATOR_CONFIRMATION)


def test_partial_capture_plan_maps_run02_to_remaining_formal_indexes(tmp_path: Path) -> None:
    plan = build_c0_batch_capture_plan(
        batch_id="c0-formal-20260921-run02",
        session_id="c0-formal-20260921-run02",
        dataset_root=tmp_path / "run02",
        evidence_root=tmp_path / "run02-evidence",
        placement_reference="90度",
        episode_count=9,
        formal_episode_offset=1,
    )

    assert plan.episode_count == 9
    assert [item.episode_index for item in plan.episodes] == list(range(9))
    assert [item.formal_episode_index for item in plan.episodes] == list(range(1, 10))
    assert [item.split_name for item in plan.episodes] == ["train"] * 7 + [
        "validation"
    ] * 2


def test_pilot_validation_extension_is_exactly_indexes_ten_and_eleven(tmp_path: Path) -> None:
    plan = build_c0_batch_capture_plan(
        batch_id="c0-pilot-validation-20260921-run05",
        session_id="c0-pilot-validation-20260921-run05",
        dataset_root=tmp_path / "run05",
        evidence_root=tmp_path / "run05-evidence",
        placement_reference="90度",
        episode_count=2,
        formal_episode_offset=10,
    )

    assert [item.episode_index for item in plan.episodes] == [0, 1]
    assert [item.formal_episode_index for item in plan.episodes] == [10, 11]
    assert [item.split_name for item in plan.episodes] == ["validation", "validation"]
    assert operator_confirmation_for_plan(plan) == C0_PILOT_VALIDATION_OPERATOR_CONFIRMATION
    assert C0_PILOT_VALIDATION_OPERATOR_CONFIRMATION in "\n".join(
        build_c0_batch_operator_checklist(plan)
    )
    assert plan.to_manifest_record()["status"].startswith("C0 pilot validation extension")


@pytest.mark.parametrize(
    ("offset", "count"),
    [(9, 2), (10, 1), (10, 3), (11, 1)],
)
def test_pilot_validation_extension_rejects_any_other_slice(
    tmp_path: Path, offset: int, count: int
) -> None:
    with pytest.raises(C0BatchCaptureError, match="exactly two episodes"):
        build_c0_batch_capture_plan(
            batch_id="invalid-extension",
            session_id="invalid-extension",
            dataset_root=tmp_path / "dataset",
            evidence_root=tmp_path / "evidence",
            placement_reference="90度",
            episode_count=count,
            formal_episode_offset=offset,
        )


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


def _record_one_session_frame(
    *,
    session: C0BatchCaptureSessionV1,
    dataset: _Dataset,
    robot: _Robot,
    episode_index: int,
) -> None:
    observer = session.episode_frame_observer()
    robot.last_c0_sensor_timestamps = _timestamps(10.0 + episode_index)
    kwargs = {
        "robot": robot,
        "dataset": dataset,
        "action": {"ee.gripper_pos": 0.0},
        "robot_action_to_send": {"ee.gripper_pos": 0.0},
        "candidate_frame_index": 0,
    }
    observer.before_send(**kwargs)
    observer.on_send_success(sent_action={"ee.gripper_pos": 0.0}, **kwargs)
    observer.on_add_frame_success(**kwargs)
    dataset.episode_buffer = {"size": 1}
    session.save_episode(robot=robot, dataset=dataset, episode_index=episode_index)


def test_batch_capture_session_persists_ten_raw_sidecars_without_final_claim(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    Path(plan.evidence_root).mkdir()
    prepared: list[int] = []
    session = C0BatchCaptureSessionV1(
        plan=plan,
        prepare_episode=lambda item: prepared.append(item.episode_index),
        human_outcome_provider=lambda _item: "success",
    )
    dataset = _Dataset()
    robot = _Robot(_timestamps(10.0))

    for episode_index in range(10):
        _record_one_session_frame(
            session=session,
            dataset=dataset,
            robot=robot,
            episode_index=episode_index,
        )
    session.on_dataset_finalize_success(dataset=dataset, recorded_episode_count=10)

    assert prepared == list(range(10))
    assert dataset.save_calls == 10
    assert session.captured_episode_count == 10
    assert session.batch_capture_complete is True
    assert len(list(Path(plan.evidence_root).glob("episode-*-capture.json"))) == 10
    first_sidecar = json.loads(
        (Path(plan.evidence_root) / "episode-00-capture.json").read_text()
    )
    assert first_sidecar["status"] == "saved"
    assert first_sidecar["human_outcome"] == "success"
    assert first_sidecar["dataset_episode_durably_saved"] is True
    record = json.loads((Path(plan.evidence_root) / "capture-session.json").read_text())
    assert record["batch_capture_complete"] is True
    assert record["formal_final_evidence_bound"] is False
    assert "final_episode_manifest_binding_pending" in record["blockers"]
    assert record["training_authorized"] is False
    assert record["policy_execution_authorized"] is False


def test_partial_capture_session_completes_without_claiming_full_batch(
    tmp_path: Path,
) -> None:
    plan = build_c0_batch_capture_plan(
        batch_id="c0-formal-20260921-run02",
        session_id="c0-formal-20260921-run02",
        dataset_root=tmp_path / "run02",
        evidence_root=tmp_path / "run02-evidence",
        placement_reference="90度",
        episode_count=9,
        formal_episode_offset=1,
    )
    Path(plan.evidence_root).mkdir()
    session = C0BatchCaptureSessionV1(
        plan=plan,
        prepare_episode=lambda _item: None,
        human_outcome_provider=lambda _item: "uncertain",
    )
    dataset = _Dataset()
    robot = _Robot(_timestamps(10.0))

    for episode_index in range(9):
        _record_one_session_frame(
            session=session,
            dataset=dataset,
            robot=robot,
            episode_index=episode_index,
        )
    session.on_dataset_finalize_success(dataset=dataset, recorded_episode_count=9)

    assert session.capture_session_complete is True
    assert session.batch_capture_complete is False
    record = session.to_manifest_record()
    assert record["capture_shard_episode_count"] == 9
    assert "ten_episode_capture_incomplete" in record["blockers"]


def test_batch_capture_session_rerecord_isolated_from_next_attempt(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    Path(plan.evidence_root).mkdir()
    session = C0BatchCaptureSessionV1(
        plan=plan,
        prepare_episode=lambda _item: None,
        human_outcome_provider=lambda _item: "failure",
    )
    dataset = _Dataset()
    robot = _Robot(_timestamps(1.0))

    first = session.episode_frame_observer()
    session.on_episode_rerecord(
        robot=robot,
        dataset=dataset,
        episode_index=0,
    )
    second = session.episode_frame_observer()

    assert first is not second
    assert first.timestamp_journal.status == "abandoned_rerecorded"
    assert second.timestamp_journal.frame_count == 0
    assert dataset.save_calls == 0
    assert list(Path(plan.evidence_root).glob("episode-*-capture.json")) == []
    record = session.to_manifest_record()
    assert record["attempts"][0]["status"] == "abandoned_rerecorded"
    assert record["attempts"][1]["status"] == "recording"


def test_batch_capture_session_rejects_sidecar_overwrite(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    evidence_root = Path(plan.evidence_root)
    evidence_root.mkdir()
    (evidence_root / "episode-00-capture.json").write_text("preserve", encoding="utf-8")
    session = C0BatchCaptureSessionV1(
        plan=plan,
        prepare_episode=lambda _item: None,
        human_outcome_provider=lambda _item: "uncertain",
    )
    dataset = _Dataset()
    robot = _Robot(_timestamps(1.0))

    with pytest.raises(C0BatchCaptureError, match="refuse overwrite"):
        _record_one_session_frame(
            session=session,
            dataset=dataset,
            robot=robot,
            episode_index=0,
        )
    assert (evidence_root / "episode-00-capture.json").read_text() == "preserve"


@pytest.mark.parametrize("value", ["success", "failure", "uncertain"])
def test_human_outcome_keeps_frozen_vocabulary(value: str) -> None:
    assert require_human_outcome(value) == value


def test_human_outcome_is_not_auto_converted() -> None:
    for value in ("passed", "yes", " uncertain ", None):
        with pytest.raises(C0BatchCaptureError, match="human outcome"):
            require_human_outcome(value)


def test_plan_record_is_deep_copy(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    record = plan.to_manifest_record()
    original = copy.deepcopy(record)
    record["episodes"][0]["episode_id"] = "mutated"
    assert plan.to_manifest_record() == original


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


def test_formal_batch_args_validate_paths_without_creating_them(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    dataset = tmp_path / "formal-dataset"
    evidence = tmp_path / "formal-evidence"
    script = _load_batch_script()
    _args, plan = script.parse_c0_batch_args(
        [
            "--batch-id",
            "c0-batch-20260921",
            "--session-id",
            "c0-session-20260921",
            "--dataset-root",
            str(dataset),
            "--evidence-root",
            str(evidence),
            "--placement-reference",
            "90-degree-scale-line",
            "--excluded-smoke-dataset-root",
            str(smoke),
        ]
    )
    assert plan.episode_count == 10
    assert not dataset.exists()
    assert not evidence.exists()


def test_formal_batch_args_support_remaining_nine_episode_shard(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    script = _load_batch_script()
    _args, plan = script.parse_c0_batch_args(
        [
            "--batch-id",
            "c0-formal-20260921-run02",
            "--session-id",
            "c0-formal-20260921-run02",
            "--dataset-root",
            str(tmp_path / "run02"),
            "--evidence-root",
            str(tmp_path / "run02-evidence"),
            "--placement-reference",
            "90度",
            "--num-episodes",
            "9",
            "--formal-episode-offset",
            "1",
            "--excluded-smoke-dataset-root",
            str(smoke),
        ]
    )

    assert plan.episode_count == 9
    assert plan.episodes[0].formal_episode_index == 1
    assert plan.episodes[-1].formal_episode_index == 9


def test_formal_batch_entry_has_sleep_dependency_for_act_style_wait_loop() -> None:
    script = _load_batch_script()
    assert script.time.sleep is time.sleep


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


def test_finalize_count_uses_already_saved_capture_sidecars(tmp_path: Path) -> None:
    script = _load_batch_script()
    plan = _plan(tmp_path)
    Path(plan.evidence_root).mkdir()
    session = C0BatchCaptureSessionV1(
        plan=plan,
        prepare_episode=lambda _item: None,
        human_outcome_provider=lambda _item: "uncertain",
    )
    session._attempts.extend(
        [
            type("Attempt", (), {"status": "saved"})(),
            type("Attempt", (), {"status": "incomplete"})(),
        ]
    )

    assert script.recorded_count_for_finalize(session, fallback=0) == 1
    assert script.recorded_count_for_finalize(None, fallback=3) == 3


def test_formal_batch_resets_every_episode_control_processor() -> None:
    script = _load_batch_script()

    class _Processor:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    teleop = _Processor()
    robot_action = _Processor()
    script.reset_c0_episode_processors(teleop, robot_action)

    assert teleop.reset_calls == 1
    assert robot_action.reset_calls == 1


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

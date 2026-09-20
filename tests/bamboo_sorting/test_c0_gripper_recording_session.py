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

"""Offline integration tests for the C0-R4C recording session coordinator.

The full generic wiring is exercised end to end: ``record_episode_sessions``
(``lerobot_record``) drives a real ``C0GripperRecordingSession`` as the
lifecycle observer, with in-memory robot/teleoperator fakes and a duck-typed
dataset whose gripper episodes exist as synthetic parquet snapshots under
``tmp_path`` (the frozen R2 auditor re-reads them during the R4B complete
phase). No hardware, camera, DO/IO, motion, real dataset recording, network,
training, or inference is involved, and nothing writes a sidecar/JSON to disk.
"""

import copy
import dataclasses
import json
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.bamboo_sorting.c0_capture_contract import (
    C0_CAPTURE_PROFILE_ID,
    C0_DATASET_FPS,
    C0_EPISODE_MANIFEST_SCHEMA_VERSION,
    C0EpisodeManifestV1,
)
from lerobot.bamboo_sorting.c0_gripper_capture_journal import (
    C0GripperPendingCaptureJournalV1,
)
from lerobot.bamboo_sorting.c0_gripper_recording_session import (
    C0_GRIPPER_EPISODE_LIFECYCLE_RECORD_SCHEMA_VERSION,
    C0GripperEpisodeLifecycleRecordV1,
    C0GripperRecordingSession,
    C0GripperRecordingSessionError,
)
from lerobot.bamboo_sorting.c0_gripper_save_finalizer import (
    C0GripperEpisodeSavePendingAuditV1,
    C0GripperSaveFinalizationError,
    complete_c0_gripper_episode_finalization,
    require_c0_gripper_finalized_evidence_bound,
)
from lerobot.bamboo_sorting.contracts import (
    ACTION_FIELD_NAMES,
    ACTION_SCHEMA_VERSION,
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_SCHEMA_VERSION,
    INSTRUCTION_SPECS,
)
from lerobot.bamboo_sorting.lerobot_bridge import (
    CAMERASET_V2_LEROBOT_BRIDGE_VERSION,
    build_camera_set_v2_lerobot_features,
)
from lerobot.bamboo_sorting.observation_contract import OBSERVATION_STATE_FIELD_NAMES
from lerobot.bamboo_sorting.rgb_gate import (
    CAMERA_SET_V2_SCHEMA_VERSION,
    FROZEN_CAMERA_SET_V2_SHA256,
)
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
    make_default_teleop_action_processor,
)
from lerobot.scripts.lerobot_record import (
    finalize_recorded_dataset,
    record_episode_sessions,
)
from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig

FPS = float(C0_DATASET_FPS)
FRAME_COUNT = 6
ON_FRAME = 2
OFF_FRAME = 4
EVIDENCE_SHA = "34" * 32

ACTION_GRIPPER_DIM = ACTION_FIELD_NAMES.index("ee.gripper_pos")
STATE_GRIPPER_DIM = OBSERVATION_STATE_FIELD_NAMES.index("gripper_pos")

AUTHORIZATION_FIELDS = {
    "training_authorized",
    "policy_execution_authorized",
    "serialized_record_grants_live_authorization",
    "hardware_access_performed_by_serialization",
    "hardware_access_performed_by_audit",
    "physical_gripper_feedback_available",
    "physical_grasp_success_proven",
    "final_sidecar_persisted_to_disk",
}


def _episode_manifest(episode_id: str, *, frame_count: int = FRAME_COUNT) -> C0EpisodeManifestV1:
    instruction = INSTRUCTION_SPECS["pick_any_collection"]
    return C0EpisodeManifestV1(
        schema_version=C0_EPISODE_MANIFEST_SCHEMA_VERSION,
        episode_id=episode_id,
        scene_id="c0-scene-00",
        session_id="c0-session-r4c",
        split_name="train",
        camera_set_schema_version=CAMERA_SET_V2_SCHEMA_VERSION,
        camera_set_sha256=FROZEN_CAMERA_SET_V2_SHA256,
        bridge_version=CAMERASET_V2_LEROBOT_BRIDGE_VERSION,
        action_schema_version=ACTION_SCHEMA_VERSION,
        instruction_schema_version=INSTRUCTION_SCHEMA_VERSION,
        instruction_id=instruction.instruction_id,
        instruction_text=instruction.text,
        instruction_language=INSTRUCTION_LANGUAGE,
        instruction_text_sha256=instruction.text_sha256,
        calibration_version="calibration-c0-v1",
        tool_version="tool-c0-v1",
        capture_profile_id=C0_CAPTURE_PROFILE_ID,
        fps=C0_DATASET_FPS,
        frame_count=frame_count,
        dataset_episode_ref=f"datasets/c0/{episode_id}",
        dataset_episode_sha256=EVIDENCE_SHA,
        timestamp_evidence_ref=f"evidence/{episode_id}-timestamps.json",
        timestamp_evidence_sha256=EVIDENCE_SHA,
        authorization_evidence_ref=f"evidence/{episode_id}-authorization.json",
        authorization_evidence_sha256=EVIDENCE_SHA,
        vlm_shadow_evidence_ref=f"evidence/{episode_id}-vlm-shadow.json",
        vlm_shadow_evidence_sha256=EVIDENCE_SHA,
        human_outcome="success",
        human_outcome_evidence_ref=f"evidence/{episode_id}-human-outcome.json",
        human_outcome_evidence_sha256=EVIDENCE_SHA,
        raw_data_immutable=True,
        finalized=True,
    )


def _session_manifests(count: int) -> list[C0EpisodeManifestV1]:
    return [
        _episode_manifest(f"c0-r4c-episode-{index:02d}") for index in range(count)
    ]


def _gripper_action(
    frame_count: int, on_frame: int | None, off_frame: int | None
) -> list[float]:
    values = [0.0] * frame_count
    if on_frame is not None:
        end = off_frame if off_frame is not None else frame_count
        for index in range(on_frame, end):
            values[index] = 100.0
    return values


def _episode_table(
    *,
    episode_index: int,
    frame_count: int = FRAME_COUNT,
    timestamp_offset: float = 0.0,
) -> pa.Table:
    action_values = _gripper_action(frame_count, ON_FRAME, OFF_FRAME)
    observation = [0.0, *action_values[:-1]]
    action = np.full((frame_count, len(ACTION_FIELD_NAMES)), 0.01, dtype=np.float64)
    state = np.full((frame_count, len(OBSERVATION_STATE_FIELD_NAMES)), 0.02, dtype=np.float64)
    action[:, ACTION_GRIPPER_DIM] = action_values
    state[:, STATE_GRIPPER_DIM] = observation
    return pa.table(
        {
            "action": pa.array(action.tolist(), type=pa.list_(pa.float64())),
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float64())),
            "episode_index": pa.array([episode_index] * frame_count, type=pa.int64()),
            "frame_index": pa.array(list(range(frame_count)), type=pa.int64()),
            "timestamp": pa.array(
                [frame / FPS + timestamp_offset for frame in range(frame_count)],
                type=pa.float64(),
            ),
            "index": pa.array(list(range(frame_count)), type=pa.int64()),
        }
    )


def _write_dataset(
    root: Path,
    *,
    episode_count: int,
    frame_count: int = FRAME_COUNT,
    timestamp_offset: float = 0.0,
) -> None:
    """Write closed synthetic saved-episode snapshots for episodes 0..N-1."""

    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "fps": FPS,
        "features": json.loads(json.dumps(build_camera_set_v2_lerobot_features())),
        "total_episodes": episode_count,
        "total_frames": frame_count * episode_count,
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    for episode_index in range(episode_count):
        table = _episode_table(
            episode_index=episode_index,
            frame_count=frame_count,
            timestamp_offset=timestamp_offset,
        )
        pq.write_table(table, data_dir / f"episode_{episode_index:06d}.parquet")


def _success_trace(target_on: bool) -> dict:
    if target_on:
        return {
            "requested_transition": True,
            "target_suction_on": True,
            "requested_do": {"2": True, "3": False},
            "do_writes": [
                {"pin": 3, "value": False, "return_code": 0},
                {"pin": 2, "value": True, "return_code": 0},
            ],
            "do_write_attempt_count": 2,
            "do_api_success": True,
            "do_readback_supported": True,
            "do_readback": {"2": True, "3": False},
            "do_readback_after_failure": None,
            "do_readback_error": None,
            "controller_output_state_known": True,
            "controller_output_matches_requested": True,
            "partial_write_possible": False,
            "commanded_state_before": False,
            "commanded_state_after": True,
            "error": None,
            "do_readback_attempts": 1,
        }
    return {
        "requested_transition": True,
        "target_suction_on": False,
        "requested_do": {"2": False, "3": True},
        "do_writes": [
            {"pin": 2, "value": False, "return_code": 0},
            {"pin": 3, "value": True, "return_code": 0},
        ],
        "do_write_attempt_count": 2,
        "do_api_success": True,
        "do_readback_supported": True,
        "do_readback": {"2": False, "3": True},
        "do_readback_after_failure": None,
        "do_readback_error": None,
        "controller_output_state_known": True,
        "controller_output_matches_requested": True,
        "partial_write_possible": False,
        "commanded_state_before": True,
        "commanded_state_after": False,
        "error": None,
        "do_readback_attempts": 1,
    }


def _hold_trace(on: bool) -> dict:
    return {
        "requested_transition": False,
        "target_suction_on": on,
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
        "commanded_state_before": on,
        "commanded_state_after": on,
        "error": None,
    }


def _episode_traces(*, frame_count: int = FRAME_COUNT) -> list[dict]:
    """One trace per frame matching the saved dataset gripper trajectory."""

    values = _gripper_action(frame_count, ON_FRAME, OFF_FRAME)
    traces = []
    for frame, gripper in enumerate(values):
        if frame == ON_FRAME:
            traces.append(_success_trace(True))
        elif frame == OFF_FRAME:
            traces.append(_success_trace(False))
        else:
            traces.append(_hold_trace(gripper == 100.0))
    return traces


def _gripper_teleop_action(value: float) -> dict:
    return {"ee.gripper_pos": value}


def _episode_actions(*, frame_count: int = FRAME_COUNT) -> list[dict]:
    return [_gripper_teleop_action(v) for v in _gripper_action(frame_count, ON_FRAME, OFF_FRAME)]


class FakeRobot:
    """Duck-typed robot with scripted per-cycle gripper traces and event triggers."""

    name = "fake_robot"

    def __init__(
        self,
        *,
        traces=(),
        exit_afters=(),
        rerecord_afters=(),
        stop_afters=(),
        send_error: Exception | None = None,
        error_after: int | None = None,
    ):
        self.traces = list(traces)
        self.exit_afters = set(exit_afters)
        self.rerecord_afters = set(rerecord_afters)
        self.stop_afters = set(stop_afters)
        self.send_error = send_error
        self.error_after = error_after
        self.sent_actions: list[dict] = []
        self.last_gripper_command_trace: dict | None = None
        self.events: dict | None = None

    def get_observation(self) -> dict:
        return {"a": 0.0, "b": 0.0}

    def send_action(self, action: dict) -> dict:
        self.sent_actions.append(action)
        index = len(self.sent_actions)
        self.last_gripper_command_trace = self.traces[index - 1] if index <= len(self.traces) else None
        if self.send_error is not None and self.error_after is not None and index >= self.error_after:
            raise self.send_error
        events = self.events
        if events is not None:
            if index in self.exit_afters:
                events["exit_early"] = True
            if index in self.rerecord_afters:
                events["rerecord_episode"] = True
                events["exit_early"] = True
            if index in self.stop_afters:
                events["stop_recording"] = True
                events["exit_early"] = True
        return action


class FakeTeleop(Teleoperator):
    config_class = TeleoperatorConfig
    name = "fake_teleop"

    def __init__(self, actions: list[dict]):
        # Deliberately skip Teleoperator.__init__ (no calibration dirs on disk).
        self._actions = iter(actions)

    @property
    def action_features(self) -> dict:
        return {"ee.gripper_pos": float}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return False

    def connect(self, calibrate: bool = True) -> None:
        raise NotImplementedError

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        raise NotImplementedError

    def configure(self) -> None:
        raise NotImplementedError

    def get_action(self) -> dict:
        return next(self._actions)

    def send_feedback(self, feedback: dict) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        pass


class FakeDataset:
    """Duck-typed dataset mimicking the real save/finalize lifecycle.

    ``episode_buffer`` is open while an episode is being recorded (size and
    episode_index of the pending episode), cleared on save/clear like
    ``LeRobotDataset``; ``meta.total_episodes`` mirrors the saved count.
    """

    def __init__(
        self,
        root: Path,
        *,
        fps: float = FPS,
        save_error: Exception | None = None,
        finalize_error: Exception | None = None,
        add_frame_error: Exception | None = None,
    ):
        self.root = root
        self.fps = fps
        self.features = {
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "action.ee.gripper_pos": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["ee.gripper_pos"],
            },
        }
        self.episode_buffer: dict | None = None
        self.meta = types.SimpleNamespace(total_episodes=0)
        self.save_calls = 0
        self.clear_calls = 0
        self.finalize_calls = 0
        self.save_error = save_error
        self.finalize_error = finalize_error
        self.add_frame_error = add_frame_error
        self.frames: list[dict] = []

    @property
    def num_episodes(self) -> int:
        return self.meta.total_episodes

    def add_frame(self, frame: dict) -> None:
        if self.episode_buffer is None:
            self.episode_buffer = {"size": 0, "episode_index": self.meta.total_episodes}
        if self.add_frame_error is not None:
            raise self.add_frame_error
        self.episode_buffer["size"] += 1
        self.frames.append(frame)

    def clear_episode_buffer(self) -> None:
        self.clear_calls += 1
        self.episode_buffer = None

    def save_episode(self) -> None:
        self.save_calls += 1
        if self.save_error is not None:
            raise self.save_error
        self.episode_buffer = None
        self.meta.total_episodes += 1

    def finalize(self) -> None:
        self.finalize_calls += 1
        if self.finalize_error is not None:
            raise self.finalize_error


def _make_session(manifest_count: int) -> C0GripperRecordingSession:
    return C0GripperRecordingSession(episode_manifests=_session_manifests(manifest_count), fps=FPS)


def _run_session(
    *,
    session: C0GripperRecordingSession,
    dataset: FakeDataset,
    robot: FakeRobot,
    actions: list[dict],
    num_episodes: int,
    reset_time_s: float = 0.0,
    observer_sink: list | None = None,
):
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    robot.events = events
    if observer_sink is not None:
        original = session.episode_frame_observer

        def spy_frame_observer():
            journal = original()
            observer_sink.append(journal)
            return journal

        session.episode_frame_observer = spy_frame_observer  # type: ignore[assignment]
    return record_episode_sessions(
        robot=robot,
        teleop=FakeTeleop(actions),
        policy=None,
        preprocessor=None,
        postprocessor=None,
        teleop_action_processor=make_default_teleop_action_processor(),
        robot_action_processor=make_default_robot_action_processor(),
        robot_observation_processor=make_default_robot_observation_processor(),
        dataset=dataset,
        events=events,
        fps=FPS,
        num_episodes=num_episodes,
        episode_time_s=60,
        reset_time_s=reset_time_s,
        single_task="sort bamboo",
        play_sounds=False,
        display_data=False,
        display_compressed_images=False,
        lifecycle=session,
    )


# --------------------------------------------------------------- happy path


def test_two_episodes_full_lifecycle_saves_and_completes_each_once(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=2)
    session = _make_session(2)
    dataset = FakeDataset(tmp_path)
    observers: list[C0GripperPendingCaptureJournalV1] = []

    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(
            traces=[*_episode_traces(), *_episode_traces()], exit_afters={FRAME_COUNT, 2 * FRAME_COUNT}
        ),
        actions=[*_episode_actions(), *_episode_actions()],
        num_episodes=2,
        observer_sink=observers,
    )
    finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)

    assert recorded == 2
    # dataset.save_episode: exactly once per episode; finalize: exactly once;
    # complete: exactly once per episode (two promotion results).
    assert dataset.save_calls == 2
    assert dataset.finalize_calls == 1
    assert session.saved_episode_count == 2
    assert session.pending_audit_count == 0
    assert session.finalized_episode_count == 2
    assert session.batch_finalization_complete is True

    # Two independent journals, one per episode, from frame 0.
    first, second = observers
    assert first is not second
    assert first.episode_index == 0
    assert second.episode_index == 1
    assert first.episode_id == "c0-r4c-episode-00"
    assert second.episode_id == "c0-r4c-episode-01"
    assert [c["candidate_frame_index"] for c in first.cycles] == list(range(FRAME_COUNT))
    assert [c["candidate_frame_index"] for c in second.cycles] == list(range(FRAME_COUNT))

    records = session.episode_records
    assert [record.outcome for record in records] == ["finalized", "finalized"]
    assert [record.episode_index for record in records] == [0, 1]
    results = session.finalization_results
    assert [result.episode_index for result in results] == [0, 1]
    assert all(result.structurally_finalized is True for result in results)


def test_reset_loop_generates_no_journal_cycles(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=2)
    session = _make_session(2)
    dataset = FakeDataset(tmp_path)
    observers: list[C0GripperPendingCaptureJournalV1] = []

    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(
            # 6 episode-0 frames, 1 reset-loop send, 6 episode-1 frames.
            traces=[*_episode_traces(), _hold_trace(False), *_episode_traces()],
            exit_afters={FRAME_COUNT, FRAME_COUNT + 1, 2 * FRAME_COUNT + 1},
        ),
        actions=[*_episode_actions(), _gripper_teleop_action(0.0), *_episode_actions()],
        num_episodes=2,
        reset_time_s=1.0,
        observer_sink=observers,
    )
    finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)

    assert recorded == 2
    assert dataset.save_calls == 2
    for journal in observers:
        assert [c["candidate_frame_index"] for c in journal.cycles] == list(range(FRAME_COUNT))
        assert len(journal.controller_attempts) == 2


# ----------------------------------------------------------------- rerecord


def test_rerecord_abandons_old_journal_and_saves_only_new_attempt(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path)
    observers: list[C0GripperPendingCaptureJournalV1] = []

    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(
            traces=[*_episode_traces(), *_episode_traces()],
            rerecord_afters={FRAME_COUNT},
            exit_afters={FRAME_COUNT, 2 * FRAME_COUNT},
        ),
        actions=[*_episode_actions(), *_episode_actions()],
        num_episodes=1,
        observer_sink=observers,
    )
    finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)

    # Rerecord itself never counts as a recorded episode.
    assert recorded == 1
    assert dataset.save_calls == 1
    assert dataset.clear_calls == 1

    old, new = observers
    assert old is not new
    records = session.episode_records
    assert [record.outcome for record in records] == ["abandoned_rerecorded", "finalized"]
    assert records[0].journal.status == "abandoned_rerecorded"
    assert records[0].journal.to_manifest_record()["abandon_reason"] == "rerecord requested by operator"

    # No cycle/event/attempt leakage: the new journal starts at frame 0 with
    # exactly this attempt's two controller attempts.
    assert [c["candidate_frame_index"] for c in new.cycles] == list(range(FRAME_COUNT))
    assert len(new.controller_attempts) == 2
    assert new.cycles[0]["cycle_index"] == 0
    assert len(old.controller_attempts) == 2
    assert session.finalized_episode_count == 1
    assert session.batch_finalization_complete is True


# ------------------------------------------------- stop / failure before save


def test_stop_before_save_keeps_pending_token_and_skips_incomplete_episode(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=2)
    session = _make_session(2)
    dataset = FakeDataset(tmp_path)

    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(
            traces=[*_episode_traces(), *_episode_traces()],
            stop_afters={FRAME_COUNT + 1},
            exit_afters={FRAME_COUNT, FRAME_COUNT + 1},
        ),
        actions=[*_episode_actions(), *_episode_actions()],
        num_episodes=2,
    )
    finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)

    assert recorded == 1
    assert dataset.save_calls == 1
    records = session.episode_records
    assert [record.outcome for record in records] == ["finalized", "incomplete"]
    assert records[1].journal.status == "incomplete"
    # The saved episode's pending token survived the stop and was completed.
    assert session.finalized_episode_count == 1
    assert session.pending_audit_count == 0
    assert session.batch_finalization_complete is False


def test_send_action_failure_aborts_without_save_or_finalized_sidecar(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path)

    with pytest.raises(RuntimeError, match="gripper DO failed"):
        try:
            _run_session(
                session=session,
                dataset=dataset,
                robot=FakeRobot(
                    traces=_episode_traces(),
                    send_error=RuntimeError("gripper DO failed"),
                    error_after=ON_FRAME,
                ),
                actions=_episode_actions(),
                num_episodes=1,
            )
        finally:
            # Mirrors record()'s finally: the dataset still finalizes.
            finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=0)

    assert dataset.save_calls == 0
    assert dataset.finalize_calls == 1
    (record,) = session.episode_records
    assert record.outcome == "incomplete"
    assert "record_loop raised RuntimeError" in record.journal.to_manifest_record()["incomplete_reason"]
    # No finalized sidecar/result was fabricated for the failed episode.
    assert session.finalization_results == ()
    assert session.pending_audit_count == 0
    assert session.batch_finalization_complete is False


def test_add_frame_failure_aborts_without_save_or_finalized_sidecar(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path, add_frame_error=RuntimeError("parquet flush failed"))

    with pytest.raises(RuntimeError, match="parquet flush failed"):
        try:
            _run_session(
                session=session,
                dataset=dataset,
                robot=FakeRobot(traces=_episode_traces()),
                actions=_episode_actions(),
                num_episodes=1,
            )
        finally:
            finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=0)

    assert dataset.save_calls == 0
    assert dataset.finalize_calls == 1
    (record,) = session.episode_records
    assert record.outcome == "incomplete"
    assert session.finalization_results == ()
    assert session.pending_audit_count == 0


# --------------------------------------------------------------- save failure


def test_save_failure_enters_save_uncertain_without_retry_or_recorded_count(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path, save_error=RuntimeError("flush crashed"))

    with pytest.raises(RuntimeError, match="flush crashed"):
        try:
            _run_session(
                session=session,
                dataset=dataset,
                robot=FakeRobot(traces=_episode_traces(), exit_afters={FRAME_COUNT}),
                actions=_episode_actions(),
                num_episodes=1,
            )
        finally:
            finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=0)

    # Exactly one save attempt, no automatic retry, no recorded episode.
    assert dataset.save_calls == 1
    assert session.saved_episode_count == 0
    (record,) = session.episode_records
    assert record.outcome == "save_uncertain"
    assert record.pending_audit is None
    assert record.finalization_result is None
    # The journal keeps the R4B terminal save-uncertain state and is never
    # promoted, even though the dataset finalize itself succeeded.
    assert record.journal._c0_r4b_state == "save_uncertain"
    assert session.finalization_results == ()
    assert session.pending_audit_count == 0


def test_frame_count_mismatch_fails_closed_before_any_save(tmp_path) -> None:
    """Recorded frames (4) < manifest frame_count (6): R4B coverage rejects."""
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path)

    with pytest.raises(C0GripperSaveFinalizationError, match="cover exactly"):
        try:
            _run_session(
                session=session,
                dataset=dataset,
                robot=FakeRobot(traces=_episode_traces(), exit_afters={4}),
                actions=_episode_actions()[:4],
                num_episodes=1,
            )
        finally:
            finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=0)

    assert dataset.save_calls == 0
    assert session.saved_episode_count == 0
    (record,) = session.episode_records
    assert record.outcome == "pre_save_rejected"
    assert session.finalization_results == ()
    assert session.batch_finalization_complete is False


# ------------------------------------------------------------ finalize failure


def test_finalize_failure_keeps_all_tokens_pending_and_completes_none(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=2)
    session = _make_session(2)
    dataset = FakeDataset(tmp_path, finalize_error=RuntimeError("writer close failed"))

    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(
            traces=[*_episode_traces(), *_episode_traces()], exit_afters={FRAME_COUNT, 2 * FRAME_COUNT}
        ),
        actions=[*_episode_actions(), *_episode_actions()],
        num_episodes=2,
    )

    with pytest.raises(RuntimeError, match="writer close failed"):
        finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)

    assert dataset.finalize_calls == 1
    # Nothing was completed and no finalized sidecar was fabricated; both
    # tokens stay saved_pending_audit.
    assert session.finalized_episode_count == 0
    assert session.pending_audit_count == 2
    assert session.batch_finalization_complete is False
    assert session.finalize_failed is True
    assert [record.outcome for record in session.episode_records] == [
        "saved_pending_audit",
        "saved_pending_audit",
    ]


# ----------------------------------------------------- partial complete failure


def test_second_complete_failure_stops_batch_without_resaving(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=2)
    session = _make_session(2)
    dataset = FakeDataset(tmp_path)

    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(
            traces=[*_episode_traces(), *_episode_traces()], exit_afters={FRAME_COUNT, 2 * FRAME_COUNT}
        ),
        actions=[*_episode_actions(), *_episode_actions()],
        num_episodes=2,
    )

    # Sabotage the saved snapshot of episode 1 before the audit phase.
    (tmp_path / "data" / "chunk-000" / "episode_000001.parquet").unlink()
    with pytest.raises(C0GripperSaveFinalizationError, match="structurally invalid"):
        finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)

    # The batch must not claim completion; the second token stays pending;
    # no episode is ever saved again.
    assert dataset.save_calls == 2
    assert session.finalized_episode_count == 1
    assert session.pending_audit_count == 1
    assert session.batch_finalization_complete is False
    assert [record.outcome for record in session.episode_records] == [
        "finalized",
        "saved_pending_audit",
    ]

    # Audit-only retry (no re-save) completes the remaining token once the
    # snapshot is restored.
    _write_dataset(tmp_path, episode_count=2)
    session.on_dataset_finalize_success(dataset=dataset, recorded_episode_count=recorded)
    assert dataset.save_calls == 2
    assert session.pending_audit_count == 0
    assert session.finalized_episode_count == 2
    assert session.batch_finalization_complete is True


# --------------------------------------------------- manifest / index mismatch


def test_manifest_count_mismatch_fails_closed_before_second_episode(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path)

    with pytest.raises(C0GripperRecordingSessionError, match="no episode manifest"):
        _run_session(
            session=session,
            dataset=dataset,
            robot=FakeRobot(
                traces=[*_episode_traces(), *_episode_traces()],
                exit_afters={FRAME_COUNT, 2 * FRAME_COUNT},
            ),
            actions=[*_episode_actions(), *_episode_actions()],
            num_episodes=2,
        )

    # Episode 0 was recorded and saved before the mismatch aborted the run.
    assert dataset.save_calls == 1
    assert session.saved_episode_count == 1


def test_manifest_sequence_validation_fails_closed() -> None:
    with pytest.raises(C0GripperRecordingSessionError, match="at least one"):
        C0GripperRecordingSession(episode_manifests=[], fps=FPS)

    manifests = _session_manifests(2)
    with pytest.raises(C0GripperRecordingSessionError, match="duplicate episode_id"):
        C0GripperRecordingSession(
            episode_manifests=[manifests[0], dataclasses.replace(manifests[0])], fps=FPS
        )

    with pytest.raises(C0GripperRecordingSessionError, match="fps"):
        C0GripperRecordingSession(episode_manifests=manifests, fps=30.0)

    with pytest.raises(TypeError, match="C0EpisodeManifestV1"):
        C0GripperRecordingSession(episode_manifests=[manifests[0], "not-a-manifest"], fps=FPS)

    with pytest.raises(ValueError, match="fps must be a finite positive number"):
        C0GripperRecordingSession(episode_manifests=manifests, fps=0.0)


def test_wrong_episode_index_and_double_observer_fail_closed(tmp_path) -> None:
    session = _make_session(1)

    with pytest.raises(C0GripperRecordingSessionError, match="no open episode journal"):
        session.on_episode_incomplete(
            robot=None, dataset=None, episode_index=0, reason="no journal open"
        )

    journal = session.episode_frame_observer()
    assert isinstance(journal, C0GripperPendingCaptureJournalV1)

    # A second observer request while one journal is open is refused.
    with pytest.raises(C0GripperRecordingSessionError, match="still open"):
        session.episode_frame_observer()

    with pytest.raises(C0GripperRecordingSessionError, match="does not match"):
        session.save_episode(robot=None, dataset=None, episode_index=1)

    session.on_episode_incomplete(robot=None, dataset=None, episode_index=0, reason="test cleanup")
    assert session.episode_records[0].outcome == "incomplete"


# ------------------------------------------------------------ detached views


def test_episode_records_and_tokens_are_detached_deep_copies(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path)
    observers: list[C0GripperPendingCaptureJournalV1] = []

    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(traces=_episode_traces(), exit_afters={FRAME_COUNT}),
        actions=_episode_actions(),
        num_episodes=1,
        observer_sink=observers,
    )
    (live_journal,) = observers

    records_a = session.episode_records
    records_b = session.episode_records
    assert records_a[0] is not records_b[0]
    assert records_a[0].journal is not records_b[0].journal
    assert records_a[0].journal is not live_journal

    # Mutating a returned journal copy must not rewrite live evidence.
    tampered = records_a[0].journal
    tampered.mark_incomplete(reason="tampering attempt")
    tampered._cycles.clear()
    assert live_journal.status == "recording"
    assert len(live_journal.cycles) == FRAME_COUNT

    tokens = session.pending_audit_tokens
    finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)
    assert tokens[0] is not None
    assert session.pending_audit_count == 0
    results = session.finalization_results
    assert len(results) == 1
    assert results[0].episode_index == 0


def test_session_record_keeps_every_authorization_field_false(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=2)
    session = _make_session(2)
    dataset = FakeDataset(tmp_path)

    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(
            traces=[*_episode_traces(), *_episode_traces()], exit_afters={FRAME_COUNT, 2 * FRAME_COUNT}
        ),
        actions=[*_episode_actions(), *_episode_actions()],
        num_episodes=2,
    )
    finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)

    record = session.to_manifest_record()

    def walk_false(node) -> list[str]:
        bad: list[str] = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key in AUTHORIZATION_FIELDS and value is not False:
                    bad.append(key)
                bad.extend(walk_false(value))
        elif isinstance(node, list):
            for item in node:
                bad.extend(walk_false(item))
        return bad

    assert walk_false(record) == []
    json.dumps(record, allow_nan=False)  # pure JSON


def _pending_token(
    *,
    episode_id: str = "c0-r4c-episode-00",
    episode_index: int = 0,
    fps: float = FPS,
    frame_count: int = FRAME_COUNT,
    dataset_root: str = "/tmp/c0-r4c-fake-root",
) -> C0GripperEpisodeSavePendingAuditV1:
    return C0GripperEpisodeSavePendingAuditV1(
        schema_version="C0GripperEpisodeSavePendingAuditV1",
        episode_id=episode_id,
        episode_index=episode_index,
        fps=fps,
        frame_count=frame_count,
        dataset_root=dataset_root,
        candidate_frame_indices=tuple(range(frame_count)),
    )


def _lifecycle_record(**overrides):
    manifest = overrides.pop("manifest", None) or _episode_manifest("c0-r4c-episode-00")
    journal = overrides.pop("journal", None) or C0GripperPendingCaptureJournalV1(
        episode_id=manifest.episode_id, episode_index=0, fps=FPS
    )
    values = {
        "schema_version": C0_GRIPPER_EPISODE_LIFECYCLE_RECORD_SCHEMA_VERSION,
        "episode_index": 0,
        "episode_id": manifest.episode_id,
        "fps": FPS,
        "frame_count": manifest.frame_count,
        "outcome": "recording",
        "manifest": manifest,
        "journal": journal,
        "pending_audit": None,
        "finalization_result": None,
    }
    values.update(overrides)
    return C0GripperEpisodeLifecycleRecordV1(**values)


def _walk_authorization_false(node) -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in AUTHORIZATION_FIELDS and value is not False:
                bad.append(key)
            bad.extend(_walk_authorization_false(value))
    elif isinstance(node, list):
        for item in node:
            bad.extend(_walk_authorization_false(item))
    return bad


def _assert_pure_json_record(record: dict) -> None:
    assert _walk_authorization_false(record) == []
    encoded = json.dumps(record, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == json.loads(json.dumps(record, allow_nan=False, sort_keys=True))


# ------------------------------------------------- batch completeness contract


def test_partial_stopped_batch_promotes_saved_tokens_but_is_not_batch_complete(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=2)
    session = _make_session(2)
    dataset = FakeDataset(tmp_path)

    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(
            traces=[*_episode_traces(), *_episode_traces()],
            stop_afters={FRAME_COUNT + 1},
            exit_afters={FRAME_COUNT, FRAME_COUNT + 1},
        ),
        actions=[*_episode_actions(), *_episode_actions()],
        num_episodes=2,
    )
    finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)

    assert recorded == 1
    assert [record.outcome for record in session.episode_records] == ["finalized", "incomplete"]
    assert session.finalized_episode_count == 1
    assert session.pending_audit_count == 0
    assert session.batch_finalization_complete is False
    assert session.to_manifest_record()["batch_finalization_complete"] is False


def test_failed_only_batch_is_not_batch_complete(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path)

    with pytest.raises(RuntimeError, match="gripper DO failed"):
        try:
            _run_session(
                session=session,
                dataset=dataset,
                robot=FakeRobot(
                    traces=_episode_traces(),
                    send_error=RuntimeError("gripper DO failed"),
                    error_after=ON_FRAME,
                ),
                actions=_episode_actions(),
                num_episodes=1,
            )
        finally:
            finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=0)

    assert dataset.save_calls == 0
    assert session.episode_records[0].outcome == "incomplete"
    assert session.batch_finalization_complete is False
    assert session.to_manifest_record()["batch_finalization_complete"] is False


def test_all_manifests_saved_and_promoted_is_batch_complete(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=2)
    session = _make_session(2)
    dataset = FakeDataset(tmp_path)

    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(
            traces=[*_episode_traces(), *_episode_traces()],
            exit_afters={FRAME_COUNT, 2 * FRAME_COUNT},
        ),
        actions=[*_episode_actions(), *_episode_actions()],
        num_episodes=2,
    )
    finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)

    assert recorded == 2
    assert [record.outcome for record in session.episode_records] == ["finalized", "finalized"]
    assert session.batch_finalization_complete is True
    assert session.to_manifest_record()["batch_finalization_complete"] is True


# ------------------------------------------- pre-save rejection vs uncertainty


def test_pre_save_validation_failure_is_not_save_uncertain(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path)

    with pytest.raises(C0GripperSaveFinalizationError, match="cover exactly"):
        try:
            _run_session(
                session=session,
                dataset=dataset,
                robot=FakeRobot(traces=_episode_traces(), exit_afters={4}),
                actions=_episode_actions()[:4],
                num_episodes=1,
            )
        finally:
            finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=0)

    assert dataset.save_calls == 0
    (record,) = session.episode_records
    assert record.outcome == "pre_save_rejected"
    assert record.pending_audit is None
    assert record.finalization_result is None
    assert getattr(record.journal, "_c0_r4b_state", "idle") == "idle"
    assert session.batch_finalization_complete is False
    payload = record.to_manifest_record()
    assert payload["outcome"] == "pre_save_rejected"
    _assert_pure_json_record(payload)


def test_save_call_failure_is_save_uncertain_and_never_retried(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path, save_error=RuntimeError("flush crashed"))

    with pytest.raises(RuntimeError, match="flush crashed"):
        try:
            _run_session(
                session=session,
                dataset=dataset,
                robot=FakeRobot(traces=_episode_traces(), exit_afters={FRAME_COUNT}),
                actions=_episode_actions(),
                num_episodes=1,
            )
        finally:
            finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=0)

    assert dataset.save_calls == 1
    with pytest.raises(C0GripperRecordingSessionError, match="no open episode journal"):
        session.save_episode(robot=None, dataset=dataset, episode_index=0)
    assert dataset.save_calls == 1
    (record,) = session.episode_records
    assert record.outcome == "save_uncertain"
    assert record.pending_audit is None
    assert record.finalization_result is None
    assert record.journal._c0_r4b_state == "save_uncertain"
    assert session.saved_episode_count == 0
    assert session.batch_finalization_complete is False
    payload = record.to_manifest_record()
    assert payload["outcome"] == "save_uncertain"
    _assert_pure_json_record(payload)
    _assert_pure_json_record(session.to_manifest_record())


# ------------------------------------------------- lifecycle record fail-closed


def test_lifecycle_record_rejects_forged_state_combinations(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path)
    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(traces=_episode_traces(), exit_afters={FRAME_COUNT}),
        actions=_episode_actions(),
        num_episodes=1,
    )
    finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)
    good = session.episode_records[0]
    token = _pending_token()
    result = good.finalization_result

    with pytest.raises(ValueError, match="requires a pending token"):
        _lifecycle_record(outcome="saved_pending_audit", pending_audit=None)

    with pytest.raises(ValueError, match="must not carry a pending token"):
        _lifecycle_record(outcome="recording", pending_audit=token)

    with pytest.raises(ValueError, match="must not carry a pending token"):
        _lifecycle_record(outcome="incomplete", pending_audit=token)

    with pytest.raises(ValueError, match="requires a finalization result"):
        _lifecycle_record(outcome="finalized", pending_audit=token, finalization_result=None)

    with pytest.raises(ValueError, match="must not carry a finalization result"):
        _lifecycle_record(outcome="incomplete", finalization_result=result)

    with pytest.raises(ValueError, match="pending token episode_id"):
        _lifecycle_record(
            outcome="saved_pending_audit",
            pending_audit=_pending_token(episode_id="foreign-episode"),
        )

    with pytest.raises(ValueError, match="pending token episode_index"):
        _lifecycle_record(
            outcome="saved_pending_audit",
            pending_audit=_pending_token(episode_index=1),
        )

    with pytest.raises(ValueError, match="pending token fps"):
        _lifecycle_record(
            outcome="saved_pending_audit",
            pending_audit=_pending_token(fps=30.0),
        )

    with pytest.raises(ValueError, match="pending token frame_count"):
        _lifecycle_record(
            outcome="saved_pending_audit",
            pending_audit=_pending_token(frame_count=8),
        )

    foreign_manifest = _episode_manifest("c0-r4c-episode-99")
    foreign_journal = C0GripperPendingCaptureJournalV1(
        episode_id="c0-r4c-episode-99", episode_index=0, fps=FPS
    )
    with pytest.raises(ValueError, match="finalization result episode_id"):
        C0GripperEpisodeLifecycleRecordV1(
            schema_version=C0_GRIPPER_EPISODE_LIFECYCLE_RECORD_SCHEMA_VERSION,
            episode_index=0,
            episode_id="c0-r4c-episode-99",
            fps=FPS,
            frame_count=FRAME_COUNT,
            outcome="finalized",
            manifest=foreign_manifest,
            journal=foreign_journal,
            pending_audit=_pending_token(episode_id="c0-r4c-episode-99"),
            finalization_result=result,
        )


def _run_one_episode_to_outcome(tmp_path, *, finalize: bool):
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path)
    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(traces=_episode_traces(), exit_afters={FRAME_COUNT}),
        actions=_episode_actions(),
        num_episodes=1,
    )
    if finalize:
        finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)
    return session, recorded


def test_lifecycle_record_rejects_incomplete_with_recording_journal() -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id="c0-r4c-episode-00", episode_index=0, fps=FPS
    )
    assert journal.status == "recording"
    with pytest.raises(ValueError, match="outcome 'incomplete' requires journal status 'incomplete'"):
        _lifecycle_record(outcome="incomplete", journal=journal)


def test_lifecycle_record_rejects_abandoned_with_recording_journal() -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id="c0-r4c-episode-00", episode_index=0, fps=FPS
    )
    assert journal.status == "recording"
    with pytest.raises(
        ValueError,
        match="outcome 'abandoned_rerecorded' requires journal status 'abandoned_rerecorded'",
    ):
        _lifecycle_record(outcome="abandoned_rerecorded", journal=journal)


def test_lifecycle_record_rejects_save_uncertain_with_idle_r4b_state() -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id="c0-r4c-episode-00", episode_index=0, fps=FPS
    )
    assert journal.status == "recording"
    assert getattr(journal, "_c0_r4b_state", "idle") == "idle"
    with pytest.raises(ValueError, match="journal save state is 'idle'; expected 'save_uncertain'"):
        _lifecycle_record(outcome="save_uncertain", journal=journal)


def test_lifecycle_record_rejects_pending_with_idle_r4b_state() -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id="c0-r4c-episode-00", episode_index=0, fps=FPS
    )
    token = _pending_token()
    assert journal.status == "recording"
    assert getattr(journal, "_c0_r4b_state", "idle") == "idle"
    assert token.episode_id == journal.episode_id
    assert token.episode_index == journal.episode_index
    assert float(token.fps) == float(journal.fps)
    assert token.frame_count == FRAME_COUNT
    with pytest.raises(
        ValueError, match="journal save state is 'idle'; expected 'saved_pending_audit'"
    ):
        _lifecycle_record(outcome="saved_pending_audit", journal=journal, pending_audit=token)


def test_lifecycle_record_rejects_same_identity_foreign_pending_token(tmp_path) -> None:
    session, _recorded = _run_one_episode_to_outcome(tmp_path, finalize=False)
    live = session.episode_records[0]
    assert live.outcome == "saved_pending_audit"
    foreign_root = tmp_path / "foreign-root"
    foreign_root.mkdir()
    foreign = _pending_token(
        episode_id=live.episode_id,
        episode_index=live.episode_index,
        fps=live.fps,
        frame_count=live.frame_count,
        dataset_root=str(foreign_root.resolve()),
    )
    assert foreign.episode_id == live.episode_id
    assert foreign.episode_index == live.episode_index
    assert float(foreign.fps) == float(live.fps)
    assert foreign.frame_count == live.frame_count
    assert foreign.dataset_root != live.pending_audit.dataset_root
    with pytest.raises(ValueError, match="pending token is not the one produced by this journal"):
        C0GripperEpisodeLifecycleRecordV1(
            schema_version=live.schema_version,
            episode_index=live.episode_index,
            episode_id=live.episode_id,
            fps=live.fps,
            frame_count=live.frame_count,
            outcome="saved_pending_audit",
            manifest=live.manifest,
            journal=live.journal,
            pending_audit=foreign,
            finalization_result=None,
        )


def test_finalized_record_rejects_token_root_mismatch_with_dataset_report(tmp_path) -> None:
    session, _recorded = _run_one_episode_to_outcome(tmp_path, finalize=True)
    live = session.episode_records[0]
    assert live.outcome == "finalized"
    foreign_root = tmp_path / "other-dataset"
    foreign_root.mkdir()
    foreign_token = dataclasses.replace(
        live.pending_audit, dataset_root=str(foreign_root.resolve())
    )
    assert foreign_token.episode_id == live.episode_id
    assert foreign_token.episode_index == live.episode_index
    assert float(foreign_token.fps) == float(live.fps)
    assert foreign_token.frame_count == live.frame_count
    assert foreign_token.dataset_root != live.finalization_result.dataset_report.dataset_root
    with pytest.raises(ValueError, match="pending token is not the one produced by this journal"):
        C0GripperEpisodeLifecycleRecordV1(
            schema_version=live.schema_version,
            episode_index=live.episode_index,
            episode_id=live.episode_id,
            fps=live.fps,
            frame_count=live.frame_count,
            outcome="finalized",
            manifest=live.manifest,
            journal=live.journal,
            pending_audit=foreign_token,
            finalization_result=live.finalization_result,
        )


def test_finalized_record_rejects_result_from_another_same_identity_save(tmp_path) -> None:
    root_a = tmp_path / "save-a"
    root_b = tmp_path / "save-b"
    session_a, _ = _run_one_episode_to_outcome(root_a, finalize=True)
    session_b, _ = _run_one_episode_to_outcome(root_b, finalize=True)
    rec_a = session_a.episode_records[0]
    rec_b = session_b.episode_records[0]
    assert rec_a.episode_id == rec_b.episode_id
    assert rec_a.episode_index == rec_b.episode_index
    assert rec_a.fps == rec_b.fps
    assert rec_a.frame_count == rec_b.frame_count
    assert rec_a.pending_audit.dataset_root != rec_b.pending_audit.dataset_root
    assert (
        rec_a.finalization_result.dataset_report.dataset_root
        != rec_b.finalization_result.dataset_report.dataset_root
    )
    with pytest.raises(ValueError, match="dataset_root"):
        C0GripperEpisodeLifecycleRecordV1(
            schema_version=rec_a.schema_version,
            episode_index=rec_a.episode_index,
            episode_id=rec_a.episode_id,
            fps=rec_a.fps,
            frame_count=rec_a.frame_count,
            outcome="finalized",
            manifest=rec_a.manifest,
            journal=rec_a.journal,
            pending_audit=rec_a.pending_audit,
            finalization_result=rec_b.finalization_result,
        )


def test_lifecycle_record_rejects_same_root_foreign_finalized_result(tmp_path) -> None:
    _write_dataset(tmp_path, episode_count=1)
    session = _make_session(1)
    dataset = FakeDataset(tmp_path)
    recorded = _run_session(
        session=session,
        dataset=dataset,
        robot=FakeRobot(traces=_episode_traces(), exit_afters={FRAME_COUNT}),
        actions=_episode_actions(),
        num_episodes=1,
    )
    pending_record = session.episode_records[0]
    journal_b = copy.deepcopy(pending_record.journal)
    pending_b = copy.deepcopy(pending_record.pending_audit)
    finalize_recorded_dataset(dataset, lifecycle=session, recorded_episode_count=recorded)
    rec_a = session.episode_records[0]
    _write_dataset(tmp_path, episode_count=1, timestamp_offset=0.01)
    result_b = complete_c0_gripper_episode_finalization(
        journal=journal_b,
        dataset=dataset,
        pending=pending_b,
        episode_manifest=rec_a.manifest,
    )
    assert rec_a.episode_id == result_b.episode_id
    assert rec_a.episode_index == result_b.episode_index
    assert rec_a.fps == result_b.fps
    assert rec_a.frame_count == result_b.frame_count
    assert rec_a.pending_audit.dataset_root == result_b.dataset_report.dataset_root
    assert rec_a.pending_audit.to_manifest_record() == pending_b.to_manifest_record()
    assert rec_a.finalization_result.dataset_gripper_slice_sha256 != result_b.dataset_gripper_slice_sha256
    with pytest.raises(ValueError, match="finalization result fingerprint"):
        require_c0_gripper_finalized_evidence_bound(
            journal=rec_a.journal, pending=rec_a.pending_audit, result=result_b
        )
    with pytest.raises(ValueError, match="finalization result fingerprint"):
        C0GripperEpisodeLifecycleRecordV1(
            schema_version=rec_a.schema_version,
            episode_index=rec_a.episode_index,
            episode_id=rec_a.episode_id,
            fps=rec_a.fps,
            frame_count=rec_a.frame_count,
            outcome="finalized",
            manifest=rec_a.manifest,
            journal=rec_a.journal,
            pending_audit=rec_a.pending_audit,
            finalization_result=result_b,
        )


def test_all_seven_lifecycle_outcomes_construct_and_serialize_deterministically(tmp_path) -> None:
    constructed: list[C0GripperEpisodeLifecycleRecordV1] = []

    recording_session = _make_session(1)
    recording_session.episode_frame_observer()
    recording = recording_session.episode_records[0]
    assert recording.outcome == "recording"
    assert recording.journal.status == "recording"
    assert recording.pending_audit is None
    constructed.append(recording)

    abandoned_session = _make_session(1)
    abandoned_session.episode_frame_observer()
    abandoned_session.on_episode_rerecord(robot=None, dataset=None, episode_index=0)
    abandoned = abandoned_session.episode_records[0]
    assert abandoned.outcome == "abandoned_rerecorded"
    assert abandoned.journal.status == "abandoned_rerecorded"
    constructed.append(abandoned)

    incomplete_session = _make_session(1)
    incomplete_session.episode_frame_observer()
    incomplete_session.on_episode_incomplete(
        robot=None, dataset=None, episode_index=0, reason="operator stop"
    )
    incomplete = incomplete_session.episode_records[0]
    assert incomplete.outcome == "incomplete"
    assert incomplete.journal.status == "incomplete"
    constructed.append(incomplete)

    rejected_root = tmp_path / "pre-save"
    _write_dataset(rejected_root, episode_count=1)
    rejected_session = _make_session(1)
    rejected_dataset = FakeDataset(rejected_root)
    with pytest.raises(C0GripperSaveFinalizationError, match="cover exactly"):
        _run_session(
            session=rejected_session,
            dataset=rejected_dataset,
            robot=FakeRobot(traces=_episode_traces(), exit_afters={4}),
            actions=_episode_actions()[:4],
            num_episodes=1,
        )
    rejected = rejected_session.episode_records[0]
    assert rejected.outcome == "pre_save_rejected"
    assert rejected.journal.status == "recording"
    constructed.append(rejected)

    uncertain_root = tmp_path / "uncertain"
    _write_dataset(uncertain_root, episode_count=1)
    uncertain_session = _make_session(1)
    uncertain_dataset = FakeDataset(uncertain_root, save_error=RuntimeError("flush crashed"))
    with pytest.raises(RuntimeError, match="flush crashed"):
        _run_session(
            session=uncertain_session,
            dataset=uncertain_dataset,
            robot=FakeRobot(traces=_episode_traces(), exit_afters={FRAME_COUNT}),
            actions=_episode_actions(),
            num_episodes=1,
        )
    uncertain = uncertain_session.episode_records[0]
    assert uncertain.outcome == "save_uncertain"
    assert uncertain.journal.status == "recording"
    constructed.append(uncertain)

    pending_root = tmp_path / "pending"
    pending_session, _ = _run_one_episode_to_outcome(pending_root, finalize=False)
    pending = pending_session.episode_records[0]
    assert pending.outcome == "saved_pending_audit"
    assert pending.journal.status == "recording"
    constructed.append(pending)

    finalized_root = tmp_path / "finalized"
    finalized_session, _ = _run_one_episode_to_outcome(finalized_root, finalize=True)
    finalized = finalized_session.episode_records[0]
    assert finalized.outcome == "finalized"
    assert finalized.journal.status == "recording"
    constructed.append(finalized)

    assert [record.outcome for record in constructed] == [
        "recording",
        "abandoned_rerecorded",
        "incomplete",
        "pre_save_rejected",
        "save_uncertain",
        "saved_pending_audit",
        "finalized",
    ]
    for record in constructed:
        payload = record.to_manifest_record()
        _assert_pure_json_record(payload)
        again = record.to_manifest_record()
        assert payload == again
        assert json.dumps(payload, sort_keys=True, allow_nan=False) == json.dumps(
            again, sort_keys=True, allow_nan=False
        )


def test_mutating_returned_lifecycle_evidence_does_not_rewrite_session(tmp_path) -> None:
    session, _ = _run_one_episode_to_outcome(tmp_path, finalize=True)
    record = session.episode_records[0]
    original_root = record.pending_audit.dataset_root
    original_episode_id = record.finalization_result.episode_id
    original_status = record.journal.status

    record.journal.mark_incomplete(reason="external tamper")
    record.journal._cycles.clear()
    object.__setattr__(record.pending_audit, "dataset_root", "/forged-root")
    object.__setattr__(record.finalization_result, "episode_id", "forged-episode")
    payload = record.to_manifest_record()
    payload["outcome"] = "forged"
    payload["episode_id"] = "forged-episode"

    live = session.episode_records[0]
    assert live.outcome == "finalized"
    assert live.journal.status == original_status == "recording"
    assert len(live.journal.cycles) == FRAME_COUNT
    assert live.pending_audit.dataset_root == original_root
    assert live.finalization_result.episode_id == original_episode_id
    assert session.batch_finalization_complete is True

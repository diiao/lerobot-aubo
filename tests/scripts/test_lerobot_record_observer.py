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

"""Offline integration tests for the optional record_loop observer hook.

Everything here is mocked: the robot, the teleoperator, and the dataset are
in-memory fakes. No camera, AUBO, DO/IO, motion, real dataset on disk,
network, training, or inference is involved. ``tmp_path`` is only used to
persist journal snapshots for inspection, never as a dataset root.
"""

import json

import pytest

from lerobot.bamboo_sorting.c0_gripper_capture_journal import (
    C0GripperPendingCaptureJournalV1,
    C0PendingCaptureError,
)
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
    make_default_teleop_action_processor,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig

EPISODE_ID = "c0-record-loop-observer-episode-00"
FPS = 30.0


def _success_trace(target_on: bool) -> dict:
    """Mirror AuboI10Robot._set_suction_outputs for a fully successful switch."""

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


def _failure_trace() -> dict:
    return {
        "requested_transition": True,
        "target_suction_on": True,
        "requested_do": {"2": True, "3": False},
        "do_writes": [{"pin": 3, "value": False, "return_code": 0}],
        "do_write_attempt_count": 2,
        "do_api_success": False,
        "do_readback_supported": True,
        "do_readback": None,
        "do_readback_after_failure": {"2": False, "3": False},
        "do_readback_error": None,
        "controller_output_state_known": True,
        "controller_output_matches_requested": False,
        "partial_write_possible": True,
        "commanded_state_before": False,
        "commanded_state_after": False,
        "error": "setStandardDigitalOutput(pin=2, value=True) returned -1",
    }


def _no_transition_trace(on: bool) -> dict:
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


class FakeRobot:
    """Duck-typed robot with scripted per-cycle gripper traces."""

    name = "fake_robot"

    def __init__(
        self,
        traces=(),
        send_error: Exception | None = None,
        exit_after: int = 1,
        returned_action: dict | None = None,
    ):
        self.traces = list(traces)
        self.send_error = send_error
        self.exit_after = exit_after
        self.returned_action = returned_action
        self.sent_actions: list[dict] = []
        self.last_gripper_command_trace: dict | None = None
        self.events: dict | None = None

    def get_observation(self) -> dict:
        return {"a": 0.0, "b": 0.0}

    def send_action(self, action: dict) -> dict:
        self.sent_actions.append(action)
        # Fresh trace per cycle, mimicking the AUBO driver freshness contract.
        index = len(self.sent_actions) - 1
        self.last_gripper_command_trace = self.traces[index] if index < len(self.traces) else None
        if self.send_error is not None and len(self.sent_actions) >= self.exit_after:
            raise self.send_error
        if len(self.sent_actions) >= self.exit_after and self.events is not None:
            self.events["exit_early"] = True
        return action if self.returned_action is None else self.returned_action


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
    """Duck-typed LeRobotDataset stand-in (features, episode_buffer, add_frame)."""

    def __init__(
        self,
        *,
        fps: float = FPS,
        initial_buffer_size: int | None = None,
        add_frame_error: Exception | None = None,
        partial_mutation_before_error: bool = False,
    ):
        self.fps = fps
        self.features = {
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "action.ee.gripper_pos": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["ee.gripper_pos"],
            },
        }
        self.episode_buffer = (
            {"size": initial_buffer_size} if initial_buffer_size is not None else None
        )
        self.add_frame_error = add_frame_error
        self.partial_mutation_before_error = partial_mutation_before_error
        self.frames: list[dict] = []
        self.add_frame_calls = 0

    def add_frame(self, frame: dict) -> None:
        self.add_frame_calls += 1
        if self.episode_buffer is None:
            self.episode_buffer = {"size": 0}
        if self.partial_mutation_before_error:
            # Simulate a buffer that was partially mutated before raising.
            self.episode_buffer["size"] += 1
        if self.add_frame_error is not None:
            raise self.add_frame_error
        self.episode_buffer["size"] += 1
        self.frames.append(frame)


class GripperFlipPipeline:
    """Robot action pipeline that silently flips the gripper command."""

    def __call__(self, pair) -> dict:
        action, _obs = pair
        flipped = dict(action)
        flipped["ee.gripper_pos"] = 100.0 - float(action["ee.gripper_pos"])
        return flipped


class CountingObserver:
    """Generic observer that counts hook invocations per stage."""

    def __init__(self, raise_at: str | None = None):
        self.raise_at = raise_at
        self.calls: list[tuple[str, int]] = []

    def _record(self, stage: str, candidate_frame_index: int) -> None:
        self.calls.append((stage, candidate_frame_index))
        if self.raise_at == stage:
            raise RuntimeError(f"boom in {stage}")

    def before_send(self, **kwargs) -> None:
        self._record("before_send", kwargs["candidate_frame_index"])

    def on_send_success(self, **kwargs) -> None:
        self._record("on_send_success", kwargs["candidate_frame_index"])

    def on_send_failure(self, **kwargs) -> None:
        self._record("on_send_failure", kwargs["candidate_frame_index"])

    def on_add_frame_success(self, **kwargs) -> None:
        self._record("on_add_frame_success", kwargs["candidate_frame_index"])

    def on_add_frame_failure(self, **kwargs) -> None:
        self._record("on_add_frame_failure", kwargs["candidate_frame_index"])


def _run_record_loop(
    *,
    robot: FakeRobot,
    teleop_actions: list[dict],
    dataset=None,
    observer=None,
    robot_action_processor=None,
    control_time_s: float = 60.0,
):
    events = {"exit_early": False}
    robot.events = events
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=make_default_teleop_action_processor(),
        robot_action_processor=robot_action_processor or make_default_robot_action_processor(),
        robot_observation_processor=make_default_robot_observation_processor(),
        dataset=dataset,
        teleop=FakeTeleop(teleop_actions),
        control_time_s=control_time_s,
        single_task="sort bamboo",
        observer=observer,
    )


def _gripper_action(value: float) -> dict:
    return {"ee.gripper_pos": value}


def _snapshot(journal: C0GripperPendingCaptureJournalV1, tmp_path) -> dict:
    record = journal.to_manifest_record()
    out = tmp_path / f"{journal.episode_id}.json"
    out.write_text(json.dumps(record, sort_keys=True, allow_nan=False))
    return json.loads(out.read_text())


# ------------------------------------------------------------- default behavior


def test_observer_none_keeps_original_record_loop_behavior(tmp_path) -> None:
    robot = FakeRobot(traces=[_no_transition_trace(on=False)])
    dataset = FakeDataset()

    _run_record_loop(
        robot=robot,
        teleop_actions=[_gripper_action(0.0)],
        dataset=dataset,
        observer=None,
    )

    assert len(robot.sent_actions) == 1
    assert dataset.add_frame_calls == 1
    (frame,) = dataset.frames
    assert "action.ee.gripper_pos" in frame
    assert frame["task"] == "sort bamboo"


def test_send_action_failure_skips_add_frame_without_observer(tmp_path) -> None:
    robot = FakeRobot(send_error=RuntimeError("motion failed"))
    dataset = FakeDataset()

    with pytest.raises(RuntimeError, match="motion failed"):
        _run_record_loop(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)],
            dataset=dataset,
        )

    assert len(robot.sent_actions) == 1
    assert dataset.add_frame_calls == 0


# ------------------------------------------------------- journal integration


def test_command_on_success_full_cycle(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    fresh_trace = _success_trace(target_on=True)
    robot = FakeRobot(traces=[fresh_trace])
    dataset = FakeDataset()

    _run_record_loop(
        robot=robot,
        teleop_actions=[_gripper_action(100.0)],
        dataset=dataset,
        observer=journal,
    )

    assert len(robot.sent_actions) == 1
    assert dataset.add_frame_calls == 1

    (cycle,) = journal.cycles
    assert cycle["candidate_frame_index"] == 0
    assert cycle["dataset_action_gripper_pos"] == 100.0
    assert cycle["sent_action_gripper_command"] == 100.0
    assert cycle["action_consistency"] == "matched"
    assert cycle["send_action_returned"] is True
    assert cycle["requested_transition"] is True
    assert cycle["frame_entered_episode_buffer"] is True
    assert cycle["dataset_episode_durably_saved"] is False
    # Fresh trace captured as a deep copy, not the live robot object.
    assert cycle["trace"] == fresh_trace
    assert cycle["trace"] is not fresh_trace

    (event,) = journal.pending_events
    (attempt,) = journal.controller_attempts
    assert event.event_type == "command_on"
    assert event.frame_index == 0
    assert event.event_index == attempt.attempt_index == 0
    # R4A: add_frame success only means the frame entered the episode buffer;
    # the pending attempt stays uncommitted until R4B durable save + digest.
    assert attempt.dataset_frame_committed is False
    assert attempt.event == event

    record = _snapshot(journal, tmp_path)
    assert record["dataset_episode_durably_saved"] is False
    assert record["training_authorized"] is False
    assert record["final_sidecar_published"] is False


def test_command_off_success_full_cycle(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    robot = FakeRobot(traces=[_success_trace(target_on=False)])

    _run_record_loop(
        robot=robot,
        teleop_actions=[_gripper_action(0.0)],
        dataset=FakeDataset(),
        observer=journal,
    )

    (event,) = journal.pending_events
    assert event.event_type == "command_off"
    assert event.requested_state_after == 0.0
    assert journal.controller_attempts[0].dataset_frame_committed is False


def test_requested_transition_false_records_evidence_without_controller_attempt(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    robot = FakeRobot(traces=[_no_transition_trace(on=True)])

    _run_record_loop(
        robot=robot,
        teleop_actions=[_gripper_action(100.0)],
        dataset=FakeDataset(),
        observer=journal,
    )

    (cycle,) = journal.cycles
    assert cycle["trace_present"] is True
    assert cycle["requested_transition"] is False
    assert cycle["frame_entered_episode_buffer"] is True
    assert journal.pending_events == ()
    assert journal.controller_attempts == ()


def test_do_failure_keeps_trace_buffered_false_no_resend_no_add_frame(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    robot = FakeRobot(
        traces=[_failure_trace()],
        send_error=RuntimeError("夹爪 DO 状态切换失败"),
    )
    dataset = FakeDataset()

    with pytest.raises(RuntimeError, match="夹爪 DO 状态切换失败"):
        _run_record_loop(
            robot=robot,
            teleop_actions=[_gripper_action(100.0)],
            dataset=dataset,
            observer=journal,
        )

    # No resend, no add_frame.
    assert len(robot.sent_actions) == 1
    assert dataset.add_frame_calls == 0

    (cycle,) = journal.cycles
    assert cycle["send_action_returned"] is False
    assert cycle["failure_stage"] == "send_action"
    assert cycle["frame_entered_episode_buffer"] is False
    assert cycle["trace"]["do_api_success"] is False
    assert journal.episode_failed is True

    (attempt,) = journal.controller_attempts
    assert attempt.dataset_frame_committed is False
    assert attempt.event.do_api_success is False

    record = _snapshot(journal, tmp_path)
    assert record["dataset_episode_durably_saved"] is False


def test_do_success_but_add_frame_failure_fails_closed(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    robot = FakeRobot(traces=[_success_trace(target_on=True)])
    dataset = FakeDataset(
        add_frame_error=RuntimeError("parquet flush failed"),
        partial_mutation_before_error=True,
    )

    with pytest.raises(RuntimeError, match="parquet flush failed"):
        _run_record_loop(
            robot=robot,
            teleop_actions=[_gripper_action(100.0)],
            dataset=dataset,
            observer=journal,
        )

    # No resend, no repeated add_frame.
    assert len(robot.sent_actions) == 1
    assert dataset.add_frame_calls == 1

    (cycle,) = journal.cycles
    assert cycle["send_action_returned"] is True
    assert cycle["failure_stage"] == "add_frame"
    # Even though the underlying buffer size was partially mutated, the frame
    # did not enter the buffer from the journal's point of view.
    assert cycle["frame_entered_episode_buffer"] is False
    assert cycle["dataset_episode_durably_saved"] is False
    assert journal.episode_failed is True
    assert journal.controller_attempts[0].dataset_frame_committed is False

    journal.mark_incomplete(reason="add_frame raised; episode abandoned by control loop")
    record = _snapshot(journal, tmp_path)
    assert record["status"] == "incomplete"
    assert record["dataset_episode_durably_saved"] is False
    assert record["finalized"] is False


def test_action_mismatch_aborts_before_send(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    robot = FakeRobot(traces=[_success_trace(target_on=True)])
    dataset = FakeDataset()

    with pytest.raises(C0PendingCaptureError, match="action mismatch"):
        _run_record_loop(
            robot=robot,
            teleop_actions=[_gripper_action(100.0)],
            dataset=dataset,
            observer=journal,
            robot_action_processor=GripperFlipPipeline(),
        )

    # send_action and add_frame must never run.
    assert len(robot.sent_actions) == 0
    assert dataset.add_frame_calls == 0

    (cycle,) = journal.cycles
    assert cycle["failure_stage"] == "before_send"
    assert cycle["error_type"] == "action_mismatch"
    assert cycle["frame_entered_episode_buffer"] is False
    assert journal.episode_failed is True
    assert journal.controller_attempts == ()


def test_returned_action_gripper_contradiction_fails_before_add_frame(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    # robot_action_to_send gripper=100 and the trace points "on", but
    # send_action returns an independent dict with gripper=0.
    robot = FakeRobot(
        traces=[_success_trace(target_on=True)],
        returned_action=_gripper_action(0.0),
    )
    dataset = FakeDataset()

    with pytest.raises(C0PendingCaptureError, match="returned action gripper contradicts"):
        _run_record_loop(
            robot=robot,
            teleop_actions=[_gripper_action(100.0)],
            dataset=dataset,
            observer=journal,
        )

    # Exactly one send, zero add_frame, journal fail closed.
    assert len(robot.sent_actions) == 1
    assert dataset.add_frame_calls == 0
    (cycle,) = journal.cycles
    assert cycle["error_type"] == "sent_action_mismatch"
    assert cycle["failure_stage"] == "send_action"
    assert cycle["frame_entered_episode_buffer"] is False
    assert journal.episode_failed is True
    assert journal.pending_events == ()
    assert journal.controller_attempts == ()


def test_send_success_with_missing_trace_fails_before_add_frame(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    robot = FakeRobot(traces=[])
    dataset = FakeDataset()

    with pytest.raises(C0PendingCaptureError, match="trace is None"):
        _run_record_loop(
            robot=robot,
            teleop_actions=[_gripper_action(100.0)],
            dataset=dataset,
            observer=journal,
        )

    assert len(robot.sent_actions) == 1
    assert dataset.add_frame_calls == 0
    (cycle,) = journal.cycles
    assert cycle["error_type"] == "trace_missing"
    assert cycle["frame_entered_episode_buffer"] is False
    assert journal.episode_failed is True


def test_candidate_frame_index_comes_from_episode_buffer_size(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    robot = FakeRobot(
        traces=[_no_transition_trace(on=False), _no_transition_trace(on=False)],
        exit_after=2,
    )
    dataset = FakeDataset(initial_buffer_size=5)

    _run_record_loop(
        robot=robot,
        teleop_actions=[_gripper_action(0.0), _gripper_action(0.0)],
        dataset=dataset,
        observer=journal,
    )

    # Buffer-derived: 5, 6. An observer-side frame counter would say 0, 1.
    assert [cycle["candidate_frame_index"] for cycle in journal.cycles] == [5, 6]
    assert dataset.episode_buffer["size"] == 7


def test_reset_loop_with_dataset_none_generates_no_journal_cycles(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    counting = CountingObserver()
    robot = FakeRobot(traces=[_no_transition_trace(on=False)])

    # dataset=None models the reset loop; even with observers passed in, no
    # dataset capture lifecycle may occur.
    _run_record_loop(
        robot=robot,
        teleop_actions=[_gripper_action(0.0)],
        dataset=None,
        observer=counting,
    )
    _run_record_loop(
        robot=robot,
        teleop_actions=[_gripper_action(0.0)],
        dataset=None,
        observer=journal,
    )

    assert counting.calls == []
    assert journal.cycles == ()
    assert journal.pending_events == ()
    assert journal.controller_attempts == ()


def test_rerecord_isolates_new_journal_from_old_attempts(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    robot = FakeRobot(traces=[_success_trace(target_on=True)])
    _run_record_loop(
        robot=robot,
        teleop_actions=[_gripper_action(100.0)],
        dataset=FakeDataset(),
        observer=journal,
    )
    assert len(journal.controller_attempts) == 1

    journal.mark_abandoned(reason="operator pressed rerecord")
    nxt = journal.fork_for_rerecord(episode_id=EPISODE_ID, episode_index=0)
    # Old attempts never leak into the new generation.
    assert nxt.cycles == ()
    assert nxt.controller_attempts == ()
    assert nxt.pending_events == ()
    assert nxt.status == "recording"

    old_record = _snapshot(journal, tmp_path)
    assert old_record["status"] == "abandoned_rerecorded"
    assert old_record["abandon_reason"] == "operator pressed rerecord"


def test_stop_recording_marks_incomplete_with_all_final_fields_false(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    robot = FakeRobot(traces=[_success_trace(target_on=True)])
    _run_record_loop(
        robot=robot,
        teleop_actions=[_gripper_action(100.0)],
        dataset=FakeDataset(),
        observer=journal,
    )

    journal.mark_incomplete(reason="operator stopped recording")
    record = _snapshot(journal, tmp_path)
    assert record["status"] == "incomplete"
    assert record["finalized"] is False
    assert record["dataset_episode_durably_saved"] is False
    assert record["dataset_gripper_slice_digest_bound"] is False
    assert record["final_sidecar_published"] is False
    assert record["physical_gripper_feedback_available"] is False
    assert record["physical_grasp_success_proven"] is False
    assert record["training_authorized"] is False
    assert record["policy_execution_authorized"] is False
    assert record["serialized_record_grants_live_authorization"] is False
    assert record["hardware_access_performed_by_serialization"] is False


def test_trace_deep_copy_survives_robot_overwrite_in_next_cycle(tmp_path) -> None:
    journal = C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )
    first_trace = _success_trace(target_on=True)
    second_trace = _failure_trace()
    robot = FakeRobot(
        traces=[first_trace, second_trace],
        send_error=RuntimeError("夹爪 DO 状态切换失败"),
        exit_after=2,
    )
    dataset = FakeDataset()

    with pytest.raises(RuntimeError, match="夹爪 DO 状态切换失败"):
        _run_record_loop(
            robot=robot,
            teleop_actions=[_gripper_action(100.0), _gripper_action(100.0)],
            dataset=dataset,
            observer=journal,
        )

    first, second = journal.cycles
    # First cycle keeps its own fresh success trace even though the robot
    # later overwrote last_gripper_command_trace with a failed one.
    assert first["trace"] == first_trace
    assert first["trace"]["do_api_success"] is True
    assert first["frame_entered_episode_buffer"] is True
    assert second["trace"] == second_trace
    assert second["frame_entered_episode_buffer"] is False


# ----------------------------------------------------------- observer failures


def test_observer_raising_in_before_send_sends_nothing(tmp_path) -> None:
    robot = FakeRobot(traces=[_no_transition_trace(on=False)])
    dataset = FakeDataset()
    observer = CountingObserver(raise_at="before_send")

    with pytest.raises(RuntimeError, match="boom in before_send"):
        _run_record_loop(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)],
            dataset=dataset,
            observer=observer,
        )

    assert len(robot.sent_actions) == 0
    assert dataset.add_frame_calls == 0
    assert observer.calls == [("before_send", 0)]


def test_observer_raising_after_send_success_skips_add_frame_without_resend(tmp_path) -> None:
    robot = FakeRobot(traces=[_no_transition_trace(on=False)])
    dataset = FakeDataset()
    observer = CountingObserver(raise_at="on_send_success")

    with pytest.raises(RuntimeError, match="boom in on_send_success"):
        _run_record_loop(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)],
            dataset=dataset,
            observer=observer,
        )

    # Exactly one send, zero add_frame, no resend.
    assert len(robot.sent_actions) == 1
    assert dataset.add_frame_calls == 0
    assert observer.calls == [("before_send", 0), ("on_send_success", 0)]


def test_observer_raising_after_add_frame_success_never_repeats(tmp_path) -> None:
    robot = FakeRobot(traces=[_no_transition_trace(on=False)])
    dataset = FakeDataset()
    observer = CountingObserver(raise_at="on_add_frame_success")

    with pytest.raises(RuntimeError, match="boom in on_add_frame_success"):
        _run_record_loop(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)],
            dataset=dataset,
            observer=observer,
        )

    # The stage is identifiable from which hook raised; no resend, no repeat.
    assert len(robot.sent_actions) == 1
    assert dataset.add_frame_calls == 1
    assert observer.calls == [
        ("before_send", 0),
        ("on_send_success", 0),
        ("on_add_frame_success", 0),
    ]


def test_observer_sees_send_failure_stage(tmp_path) -> None:
    robot = FakeRobot(send_error=RuntimeError("servo queue full"))
    dataset = FakeDataset()
    observer = CountingObserver()

    with pytest.raises(RuntimeError, match="servo queue full"):
        _run_record_loop(
            robot=robot,
            teleop_actions=[_gripper_action(0.0)],
            dataset=dataset,
            observer=observer,
        )

    assert [stage for stage, _ in observer.calls] == ["before_send", "on_send_failure"]
    assert dataset.add_frame_calls == 0

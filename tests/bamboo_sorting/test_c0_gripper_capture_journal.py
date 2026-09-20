# Copyright 2026 The HuggingFace Inc. All rights reserved.
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

"""Offline unit tests for the R4A in-memory pending capture journal.

These tests use a fake robot object exposing ``last_gripper_command_trace``
and plain dict actions only. No camera, AUBO, DO/IO, motion, real dataset,
network, training, or inference is involved.
"""

import json
import math
import subprocess
import sys

import pytest

from lerobot.bamboo_sorting.c0_gripper_capture_journal import (
    C0_GRIPPER_PENDING_CAPTURE_JOURNAL_SCHEMA_VERSION,
    C0GripperPendingCaptureJournalV1,
    C0PendingCaptureError,
    extract_c0_dataset_gripper_pos,
    resolve_aubo_sent_gripper_command,
)

EPISODE_ID = "c0-pending-journal-episode-00"
FPS = 30.0

FIXED_FALSE_FIELDS = (
    "dataset_episode_durably_saved",
    "dataset_gripper_slice_digest_bound",
    "final_sidecar_published",
    "physical_gripper_feedback_available",
    "physical_grasp_success_proven",
    "training_authorized",
    "policy_execution_authorized",
    "serialized_record_grants_live_authorization",
    "hardware_access_performed_by_serialization",
)


def _success_trace(target_on: bool = True, **changes) -> dict:
    """Mirror AuboI10Robot._set_suction_outputs for a fully successful switch."""

    if target_on:
        trace = {
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
    else:
        trace = {
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
    trace.update(changes)
    return trace


def _failure_trace(**changes) -> dict:
    """Second DO write fails: one recorded write, two attempts, failed latch."""

    trace = {
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
    trace.update(changes)
    return trace


def _no_transition_trace(on: bool = False) -> dict:
    """requested_transition=False freshness evidence (driver hold path)."""

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
    """Duck-typed stand-in exposing only last_gripper_command_trace."""

    def __init__(self, trace):
        self.last_gripper_command_trace = trace


def _journal() -> C0GripperPendingCaptureJournalV1:
    return C0GripperPendingCaptureJournalV1(
        episode_id=EPISODE_ID, episode_index=0, fps=FPS
    )


def _action(gripper: float) -> dict:
    return {"ee.gripper_pos": gripper}


def _run_cycle(
    journal: C0GripperPendingCaptureJournalV1,
    robot: FakeRobot,
    *,
    gripper: float = 100.0,
    candidate_frame_index: int = 0,
    send_error: Exception | None = None,
    add_frame_error: Exception | None = None,
    sent_action: dict | None = None,
) -> None:
    """Drive one journal cycle the same way record_loop drives its observer."""

    action = _action(gripper)
    to_send = action
    journal.before_send(
        robot=robot,
        dataset=None,
        action=action,
        robot_action_to_send=to_send,
        candidate_frame_index=candidate_frame_index,
    )
    if send_error is not None:
        journal.on_send_failure(
            robot=robot,
            dataset=None,
            action=action,
            robot_action_to_send=to_send,
            candidate_frame_index=candidate_frame_index,
            error=send_error,
        )
        return
    journal.on_send_success(
        robot=robot,
        dataset=None,
        action=action,
        robot_action_to_send=to_send,
        sent_action=to_send if sent_action is None else sent_action,
        candidate_frame_index=candidate_frame_index,
    )
    if add_frame_error is not None:
        journal.on_add_frame_failure(
            robot=robot,
            dataset=None,
            action=action,
            robot_action_to_send=to_send,
            candidate_frame_index=candidate_frame_index,
            error=add_frame_error,
        )
    else:
        journal.on_add_frame_success(
            robot=robot,
            dataset=None,
            action=action,
            robot_action_to_send=to_send,
            candidate_frame_index=candidate_frame_index,
        )


# --------------------------------------------------------------- resolution


def test_resolve_sent_gripper_uses_aubo_alias_priority() -> None:
    assert resolve_aubo_sent_gripper_command({"ee.gripper_pos": 100.0}) == 100.0
    assert resolve_aubo_sent_gripper_command({"gripper_pos": 0.0}) == 0.0
    assert resolve_aubo_sent_gripper_command({"gripper.pos": 100.0}) == 100.0
    # All aliases present and agreeing resolve exactly like the AUBO priority
    # chain; conflicting values are rejected instead of silently picked.
    assert (
        resolve_aubo_sent_gripper_command(
            {"gripper.pos": 100.0, "gripper_pos": 100.0, "ee.gripper_pos": 100.0}
        )
        == 100.0
    )


def test_resolve_sent_gripper_agreeing_aliases_are_accepted() -> None:
    assert (
        resolve_aubo_sent_gripper_command(
            {"gripper_pos": 100.0, "ee.gripper_pos": 100.0}
        )
        == 100.0
    )


@pytest.mark.parametrize(
    "action",
    [
        {},  # missing gripper entirely
        {"gripper.pos": True},  # bool
        {"gripper_pos": math.nan},  # NaN
        {"ee.gripper_pos": math.inf},  # Inf
        {"ee.gripper_pos": 50.0},  # non C0 binary value
        {"gripper_pos": 100.0, "ee.gripper_pos": 0.0},  # conflicting aliases
    ],
)
def test_resolve_sent_gripper_fails_closed(action: dict) -> None:
    with pytest.raises(ValueError):
        resolve_aubo_sent_gripper_command(action)


@pytest.mark.parametrize(
    "action",
    [
        {},  # missing official field
        {"ee.gripper_pos": False},
        {"ee.gripper_pos": math.nan},
        {"ee.gripper_pos": math.inf},
        {"ee.gripper_pos": 50.0},
    ],
)
def test_extract_dataset_gripper_fails_closed(action: dict) -> None:
    with pytest.raises(ValueError):
        extract_c0_dataset_gripper_pos(action)


# --------------------------------------------------------- before_send layer


def test_before_send_action_mismatch_aborts_and_marks_journal() -> None:
    journal = _journal()
    action = _action(100.0)
    sent = _action(0.0)

    with pytest.raises(C0PendingCaptureError, match="action mismatch"):
        journal.before_send(
            robot=FakeRobot(None),
            dataset=None,
            action=action,
            robot_action_to_send=sent,
            candidate_frame_index=0,
        )

    (cycle,) = journal.cycles
    assert cycle["failure_stage"] == "before_send"
    assert cycle["error_type"] == "action_mismatch"
    assert cycle["action_consistency"] == "mismatch"
    assert cycle["frame_entered_episode_buffer"] is False
    assert journal.episode_failed is True
    assert journal.finalization_blocked is True

    # A failed episode must not accept further cycles.
    with pytest.raises(C0PendingCaptureError, match="failed closed"):
        journal.before_send(
            robot=FakeRobot(None),
            dataset=None,
            action=action,
            robot_action_to_send=action,
            candidate_frame_index=1,
        )


def test_before_send_rejects_bool_and_float_candidate_index() -> None:
    journal = _journal()
    action = _action(100.0)
    for bad_index in (True, 1.5, -1):
        with pytest.raises(ValueError):
            journal.before_send(
                robot=FakeRobot(None),
                dataset=None,
                action=action,
                robot_action_to_send=action,
                candidate_frame_index=bad_index,
            )


# ---------------------------------------------------------- transition layer


def test_command_on_success_creates_aligned_pending_attempt() -> None:
    journal = _journal()
    _run_cycle(journal, FakeRobot(_success_trace(target_on=True)), gripper=100.0, candidate_frame_index=7)

    (cycle,) = journal.cycles
    assert cycle["candidate_frame_index"] == 7
    assert cycle["dataset_action_gripper_pos"] == 100.0
    assert cycle["sent_action_gripper_command"] == 100.0
    assert cycle["action_consistency"] == "matched"
    assert cycle["send_action_returned"] is True
    assert cycle["requested_transition"] is True
    assert cycle["trace_present"] is True
    assert cycle["frame_entered_episode_buffer"] is True
    assert cycle["dataset_episode_durably_saved"] is False

    (event,) = journal.pending_events
    (attempt,) = journal.controller_attempts
    assert event.event_type == "command_on"
    assert event.frame_index == 7
    assert event.event_index == attempt.attempt_index == 0
    assert attempt.candidate_frame_index == 7
    # R4A never marks an attempt committed: durable save is layer 3 and the
    # add_frame success only means the frame entered the episode buffer.
    assert attempt.dataset_frame_committed is False
    assert attempt.event == event


def test_command_off_success_creates_aligned_pending_attempt() -> None:
    journal = _journal()
    _run_cycle(journal, FakeRobot(_success_trace(target_on=False)), gripper=0.0, candidate_frame_index=3)

    (event,) = journal.pending_events
    assert event.event_type == "command_off"
    assert event.frame_index == 3
    assert journal.controller_attempts[0].dataset_frame_committed is False


def test_no_transition_cycle_keeps_evidence_without_controller_attempt() -> None:
    journal = _journal()
    _run_cycle(
        journal,
        FakeRobot(_no_transition_trace(on=False)),
        gripper=0.0,
        candidate_frame_index=0,
    )

    (cycle,) = journal.cycles
    assert cycle["trace_present"] is True
    assert cycle["requested_transition"] is False
    assert cycle["frame_entered_episode_buffer"] is True
    assert journal.pending_events == ()
    assert journal.controller_attempts == ()


def test_send_success_with_missing_trace_fails_closed() -> None:
    journal = _journal()
    with pytest.raises(C0PendingCaptureError, match="trace is None"):
        _run_cycle(journal, FakeRobot(None), gripper=100.0)

    (cycle,) = journal.cycles
    assert cycle["trace"] is None
    assert cycle["trace_present"] is False
    assert cycle["send_action_returned"] is True
    assert cycle["failure_stage"] == "send_action"
    assert cycle["error_type"] == "trace_missing"
    assert cycle["frame_entered_episode_buffer"] is False
    assert journal.episode_failed is True
    assert journal.pending_events == ()
    assert journal.controller_attempts == ()


def test_send_success_with_partial_hold_trace_fails_closed() -> None:
    journal = _journal()
    partial = {"requested_transition": False}
    with pytest.raises(C0PendingCaptureError, match="exactly the AUBO driver field set"):
        _run_cycle(journal, FakeRobot(partial), gripper=0.0)

    (cycle,) = journal.cycles
    assert cycle["error_type"] == "hold_trace_invalid"
    assert journal.episode_failed is True


@pytest.mark.parametrize(
    ("gripper", "target_on"),
    [
        (100.0, False),  # sent command_on but the latch stays off: contradiction
        (0.0, True),  # sent command_off but the latch stays on: contradiction
    ],
)
def test_hold_trace_target_must_match_sent_gripper(gripper: float, target_on: bool) -> None:
    journal = _journal()
    with pytest.raises(C0PendingCaptureError, match="contradicts the sent gripper"):
        _run_cycle(journal, FakeRobot(_no_transition_trace(on=target_on)), gripper=gripper)

    (cycle,) = journal.cycles
    assert cycle["error_type"] == "hold_trace_invalid"
    assert cycle["frame_entered_episode_buffer"] is False
    assert journal.episode_failed is True
    assert journal.controller_attempts == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"do_writes": [{"pin": 3, "value": False, "return_code": 0}]},
        {"do_write_attempt_count": 1},
        {"do_api_success": True},
        {"do_readback": {"2": False, "3": False}},
        {"do_readback_after_failure": {"2": False, "3": False}},
        {"do_readback_error": "stale"},
        {"controller_output_state_known": True},
        {"controller_output_matches_requested": True},
        {"partial_write_possible": True},
        {"commanded_state_after": True},
        {"commanded_state_before": True},
        {"error": "leftover error"},
        {"requested_do": {"2": True, "3": False}},
        {"do_readback_attempts": 1},  # unknown field in the hold path
    ],
)
def test_hold_trace_field_contradictions_fail_closed(changes: dict) -> None:
    journal = _journal()
    trace = _no_transition_trace(on=False)
    trace.update(changes)
    with pytest.raises(C0PendingCaptureError, match="hold trace"):
        _run_cycle(journal, FakeRobot(trace), gripper=0.0)

    (cycle,) = journal.cycles
    assert cycle["error_type"] == "hold_trace_invalid"
    assert cycle["frame_entered_episode_buffer"] is False
    assert journal.episode_failed is True
    assert journal.controller_attempts == ()


@pytest.mark.parametrize(
    "trace",
    [
        {"do_api_success": math.nan, "requested_transition": True},
        {"do_api_success": True, "requested_transition": "yes"},
    ],
)
def test_on_send_success_fails_closed_on_impure_or_invalid_trace(trace: dict) -> None:
    journal = _journal()
    action = _action(100.0)
    journal.before_send(
        robot=FakeRobot(None),
        dataset=None,
        action=action,
        robot_action_to_send=action,
        candidate_frame_index=0,
    )
    with pytest.raises(C0PendingCaptureError):
        journal.on_send_success(
            robot=FakeRobot(trace),
            dataset=None,
            action=action,
            robot_action_to_send=action,
            sent_action=action,
            candidate_frame_index=0,
        )
    assert journal.episode_failed is True
    assert journal.pending_events == ()


def test_post_send_action_contradiction_fails_closed() -> None:
    journal = _journal()
    action = _action(100.0)
    journal.before_send(
        robot=FakeRobot(None),
        dataset=None,
        action=action,
        robot_action_to_send=action,
        candidate_frame_index=0,
    )
    contradictory_returned = _action(0.0)
    with pytest.raises(C0PendingCaptureError, match="returned action gripper contradicts"):
        journal.on_send_success(
            robot=FakeRobot(_success_trace(target_on=True)),
            dataset=None,
            action=action,
            robot_action_to_send=action,
            sent_action=contradictory_returned,
            candidate_frame_index=0,
        )
    (cycle,) = journal.cycles
    assert cycle["error_type"] == "sent_action_mismatch"
    assert cycle["failure_stage"] == "send_action"
    assert cycle["frame_entered_episode_buffer"] is False
    assert journal.episode_failed is True
    assert journal.pending_events == ()


# ------------------------------------------------------------- failure layer


def test_send_failure_preserves_failed_trace_as_uncommitted_attempt() -> None:
    journal = _journal()
    _run_cycle(
        journal,
        FakeRobot(_failure_trace()),
        gripper=100.0,
        candidate_frame_index=5,
        send_error=RuntimeError("夹爪 DO 状态切换失败"),
    )

    (cycle,) = journal.cycles
    assert cycle["send_action_returned"] is False
    assert cycle["failure_stage"] == "send_action"
    assert cycle["error_type"] == "RuntimeError"
    assert cycle["frame_entered_episode_buffer"] is False
    assert cycle["trace"]["do_api_success"] is False
    assert journal.episode_failed is True

    (attempt,) = journal.controller_attempts
    assert attempt.dataset_frame_committed is False
    assert attempt.event.do_api_success is False
    assert attempt.event.error


def test_send_failure_with_none_trace_never_reuses_previous_trace() -> None:
    journal = _journal()
    _run_cycle(
        journal,
        FakeRobot(None),
        gripper=100.0,
        candidate_frame_index=0,
        send_error=RuntimeError("设置速度比例失败"),
    )

    (cycle,) = journal.cycles
    assert cycle["trace"] is None
    assert cycle["trace_present"] is False
    assert cycle["requested_transition"] is None
    assert journal.pending_events == ()
    assert journal.controller_attempts == ()


def test_send_failure_with_impure_trace_drops_trace_but_keeps_error() -> None:
    journal = _journal()
    impure = _no_transition_trace()
    impure["do_api_success"] = math.nan
    _run_cycle(
        journal,
        FakeRobot(impure),
        gripper=0.0,
        candidate_frame_index=0,
        send_error=RuntimeError("io_control is None"),
    )

    (cycle,) = journal.cycles
    assert cycle["trace"] is None
    assert "trace dropped" in cycle["error_message"]


def test_add_frame_failure_keeps_buffered_false_even_after_partial_mutation() -> None:
    journal = _journal()
    _run_cycle(
        journal,
        FakeRobot(_success_trace(target_on=True)),
        gripper=100.0,
        candidate_frame_index=0,
        add_frame_error=RuntimeError("episode_buffer flush failed"),
    )

    (cycle,) = journal.cycles
    assert cycle["send_action_returned"] is True
    assert cycle["failure_stage"] == "add_frame"
    assert cycle["frame_entered_episode_buffer"] is False
    assert cycle["dataset_episode_durably_saved"] is False
    assert journal.episode_failed is True
    # The attempt stays uncommitted: the frame never entered the buffer.
    assert journal.controller_attempts[0].dataset_frame_committed is False


# ----------------------------------------------------------------- isolation


def test_trace_is_deep_copied_and_later_robot_mutation_is_invisible() -> None:
    journal = _journal()
    robot = FakeRobot(_success_trace(target_on=True))
    _run_cycle(journal, robot, gripper=100.0, candidate_frame_index=0)

    # The robot overwrites and mutates its trace after the capture.
    robot.last_gripper_command_trace["do_writes"][0]["return_code"] = -99
    robot.last_gripper_command_trace = None

    (cycle,) = journal.cycles
    assert cycle["trace"]["do_writes"][0]["return_code"] == 0
    assert cycle["trace"]["do_api_success"] is True


def test_cycles_property_returns_detached_copies() -> None:
    journal = _journal()
    _run_cycle(journal, FakeRobot(_success_trace()), gripper=100.0)

    (cycle,) = journal.cycles
    cycle["trace"]["do_writes"][0]["return_code"] = -99
    cycle["frame_entered_episode_buffer"] = False

    (fresh,) = journal.cycles
    assert fresh["trace"]["do_writes"][0]["return_code"] == 0
    assert fresh["frame_entered_episode_buffer"] is True


# ------------------------------------------------------------------ lifecycle


def test_rerecord_abandons_journal_and_forks_isolated_generation() -> None:
    journal = _journal()
    _run_cycle(journal, FakeRobot(_success_trace()), gripper=100.0, candidate_frame_index=0)
    assert len(journal.controller_attempts) == 1

    journal.mark_abandoned(reason="operator requested rerecord")
    assert journal.status == "abandoned_rerecorded"
    with pytest.raises(C0PendingCaptureError, match="abandoned"):
        journal.before_send(
            robot=FakeRobot(None),
            dataset=None,
            action=_action(100.0),
            robot_action_to_send=_action(100.0),
            candidate_frame_index=1,
        )

    nxt = journal.fork_for_rerecord(episode_id=EPISODE_ID, episode_index=0)
    assert nxt is not journal
    assert nxt.cycles == ()
    assert nxt.pending_events == ()
    assert nxt.controller_attempts == ()
    assert nxt.status == "recording"


def test_fork_for_rerecord_requires_abandoned_journal() -> None:
    journal = _journal()
    with pytest.raises(C0PendingCaptureError, match="abandoned"):
        journal.fork_for_rerecord(episode_id=EPISODE_ID, episode_index=0)


def test_stop_marks_journal_incomplete_and_never_finalized() -> None:
    journal = _journal()
    _run_cycle(journal, FakeRobot(_success_trace()), gripper=100.0)
    journal.mark_incomplete(reason="operator stopped recording early")

    assert journal.status == "incomplete"
    assert journal.finalized is False
    record = journal.to_manifest_record()
    assert record["status"] == "incomplete"
    assert record["incomplete_reason"] == "operator stopped recording early"


# ------------------------------------------------------------- serialization


def test_manifest_record_is_deterministic_pure_json_with_fixed_false_fields() -> None:
    journal = _journal()
    _run_cycle(journal, FakeRobot(_success_trace()), gripper=100.0, candidate_frame_index=2)
    _run_cycle(
        journal,
        FakeRobot(_no_transition_trace(on=True)),
        gripper=100.0,
        candidate_frame_index=3,
    )

    first = journal.to_manifest_record()
    second = journal.to_manifest_record()
    assert first == second
    assert first["schema_version"] == C0_GRIPPER_PENDING_CAPTURE_JOURNAL_SCHEMA_VERSION
    encoded = json.dumps(first, sort_keys=True, allow_nan=False)
    assert json.loads(encoded) == first

    for field in FIXED_FALSE_FIELDS:
        assert first[field] is False, field
    assert first["finalized"] is False
    assert "dataset_gripper_slice_sha256" not in json.dumps(first)
    assert "C0GripperControllerSidecarV1" not in json.dumps(
        {key: value for key, value in first.items() if key != "schema_version"}
    )

    # R4A recursion guard: no pending record anywhere may claim the frame was
    # committed to the dataset; add_frame only entered the episode buffer.
    def _contains_committed_true(node) -> bool:
        if isinstance(node, dict):
            if node.get("dataset_frame_committed") is True:
                return True
            return any(_contains_committed_true(value) for value in node.values())
        if isinstance(node, list):
            return any(_contains_committed_true(item) for item in node)
        return False

    assert not _contains_committed_true(first)

    # The returned record is detached: mutating it cannot rewrite the journal.
    first["cycles"][0]["trace"]["do_api_success"] = False
    first["training_authorized"] = True
    third = journal.to_manifest_record()
    assert third["cycles"][0]["trace"]["do_api_success"] is True
    assert third["training_authorized"] is False


def test_pending_events_and_attempts_properties_return_deep_copies() -> None:
    journal = _journal()
    _run_cycle(journal, FakeRobot(_success_trace()), gripper=100.0, candidate_frame_index=0)

    (event,) = journal.pending_events
    (attempt,) = journal.controller_attempts
    # Mutate the returned nested objects in every possible way.
    event.requested_do["2"] = False
    event.do_writes[0]["return_code"] = -99
    attempt.source_trace["do_api_success"] = False
    attempt.source_trace["do_writes"][1]["return_code"] = -99
    attempt.event.requested_do["2"] = False

    # Re-reading the journal shows unchanged internal values.
    (fresh_event,) = journal.pending_events
    (fresh_attempt,) = journal.controller_attempts
    assert fresh_event.requested_do == {"2": True, "3": False}
    assert fresh_event.do_writes[0]["return_code"] == 0
    assert fresh_attempt.source_trace["do_api_success"] is True
    assert fresh_attempt.source_trace["do_writes"][1]["return_code"] == 0
    assert fresh_attempt.event.requested_do == {"2": True, "3": False}
    assert fresh_attempt.dataset_frame_committed is False

    # Serialization still succeeds and content is unchanged.
    before = journal.to_manifest_record()
    assert before["controller_attempts"][0]["source_trace"]["do_api_success"] is True
    assert before["controller_attempts"][0]["event"]["requested_do"] == {"2": True, "3": False}


def test_serialization_fails_closed_on_non_json_sdk_object_in_trace() -> None:
    journal = _journal()
    robot = FakeRobot(_success_trace())
    _run_cycle(journal, robot, gripper=100.0)
    # Simulate a post-hoc corruption with an SDK-like object sneaking in.
    journal._cycles[0]["trace"]["sdk_handle"] = object()
    with pytest.raises(ValueError, match="pure JSON"):
        journal.to_manifest_record()


def test_constructor_rejects_non_pure_identity_fields() -> None:
    with pytest.raises(ValueError):
        C0GripperPendingCaptureJournalV1(episode_id=EPISODE_ID, episode_index=True, fps=FPS)
    with pytest.raises(ValueError):
        C0GripperPendingCaptureJournalV1(episode_id="", episode_index=0, fps=FPS)
    with pytest.raises(ValueError):
        C0GripperPendingCaptureJournalV1(episode_id=EPISODE_ID, episode_index=0, fps=math.nan)


# ------------------------------------------------------ import side-effect probe

_IMPORT_PROBE = r"""
import importlib.util
import json
import os
import sys
import threading
import types

def _stub_package(name):
    spec = importlib.util.find_spec(name)
    module = types.ModuleType(name)
    module.__path__ = list(spec.submodule_search_locations)
    sys.modules[name] = module

_stub_package("lerobot")
_stub_package("lerobot.bamboo_sorting")

baseline_modules = set(sys.modules)

import lerobot.bamboo_sorting.c0_gripper_capture_journal  # noqa: F401

forbidden_roots = {"cv2", "pyaubo_sdk", "torch"}
new_modules = sorted(set(sys.modules) - baseline_modules)
forbidden_modules = [name for name in new_modules if name.split(".")[0] in forbidden_roots]
extra_threads = [
    thread.name for thread in threading.enumerate() if thread is not threading.main_thread()
]
sockets = []
for fd in os.listdir("/proc/self/fd"):
    path = f"/proc/self/fd/{fd}"
    if os.path.islink(path) and "socket" in os.readlink(path):
        sockets.append(fd)
print(
    json.dumps(
        {
            "forbidden_modules": forbidden_modules,
            "extra_threads": extra_threads,
            "sockets": sockets,
            "package_init_executed": "lerobot.bamboo_sorting.smolvla_adapter" in sys.modules,
            "module_loaded": "lerobot.bamboo_sorting.c0_gripper_capture_journal" in sys.modules,
        }
    )
)
"""


def test_importing_journal_module_touches_no_hardware_network_threads_or_files(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
        cwd=tmp_path,
    )
    state = json.loads(result.stdout.strip().splitlines()[-1])

    assert state["module_loaded"] is True
    assert state["package_init_executed"] is False
    assert state["forbidden_modules"] == []
    assert state["extra_threads"] == []
    assert state["sockets"] == []

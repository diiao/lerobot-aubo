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

import json

import pytest

from lerobot.bamboo_sorting.c0_gripper_auditor import (
    C0_GRIPPER_AUDIT_REPORT_SCHEMA_VERSION,
    C0GripperAuditReportV1,
    audit_gripper_episode,
)
from lerobot.bamboo_sorting.c0_gripper_contract import (
    C0_GRIPPER_EPISODE_TRACE_SCHEMA_VERSION,
    C0_GRIPPER_EVENT_SCHEMA_VERSION,
    C0GripperEpisodeTraceV1,
    C0GripperEventV1,
)

EPISODE_ID = "c0-gripper-audit-00"
FPS = 25.0


def _event(
    event_index: int,
    frame_index: int,
    event_type: str,
    *,
    latch_before: float,
    failed: bool = False,
    readback_mismatch: bool = False,
    readback_supported: bool = True,
) -> C0GripperEventV1:
    """Build one event consistent with the execution-layer latch semantics.

    ``latch_before`` is the actual software latch before the request. A failed
    event keeps ``commanded_state_after == latch_before`` and preserves its
    failure evidence; a successful one moves the latch to the requested state.
    """

    if event_type == "command_on":
        requested, action_value = 100.0, 100.0
        requested_do = {"2": True, "3": False}
        do_writes = [
            {"pin": 3, "value": False, "return_code": 0},
            {"pin": 2, "value": True, "return_code": 0},
        ]
        matching_readback = {"2": True, "3": False}
        mismatching_readback = {"2": False, "3": False}
    else:
        requested, action_value = 0.0, 0.0
        requested_do = {"2": False, "3": True}
        do_writes = [
            {"pin": 2, "value": False, "return_code": 0},
            {"pin": 3, "value": True, "return_code": 0},
        ]
        matching_readback = {"2": False, "3": True}
        mismatching_readback = {"2": True, "3": False}

    do_readback = mismatching_readback if readback_mismatch else matching_readback
    if not readback_supported:
        do_readback = None
    succeeded = not failed and not readback_mismatch
    return C0GripperEventV1(
        schema_version=C0_GRIPPER_EVENT_SCHEMA_VERSION,
        episode_id=EPISODE_ID,
        event_index=event_index,
        frame_index=frame_index,
        sync_timestamp_s=frame_index / FPS,
        event_type=event_type,
        action_gripper_pos=action_value,
        requested_state_after=requested,
        commanded_state_before=latch_before,
        commanded_state_after=latch_before if not succeeded else requested,
        transition_requested=True,
        requested_do=requested_do,
        do_writes=do_writes,
        do_write_attempt_count=len(do_writes),
        do_api_success=succeeded,
        do_readback_supported=readback_supported,
        do_readback=do_readback,
        do_readback_after_failure=None if succeeded else do_readback,
        do_readback_error=None,
        controller_output_state_known=readback_supported,
        controller_output_matches_requested=(
            None if not readback_supported else not readback_mismatch
        ),
        partial_write_possible=not succeeded,
        error=None if succeeded else "controller DO write or readback failed",
    )


def _build_trace(
    action: list[float],
    events: list[C0GripperEventV1],
    *,
    frame_count: int = 100,
    lift_start: int | None = 45,
    placement_complete: int | None = 78,
    human_outcome: str = "success",
) -> C0GripperEpisodeTraceV1:
    """Assemble a trace, simulating the observation latch chain from events."""

    events_by_frame = {event.frame_index: event for event in events}
    observation = [0.0]
    latch = 0.0
    for index in range(1, frame_count):
        previous = index - 1
        if action[previous] != latch:
            latch = events_by_frame[previous].commanded_state_after
        observation.append(latch)
    return C0GripperEpisodeTraceV1(
        schema_version=C0_GRIPPER_EPISODE_TRACE_SCHEMA_VERSION,
        episode_id=EPISODE_ID,
        fps=FPS,
        frame_count=frame_count,
        action_gripper_values=action,
        observation_gripper_values=observation,
        events=tuple(events),
        lift_start_frame_index=lift_start,
        placement_complete_frame_index=placement_complete,
        human_outcome=human_outcome,
        synthetic=True,
        finalized=True,
    )


def _switched_action(switches: tuple[tuple[int, str], ...], frame_count: int = 100) -> list[float]:
    """Build the requested action trajectory from (frame, event_type) switches."""

    action = [0.0] * frame_count
    for frame, event_type in switches:
        value = 100.0 if event_type == "command_on" else 0.0
        for index in range(frame, frame_count):
            action[index] = value
    return action


def _successful_events(action: list[float]) -> list[C0GripperEventV1]:
    """Create one successful event per requested transition frame."""

    events: list[C0GripperEventV1] = []
    latch = 0.0
    for frame, value in enumerate(action):
        if value == latch:
            continue
        event_type = "command_on" if value == 100.0 else "command_off"
        events.append(
            _event(len(events), frame, event_type, latch_before=latch)
        )
        latch = value
    return events


def _valid_trace(**changes) -> C0GripperEpisodeTraceV1:
    action = _switched_action(((40, "command_on"), (80, "command_off")))
    events = _successful_events(action)
    return _build_trace(action, events, **changes)


def test_valid_single_strip_passes_structural_audit() -> None:
    report = audit_gripper_episode(_valid_trace())

    assert report.schema_version == C0_GRIPPER_AUDIT_REPORT_SCHEMA_VERSION
    assert report.episode_id == EPISODE_ID
    assert report.errors == ()
    assert report.blockers == ()
    assert report.derived_on_frames == (40,)
    assert report.derived_off_frames == (80,)
    assert report.event_count == 2
    assert report.gripper_sequence_valid_for_c0_review is True

    record = report.to_manifest_record()
    assert record["synthetic"] is True
    assert record["training_authorized"] is False
    assert record["policy_execution_authorized"] is False
    assert record["serialized_record_grants_live_authorization"] is False
    assert record["hardware_access_performed_by_serialization"] is False
    assert record["physical_gripper_feedback_available"] is False
    assert record["physical_grasp_success_proven"] is False
    assert "data_ready_for_training" not in record
    json.dumps(record, allow_nan=False)


def test_placement_complete_may_share_command_off_frame() -> None:
    report = audit_gripper_episode(_valid_trace(placement_complete=80))
    assert report.errors == ()
    assert report.gripper_sequence_valid_for_c0_review is True


def test_missing_command_on_rejected() -> None:
    report = audit_gripper_episode(
        _build_trace(
            [0.0] * 100,
            [],
            lift_start=None,
            placement_complete=None,
            human_outcome="failure",
        )
    )
    assert "missing_command_on" in report.errors
    assert "missing_command_off" in report.errors
    assert report.gripper_sequence_valid_for_c0_review is False


def test_missing_command_off_rejected() -> None:
    action = _switched_action(((40, "command_on"),))
    report = audit_gripper_episode(
        _build_trace(action, _successful_events(action), human_outcome="failure")
    )
    assert "missing_command_off" in report.errors
    assert "action_must_end_released" in report.errors
    assert report.gripper_sequence_valid_for_c0_review is False


def test_chatter_rejected() -> None:
    action = _switched_action(
        ((20, "command_on"), (30, "command_off"), (40, "command_on"), (50, "command_off"))
    )
    report = audit_gripper_episode(
        _build_trace(action, _successful_events(action), human_outcome="failure")
    )
    assert "multiple_command_on" in report.errors
    assert "multiple_command_off" in report.errors
    assert report.gripper_sequence_valid_for_c0_review is False


def test_do_readback_mismatch_rejected() -> None:
    # The command_on request fails on readback mismatch: the latch stays
    # released, the operator stops requesting (action returns to 0), and the
    # episode must be rejected on the DO evidence, not rewritten as success.
    action = [0.0] * 100
    action[40] = 100.0  # single-frame on request at frame 40
    failed_on = _event(0, 40, "command_on", latch_before=0.0, readback_mismatch=True)
    report = audit_gripper_episode(
        _build_trace(action, [failed_on], human_outcome="failure")
    )
    assert "event_0000:do_api_failure" in report.errors
    assert "event_0000:readback_mismatch" in report.errors
    assert "missing_command_off" in report.errors
    assert report.gripper_sequence_valid_for_c0_review is False


def test_do_write_failure_without_readback_rejected() -> None:
    action = [0.0] * 100
    action[40] = 100.0  # single-frame on request at frame 40
    failed_on = _event(
        0, 40, "command_on", latch_before=0.0, failed=True, readback_supported=False
    )
    report = audit_gripper_episode(
        _build_trace(action, [failed_on], human_outcome="failure")
    )
    assert "event_0000:do_api_failure" in report.errors
    assert not any("readback_mismatch" in error for error in report.errors)
    assert report.gripper_sequence_valid_for_c0_review is False


def test_failed_command_off_keeps_latch_and_rejects() -> None:
    # on@40 succeeds; off@80 fails so the latch stays commanded-on; the
    # operator stops requesting, and the episode ends still commanded on.
    action = _switched_action(((40, "command_on"),))
    action[80] = 0.0  # single-frame release request at frame 80
    events = _successful_events(action[:80])
    failed_off = _event(1, 80, "command_off", latch_before=100.0, failed=True)
    report = audit_gripper_episode(
        _build_trace(action, [*events, failed_off], human_outcome="failure")
    )
    assert "event_0001:do_api_failure" in report.errors
    assert "action_must_end_released" in report.errors
    assert report.gripper_sequence_valid_for_c0_review is False


def test_lift_before_command_on_rejected() -> None:
    report = audit_gripper_episode(_valid_trace(lift_start=35))
    assert "lift_start_before_command_on" in report.errors
    assert report.gripper_sequence_valid_for_c0_review is False


def test_command_off_before_placement_complete_rejected() -> None:
    report = audit_gripper_episode(_valid_trace(placement_complete=85))
    assert "command_off_before_placement_complete" in report.errors
    assert report.gripper_sequence_valid_for_c0_review is False


@pytest.mark.parametrize("human_outcome", ["failure", "uncertain"])
def test_non_success_outcomes_audited_but_never_marked_successful(human_outcome: str) -> None:
    report = audit_gripper_episode(_valid_trace(human_outcome=human_outcome))

    assert report.errors == ()
    assert report.blockers == ("human_outcome_not_success",)
    assert report.gripper_sequence_valid_for_c0_review is False
    record = report.to_manifest_record()
    assert record["human_outcome"] == human_outcome
    assert record["gripper_sequence_valid_for_c0_review"] is False


def test_audit_is_deterministic_and_does_not_modify_input() -> None:
    trace = _valid_trace()
    before = trace.to_manifest_record()

    first = audit_gripper_episode(trace)
    second = audit_gripper_episode(trace)

    assert first == second
    assert first.to_manifest_record() == second.to_manifest_record()
    assert trace.to_manifest_record() == before


def test_report_results_are_computed_not_supplied() -> None:
    # The report constructor accepts only the trace; callers cannot pass or
    # forge errors, blockers, derived counts, or the validity flag.
    action = _switched_action(
        ((20, "command_on"), (30, "command_off"), (40, "command_on"), (50, "command_off"))
    )
    chatter_trace = _build_trace(action, _successful_events(action), human_outcome="failure")

    with pytest.raises(TypeError):
        C0GripperAuditReportV1(
            schema_version=C0_GRIPPER_AUDIT_REPORT_SCHEMA_VERSION,
            trace=chatter_trace,
            errors=(),
            blockers=(),
        )
    with pytest.raises(TypeError):
        C0GripperAuditReportV1(
            schema_version=C0_GRIPPER_AUDIT_REPORT_SCHEMA_VERSION,
            trace=chatter_trace,
            derived_on_frames=(1, 2, 3),
        )
    with pytest.raises(TypeError):
        C0GripperAuditReportV1(
            schema_version=C0_GRIPPER_AUDIT_REPORT_SCHEMA_VERSION,
            trace=chatter_trace,
            gripper_sequence_valid_for_c0_review=True,
        )

    report = C0GripperAuditReportV1(
        schema_version=C0_GRIPPER_AUDIT_REPORT_SCHEMA_VERSION,
        trace=chatter_trace,
    )
    assert "multiple_command_on" in report.errors
    assert "multiple_command_off" in report.errors
    assert report.gripper_sequence_valid_for_c0_review is False


def test_report_validity_is_derived_from_trace_only() -> None:
    report = audit_gripper_episode(
        _build_trace(
            [0.0] * 100, [], lift_start=None, placement_complete=None, human_outcome="failure"
        )
    )
    assert report.gripper_sequence_valid_for_c0_review is False
    record = report.to_manifest_record()
    assert record["errors"] == sorted(record["errors"])
    assert record["training_authorized"] is False
    assert record["policy_execution_authorized"] is False

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

"""Suction trace adapters and per-command integrity used by current capture."""

import copy
import json

import pytest

from lerobot.bamboo_sorting.c0_gripper_contract import C0_GRIPPER_EVENT_SCHEMA_VERSION
from lerobot.bamboo_sorting.c0_gripper_controller_sidecar import (
    C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION,
    C0GripperControllerAttemptV1,
    build_c0_gripper_event_from_aubo_trace,
    validate_aubo_gripper_transition_trace,
    verify_c0_gripper_controller_attempt_integrity,
)

EPISODE_ID = "c0-controller-sidecar-episode-00"

FPS = 25.0

ON_FRAME = 10

OFF_FRAME = 50


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


def _return_code_failure_trace(**changes) -> dict:
    """First DO write returns a non-zero code; SDK itself did not raise."""

    trace = {
        "requested_transition": True,
        "target_suction_on": True,
        "requested_do": {"2": True, "3": False},
        "do_writes": [{"pin": 3, "value": False, "return_code": -1}],
        "do_write_attempt_count": 1,
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
        "error": "setStandardDigitalOutput(pin=3, value=False) returned -1",
    }
    trace.update(changes)
    return trace


def _first_call_exception_trace(**changes) -> dict:
    """The first setStandardDigitalOutput call raised: no write recorded."""

    trace = {
        "requested_transition": True,
        "target_suction_on": True,
        "requested_do": {"2": True, "3": False},
        "do_writes": [],
        "do_write_attempt_count": 1,
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
        "error": "sdk connection lost",
    }
    trace.update(changes)
    return trace


def _second_call_exception_trace(**changes) -> dict:
    """The second setStandardDigitalOutput call raised: one write recorded."""

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
        "error": "sdk timeout on second write",
    }
    trace.update(changes)
    return trace


def _readback_mismatch_trace(**changes) -> dict:
    """Both DO writes succeeded but the readback never matched the request."""

    trace = {
        "requested_transition": True,
        "target_suction_on": True,
        "requested_do": {"2": True, "3": False},
        "do_writes": [
            {"pin": 3, "value": False, "return_code": 0},
            {"pin": 2, "value": True, "return_code": 0},
        ],
        "do_write_attempt_count": 2,
        "do_api_success": False,
        "do_readback_supported": True,
        "do_readback": {"2": False, "3": False},
        "do_readback_after_failure": {"2": False, "3": False},
        "do_readback_error": None,
        "controller_output_state_known": True,
        "controller_output_matches_requested": False,
        "partial_write_possible": True,
        "commanded_state_before": False,
        "commanded_state_after": False,
        "error": "controller DO readback mismatch: requested={'2': True, '3': False}",
        "do_readback_attempts": 8,
    }
    trace.update(changes)
    return trace


def _readback_exception_trace(**changes) -> dict:
    """Both DO writes succeeded but the readback API raised every time."""

    trace = {
        "requested_transition": True,
        "target_suction_on": True,
        "requested_do": {"2": True, "3": False},
        "do_writes": [
            {"pin": 3, "value": False, "return_code": 0},
            {"pin": 2, "value": True, "return_code": 0},
        ],
        "do_write_attempt_count": 2,
        "do_api_success": False,
        "do_readback_supported": True,
        "do_readback": None,
        "do_readback_after_failure": None,
        "do_readback_error": "getStandardDigitalOutput raised: io offline",
        "controller_output_state_known": False,
        "controller_output_matches_requested": None,
        "partial_write_possible": True,
        "commanded_state_before": False,
        "commanded_state_after": False,
        "error": "getStandardDigitalOutput raised: io offline",
    }
    trace.update(changes)
    return trace


def _adapt(
    trace: dict,
    *,
    episode_id: str = EPISODE_ID,
    event_index: int = 0,
    frame_index: int = ON_FRAME,
    dataset_timestamp_s: float | None = None,
    action_gripper_pos: float | None = None,
):
    if action_gripper_pos is None:
        action_gripper_pos = 100.0 if trace["target_suction_on"] else 0.0
    if dataset_timestamp_s is None:
        dataset_timestamp_s = frame_index / FPS
    return build_c0_gripper_event_from_aubo_trace(
        episode_id=episode_id,
        event_index=event_index,
        candidate_frame_index=frame_index,
        dataset_timestamp_s=dataset_timestamp_s,
        action_gripper_pos=action_gripper_pos,
        source_trace=trace,
    )


def _attempt(
    trace: dict | None = None,
    *,
    target_suction_on: bool = True,
    attempt_index: int = 0,
    event_index: int | None = None,
    frame_index: int = ON_FRAME,
    committed: bool = True,
    episode_id: str = EPISODE_ID,
    dataset_timestamp_s: float | None = None,
) -> C0GripperControllerAttemptV1:
    if trace is None:
        trace = _success_trace(target_suction_on)
    event = _adapt(
        trace,
        episode_id=episode_id,
        event_index=event_index if event_index is not None else attempt_index,
        frame_index=frame_index,
        dataset_timestamp_s=dataset_timestamp_s,
    )
    return C0GripperControllerAttemptV1(
        schema_version=C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION,
        attempt_index=attempt_index,
        candidate_frame_index=frame_index,
        dataset_frame_committed=committed,
        event=event,
        source_trace=trace,
    )


def test_adapter_command_on_success_full_writes_and_matching_readback() -> None:
    event = _adapt(_success_trace(True), frame_index=ON_FRAME)

    assert event.schema_version == C0_GRIPPER_EVENT_SCHEMA_VERSION
    assert event.event_type == "command_on"
    assert event.requested_state_after == 100.0
    assert event.action_gripper_pos == 100.0
    assert event.commanded_state_before == 0.0
    assert event.commanded_state_after == 100.0
    assert event.frame_index == ON_FRAME
    assert event.sync_timestamp_s == ON_FRAME / FPS
    assert event.requested_do == {"2": True, "3": False}
    assert event.do_writes == (
        {"pin": 3, "value": False, "return_code": 0},
        {"pin": 2, "value": True, "return_code": 0},
    )
    assert event.do_write_attempt_count == 2
    assert event.do_api_success is True
    assert event.do_readback == {"2": True, "3": False}
    assert event.controller_output_state_known is True
    assert event.controller_output_matches_requested is True
    assert event.partial_write_possible is False
    assert event.error is None


def test_adapter_command_off_success() -> None:
    event = _adapt(_success_trace(False), frame_index=OFF_FRAME)

    assert event.event_type == "command_off"
    assert event.requested_state_after == 0.0
    assert event.action_gripper_pos == 0.0
    assert event.commanded_state_before == 100.0
    assert event.commanded_state_after == 0.0
    assert event.requested_do == {"2": False, "3": True}
    assert event.do_writes == (
        {"pin": 2, "value": False, "return_code": 0},
        {"pin": 3, "value": True, "return_code": 0},
    )
    assert event.do_readback == {"2": False, "3": True}
    assert event.do_api_success is True


def test_adapter_preserves_nonzero_return_code_failure() -> None:
    event = _adapt(_return_code_failure_trace())

    assert event.do_api_success is False
    assert event.do_writes == ({"pin": 3, "value": False, "return_code": -1},)
    assert event.do_write_attempt_count == 1
    assert event.error == "setStandardDigitalOutput(pin=3, value=False) returned -1"
    assert event.partial_write_possible is True
    # Failed event keeps the latch: after == before, never rewritten to success.
    assert event.commanded_state_before == 0.0
    assert event.commanded_state_after == 0.0
    assert event.do_readback_after_failure == {"2": False, "3": False}
    assert event.controller_output_matches_requested is False


def test_adapter_first_sdk_call_exception_keeps_attempt_semantics() -> None:
    event = _adapt(_first_call_exception_trace())

    assert event.do_api_success is False
    assert event.do_writes == ()
    assert event.do_write_attempt_count == 1  # len(do_writes) + 1
    assert event.partial_write_possible is True
    assert event.error == "sdk connection lost"
    assert event.commanded_state_after == event.commanded_state_before


def test_adapter_second_sdk_call_exception_keeps_attempt_semantics() -> None:
    event = _adapt(_second_call_exception_trace())

    assert event.do_api_success is False
    assert event.do_writes == ({"pin": 3, "value": False, "return_code": 0},)
    assert event.do_write_attempt_count == 2  # len(do_writes) + 1
    assert event.partial_write_possible is True
    assert event.error == "sdk timeout on second write"


def test_adapter_readback_mismatch_preserved_as_failure() -> None:
    event = _adapt(_readback_mismatch_trace())

    assert event.do_api_success is False
    assert event.do_readback == {"2": False, "3": False}
    assert event.do_readback_after_failure == {"2": False, "3": False}
    assert event.controller_output_state_known is True
    assert event.controller_output_matches_requested is False
    assert "readback mismatch" in event.error
    assert event.commanded_state_after == event.commanded_state_before


def test_adapter_readback_api_exception_preserved() -> None:
    event = _adapt(_readback_exception_trace())

    assert event.do_api_success is False
    assert event.do_readback is None
    assert event.do_readback_after_failure is None
    assert event.do_readback_error == "getStandardDigitalOutput raised: io offline"
    assert event.controller_output_state_known is False
    assert event.controller_output_matches_requested is None


def test_adapter_rejects_requested_transition_false() -> None:
    trace = _success_trace(True, requested_transition=False)
    with pytest.raises(ValueError, match="requested_transition"):
        _adapt(trace)
    with pytest.raises(ValueError, match="requested_transition"):
        validate_aubo_gripper_transition_trace(trace)


def test_adapter_rejects_action_contradicting_target() -> None:
    with pytest.raises(ValueError, match="contradicts"):
        _adapt(_success_trace(True), action_gripper_pos=0.0)
    with pytest.raises(ValueError, match="contradicts"):
        _adapt(_success_trace(False), action_gripper_pos=100.0)


def test_adapter_rejects_missing_field() -> None:
    trace = _success_trace(True)
    del trace["do_writes"]
    with pytest.raises(ValueError, match="missing required fields"):
        _adapt(trace)


def test_adapter_rejects_wrong_field_type() -> None:
    with pytest.raises(ValueError, match="must be bool"):
        _adapt(_success_trace(True, target_suction_on="yes"))
    with pytest.raises(ValueError, match="non-negative integer"):
        _adapt(_success_trace(True, do_write_attempt_count=2.0))
    with pytest.raises(ValueError, match="must be a list"):
        _adapt(_success_trace(True, do_writes={"pin": 3}))
    with pytest.raises(ValueError, match="pure JSON"):
        _adapt(_success_trace(True, error=float("nan")))


def test_adapter_rejects_unknown_field_fail_closed() -> None:
    trace = _success_trace(True, vendor_extra_debug={"x": 1})
    with pytest.raises(ValueError, match="unknown fields"):
        _adapt(trace)


def test_adapter_rejects_malformed_write_records() -> None:
    with pytest.raises(ValueError, match="exactly"):
        _adapt(_success_trace(True, do_writes=[{"pin": 3, "value": False}]))
    with pytest.raises(ValueError, match=r"\['pin'\] must be an integer"):
        _adapt(
            _success_trace(
                True,
                do_writes=[
                    {"pin": True, "value": False, "return_code": 0},
                    {"pin": 2, "value": True, "return_code": 0},
                ],
            )
        )


def test_adapter_do_readback_attempts_valid_values() -> None:
    event_one = _adapt(_success_trace(True, do_readback_attempts=1))
    event_many = _adapt(_readback_mismatch_trace())  # do_readback_attempts=8
    assert event_one.do_readback == {"2": True, "3": False}
    assert event_many.do_readback == {"2": False, "3": False}
    # Absent is legal only when no readback was recorded (readback unsupported).
    unsupported = _success_trace(
        True,
        do_readback_supported=False,
        do_readback=None,
        controller_output_state_known=False,
        controller_output_matches_requested=None,
    )
    del unsupported["do_readback_attempts"]
    event_unsupported = _adapt(unsupported)
    assert event_unsupported.do_readback_supported is False


def test_adapter_do_readback_attempts_invalid_values() -> None:
    for bad in (0, -1, 2.5, True, "2"):
        with pytest.raises(ValueError, match="positive integer"):
            _adapt(_success_trace(True, do_readback_attempts=bad))
    # Attempt count without a recorded readback contradicts the driver semantics.
    with pytest.raises(ValueError, match="requires a recorded do_readback"):
        _adapt(_readback_exception_trace(do_readback_attempts=1))
    # Attempt count with unsupported readback contradicts the driver semantics.
    with pytest.raises(ValueError, match="do_readback_supported"):
        _adapt(
            _success_trace(
                True,
                do_readback_attempts=1,
                do_readback_supported=False,
                do_readback=None,
                controller_output_state_known=False,
                controller_output_matches_requested=None,
            )
        )
    # Recorded readback without the attempt count contradicts the driver semantics.
    trace = _success_trace(True)
    del trace["do_readback_attempts"]
    with pytest.raises(ValueError, match="no do_readback_attempts"):
        _adapt(trace)


def test_adapter_does_not_mutate_source_trace() -> None:
    trace = _readback_mismatch_trace()
    snapshot = copy.deepcopy(trace)
    _adapt(trace)
    assert trace == snapshot
    assert json.dumps(trace, sort_keys=True) == json.dumps(snapshot, sort_keys=True)


def test_attempt_deep_copy_isolation() -> None:
    trace = _success_trace(True)
    attempt = _attempt(trace)
    trace["error"] = "mutated after construction"
    trace["do_writes"].append({"pin": 9, "value": True, "return_code": 0})
    trace["do_readback"]["2"] = False

    assert attempt.source_trace["error"] is None
    assert len(attempt.source_trace["do_writes"]) == 2
    assert attempt.source_trace["do_readback"] == {"2": True, "3": False}
    # The digest is pinned to the construction-time content.
    assert attempt.source_trace_sha256 == _attempt(_success_trace(True)).source_trace_sha256


def test_source_trace_sha256_deterministic_and_content_sensitive() -> None:
    first = _attempt(_success_trace(True))
    second = _attempt(_success_trace(True))
    assert first.source_trace_sha256 == second.source_trace_sha256
    assert len(first.source_trace_sha256) == 64

    perturbed = _attempt(_success_trace(True, do_readback_attempts=2))
    assert perturbed.source_trace_sha256 != first.source_trace_sha256

    failed = _attempt(_first_call_exception_trace(), committed=False)
    assert failed.source_trace_sha256 != first.source_trace_sha256


def test_failed_attempt_cannot_be_marked_committed() -> None:
    with pytest.raises(ValueError, match="do_api_success"):
        _attempt(_return_code_failure_trace(), committed=True)


def test_attempt_integrity_detects_target_suction_on_tampering() -> None:
    attempt = _attempt()
    assert verify_c0_gripper_controller_attempt_integrity(attempt) == ()

    attempt.source_trace["target_suction_on"] = False
    details = verify_c0_gripper_controller_attempt_integrity(attempt)
    assert "source_trace_sha256_mismatch" in details
    assert "event_trace_inconsistent" in details
    # The pinned digest itself is unchanged; detection comes from recomputation.
    assert attempt.source_trace_sha256 == _attempt().source_trace_sha256


def test_attempt_integrity_detects_error_tampering() -> None:
    attempt = _attempt(_return_code_failure_trace(), committed=False)
    assert verify_c0_gripper_controller_attempt_integrity(attempt) == ()

    attempt.source_trace["error"] = "tampered after validation"
    details = verify_c0_gripper_controller_attempt_integrity(attempt)
    assert "source_trace_sha256_mismatch" in details
    assert "event_trace_inconsistent" in details


def test_attempt_integrity_detects_event_field_tampering() -> None:
    # requested_do
    attempt = _attempt()
    attempt.event.requested_do["2"] = False
    details = verify_c0_gripper_controller_attempt_integrity(attempt)
    assert "event_trace_inconsistent" in details
    assert "source_trace_sha256_mismatch" not in details

    # do_readback
    attempt = _attempt()
    attempt.event.do_readback["2"] = False
    assert "event_trace_inconsistent" in verify_c0_gripper_controller_attempt_integrity(
        attempt
    )

    # a dict inside do_writes
    attempt = _attempt()
    attempt.event.do_writes[0]["return_code"] = -1
    assert "event_trace_inconsistent" in verify_c0_gripper_controller_attempt_integrity(
        attempt
    )

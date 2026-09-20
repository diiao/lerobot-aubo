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
import subprocess
import sys

import pytest

from lerobot.bamboo_sorting.c0_capture_contract import (
    C0_CAPTURE_PROFILE_ID,
    C0_DATASET_FPS,
    C0_EPISODE_MANIFEST_SCHEMA_VERSION,
    C0EpisodeManifestV1,
)
from lerobot.bamboo_sorting.c0_gripper_contract import (
    C0_GRIPPER_EPISODE_BINDING_SCHEMA_VERSION,
    C0_GRIPPER_EPISODE_TRACE_SCHEMA_VERSION,
    C0_GRIPPER_EVENT_SCHEMA_VERSION,
    C0GripperEpisodeBindingV1,
    C0GripperEpisodeTraceV1,
    C0GripperEventV1,
    derive_gripper_transition_frames,
)
from lerobot.bamboo_sorting.contracts import (
    ACTION_SCHEMA_VERSION,
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_SCHEMA_VERSION,
    INSTRUCTION_SPECS,
)
from lerobot.bamboo_sorting.lerobot_bridge import CAMERASET_V2_LEROBOT_BRIDGE_VERSION
from lerobot.bamboo_sorting.rgb_gate import (
    CAMERA_SET_V2_SCHEMA_VERSION,
    FROZEN_CAMERA_SET_V2_SHA256,
)

EPISODE_ID = "c0-gripper-episode-00"
FPS = 25.0
EVIDENCE_SHA = "12" * 32


def _event(event_index: int = 0, frame_index: int = 40, event_type: str = "command_on", **changes):
    if event_type == "command_on":
        before, requested, action_value = 0.0, 100.0, 100.0
        requested_do = {"2": True, "3": False}
        do_writes = [
            {"pin": 3, "value": False, "return_code": 0},
            {"pin": 2, "value": True, "return_code": 0},
        ]
        do_readback = {"2": True, "3": False}
    else:
        before, requested, action_value = 100.0, 0.0, 0.0
        requested_do = {"2": False, "3": True}
        do_writes = [
            {"pin": 2, "value": False, "return_code": 0},
            {"pin": 3, "value": True, "return_code": 0},
        ]
        do_readback = {"2": False, "3": True}
    values = {
        "schema_version": C0_GRIPPER_EVENT_SCHEMA_VERSION,
        "episode_id": EPISODE_ID,
        "event_index": event_index,
        "frame_index": frame_index,
        "sync_timestamp_s": 1.6,
        "event_type": event_type,
        "action_gripper_pos": action_value,
        "requested_state_after": requested,
        "commanded_state_before": before,
        "commanded_state_after": requested,  # successful switch updates the latch
        "transition_requested": True,
        "requested_do": requested_do,
        "do_writes": do_writes,
        "do_write_attempt_count": len(do_writes),
        "do_api_success": True,
        "do_readback_supported": True,
        "do_readback": do_readback,
        "do_readback_after_failure": None,
        "do_readback_error": None,
        "controller_output_state_known": True,
        "controller_output_matches_requested": True,
        "partial_write_possible": False,
        "error": None,
    }
    values.update(changes)
    return C0GripperEventV1(**values)


def _action_values(frame_count: int, on_frame: int | None, off_frame: int | None) -> list[float]:
    values = [0.0] * frame_count
    if on_frame is not None:
        end = off_frame if off_frame is not None else frame_count
        for index in range(on_frame, end):
            values[index] = 100.0
    return values


def _observation_values(action: list[float]) -> list[float]:
    return [0.0, *action[:-1]]


def _trace(
    frame_count: int = 100,
    on_frame: int | None = 40,
    off_frame: int | None = 80,
    lift_start: int | None = 45,
    placement_complete: int | None = 78,
    human_outcome: str = "success",
    **changes,
):
    # Arrays are always built at the nominal length so invalid frame_count
    # overrides reach the constructor instead of breaking the builder.
    action = _action_values(100, on_frame, off_frame)
    events: list[C0GripperEventV1] = []
    if on_frame is not None:
        events.append(_event(event_index=0, frame_index=on_frame, event_type="command_on"))
    if off_frame is not None:
        events.append(
            _event(
                event_index=len(events),
                frame_index=off_frame,
                event_type="command_off",
            )
        )
    values = {
        "schema_version": C0_GRIPPER_EPISODE_TRACE_SCHEMA_VERSION,
        "episode_id": EPISODE_ID,
        "fps": FPS,
        "frame_count": frame_count,
        "action_gripper_values": action,
        "observation_gripper_values": _observation_values(action),
        "events": tuple(events),
        "lift_start_frame_index": lift_start,
        "placement_complete_frame_index": placement_complete,
        "human_outcome": human_outcome,
        "synthetic": True,
        "finalized": True,
    }
    values.update(changes)
    return C0GripperEpisodeTraceV1(**values)


def _episode_manifest(**changes) -> C0EpisodeManifestV1:
    instruction = INSTRUCTION_SPECS["pick_any_collection"]
    values = {
        "schema_version": C0_EPISODE_MANIFEST_SCHEMA_VERSION,
        "episode_id": EPISODE_ID,
        "scene_id": "c0-scene-00",
        "session_id": "c0-session-01",
        "split_name": "train",
        "camera_set_schema_version": CAMERA_SET_V2_SCHEMA_VERSION,
        "camera_set_sha256": FROZEN_CAMERA_SET_V2_SHA256,
        "bridge_version": CAMERASET_V2_LEROBOT_BRIDGE_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "instruction_schema_version": INSTRUCTION_SCHEMA_VERSION,
        "instruction_id": instruction.instruction_id,
        "instruction_text": instruction.text,
        "instruction_language": INSTRUCTION_LANGUAGE,
        "instruction_text_sha256": instruction.text_sha256,
        "calibration_version": "calibration-c0-v1",
        "tool_version": "tool-c0-v1",
        "capture_profile_id": C0_CAPTURE_PROFILE_ID,
        "fps": C0_DATASET_FPS,
        "frame_count": 100,
        "dataset_episode_ref": "datasets/c0/episode-00",
        "dataset_episode_sha256": EVIDENCE_SHA,
        "timestamp_evidence_ref": "evidence/timestamps-00.json",
        "timestamp_evidence_sha256": EVIDENCE_SHA,
        "authorization_evidence_ref": "evidence/authorization-00.json",
        "authorization_evidence_sha256": EVIDENCE_SHA,
        "vlm_shadow_evidence_ref": "evidence/vlm-shadow-00.json",
        "vlm_shadow_evidence_sha256": EVIDENCE_SHA,
        "human_outcome": "success",
        "human_outcome_evidence_ref": "evidence/human-outcome-00.json",
        "human_outcome_evidence_sha256": EVIDENCE_SHA,
        "raw_data_immutable": True,
        "finalized": True,
    }
    values.update(changes)
    return C0EpisodeManifestV1(**values)


def test_valid_event_serializes_with_fixed_no_physical_claim_fields() -> None:
    event = _event()
    record = event.to_manifest_record()

    assert record["schema_version"] == C0_GRIPPER_EVENT_SCHEMA_VERSION
    assert record["event_type"] == "command_on"
    assert record["requested_state_after"] == 100.0
    assert record["commanded_state_after"] == 100.0
    assert record["physical_gripper_feedback_available"] is False
    assert record["physical_grasp_success_proven"] is False
    assert record["serialized_record_grants_live_authorization"] is False
    assert record["hardware_access_performed_by_serialization"] is False
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("field", ["event_index", "frame_index", "do_write_attempt_count"])
@pytest.mark.parametrize("bad_value", [True, -1, 1.5, "3"])
def test_event_rejects_invalid_indices(field: str, bad_value: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        _event(**{field: bad_value})


@pytest.mark.parametrize("bad_value", [True, -0.1, float("nan"), float("inf")])
def test_event_rejects_invalid_sync_timestamp(bad_value: object) -> None:
    with pytest.raises(ValueError, match="sync_timestamp_s"):
        _event(sync_timestamp_s=bad_value)


@pytest.mark.parametrize(
    "field",
    [
        "action_gripper_pos",
        "requested_state_after",
        "commanded_state_before",
        "commanded_state_after",
    ],
)
@pytest.mark.parametrize("bad_value", [50.0, True, "100"])
def test_event_rejects_non_binary_gripper_values(field: str, bad_value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _event(**{field: bad_value})


def test_command_on_must_request_zero_to_hundred() -> None:
    with pytest.raises(ValueError, match=r"must request a 0.0 -> 100.0"):
        _event(commanded_state_before=100.0)
    with pytest.raises(ValueError, match="action_gripper_pos == 100.0"):
        _event(action_gripper_pos=0.0)


def test_command_off_must_request_hundred_to_zero() -> None:
    with pytest.raises(ValueError, match=r"must request a 100.0 -> 0.0"):
        _event(event_type="command_off", requested_state_after=100.0)
    with pytest.raises(ValueError, match="action_gripper_pos == 0.0"):
        _event(event_type="command_off", action_gripper_pos=100.0)


def test_event_rejects_unknown_event_type_and_missing_transition() -> None:
    with pytest.raises(ValueError, match="event_type must be one of"):
        _event(event_type="physical_closed")
    with pytest.raises(ValueError, match="transition_requested=True"):
        _event(transition_requested=False)
    with pytest.raises(ValueError, match="transition_requested must be bool"):
        _event(transition_requested=1)


def test_event_rejects_non_json_evidence() -> None:
    class SdkHandle:
        pass

    with pytest.raises(ValueError, match="string pin ids"):
        _event(requested_do={2: True})
    with pytest.raises(ValueError, match="must be bool"):
        _event(requested_do={"2": 1})
    with pytest.raises(ValueError, match="pure JSON"):
        _event(do_writes=[{"pin": 2, "value": True, "handle": SdkHandle()}])
    with pytest.raises(ValueError, match="non-finite"):
        _event(do_writes=[{"pin": 2, "value": True, "return_code": float("nan")}])
    with pytest.raises(ValueError, match="do_readback"):
        _event(do_readback={"2": True, "3": "yes"})


def test_failed_event_keeps_latch_unchanged_and_preserves_evidence() -> None:
    # Matches the execution layer: a failed DO switch must not update the
    # software latch, and the failure evidence must stay attached.
    failed = _event(
        do_api_success=False,
        commanded_state_after=0.0,  # latch unchanged: still released
        error="setStandardDigitalOutput(pin=2, value=True) returned 7",
        do_writes=[{"pin": 3, "value": False, "return_code": 0}],
        do_write_attempt_count=1,
        partial_write_possible=True,
        do_readback={"2": False, "3": False},
        do_readback_after_failure={"2": False, "3": False},
        controller_output_matches_requested=False,
    )
    record = failed.to_manifest_record()
    assert record["do_api_success"] is False
    assert record["commanded_state_after"] == 0.0
    assert record["commanded_state_before"] == 0.0
    assert record["requested_state_after"] == 100.0
    assert record["partial_write_possible"] is True
    assert record["do_readback_after_failure"] == {"2": False, "3": False}
    assert "returned 7" in record["error"]
    json.dumps(record, allow_nan=False)


def test_failed_event_cannot_claim_latch_switch() -> None:
    with pytest.raises(ValueError, match="must not update the latch"):
        _event(do_api_success=False, error="write failed")
    with pytest.raises(ValueError, match="non-empty error"):
        _event(do_api_success=False, commanded_state_after=0.0, error=None)
    with pytest.raises(ValueError, match="do_api_success=True forbids"):
        _event(error="write failed")


def test_successful_event_must_update_latch_to_requested_state() -> None:
    with pytest.raises(ValueError, match="must update the latch"):
        _event(commanded_state_after=0.0)


def test_event_rejects_missing_or_inconsistent_write_evidence() -> None:
    with pytest.raises(ValueError, match="must record both DO writes"):
        _event(do_writes=[], do_write_attempt_count=0)
    with pytest.raises(ValueError, match="non-success return_code"):
        _event(
            do_writes=[
                {"pin": 3, "value": False, "return_code": 0},
                {"pin": 2, "value": True, "return_code": 7},
            ]
        )
    with pytest.raises(ValueError, match="safe mutual-exclusion order"):
        # Writes target pins that were never requested.
        _event(
            do_writes=[
                {"pin": 8, "value": False, "return_code": 0},
                {"pin": 9, "value": True, "return_code": 0},
            ]
        )
    with pytest.raises(ValueError, match="safe mutual-exclusion order"):
        _event(do_writes=[{"pin": 2, "value": True, "return_code": 0}],
               do_write_attempt_count=1)
    with pytest.raises(ValueError, match="do_write_attempt_count must equal"):
        _event(do_write_attempt_count=1)


def test_failed_event_allows_sdk_exception_without_write_record() -> None:
    # The driver increments the attempt counter before the SDK call and only
    # appends the record after it returns, so an SDK/communication exception
    # leaves one more attempt than recorded writes.
    raised_on_first_call = _event(
        do_api_success=False,
        commanded_state_after=0.0,
        error="setStandardDigitalOutput raised: connection lost",
        do_writes=[],
        do_write_attempt_count=1,
        partial_write_possible=True,
        do_readback_supported=False,
        do_readback=None,
        controller_output_state_known=False,
        controller_output_matches_requested=None,
    )
    assert raised_on_first_call.do_write_attempt_count == 1
    assert raised_on_first_call.do_writes == ()

    raised_on_second_call = _event(
        do_api_success=False,
        commanded_state_after=0.0,
        error="setStandardDigitalOutput raised: timeout",
        do_writes=[{"pin": 3, "value": False, "return_code": 0}],
        do_write_attempt_count=2,
        partial_write_possible=True,
        do_readback_supported=False,
        do_readback=None,
        controller_output_state_known=False,
        controller_output_matches_requested=None,
    )
    assert raised_on_second_call.do_write_attempt_count == 2
    assert len(raised_on_second_call.do_writes) == 1

    with pytest.raises(ValueError, match="exceed them by one"):
        _event(
            do_api_success=False,
            commanded_state_after=0.0,
            error="write failed",
            do_writes=[],
            do_write_attempt_count=2,
            partial_write_possible=True,
            do_readback_supported=False,
            do_readback=None,
            controller_output_state_known=False,
            controller_output_matches_requested=None,
        )
    with pytest.raises(ValueError, match="cannot exceed the number of requested outputs"):
        _event(do_write_attempt_count=3)


def test_event_requires_exactly_one_mutual_exclusion_pair() -> None:
    with pytest.raises(ValueError, match="exactly two pins"):
        _event(requested_do={"2": True}, do_writes=[{"pin": 2, "value": True, "return_code": 0}],
               do_write_attempt_count=1)
    with pytest.raises(ValueError, match="one True and"):
        _event(
            requested_do={"2": True, "3": True},
            do_writes=[
                {"pin": 3, "value": True, "return_code": 0},
                {"pin": 2, "value": True, "return_code": 0},
            ],
        )
    with pytest.raises(ValueError, match="one True and"):
        _event(
            requested_do={"2": False, "3": False},
            do_writes=[
                {"pin": 3, "value": False, "return_code": 0},
                {"pin": 2, "value": False, "return_code": 0},
            ],
        )
    with pytest.raises(ValueError, match="decimal string pin ids"):
        _event(requested_do={"vacuum": True, "vent": False})


def test_event_rejects_duplicate_pin_and_unsafe_write_order() -> None:
    with pytest.raises(ValueError, match="must not repeat a pin"):
        _event(
            do_writes=[
                {"pin": 3, "value": False, "return_code": 0},
                {"pin": 3, "value": False, "return_code": 0},
            ]
        )
    # Dangerous order: the True-target pin written first leaves both outputs
    # briefly active.
    with pytest.raises(ValueError, match="safe mutual-exclusion order"):
        _event(
            do_writes=[
                {"pin": 2, "value": True, "return_code": 0},
                {"pin": 3, "value": False, "return_code": 0},
            ]
        )
    # The order rule also constrains failed events' recorded prefixes.
    with pytest.raises(ValueError, match="safe mutual-exclusion order"):
        _event(
            do_api_success=False,
            commanded_state_after=0.0,
            error="write failed",
            do_writes=[{"pin": 2, "value": True, "return_code": 0}],
            do_write_attempt_count=1,
            partial_write_possible=True,
            do_readback_supported=False,
            do_readback=None,
            controller_output_state_known=False,
            controller_output_matches_requested=None,
        )


def test_event_rejects_contradictory_failure_readbacks() -> None:
    base = {
        "do_api_success": False,
        "commanded_state_after": 0.0,
        "error": "controller DO readback mismatch",
        "partial_write_possible": True,
        "controller_output_matches_requested": False,
    }
    with pytest.raises(ValueError, match="must agree"):
        _event(
            **base,
            do_readback={"2": False, "3": False},
            do_readback_after_failure={"2": True, "3": False},
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _event(
            **base,
            do_readback=None,
            do_readback_after_failure={"2": False, "3": False},
            do_readback_error="readback raised",
        )
    with pytest.raises(ValueError, match="non-empty string"):
        _event(**base, do_readback={"2": False, "3": False}, do_readback_error="")


def test_event_rejects_missing_or_inconsistent_readback_evidence() -> None:
    # Successful event declaring readback support but recording no readback.
    with pytest.raises(ValueError, match="must record do_readback"):
        _event(do_readback=None, controller_output_state_known=False,
               controller_output_matches_requested=None)
    # Known controller state without any readback data.
    with pytest.raises(ValueError, match="requires do_readback"):
        _event(do_readback=None, controller_output_matches_requested=None)
    # Readback data without an explicit match verdict.
    with pytest.raises(ValueError, match="explicit controller_output_matches_requested"):
        _event(controller_output_matches_requested=None)
    # Verdict contradicting the recorded readback.
    with pytest.raises(ValueError, match="contradicts readback"):
        _event(
            do_api_success=False,
            commanded_state_after=0.0,
            error="controller DO readback mismatch",
            partial_write_possible=True,
            controller_output_matches_requested=True,
            do_readback={"2": False, "3": True},
        )
    # Successful readback verdict cannot coexist with API failure semantics.
    with pytest.raises(ValueError, match="must stay a failure"):
        _event(controller_output_matches_requested=False,
               do_readback={"2": False, "3": True})
    # After-failure evidence on a successful event.
    with pytest.raises(ValueError, match="do_readback_after_failure requires"):
        _event(do_readback_after_failure={"2": True, "3": False})
    with pytest.raises(ValueError, match="do_readback_error requires"):
        _event(do_readback_error="readback raised")


def test_event_partial_write_flag_must_match_attempts() -> None:
    with pytest.raises(ValueError, match="partial_write_possible"):
        _event(partial_write_possible=True)
    with pytest.raises(ValueError, match="partial_write_possible"):
        _event(
            do_api_success=False,
            commanded_state_after=0.0,
            error="io_control is None",
            do_writes=[],
            do_write_attempt_count=0,
            partial_write_possible=True,
            do_readback_supported=False,
            do_readback=None,
            controller_output_state_known=False,
            controller_output_matches_requested=None,
        )


def test_event_inputs_and_serialization_are_deep_copied() -> None:
    requested = {"2": True, "3": False}
    writes = [{"pin": 3, "value": False, "return_code": 0}, {"pin": 2, "value": True, "return_code": 0}]
    readback = {"2": True, "3": False}
    event = _event(requested_do=requested, do_writes=writes, do_readback=readback)

    requested["2"] = False
    writes[0]["value"] = True
    readback["2"] = False
    assert event.requested_do == {"2": True, "3": False}
    assert event.do_writes[0]["value"] is False
    assert event.do_readback == {"2": True, "3": False}

    record = event.to_manifest_record()
    record["requested_do"]["2"] = False
    record["do_writes"][0]["pin"] = 99
    fresh = event.to_manifest_record()
    assert fresh["requested_do"] == {"2": True, "3": False}
    assert fresh["do_writes"][0]["pin"] == 3


def test_derive_gripper_transition_frames_uses_request_frames() -> None:
    action = (0.0, 0.0, 100.0, 100.0, 0.0, 0.0)
    observation = (0.0, 0.0, 0.0, 100.0, 100.0, 0.0)
    on_frames, off_frames = derive_gripper_transition_frames(action, observation)
    assert on_frames == (2,)
    assert off_frames == (4,)
    assert derive_gripper_transition_frames((0.0, 0.0), (0.0, 0.0)) == ((), ())
    with pytest.raises(ValueError, match="equal length"):
        derive_gripper_transition_frames((0.0,), (0.0, 0.0))


def test_valid_trace_serializes_with_fixed_offline_fields() -> None:
    trace = _trace()
    record = trace.to_manifest_record()

    assert record["schema_version"] == C0_GRIPPER_EPISODE_TRACE_SCHEMA_VERSION
    assert record["synthetic"] is True
    assert record["real_capture_evidence"] is False
    assert record["physical_gripper_feedback_available"] is False
    assert record["physical_grasp_success_proven"] is False
    assert record["training_authorized"] is False
    assert record["policy_execution_authorized"] is False
    assert record["serialized_record_grants_live_authorization"] is False
    assert record["hardware_access_performed_by_serialization"] is False
    assert record["derived_command_on_frames"] == [40]
    assert record["derived_command_off_frames"] == [80]
    json.dumps(record, allow_nan=False)


def test_trace_rejects_length_mismatch() -> None:
    action = _action_values(100, 40, 80)
    with pytest.raises(ValueError, match="length must equal frame_count"):
        _trace(action_gripper_values=action[:-1])
    with pytest.raises(ValueError, match="length must equal frame_count"):
        _trace(observation_gripper_values=[*_observation_values(action), 0.0])


def test_trace_rejects_non_binary_bool_and_nan_values() -> None:
    action = _action_values(100, 40, 80)
    action[50] = 50.0
    with pytest.raises(ValueError, match="exactly one of"):
        _trace(action_gripper_values=action)

    bool_action = _action_values(100, 40, 80)
    bool_action[50] = True
    with pytest.raises(ValueError, match="not bool"):
        _trace(action_gripper_values=bool_action)

    nan_observation = _observation_values(_action_values(100, 40, 80))
    nan_observation[10] = float("nan")
    with pytest.raises(ValueError, match="exactly one of"):
        _trace(observation_gripper_values=nan_observation)


@pytest.mark.parametrize("frame_count", [True, 0, -5, 2.5])
def test_trace_rejects_invalid_frame_count(frame_count: object) -> None:
    with pytest.raises(ValueError, match="frame_count must be a positive integer"):
        _trace(frame_count=frame_count)


@pytest.mark.parametrize("fps", [True, 0, -25.0, float("nan"), float("inf")])
def test_trace_rejects_invalid_fps(fps: object) -> None:
    with pytest.raises(ValueError, match="fps must be a finite positive number"):
        _trace(fps=fps)


def test_trace_rejects_duplicate_or_unordered_event_frames() -> None:
    duplicate = (_event(event_index=0, frame_index=40), _event(event_index=1, frame_index=40))
    with pytest.raises(ValueError, match="strictly increasing"):
        _trace(events=duplicate)

    unordered = (
        _event(event_index=0, frame_index=80, event_type="command_off"),
        _event(event_index=1, frame_index=40),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        _trace(events=unordered)


def test_trace_rejects_event_frame_outside_episode() -> None:
    with pytest.raises(ValueError, match="inside the episode"):
        _trace(events=(_event(event_index=0, frame_index=100),), on_frame=None, off_frame=None)


def test_trace_rejects_event_episode_id_and_index_mismatch() -> None:
    wrong_episode = _event(event_index=0, frame_index=40, episode_id="other-episode")
    with pytest.raises(ValueError, match="episode_id must equal"):
        _trace(events=(wrong_episode,))

    wrong_index = (
        _event(event_index=0, frame_index=40),
        _event(event_index=0, frame_index=80, event_type="command_off"),
    )
    with pytest.raises(ValueError, match="event_index must be exactly"):
        _trace(events=wrong_index)


def test_trace_rejects_event_action_transition_mismatch() -> None:
    shifted = (
        _event(event_index=0, frame_index=41),
        _event(event_index=1, frame_index=80, event_type="command_off"),
    )
    with pytest.raises(ValueError, match="one-to-one with requested action gripper transitions"):
        _trace(events=shifted)

    missing_off = (_event(event_index=0, frame_index=40),)
    with pytest.raises(ValueError, match="one-to-one"):
        _trace(events=missing_off)


def test_trace_rejects_observation_action_lag_violation() -> None:
    # A corrupted observation shifts the request frames, breaking the
    # one-to-one event correspondence before any state check runs.
    action = _action_values(100, 40, 80)
    lagged = _observation_values(action)
    lagged[41] = 0.0  # must carry the latch after action[40] == 100.0
    with pytest.raises(ValueError, match="one-to-one"):
        _trace(observation_gripper_values=lagged)

    bad_start = _observation_values(action)
    bad_start[0] = 100.0
    with pytest.raises(ValueError, match="released at start"):
        _trace(observation_gripper_values=bad_start)


def test_trace_does_not_require_observation_equal_action_same_frame() -> None:
    # At the transition frame the observation still holds the previous latch.
    trace = _trace()
    assert trace.observation_gripper_values[40] == 0.0
    assert trace.action_gripper_values[40] == 100.0


def test_trace_human_outcome_enum_and_uncertain_preserved() -> None:
    with pytest.raises(ValueError, match="human_outcome must be one of"):
        _trace(human_outcome="passed")

    for outcome in ("success", "failure", "uncertain"):
        trace = _trace(human_outcome=outcome)
        assert trace.human_outcome == outcome
        assert trace.to_manifest_record()["human_outcome"] == outcome


def test_trace_requires_explicit_synthetic_flag() -> None:
    with pytest.raises(ValueError, match="synthetic must be bool"):
        _trace(synthetic="yes")
    with pytest.raises(ValueError, match="synthetic must be True"):
        _trace(synthetic=False)


def test_trace_requires_finalized() -> None:
    with pytest.raises(ValueError, match="finalized must be True"):
        _trace(finalized=False)
    with pytest.raises(ValueError, match="finalized must be bool"):
        _trace(finalized=1)


@pytest.mark.parametrize("field", ["lift_start_frame_index", "placement_complete_frame_index"])
def test_trace_rejects_invalid_annotations(field: str) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        _trace(**{field: True})
    with pytest.raises(ValueError, match="inside the episode"):
        _trace(**{field: 100})
    with pytest.raises(ValueError, match="non-negative integer"):
        _trace(**{field: 2.5})


def test_trace_serialization_is_detached_deep_copy() -> None:
    trace = _trace()
    record = trace.to_manifest_record()
    record["action_gripper_values"][40] = 0.0
    record["events"][0]["requested_do"]["2"] = False
    record["derived_command_on_frames"].append(99)

    fresh = trace.to_manifest_record()
    assert fresh["action_gripper_values"][40] == 100.0
    assert fresh["events"][0]["requested_do"] == {"2": True, "3": False}
    assert fresh["derived_command_on_frames"] == [40]
    assert trace.events[0].requested_do == {"2": True, "3": False}


def test_binding_cross_checks_trace_and_manifest() -> None:
    binding = C0GripperEpisodeBindingV1(
        schema_version=C0_GRIPPER_EPISODE_BINDING_SCHEMA_VERSION,
        trace=_trace(),
        episode_manifest=_episode_manifest(),
    )
    record = binding.to_manifest_record()
    assert record["episode_id"] == EPISODE_ID
    assert record["dataset_content_rehashed_by_binding"] is False
    assert record["binding_is_real_capture_evidence"] is False
    assert record["training_authorized"] is False
    assert record["policy_execution_authorized"] is False
    assert record["serialized_record_grants_live_authorization"] is False
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("episode_id", "other-episode", "episode_id must equal"),
        ("frame_count", 101, "frame_count must equal"),
        ("human_outcome", "failure", "human_outcome must equal"),
    ],
)
def test_binding_rejects_inconsistent_manifest(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        C0GripperEpisodeBindingV1(
            schema_version=C0_GRIPPER_EPISODE_BINDING_SCHEMA_VERSION,
            trace=_trace(),
            episode_manifest=_episode_manifest(**{field: value}),
        )


def test_binding_fps_drift_is_blocked_at_manifest_level() -> None:
    # C0EpisodeManifestV1 pins fps to the frozen 25 Hz capture profile, so an
    # fps drift can never reach the binding cross-check.
    with pytest.raises(ValueError, match="fps must be 25"):
        _episode_manifest(fps=30.0)


_IMPORT_PROBE = r"""
import importlib.util
import json
import os
import sys
import threading
import types

# Stub the parent packages so importing the new modules does NOT execute the
# pre-existing lerobot.bamboo_sorting __init__ (which pulls in heavy modules
# such as torch via smolvla_adapter). Only the new modules and their genuine
# import closure are loaded, so the forbidden-module check is meaningful.
def _stub_package(name):
    spec = importlib.util.find_spec(name)
    module = types.ModuleType(name)
    module.__path__ = list(spec.submodule_search_locations)
    sys.modules[name] = module

_stub_package("lerobot")
_stub_package("lerobot.bamboo_sorting")

baseline_modules = set(sys.modules)

import lerobot.bamboo_sorting.c0_gripper_auditor  # noqa: F401
import lerobot.bamboo_sorting.c0_gripper_contract  # noqa: F401

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
            "contract_loaded": "lerobot.bamboo_sorting.c0_gripper_contract" in sys.modules,
            "auditor_loaded": "lerobot.bamboo_sorting.c0_gripper_auditor" in sys.modules,
        }
    )
)
"""


def test_importing_gripper_modules_touches_no_hardware_network_or_threads() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    state = json.loads(result.stdout.strip().splitlines()[-1])

    # The probe is only meaningful if the modules really loaded and the heavy
    # package __init__ was genuinely bypassed.
    assert state["contract_loaded"] is True
    assert state["auditor_loaded"] is True
    assert state["package_init_executed"] is False
    assert state["forbidden_modules"] == []
    assert state["extra_threads"] == []
    assert state["sockets"] == []

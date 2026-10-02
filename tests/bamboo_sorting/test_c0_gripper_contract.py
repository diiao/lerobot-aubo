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

"""Suction event validation and import isolation for current capture."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot.bamboo_sorting.c0_gripper_contract import C0_GRIPPER_EVENT_SCHEMA_VERSION, C0GripperEventV1

EPISODE_ID = "c0-gripper-episode-00"


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


_IMPORT_PROBE = r"""
import importlib
import json
import os
import sys
import threading

# Import the real parent packages too, from this checkout in a fresh process.
sys.path.insert(0, sys.argv[2])
baseline_modules = set(sys.modules)

importlib.import_module(sys.argv[1])

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
            "module_loaded": sys.argv[1] in sys.modules,
        }
    )
)
"""


@pytest.mark.parametrize(
    "module_name",
    ["c0_gripper_contract", "c0_gripper_capture_journal", "c0_gripper_controller_sidecar"],
)
def test_importing_gripper_modules_touches_no_hardware_network_threads_or_files(
    module_name: str, tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable, "-B", "-c", _IMPORT_PROBE,
            f"lerobot.bamboo_sorting.{module_name}", str(Path(__file__).resolve().parents[2] / "src"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
        cwd=tmp_path,
    )
    state = json.loads(result.stdout.strip().splitlines()[-1])

    assert state["module_loaded"] is True
    assert state["forbidden_modules"] == []
    assert state["extra_threads"] == []
    assert state["sockets"] == []
    assert list(tmp_path.iterdir()) == []

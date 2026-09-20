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

"""Offline-only C0 gripper command event and episode trace contracts.

These records describe gripper *command* switches and their controller
evidence after the fact. Importing, constructing, or serializing them never
reads a camera, connects to AUBO, performs gripper IO, starts training or
inference, or creates live authorization.

The recorded gripper values are software command latches (the last commanded
0/100 state), not vacuum pressure, not physical jaw position, and not grasp
success feedback. Controller DO readback only proves what the controller
output register holds; it is never physical gripper feedback. This contract
does not assign physical meaning to specific DO pin numbers: it records which
pins were requested and what was read back, nothing more.

Latch semantics follow the AUBO execution layer: the software latch is only
updated after the DO writes (and readback, when supported) succeed. A failed
transition request therefore keeps ``commanded_state_after ==
commanded_state_before`` and must retain its failure evidence (``error``,
partial-write flag, after-failure readback) instead of being rewritten into a
successful switch.

Timing model follows ``record_loop``: the observation is captured *before*
the action is sent on each control cycle. A transition is *requested* at frame
``t`` exactly when ``action[t] != observation[t]``; each requested transition
owns exactly one event, and ``observation[t+1]`` equals that event's actual
``commanded_state_after``. When every event succeeds this reduces to the
normal ``observation[t] == action[t-1]`` relation; failed events deliberately
break it, and the contract models that instead of hiding it.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .c0_capture_contract import C0_HUMAN_OUTCOMES, C0EpisodeManifestV1
from .contracts import GRIPPER_BINARY_VALUES

C0_GRIPPER_EVENT_SCHEMA_VERSION: Final = "C0GripperEventV1"
C0_GRIPPER_EPISODE_TRACE_SCHEMA_VERSION: Final = "C0GripperEpisodeTraceV1"
C0_GRIPPER_EPISODE_BINDING_SCHEMA_VERSION: Final = "C0GripperEpisodeBindingV1"

C0_GRIPPER_EVENT_TYPES: Final = frozenset({"command_on", "command_off"})

GRIPPER_RELEASED: Final = 0.0
GRIPPER_COMMANDED_ON: Final = 100.0
DO_WRITE_SUCCESS_RETURN_CODES: Final = (0, None)


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer, not bool or float")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_non_negative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number, not bool")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return converted


def _require_gripper_value(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be one of {GRIPPER_BINARY_VALUES}, not bool")
    converted = float(value)
    if converted not in GRIPPER_BINARY_VALUES:
        raise ValueError(f"{name} must be exactly one of {GRIPPER_BINARY_VALUES}")
    return converted


def _require_json_pure(name: str, value: object) -> None:
    """Reject anything that is not plain JSON data (no SDK objects, no NaN)."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be pure JSON data: non-finite float")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_json_pure(f"{name}[{index}]", item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} must be pure JSON data: non-string key")
            _require_json_pure(f"{name}[{key!r}]", item)
        return
    raise ValueError(f"{name} must be pure JSON data, got {type(value).__name__}")


def _require_str_bool_map(name: str, value: object) -> dict[str, bool]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty mapping of string pin id to bool")
    result: dict[str, bool] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty string pin ids")
        if not isinstance(item, bool):
            raise ValueError(f"{name}[{key!r}] must be bool")
        result[key] = item
    return result


def _require_optional_str_bool_map(name: str, value: object) -> dict[str, bool] | None:
    if value is None:
        return None
    return _require_str_bool_map(name, value)


def derive_gripper_transition_frames(
    action_gripper_values: Sequence[float],
    observation_gripper_values: Sequence[float],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Derive (command_on frames, command_off frames) of requested transitions.

    A transition is requested at frame ``t`` exactly when
    ``action[t] != observation[t]`` (the observation holds the latch from
    before the action is applied). Both sequences must already be validated
    binary 0/100 values of equal length. This helper is pure and never mutates
    its inputs.
    """

    if len(action_gripper_values) != len(observation_gripper_values):
        raise ValueError("action and observation sequences must have equal length")
    on_frames: list[int] = []
    off_frames: list[int] = []
    for index, (observation, action) in enumerate(
        zip(observation_gripper_values, action_gripper_values, strict=True)
    ):
        if action == observation:
            continue
        if observation == GRIPPER_RELEASED and action == GRIPPER_COMMANDED_ON:
            on_frames.append(index)
        else:
            off_frames.append(index)
    return tuple(on_frames), tuple(off_frames)


@dataclass(frozen=True)
class C0GripperEventV1:
    """One requested gripper command switch with its controller evidence.

    ``requested_state_after`` is the requested latch target (100.0 for
    ``command_on``). ``commanded_state_before``/``commanded_state_after`` are
    the actual software latch before/after: matching the AUBO execution layer,
    the latch is only updated after the DO writes (and readback, when
    supported) succeed, so a failed request must keep
    ``commanded_state_after == commanded_state_before``. Failure evidence
    (``error``, ``partial_write_possible``, ``do_readback_after_failure``,
    ``do_readback_error``) is preserved, never normalized into success. None
    of these fields is physical gripper feedback.

    ``requested_do`` must be exactly one pair of mutually exclusive outputs
    (two distinct pins, one True and one False target), and ``do_writes`` must
    follow the safe order of the execution layer: the False-target pin is
    written before the True-target pin so both outputs are never active at
    once. ``do_write_attempt_count`` counts SDK calls; because the driver
    increments it before the call but appends the write record only after the
    SDK returns, a failed event may hold one more attempt than recorded
    writes.
    """

    schema_version: str
    episode_id: str
    event_index: int
    frame_index: int
    sync_timestamp_s: float
    event_type: str
    action_gripper_pos: float
    requested_state_after: float
    commanded_state_before: float
    commanded_state_after: float
    transition_requested: bool
    requested_do: dict[str, bool]
    do_writes: Sequence[dict[str, object]]
    do_write_attempt_count: int
    do_api_success: bool
    do_readback_supported: bool
    do_readback: dict[str, bool] | None
    do_readback_after_failure: dict[str, bool] | None
    do_readback_error: str | None
    controller_output_state_known: bool
    controller_output_matches_requested: bool | None
    partial_write_possible: bool
    error: str | None

    def __post_init__(self) -> None:
        if self.schema_version != C0_GRIPPER_EVENT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {C0_GRIPPER_EVENT_SCHEMA_VERSION!r}")
        _require_nonempty_string("episode_id", self.episode_id)
        _require_non_negative_int("event_index", self.event_index)
        _require_non_negative_int("frame_index", self.frame_index)
        object.__setattr__(
            self, "sync_timestamp_s", _finite_non_negative("sync_timestamp_s", self.sync_timestamp_s)
        )

        if self.event_type not in C0_GRIPPER_EVENT_TYPES:
            raise ValueError(f"event_type must be one of {sorted(C0_GRIPPER_EVENT_TYPES)}")
        action_value = _require_gripper_value("action_gripper_pos", self.action_gripper_pos)
        requested = _require_gripper_value("requested_state_after", self.requested_state_after)
        before = _require_gripper_value("commanded_state_before", self.commanded_state_before)
        after = _require_gripper_value("commanded_state_after", self.commanded_state_after)
        object.__setattr__(self, "action_gripper_pos", action_value)
        object.__setattr__(self, "requested_state_after", requested)
        object.__setattr__(self, "commanded_state_before", before)
        object.__setattr__(self, "commanded_state_after", after)
        if self.event_type == "command_on":
            if before != GRIPPER_RELEASED or requested != GRIPPER_COMMANDED_ON:
                raise ValueError("command_on must request a 0.0 -> 100.0 command switch")
            if action_value != GRIPPER_COMMANDED_ON:
                raise ValueError("command_on requires action_gripper_pos == 100.0")
        else:
            if before != GRIPPER_COMMANDED_ON or requested != GRIPPER_RELEASED:
                raise ValueError("command_off must request a 100.0 -> 0.0 command switch")
            if action_value != GRIPPER_RELEASED:
                raise ValueError("command_off requires action_gripper_pos == 0.0")

        if not isinstance(self.transition_requested, bool):
            raise ValueError("transition_requested must be bool")
        if not self.transition_requested:
            raise ValueError("an event requires transition_requested=True")
        for name in (
            "do_api_success",
            "do_readback_supported",
            "controller_output_state_known",
            "partial_write_possible",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if not (
            self.controller_output_matches_requested is None
            or isinstance(self.controller_output_matches_requested, bool)
        ):
            raise ValueError("controller_output_matches_requested must be bool or None")

        requested_do = _require_str_bool_map("requested_do", self.requested_do)
        object.__setattr__(self, "requested_do", copy.deepcopy(requested_do))

        # The gripper is driven by exactly one pair of mutually exclusive
        # outputs: two distinct pins, one targeted True and one targeted False.
        # This constrains only the mutual-exclusion relation, never any
        # physical meaning of specific pin numbers.
        pin_targets: dict[int, bool] = {}
        for key, value in requested_do.items():
            if not key.isdigit() or key != str(int(key)):
                raise ValueError("requested_do keys must be decimal string pin ids")
            pin_targets[int(key)] = value
        false_pins = sorted(pin for pin, target in pin_targets.items() if not target)
        true_pins = sorted(pin for pin, target in pin_targets.items() if target)
        if len(pin_targets) != 2 or len(false_pins) != 1 or len(true_pins) != 1:
            raise ValueError(
                "requested_do must contain exactly two pins with one True and "
                "one False target"
            )
        # Safe write order of the execution layer: the output going False is
        # written before the output going True, so both outputs are never
        # active at the same time.
        safe_sequence: tuple[tuple[int, bool], ...] = (
            (false_pins[0], False),
            (true_pins[0], True),
        )

        if isinstance(self.do_writes, (str, bytes)):
            raise ValueError("do_writes must be a sequence of pure JSON write records")
        writes: list[dict[str, object]] = []
        for index, write in enumerate(self.do_writes):
            if not isinstance(write, dict):
                raise ValueError(f"do_writes[{index}] must be a pure JSON object")
            _require_json_pure(f"do_writes[{index}]", write)
            pin = write.get("pin")
            if isinstance(pin, bool) or not isinstance(pin, int):
                raise ValueError(f"do_writes[{index}]['pin'] must be an integer, not bool")
            if not isinstance(write.get("value"), bool):
                raise ValueError(f"do_writes[{index}]['value'] must be bool")
            return_code = write.get("return_code")
            if return_code is not None and (
                isinstance(return_code, bool) or not isinstance(return_code, int)
            ):
                raise ValueError(f"do_writes[{index}]['return_code'] must be an integer or None")
            writes.append(copy.deepcopy(dict(write)))
        object.__setattr__(self, "do_writes", tuple(writes))

        # Recorded writes must be a prefix of the safe sequence: no duplicate
        # pins, no unknown pins, and never the True-target pin first. A failed
        # event may hold any prefix; a successful one must hold the full pair.
        written_pins = [write["pin"] for write in writes]
        if len(set(written_pins)) != len(written_pins):
            raise ValueError("do_writes must not repeat a pin")
        if len(writes) > len(safe_sequence):
            raise ValueError("do_writes cannot exceed the two mutual-exclusion outputs")
        for position, (write, (expected_pin, expected_value)) in enumerate(
            zip(writes, safe_sequence, strict=False)
        ):
            if write["pin"] != expected_pin or write["value"] != expected_value:
                raise ValueError(
                    f"do_writes[{position}] violates the safe mutual-exclusion order: "
                    "the False-target pin must be written before the True-target pin"
                )

        # The driver increments the attempt counter before each SDK call but
        # only appends the write record after the SDK returns, so an SDK or
        # communication exception leaves attempt_count == len(do_writes) + 1
        # on the failing call. Success requires every attempt recorded.
        attempt_count = _require_non_negative_int(
            "do_write_attempt_count", self.do_write_attempt_count
        )
        if attempt_count > len(safe_sequence):
            raise ValueError("do_write_attempt_count cannot exceed the number of requested outputs")
        if self.do_api_success:
            if len(writes) != len(safe_sequence) or attempt_count != len(writes):
                raise ValueError(
                    "a successful event must record both DO writes and "
                    "do_write_attempt_count must equal them"
                )
        elif attempt_count not in (len(writes), len(writes) + 1):
            raise ValueError(
                "a failed event's do_write_attempt_count must equal the recorded "
                "do_writes, or exceed them by one when the SDK call raised"
            )

        readback = _require_optional_str_bool_map("do_readback", self.do_readback)
        object.__setattr__(self, "do_readback", copy.deepcopy(readback))
        readback_after_failure = _require_optional_str_bool_map(
            "do_readback_after_failure", self.do_readback_after_failure
        )
        object.__setattr__(
            self, "do_readback_after_failure", copy.deepcopy(readback_after_failure)
        )
        for name in ("do_readback_error", "error"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")

        # Latch semantics of the execution layer: the software latch only
        # changes after a fully successful DO switch.
        if self.do_api_success:
            if after != requested:
                raise ValueError(
                    "a successful event must update the latch: "
                    "commanded_state_after must equal requested_state_after"
                )
        elif after != before:
            raise ValueError(
                "a failed event must not update the latch: "
                "commanded_state_after must equal commanded_state_before"
            )

        # Failure evidence must survive; it can never be normalized into success.
        if self.do_api_success:
            if self.error is not None:
                raise ValueError("do_api_success=True forbids a non-empty error")
        elif not (isinstance(self.error, str) and self.error):
            raise ValueError(
                "do_api_success=False requires a non-empty error; failure must be preserved"
            )
        if self.partial_write_possible != (not self.do_api_success and attempt_count > 0):
            raise ValueError(
                "partial_write_possible must be True exactly when a failed event "
                "attempted at least one DO write"
            )
        if readback_after_failure is not None and self.do_api_success:
            raise ValueError("do_readback_after_failure requires do_api_success=False")
        if self.do_readback_error is not None:
            if self.do_api_success:
                raise ValueError("do_readback_error requires do_api_success=False")
            if not self.do_readback_error:
                raise ValueError("do_readback_error must be a non-empty string")

        # Every write of a successful event must have returned success.
        if self.do_api_success:
            for index, write in enumerate(writes):
                if write["return_code"] not in DO_WRITE_SUCCESS_RETURN_CODES:
                    raise ValueError(
                        f"do_writes[{index}] has non-success return_code "
                        f"{write['return_code']!r} but do_api_success=True"
                    )

        # Readback consistency.
        if not self.do_readback_supported:
            if readback is not None or readback_after_failure is not None:
                raise ValueError("do_readback_supported=False forbids do_readback data")
            if self.controller_output_state_known:
                raise ValueError(
                    "do_readback_supported=False forbids controller_output_state_known=True"
                )
            if self.controller_output_matches_requested is not None:
                raise ValueError(
                    "do_readback_supported=False requires controller_output_matches_requested=None"
                )
            if self.do_readback_error is not None:
                raise ValueError("do_readback_supported=False forbids do_readback_error")
        if self.controller_output_state_known and readback is None and readback_after_failure is None:
            raise ValueError(
                "controller_output_state_known=True requires do_readback or "
                "do_readback_after_failure data"
            )
        # Two recorded readbacks must never contradict each other: the driver
        # copies the mismatching readback into do_readback_after_failure, so
        # they describe the same controller state. An after-failure readback
        # also excludes a readback error on the same event.
        if (
            readback is not None
            and readback_after_failure is not None
            and readback != readback_after_failure
        ):
            raise ValueError(
                "do_readback and do_readback_after_failure must agree when both are recorded"
            )
        if readback_after_failure is not None and self.do_readback_error is not None:
            raise ValueError(
                "do_readback_after_failure and do_readback_error are mutually exclusive"
            )
        readback_reference = readback if readback is not None else readback_after_failure
        if readback_reference is not None and self.controller_output_matches_requested is None:
            raise ValueError(
                "recorded readback data requires an explicit controller_output_matches_requested"
            )
        if (
            self.controller_output_matches_requested is not None
            and not self.controller_output_state_known
        ):
            raise ValueError(
                "controller_output_state_known=False requires "
                "controller_output_matches_requested=None"
            )
        if self.controller_output_matches_requested is False and self.do_api_success:
            raise ValueError(
                "controller_output_matches_requested=False must stay a failure: "
                "do_api_success must be False"
            )
        if (
            readback_reference is not None
            and self.controller_output_matches_requested is not None
            and self.controller_output_matches_requested != (readback_reference == requested_do)
        ):
            raise ValueError(
                "controller_output_matches_requested contradicts readback vs requested_do"
            )
        if self.do_api_success and self.do_readback_supported:
            if readback is None or not self.controller_output_state_known:
                raise ValueError(
                    "a successful event with supported readback must record do_readback "
                    "and a known controller output state"
                )
            if self.controller_output_matches_requested is not True:
                raise ValueError(
                    "a successful event with supported readback requires "
                    "controller_output_matches_requested=True"
                )

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible record with fixed no-physical-claim fields."""

        record = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "event_index": self.event_index,
            "frame_index": self.frame_index,
            "sync_timestamp_s": self.sync_timestamp_s,
            "event_type": self.event_type,
            "action_gripper_pos": self.action_gripper_pos,
            "requested_state_after": self.requested_state_after,
            "commanded_state_before": self.commanded_state_before,
            "commanded_state_after": self.commanded_state_after,
            "transition_requested": self.transition_requested,
            "requested_do": self.requested_do,
            "do_writes": list(self.do_writes),
            "do_write_attempt_count": self.do_write_attempt_count,
            "do_api_success": self.do_api_success,
            "do_readback_supported": self.do_readback_supported,
            "do_readback": self.do_readback,
            "do_readback_after_failure": self.do_readback_after_failure,
            "do_readback_error": self.do_readback_error,
            "controller_output_state_known": self.controller_output_state_known,
            "controller_output_matches_requested": self.controller_output_matches_requested,
            "partial_write_possible": self.partial_write_possible,
            "error": self.error,
            "physical_gripper_feedback_available": False,
            "physical_grasp_success_proven": False,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_serialization": False,
        }
        return copy.deepcopy(record)


@dataclass(frozen=True)
class C0GripperEpisodeTraceV1:
    """Pure-data gripper trajectory for one synthetic C0 episode.

    ``synthetic`` must stay ``True`` in this schema version: a synthetic trace
    can never be presented as real capture evidence. Binding real captures
    requires a new schema version with explicit capture evidence.

    Structural invariants enforced at construction:

    - action and observation lengths equal ``frame_count`` and hold only 0/100;
    - a transition request exists at frame ``t`` exactly when
      ``action[t] != observation[t]``; events correspond one-to-one with those
      frames, in direction and in requested/before states;
    - the latch chain holds: ``observation[0] == 0.0``, and for ``t >= 1``,
      ``observation[t]`` equals the previous latch when no transition was
      requested, or the event's actual ``commanded_state_after`` when one was
      (so failed events leave the latch unchanged);
    - event frames are strictly increasing, unique, and inside the episode.
    """

    schema_version: str
    episode_id: str
    fps: float
    frame_count: int
    action_gripper_values: Sequence[object]
    observation_gripper_values: Sequence[object]
    events: Sequence[C0GripperEventV1]
    lift_start_frame_index: int | None
    placement_complete_frame_index: int | None
    human_outcome: str
    synthetic: bool
    finalized: bool

    def __post_init__(self) -> None:
        if self.schema_version != C0_GRIPPER_EPISODE_TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_GRIPPER_EPISODE_TRACE_SCHEMA_VERSION!r}"
            )
        _require_nonempty_string("episode_id", self.episode_id)

        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise ValueError("fps must be a finite positive number, not bool")
        fps = float(self.fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a finite positive number")
        object.__setattr__(self, "fps", fps)

        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int):
            raise ValueError("frame_count must be a positive integer, not bool or float")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be a positive integer")

        for name in ("action_gripper_values", "observation_gripper_values"):
            raw = getattr(self, name)
            if isinstance(raw, (str, bytes)):
                raise ValueError(f"{name} must be a sequence of binary gripper values")
            values = tuple(
                _require_gripper_value(f"{name}[{index}]", item)
                for index, item in enumerate(raw)
            )
            if len(values) != self.frame_count:
                raise ValueError(f"{name} length must equal frame_count ({self.frame_count})")
            object.__setattr__(self, name, values)

        if isinstance(self.events, (str, bytes)):
            raise ValueError("events must be a sequence of C0GripperEventV1")
        events = tuple(self.events)
        if not all(isinstance(event, C0GripperEventV1) for event in events):
            raise ValueError("events must contain only C0GripperEventV1 records")
        object.__setattr__(self, "events", events)

        previous_frame = -1
        for position, event in enumerate(events):
            if event.episode_id != self.episode_id:
                raise ValueError("event episode_id must equal the trace episode_id")
            if event.event_index != position:
                raise ValueError("event_index must be exactly 0..N-1 in order")
            if event.frame_index >= self.frame_count:
                raise ValueError("event frame_index must be inside the episode")
            if event.frame_index <= previous_frame:
                raise ValueError("event frame_index must be strictly increasing and unique")
            previous_frame = event.frame_index

        # Latch chain: the observation carries the latch before each action.
        # The start state is checked before the event correspondence so a
        # corrupted initial latch reports its own cause.
        observation = self.observation_gripper_values
        if observation[0] != GRIPPER_RELEASED:
            raise ValueError("observation_gripper_values[0] must be 0.0 (released at start)")

        action = self.action_gripper_values
        on_frames, off_frames = derive_gripper_transition_frames(action, observation)
        event_on_frames = tuple(e.frame_index for e in events if e.event_type == "command_on")
        event_off_frames = tuple(e.frame_index for e in events if e.event_type == "command_off")
        if event_on_frames != on_frames or event_off_frames != off_frames:
            raise ValueError(
                "events must correspond one-to-one with requested action gripper transitions"
            )
        events_by_frame = {event.frame_index: event for event in events}
        for event in events:
            frame = event.frame_index
            if (
                event.commanded_state_before != observation[frame]
                or event.requested_state_after != action[frame]
                or event.action_gripper_pos != action[frame]
            ):
                raise ValueError(
                    "event commanded/requested states must match the action and "
                    "observation trajectory at its frame"
                )

        for index in range(1, self.frame_count):
            previous = index - 1
            if action[previous] == observation[previous]:
                expected = observation[previous]
            else:
                expected = events_by_frame[previous].commanded_state_after
            if observation[index] != expected:
                raise ValueError(
                    "record_loop timing requires observation[t] to carry the latch after "
                    "action[t-1]: unchanged without a request, or the event's actual "
                    "commanded_state_after"
                )

        for name in ("lift_start_frame_index", "placement_complete_frame_index"):
            value = getattr(self, name)
            if value is None:
                continue
            index = _require_non_negative_int(name, value)
            if index >= self.frame_count:
                raise ValueError(f"{name} must be inside the episode")

        if self.human_outcome not in C0_HUMAN_OUTCOMES:
            raise ValueError(f"human_outcome must be one of {sorted(C0_HUMAN_OUTCOMES)}")
        if not isinstance(self.synthetic, bool):
            raise ValueError("synthetic must be bool")
        if not self.synthetic:
            raise ValueError(
                "synthetic must be True in this schema version; real captures require "
                "a new schema version with explicit capture evidence"
            )
        if not isinstance(self.finalized, bool):
            raise ValueError("finalized must be bool")
        if not self.finalized:
            raise ValueError("finalized must be True for a completed trace")

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible record that never claims real capture."""

        on_frames, off_frames = derive_gripper_transition_frames(
            self.action_gripper_values, self.observation_gripper_values
        )
        record = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "action_gripper_values": list(self.action_gripper_values),
            "observation_gripper_values": list(self.observation_gripper_values),
            "events": [event.to_manifest_record() for event in self.events],
            "lift_start_frame_index": self.lift_start_frame_index,
            "placement_complete_frame_index": self.placement_complete_frame_index,
            "human_outcome": self.human_outcome,
            "synthetic": self.synthetic,
            "finalized": self.finalized,
            "derived_command_on_frames": list(on_frames),
            "derived_command_off_frames": list(off_frames),
            "real_capture_evidence": False,
            "physical_gripper_feedback_available": False,
            "physical_grasp_success_proven": False,
            "training_authorized": False,
            "policy_execution_authorized": False,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_serialization": False,
        }
        return copy.deepcopy(record)


@dataclass(frozen=True)
class C0GripperEpisodeBindingV1:
    """Offline binding between a gripper trace and a C0 episode manifest.

    This wrapper cross-checks identity and consistency fields only
    (``episode_id``, ``frame_count``, ``fps``, ``human_outcome``) and requires
    the manifest's dataset episode reference and SHA-256 to be present. It
    never reads or re-hashes dataset files, so it does not prove that the
    trace arrays came from the referenced dataset episode; that step belongs
    to the dataset-content audit. Because V1 traces are synthetic-only, this
    binding is currently a structural rehearsal of the future real-capture
    binding and never constitutes real capture evidence.
    """

    schema_version: str
    trace: C0GripperEpisodeTraceV1
    episode_manifest: C0EpisodeManifestV1

    def __post_init__(self) -> None:
        if self.schema_version != C0_GRIPPER_EPISODE_BINDING_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_GRIPPER_EPISODE_BINDING_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.trace, C0GripperEpisodeTraceV1):
            raise ValueError("trace must be a C0GripperEpisodeTraceV1")
        if not isinstance(self.episode_manifest, C0EpisodeManifestV1):
            raise ValueError("episode_manifest must be a C0EpisodeManifestV1")

        manifest = self.episode_manifest
        if self.trace.episode_id != manifest.episode_id:
            raise ValueError("trace episode_id must equal the manifest episode_id")
        if self.trace.frame_count != manifest.frame_count:
            raise ValueError("trace frame_count must equal the manifest frame_count")
        if self.trace.fps != manifest.fps:
            raise ValueError("trace fps must equal the manifest fps")
        if self.trace.human_outcome != manifest.human_outcome:
            raise ValueError("trace human_outcome must equal the manifest human_outcome")
        if not (manifest.dataset_episode_ref and manifest.dataset_episode_sha256):
            raise ValueError(
                "the manifest must carry dataset_episode_ref and dataset_episode_sha256"
            )
        if not (manifest.human_outcome_evidence_ref and manifest.human_outcome_evidence_sha256):
            raise ValueError(
                "the manifest must carry human outcome evidence ref and sha256"
            )

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible binding that grants no authorization."""

        record = {
            "schema_version": self.schema_version,
            "episode_id": self.trace.episode_id,
            "trace": self.trace.to_manifest_record(),
            "episode_manifest": self.episode_manifest.to_manifest_record(),
            "dataset_content_rehashed_by_binding": False,
            "binding_is_real_capture_evidence": False,
            "physical_gripper_feedback_available": False,
            "physical_grasp_success_proven": False,
            "training_authorized": False,
            "policy_execution_authorized": False,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_serialization": False,
        }
        return copy.deepcopy(record)

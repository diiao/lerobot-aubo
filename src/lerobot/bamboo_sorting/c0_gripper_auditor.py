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

"""Deterministic, side-effect-free offline auditor for C0 gripper episodes.

The auditor checks one :class:`C0GripperEpisodeTraceV1` against the
single-strip C0 command-sequence rules. It is pure: it never mutates its
input, never touches hardware, files, network, or threads, and returns the
same report for the same trace.

``C0GripperAuditReportV1`` computes its errors and blockers from the trace at
construction; callers can neither pass nor forge audit results, derived
counts, or the final review-validity flag.

A passing audit means only that the gripper *command* structure and its
*controller* evidence are complete enough to enter human C0 review. It does
not prove a physical grasp, and it never authorizes capture, training, or
policy execution.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Final

from .c0_gripper_contract import (
    GRIPPER_RELEASED,
    C0GripperEpisodeTraceV1,
    derive_gripper_transition_frames,
)

C0_GRIPPER_AUDIT_REPORT_SCHEMA_VERSION: Final = "C0GripperAuditReportV1"

_HUMAN_OUTCOME_BLOCKER: Final = "human_outcome_not_success"


def _sorted_unique(messages: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(messages)))


def _audit_trace(trace: C0GripperEpisodeTraceV1) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Compute (errors, blockers) for one trace; pure and deterministic."""

    errors: list[str] = []
    blockers: list[str] = []
    action = trace.action_gripper_values
    observation = trace.observation_gripper_values

    if action[0] != GRIPPER_RELEASED:
        errors.append("action_must_start_released")
    if action[-1] != GRIPPER_RELEASED:
        errors.append("action_must_end_released")

    on_frames, off_frames = derive_gripper_transition_frames(action, observation)
    if not on_frames:
        errors.append("missing_command_on")
    elif len(on_frames) > 1:
        errors.append("multiple_command_on")
    if not off_frames:
        errors.append("missing_command_off")
    elif len(off_frames) > 1:
        errors.append("multiple_command_off")

    if len(on_frames) == 1 and len(off_frames) == 1:
        on_frame = on_frames[0]
        off_frame = off_frames[0]
        if off_frame < on_frame:
            errors.append("command_off_before_command_on")
        lift_start = trace.lift_start_frame_index
        placement_complete = trace.placement_complete_frame_index
        if lift_start is None:
            errors.append("missing_lift_start_annotation")
        elif lift_start < on_frame:
            errors.append("lift_start_before_command_on")
        if placement_complete is None:
            errors.append("missing_placement_complete_annotation")
        else:
            if off_frame < placement_complete:
                errors.append("command_off_before_placement_complete")
            if lift_start is not None and lift_start > placement_complete:
                errors.append("lift_start_after_placement_complete")

    # Re-verify event/trajectory consistency already enforced at construction,
    # so future trace schema versions cannot silently drop the invariant here.
    event_on_frames = tuple(e.frame_index for e in trace.events if e.event_type == "command_on")
    event_off_frames = tuple(e.frame_index for e in trace.events if e.event_type == "command_off")
    if event_on_frames != on_frames or event_off_frames != off_frames:
        errors.append("event_action_transition_mismatch")
    events_by_frame = {event.frame_index: event for event in trace.events}
    for event in trace.events:
        frame = event.frame_index
        if (
            event.commanded_state_before != observation[frame]
            or event.requested_state_after != action[frame]
        ):
            errors.append(f"event_{event.event_index:04d}:commanded_state_mismatch")

    for index in range(1, trace.frame_count):
        previous = index - 1
        if action[previous] == observation[previous]:
            expected = observation[previous]
        else:
            expected = events_by_frame[previous].commanded_state_after
        if observation[index] != expected:
            errors.append("observation_action_latch_chain_violation")
            break

    for event in trace.events:
        if not event.do_api_success:
            errors.append(f"event_{event.event_index:04d}:do_api_failure")
        if (
            event.do_readback_supported
            and event.controller_output_state_known
            and event.controller_output_matches_requested is False
        ):
            errors.append(f"event_{event.event_index:04d}:readback_mismatch")

    if trace.human_outcome != "success":
        blockers.append(_HUMAN_OUTCOME_BLOCKER)

    return _sorted_unique(errors), _sorted_unique(blockers)


@dataclass(frozen=True)
class C0GripperAuditReportV1:
    """Deterministic audit result computed from one C0 gripper episode trace.

    The constructor accepts only the schema version and the trace; errors,
    blockers, derived transition frames, counts, and the review-validity flag
    are all computed internally and can never be supplied or forged by a
    caller. ``gripper_sequence_valid_for_c0_review`` means nothing more than
    "eligible to enter human C0 review": it never proves a physical grasp and
    never authorizes capture, training, or policy execution.
    """

    schema_version: str
    trace: C0GripperEpisodeTraceV1
    _errors: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _blockers: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != C0_GRIPPER_AUDIT_REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_GRIPPER_AUDIT_REPORT_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.trace, C0GripperEpisodeTraceV1):
            raise ValueError("trace must be a C0GripperEpisodeTraceV1")
        errors, blockers = _audit_trace(self.trace)
        object.__setattr__(self, "_errors", errors)
        object.__setattr__(self, "_blockers", blockers)

    @property
    def episode_id(self) -> str:
        return self.trace.episode_id

    @property
    def errors(self) -> tuple[str, ...]:
        """Audit errors in stable sorted order."""

        return self._errors

    @property
    def blockers(self) -> tuple[str, ...]:
        """Review blockers in stable sorted order."""

        return self._blockers

    @property
    def derived_on_frames(self) -> tuple[int, ...]:
        on_frames, _ = derive_gripper_transition_frames(
            self.trace.action_gripper_values, self.trace.observation_gripper_values
        )
        return on_frames

    @property
    def derived_off_frames(self) -> tuple[int, ...]:
        _, off_frames = derive_gripper_transition_frames(
            self.trace.action_gripper_values, self.trace.observation_gripper_values
        )
        return off_frames

    @property
    def event_count(self) -> int:
        return len(self.trace.events)

    @property
    def gripper_sequence_valid_for_c0_review(self) -> bool:
        """Return structural review eligibility only; never authorization."""

        return not self._errors and not self._blockers

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible report that grants no authorization."""

        record = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "synthetic": self.trace.synthetic,
            "human_outcome": self.trace.human_outcome,
            "fps": self.trace.fps,
            "frame_count": self.trace.frame_count,
            "derived_on_frames": list(self.derived_on_frames),
            "derived_off_frames": list(self.derived_off_frames),
            "event_count": self.event_count,
            "errors": list(self._errors),
            "blockers": list(self._blockers),
            "gripper_sequence_valid_for_c0_review": self.gripper_sequence_valid_for_c0_review,
            "trace": self.trace.to_manifest_record(),
            "physical_gripper_feedback_available": False,
            "physical_grasp_success_proven": False,
            "training_authorized": False,
            "policy_execution_authorized": False,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_serialization": False,
        }
        return copy.deepcopy(record)


def audit_gripper_episode(trace: C0GripperEpisodeTraceV1) -> C0GripperAuditReportV1:
    """Audit one C0 gripper episode trace against the single-strip rules.

    Required sequence for a C0 single-strip episode:
    ``command_on`` <= ``lift_start`` <= ``placement_complete`` <= ``command_off``.
    Equality is allowed at every step: the lift may start on the same frame the
    command is issued, and the release command may share the frame at which
    placement completes. ``lift_start``/``placement_complete`` come from
    explicit episode annotations, never from geometric heuristics.

    Episodes with ``human_outcome`` of ``failure`` or ``uncertain`` are still
    structurally audited, but a blocker keeps them from ever being marked as
    successful demonstrations; ``uncertain`` is never rewritten.
    """

    if not isinstance(trace, C0GripperEpisodeTraceV1):
        raise TypeError("trace must be a C0GripperEpisodeTraceV1")
    return C0GripperAuditReportV1(
        schema_version=C0_GRIPPER_AUDIT_REPORT_SCHEMA_VERSION,
        trace=trace,
    )

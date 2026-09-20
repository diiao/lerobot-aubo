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

"""In-memory pending capture journal for C0 gripper record_loop integration (R4A).

This journal is the C0-side implementation of the generic
``RecordLoopObserver`` hook surface of ``record_loop``. It is pure in-memory
data plumbing: it never reads a camera, never connects to AUBO, never performs
DO/IO, never moves the arm, never touches a real dataset on disk, and never
grants training or policy authorization.

Scope and trust boundaries:

- The journal tracks exactly two lifecycle layers: (1) a control attempt
  happened, and (2) a frame entered the episode buffer. The third layer
  (episode durably saved) is ALWAYS reported as false in R4A; nothing here
  asserts parquet persistence.
- ``frame_entered_episode_buffer`` is set only after ``dataset.add_frame``
  returns normally. A send_action failure or an add_frame exception keeps it
  false even when the underlying buffer lists were partially mutated.
- Only ``requested_transition=True`` cycles create controller events/attempts
  through the frozen adapters. ``requested_transition=False`` cycles keep
  freshness/cycle evidence only.
- Every trace is deep-copied at capture time; later robot-side mutation or
  replacement of ``last_gripper_command_trace`` can never rewrite journal
  content.
- Gripper action consistency is checked fail-closed before ``send_action``:
  the dataset action ``ee.gripper_pos`` must exactly equal the gripper command
  resolved from the action handed to the AUBO state machine. Any mismatch
  aborts before sending, never silently picks one value, and never rewrites an
  action to force equality.
- The serialized record fixes every authorization/physical/durability field to
  false; callers cannot set them to True, no dataset gripper slice digest is
  fabricated, and no final ``C0GripperControllerSidecarV1`` is produced here.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, Final

from lerobot.bamboo_sorting.c0_gripper_contract import (
    C0_GRIPPER_EVENT_SCHEMA_VERSION,
    GRIPPER_COMMANDED_ON,
    GRIPPER_RELEASED,
    C0GripperEventV1,
)
from lerobot.bamboo_sorting.c0_gripper_controller_sidecar import (
    AUBO_GRIPPER_TRACE_REQUIRED_FIELDS,
    C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION,
    C0GripperControllerAttemptV1,
    build_c0_gripper_event_from_aubo_trace,
)

C0_GRIPPER_PENDING_CAPTURE_JOURNAL_SCHEMA_VERSION: Final = "C0GripperPendingCaptureJournalV1"

# Gripper command aliases understood by the AUBO execution layer, in the exact
# priority order of ``AuboI10Robot.send_action`` (first present alias wins).
AUBO_GRIPPER_COMMAND_ALIASES: Final = ("gripper.pos", "gripper_pos", "ee.gripper_pos")

# The single official C0 dataset action field for the gripper command.
C0_DATASET_GRIPPER_FIELD: Final = "ee.gripper_pos"

# Journal lifecycle states.
JOURNAL_STATUS_RECORDING: Final = "recording"
JOURNAL_STATUS_INCOMPLETE: Final = "incomplete"
JOURNAL_STATUS_ABANDONED_RERECORDED: Final = "abandoned_rerecorded"

# Failure stages reported in pure-JSON cycle records.
FAILURE_STAGE_BEFORE_SEND: Final = "before_send"
FAILURE_STAGE_SEND_ACTION: Final = "send_action"
FAILURE_STAGE_ADD_FRAME: Final = "add_frame"


class C0PendingCaptureError(ValueError):
    """Fail-closed journal error: the episode must not proceed as if valid."""


def _require_pure_json(name: str, value: Any) -> None:
    """Reject anything that is not plain JSON data (no SDK objects, no NaN/Inf)."""

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
            _require_pure_json(f"{name}[{index}]", item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} must be pure JSON data: non-string key")
            _require_pure_json(f"{name}[{key!r}]", item)
        return
    raise ValueError(f"{name} must be pure JSON data, got {type(value).__name__}")


def _require_binary_gripper(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be exactly one of 0.0 or 100.0, not bool")
    converted = float(value)
    if not math.isfinite(converted) or converted not in (GRIPPER_RELEASED, GRIPPER_COMMANDED_ON):
        raise ValueError(f"{name} must be exactly one of 0.0 or 100.0")
    return converted


def _require_candidate_frame_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("candidate_frame_index must be a non-negative integer, not bool or float")
    if value < 0:
        raise ValueError("candidate_frame_index must be a non-negative integer")
    return value


def _require_reason(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def extract_c0_dataset_gripper_pos(action: Mapping[str, Any]) -> float:
    """Extract the official C0 dataset gripper field, fail-closed.

    The field must exist and hold exactly 0.0 or 100.0; bool, NaN/Inf, and any
    other value are rejected.
    """

    if C0_DATASET_GRIPPER_FIELD not in action:
        raise ValueError(f"dataset action is missing {C0_DATASET_GRIPPER_FIELD!r}")
    return _require_binary_gripper(C0_DATASET_GRIPPER_FIELD, action[C0_DATASET_GRIPPER_FIELD])


def resolve_aubo_sent_gripper_command(action: Mapping[str, Any]) -> float:
    """Resolve the gripper command actually handed to the AUBO state machine.

    Mirrors the AUBO alias priority (``gripper.pos`` > ``gripper_pos`` >
    ``ee.gripper_pos``) but fails closed: a missing gripper, a bool, NaN/Inf,
    a non 0/100 value, or several simultaneously present aliases with
    conflicting values are all rejected instead of silently picking one.
    """

    present = [
        (alias, action[alias]) for alias in AUBO_GRIPPER_COMMAND_ALIASES if alias in action
    ]
    if not present:
        raise ValueError(f"sent action holds none of the gripper aliases {AUBO_GRIPPER_COMMAND_ALIASES}")
    resolved = [_require_binary_gripper(alias, value) for alias, value in present]
    if len(set(resolved)) != 1:
        raise ValueError(
            f"conflicting gripper alias values {[value for _, value in present]!r}; "
            "refusing to silently pick one"
        )
    return resolved[0]


def _error_description(error: BaseException) -> tuple[str, str]:
    error_type = type(error).__name__
    message = str(error)
    if not message:
        message = error_type
    return error_type, message


def _validate_hold_trace(trace: Mapping[str, Any], *, sent_gripper: float) -> None:
    """Fail closed unless ``trace`` is a complete, consistent AUBO hold trace.

    A hold trace (``requested_transition=False``) must hold exactly the AUBO
    driver field set — no missing and no unknown fields — and every field must
    match the semantics of the driver's no-switch path: no requested DO, no
    writes, no API outcome, no readback evidence, an unchanged software latch,
    and no error. ``target_suction_on`` must also agree with the gripper
    command that was actually sent (100.0 -> on, 0.0 -> off).
    """

    fields = set(trace)
    if fields != AUBO_GRIPPER_TRACE_REQUIRED_FIELDS:
        unknown = sorted(fields - AUBO_GRIPPER_TRACE_REQUIRED_FIELDS)
        missing = sorted(AUBO_GRIPPER_TRACE_REQUIRED_FIELDS - fields)
        raise ValueError(
            f"hold trace must hold exactly the AUBO driver field set; unknown={unknown}, missing={missing}"
        )
    if trace["requested_transition"] is not False:
        raise ValueError("hold trace requires requested_transition is False")
    target = trace["target_suction_on"]
    if not isinstance(target, bool):
        raise ValueError("hold trace requires target_suction_on is bool")
    if trace["requested_do"] is not None:
        raise ValueError("hold trace requires requested_do is None")
    if trace["do_writes"] != []:
        raise ValueError("hold trace requires do_writes == []")
    if isinstance(trace["do_write_attempt_count"], bool) or trace["do_write_attempt_count"] != 0:
        raise ValueError("hold trace requires do_write_attempt_count == 0")
    if trace["do_api_success"] is not None:
        raise ValueError("hold trace requires do_api_success is None")
    if not isinstance(trace["do_readback_supported"], bool):
        raise ValueError("hold trace requires do_readback_supported is bool")
    if trace["do_readback"] is not None:
        raise ValueError("hold trace requires do_readback is None")
    if trace["do_readback_after_failure"] is not None:
        raise ValueError("hold trace requires do_readback_after_failure is None")
    if trace["do_readback_error"] is not None:
        raise ValueError("hold trace requires do_readback_error is None")
    if trace["controller_output_state_known"] is not False:
        raise ValueError("hold trace requires controller_output_state_known is False")
    if trace["controller_output_matches_requested"] is not None:
        raise ValueError("hold trace requires controller_output_matches_requested is None")
    if trace["partial_write_possible"] is not False:
        raise ValueError("hold trace requires partial_write_possible is False")
    latch_before = trace["commanded_state_before"]
    latch_after = trace["commanded_state_after"]
    if not isinstance(latch_before, bool) or not isinstance(latch_after, bool):
        raise ValueError("hold trace requires bool commanded latch states")
    if latch_before != target or latch_after != target:
        raise ValueError(
            "hold trace requires commanded_state_before == commanded_state_after == target_suction_on"
        )
    if trace["error"] is not None:
        raise ValueError("hold trace requires error is None")
    expected_target = sent_gripper == GRIPPER_COMMANDED_ON
    if target is not expected_target:
        raise ValueError(
            f"hold trace target_suction_on={target} contradicts the sent gripper command {sent_gripper}"
        )


class C0GripperPendingCaptureJournalV1:
    """In-memory pending capture journal for one C0 episode recording attempt.

    Implements the generic ``RecordLoopObserver`` protocol of
    ``record_loop``. It distinguishes strictly between (1) a control attempt
    happening and (2) a frame entering the episode buffer; the episode being
    durably saved is always reported false. A journal whose episode failed can
    never be finalized, and a failed cycle never masquerades as finalizable.
    """

    def __init__(self, *, episode_id: str, episode_index: int, fps: int | float) -> None:
        _require_pure_json("episode_id", episode_id)
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError("episode_index must be a non-negative integer, not bool")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise ValueError("fps must be a finite positive number, not bool")
        fps_value = float(fps)
        if not math.isfinite(fps_value) or fps_value <= 0:
            raise ValueError("fps must be a finite positive number")
        self.episode_id = episode_id
        self.episode_index = episode_index
        self.fps = fps_value
        self._status = JOURNAL_STATUS_RECORDING
        self._cycles: list[dict[str, Any]] = []
        self._attempts: list[C0GripperControllerAttemptV1] = []
        self._events: list[C0GripperEventV1] = []
        self._current_cycle: dict[str, Any] | None = None
        self._episode_failed = False
        self._incomplete_reason: str | None = None
        self._abandon_reason: str | None = None

    # ------------------------------------------------------------------ state

    @property
    def status(self) -> str:
        return self._status

    @property
    def finalized(self) -> bool:
        """R4A pending journals are never finalized."""

        return False

    @property
    def dataset_episode_durably_saved(self) -> bool:
        """Layer 3 (durable save) is out of R4A scope and always false."""

        return False

    @property
    def episode_failed(self) -> bool:
        """True once any cycle failed; blocks finalization and new cycles."""

        return self._episode_failed

    @property
    def finalization_blocked(self) -> bool:
        """A pending journal can never finalize; failures reinforce this."""

        return True

    @property
    def cycles(self) -> tuple[dict[str, Any], ...]:
        return copy.deepcopy(tuple(self._cycles))

    @property
    def controller_attempts(self) -> tuple[C0GripperControllerAttemptV1, ...]:
        """Detached deep copies; mutating a return value never rewrites the journal."""

        return copy.deepcopy(tuple(self._attempts))

    @property
    def pending_events(self) -> tuple[C0GripperEventV1, ...]:
        """Detached deep copies; mutating a return value never rewrites the journal."""

        return copy.deepcopy(tuple(self._events))

    # -------------------------------------------------------- observer hooks

    def before_send(
        self,
        *,
        robot: Any,
        dataset: Any,
        action: Mapping[str, Any],
        robot_action_to_send: Mapping[str, Any],
        candidate_frame_index: int,
    ) -> None:
        """Validate action consistency; abort before send_action on mismatch."""

        if self._status != JOURNAL_STATUS_RECORDING:
            raise C0PendingCaptureError(
                f"journal is {self._status!r}; no further capture cycles allowed"
            )
        if self._episode_failed:
            raise C0PendingCaptureError(
                "episode already failed closed; refusing another capture cycle"
            )
        candidate = _require_candidate_frame_index(candidate_frame_index)
        cycle: dict[str, Any] = {
            "cycle_index": len(self._cycles),
            "candidate_frame_index": candidate,
            "dataset_action_gripper_pos": None,
            "sent_action_gripper_command": None,
            "action_consistency": "not_checked",
            "send_action_returned": False,
            "trace": None,
            "trace_present": False,
            "requested_transition": None,
            "frame_entered_episode_buffer": False,
            "dataset_episode_durably_saved": False,
            "failure_stage": None,
            "error_type": None,
            "error_message": None,
        }
        self._current_cycle = cycle
        try:
            dataset_gripper = extract_c0_dataset_gripper_pos(action)
            sent_gripper = resolve_aubo_sent_gripper_command(robot_action_to_send)
            if dataset_gripper != sent_gripper:
                raise ValueError(
                    "action mismatch: dataset ee.gripper_pos="
                    f"{dataset_gripper} but the gripper command handed to the "
                    f"robot resolves to {sent_gripper}; refusing to send"
                )
        except ValueError as error:
            cycle["failure_stage"] = FAILURE_STAGE_BEFORE_SEND
            cycle["error_type"] = "action_mismatch"
            cycle["error_message"] = str(error)
            cycle["action_consistency"] = "mismatch"
            self._cycles.append(cycle)
            self._episode_failed = True
            raise C0PendingCaptureError(str(error)) from error
        cycle["dataset_action_gripper_pos"] = dataset_gripper
        cycle["sent_action_gripper_command"] = sent_gripper
        cycle["action_consistency"] = "matched"
        # The cycle enters the journal at creation; later hooks update it in
        # place so a partially-updated cycle never reads as finalizable.
        self._cycles.append(cycle)

    def on_send_success(
        self,
        *,
        robot: Any,
        dataset: Any,
        action: Mapping[str, Any],
        robot_action_to_send: Mapping[str, Any],
        sent_action: Any,
        candidate_frame_index: int,
    ) -> None:
        """Capture the fresh trace immediately; build a pending event on transitions.

        A successful ``send_action`` must leave a complete fresh trace: a None
        trace, a partial trace, or any contradiction with the gripper command
        that was actually sent fails closed before ``add_frame`` and marks the
        episode failed.
        """

        cycle = self._require_current_cycle(candidate_frame_index)
        cycle["send_action_returned"] = True
        # Deep-copy the trace right after send_action returned, before any next
        # robot action or add_frame can overwrite it. Later robot-side mutation
        # or replacement can never rewrite this evidence.
        trace = copy.deepcopy(getattr(robot, "last_gripper_command_trace", None))
        if trace is None:
            error = ValueError(
                "send_action returned but last_gripper_command_trace is None; "
                "no fresh gripper evidence exists for this cycle"
            )
            self._fail_cycle(cycle, FAILURE_STAGE_SEND_ACTION, "trace_missing", error)
            raise C0PendingCaptureError(str(error)) from error
        try:
            _require_pure_json("last_gripper_command_trace", trace)
        except ValueError as error:
            self._fail_cycle(cycle, FAILURE_STAGE_SEND_ACTION, "trace_not_pure_json", error)
            raise C0PendingCaptureError(str(error)) from error
        cycle["trace"] = trace
        cycle["trace_present"] = True

        # The returned action is checked before add_frame: it must be a mapping
        # with a legal gripper command that matches both the dataset action and
        # the action handed to the robot.
        try:
            returned_gripper = self._validate_sent_action(cycle, robot_action_to_send, sent_action)
        except ValueError as error:
            self._fail_cycle(cycle, FAILURE_STAGE_SEND_ACTION, "sent_action_mismatch", error)
            raise C0PendingCaptureError(str(error)) from error

        requested = trace.get("requested_transition")
        if not isinstance(requested, bool):
            error = ValueError("last_gripper_command_trace['requested_transition'] must be bool")
            self._fail_cycle(cycle, FAILURE_STAGE_SEND_ACTION, "trace_invalid", error)
            raise C0PendingCaptureError(str(error)) from error
        cycle["requested_transition"] = requested
        if not requested:
            # Hold path: keep validated freshness evidence only, never a
            # controller event/attempt. The full hold-trace semantics are
            # validated, and any contradiction fails closed before add_frame.
            try:
                _validate_hold_trace(trace, sent_gripper=returned_gripper)
            except ValueError as error:
                self._fail_cycle(cycle, FAILURE_STAGE_SEND_ACTION, "hold_trace_invalid", error)
                raise C0PendingCaptureError(str(error)) from error
            return
        try:
            event = build_c0_gripper_event_from_aubo_trace(
                episode_id=self.episode_id,
                event_index=len(self._events),
                candidate_frame_index=cycle["candidate_frame_index"],
                dataset_timestamp_s=cycle["candidate_frame_index"] / self.fps,
                action_gripper_pos=cycle["dataset_action_gripper_pos"],
                source_trace=trace,
            )
        except ValueError as error:
            self._fail_cycle(cycle, FAILURE_STAGE_SEND_ACTION, "event_adapter_rejected", error)
            raise C0PendingCaptureError(str(error)) from error
        expected_target = returned_gripper == GRIPPER_COMMANDED_ON
        if trace["target_suction_on"] is not expected_target:
            error = ValueError(
                "post-send trace contradiction: trace['target_suction_on']="
                f"{trace['target_suction_on']} contradicts the returned gripper command "
                f"{returned_gripper}"
            )
            self._fail_cycle(cycle, FAILURE_STAGE_SEND_ACTION, "post_send_contradiction", error)
            raise C0PendingCaptureError(str(error)) from error
        attempt = C0GripperControllerAttemptV1(
            schema_version=C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION,
            attempt_index=len(self._attempts),
            candidate_frame_index=cycle["candidate_frame_index"],
            dataset_frame_committed=False,
            event=event,
            source_trace=trace,
        )
        cycle["event_index"] = event.event_index
        cycle["attempt_index"] = attempt.attempt_index
        self._events.append(event)
        self._attempts.append(attempt)

    def on_send_failure(
        self,
        *,
        robot: Any,
        dataset: Any,
        action: Mapping[str, Any],
        robot_action_to_send: Mapping[str, Any],
        candidate_frame_index: int,
        error: Exception,
    ) -> None:
        """Record send_action failure; keep any fresh failure trace as evidence."""

        cycle = self._require_current_cycle(candidate_frame_index)
        # Immediate deep-copy even on failure: a DO-stage failure leaves a
        # failed trace; a motion-stage failure leaves None (never the stale
        # previous cycle's trace).
        trace = copy.deepcopy(getattr(robot, "last_gripper_command_trace", None))
        error_type, error_message = _error_description(error)
        if trace is not None:
            try:
                _require_pure_json("last_gripper_command_trace", trace)
            except ValueError as purity_error:
                trace = None
                error_message = f"{error_message}; trace dropped: {purity_error}"
        cycle["trace"] = trace
        cycle["trace_present"] = trace is not None
        cycle["send_action_returned"] = False
        cycle["failure_stage"] = FAILURE_STAGE_SEND_ACTION
        cycle["error_type"] = error_type
        cycle["error_message"] = error_message
        requested = trace.get("requested_transition") if isinstance(trace, dict) else None
        cycle["requested_transition"] = requested if isinstance(requested, bool) else None
        if isinstance(trace, dict) and requested is True:
            # Failure evidence preserved as an uncommitted controller attempt.
            try:
                event = build_c0_gripper_event_from_aubo_trace(
                    episode_id=self.episode_id,
                    event_index=len(self._events),
                    candidate_frame_index=cycle["candidate_frame_index"],
                    dataset_timestamp_s=cycle["candidate_frame_index"] / self.fps,
                    action_gripper_pos=cycle["dataset_action_gripper_pos"],
                    source_trace=trace,
                )
                attempt = C0GripperControllerAttemptV1(
                    schema_version=C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION,
                    attempt_index=len(self._attempts),
                    candidate_frame_index=cycle["candidate_frame_index"],
                    dataset_frame_committed=False,
                    event=event,
                    source_trace=trace,
                )
            except ValueError as adapter_error:
                # The raw trace is still preserved in the cycle; only the
                # controller-attempt layer is skipped when adaptation fails.
                cycle["error_message"] = f"{cycle['error_message']}; {adapter_error}"
            else:
                cycle["event_index"] = event.event_index
                cycle["attempt_index"] = attempt.attempt_index
                self._events.append(event)
                self._attempts.append(attempt)
        self._episode_failed = True

    def on_add_frame_success(
        self,
        *,
        robot: Any,
        dataset: Any,
        action: Mapping[str, Any],
        robot_action_to_send: Mapping[str, Any],
        candidate_frame_index: int,
    ) -> None:
        """Mark the frame as entered the episode buffer (not durably saved).

        R4A never rewrites ``dataset_frame_committed``: durable save plus the
        R2 digest audit belong to R4B, so every pending attempt here stays
        ``dataset_frame_committed=False``.
        """

        cycle = self._require_current_cycle(candidate_frame_index)
        cycle["frame_entered_episode_buffer"] = True

    def on_add_frame_failure(
        self,
        *,
        robot: Any,
        dataset: Any,
        action: Mapping[str, Any],
        robot_action_to_send: Mapping[str, Any],
        candidate_frame_index: int,
        error: Exception,
    ) -> None:
        """add_frame did not return normally: buffered stays false, fail closed."""

        cycle = self._require_current_cycle(candidate_frame_index)
        cycle["frame_entered_episode_buffer"] = False
        cycle["failure_stage"] = FAILURE_STAGE_ADD_FRAME
        cycle["error_type"], cycle["error_message"] = _error_description(error)
        self._episode_failed = True

    # ------------------------------------------------------------- lifecycle

    def mark_incomplete(self, *, reason: str) -> None:
        """Explicit stop: the episode ends incomplete, never finalized."""

        reason = _require_reason("reason", reason)
        if self._status == JOURNAL_STATUS_RECORDING:
            self._status = JOURNAL_STATUS_INCOMPLETE
        self._incomplete_reason = reason

    def mark_abandoned(self, *, reason: str) -> None:
        """Rerecord: this attempt is abandoned and can accept no new cycles."""

        reason = _require_reason("reason", reason)
        if self._status != JOURNAL_STATUS_RECORDING:
            raise C0PendingCaptureError(f"journal is {self._status!r}; cannot abandon")
        self._status = JOURNAL_STATUS_ABANDONED_RERECORDED
        self._abandon_reason = reason

    def fork_for_rerecord(self, *, episode_id: str, episode_index: int) -> C0GripperPendingCaptureJournalV1:
        """Start a fully isolated new journal for the next rerecord attempt.

        The new journal carries no cycles, events, or attempts from this one.
        """

        if self._status != JOURNAL_STATUS_ABANDONED_RERECORDED:
            raise C0PendingCaptureError("fork_for_rerecord requires an abandoned journal")
        return C0GripperPendingCaptureJournalV1(
            episode_id=episode_id, episode_index=episode_index, fps=self.fps
        )

    # --------------------------------------------------------- serialization

    def to_manifest_record(self) -> dict[str, Any]:
        """Detached pure-JSON snapshot; every authorization field fixed false."""

        record: dict[str, Any] = {
            "schema_version": C0_GRIPPER_PENDING_CAPTURE_JOURNAL_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "fps": self.fps,
            "status": self._status,
            "finalized": False,
            "incomplete_reason": self._incomplete_reason,
            "abandon_reason": self._abandon_reason,
            "cycles": copy.deepcopy(self._cycles),
            "controller_attempts": [attempt.to_manifest_record() for attempt in self._attempts],
            "event_schema_version": C0_GRIPPER_EVENT_SCHEMA_VERSION,
            # Layer 3 and every downstream authorization stay false in R4A.
            "dataset_episode_durably_saved": False,
            "dataset_gripper_slice_digest_bound": False,
            "final_sidecar_published": False,
            "physical_gripper_feedback_available": False,
            "physical_grasp_success_proven": False,
            "training_authorized": False,
            "policy_execution_authorized": False,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_serialization": False,
        }
        _require_pure_json("journal_record", record)
        return copy.deepcopy(record)

    # -------------------------------------------------------------- internal

    def _require_current_cycle(self, candidate_frame_index: int) -> dict[str, Any]:
        candidate = _require_candidate_frame_index(candidate_frame_index)
        cycle = self._current_cycle
        if cycle is None or cycle["candidate_frame_index"] != candidate:
            raise C0PendingCaptureError(
                "no open cycle for candidate_frame_index "
                f"{candidate}; before_send must run first in this cycle"
            )
        return cycle

    def _fail_cycle(
        self, cycle: dict[str, Any], failure_stage: str, error_type: str, error: BaseException
    ) -> None:
        _, message = _error_description(error)
        cycle["failure_stage"] = failure_stage
        cycle["error_type"] = error_type
        cycle["error_message"] = message
        cycle["frame_entered_episode_buffer"] = False
        self._episode_failed = True

    def _validate_sent_action(
        self,
        cycle: dict[str, Any],
        robot_action_to_send: Mapping[str, Any],
        sent_action: Any,
    ) -> float:
        """Validate the action returned by ``send_action`` against pre-send state.

        The returned action must be a mapping with a legal gripper command that
        equals both the dataset action gripper saved in ``before_send`` and the
        gripper command resolved from ``robot_action_to_send``.
        """

        if not isinstance(sent_action, Mapping):
            raise ValueError(
                f"sent_action must be a mapping of the returned action, got {type(sent_action).__name__}"
            )
        returned_gripper = resolve_aubo_sent_gripper_command(sent_action)
        dataset_gripper = cycle["dataset_action_gripper_pos"]
        if returned_gripper != dataset_gripper:
            raise ValueError(
                "returned action gripper contradicts the dataset action: "
                f"returned {returned_gripper}, dataset ee.gripper_pos {dataset_gripper}"
            )
        to_send_gripper = resolve_aubo_sent_gripper_command(robot_action_to_send)
        if returned_gripper != to_send_gripper:
            raise ValueError(
                "returned action gripper contradicts the action handed to the robot: "
                f"returned {returned_gripper}, robot_action_to_send resolves to {to_send_gripper}"
            )
        return returned_gripper

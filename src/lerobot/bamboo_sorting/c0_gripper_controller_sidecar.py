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

"""Offline-only C0 gripper controller attempt records and episode sidecar.

This module adapts the *already serialized* AUBO ``last_gripper_command_trace``
dict into the frozen :class:`C0GripperEventV1` contract, and wraps one or more
such events into per-attempt records and a per-episode controller sidecar. It
is pure data plumbing: importing, constructing, adapting, or serializing these
records never reads a camera, connects to AUBO, performs gripper DO/IO, moves
the arm, starts training or inference, or creates live authorization.

Scope and trust boundaries:

- The adapter only handles *switch* events (``requested_transition=True``).
  It never re-invents an event outcome from the action value, never rewrites a
  failed event into a success, and never drops the SDK-exception semantics
  where ``do_write_attempt_count == len(do_writes) + 1``.
- DO readback only proves what the controller output register holds. It is NOT
  vacuum pressure, NOT physical jaw position, and NOT grasp success feedback.
- ``source_trace_sha256`` is a content-change detector over the stored trace
  copy. It is NOT a signature, NOT authentication, and can never prove that a
  trace genuinely came from the real AUBO driver.
- The sidecar is not wired into ``record_loop`` in this round, so it never
  claims ``live_capture_integration_verified``.
- ``dataclass(frozen=True)`` does not freeze nested dicts/lists. Every later
  use of these records therefore re-validates the CURRENT content with the
  deterministic :func:`verify_c0_gripper_controller_attempt_integrity` /
  :func:`verify_c0_gripper_controller_sidecar_integrity` checks (schema,
  recomputed digests, event-vs-trace field consistency, frame and commit
  semantics), and ``to_manifest_record`` fails closed instead of serializing
  a record that would mix changed content with a stale digest.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Final

from .c0_capture_contract import _require_sha256
from .c0_gripper_contract import (
    C0_GRIPPER_EVENT_SCHEMA_VERSION,
    GRIPPER_COMMANDED_ON,
    GRIPPER_RELEASED,
    C0GripperEventV1,
    _require_json_pure,
    _require_non_negative_int,
    _require_nonempty_string,
)

C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION: Final = "C0GripperControllerAttemptV1"
C0_GRIPPER_CONTROLLER_SIDECAR_SCHEMA_VERSION: Final = "C0GripperControllerSidecarV1"

# Exact field set of the current AUBO driver trace as produced by
# ``AuboI10Robot._set_suction_outputs`` for a requested transition. Any driver
# schema change must bump these contracts to a new schema version; V1 fails
# closed on unknown or missing fields instead of silently ignoring them.
AUBO_GRIPPER_TRACE_REQUIRED_FIELDS: Final = frozenset(
    {
        "requested_transition",
        "target_suction_on",
        "requested_do",
        "do_writes",
        "do_write_attempt_count",
        "do_api_success",
        "do_readback_supported",
        "do_readback",
        "do_readback_after_failure",
        "do_readback_error",
        "controller_output_state_known",
        "controller_output_matches_requested",
        "partial_write_possible",
        "commanded_state_before",
        "commanded_state_after",
        "error",
    }
)
AUBO_GRIPPER_TRACE_OPTIONAL_FIELDS: Final = frozenset({"do_readback_attempts"})
AUBO_GRIPPER_TRACE_WRITE_RECORD_FIELDS: Final = frozenset({"pin", "value", "return_code"})

_NO_TRANSITION_ERROR: Final = (
    "source_trace with requested_transition=False is not a switch event; "
    "only requested transitions can be adapted into a C0GripperEventV1"
)


def _canonical_json_sha256(payload: dict[str, object]) -> str:
    """Return SHA-256 over the canonical JSON encoding of a pure-JSON payload."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _require_bool_field(trace: dict[str, object], name: str) -> bool:
    value = trace[name]
    if not isinstance(value, bool):
        raise ValueError(f"source_trace[{name!r}] must be bool")
    return value


def _require_optional_str_bool_map_field(
    trace: dict[str, object], name: str
) -> dict[str, bool] | None:
    value = trace[name]
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError(f"source_trace[{name!r}] must be None or a non-empty str->bool mapping")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"source_trace[{name!r}] keys must be non-empty string pin ids")
        if not isinstance(item, bool):
            raise ValueError(f"source_trace[{name!r}][{key!r}] must be bool")
    return value


def validate_aubo_gripper_transition_trace(source_trace: object) -> None:
    """Fail closed unless ``source_trace`` matches the current AUBO driver trace schema.

    The trace must be pure JSON data (no SDK objects, no NaN/Inf, no non-string
    dict keys), must hold exactly the required V1 field set plus the optional
    ``do_readback_attempts``, and must describe a *requested* transition.
    Unknown fields are rejected: a future driver trace schema change requires a
    new schema version, never silent field dropping. This function never
    mutates ``source_trace``.
    """

    if not isinstance(source_trace, dict):
        raise ValueError("source_trace must be a dict of pure JSON data")
    _require_json_pure("source_trace", source_trace)

    fields = set(source_trace)
    unknown = sorted(fields - AUBO_GRIPPER_TRACE_REQUIRED_FIELDS - AUBO_GRIPPER_TRACE_OPTIONAL_FIELDS)
    if unknown:
        raise ValueError(
            f"source_trace holds unknown fields {unknown}; a driver trace schema change "
            "requires a new schema version instead of silently ignoring fields"
        )
    missing = sorted(AUBO_GRIPPER_TRACE_REQUIRED_FIELDS - fields)
    if missing:
        raise ValueError(f"source_trace is missing required fields {missing}")

    requested = source_trace["requested_transition"]
    if not isinstance(requested, bool):
        raise ValueError("source_trace['requested_transition'] must be bool")
    if not requested:
        raise ValueError(_NO_TRANSITION_ERROR)

    _require_bool_field(source_trace, "target_suction_on")
    _require_bool_field(source_trace, "do_api_success")
    _require_bool_field(source_trace, "do_readback_supported")
    _require_bool_field(source_trace, "controller_output_state_known")
    _require_bool_field(source_trace, "partial_write_possible")
    _require_bool_field(source_trace, "commanded_state_before")
    _require_bool_field(source_trace, "commanded_state_after")

    matches = source_trace["controller_output_matches_requested"]
    if matches is not None and not isinstance(matches, bool):
        raise ValueError("source_trace['controller_output_matches_requested'] must be bool or None")

    requested_do = source_trace["requested_do"]
    if not isinstance(requested_do, dict) or not requested_do:
        raise ValueError("source_trace['requested_do'] must be a non-empty str->bool mapping")
    for key, item in requested_do.items():
        if not isinstance(key, str) or not key:
            raise ValueError("source_trace['requested_do'] keys must be non-empty string pin ids")
        if not isinstance(item, bool):
            raise ValueError(f"source_trace['requested_do'][{key!r}] must be bool")

    do_writes = source_trace["do_writes"]
    if not isinstance(do_writes, list):
        raise ValueError("source_trace['do_writes'] must be a list of write records")
    for index, write in enumerate(do_writes):
        if not isinstance(write, dict):
            raise ValueError(f"source_trace['do_writes'][{index}] must be a dict")
        write_fields = set(write)
        if write_fields != AUBO_GRIPPER_TRACE_WRITE_RECORD_FIELDS:
            raise ValueError(
                f"source_trace['do_writes'][{index}] must hold exactly "
                f"{sorted(AUBO_GRIPPER_TRACE_WRITE_RECORD_FIELDS)}"
            )
        pin = write["pin"]
        if isinstance(pin, bool) or not isinstance(pin, int):
            raise ValueError(f"source_trace['do_writes'][{index}]['pin'] must be an integer")
        if not isinstance(write["value"], bool):
            raise ValueError(f"source_trace['do_writes'][{index}]['value'] must be bool")
        return_code = write["return_code"]
        if return_code is not None and (
            isinstance(return_code, bool) or not isinstance(return_code, int)
        ):
            raise ValueError(
                f"source_trace['do_writes'][{index}]['return_code'] must be an integer or None"
            )

    attempt_count = source_trace["do_write_attempt_count"]
    if isinstance(attempt_count, bool) or not isinstance(attempt_count, int) or attempt_count < 0:
        raise ValueError("source_trace['do_write_attempt_count'] must be a non-negative integer")

    for name in ("do_readback_error", "error"):
        value = source_trace[name]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"source_trace[{name!r}] must be a string or None")

    readback = _require_optional_str_bool_map_field(source_trace, "do_readback")
    _require_optional_str_bool_map_field(source_trace, "do_readback_after_failure")

    # do_readback_attempts is written by the driver together with do_readback
    # inside the readback loop, so the two are co-present: a recorded readback
    # implies a positive attempt count, and an attempt count implies a recorded
    # readback from a supported readback path.
    if "do_readback_attempts" in source_trace:
        attempts = source_trace["do_readback_attempts"]
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError("source_trace['do_readback_attempts'] must be a positive integer")
        if not source_trace["do_readback_supported"]:
            raise ValueError(
                "source_trace['do_readback_attempts'] requires do_readback_supported=True"
            )
        if readback is None:
            raise ValueError(
                "source_trace['do_readback_attempts'] requires a recorded do_readback"
            )
    elif readback is not None:
        raise ValueError(
            "source_trace records do_readback but no do_readback_attempts; "
            "the driver always writes them together"
        )


def _bool_to_commanded_state(value: bool) -> float:
    return GRIPPER_COMMANDED_ON if value else GRIPPER_RELEASED


def _derive_event_fields_from_aubo_trace(trace: dict[str, object]) -> dict[str, object]:
    """Map a validated AUBO transition trace onto normalized C0GripperEventV1 fields.

    The bool latch states map to 0.0/100.0; every piece of controller evidence
    (requested DO, writes, attempt count, readbacks, readback error, output
    state flags, partial-write flag, error) is carried over unchanged. Nothing
    is recomputed from the action value and failures are never rewritten into
    successes.
    """

    target_on = trace["target_suction_on"]
    assert isinstance(target_on, bool)
    return {
        "event_type": "command_on" if target_on else "command_off",
        "action_gripper_pos": GRIPPER_COMMANDED_ON if target_on else GRIPPER_RELEASED,
        "requested_state_after": GRIPPER_COMMANDED_ON if target_on else GRIPPER_RELEASED,
        "commanded_state_before": _bool_to_commanded_state(trace["commanded_state_before"]),
        "commanded_state_after": _bool_to_commanded_state(trace["commanded_state_after"]),
        "requested_do": copy.deepcopy(trace["requested_do"]),
        "do_writes": tuple(copy.deepcopy(write) for write in trace["do_writes"]),
        "do_write_attempt_count": trace["do_write_attempt_count"],
        "do_api_success": trace["do_api_success"],
        "do_readback_supported": trace["do_readback_supported"],
        "do_readback": copy.deepcopy(trace["do_readback"]),
        "do_readback_after_failure": copy.deepcopy(trace["do_readback_after_failure"]),
        "do_readback_error": trace["do_readback_error"],
        "controller_output_state_known": trace["controller_output_state_known"],
        "controller_output_matches_requested": trace["controller_output_matches_requested"],
        "partial_write_possible": trace["partial_write_possible"],
        "error": trace["error"],
    }


def build_c0_gripper_event_from_aubo_trace(
    *,
    episode_id: str,
    event_index: int,
    candidate_frame_index: int,
    dataset_timestamp_s: float,
    action_gripper_pos: float,
    source_trace: dict[str, object],
) -> C0GripperEventV1:
    """Adapt one serialized AUBO ``last_gripper_command_trace`` into C0GripperEventV1.

    Only ``requested_transition=True`` switch events are accepted. The caller
    must supply the episode coordinates explicitly: ``dataset_timestamp_s`` is
    the relative timestamp of the candidate dataset frame, so this function
    never calls ``time.time()`` or ``time.monotonic()`` itself. The mapping is
    ``target_suction_on=True -> command_on / requested_state_after=100.0`` and
    ``target_suction_on=False -> command_off / requested_state_after=0.0``;
    the bool latch states map to 0.0/100.0. ``source_trace`` is validated
    against the exact current driver field set and is never modified.
    """

    validate_aubo_gripper_transition_trace(source_trace)
    assert isinstance(source_trace, dict)

    if isinstance(action_gripper_pos, bool) or not isinstance(action_gripper_pos, (int, float)):
        raise ValueError("action_gripper_pos must be exactly one of 0.0 or 100.0, not bool")
    action_value = float(action_gripper_pos)
    fields = _derive_event_fields_from_aubo_trace(source_trace)
    if action_value != fields["action_gripper_pos"]:
        raise ValueError(
            "action_gripper_pos contradicts source_trace['target_suction_on']: "
            f"target_suction_on={source_trace['target_suction_on']} requires "
            f"action_gripper_pos == {fields['action_gripper_pos']}, got {action_value}"
        )

    return C0GripperEventV1(
        schema_version=C0_GRIPPER_EVENT_SCHEMA_VERSION,
        episode_id=episode_id,
        event_index=event_index,
        frame_index=candidate_frame_index,
        sync_timestamp_s=dataset_timestamp_s,
        transition_requested=True,
        **fields,
    )


def _source_trace_sha256(source_trace: dict[str, object]) -> str:
    """Derive the attempt's trace digest internally; callers can never inject one.

    The payload is exactly ``{"schema_version": ..., "source_trace": <stored
    deep copy>}`` under the attempt schema version, canonicalized with sorted
    keys, compact separators, UTF-8, and ``allow_nan=False``. This digest only
    detects content changes of the stored trace; it is not authentication and
    never proves the trace came from the real AUBO driver.
    """

    payload = {
        "schema_version": C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION,
        "source_trace": source_trace,
    }
    return _canonical_json_sha256(payload)


@dataclass(frozen=True)
class C0GripperControllerAttemptV1:
    """One gripper switch *attempt* that actually happened, success or failure.

    ``dataset_frame_committed=True`` asserts that ``send_action`` succeeded and
    the candidate frame was finally committed into the dataset; that requires
    ``event.do_api_success=True`` and ``event.frame_index ==
    candidate_frame_index``. ``dataset_frame_committed=False`` preserves
    send_action/DO failure evidence for a candidate frame that never entered
    the parquet data; the failure evidence (``error``, partial write,
    after-failure readback) is stored unchanged, and such an attempt blocks the
    episode controller binding audit.

    ``source_trace`` is deep-copied at construction so later external mutation
    cannot rewrite history. ``source_trace_sha256`` is derived internally from
    the stored copy and can never be supplied by a caller. The adapted event is
    cross-validated field by field against the trace, so a caller cannot pair
    a clean event with a failed or opposite-direction trace.
    """

    schema_version: str
    attempt_index: int
    candidate_frame_index: int
    dataset_frame_committed: bool
    event: C0GripperEventV1
    source_trace: dict[str, object]
    _source_trace_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION!r}"
            )
        _require_non_negative_int("attempt_index", self.attempt_index)
        _require_non_negative_int("candidate_frame_index", self.candidate_frame_index)
        if not isinstance(self.dataset_frame_committed, bool):
            raise ValueError("dataset_frame_committed must be bool")
        if not isinstance(self.event, C0GripperEventV1):
            raise ValueError("event must be a C0GripperEventV1")

        validate_aubo_gripper_transition_trace(self.source_trace)
        stored_trace = copy.deepcopy(self.source_trace)
        object.__setattr__(self, "source_trace", stored_trace)
        object.__setattr__(self, "_source_trace_sha256", _source_trace_sha256(stored_trace))

        # Cross-validate the adapted event against the trace field by field.
        expected = _derive_event_fields_from_aubo_trace(stored_trace)
        for name, expected_value in expected.items():
            if getattr(self.event, name) != expected_value:
                raise ValueError(
                    f"event.{name} contradicts source_trace: event holds "
                    f"{getattr(self.event, name)!r}, trace implies {expected_value!r}"
                )
        if self.event.frame_index != self.candidate_frame_index:
            raise ValueError("event.frame_index must equal candidate_frame_index")

        if self.dataset_frame_committed and not self.event.do_api_success:
            raise ValueError(
                "a dataset-committed attempt requires event.do_api_success=True; "
                "failure evidence must stay an uncommitted attempt"
            )

    @property
    def source_trace_sha256(self) -> str:
        """Internally derived digest of the stored trace copy (never caller-supplied)."""

        return self._source_trace_sha256

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible record with fixed boundary fields.

        Fail closed: if the current content no longer passes the deterministic
        integrity revalidation, this raises ValueError instead of serializing a
        record that would mix changed content with the pinned digest.
        """

        details = verify_c0_gripper_controller_attempt_integrity(self)
        if details:
            raise ValueError(
                f"controller_attempt_integrity_mismatch; refusing to serialize a "
                f"record mixing changed content with the pinned digest: {list(details)}"
            )
        record = {
            "schema_version": self.schema_version,
            "attempt_index": self.attempt_index,
            "candidate_frame_index": self.candidate_frame_index,
            "dataset_frame_committed": self.dataset_frame_committed,
            "event": self.event.to_manifest_record(),
            "source_trace": self.source_trace,
            "source_trace_sha256": self._source_trace_sha256,
            "physical_gripper_feedback_available": False,
            "physical_grasp_success_proven": False,
            "controller_event_origin_authenticated": False,
            "live_capture_integration_verified": False,
            "training_authorized": False,
            "policy_execution_authorized": False,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_serialization": False,
        }
        return copy.deepcopy(record)


def _validate_sidecar_attempts(sidecar: C0GripperControllerSidecarV1) -> None:
    """Validate the CURRENT attempts sequence of a sidecar; raises ValueError.

    Shared by ``__post_init__`` and the post-construction integrity check so a
    structurally corrupted sidecar (for example via ``object.__setattr__``) is
    judged by exactly the same rules as a freshly built one. Attempt frames
    must be strictly increasing and unique: the episode timeline is ordered,
    so an off event can never precede an on event at a lower frame.
    """

    previous_frame = -1
    for position, attempt in enumerate(sidecar.attempts):
        if attempt.attempt_index != position:
            raise ValueError("attempt_index must be exactly 0..N-1 in order, no gaps")
        if attempt.event.event_index != position:
            raise ValueError("event.event_index must equal the attempt position 0..N-1")
        if attempt.event.episode_id != sidecar.episode_id:
            raise ValueError("attempt event episode_id must equal the sidecar episode_id")
        if attempt.dataset_frame_committed:
            if attempt.candidate_frame_index >= sidecar.frame_count:
                raise ValueError(
                    "a committed attempt must reference an existing dataset frame: "
                    "candidate_frame_index < frame_count"
                )
        elif attempt.candidate_frame_index > sidecar.frame_count:
            raise ValueError(
                "an uncommitted attempt may use candidate_frame_index == frame_count "
                "for a candidate after the last saved frame, never beyond it"
            )
        if attempt.candidate_frame_index <= previous_frame:
            raise ValueError(
                "candidate_frame_index must be strictly increasing and unique across attempts"
            )
        previous_frame = attempt.candidate_frame_index


def verify_c0_gripper_controller_attempt_integrity(
    attempt: C0GripperControllerAttemptV1,
) -> tuple[str, ...]:
    """Deterministically re-validate the CURRENT content of an attempt.

    ``dataclass(frozen=True)`` does not freeze nested dicts/lists, so this
    check exists for every later use of the object: the sidecar constructor,
    the binding audit, and ``to_manifest_record`` all call it. It re-checks
    that the current ``source_trace`` still satisfies the driver trace schema,
    that its recomputed SHA-256 equals the pinned ``source_trace_sha256``, that
    the normalized event still matches the trace field by field (which also
    covers tampering of nested event fields such as ``requested_do``,
    ``do_readback``, or the ``do_writes`` dicts), that ``event.frame_index``
    still equals ``candidate_frame_index``, and that the committed/failure
    semantics still hold.

    Returns a sorted tuple of stable detail codes; an empty tuple means the
    attempt is intact. Detail codes: ``source_trace_schema_invalid``,
    ``source_trace_sha256_mismatch``, ``event_trace_inconsistent``,
    ``candidate_frame_mismatch``, ``commit_semantics_mismatch``.
    """

    if not isinstance(attempt, C0GripperControllerAttemptV1):
        raise TypeError("attempt must be a C0GripperControllerAttemptV1")
    details: list[str] = []
    try:
        validate_aubo_gripper_transition_trace(attempt.source_trace)
    except ValueError:
        details.append("source_trace_schema_invalid")
    else:
        assert isinstance(attempt.source_trace, dict)
        if _source_trace_sha256(attempt.source_trace) != attempt.source_trace_sha256:
            details.append("source_trace_sha256_mismatch")
        expected = _derive_event_fields_from_aubo_trace(attempt.source_trace)
        for name, expected_value in expected.items():
            if getattr(attempt.event, name) != expected_value:
                details.append("event_trace_inconsistent")
                break
    if attempt.event.frame_index != attempt.candidate_frame_index:
        details.append("candidate_frame_mismatch")
    if attempt.dataset_frame_committed and not attempt.event.do_api_success:
        details.append("commit_semantics_mismatch")
    return tuple(sorted(set(details)))


def verify_c0_gripper_controller_sidecar_integrity(
    sidecar: C0GripperControllerSidecarV1,
) -> tuple[str, ...]:
    """Deterministically re-validate the CURRENT content of a sidecar.

    Re-runs the full attempts structural rules on the current state, re-verifies
    every attempt, and recomputes the sidecar digest against the pinned
    ``sidecar_sha256``. Returns a sorted tuple of stable detail codes; an empty
    tuple means the sidecar is intact. Detail codes:
    ``sidecar_structure_invalid``, ``attempt_integrity_mismatch``,
    ``sidecar_sha256_mismatch``.
    """

    if not isinstance(sidecar, C0GripperControllerSidecarV1):
        raise TypeError("sidecar must be a C0GripperControllerSidecarV1")
    details: list[str] = []
    try:
        _validate_sidecar_attempts(sidecar)
    except ValueError:
        details.append("sidecar_structure_invalid")
    attempts_intact = True
    for attempt in sidecar.attempts:
        if verify_c0_gripper_controller_attempt_integrity(attempt):
            attempts_intact = False
    if not attempts_intact:
        details.append("attempt_integrity_mismatch")
    else:
        # The digest payload renders each event; only recompute it when every
        # attempt is intact so tampered content can never crash the check.
        try:
            if _sidecar_sha256(sidecar) != sidecar.sidecar_sha256:
                details.append("sidecar_sha256_mismatch")
        except ValueError:
            details.append("sidecar_sha256_mismatch")
    return tuple(sorted(set(details)))


def _sidecar_sha256(sidecar: C0GripperControllerSidecarV1) -> str:
    """Derive the sidecar digest internally; callers can never inject one.

    The payload is exactly: the sidecar schema version, ``episode_id``,
    ``episode_index``, ``fps``, ``frame_count``,
    ``dataset_gripper_slice_sha256``, ``finalized``, and for each attempt in
    order its ``attempt_index``, ``candidate_frame_index``,
    ``dataset_frame_committed``, the event's manifest record, and the attempt's
    ``source_trace_sha256``. Canonicalized with sorted keys, compact
    separators, UTF-8, and ``allow_nan=False``. Content-change detection only;
    not authentication.
    """

    payload = {
        "schema_version": sidecar.schema_version,
        "episode_id": sidecar.episode_id,
        "episode_index": sidecar.episode_index,
        "fps": sidecar.fps,
        "frame_count": sidecar.frame_count,
        "dataset_gripper_slice_sha256": sidecar.dataset_gripper_slice_sha256,
        "finalized": sidecar.finalized,
        "attempts": [
            {
                "attempt_index": attempt.attempt_index,
                "candidate_frame_index": attempt.candidate_frame_index,
                "dataset_frame_committed": attempt.dataset_frame_committed,
                "event": attempt.event.to_manifest_record(),
                "source_trace_sha256": attempt.source_trace_sha256,
            }
            for attempt in sidecar.attempts
        ],
    }
    return _canonical_json_sha256(payload)


@dataclass(frozen=True)
class C0GripperControllerSidecarV1:
    """Per-episode sidecar of gripper controller switch attempts.

    Committed attempts reference dataset frames (``0 <= candidate_frame_index <
    frame_count``). An uncommitted failure attempt may use
    ``candidate_frame_index == frame_count`` to mark a candidate frame that
    occurred after the last saved frame and never entered the parquet data; it
    must never be presented as an existing dataset frame. ``finalized=False``
    sidecars preserve interrupted/failed capture evidence but cannot pass the
    final binding audit.

    ``sidecar_sha256`` is derived internally over the canonical payload (see
    ``_sidecar_sha256``) and can never be supplied by a caller.
    """

    schema_version: str
    episode_id: str
    episode_index: int
    fps: float
    frame_count: int
    dataset_gripper_slice_sha256: str
    attempts: Sequence[C0GripperControllerAttemptV1]
    finalized: bool
    _sidecar_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != C0_GRIPPER_CONTROLLER_SIDECAR_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_GRIPPER_CONTROLLER_SIDECAR_SCHEMA_VERSION!r}"
            )
        _require_nonempty_string("episode_id", self.episode_id)
        _require_non_negative_int("episode_index", self.episode_index)

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

        _require_sha256("dataset_gripper_slice_sha256", self.dataset_gripper_slice_sha256)

        if isinstance(self.attempts, (str, bytes)):
            raise ValueError("attempts must be a sequence of C0GripperControllerAttemptV1")
        attempts = tuple(self.attempts)
        if not all(isinstance(attempt, C0GripperControllerAttemptV1) for attempt in attempts):
            raise ValueError("attempts must contain only C0GripperControllerAttemptV1 records")
        object.__setattr__(self, "attempts", attempts)

        _validate_sidecar_attempts(self)

        if not isinstance(self.finalized, bool):
            raise ValueError("finalized must be bool")

        # Construction-time integrity gate: every attempt must pass the same
        # deterministic revalidation that later audits apply, so a corrupted
        # attempt can never enter a sidecar silently.
        for attempt in attempts:
            details = verify_c0_gripper_controller_attempt_integrity(attempt)
            if details:
                raise ValueError(
                    f"controller_attempt_integrity_mismatch at construction: {list(details)}"
                )

        object.__setattr__(self, "_sidecar_sha256", _sidecar_sha256(self))

    @property
    def sidecar_sha256(self) -> str:
        """Internally derived digest of the sidecar payload (never caller-supplied)."""

        return self._sidecar_sha256

    @property
    def committed_attempts(self) -> tuple[C0GripperControllerAttemptV1, ...]:
        return tuple(attempt for attempt in self.attempts if attempt.dataset_frame_committed)

    @property
    def failed_or_uncommitted_attempt_count(self) -> int:
        return sum(
            1
            for attempt in self.attempts
            if not attempt.dataset_frame_committed or not attempt.event.do_api_success
        )

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible record with fixed boundary fields.

        The sidecar is not wired into ``record_loop`` in this round, so
        ``live_capture_integration_verified`` stays False; serialization never
        grants capture, training, or execution authorization.

        Fail closed: if the current content no longer passes the deterministic
        integrity revalidation, this raises ValueError instead of serializing a
        record that would mix changed content with the pinned digest.
        """

        details = verify_c0_gripper_controller_sidecar_integrity(self)
        if details:
            raise ValueError(
                f"sidecar_integrity_mismatch; refusing to serialize a record "
                f"mixing changed content with the pinned digest: {list(details)}"
            )
        record = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "dataset_gripper_slice_sha256": self.dataset_gripper_slice_sha256,
            "sidecar_sha256": self._sidecar_sha256,
            "finalized": self.finalized,
            "attempt_count": len(self.attempts),
            "committed_attempt_count": len(self.committed_attempts),
            "failed_or_uncommitted_attempt_count": self.failed_or_uncommitted_attempt_count,
            "attempts": [attempt.to_manifest_record() for attempt in self.attempts],
            "physical_gripper_feedback_available": False,
            "physical_grasp_success_proven": False,
            "controller_event_origin_authenticated": False,
            "live_capture_integration_verified": False,
            "training_authorized": False,
            "policy_execution_authorized": False,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_serialization": False,
        }
        return copy.deepcopy(record)

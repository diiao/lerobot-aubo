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

"""Offline binding audit between a C0 dataset gripper slice and a controller sidecar.

This module cross-checks a frozen :class:`C0GripperDatasetAuditReportV1`
(dataset content evidence) against a :class:`C0GripperControllerSidecarV1`
(controller DO switch evidence) for the same episode. It is pure and
deterministic: it never reads or writes dataset files, never touches cameras,
AUBO, the network, threads, or sockets, and never starts capture, training, or
policy execution.

Trust boundaries:

- Only ``dataset_frame_committed=True`` success events are used for the
  parquet transition binding; uncommitted attempts preserve failure evidence
  and always block the binding.
- DO readback only proves the controller output register state. It is never
  vacuum pressure, physical jaw position, or grasp success feedback.
- The event timestamp tolerance reuses the dataset auditor convention of half
  a control cycle (``abs(timestamp - frame_index / fps) <= 0.5 / fps``), which
  at the frozen 25 Hz profile equals ``C0_TIMESTAMP_ABS_TOLERANCE_S``.
- ``complete_gripper_audit_ready`` is fixed False in this schema version: the
  sidecar is not wired into the real ``record_loop`` yet and the controller
  event origin is not authenticated. A structurally valid binding is never a
  training authorization.
- Frozen dataclasses do not freeze nested dicts/lists, so the audit first
  re-validates the CURRENT content of every attempt and of the sidecar itself
  (schema, recomputed digests, event-vs-trace consistency). Any mismatch
  yields stable error codes and never lets the binding pass.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Final

from .c0_gripper_controller_sidecar import (
    C0GripperControllerSidecarV1,
    verify_c0_gripper_controller_attempt_integrity,
    verify_c0_gripper_controller_sidecar_integrity,
)
from .c0_gripper_dataset_auditor import C0GripperDatasetAuditReportV1

C0_GRIPPER_CONTROLLER_BINDING_REPORT_SCHEMA_VERSION: Final = (
    "C0GripperControllerBindingReportV1"
)


def _sorted_unique(messages: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(messages)))


def _audit_binding(
    dataset_report: C0GripperDatasetAuditReportV1,
    sidecar: C0GripperControllerSidecarV1,
) -> dict[str, object]:
    """Run the full deterministic binding audit; pure with respect to its inputs."""

    errors: list[str] = []
    blockers: list[str] = []

    # 0. Post-construction integrity. Frozen dataclasses do not freeze nested
    # dicts/lists, so the current content of every attempt and of the sidecar
    # itself is re-validated before anything is trusted. A failure never
    # crashes the audit and never lets it pass: it yields stable error codes.
    for attempt in sidecar.attempts:
        if verify_c0_gripper_controller_attempt_integrity(attempt):
            errors.append("controller_attempt_integrity_mismatch")
    if verify_c0_gripper_controller_sidecar_integrity(sidecar):
        errors.append("sidecar_integrity_mismatch")

    # 1. The dataset report itself must be structurally valid and carry the
    # metadata the binding compares against.
    if not dataset_report.dataset_slice_structurally_valid:
        errors.append("dataset_report_structurally_invalid")
    report_digest = dataset_report.gripper_slice_sha256
    if not report_digest:
        errors.append("dataset_gripper_slice_digest_missing")
    report_frame_count = dataset_report.frame_count
    if report_frame_count is None:
        errors.append("dataset_frame_count_missing")
    report_fps = dataset_report.fps
    if report_fps is None:
        errors.append("dataset_fps_missing")

    # 2. Identity fields must agree exactly. fps uses strict float equality on
    # values already validated finite and positive by both constructors; the
    # isfinite guard keeps NaN from ever bypassing the comparison.
    if sidecar.episode_id != dataset_report.episode_id:
        errors.append("episode_id_mismatch")
    if sidecar.episode_index != dataset_report.episode_index:
        errors.append("episode_index_mismatch")
    if report_frame_count is not None and sidecar.frame_count != report_frame_count:
        errors.append("frame_count_mismatch")
    if report_fps is not None and (
        not math.isfinite(report_fps)
        or not math.isfinite(sidecar.fps)
        or float(report_fps) != float(sidecar.fps)
    ):
        errors.append("fps_mismatch")
    if report_digest and sidecar.dataset_gripper_slice_sha256 != report_digest:
        errors.append("dataset_gripper_slice_sha256_mismatch")

    # 3. Only finalized sidecars can pass the final binding.
    if not sidecar.finalized:
        errors.append("sidecar_not_finalized")

    attempts = sidecar.attempts
    committed = tuple(attempt for attempt in attempts if attempt.dataset_frame_committed)

    # 4. Uncommitted attempts and failed events preserve failure evidence; both
    # block the binding from passing.
    if any(not attempt.dataset_frame_committed for attempt in attempts):
        errors.append("uncommitted_controller_attempt")
    if any(not attempt.event.do_api_success for attempt in attempts):
        errors.append("controller_event_failure")

    # 5./6. Committed on/off frames must equal the dataset-derived transition
    # frames exactly: direction, frame numbers, and order — never just counts.
    committed_on_frames = tuple(
        attempt.event.frame_index
        for attempt in committed
        if attempt.event.event_type == "command_on"
    )
    committed_off_frames = tuple(
        attempt.event.frame_index
        for attempt in committed
        if attempt.event.event_type == "command_off"
    )
    expected_on = dataset_report.derived_command_on_frames
    expected_off = dataset_report.derived_command_off_frames
    expected_total = len(expected_on) + len(expected_off)
    if len(committed) < expected_total:
        errors.append("missing_committed_controller_event")
    elif len(committed) > expected_total:
        errors.append("unexpected_committed_controller_event")
    if committed_on_frames != expected_on or committed_off_frames != expected_off:
        if set(committed_on_frames) & set(expected_off) or set(committed_off_frames) & set(
            expected_on
        ):
            errors.append("controller_event_direction_mismatch")
        if len(committed_on_frames) == len(expected_on) and committed_on_frames != expected_on:
            errors.append("controller_event_frame_mismatch")
        if len(committed_off_frames) == len(expected_off) and committed_off_frames != expected_off:
            errors.append("controller_event_frame_mismatch")

    # The full ordered (frame_index, event_type) sequence of committed events
    # must equal the dataset-derived expectation item by item — per-direction
    # frame sets alone cannot catch a cross-direction order swap. The expected
    # sequence is the merge of both derived frame lists sorted by frame_index.
    # Defense in depth: this also fires for objects corrupted around the
    # sidecar constructor's strictly-increasing-frames rule.
    expected_sequence = tuple(
        sorted(
            [(frame, "command_on") for frame in expected_on]
            + [(frame, "command_off") for frame in expected_off]
        )
    )
    actual_sequence = tuple(
        (attempt.event.frame_index, attempt.event.event_type) for attempt in committed
    )
    if actual_sequence != expected_sequence and sorted(actual_sequence) == sorted(
        expected_sequence
    ):
        errors.append("controller_event_order_mismatch")

    # 7. Each committed event's dataset timestamp must reproduce
    # frame_index / fps within half a control cycle — the same convention the
    # frozen dataset auditor applies to parquet timestamps.
    timestamp_tolerance_s = 0.5 / sidecar.fps
    for attempt in committed:
        event = attempt.event
        expected_timestamp = event.frame_index / sidecar.fps
        if abs(event.sync_timestamp_s - expected_timestamp) > timestamp_tolerance_s:
            errors.append("controller_event_timestamp_mismatch")

    # 8./9. Controller output evidence. Readback-not-supported is expressible
    # but yields a stable blocker and never claims verified controller output.
    for attempt in committed:
        if not attempt.event.do_readback_supported:
            blockers.append("controller_readback_not_supported")
    for attempt in attempts:
        event = attempt.event
        if event.do_readback_supported and not event.controller_output_state_known:
            errors.append("controller_output_state_unknown")
        if event.controller_output_matches_requested is False:
            errors.append("controller_output_mismatch")

    failed_or_uncommitted_count = sum(
        1
        for attempt in attempts
        if not attempt.dataset_frame_committed or not attempt.event.do_api_success
    )

    return {
        "errors": _sorted_unique(errors),
        "blockers": _sorted_unique(blockers),
        "committed_on_frames": committed_on_frames,
        "committed_off_frames": committed_off_frames,
        "failed_or_uncommitted_attempt_count": failed_or_uncommitted_count,
    }


@dataclass(frozen=True)
class C0GripperControllerBindingReportV1:
    """Deterministic offline binding audit report for one episode.

    The constructor accepts only the schema version, the frozen dataset audit
    report, and the controller sidecar; every error, blocker, derived frame
    list, counter, and validity flag is computed internally and can never be
    supplied or forged by a caller.

    Lifecycle semantics — audit snapshot plus pre-use re-verification:

    - At construction both inputs are deep-copied into a snapshot that is
      fully detached from caller-held objects, and the audit runs against that
      snapshot. A binding report is a historical audit snapshot: later
      mutation of the caller's original sidecar can never rewrite it.
    - Frozen dataclasses do not freeze nested dicts/lists, so before any use
      (``errors``, the validity flags, ``to_manifest_record``) the report
      re-verifies the CURRENT integrity of its own snapshot with the
      deterministic sidecar integrity check. Tampering the report's own
      snapshot yields the stable error code ``binding_input_integrity_mismatch``,
      forces both validity flags to False, and makes ``to_manifest_record``
      fail closed with ValueError — a mixed record of stale audit results and
      changed content is never serialized. Every serialized field
      (``episode_id``, ``episode_index``, ``frame_count``, ``fps``,
      ``dataset_gripper_slice_sha256``, ``sidecar_sha256``,
      ``sidecar_finalized``, the committed frame lists, ``errors``,
      ``blockers``, and the validity flags) comes from that one snapshot.

    ``controller_event_binding_structurally_valid`` means the committed
    controller events correspond one-to-one with the dataset-derived gripper
    transitions in direction, frame, order, and timestamp, with no failure or
    uncommitted evidence outstanding. ``controller_output_evidence_complete``
    additionally requires every committed event to prove a successful DO write
    with supported, known, matching controller readback.
    ``complete_gripper_audit_ready`` stays False in this schema version: the
    sidecar is not yet wired into the real ``record_loop`` and the controller
    event origin is not authenticated. This report never authorizes capture,
    training, or policy execution and never proves a physical grasp.
    """

    schema_version: str
    dataset_report: C0GripperDatasetAuditReportV1
    controller_sidecar: C0GripperControllerSidecarV1
    _errors: tuple[str, ...] = field(init=False, repr=False)
    _blockers: tuple[str, ...] = field(init=False, repr=False)
    _committed_on_frames: tuple[int, ...] = field(init=False, repr=False)
    _committed_off_frames: tuple[int, ...] = field(init=False, repr=False)
    _failed_or_uncommitted_attempt_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != C0_GRIPPER_CONTROLLER_BINDING_REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_GRIPPER_CONTROLLER_BINDING_REPORT_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.dataset_report, C0GripperDatasetAuditReportV1):
            raise ValueError("dataset_report must be a C0GripperDatasetAuditReportV1")
        if not isinstance(self.controller_sidecar, C0GripperControllerSidecarV1):
            raise ValueError("controller_sidecar must be a C0GripperControllerSidecarV1")

        # Detach from the caller: the audit snapshot is a deep copy, so later
        # mutation of the caller's original objects cannot reach this report.
        object.__setattr__(self, "dataset_report", copy.deepcopy(self.dataset_report))
        object.__setattr__(self, "controller_sidecar", copy.deepcopy(self.controller_sidecar))

        outcome = _audit_binding(self.dataset_report, self.controller_sidecar)
        object.__setattr__(self, "_errors", outcome["errors"])
        object.__setattr__(self, "_blockers", outcome["blockers"])
        object.__setattr__(self, "_committed_on_frames", outcome["committed_on_frames"])
        object.__setattr__(self, "_committed_off_frames", outcome["committed_off_frames"])
        object.__setattr__(
            self,
            "_failed_or_uncommitted_attempt_count",
            outcome["failed_or_uncommitted_attempt_count"],
        )

    def _snapshot_integrity_details(self) -> tuple[str, ...]:
        """Re-verify the CURRENT integrity of the report's own audit snapshot."""

        return verify_c0_gripper_controller_sidecar_integrity(self.controller_sidecar)

    @property
    def errors(self) -> tuple[str, ...]:
        """Binding errors in stable sorted order.

        If the report's own snapshot was tampered with after construction, the
        stable code ``binding_input_integrity_mismatch`` is included.
        """

        if self._snapshot_integrity_details():
            return _sorted_unique([*self._errors, "binding_input_integrity_mismatch"])
        return self._errors

    @property
    def blockers(self) -> tuple[str, ...]:
        """Binding blockers in stable sorted order."""

        return self._blockers

    @property
    def committed_on_frames(self) -> tuple[int, ...]:
        return self._committed_on_frames

    @property
    def committed_off_frames(self) -> tuple[int, ...]:
        return self._committed_off_frames

    @property
    def failed_or_uncommitted_attempt_count(self) -> int:
        return self._failed_or_uncommitted_attempt_count

    @property
    def controller_event_binding_structurally_valid(self) -> bool:
        """Structural one-to-one binding between committed events and transitions.

        Uses the integrity-guarded ``errors`` property: a tampered snapshot
        always yields False.
        """

        return not self.errors and not self._blockers

    @property
    def controller_output_evidence_complete(self) -> bool:
        """Every committed event proves a successful, readback-verified DO switch."""

        if not self.controller_event_binding_structurally_valid:
            return False
        return all(
            attempt.event.do_api_success
            and attempt.event.do_readback_supported
            and attempt.event.controller_output_state_known
            and attempt.event.controller_output_matches_requested is True
            for attempt in self.controller_sidecar.committed_attempts
        )

    @property
    def complete_gripper_audit_ready(self) -> bool:
        """Always False: the sidecar is not wired into record_loop and the origin is unauthenticated."""

        return False

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible report that grants no authorization.

        Fail closed: if the report's own audit snapshot was tampered with after
        construction, this raises ValueError instead of serializing a mixed
        record of stale audit results and changed content.
        """

        details = self._snapshot_integrity_details()
        if details:
            raise ValueError(
                f"binding_input_integrity_mismatch; refusing to serialize a report "
                f"mixing stale audit results with changed content: {list(details)}"
            )
        sidecar = self.controller_sidecar
        record = {
            "schema_version": self.schema_version,
            "episode_id": sidecar.episode_id,
            "episode_index": sidecar.episode_index,
            "frame_count": sidecar.frame_count,
            "fps": sidecar.fps,
            "dataset_gripper_slice_sha256": sidecar.dataset_gripper_slice_sha256,
            "sidecar_sha256": sidecar.sidecar_sha256,
            "sidecar_finalized": sidecar.finalized,
            "committed_on_frames": list(self._committed_on_frames),
            "committed_off_frames": list(self._committed_off_frames),
            "failed_or_uncommitted_attempt_count": self._failed_or_uncommitted_attempt_count,
            "errors": list(self._errors),
            "blockers": list(self._blockers),
            "dataset_report_blockers": list(self.dataset_report.blockers),
            "controller_event_binding_structurally_valid": (
                self.controller_event_binding_structurally_valid
            ),
            "controller_output_evidence_complete": self.controller_output_evidence_complete,
            "controller_event_origin_authenticated": False,
            "live_capture_integration_verified": False,
            "complete_gripper_audit_ready": False,
            "physical_gripper_feedback_available": False,
            "physical_grasp_success_proven": False,
            "training_authorized": False,
            "policy_execution_authorized": False,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_audit": False,
            "dataset_files_modified_by_audit": False,
        }
        return copy.deepcopy(record)


def audit_c0_gripper_controller_binding(
    dataset_report: C0GripperDatasetAuditReportV1,
    controller_sidecar: C0GripperControllerSidecarV1,
) -> C0GripperControllerBindingReportV1:
    """Audit the binding between a dataset gripper slice report and a controller sidecar.

    Both inputs must already be constructed (the dataset report by the frozen
    read-only dataset auditor, the sidecar from validated AUBO traces). The
    audit is pure and deterministic: identical inputs always produce identical
    reports. A passing binding proves only structural correspondence between
    committed controller events and the saved gripper command transitions; it
    never proves a physical grasp and never authorizes training or policy
    execution.
    """

    if not isinstance(dataset_report, C0GripperDatasetAuditReportV1):
        raise TypeError("dataset_report must be a C0GripperDatasetAuditReportV1")
    if not isinstance(controller_sidecar, C0GripperControllerSidecarV1):
        raise TypeError("controller_sidecar must be a C0GripperControllerSidecarV1")
    return C0GripperControllerBindingReportV1(
        schema_version=C0_GRIPPER_CONTROLLER_BINDING_REPORT_SCHEMA_VERSION,
        dataset_report=dataset_report,
        controller_sidecar=controller_sidecar,
    )

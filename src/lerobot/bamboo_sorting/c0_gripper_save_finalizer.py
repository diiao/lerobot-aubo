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

"""C0-R4B post-save offline evidence promotion for one recorded episode.

Why two phases: ``LeRobotDataset.save_episode()`` appends to a persistently
open ParquetWriter that is only closed when the dataset is finalized, so the
saved episode is NOT re-readable from parquet at the moment ``save_episode``
returns. A single-phase "save then immediately audit" coordinator could never
be attached to the real save lifecycle. This module therefore splits the
promotion into two explicit phases:

1. ``begin_c0_gripper_episode_finalization`` runs BEFORE the save and proves
   everything that can be proven without touching disk: journal identity and
   health, episode id/index agreement, manifest frame count, the dataset's
   in-memory episode buffer size, and exact journal cycle coverage of frames
   ``0..frame_count-1``. Only then does it call ``dataset.save_episode()``.
   After a normal return the journal enters the ``saved_pending_audit``
   state and must never be saved again through this coordinator.

2. ``complete_c0_gripper_episode_finalization`` runs AFTER the caller has
   safely closed the writer / finalized the dataset (a public lifecycle
   step, never a private writer poke). It derives the audited root strictly
   from ``dataset.root``, re-reads the SAVED episode with the frozen R2
   auditor, cross-checks the pending save token against the live journal,
   rebuilds NEW committed attempts (the R4A pending attempts always stay
   ``dataset_frame_committed=False``), builds the finalized sidecar with the
   real R2 digest, and runs the frozen R3 binding audit. Audit-only retries
   are allowed while the journal is in ``saved_pending_audit``; a save that
   raised leaves the journal in ``save_uncertain``, a terminal state that
   requires manual recovery and never re-saves automatically.

State machine (stored as a private attribute on the journal, owned by this
module so the R4A journal file stays untouched):

    (absent)          -> save never invoked through this coordinator
    save_uncertain    -> save_episode() raised; terminal, manual recovery
    saved_pending_audit -> save returned; audit-only retries permitted
    finalized         -> promotion complete; any further call is refused

This is offline software evidence plumbing only: nothing here connects to a
robot or camera, performs DO/IO, moves an arm, trains, runs inference, or
grants capture/training/policy authorization. This module does not implement
any file writing itself, but it DOES explicitly invoke the public
``dataset.save_episode()``; all persistence semantics (buffer flush, parquet
writer lifetime, metadata) belong to the dataset implementation. A serialized
result grants no live authorization, and no software command, DO API success,
or controller readback is interpreted as physical gripper motion or grasp
success.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from lerobot.bamboo_sorting.c0_capture_contract import C0EpisodeManifestV1
from lerobot.bamboo_sorting.c0_gripper_capture_journal import (
    JOURNAL_STATUS_RECORDING,
    C0GripperPendingCaptureJournalV1,
    _require_pure_json,
    _validate_hold_trace,
)
from lerobot.bamboo_sorting.c0_gripper_controller_binding import (
    C0GripperControllerBindingReportV1,
    audit_c0_gripper_controller_binding,
)
from lerobot.bamboo_sorting.c0_gripper_controller_sidecar import (
    C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION,
    C0_GRIPPER_CONTROLLER_SIDECAR_SCHEMA_VERSION,
    C0GripperControllerAttemptV1,
    C0GripperControllerSidecarV1,
    verify_c0_gripper_controller_attempt_integrity,
)
from lerobot.bamboo_sorting.c0_gripper_dataset_auditor import (
    C0_GRIPPER_DATASET_AUDIT_REPORT_SCHEMA_VERSION,
    C0GripperDatasetAuditReportV1,
    audit_c0_gripper_dataset_episode,
)

C0_GRIPPER_SAVE_PENDING_AUDIT_SCHEMA_VERSION: Final = "C0GripperEpisodeSavePendingAuditV1"
C0_GRIPPER_SAVE_FINALIZATION_RESULT_SCHEMA_VERSION: Final = (
    "C0GripperSaveFinalizationResultV1"
)

# Journal state-machine attribute owned by this module (the R4A journal class
# is not modified).
_STATE_ATTR: Final = "_c0_r4b_state"
_SNAPSHOT_SHA_ATTR: Final = "_c0_r4b_snapshot_sha256"
_TOKEN_FINGERPRINT_ATTR: Final = "_c0_r4b_token_fingerprint"
_RESULT_FINGERPRINT_ATTR: Final = "_c0_r4b_result_fingerprint"
STATE_IDLE: Final = "idle"
STATE_SAVE_UNCERTAIN: Final = "save_uncertain"
STATE_SAVED_PENDING_AUDIT: Final = "saved_pending_audit"
STATE_FINALIZED: Final = "finalized"
_KNOWN_STATES: Final = frozenset(
    {STATE_IDLE, STATE_SAVE_UNCERTAIN, STATE_SAVED_PENDING_AUDIT, STATE_FINALIZED}
)


class C0GripperSaveFinalizationError(ValueError):
    """Fail-closed R4B error: no committed attempt or sidecar was produced."""


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _journal_snapshot_sha256(
    journal: C0GripperPendingCaptureJournalV1, *, frame_count: int
) -> str:
    """Deterministic digest of the ENTIRE save-time journal evidence.

    Covers every cycle field (including the traces and the dataset/sent
    gripper values), every attempt (event fields plus the pinned source-trace
    digest), and the episode identity, fps, and frame count. Any later
    mutation — even to another internally-coherent valid value — changes the
    digest and is rejected by the audit phase.
    """

    return _canonical_sha256(
        {
            "schema_version": "C0GripperJournalSnapshotV1",
            "episode_id": journal.episode_id,
            "episode_index": journal.episode_index,
            "fps": journal.fps,
            "frame_count": frame_count,
            "cycles": [dict(cycle) for cycle in journal.cycles],
            "attempts": [
                {
                    "attempt_index": attempt.attempt_index,
                    "candidate_frame_index": attempt.candidate_frame_index,
                    "dataset_frame_committed": attempt.dataset_frame_committed,
                    "event": dataclasses.asdict(attempt.event),
                    "source_trace_sha256": attempt.source_trace_sha256,
                }
                for attempt in journal.controller_attempts
            ],
        }
    )


def _token_fingerprint(token: C0GripperEpisodeSavePendingAuditV1) -> str:
    """Digest binding one pending token to one journal save attempt."""

    return _canonical_sha256(
        {
            "schema_version": token.schema_version,
            "episode_id": token.episode_id,
            "episode_index": token.episode_index,
            "fps": token.fps,
            "frame_count": token.frame_count,
            "dataset_root": token.dataset_root,
            "candidate_frame_indices": list(token.candidate_frame_indices),
        }
    )


def _result_fingerprint(result: C0GripperSaveFinalizationResultV1) -> str:
    """Digest binding one complete() result to the journal that produced it."""

    if not isinstance(result, C0GripperSaveFinalizationResultV1):
        raise TypeError("result must be a C0GripperSaveFinalizationResultV1")
    return _canonical_sha256(
        {
            "schema_version": "C0GripperSaveFinalizationResultFingerprintV1",
            "record": result.to_manifest_record(),
        }
    )


def _journal_state(journal: C0GripperPendingCaptureJournalV1) -> str:
    state = getattr(journal, _STATE_ATTR, STATE_IDLE)
    if state not in _KNOWN_STATES:
        raise C0GripperSaveFinalizationError(
            f"journal holds unknown save state {state!r}; refusing to guess"
        )
    return state


def _set_journal_state(journal: C0GripperPendingCaptureJournalV1, state: str) -> None:
    setattr(journal, _STATE_ATTR, state)


def _require_finalizable_journal(
    journal: Any,
) -> tuple[tuple[dict[str, Any], ...], tuple[C0GripperControllerAttemptV1, ...]]:
    """Shared journal health check; returns (cycles, attempts) snapshots."""

    if not isinstance(journal, C0GripperPendingCaptureJournalV1):
        raise TypeError("journal must be a C0GripperPendingCaptureJournalV1")
    if journal.status != JOURNAL_STATUS_RECORDING:
        raise C0GripperSaveFinalizationError(
            f"journal status is {journal.status!r}; only a recording journal can be finalized"
        )
    if journal.episode_failed:
        raise C0GripperSaveFinalizationError(
            "journal episode_failed is True; failed episodes cannot be finalized"
        )
    for cycle in journal.cycles:
        if cycle["frame_entered_episode_buffer"] is not True:
            raise C0GripperSaveFinalizationError(
                f"cycle {cycle['cycle_index']} never entered the episode buffer; "
                "the episode is not durably saveable"
            )
        if cycle["failure_stage"] is not None:
            raise C0GripperSaveFinalizationError(
                f"cycle {cycle['cycle_index']} holds failure_stage "
                f"{cycle['failure_stage']!r}; failed cycles block finalization"
            )
        if cycle["action_consistency"] != "matched":
            raise C0GripperSaveFinalizationError(
                f"cycle {cycle['cycle_index']} action consistency is not matched"
            )
        if cycle["trace_present"] is not True:
            raise C0GripperSaveFinalizationError(
                f"cycle {cycle['cycle_index']} has no trace evidence"
            )
    attempts = journal.controller_attempts
    attempts_by_index = {attempt.attempt_index: attempt for attempt in attempts}
    for cycle in journal.cycles:
        # The cycle trace and the attempt source trace were captured as two
        # deep copies of the same driver evidence; any divergence means the
        # journal content was tampered with after capture.
        attempt_index = cycle.get("attempt_index")
        if attempt_index is not None:
            attempt = attempts_by_index.get(attempt_index)
            if attempt is None or cycle["trace"] != attempt.source_trace:
                raise C0GripperSaveFinalizationError(
                    f"cycle {cycle['cycle_index']} trace contradicts attempt "
                    f"{attempt_index} source trace; journal evidence was tampered with"
                )
        elif cycle["trace"].get("requested_transition") is False:
            # Hold cycles keep their trace as the only evidence; re-validate
            # the full hold semantics so post-capture tampering fails closed.
            try:
                _validate_hold_trace(
                    cycle["trace"],
                    sent_gripper=cycle["sent_action_gripper_command"],
                )
            except ValueError as error:
                raise C0GripperSaveFinalizationError(
                    f"cycle {cycle['cycle_index']} hold trace was tampered with: {error}"
                ) from error
    for attempt in attempts:
        details = verify_c0_gripper_controller_attempt_integrity(attempt)
        if details:
            raise C0GripperSaveFinalizationError(
                f"pending attempt {attempt.attempt_index} is corrupted: {list(details)}"
            )
        if attempt.dataset_frame_committed is not False:
            raise C0GripperSaveFinalizationError(
                f"pending attempt {attempt.attempt_index} is already committed; "
                "R4A pending attempts must stay uncommitted"
            )
        if attempt.event.do_api_success is not True:
            raise C0GripperSaveFinalizationError(
                f"pending attempt {attempt.attempt_index} is a failed event; "
                "failed controller evidence blocks finalization"
            )
    return tuple(journal.cycles), attempts


def _require_identity_agreement(
    *,
    journal: C0GripperPendingCaptureJournalV1,
    episode_index: int,
    episode_manifest: C0EpisodeManifestV1,
) -> int:
    """Pre-save identity and size agreement; nothing here touches the disk."""

    if not isinstance(episode_manifest, C0EpisodeManifestV1):
        raise TypeError("episode_manifest must be a C0EpisodeManifestV1")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError("episode_index must be a non-negative integer, not bool")
    frame_count = episode_manifest.frame_count
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise C0GripperSaveFinalizationError(
            "episode manifest frame_count must be a positive integer"
        )
    if journal.episode_index != episode_index:
        raise C0GripperSaveFinalizationError(
            f"journal episode_index {journal.episode_index} != explicit episode_index "
            f"{episode_index}"
        )
    if journal.episode_id != episode_manifest.episode_id:
        raise C0GripperSaveFinalizationError(
            f"journal episode_id {journal.episode_id!r} != manifest episode_id "
            f"{episode_manifest.episode_id!r}"
        )
    if not math.isfinite(journal.fps) or float(journal.fps) != float(episode_manifest.fps):
        raise C0GripperSaveFinalizationError(
            f"journal fps {journal.fps} != manifest fps {episode_manifest.fps}"
        )
    return frame_count


def _require_exact_frame_coverage(
    cycles: tuple[dict[str, Any], ...], *, frame_count: int
) -> tuple[int, ...]:
    """Journal cycles must cover exactly frames 0..frame_count-1, once each."""

    candidates = tuple(cycle["candidate_frame_index"] for cycle in cycles)
    expected = tuple(range(frame_count))
    if len(cycles) != frame_count or candidates != expected:
        raise C0GripperSaveFinalizationError(
            f"journal cycles must cover exactly frames 0..{frame_count - 1} once each: "
            f"got {len(cycles)} cycles with candidates {list(candidates)}"
        )
    return candidates


def _dataset_root(dataset: Any) -> Path:
    """Canonical resolved root of the dataset that is actually being saved.

    The audited root is ALWAYS derived from the saved dataset itself and is
    resolved with ``strict=True`` so a symlink retargeted between the save
    and the audit changes the canonical path and fails closed.
    """

    root = getattr(dataset, "root", None)
    if not isinstance(root, (str, Path)) or not str(root):
        raise TypeError("dataset must expose a non-empty root path")
    try:
        return Path(root).resolve(strict=True)
    except OSError as error:
        raise C0GripperSaveFinalizationError(
            f"dataset root {str(root)!r} does not resolve to an existing path: {error}"
        ) from error


def _require_saveable_buffer(dataset: Any, *, episode_index: int, frame_count: int) -> None:
    """Pre-save: the in-memory episode buffer must hold exactly this episode."""

    save_episode = getattr(dataset, "save_episode", None)
    if not callable(save_episode):
        raise TypeError("dataset must provide a callable save_episode()")
    buffer = getattr(dataset, "episode_buffer", None)
    if not isinstance(buffer, dict) or "size" not in buffer:
        raise C0GripperSaveFinalizationError(
            "dataset episode_buffer is not open; there is no episode to save"
        )
    if "episode_index" not in buffer:
        raise C0GripperSaveFinalizationError(
            "dataset episode_buffer has no episode_index; refusing to guess which "
            "episode would be saved"
        )
    size = buffer["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size != frame_count:
        raise C0GripperSaveFinalizationError(
            f"dataset episode_buffer size {size!r} != manifest frame_count {frame_count}; "
            "refusing to save an episode that does not match the manifest"
        )
    buffer_index = buffer["episode_index"]
    if (
        isinstance(buffer_index, bool)
        or not isinstance(buffer_index, int)
        or buffer_index != episode_index
    ):
        raise C0GripperSaveFinalizationError(
            f"dataset episode_buffer episode_index {buffer_index!r} != explicit "
            f"episode_index {episode_index}; refusing to save the wrong episode"
        )
    meta = getattr(dataset, "meta", None)
    if meta is not None:
        total = getattr(meta, "total_episodes", None)
        if isinstance(total, bool) or not isinstance(total, int):
            raise C0GripperSaveFinalizationError(
                "dataset meta.total_episodes is missing or not an integer"
            )
        if total != episode_index:
            raise C0GripperSaveFinalizationError(
                f"dataset meta.total_episodes {total} != explicit episode_index "
                f"{episode_index}; the buffer would be saved as a different episode"
            )


def _require_consistent_report(
    *,
    journal: C0GripperPendingCaptureJournalV1,
    report: C0GripperDatasetAuditReportV1,
    episode_index: int,
    episode_manifest: C0EpisodeManifestV1,
) -> None:
    """Fail closed unless the frozen R2 report is valid and identity-consistent."""

    if report.schema_version != C0_GRIPPER_DATASET_AUDIT_REPORT_SCHEMA_VERSION:
        raise C0GripperSaveFinalizationError("dataset audit report schema version mismatch")
    if not report.dataset_slice_structurally_valid:
        raise C0GripperSaveFinalizationError(
            f"dataset audit report is structurally invalid: {list(report.errors)}"
        )
    digest = report.gripper_slice_sha256
    if not isinstance(digest, str) or not digest:
        raise C0GripperSaveFinalizationError(
            "dataset audit report has no gripper slice digest; refusing to fabricate one"
        )
    if report.frame_count is None:
        raise C0GripperSaveFinalizationError("dataset audit report has no frame_count")
    if report.fps is None:
        raise C0GripperSaveFinalizationError("dataset audit report has no fps")
    if report.episode_id != journal.episode_id:
        raise C0GripperSaveFinalizationError(
            f"dataset report episode_id {report.episode_id!r} != journal episode_id "
            f"{journal.episode_id!r}"
        )
    if episode_manifest.episode_id != journal.episode_id:
        raise C0GripperSaveFinalizationError(
            f"episode manifest episode_id {episode_manifest.episode_id!r} != journal "
            f"episode_id {journal.episode_id!r}"
        )
    if report.episode_index != episode_index:
        raise C0GripperSaveFinalizationError(
            f"dataset report episode_index {report.episode_index} != explicit "
            f"episode_index {episode_index}"
        )
    if journal.episode_index != episode_index:
        raise C0GripperSaveFinalizationError(
            f"journal episode_index {journal.episode_index} != explicit episode_index "
            f"{episode_index}"
        )
    if float(report.fps) != journal.fps:
        raise C0GripperSaveFinalizationError(
            f"dataset report fps {report.fps} != journal fps {journal.fps}"
        )
    if episode_manifest.frame_count != report.frame_count:
        raise C0GripperSaveFinalizationError(
            f"episode manifest frame_count {episode_manifest.frame_count} != dataset "
            f"report frame_count {report.frame_count}"
        )


def _require_transition_correspondence(
    *,
    attempts: tuple[C0GripperControllerAttemptV1, ...],
    report: C0GripperDatasetAuditReportV1,
) -> None:
    """Pending transition attempts must match dataset-derived transitions 1:1."""

    expected_sequence = tuple(
        sorted(
            [(frame, "command_on") for frame in report.derived_command_on_frames]
            + [(frame, "command_off") for frame in report.derived_command_off_frames]
        )
    )
    actual_sequence = tuple(
        (attempt.event.frame_index, attempt.event.event_type) for attempt in attempts
    )
    if actual_sequence != expected_sequence:
        raise C0GripperSaveFinalizationError(
            "pending controller events do not correspond one-to-one with the "
            f"dataset-derived transitions: journal {list(actual_sequence)} vs dataset "
            f"{list(expected_sequence)}"
        )


def _build_committed_attempts(
    attempts: tuple[C0GripperControllerAttemptV1, ...],
) -> tuple[C0GripperControllerAttemptV1, ...]:
    """Rebuild NEW committed attempts; never mutate the pending ones in place."""

    committed: list[C0GripperControllerAttemptV1] = []
    for position, pending in enumerate(attempts):
        committed.append(
            C0GripperControllerAttemptV1(
                schema_version=C0_GRIPPER_CONTROLLER_ATTEMPT_SCHEMA_VERSION,
                attempt_index=position,
                candidate_frame_index=pending.candidate_frame_index,
                dataset_frame_committed=True,
                event=copy.deepcopy(pending.event),
                source_trace=copy.deepcopy(pending.source_trace),
            )
        )
    return tuple(committed)


@dataclasses.dataclass(frozen=True)
class C0GripperEpisodeSavePendingAuditV1:
    """Proof that the save was invoked and returned; consumed by the audit phase.

    The token snapshots every pre-save validation result so the audit phase
    can prove the episode being audited is the same episode that was saved:
    identity, frame count, cycle coverage, and the exact dataset root that was
    saved. A journal can hold at most one live token at a time.
    """

    schema_version: str
    episode_id: str
    episode_index: int
    fps: float
    frame_count: int
    dataset_root: str
    candidate_frame_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != C0_GRIPPER_SAVE_PENDING_AUDIT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_GRIPPER_SAVE_PENDING_AUDIT_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, int)
            or self.episode_index < 0
        ):
            raise ValueError("episode_index must be a non-negative integer, not bool")
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count <= 0
        ):
            raise ValueError("frame_count must be a positive integer, not bool")
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise ValueError("fps must be a finite positive number, not bool")
        fps = float(self.fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a finite positive number")
        object.__setattr__(self, "fps", fps)
        if not isinstance(self.dataset_root, str) or not self.dataset_root:
            raise ValueError("dataset_root must be a non-empty string path")
        if isinstance(self.candidate_frame_indices, (str, bytes)) or not isinstance(
            self.candidate_frame_indices, (tuple, list)
        ):
            raise ValueError("candidate_frame_indices must be a tuple or list of integers")
        candidates = tuple(self.candidate_frame_indices)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in candidates):
            raise ValueError("candidate_frame_indices must contain only integers, not bool")
        if candidates != tuple(range(self.frame_count)):
            raise ValueError(
                "candidate_frame_indices must be exactly 0..frame_count-1 in order, "
                f"got {list(candidates)} for frame_count {self.frame_count}"
            )
        object.__setattr__(self, "candidate_frame_indices", candidates)

    def to_manifest_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "dataset_root": self.dataset_root,
            "candidate_frame_indices": list(self.candidate_frame_indices),
            "save_episode_returned": True,
            "dataset_episode_durably_saved": False,
            "final_sidecar_published": False,
            "physical_gripper_feedback_available": False,
            "physical_grasp_success_proven": False,
            "training_authorized": False,
            "policy_execution_authorized": False,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_serialization": False,
        }
        _require_pure_json("pending_audit_record", record)
        return copy.deepcopy(record)


@dataclasses.dataclass(frozen=True)
class C0GripperSaveFinalizationResultV1:
    """Deterministic offline evidence snapshot of one promoted episode save.

    Cross-validated at construction: top-level identity fields must agree
    with the sidecar and the dataset report, the sidecar must be finalized
    with the report digest, and the binding report must be error-free over
    the same objects. Both the underlying objects and the serialized record
    are detached deep copies with every authorization/physical field False.
    """

    schema_version: str
    episode_id: str
    episode_index: int
    fps: float
    frame_count: int
    dataset_gripper_slice_sha256: str
    sidecar: C0GripperControllerSidecarV1
    dataset_report: C0GripperDatasetAuditReportV1
    binding_report: C0GripperControllerBindingReportV1

    def __post_init__(self) -> None:
        if self.schema_version != C0_GRIPPER_SAVE_FINALIZATION_RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_GRIPPER_SAVE_FINALIZATION_RESULT_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.sidecar, C0GripperControllerSidecarV1):
            raise ValueError("sidecar must be a C0GripperControllerSidecarV1")
        if not isinstance(self.dataset_report, C0GripperDatasetAuditReportV1):
            raise ValueError("dataset_report must be a C0GripperDatasetAuditReportV1")
        if not isinstance(self.binding_report, C0GripperControllerBindingReportV1):
            raise ValueError("binding_report must be a C0GripperControllerBindingReportV1")
        # Cross-validation: the result can never present mutually contradictory
        # top-level fields, sidecar, dataset report, and binding report.
        if self.binding_report.errors:
            raise ValueError(
                f"binding report holds structural errors: {list(self.binding_report.errors)}"
            )
        if self.sidecar.finalized is not True:
            raise ValueError("sidecar must be finalized in a finalization result")
        if self.sidecar.episode_id != self.episode_id:
            raise ValueError("top-level episode_id contradicts sidecar episode_id")
        if self.sidecar.episode_index != self.episode_index:
            raise ValueError("top-level episode_index contradicts sidecar episode_index")
        if self.sidecar.frame_count != self.frame_count:
            raise ValueError("top-level frame_count contradicts sidecar frame_count")
        if not math.isfinite(self.fps) or float(self.sidecar.fps) != float(self.fps):
            raise ValueError("top-level fps contradicts sidecar fps")
        if self.sidecar.dataset_gripper_slice_sha256 != self.dataset_gripper_slice_sha256:
            raise ValueError("top-level digest contradicts sidecar digest")
        if self.dataset_report.episode_id != self.episode_id:
            raise ValueError("dataset report episode_id contradicts top-level episode_id")
        if self.dataset_report.episode_index != self.episode_index:
            raise ValueError("dataset report episode_index contradicts top-level episode_index")
        if self.dataset_report.frame_count != self.frame_count:
            raise ValueError("dataset report frame_count contradicts top-level frame_count")
        if self.dataset_report.gripper_slice_sha256 != self.dataset_gripper_slice_sha256:
            raise ValueError("dataset report digest contradicts top-level digest")
        # The binding audit must be over THIS sidecar and THIS dataset report.
        if self.binding_report.controller_sidecar.sidecar_sha256 != self.sidecar.sidecar_sha256:
            raise ValueError("binding report was not produced over this sidecar")
        if (
            self.binding_report.dataset_report.gripper_slice_sha256
            != self.dataset_gripper_slice_sha256
        ):
            raise ValueError("binding report was not produced over this dataset report")
        object.__setattr__(self, "sidecar", copy.deepcopy(self.sidecar))
        object.__setattr__(self, "dataset_report", copy.deepcopy(self.dataset_report))
        object.__setattr__(self, "binding_report", copy.deepcopy(self.binding_report))

    @property
    def structurally_finalized(self) -> bool:
        """Aligned with R3 ``controller_event_binding_structurally_valid``.

        A binding blocker (for example ``controller_readback_not_supported``)
        therefore makes this False instead of contradicting the binding
        report; the blockers are preserved truthfully in the record.
        """

        return self.binding_report.controller_event_binding_structurally_valid

    def to_manifest_record(self) -> dict[str, Any]:
        """Detached pure-JSON record; recursive authorization fields stay False."""

        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "dataset_gripper_slice_sha256": self.dataset_gripper_slice_sha256,
            "structurally_finalized": self.structurally_finalized,
            "binding_errors": list(self.binding_report.errors),
            "binding_blockers": list(self.binding_report.blockers),
            "sidecar": self.sidecar.to_manifest_record(),
            "dataset_report": self.dataset_report.to_manifest_record(),
            "binding_report": self.binding_report.to_manifest_record(),
            # Fixed boundary fields: software evidence never grants authorization
            # and never claims physical gripper feedback or grasp success.
            "physical_gripper_feedback_available": False,
            "physical_grasp_success_proven": False,
            "training_authorized": False,
            "policy_execution_authorized": False,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_serialization": False,
        }
        _require_pure_json("finalization_record", record)
        return copy.deepcopy(record)


def c0_gripper_journal_save_state(journal: C0GripperPendingCaptureJournalV1) -> str:
    """Read-only current R4B save state. Never writes, never saves, never audits."""

    if not isinstance(journal, C0GripperPendingCaptureJournalV1):
        raise TypeError("journal must be a C0GripperPendingCaptureJournalV1")
    return _journal_state(journal)


def require_c0_gripper_journal_save_state(
    journal: C0GripperPendingCaptureJournalV1, *, expected: str
) -> str:
    """Fail-closed read of the journal's R4B save state."""

    if expected not in _KNOWN_STATES:
        raise C0GripperSaveFinalizationError(
            f"unknown expected save state {expected!r}; refusing to guess"
        )
    state = c0_gripper_journal_save_state(journal)
    if state != expected:
        raise C0GripperSaveFinalizationError(
            f"journal save state is {state!r}; expected {expected!r}"
        )
    return state


def require_c0_gripper_pending_token_bound_to_journal(
    *,
    journal: C0GripperPendingCaptureJournalV1,
    pending: C0GripperEpisodeSavePendingAuditV1,
) -> None:
    """Prove ``pending`` is the token pinned by this journal's begin phase.

    Compares the deterministic token fingerprint and the save-time journal
    snapshot digest. Does not write, save, or re-run the R2 dataset audit.
    """

    if not isinstance(pending, C0GripperEpisodeSavePendingAuditV1):
        raise TypeError("pending must be a C0GripperEpisodeSavePendingAuditV1")
    state = c0_gripper_journal_save_state(journal)
    if state not in {STATE_SAVED_PENDING_AUDIT, STATE_FINALIZED}:
        raise C0GripperSaveFinalizationError(
            f"journal save state is {state!r}; token binding requires "
            f"{STATE_SAVED_PENDING_AUDIT!r} or {STATE_FINALIZED!r}"
        )
    if pending.episode_id != journal.episode_id or pending.episode_index != journal.episode_index:
        raise C0GripperSaveFinalizationError(
            "pending token identity does not match the journal"
        )
    if not math.isfinite(pending.fps) or float(pending.fps) != float(journal.fps):
        raise C0GripperSaveFinalizationError(
            f"pending token fps {pending.fps} != journal fps {journal.fps}"
        )
    if _token_fingerprint(pending) != getattr(journal, _TOKEN_FINGERPRINT_ATTR, None):
        raise C0GripperSaveFinalizationError(
            "pending token is not the one produced by this journal's save; "
            "refusing a reconstructed or foreign token"
        )
    current_snapshot = _journal_snapshot_sha256(journal, frame_count=pending.frame_count)
    if current_snapshot != getattr(journal, _SNAPSHOT_SHA_ATTR, None):
        raise C0GripperSaveFinalizationError(
            "journal evidence does not match the save-time snapshot digest; "
            "refusing a journal that is not the one that was saved"
        )


def require_c0_gripper_finalized_evidence_bound(
    *,
    journal: C0GripperPendingCaptureJournalV1,
    pending: C0GripperEpisodeSavePendingAuditV1,
    result: C0GripperSaveFinalizationResultV1,
) -> None:
    """Prove journal, pending token, and result are one save/audit chain.

    Read-only: does not write, save, or re-run the R2 dataset audit.
    """

    if not isinstance(result, C0GripperSaveFinalizationResultV1):
        raise TypeError("result must be a C0GripperSaveFinalizationResultV1")
    require_c0_gripper_journal_save_state(journal, expected=STATE_FINALIZED)
    require_c0_gripper_pending_token_bound_to_journal(journal=journal, pending=pending)
    if str(pending.dataset_root) != str(result.dataset_report.dataset_root):
        raise C0GripperSaveFinalizationError(
            "pending token dataset_root does not match finalization result "
            "dataset_report.dataset_root; refusing a mixed save/audit chain"
        )
    pending_attempts = journal.controller_attempts
    committed_attempts = result.sidecar.attempts
    if len(pending_attempts) != len(committed_attempts):
        raise C0GripperSaveFinalizationError(
            "journal pending attempts and finalized committed attempts differ in count"
        )
    for pending_attempt, committed_attempt in zip(
        pending_attempts, committed_attempts, strict=True
    ):
        if pending_attempt.attempt_index != committed_attempt.attempt_index:
            raise C0GripperSaveFinalizationError(
                "committed attempt_index does not match the journal pending attempt"
            )
        if pending_attempt.candidate_frame_index != committed_attempt.candidate_frame_index:
            raise C0GripperSaveFinalizationError(
                "committed candidate_frame_index does not match the journal pending attempt"
            )
        if dataclasses.asdict(pending_attempt.event) != dataclasses.asdict(
            committed_attempt.event
        ):
            raise C0GripperSaveFinalizationError(
                "committed attempt event does not match the journal pending attempt"
            )
        if pending_attempt.source_trace_sha256 != committed_attempt.source_trace_sha256:
            raise C0GripperSaveFinalizationError(
                "committed source_trace_sha256 does not match the journal pending attempt"
            )
        if pending_attempt.dataset_frame_committed is not False:
            raise C0GripperSaveFinalizationError(
                "journal pending attempts must remain dataset_frame_committed=False"
            )
        if committed_attempt.dataset_frame_committed is not True:
            raise C0GripperSaveFinalizationError(
                "finalized committed attempts must have dataset_frame_committed=True"
            )
    pinned = getattr(journal, _RESULT_FINGERPRINT_ATTR, None)
    if not isinstance(pinned, str) or not pinned:
        raise C0GripperSaveFinalizationError(
            "journal has no finalization result fingerprint; refusing an unbound "
            "finalized journal"
        )
    if _result_fingerprint(result) != pinned:
        raise C0GripperSaveFinalizationError(
            "finalization result fingerprint does not match the journal's pinned "
            "result; refusing a reconstructed or foreign result"
        )


def begin_c0_gripper_episode_finalization(
    *,
    journal: C0GripperPendingCaptureJournalV1,
    dataset: Any,
    episode_index: int,
    episode_manifest: C0EpisodeManifestV1,
) -> C0GripperEpisodeSavePendingAuditV1:
    """Pre-save validation phase; performs the one and only allowed save call.

    Everything checkable without touching disk is proven BEFORE
    ``dataset.save_episode()`` runs: journal identity and health, episode
    id/index agreement, manifest frame count, the dataset's episode buffer
    size AND buffer episode index, and exact cycle coverage of frames
    ``0..frame_count-1``. A deterministic digest of the ENTIRE journal
    evidence is pinned before the save. The journal enters ``save_uncertain``
    BEFORE the save call, so any exception — including KeyboardInterrupt —
    leaves it there (terminal; manual recovery; never an automatic re-save).
    Only a normal return moves it to ``saved_pending_audit`` and produces the
    token the audit phase must consume.
    """

    if _journal_state(journal) != STATE_IDLE:
        raise C0GripperSaveFinalizationError(
            f"journal save state is {_journal_state(journal)!r}; a save was already "
            "invoked through this coordinator and must not be repeated"
        )
    cycles, _ = _require_finalizable_journal(journal)
    frame_count = _require_identity_agreement(
        journal=journal, episode_index=episode_index, episode_manifest=episode_manifest
    )
    candidates = _require_exact_frame_coverage(cycles, frame_count=frame_count)
    _require_saveable_buffer(dataset, episode_index=episode_index, frame_count=frame_count)
    root = _dataset_root(dataset)
    snapshot_sha256 = _journal_snapshot_sha256(journal, frame_count=frame_count)

    _set_journal_state(journal, STATE_SAVE_UNCERTAIN)
    dataset.save_episode()

    pending = C0GripperEpisodeSavePendingAuditV1(
        schema_version=C0_GRIPPER_SAVE_PENDING_AUDIT_SCHEMA_VERSION,
        episode_id=journal.episode_id,
        episode_index=journal.episode_index,
        fps=journal.fps,
        frame_count=frame_count,
        dataset_root=str(root),
        candidate_frame_indices=candidates,
    )
    setattr(journal, _SNAPSHOT_SHA_ATTR, snapshot_sha256)
    setattr(journal, _TOKEN_FINGERPRINT_ATTR, _token_fingerprint(pending))
    _set_journal_state(journal, STATE_SAVED_PENDING_AUDIT)
    return pending


def complete_c0_gripper_episode_finalization(
    *,
    journal: C0GripperPendingCaptureJournalV1,
    dataset: Any,
    pending: C0GripperEpisodeSavePendingAuditV1,
    episode_manifest: C0EpisodeManifestV1,
) -> C0GripperSaveFinalizationResultV1:
    """Post-close audit phase; NEVER saves again.

    Call this only after the ParquetWriter / dataset has been safely closed
    through its public lifecycle. The audited root is derived from
    ``dataset.root`` and must equal the root recorded in the pending token,
    so the episode being audited is structurally the episode that was saved.
    Audit or binding failures raise and leave the journal in
    ``saved_pending_audit``, permitting audit-only retries; nothing here
    calls ``save_episode()`` again.
    """

    if not isinstance(pending, C0GripperEpisodeSavePendingAuditV1):
        raise TypeError("pending must be a C0GripperEpisodeSavePendingAuditV1")
    state = _journal_state(journal)
    if state != STATE_SAVED_PENDING_AUDIT:
        raise C0GripperSaveFinalizationError(
            f"journal save state is {state!r}; the audit phase requires "
            f"{STATE_SAVED_PENDING_AUDIT!r}"
        )
    if _token_fingerprint(pending) != getattr(journal, _TOKEN_FINGERPRINT_ATTR, None):
        raise C0GripperSaveFinalizationError(
            "pending token is not the one produced by this journal's save; "
            "refusing a reconstructed or foreign token"
        )
    if pending.episode_id != journal.episode_id or pending.episode_index != journal.episode_index:
        raise C0GripperSaveFinalizationError(
            "pending token identity does not match the journal"
        )
    if not isinstance(episode_manifest, C0EpisodeManifestV1):
        raise TypeError("episode_manifest must be a C0EpisodeManifestV1")
    if episode_manifest.episode_id != journal.episode_id:
        raise C0GripperSaveFinalizationError(
            "episode manifest episode_id changed between save and audit"
        )
    if episode_manifest.frame_count != pending.frame_count:
        raise C0GripperSaveFinalizationError(
            "episode manifest frame_count changed between save and audit"
        )
    if not math.isfinite(pending.fps) or float(pending.fps) != float(journal.fps):
        raise C0GripperSaveFinalizationError(
            f"pending token fps {pending.fps} != journal fps {journal.fps}"
        )
    if float(pending.fps) != float(episode_manifest.fps):
        raise C0GripperSaveFinalizationError(
            f"pending token fps {pending.fps} != manifest fps {episode_manifest.fps}"
        )
    root = _dataset_root(dataset)
    if str(root) != pending.dataset_root:
        raise C0GripperSaveFinalizationError(
            f"resolved dataset root {str(root)!r} no longer matches the saved root "
            f"{pending.dataset_root!r}; refusing to audit a different dataset"
        )

    # Re-validate the live journal and prove it is BIT-IDENTICAL to the
    # save-time evidence: the deterministic snapshot digest must match, not
    # merely be internally coherent.
    cycles, pending_attempts = _require_finalizable_journal(journal)
    candidates = _require_exact_frame_coverage(cycles, frame_count=pending.frame_count)
    if candidates != pending.candidate_frame_indices:
        raise C0GripperSaveFinalizationError(
            "journal cycles changed between save and audit; refusing to bind "
            "stale save evidence to new journal content"
        )
    current_snapshot = _journal_snapshot_sha256(journal, frame_count=pending.frame_count)
    if current_snapshot != getattr(journal, _SNAPSHOT_SHA_ATTR, None):
        raise C0GripperSaveFinalizationError(
            "journal evidence changed after the save (snapshot digest mismatch); "
            "refusing to bind stale save evidence to modified journal content"
        )

    # Re-read the SAVED episode from parquet with the frozen read-only R2
    # auditor; the digest and transition frames are derived there, never
    # computed from the in-memory buffer and never supplied by the caller.
    report = audit_c0_gripper_dataset_episode(
        root,
        pending.episode_index,
        episode_manifest,
    )
    _require_consistent_report(
        journal=journal,
        report=report,
        episode_index=pending.episode_index,
        episode_manifest=episode_manifest,
    )
    _require_transition_correspondence(attempts=pending_attempts, report=report)

    # Only now may committed attempts exist, as NEW objects built from the
    # pending ones; the R4A journal and its attempts are never modified.
    committed_attempts = _build_committed_attempts(pending_attempts)
    assert report.frame_count is not None
    assert report.fps is not None
    sidecar = C0GripperControllerSidecarV1(
        schema_version=C0_GRIPPER_CONTROLLER_SIDECAR_SCHEMA_VERSION,
        episode_id=journal.episode_id,
        episode_index=pending.episode_index,
        fps=float(report.fps),
        frame_count=report.frame_count,
        dataset_gripper_slice_sha256=report.gripper_slice_sha256,
        attempts=committed_attempts,
        finalized=True,
    )

    # Frozen R3 binding audit. Structural errors fail closed; blockers are
    # preserved truthfully in the result and never forged away.
    binding_report = audit_c0_gripper_controller_binding(report, sidecar)
    if binding_report.errors:
        raise C0GripperSaveFinalizationError(
            f"controller binding audit failed with structural errors: "
            f"{list(binding_report.errors)}"
        )

    result = C0GripperSaveFinalizationResultV1(
        schema_version=C0_GRIPPER_SAVE_FINALIZATION_RESULT_SCHEMA_VERSION,
        episode_id=journal.episode_id,
        episode_index=pending.episode_index,
        fps=float(report.fps),
        frame_count=report.frame_count,
        dataset_gripper_slice_sha256=report.gripper_slice_sha256,
        sidecar=sidecar,
        dataset_report=report,
        binding_report=binding_report,
    )
    setattr(journal, _RESULT_FINGERPRINT_ATTR, _result_fingerprint(result))
    _set_journal_state(journal, STATE_FINALIZED)
    return result

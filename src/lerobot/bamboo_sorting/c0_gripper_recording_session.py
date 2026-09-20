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

"""C0-R4C offline multi-episode recording session coordinator.

This module is the C0-side implementation of the generic
``RecordEpisodeLifecycleObserver`` hook surface of ``lerobot_record``. It wires
one explicitly provided sequence of ``C0EpisodeManifestV1`` objects into the
per-episode capture/save/finalize lifecycle:

1. Each episode attempt opens exactly one fresh R4A pending capture journal
   (created from the manifest at that episode index — never guessed), which
   the generic record loop passes to ``record_loop`` as the frame observer.
2. A rerecord abandons the open journal (no new cycles allowed) and the next
   attempt opens a fully isolated journal with no cycles/events/attempts from
   the abandoned one. The generic loop performs the single
   ``dataset.clear_episode_buffer()`` call.
    3. A completed episode is saved exclusively through the R4B two-phase
    coordinator: ``begin_c0_gripper_episode_finalization`` runs every
    pre-save validation (journal identity/health, manifest id/index/frame
    count agreement, exact cycle coverage, buffer size/index) and performs the
    one and only ``dataset.save_episode()`` call; on success the session holds
    the pending audit token. A pre-save validation refusal is
    ``pre_save_rejected`` (the save call never started). An exception after
    ``dataset.save_episode`` has been entered is ``save_uncertain``. Neither
    outcome retries the save or counts the episode as recorded.
4. Only after the dataset finalized/closed its writer through the public
   lifecycle does the session run ``complete_c0_gripper_episode_finalization``
   per saved episode, in save order, without ever saving again. The first
   complete failure stops the batch deterministically: later promotions are
   skipped, their tokens stay ``saved_pending_audit``, and the session never
   claims batch completion. A finalize failure keeps every token pending and
   fabricates nothing.
5. A stopped or failed episode is marked incomplete and is never saved;
   tokens of previously saved episodes are never dropped.

Trust boundaries: the coordinator never connects to a robot, camera, or
dataset by itself and never performs DO/IO, motion, or a real recording; it is
driven by the generic record loop. Every manifest must be provided explicitly
by the caller — episode ids come from manifests, episode indexes are
positional in the manifest sequence, frame counts come from manifests and are
re-validated against the journal/dataset at save time by the frozen R4B
checks. Any mismatch fails closed before the episode proceeds (or, for frame
coverage, before the save). Nothing here writes a sidecar/JSON to disk:
evidence-path freezing, atomic persistence, and cross-process recovery are
R4D scope. All serialized authorization/physical fields stay recursively
False and no live authorization is ever granted.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Sequence
from typing import Any, Final

from lerobot.bamboo_sorting.c0_capture_contract import C0EpisodeManifestV1
from lerobot.bamboo_sorting.c0_gripper_capture_journal import (
    JOURNAL_STATUS_ABANDONED_RERECORDED,
    JOURNAL_STATUS_INCOMPLETE,
    JOURNAL_STATUS_RECORDING,
    C0GripperPendingCaptureJournalV1,
    _require_pure_json,
)
from lerobot.bamboo_sorting.c0_gripper_save_finalizer import (
    STATE_FINALIZED,
    STATE_IDLE,
    STATE_SAVE_UNCERTAIN,
    STATE_SAVED_PENDING_AUDIT,
    C0GripperEpisodeSavePendingAuditV1,
    C0GripperSaveFinalizationResultV1,
    begin_c0_gripper_episode_finalization,
    complete_c0_gripper_episode_finalization,
    require_c0_gripper_finalized_evidence_bound,
    require_c0_gripper_journal_save_state,
    require_c0_gripper_pending_token_bound_to_journal,
)

C0_GRIPPER_EPISODE_LIFECYCLE_RECORD_SCHEMA_VERSION: Final = "C0GripperEpisodeLifecycleRecordV1"
C0_GRIPPER_RECORDING_SESSION_RECORD_SCHEMA_VERSION: Final = "C0GripperRecordingSessionRecordV1"

# Coordinator-owned per-attempt outcomes.
# ``pre_save_rejected``: R4B refused before ``dataset.save_episode`` started.
# ``save_uncertain``: ``dataset.save_episode`` was entered and raised; the
# journal keeps the R4B ``save_uncertain`` terminal state and is never retried.
OUTCOME_RECORDING: Final = "recording"
OUTCOME_ABANDONED_RERECORDED: Final = "abandoned_rerecorded"
OUTCOME_INCOMPLETE: Final = "incomplete"
OUTCOME_PRE_SAVE_REJECTED: Final = "pre_save_rejected"
OUTCOME_SAVE_UNCERTAIN: Final = "save_uncertain"
OUTCOME_SAVED_PENDING_AUDIT: Final = "saved_pending_audit"
OUTCOME_FINALIZED: Final = "finalized"
_KNOWN_OUTCOMES: Final = frozenset(
    {
        OUTCOME_RECORDING,
        OUTCOME_ABANDONED_RERECORDED,
        OUTCOME_INCOMPLETE,
        OUTCOME_PRE_SAVE_REJECTED,
        OUTCOME_SAVE_UNCERTAIN,
        OUTCOME_SAVED_PENDING_AUDIT,
        OUTCOME_FINALIZED,
    }
)
_OUTCOMES_WITHOUT_SAVE_PAYLOAD: Final = frozenset(
    {
        OUTCOME_RECORDING,
        OUTCOME_ABANDONED_RERECORDED,
        OUTCOME_INCOMPLETE,
        OUTCOME_PRE_SAVE_REJECTED,
        OUTCOME_SAVE_UNCERTAIN,
    }
)
# R4A journal.status vs R4B save-state are independent machines.
_OUTCOME_JOURNAL_STATUS: Final = {
    OUTCOME_RECORDING: JOURNAL_STATUS_RECORDING,
    OUTCOME_ABANDONED_RERECORDED: JOURNAL_STATUS_ABANDONED_RERECORDED,
    OUTCOME_INCOMPLETE: JOURNAL_STATUS_INCOMPLETE,
    OUTCOME_PRE_SAVE_REJECTED: JOURNAL_STATUS_RECORDING,
    OUTCOME_SAVE_UNCERTAIN: JOURNAL_STATUS_RECORDING,
    OUTCOME_SAVED_PENDING_AUDIT: JOURNAL_STATUS_RECORDING,
    OUTCOME_FINALIZED: JOURNAL_STATUS_RECORDING,
}
_OUTCOME_R4B_STATE: Final = {
    OUTCOME_RECORDING: STATE_IDLE,
    OUTCOME_ABANDONED_RERECORDED: STATE_IDLE,
    OUTCOME_INCOMPLETE: STATE_IDLE,
    OUTCOME_PRE_SAVE_REJECTED: STATE_IDLE,
    OUTCOME_SAVE_UNCERTAIN: STATE_SAVE_UNCERTAIN,
    OUTCOME_SAVED_PENDING_AUDIT: STATE_SAVED_PENDING_AUDIT,
    OUTCOME_FINALIZED: STATE_FINALIZED,
}

# Authorization/physical fields every serialized record of this module fixes
# to False; callers cannot set them to True.
_AUTHORIZATION_FIELDS: Final = (
    "physical_gripper_feedback_available",
    "physical_grasp_success_proven",
    "training_authorized",
    "policy_execution_authorized",
    "serialized_record_grants_live_authorization",
    "hardware_access_performed_by_serialization",
    "hardware_access_performed_by_audit",
    "final_sidecar_persisted_to_disk",
)


class C0GripperRecordingSessionError(ValueError):
    """Fail-closed session error: the mismatched episode must not proceed."""


class _DatasetSaveEntryProbe:
    """Delegates dataset access while recording whether save_episode was entered.

    R4B owns the private journal save-state machine. This probe is the public
    distinction between a pre-save refusal and an in-flight save: it never
    reads R4B private attributes and never consults test counters.
    """

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset
        self.save_entered = False

    def save_episode(self, *args: Any, **kwargs: Any) -> Any:
        self.save_entered = True
        return self._dataset.save_episode(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)


@dataclasses.dataclass
class _AttemptSlot:
    """Internal mutable state for one episode recording attempt."""

    episode_index: int
    manifest: C0EpisodeManifestV1
    journal: C0GripperPendingCaptureJournalV1
    outcome: str
    pending_audit: C0GripperEpisodeSavePendingAuditV1 | None = None
    finalization_result: C0GripperSaveFinalizationResultV1 | None = None


@dataclasses.dataclass(frozen=True)
class C0GripperEpisodeLifecycleRecordV1:
    """Detached, deep-copied snapshot of one episode attempt's lifecycle.

    Built by the session on demand: mutating the returned journal, manifest,
    token, or result can never rewrite the session's internal evidence. The
    outcome is coordinator-owned vocabulary (see ``OUTCOME_*``); the embedded
    journal snapshot carries its own R4A status and R4B save state.
    """

    schema_version: str
    episode_index: int
    episode_id: str
    fps: float
    frame_count: int
    outcome: str
    manifest: C0EpisodeManifestV1
    journal: C0GripperPendingCaptureJournalV1
    pending_audit: C0GripperEpisodeSavePendingAuditV1 | None
    finalization_result: C0GripperSaveFinalizationResultV1 | None

    def __post_init__(self) -> None:
        if self.schema_version != C0_GRIPPER_EPISODE_LIFECYCLE_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_GRIPPER_EPISODE_LIFECYCLE_RECORD_SCHEMA_VERSION!r}"
            )
        if isinstance(self.episode_index, bool) or not isinstance(self.episode_index, int):
            raise ValueError("episode_index must be an integer, not bool")
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
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
        if self.outcome not in _KNOWN_OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(_KNOWN_OUTCOMES)}")
        if not isinstance(self.manifest, C0EpisodeManifestV1):
            raise TypeError("manifest must be a C0EpisodeManifestV1")
        if not isinstance(self.journal, C0GripperPendingCaptureJournalV1):
            raise TypeError("journal must be a C0GripperPendingCaptureJournalV1")
        if self.manifest.episode_id != self.episode_id:
            raise ValueError("episode_id contradicts manifest episode_id")
        if self.journal.episode_id != self.episode_id:
            raise ValueError("episode_id contradicts journal episode_id")
        if self.journal.episode_index != self.episode_index:
            raise ValueError("episode_index contradicts journal episode_index")
        if self.manifest.frame_count != self.frame_count:
            raise ValueError("frame_count contradicts manifest frame_count")
        if float(self.journal.fps) != float(self.fps):
            raise ValueError("fps contradicts journal fps")
        if float(self.manifest.fps) != float(self.fps):
            raise ValueError("fps contradicts manifest fps")
        if self.pending_audit is not None and not isinstance(
            self.pending_audit, C0GripperEpisodeSavePendingAuditV1
        ):
            raise TypeError("pending_audit must be a C0GripperEpisodeSavePendingAuditV1 or None")
        if self.finalization_result is not None and not isinstance(
            self.finalization_result, C0GripperSaveFinalizationResultV1
        ):
            raise TypeError(
                "finalization_result must be a C0GripperSaveFinalizationResultV1 or None"
            )
        has_token = self.pending_audit is not None
        has_result = self.finalization_result is not None
        if self.outcome in _OUTCOMES_WITHOUT_SAVE_PAYLOAD:
            if has_token:
                raise ValueError(f"outcome {self.outcome!r} must not carry a pending token")
            if has_result:
                raise ValueError(
                    f"outcome {self.outcome!r} must not carry a finalization result"
                )
        elif self.outcome == OUTCOME_SAVED_PENDING_AUDIT:
            if not has_token:
                raise ValueError("outcome 'saved_pending_audit' requires a pending token")
            if has_result:
                raise ValueError(
                    "outcome 'saved_pending_audit' must not carry a finalization result"
                )
        elif self.outcome == OUTCOME_FINALIZED:
            if not has_result:
                raise ValueError("outcome 'finalized' requires a finalization result")
            if not has_token:
                raise ValueError(
                    "outcome 'finalized' requires the pending token that was promoted"
                )
        if has_token:
            self._require_identity_payload(
                "pending token",
                episode_id=self.pending_audit.episode_id,
                episode_index=self.pending_audit.episode_index,
                fps=self.pending_audit.fps,
                frame_count=self.pending_audit.frame_count,
            )
        if has_result:
            self._require_identity_payload(
                "finalization result",
                episode_id=self.finalization_result.episode_id,
                episode_index=self.finalization_result.episode_index,
                fps=self.finalization_result.fps,
                frame_count=self.finalization_result.frame_count,
            )
        expected_status = _OUTCOME_JOURNAL_STATUS[self.outcome]
        if self.journal.status != expected_status:
            raise ValueError(
                f"outcome {self.outcome!r} requires journal status {expected_status!r}, "
                f"got {self.journal.status!r}"
            )
        require_c0_gripper_journal_save_state(
            self.journal, expected=_OUTCOME_R4B_STATE[self.outcome]
        )
        if self.outcome == OUTCOME_SAVED_PENDING_AUDIT:
            require_c0_gripper_pending_token_bound_to_journal(
                journal=self.journal, pending=self.pending_audit
            )
        elif self.outcome == OUTCOME_FINALIZED:
            require_c0_gripper_finalized_evidence_bound(
                journal=self.journal,
                pending=self.pending_audit,
                result=self.finalization_result,
            )
        # Detach everything mutable at construction time.
        object.__setattr__(self, "manifest", copy.deepcopy(self.manifest))
        object.__setattr__(self, "journal", copy.deepcopy(self.journal))
        object.__setattr__(self, "pending_audit", copy.deepcopy(self.pending_audit))
        object.__setattr__(self, "finalization_result", copy.deepcopy(self.finalization_result))

    def _require_identity_payload(
        self,
        label: str,
        *,
        episode_id: str,
        episode_index: int,
        fps: float,
        frame_count: int,
    ) -> None:
        if episode_id != self.episode_id:
            raise ValueError(f"{label} episode_id contradicts top-level episode_id")
        if episode_id != self.manifest.episode_id or episode_id != self.journal.episode_id:
            raise ValueError(f"{label} episode_id contradicts manifest or journal")
        if episode_index != self.episode_index:
            raise ValueError(f"{label} episode_index contradicts top-level episode_index")
        if episode_index != self.journal.episode_index:
            raise ValueError(f"{label} episode_index contradicts journal")
        if frame_count != self.frame_count:
            raise ValueError(f"{label} frame_count contradicts top-level frame_count")
        if frame_count != self.manifest.frame_count:
            raise ValueError(f"{label} frame_count contradicts manifest")
        if float(fps) != float(self.fps):
            raise ValueError(f"{label} fps contradicts top-level fps")
        if float(fps) != float(self.journal.fps) or float(fps) != float(self.manifest.fps):
            raise ValueError(f"{label} fps contradicts journal or manifest")

    def to_manifest_record(self) -> dict[str, Any]:
        """Detached pure-JSON snapshot; every authorization field stays False."""

        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "episode_index": self.episode_index,
            "episode_id": self.episode_id,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "outcome": self.outcome,
            "journal_status": self.journal.status,
            "manifest": self.manifest.to_manifest_record(),
            "journal": self.journal.to_manifest_record(),
            "pending_audit": (
                self.pending_audit.to_manifest_record() if self.pending_audit is not None else None
            ),
            "finalization_result": (
                self.finalization_result.to_manifest_record()
                if self.finalization_result is not None
                else None
            ),
        }
        for field_name in _AUTHORIZATION_FIELDS:
            record[field_name] = False
        _require_pure_json("episode_lifecycle_record", record)
        return copy.deepcopy(record)


class C0GripperRecordingSession:
    """Offline coordinator wiring manifests -> journals -> R4B save -> R4B complete.

    Implements the generic ``RecordEpisodeLifecycleObserver`` protocol of
    ``lerobot.scripts.lerobot_record``. Exactly one journal is open at a time;
    episode ids come from the explicit manifests, episode indexes are
    positional in the manifest sequence, and every index/id/frame-count
    mismatch fails closed (before the episode starts, or before the save for
    frame coverage, via the frozen R4B checks).
    """

    def __init__(self, *, episode_manifests: Sequence[C0EpisodeManifestV1], fps: int | float) -> None:
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise ValueError("fps must be a finite positive number, not bool")
        fps_value = float(fps)
        if not math.isfinite(fps_value) or fps_value <= 0:
            raise ValueError("fps must be a finite positive number")
        if isinstance(episode_manifests, (str, bytes)):
            raise TypeError("episode_manifests must be a sequence of C0EpisodeManifestV1")
        try:
            manifests = tuple(episode_manifests)
        except TypeError as error:
            raise TypeError(
                "episode_manifests must be a sequence of C0EpisodeManifestV1"
            ) from error
        if not manifests:
            raise C0GripperRecordingSessionError(
                "at least one explicit episode manifest is required; refusing to "
                "record episodes without manifests"
            )
        episode_ids: set[str] = set()
        for position, manifest in enumerate(manifests):
            if not isinstance(manifest, C0EpisodeManifestV1):
                raise TypeError(
                    f"episode_manifests[{position}] must be a C0EpisodeManifestV1, got "
                    f"{type(manifest).__name__}"
                )
            if manifest.episode_id in episode_ids:
                raise C0GripperRecordingSessionError(
                    f"duplicate episode_id {manifest.episode_id!r} at manifest position "
                    f"{position}; episode indexes are positional and ids must be unique"
                )
            episode_ids.add(manifest.episode_id)
            if float(manifest.fps) != fps_value:
                raise C0GripperRecordingSessionError(
                    f"manifest at position {position} has fps {manifest.fps} != session "
                    f"fps {fps_value}; refusing to mix capture profiles in one session"
                )
            if (
                isinstance(manifest.frame_count, bool)
                or not isinstance(manifest.frame_count, int)
                or manifest.frame_count <= 0
            ):
                raise C0GripperRecordingSessionError(
                    f"manifest at position {position} has invalid frame_count "
                    f"{manifest.frame_count!r}; refusing to start the episode"
                )
        self._manifests: tuple[C0EpisodeManifestV1, ...] = copy.deepcopy(manifests)
        self._fps = fps_value
        self._current_index = 0
        self._active: _AttemptSlot | None = None
        self._attempts: list[_AttemptSlot] = []
        self._pending: list[_AttemptSlot] = []
        self._batch_finalized = False
        self._finalize_failed = False

    # ------------------------------------------------------------------ state

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def manifest_count(self) -> int:
        return len(self._manifests)

    @property
    def next_episode_index(self) -> int:
        """Index of the episode the next ``episode_frame_observer`` will open."""

        return self._current_index

    @property
    def saved_episode_count(self) -> int:
        """Episodes whose save returned and holds a pending (or finalized) token."""

        return len(self._pending) + sum(
            1 for slot in self._attempts if slot.finalization_result is not None
        )

    @property
    def pending_audit_count(self) -> int:
        return len(self._pending)

    @property
    def finalized_episode_count(self) -> int:
        return sum(1 for slot in self._attempts if slot.finalization_result is not None)

    @property
    def batch_finalization_complete(self) -> bool:
        """True only when every planned manifest was saved and promoted.

        Stopped, incomplete, failed, pre-save rejected, save-uncertain, or
        otherwise unfinished batches stay False even if already-saved tokens
        were audited. A rerecord that later finishes every manifest can be True.
        """

        return self._batch_finalized

    @property
    def finalize_failed(self) -> bool:
        return self._finalize_failed

    @property
    def episode_records(self) -> tuple[C0GripperEpisodeLifecycleRecordV1, ...]:
        """Detached deep-copied records for every attempt, in creation order."""

        return tuple(self._build_record(slot) for slot in self._attempts)

    @property
    def pending_audit_tokens(self) -> tuple[C0GripperEpisodeSavePendingAuditV1, ...]:
        """Detached copies of the tokens not yet promoted to finalized."""

        return tuple(copy.deepcopy(slot.pending_audit) for slot in self._pending)

    @property
    def finalization_results(self) -> tuple[C0GripperSaveFinalizationResultV1, ...]:
        """Detached copies of every completed promotion result, in save order."""

        return tuple(
            copy.deepcopy(slot.finalization_result)
            for slot in self._attempts
            if slot.finalization_result is not None
        )

    def to_manifest_record(self) -> dict[str, Any]:
        """Detached pure-JSON session record; authorization fields stay False."""

        record: dict[str, Any] = {
            "schema_version": C0_GRIPPER_RECORDING_SESSION_RECORD_SCHEMA_VERSION,
            "fps": self._fps,
            "manifest_count": len(self._manifests),
            "next_episode_index": self._current_index,
            "saved_episode_count": self.saved_episode_count,
            "pending_audit_count": self.pending_audit_count,
            "finalized_episode_count": self.finalized_episode_count,
            "batch_finalization_complete": self._batch_finalized,
            "finalize_failed": self._finalize_failed,
            "episodes": [episode.to_manifest_record() for episode in self.episode_records],
        }
        for field_name in _AUTHORIZATION_FIELDS:
            record[field_name] = False
        _require_pure_json("session_record", record)
        return copy.deepcopy(record)

    # ------------------------------------------------------ lifecycle protocol

    def episode_frame_observer(self) -> C0GripperPendingCaptureJournalV1:
        """Open the isolated journal for the next episode attempt (fail closed)."""

        if self._batch_finalized:
            raise C0GripperRecordingSessionError(
                "session batch is already finalized; no new episodes may start"
            )
        if self._finalize_failed:
            raise C0GripperRecordingSessionError(
                "dataset finalize failed; no new episodes may start"
            )
        if self._active is not None:
            raise C0GripperRecordingSessionError(
                f"episode journal {self._active.journal.episode_id!r} (index "
                f"{self._active.episode_index}) is still open; refusing to open a "
                "second journal for the same session"
            )
        index = self._current_index
        if index >= len(self._manifests):
            raise C0GripperRecordingSessionError(
                f"no episode manifest for episode_index {index} (manifest count "
                f"{len(self._manifests)}); refusing to record an episode without an "
                "explicit manifest"
            )
        manifest = self._manifests[index]
        journal = C0GripperPendingCaptureJournalV1(
            episode_id=manifest.episode_id, episode_index=index, fps=self._fps
        )
        slot = _AttemptSlot(
            episode_index=index,
            manifest=manifest,
            journal=journal,
            outcome=OUTCOME_RECORDING,
        )
        self._attempts.append(slot)
        self._active = slot
        return journal

    def on_episode_rerecord(self, *, robot: Any, dataset: Any, episode_index: int) -> None:
        """Abandon the open journal; the next attempt starts fully isolated."""

        slot = self._require_active_slot(episode_index, hook="on_episode_rerecord")
        slot.journal.mark_abandoned(reason="rerecord requested by operator")
        slot.outcome = OUTCOME_ABANDONED_RERECORDED
        # The episode index is NOT advanced: the next attempt re-uses the same
        # manifest. The generic loop performs the single buffer clear.
        self._active = None

    def on_episode_incomplete(self, *, robot: Any, dataset: Any, episode_index: int, reason: str) -> None:
        """Mark the open journal incomplete; the unsaved episode is dropped."""

        slot = self._require_active_slot(episode_index, hook="on_episode_incomplete")
        slot.journal.mark_incomplete(reason=reason)
        slot.outcome = OUTCOME_INCOMPLETE
        self._active = None

    def save_episode(self, *, robot: Any, dataset: Any, episode_index: int) -> None:
        """Save the completed episode exclusively through the R4B begin phase.

        Exactly one ``dataset.save_episode()`` happens here, inside the frozen
        R4B coordinator, after every pre-save validation passed. On failure
        the exception propagates: the journal keeps its R4B save state, the
        save is never retried, and the episode is never counted as recorded.
        """

        slot = self._require_active_slot(episode_index, hook="save_episode")
        manifest = self._manifests[episode_index]
        probed = _DatasetSaveEntryProbe(dataset)
        try:
            token = begin_c0_gripper_episode_finalization(
                journal=slot.journal,
                dataset=probed,
                episode_index=episode_index,
                episode_manifest=manifest,
            )
        except BaseException:
            save_entered = probed.save_entered
            if save_entered:
                slot.outcome = OUTCOME_SAVE_UNCERTAIN
            else:
                slot.outcome = OUTCOME_PRE_SAVE_REJECTED
            self._active = None
            raise
        slot.pending_audit = token
        slot.outcome = OUTCOME_SAVED_PENDING_AUDIT
        self._pending.append(slot)
        self._active = None
        self._current_index = episode_index + 1

    def on_dataset_finalize_success(self, *, dataset: Any, recorded_episode_count: int) -> None:
        """Promote every pending token via the R4B complete phase, in save order.

        This runs only after the dataset finalized through its public
        lifecycle; it never saves again. The first complete failure stops the
        batch: later promotions are skipped, their tokens stay
        ``saved_pending_audit``, no save is retried, and
        ``batch_finalization_complete`` stays False.
        """

        if self._batch_finalized:
            raise C0GripperRecordingSessionError(
                "session batch is already finalized; refusing to finalize twice"
            )
        if self._active is not None:
            raise C0GripperRecordingSessionError(
                f"episode journal {self._active.journal.episode_id!r} (index "
                f"{self._active.episode_index}) is still open at dataset finalize; "
                "refusing to silently drop an unsaved episode"
            )
        if isinstance(recorded_episode_count, bool) or not isinstance(
            recorded_episode_count, int
        ):
            raise C0GripperRecordingSessionError(
                "recorded_episode_count must be an integer, not bool"
            )
        if recorded_episode_count != self.saved_episode_count:
            raise C0GripperRecordingSessionError(
                f"recorded_episode_count {recorded_episode_count} != saved episode "
                f"count {self.saved_episode_count}; refusing a partial or forged batch"
            )
        while self._pending:
            slot = self._pending[0]
            result = complete_c0_gripper_episode_finalization(
                journal=slot.journal,
                dataset=dataset,
                pending=slot.pending_audit,
                episode_manifest=slot.manifest,
            )
            slot.finalization_result = result
            slot.outcome = OUTCOME_FINALIZED
            self._pending.pop(0)
        finalized_indexes = {
            slot.episode_index
            for slot in self._attempts
            if slot.outcome == OUTCOME_FINALIZED and slot.finalization_result is not None
        }
        self._batch_finalized = finalized_indexes == set(range(len(self._manifests)))

    def on_dataset_finalize_failure(self, *, dataset: Any, error: BaseException) -> None:
        """Keep every token pending; never complete, never fabricate evidence."""

        self._finalize_failed = True

    # -------------------------------------------------------------- internal

    def _require_active_slot(self, episode_index: int, *, hook: str) -> _AttemptSlot:
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise C0GripperRecordingSessionError(
                f"{hook}: episode_index must be an integer, not bool"
            )
        if episode_index < 0:
            raise C0GripperRecordingSessionError(
                f"{hook}: episode_index must be non-negative"
            )
        slot = self._active
        if slot is None:
            raise C0GripperRecordingSessionError(
                f"{hook}: no open episode journal for episode_index {episode_index}; "
                "episode_frame_observer must run before this hook"
            )
        if episode_index != self._current_index or slot.episode_index != episode_index:
            raise C0GripperRecordingSessionError(
                f"{hook}: episode_index {episode_index} does not match the open "
                f"journal episode_index {slot.episode_index} (session is at episode "
                f"{self._current_index}); refusing to act on the wrong episode"
            )
        return slot

    def _build_record(self, slot: _AttemptSlot) -> C0GripperEpisodeLifecycleRecordV1:
        return C0GripperEpisodeLifecycleRecordV1(
            schema_version=C0_GRIPPER_EPISODE_LIFECYCLE_RECORD_SCHEMA_VERSION,
            episode_index=slot.episode_index,
            episode_id=slot.manifest.episode_id,
            fps=self._fps,
            frame_count=slot.manifest.frame_count,
            outcome=slot.outcome,
            manifest=slot.manifest,
            journal=slot.journal,
            pending_audit=slot.pending_audit,
            finalization_result=slot.finalization_result,
        )

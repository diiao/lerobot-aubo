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

"""Fail-closed staging contracts for the formal ten-episode C0 capture.

This module is deliberately hardware-free.  It defines the immutable plan and
the per-frame timestamp observer needed by a later, separately authorized live
entry.  A plan or a serialized journal is evidence, never permission to read a
camera, connect to AUBO, perform IO/motion, train, infer, or execute a policy.

The previously recorded one-episode smoke dataset may be referenced as
excluded diagnostic evidence, but it is never counted as one of the ten formal
episodes because it predates the required per-sensor timestamp sidecar.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from .c0_capture_contract import (
    C0_DATASET_FPS,
    C0_EXPECTED_EPISODE_COUNT,
    C0_HUMAN_OUTCOMES,
    C0_INSTRUCTION_ID,
    C0_SENSOR_STREAMS,
    C0_SPLIT_NAMES,
    C0SensorTimestampV1,
)
from .c0_gripper_capture_journal import C0GripperPendingCaptureJournalV1
from .contracts import INSTRUCTION_SPECS

C0_BATCH_CAPTURE_PLAN_SCHEMA_VERSION: Final = "C0BatchCapturePlanV1"
C0_SENSOR_TIMESTAMP_JOURNAL_SCHEMA_VERSION: Final = "C0SensorTimestampJournalV1"
C0_FORMAL_FRAME_OBSERVER_SCHEMA_VERSION: Final = "C0FormalFrameObserverV1"
C0_BATCH_CAPTURE_SESSION_SCHEMA_VERSION: Final = "C0BatchCaptureSessionV1"
C0_FORMAL_BATCH_STATUS: Final = "formal ten-episode C0 capture, pending final evidence binding"
C0_FORMAL_OPERATOR_CONFIRMATION: Final = "YES_RECORD_FORMAL_C0_10"
C0_FORMAL_EPISODE_COUNT: Final = C0_EXPECTED_EPISODE_COUNT
C0_FORMAL_TRAIN_EPISODES: Final = 8
C0_FORMAL_VALIDATION_EPISODES: Final = 2
C0_PILOT_AGGREGATE_EPISODE_COUNT: Final = 12
C0_PILOT_VALIDATION_OPERATOR_CONFIRMATION: Final = "YES_RECORD_C0_PILOT_VALIDATION_2"
C0_FORMAL_TASK_TEXT: Final = INSTRUCTION_SPECS[C0_INSTRUCTION_ID].text

_AUTHORIZATION_FALSE_FIELDS: Final = (
    "camera_access_authorized",
    "robot_state_access_authorized",
    "teleoperation_authorized",
    "robot_motion_authorized",
    "gripper_io_authorized",
    "training_authorized",
    "policy_execution_authorized",
    "serialized_record_grants_live_authorization",
    "hardware_access_performed_by_serialization",
)


class C0BatchCaptureError(ValueError):
    """Fail-closed formal capture planning or timestamp error."""


def _require_nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise C0BatchCaptureError(f"{name} must be a non-empty string")
    return value.strip()


def _authorization_false_record() -> dict[str, bool]:
    return {name: False for name in _AUTHORIZATION_FALSE_FIELDS}


@dataclass(frozen=True)
class C0EpisodeCapturePlanV1:
    """Pre-capture identity for one episode; contains no fabricated final evidence."""

    episode_index: int
    episode_id: str
    scene_id: str
    split_name: str
    placement_reference: str
    formal_episode_index: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.episode_index, bool) or not isinstance(self.episode_index, int):
            raise C0BatchCaptureError("episode_index must be an integer, not bool")
        if not 0 <= self.episode_index < C0_FORMAL_EPISODE_COUNT:
            raise C0BatchCaptureError(
                f"episode_index must be in [0, {C0_FORMAL_EPISODE_COUNT})"
            )
        formal_index = (
            self.episode_index
            if self.formal_episode_index is None
            else self.formal_episode_index
        )
        if isinstance(formal_index, bool) or not isinstance(formal_index, int):
            raise C0BatchCaptureError("formal_episode_index must be an integer, not bool")
        if not 0 <= formal_index < C0_PILOT_AGGREGATE_EPISODE_COUNT:
            raise C0BatchCaptureError(
                "formal_episode_index must be in "
                f"[0, {C0_PILOT_AGGREGATE_EPISODE_COUNT})"
            )
        object.__setattr__(self, "formal_episode_index", formal_index)
        for name in ("episode_id", "scene_id", "placement_reference"):
            object.__setattr__(self, name, _require_nonempty(name, getattr(self, name)))
        if self.split_name not in C0_SPLIT_NAMES:
            raise C0BatchCaptureError(
                f"split_name must be one of {sorted(C0_SPLIT_NAMES)}"
            )

    def to_manifest_record(self) -> dict[str, object]:
        record = asdict(self)
        record.update(
            {
                "instruction_id": C0_INSTRUCTION_ID,
                "instruction_text": C0_FORMAL_TASK_TEXT,
                "fps": float(C0_DATASET_FPS),
                "finalized": False,
                "formal_final_evidence_bound": False,
                **_authorization_false_record(),
            }
        )
        return copy.deepcopy(record)


@dataclass(frozen=True)
class C0BatchCapturePlanV1:
    """Exactly ten new formal episode identities plus an optional excluded smoke ref."""

    schema_version: str
    batch_id: str
    session_id: str
    dataset_root: str
    evidence_root: str
    episodes: tuple[C0EpisodeCapturePlanV1, ...]
    excluded_smoke_dataset_root: str | None = None
    excluded_smoke_reason: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != C0_BATCH_CAPTURE_PLAN_SCHEMA_VERSION:
            raise C0BatchCaptureError(
                f"schema_version must be {C0_BATCH_CAPTURE_PLAN_SCHEMA_VERSION!r}"
            )
        for name in ("batch_id", "session_id", "dataset_root", "evidence_root"):
            object.__setattr__(self, name, _require_nonempty(name, getattr(self, name)))
        if Path(self.dataset_root).expanduser().resolve() == Path(
            self.evidence_root
        ).expanduser().resolve():
            raise C0BatchCaptureError("dataset_root and evidence_root must be different")

        episodes = tuple(self.episodes)
        if not 1 <= len(episodes) <= C0_FORMAL_EPISODE_COUNT:
            raise C0BatchCaptureError(
                f"C0 capture shard requires 1..{C0_FORMAL_EPISODE_COUNT} episodes"
            )
        if not all(isinstance(item, C0EpisodeCapturePlanV1) for item in episodes):
            raise TypeError("episodes must contain only C0EpisodeCapturePlanV1")
        if tuple(item.episode_index for item in episodes) != tuple(range(len(episodes))):
            raise C0BatchCaptureError(
                "dataset-local episode indexes must be contiguous from zero in order"
            )
        formal_indexes = tuple(item.formal_episode_index for item in episodes)
        assert all(index is not None for index in formal_indexes)
        formal_start = formal_indexes[0]
        if formal_indexes != tuple(range(formal_start, formal_start + len(episodes))):
            raise C0BatchCaptureError(
                "formal episode indexes must be one contiguous ordered slice of 0..11"
            )
        for name in ("episode_id", "scene_id"):
            values = [getattr(item, name) for item in episodes]
            if len(set(values)) != len(values):
                raise C0BatchCaptureError(f"{name} must be unique across the formal batch")
        expected_splits = tuple(
            "train" if index < C0_FORMAL_TRAIN_EPISODES else "validation"
            for index in formal_indexes
        )
        if tuple(item.split_name for item in episodes) != expected_splits:
            raise C0BatchCaptureError(
                "formal C0 split must follow formal indexes 0..7 train and 8..9 validation"
            )
        placement_references = {item.placement_reference for item in episodes}
        if len(placement_references) != 1:
            raise C0BatchCaptureError(
                "placement_reference must stay fixed across the formal C0 batch"
            )
        object.__setattr__(self, "episodes", episodes)

        has_smoke = self.excluded_smoke_dataset_root is not None
        has_reason = self.excluded_smoke_reason is not None
        if has_smoke != has_reason:
            raise C0BatchCaptureError(
                "excluded_smoke_dataset_root and excluded_smoke_reason must be paired"
            )
        if has_smoke:
            object.__setattr__(
                self,
                "excluded_smoke_dataset_root",
                _require_nonempty(
                    "excluded_smoke_dataset_root", self.excluded_smoke_dataset_root
                ),
            )
            object.__setattr__(
                self,
                "excluded_smoke_reason",
                _require_nonempty("excluded_smoke_reason", self.excluded_smoke_reason),
            )

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    def to_manifest_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "status": (
                "C0 pilot validation extension, pending final evidence binding"
                if self.episodes[0].formal_episode_index >= C0_FORMAL_EPISODE_COUNT
                else C0_FORMAL_BATCH_STATUS
            ),
            "batch_id": self.batch_id,
            "session_id": self.session_id,
            "dataset_root": self.dataset_root,
            "evidence_root": self.evidence_root,
            "expected_episode_count": C0_FORMAL_EPISODE_COUNT,
            "capture_shard_episode_count": self.episode_count,
            "episode_count": self.episode_count,
            "episodes": [item.to_manifest_record() for item in self.episodes],
            "excluded_smoke_dataset_root": self.excluded_smoke_dataset_root,
            "excluded_smoke_reason": self.excluded_smoke_reason,
            "capture_started": False,
            "formal_final_evidence_bound": False,
            **_authorization_false_record(),
        }
        return copy.deepcopy(record)


def build_c0_batch_capture_plan(
    *,
    batch_id: str,
    session_id: str,
    dataset_root: Path,
    evidence_root: Path,
    placement_reference: str,
    excluded_smoke_dataset_root: Path | None = None,
    episode_count: int = C0_FORMAL_EPISODE_COUNT,
    formal_episode_offset: int = 0,
) -> C0BatchCapturePlanV1:
    """Build a formal C0 shard or the two-episode pilot validation extension."""

    batch_id = _require_nonempty("batch_id", batch_id)
    session_id = _require_nonempty("session_id", session_id)
    placement_reference = _require_nonempty("placement_reference", placement_reference)
    if isinstance(episode_count, bool) or not isinstance(episode_count, int):
        raise C0BatchCaptureError("episode_count must be an integer, not bool")
    if isinstance(formal_episode_offset, bool) or not isinstance(
        formal_episode_offset, int
    ):
        raise C0BatchCaptureError("formal_episode_offset must be an integer, not bool")
    is_pilot_validation_extension = (
        formal_episode_offset >= C0_FORMAL_EPISODE_COUNT
        or formal_episode_offset + episode_count > C0_FORMAL_EPISODE_COUNT
    )
    if is_pilot_validation_extension and (
        formal_episode_offset != C0_FORMAL_EPISODE_COUNT or episode_count != 2
    ):
        raise C0BatchCaptureError(
            "the pilot extension must be exactly two episodes at formal indexes 10..11"
        )
    if (
        episode_count < 1
        or formal_episode_offset < 0
        or formal_episode_offset + episode_count > C0_PILOT_AGGREGATE_EPISODE_COUNT
    ):
        raise C0BatchCaptureError(
            "formal_episode_offset + episode_count must select a non-empty slice of 0..11"
        )
    episodes = tuple(
        C0EpisodeCapturePlanV1(
            episode_index=index,
            episode_id=f"{batch_id}-episode-{index:02d}",
            scene_id=f"{batch_id}-scene-{index:02d}",
            split_name=(
                "train"
                if formal_episode_offset + index < C0_FORMAL_TRAIN_EPISODES
                else "validation"
            ),
            placement_reference=placement_reference,
            formal_episode_index=formal_episode_offset + index,
        )
        for index in range(episode_count)
    )
    excluded_root = (
        str(Path(excluded_smoke_dataset_root).expanduser().resolve())
        if excluded_smoke_dataset_root is not None
        else None
    )
    return C0BatchCapturePlanV1(
        schema_version=C0_BATCH_CAPTURE_PLAN_SCHEMA_VERSION,
        batch_id=batch_id,
        session_id=session_id,
        dataset_root=str(Path(dataset_root).expanduser().resolve()),
        evidence_root=str(Path(evidence_root).expanduser().resolve()),
        episodes=episodes,
        excluded_smoke_dataset_root=excluded_root,
        excluded_smoke_reason=(
            "one-episode smoke predates required per-sensor timestamp and final evidence sidecars"
            if excluded_root is not None
            else None
        ),
    )


def require_new_c0_batch_paths(plan: C0BatchCapturePlanV1) -> None:
    """Fail before hardware imports unless both formal output roots are new."""

    if not isinstance(plan, C0BatchCapturePlanV1):
        raise TypeError("plan must be C0BatchCapturePlanV1")
    for name in ("dataset_root", "evidence_root"):
        path = Path(getattr(plan, name)).expanduser().resolve()
        if path.exists():
            raise C0BatchCaptureError(f"{name} already exists; refuse overwrite/resume: {path}")
    if plan.excluded_smoke_dataset_root is not None:
        smoke = Path(plan.excluded_smoke_dataset_root).expanduser().resolve()
        if not smoke.is_dir():
            raise C0BatchCaptureError(f"excluded smoke dataset is missing: {smoke}")


def operator_confirmation_for_plan(plan: C0BatchCapturePlanV1) -> str:
    """Return the exact live confirmation for the selected capture scope."""

    if not isinstance(plan, C0BatchCapturePlanV1):
        raise TypeError("plan must be a C0BatchCapturePlanV1")
    if plan.episodes[0].formal_episode_index >= C0_FORMAL_EPISODE_COUNT:
        return C0_PILOT_VALIDATION_OPERATOR_CONFIRMATION
    return C0_FORMAL_OPERATOR_CONFIRMATION


def require_formal_operator_confirmation(
    value: object, expected: str = C0_FORMAL_OPERATOR_CONFIRMATION
) -> None:
    if not isinstance(expected, str) or not expected:
        raise TypeError("expected confirmation must be a non-empty string")
    if not isinstance(value, str) or value.strip() != expected:
        raise C0BatchCaptureError(
            f"operator confirmation must be {expected!r}"
        )


def build_c0_batch_operator_checklist(plan: C0BatchCapturePlanV1) -> tuple[str, ...]:
    """Return the live checklist; constructing it performs no hardware access."""

    if not isinstance(plan, C0BatchCapturePlanV1):
        raise TypeError("plan must be C0BatchCapturePlanV1")
    placement_reference = plan.episodes[0].placement_reference
    return (
        C0_FORMAL_BATCH_STATUS,
        f"Batch: {plan.batch_id}",
        f"Dataset root (must be new): {plan.dataset_root}",
        f"Evidence root (must be new): {plan.evidence_root}",
        f"Capture shard episodes: exactly {plan.episode_count}",
        (
            "Pilot aggregate indexes: "
            f"{plan.episodes[0].formal_episode_index}.."
            f"{plan.episodes[-1].formal_episode_index}"
        ),
        f"Canonical task: {C0_FORMAL_TASK_TEXT}",
        f"Placement reference: {placement_reference}",
        "The previous one-episode smoke dataset is preserved but excluded from the formal count.",
        "Confirm the two frozen RGB cameras have passed the separately authorized preflight.",
        "Confirm the operator is on site, the workspace is clear, and the e-stop is available.",
        "Confirm a new physical scene is prepared before every episode.",
        "Retained episodes are labelled uncertain; press left arrow to discard failed attempts before save.",
        "This capture does not authorize training, inference, or policy execution.",
        f"Type {operator_confirmation_for_plan(plan)} to proceed.",
    )


def _finite_monotonic_timestamp(stream: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise C0BatchCaptureError(
            f"{stream} host monotonic timestamp must be a finite non-negative number, not bool"
        )
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise C0BatchCaptureError(
            f"{stream} host monotonic timestamp must be finite and non-negative"
        )
    return converted


class C0SensorTimestampJournalV1:
    """Per-frame three-stream host-monotonic evidence for one recording attempt."""

    def __init__(self, *, episode_id: str, episode_index: int) -> None:
        self.episode_id = _require_nonempty("episode_id", episode_id)
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise C0BatchCaptureError("episode_index must be an integer, not bool")
        if episode_index < 0:
            raise C0BatchCaptureError("episode_index must be non-negative")
        self.episode_index = episode_index
        self._frames: list[dict[str, object]] = []
        self._staged: dict[str, object] | None = None
        self._status = "recording"
        self._failure: str | None = None

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def status(self) -> str:
        return self._status

    @property
    def frames(self) -> tuple[dict[str, object], ...]:
        return copy.deepcopy(tuple(self._frames))

    def before_send(self, *, robot: Any, candidate_frame_index: int, **_: Any) -> None:
        if self._status != "recording" or self._staged is not None:
            raise C0BatchCaptureError("timestamp journal is not ready for a new frame")
        if (
            isinstance(candidate_frame_index, bool)
            or not isinstance(candidate_frame_index, int)
            or candidate_frame_index != len(self._frames)
        ):
            raise C0BatchCaptureError(
                "candidate_frame_index must be the next contiguous timestamp frame"
            )
        raw = getattr(robot, "last_c0_sensor_timestamps", None)
        if not isinstance(raw, Mapping) or set(raw) != set(C0_SENSOR_STREAMS):
            raise C0BatchCaptureError(
                f"robot must expose exactly {C0_SENSOR_STREAMS} in last_c0_sensor_timestamps"
            )
        previous = self._frames[-1]["timestamps"] if self._frames else None
        timestamps: dict[str, dict[str, object]] = {}
        for stream in C0_SENSOR_STREAMS:
            value = _finite_monotonic_timestamp(stream, raw[stream])
            if previous is not None:
                previous_value = previous[stream]["host_receive_monotonic_s"]
                if value <= previous_value:
                    raise C0BatchCaptureError(
                        f"{stream} host monotonic timestamp must increase every saved frame"
                    )
            stamp = C0SensorTimestampV1(
                stream_name=stream,
                sync_timestamp_s=value,
                host_receive_monotonic_s=value,
                sync_method="host_receive",
            )
            timestamps[stream] = stamp.to_manifest_record()
        self._staged = {
            "frame_index": candidate_frame_index,
            "timestamps": timestamps,
        }

    def on_send_success(self, **_: Any) -> None:
        if self._staged is None:
            raise C0BatchCaptureError("timestamp evidence was not staged before send")

    def on_send_failure(self, *, error: BaseException, **_: Any) -> None:
        self._staged = None
        self._status = "failed"
        self._failure = f"send_action: {type(error).__name__}: {error}"

    def on_add_frame_success(self, **_: Any) -> None:
        if self._staged is None:
            raise C0BatchCaptureError("timestamp evidence was not staged before add_frame")
        self._frames.append(self._staged)
        self._staged = None

    def on_add_frame_failure(self, *, error: BaseException, **_: Any) -> None:
        self._staged = None
        self._status = "failed"
        self._failure = f"add_frame: {type(error).__name__}: {error}"

    def mark_abandoned(self, *, reason: str) -> None:
        self._staged = None
        self._status = "abandoned_rerecorded"
        self._failure = _require_nonempty("reason", reason)

    def mark_incomplete(self, *, reason: str) -> None:
        self._staged = None
        self._status = "incomplete"
        self._failure = _require_nonempty("reason", reason)

    def to_manifest_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": C0_SENSOR_TIMESTAMP_JOURNAL_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "sensor_streams": list(C0_SENSOR_STREAMS),
            "frame_count": self.frame_count,
            "status": self._status,
            "failure": self._failure,
            "frames": copy.deepcopy(self._frames),
            "finalized": False,
            "formal_final_evidence_bound": False,
            **_authorization_false_record(),
        }
        return copy.deepcopy(record)


class C0FormalFrameObserverV1:
    """Fan out one record-loop cycle to timestamp and gripper journals."""

    def __init__(
        self,
        *,
        timestamp_journal: C0SensorTimestampJournalV1,
        gripper_journal: C0GripperPendingCaptureJournalV1,
    ) -> None:
        if timestamp_journal.episode_id != gripper_journal.episode_id:
            raise C0BatchCaptureError("timestamp and gripper episode_id must match")
        if timestamp_journal.episode_index != gripper_journal.episode_index:
            raise C0BatchCaptureError("timestamp and gripper episode_index must match")
        self.timestamp_journal = timestamp_journal
        self.gripper_journal = gripper_journal

    def before_send(self, **kwargs: Any) -> None:
        self.timestamp_journal.before_send(**kwargs)
        self.gripper_journal.before_send(**kwargs)

    def on_send_success(self, **kwargs: Any) -> None:
        self.gripper_journal.on_send_success(**kwargs)
        self.timestamp_journal.on_send_success(**kwargs)

    def on_send_failure(self, **kwargs: Any) -> None:
        self.gripper_journal.on_send_failure(**kwargs)
        self.timestamp_journal.on_send_failure(**kwargs)

    def on_add_frame_success(self, **kwargs: Any) -> None:
        self.gripper_journal.on_add_frame_success(**kwargs)
        self.timestamp_journal.on_add_frame_success(**kwargs)

    def on_add_frame_failure(self, **kwargs: Any) -> None:
        self.gripper_journal.on_add_frame_failure(**kwargs)
        self.timestamp_journal.on_add_frame_failure(**kwargs)

    def to_manifest_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": C0_FORMAL_FRAME_OBSERVER_SCHEMA_VERSION,
            "episode_id": self.timestamp_journal.episode_id,
            "episode_index": self.timestamp_journal.episode_index,
            "timestamp_journal": self.timestamp_journal.to_manifest_record(),
            "gripper_journal": self.gripper_journal.to_manifest_record(),
            "dataset_episode_durably_saved": False,
            "formal_final_evidence_bound": False,
            **_authorization_false_record(),
        }
        return copy.deepcopy(record)


def require_human_outcome(value: object) -> str:
    if not isinstance(value, str) or value not in C0_HUMAN_OUTCOMES:
        raise C0BatchCaptureError(
            f"human outcome must be exactly one of {sorted(C0_HUMAN_OUTCOMES)}"
        )
    return value


@dataclass
class _CaptureAttempt:
    plan: C0EpisodeCapturePlanV1
    observer: C0FormalFrameObserverV1
    status: str = "recording"
    human_outcome: str | None = None
    dataset_episode_durably_saved: bool = False
    sidecar_path: str | None = None
    sidecar_write_error: str | None = None


def _write_json_atomic_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically create one evidence JSON; never overwrite or silently resume."""

    if path.exists():
        raise C0BatchCaptureError(f"evidence path already exists; refuse overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise C0BatchCaptureError(
            f"temporary evidence path already exists; manual review required: {temporary}"
        )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    temporary_created = False
    try:
        with temporary.open("xb") as stream:
            temporary_created = True
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # link(2) creates the destination atomically and fails with EEXIST;
        # unlike replace(), it can never overwrite evidence created by a race.
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary_created and temporary.exists():
            temporary.unlink()
        raise


class C0BatchCaptureSessionV1:
    """Lifecycle observer for ten raw formal candidates and their sidecars.

    The session intentionally stops at *capture evidence*.  It persists the
    per-frame timestamp and gripper journals after each successful dataset
    save, but it does not fabricate dataset digests, VLM shadow results, or a
    finalized :class:`C0EpisodeManifestV1`.  Those remain explicit blockers in
    the session record and must be bound by a later offline finalization step.
    """

    def __init__(
        self,
        *,
        plan: C0BatchCapturePlanV1,
        prepare_episode: Callable[[C0EpisodeCapturePlanV1], None],
        human_outcome_provider: Callable[[C0EpisodeCapturePlanV1], str],
    ) -> None:
        if not isinstance(plan, C0BatchCapturePlanV1):
            raise TypeError("plan must be C0BatchCapturePlanV1")
        if not callable(prepare_episode):
            raise TypeError("prepare_episode must be callable")
        if not callable(human_outcome_provider):
            raise TypeError("human_outcome_provider must be callable")
        evidence_root = Path(plan.evidence_root).expanduser().resolve()
        if not evidence_root.is_dir():
            raise C0BatchCaptureError(
                "evidence_root must be exclusively created before constructing the session"
            )
        self._plan = copy.deepcopy(plan)
        self._prepare_episode = prepare_episode
        self._human_outcome_provider = human_outcome_provider
        self._evidence_root = evidence_root
        self._next_episode_index = 0
        self._active: _CaptureAttempt | None = None
        self._attempts: list[_CaptureAttempt] = []
        self._dataset_finalize_succeeded = False
        self._dataset_finalize_error: str | None = None

    @property
    def captured_episode_count(self) -> int:
        return sum(1 for attempt in self._attempts if attempt.status == "saved")

    @property
    def batch_capture_complete(self) -> bool:
        return (
            self.capture_session_complete
            and self._plan.episode_count == C0_FORMAL_EPISODE_COUNT
            and self._plan.episodes[0].formal_episode_index == 0
        )

    @property
    def capture_session_complete(self) -> bool:
        return (
            self._dataset_finalize_succeeded
            and self.captured_episode_count == self._plan.episode_count
            and self._next_episode_index == self._plan.episode_count
            and self._active is None
        )

    def episode_frame_observer(self) -> C0FormalFrameObserverV1:
        if self._active is not None:
            raise C0BatchCaptureError("previous episode attempt is still open")
        if self._next_episode_index >= self._plan.episode_count:
            raise C0BatchCaptureError("all capture-shard episode plans are already consumed")
        episode_plan = self._plan.episodes[self._next_episode_index]
        self._prepare_episode(copy.deepcopy(episode_plan))
        observer = C0FormalFrameObserverV1(
            timestamp_journal=C0SensorTimestampJournalV1(
                episode_id=episode_plan.episode_id,
                episode_index=episode_plan.episode_index,
            ),
            gripper_journal=C0GripperPendingCaptureJournalV1(
                episode_id=episode_plan.episode_id,
                episode_index=episode_plan.episode_index,
                fps=C0_DATASET_FPS,
            ),
        )
        attempt = _CaptureAttempt(plan=episode_plan, observer=observer)
        self._attempts.append(attempt)
        self._active = attempt
        return observer

    def on_episode_rerecord(
        self, *, robot: Any, dataset: Any, episode_index: int
    ) -> None:
        del robot, dataset
        attempt = self._require_active(episode_index)
        attempt.observer.timestamp_journal.mark_abandoned(reason="operator requested rerecord")
        attempt.observer.gripper_journal.mark_abandoned(reason="operator requested rerecord")
        attempt.status = "abandoned_rerecorded"
        self._active = None

    def on_episode_incomplete(
        self,
        *,
        robot: Any,
        dataset: Any,
        episode_index: int,
        reason: str,
    ) -> None:
        del robot, dataset
        attempt = self._require_active(episode_index)
        attempt.observer.timestamp_journal.mark_incomplete(reason=reason)
        attempt.observer.gripper_journal.mark_incomplete(reason=reason)
        attempt.status = "incomplete"
        self._active = None

    def save_episode(self, *, robot: Any, dataset: Any, episode_index: int) -> None:
        del robot
        attempt = self._require_active(episode_index)
        episode_buffer = getattr(dataset, "episode_buffer", None)
        if not isinstance(episode_buffer, Mapping):
            raise C0BatchCaptureError("dataset episode_buffer is missing before save")
        frame_count = episode_buffer.get("size")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            raise C0BatchCaptureError("dataset episode_buffer size must be a positive integer")
        if attempt.observer.timestamp_journal.frame_count != frame_count:
            raise C0BatchCaptureError(
                "timestamp frame count does not match the dataset episode buffer"
            )
        if len(attempt.observer.gripper_journal.cycles) != frame_count:
            raise C0BatchCaptureError(
                "gripper cycle count does not match the dataset episode buffer"
            )
        if attempt.observer.gripper_journal.episode_failed:
            raise C0BatchCaptureError("gripper journal failed; refusing dataset save")

        sidecar_path = self._evidence_root / f"episode-{episode_index:02d}-capture.json"
        temporary_sidecar = sidecar_path.with_name(f".{sidecar_path.name}.tmp")
        if sidecar_path.exists() or temporary_sidecar.exists():
            raise C0BatchCaptureError(
                f"evidence path already exists; refuse overwrite: {sidecar_path}"
            )
        outcome = require_human_outcome(self._human_outcome_provider(copy.deepcopy(attempt.plan)))
        dataset.save_episode()
        attempt.dataset_episode_durably_saved = True
        attempt.human_outcome = outcome
        attempt.status = "dataset_saved_pending_capture_sidecar"
        sidecar = self._episode_record(attempt, dataset_episode_durably_saved=True)
        sidecar["status"] = "saved"
        sidecar["sidecar_path"] = str(sidecar_path)
        try:
            _write_json_atomic_exclusive(sidecar_path, sidecar)
        except BaseException as error:
            attempt.status = "dataset_saved_capture_sidecar_failed"
            attempt.sidecar_write_error = f"{type(error).__name__}: {error}"
            self._active = None
            raise
        attempt.status = "saved"
        attempt.sidecar_path = str(sidecar_path)
        self._next_episode_index += 1
        self._active = None

    def on_dataset_finalize_success(
        self, *, dataset: Any, recorded_episode_count: int
    ) -> None:
        del dataset
        if (
            isinstance(recorded_episode_count, bool)
            or not isinstance(recorded_episode_count, int)
            or recorded_episode_count != self.captured_episode_count
        ):
            raise C0BatchCaptureError(
                "recorded_episode_count does not match saved capture sidecars"
            )
        self._dataset_finalize_succeeded = True
        session_path = self._evidence_root / "capture-session.json"
        _write_json_atomic_exclusive(session_path, self.to_manifest_record())

    def on_dataset_finalize_failure(self, *, dataset: Any, error: BaseException) -> None:
        del dataset
        self._dataset_finalize_error = f"{type(error).__name__}: {error}"

    def to_manifest_record(self) -> dict[str, object]:
        blockers = [
            "dataset_episode_digest_binding_pending",
            "gripper_post_save_audit_pending",
            "vlm_shadow_evidence_pending",
            "final_episode_manifest_binding_pending",
        ]
        if not self.batch_capture_complete:
            blockers.append("ten_episode_capture_incomplete")
        record: dict[str, object] = {
            "schema_version": C0_BATCH_CAPTURE_SESSION_SCHEMA_VERSION,
            "status": C0_FORMAL_BATCH_STATUS,
            "plan": self._plan.to_manifest_record(),
            "attempts": [self._attempt_record(item) for item in self._attempts],
            "captured_episode_count": self.captured_episode_count,
            "dataset_saved_episode_count": sum(
                1 for attempt in self._attempts if attempt.dataset_episode_durably_saved
            ),
            "expected_episode_count": C0_FORMAL_EPISODE_COUNT,
            "capture_shard_episode_count": self._plan.episode_count,
            "dataset_finalize_succeeded": self._dataset_finalize_succeeded,
            "dataset_finalize_error": self._dataset_finalize_error,
            "batch_capture_complete": self.batch_capture_complete,
            "capture_session_complete": self.capture_session_complete,
            "formal_final_evidence_bound": False,
            "blockers": sorted(blockers),
            **_authorization_false_record(),
        }
        return copy.deepcopy(record)

    def _require_active(self, episode_index: int) -> _CaptureAttempt:
        if self._active is None:
            raise C0BatchCaptureError("no active episode attempt")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index != self._active.plan.episode_index
        ):
            raise C0BatchCaptureError("episode_index does not match the active plan")
        return self._active

    def _episode_record(
        self, attempt: _CaptureAttempt, *, dataset_episode_durably_saved: bool
    ) -> dict[str, object]:
        record = self._attempt_record(attempt)
        record["dataset_episode_durably_saved"] = dataset_episode_durably_saved
        return record

    def _attempt_record(self, attempt: _CaptureAttempt) -> dict[str, object]:
        return {
            "episode_plan": attempt.plan.to_manifest_record(),
            "status": attempt.status,
            "human_outcome": attempt.human_outcome,
            "sidecar_path": attempt.sidecar_path,
            "sidecar_write_error": attempt.sidecar_write_error,
            "frame_evidence": attempt.observer.to_manifest_record(),
            "dataset_episode_durably_saved": attempt.dataset_episode_durably_saved,
            "formal_final_evidence_bound": False,
            **_authorization_false_record(),
        }

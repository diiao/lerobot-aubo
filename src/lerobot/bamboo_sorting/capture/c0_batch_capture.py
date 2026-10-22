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

"""Timestamp, suction-observer and evidence-write helpers used by joint capture.

Historical schema names are retained for existing capture records."""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from .c0_capture_contract import C0_SENSOR_STREAMS, C0SensorTimestampV1
from .c0_gripper_capture_journal import C0GripperPendingCaptureJournalV1

C0_SENSOR_TIMESTAMP_JOURNAL_SCHEMA_VERSION: Final = "C0SensorTimestampJournalV1"

C0_FORMAL_FRAME_OBSERVER_SCHEMA_VERSION: Final = "C0FormalFrameObserverV1"

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

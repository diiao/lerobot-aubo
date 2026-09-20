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

"""Offline-only capture contracts for the ten-episode C0-R1 pilot batch.

The records in this module describe data and evidence after capture. Importing
or serializing them never reads a camera, connects to AUBO, performs gripper
IO, starts training or inference, or creates live authorization.

The frozen cameras have nominal profiles of 30 Hz for ``global_rgb`` and 25 Hz
for ``grasp_rgb``. C0-R1 therefore names a 25 Hz *dataset* capture profile; this
does not claim that both devices have the same native FPS.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Final

from .contracts import (
    ACTION_SCHEMA_VERSION,
    INSTRUCTION_SPECS,
    validate_instruction_fields,
)
from .lerobot_bridge import CAMERASET_V2_LEROBOT_BRIDGE_VERSION
from .rgb_gate import (
    CAMERA_SET_V2_SCHEMA_VERSION,
    FIXED_RGB_STREAMS,
    FROZEN_CAMERA_SET_V2_SHA256,
    ROBOT_STATE,
)

C0_SENSOR_TIMESTAMP_SCHEMA_VERSION: Final = "C0SensorTimestampV1"
C0_EPISODE_MANIFEST_SCHEMA_VERSION: Final = "C0EpisodeManifestV1"
C0_BATCH_MANIFEST_SCHEMA_VERSION: Final = "C0BatchManifestV1"
C0_SPLIT_RULE_VERSION: Final = "C0SceneSplitV1"

C0_INSTRUCTION_ID: Final = "pick_any_collection"
_C0_INSTRUCTION_SPEC: Final = INSTRUCTION_SPECS[C0_INSTRUCTION_ID]
C0_SENSOR_STREAMS: Final = (*FIXED_RGB_STREAMS, ROBOT_STATE)
C0_SYNC_METHODS: Final = frozenset({"host_receive", "device_to_host_calibrated"})
C0_HUMAN_OUTCOMES: Final = frozenset({"success", "failure", "uncertain"})
C0_SPLIT_NAMES: Final = frozenset({"train", "validation", "test"})
C0_EXPECTED_EPISODE_COUNT: Final = 10

C0_CAPTURE_PROFILE_ID: Final = "CameraSetV2Dataset25HzV1"
C0_DATASET_FPS: Final = 25.0
GRIPPER_STATE_SEMANTICS: Final = "last_commanded_binary_not_physical_feedback"


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")


def _require_ref_sha_pair(ref_name: str, ref: object, sha_name: str, digest: object) -> None:
    has_ref = ref is not None
    has_sha = digest is not None
    if has_ref != has_sha:
        raise ValueError(f"{ref_name} and {sha_name} must be provided together")
    if not has_ref:
        raise ValueError(f"{ref_name} and {sha_name} are required")
    _require_nonempty_string(ref_name, ref)
    _require_sha256(sha_name, digest)


def _require_utc_timestamp(name: str, value: object) -> None:
    _require_nonempty_string(name, value)
    assert isinstance(value, str)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be valid ISO 8601") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must include the UTC timezone")


def _finite_non_negative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number, not bool")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return converted


@dataclass(frozen=True)
class C0SensorTimestampV1:
    """Timing evidence for one C0 sample from one frozen sensor stream.

    The class is intentionally independent of the legacy three-RGB
    ``EmbodiedObservationV1``. It permits only the two CameraSetV2 RGB streams
    and robot state, so wrist RGB, depth, ``handeye`` and ``fixed`` cannot enter
    C0 through this contract.
    """

    stream_name: str
    sync_timestamp_s: float
    host_receive_monotonic_s: float
    sync_method: str
    device_timestamp_s: float | None = None
    device_clock_id: str | None = None
    schema_version: str = C0_SENSOR_TIMESTAMP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != C0_SENSOR_TIMESTAMP_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_SENSOR_TIMESTAMP_SCHEMA_VERSION!r}"
            )
        if self.stream_name not in C0_SENSOR_STREAMS:
            raise ValueError(f"stream_name must be one of {C0_SENSOR_STREAMS}")

        sync_timestamp = _finite_non_negative("sync_timestamp_s", self.sync_timestamp_s)
        host_timestamp = _finite_non_negative(
            "host_receive_monotonic_s", self.host_receive_monotonic_s
        )
        object.__setattr__(self, "sync_timestamp_s", sync_timestamp)
        object.__setattr__(self, "host_receive_monotonic_s", host_timestamp)

        if self.sync_method not in C0_SYNC_METHODS:
            raise ValueError(f"sync_method must be one of {sorted(C0_SYNC_METHODS)}")

        has_device_time = self.device_timestamp_s is not None
        has_device_clock = self.device_clock_id is not None
        if has_device_time != has_device_clock:
            raise ValueError(
                "device_timestamp_s and device_clock_id must be provided together"
            )
        if has_device_time:
            device_timestamp = _finite_non_negative(
                "device_timestamp_s", self.device_timestamp_s
            )
            _require_nonempty_string("device_clock_id", self.device_clock_id)
            object.__setattr__(self, "device_timestamp_s", device_timestamp)

        if self.sync_method == "host_receive" and not math.isclose(
            sync_timestamp, host_timestamp, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "host_receive sync_timestamp_s must equal host_receive_monotonic_s"
            )
        if self.sync_method == "device_to_host_calibrated" and not has_device_time:
            raise ValueError(
                "device_to_host_calibrated requires device_timestamp_s and device_clock_id"
            )

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible timing record."""

        return copy.deepcopy(asdict(self))


@dataclass(frozen=True)
class C0EpisodeManifestV1:
    """One finalized single-strip C0 teleoperation episode for offline audit.

    ``split_name`` is episode-level because scene partitioning must be checked
    before any dataset split is consumed. Authorization fields are evidence
    references only; the record never grants a current teleoperation session.
    """

    schema_version: str
    episode_id: str
    scene_id: str
    session_id: str
    split_name: str
    camera_set_schema_version: str
    camera_set_sha256: str
    bridge_version: str
    action_schema_version: str
    instruction_schema_version: str
    instruction_id: str
    instruction_text: str
    instruction_language: str
    instruction_text_sha256: str
    calibration_version: str
    tool_version: str
    capture_profile_id: str
    fps: float
    frame_count: int
    dataset_episode_ref: str | None
    dataset_episode_sha256: str | None
    timestamp_evidence_ref: str | None
    timestamp_evidence_sha256: str | None
    authorization_evidence_ref: str | None
    authorization_evidence_sha256: str | None
    vlm_shadow_evidence_ref: str | None
    vlm_shadow_evidence_sha256: str | None
    human_outcome: str
    human_outcome_evidence_ref: str | None
    human_outcome_evidence_sha256: str | None
    raw_data_immutable: bool
    finalized: bool
    gripper_state_semantics: str = GRIPPER_STATE_SEMANTICS
    gripper_physical_mapping_verified: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != C0_EPISODE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_EPISODE_MANIFEST_SCHEMA_VERSION!r}"
            )
        for name in (
            "episode_id",
            "scene_id",
            "session_id",
            "calibration_version",
            "tool_version",
        ):
            _require_nonempty_string(name, getattr(self, name))

        if self.split_name not in C0_SPLIT_NAMES:
            raise ValueError(f"split_name must be one of {sorted(C0_SPLIT_NAMES)}")
        if self.camera_set_schema_version != CAMERA_SET_V2_SCHEMA_VERSION:
            raise ValueError(
                f"camera_set_schema_version must be {CAMERA_SET_V2_SCHEMA_VERSION!r}"
            )
        _require_sha256("camera_set_sha256", self.camera_set_sha256)
        if self.camera_set_sha256 != FROZEN_CAMERA_SET_V2_SHA256:
            raise ValueError(
                "camera_set_sha256 must equal the frozen CameraSetV2 SHA-256"
            )
        if self.bridge_version != CAMERASET_V2_LEROBOT_BRIDGE_VERSION:
            raise ValueError(
                f"bridge_version must be {CAMERASET_V2_LEROBOT_BRIDGE_VERSION!r}"
            )
        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"action_schema_version must be {ACTION_SCHEMA_VERSION!r}"
            )

        instruction = validate_instruction_fields(
            schema_version=self.instruction_schema_version,
            instruction_id=self.instruction_id,
            instruction_text=self.instruction_text,
            instruction_language=self.instruction_language,
            instruction_text_sha256=self.instruction_text_sha256,
        )
        if instruction is not _C0_INSTRUCTION_SPEC:
            raise ValueError(f"C0-R1 only accepts instruction_id {C0_INSTRUCTION_ID!r}")

        if self.capture_profile_id != C0_CAPTURE_PROFILE_ID:
            raise ValueError(
                f"capture_profile_id must be {C0_CAPTURE_PROFILE_ID!r} for C0-R1"
            )
        fps = _finite_non_negative("fps", self.fps)
        if fps != C0_DATASET_FPS:
            raise ValueError(
                f"fps must be {C0_DATASET_FPS:g} for capture profile {C0_CAPTURE_PROFILE_ID!r}"
            )
        object.__setattr__(self, "fps", fps)
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count <= 0
        ):
            raise ValueError("frame_count must be a positive integer, not bool or float")

        for ref_name, sha_name in (
            ("dataset_episode_ref", "dataset_episode_sha256"),
            ("timestamp_evidence_ref", "timestamp_evidence_sha256"),
            ("authorization_evidence_ref", "authorization_evidence_sha256"),
            ("vlm_shadow_evidence_ref", "vlm_shadow_evidence_sha256"),
            ("human_outcome_evidence_ref", "human_outcome_evidence_sha256"),
        ):
            _require_ref_sha_pair(
                ref_name,
                getattr(self, ref_name),
                sha_name,
                getattr(self, sha_name),
            )

        if self.human_outcome not in C0_HUMAN_OUTCOMES:
            raise ValueError(
                f"human_outcome must be one of {sorted(C0_HUMAN_OUTCOMES)}"
            )
        for name in ("raw_data_immutable", "finalized"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
            if not getattr(self, name):
                raise ValueError(f"{name} must be True for a completed C0 episode")
        if self.gripper_state_semantics != GRIPPER_STATE_SEMANTICS:
            raise ValueError(
                f"gripper_state_semantics must be {GRIPPER_STATE_SEMANTICS!r}"
            )
        if not isinstance(self.gripper_physical_mapping_verified, bool):
            raise ValueError("gripper_physical_mapping_verified must be bool")
        if self.gripper_physical_mapping_verified:
            raise ValueError("gripper_physical_mapping_verified must remain False in C0-R1")

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible record with fixed no-authorization fields."""

        record = copy.deepcopy(asdict(self))
        record["sensor_streams"] = list(C0_SENSOR_STREAMS)
        record["serialized_record_grants_live_authorization"] = False
        record["policy_execution_authorized"] = False
        record["training_authorized"] = False
        record["hardware_access_performed_by_serialization"] = False
        return copy.deepcopy(record)


@dataclass(frozen=True)
class C0BatchManifestV1:
    """The frozen ten-episode C0 pilot batch, ready only for human review.

    C0 uses one new ``scene_id`` per episode because completing a single-strip
    pick or repositioning any strip changes the physical scene. The split is
    therefore assigned by whole scene and never by frame. Data readiness here
    means only that offline evidence is complete enough for C0 review.
    """

    schema_version: str
    batch_id: str
    episodes: tuple[C0EpisodeManifestV1, ...]
    split_rule_version: str
    finalized_at_utc: str
    frozen: bool
    expected_episode_count: int = C0_EXPECTED_EPISODE_COUNT

    def __post_init__(self) -> None:
        if self.schema_version != C0_BATCH_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_BATCH_MANIFEST_SCHEMA_VERSION!r}"
            )
        _require_nonempty_string("batch_id", self.batch_id)
        if self.split_rule_version != C0_SPLIT_RULE_VERSION:
            raise ValueError(f"split_rule_version must be {C0_SPLIT_RULE_VERSION!r}")
        _require_utc_timestamp("finalized_at_utc", self.finalized_at_utc)
        if not isinstance(self.frozen, bool):
            raise ValueError("frozen must be bool")
        if (
            isinstance(self.expected_episode_count, bool)
            or not isinstance(self.expected_episode_count, int)
            or self.expected_episode_count != C0_EXPECTED_EPISODE_COUNT
        ):
            raise ValueError(
                f"expected_episode_count must remain {C0_EXPECTED_EPISODE_COUNT}"
            )

        if isinstance(self.episodes, (str, bytes)):
            raise ValueError("episodes must contain C0EpisodeManifestV1 records")
        episodes = tuple(self.episodes)
        if len(episodes) != C0_EXPECTED_EPISODE_COUNT:
            raise ValueError(
                f"C0 batch requires exactly {C0_EXPECTED_EPISODE_COUNT} episodes"
            )
        if not all(isinstance(episode, C0EpisodeManifestV1) for episode in episodes):
            raise ValueError("episodes must contain only C0EpisodeManifestV1 records")
        object.__setattr__(self, "episodes", episodes)

        episode_ids = [episode.episode_id for episode in episodes]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("episode_id must be globally unique within a C0 batch")
        scene_ids = [episode.scene_id for episode in episodes]
        if len(set(scene_ids)) != len(scene_ids):
            scene_splits: dict[str, set[str]] = {}
            for episode in episodes:
                scene_splits.setdefault(episode.scene_id, set()).add(episode.split_name)
            cross_split = sorted(
                scene_id for scene_id, splits in scene_splits.items() if len(splits) > 1
            )
            if cross_split:
                raise ValueError(f"scene_id cannot cross split: {cross_split}")
            raise ValueError(
                "C0 requires a new scene_id after each single-strip episode"
            )

        consistent_fields = (
            "camera_set_schema_version",
            "camera_set_sha256",
            "bridge_version",
            "action_schema_version",
            "instruction_schema_version",
            "instruction_id",
            "calibration_version",
            "tool_version",
            "capture_profile_id",
            "fps",
        )
        first = episodes[0]
        for name in consistent_fields:
            if any(getattr(episode, name) != getattr(first, name) for episode in episodes[1:]):
                raise ValueError(f"C0 batch configuration drift in {name}")

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def total_frame_count(self) -> int:
        return sum(episode.frame_count for episode in self.episodes)

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple(sorted(episode.scene_id for episode in self.episodes))

    @property
    def blockers(self) -> tuple[str, ...]:
        """Return review-readiness blockers in stable sorted order."""

        blockers = []
        if not self.frozen:
            blockers.append("batch_not_frozen")
        return tuple(sorted(blockers))

    @property
    def data_ready_for_c0_review(self) -> bool:
        """Return offline review readiness, never hardware or training permission."""

        return not self.blockers

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible batch record with no authorization."""

        record = {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "episodes": [episode.to_manifest_record() for episode in self.episodes],
            "expected_episode_count": self.expected_episode_count,
            "episode_count": self.episode_count,
            "total_frame_count": self.total_frame_count,
            "scene_ids": list(self.scene_ids),
            "split_rule_version": self.split_rule_version,
            "finalized_at_utc": self.finalized_at_utc,
            "frozen": self.frozen,
            "blockers": list(self.blockers),
            "data_ready_for_c0_review": self.data_ready_for_c0_review,
            "policy_execution_authorized": False,
            "training_authorized": False,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_serialization": False,
        }
        return copy.deepcopy(record)

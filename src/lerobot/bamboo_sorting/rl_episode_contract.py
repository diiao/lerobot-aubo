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

"""Offline RL episode, reset, and replay audit contracts (R0B).

These dataclasses are immutable, pure-data audit records. They describe
resets and episodes that already happened; they never execute a reset, never
hold device handles, camera objects, RPC clients, numpy image payloads, or
callbacks, and they never grant live authorization. No in-memory replay
buffer is implemented here.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Final

from .contracts import INSTRUCTION_SPECS
from .rl_contract import RLObservationRefV1, RLTransitionV1

RESET_RECORD_SCHEMA_VERSION: Final = "ResetRecordV1"
RL_EPISODE_MANIFEST_SCHEMA_VERSION: Final = "RLEpisodeManifestV1"
RL_REPLAY_MANIFEST_SCHEMA_VERSION: Final = "RLReplayManifestV1"

RESET_REASONS: Final = frozenset(
    {"initial_setup", "after_done", "after_truncated", "operator_requested", "safety_abort_recovery"}
)
EPISODE_TERMINATION_REASONS: Final = frozenset(
    {"task_success", "task_failure", "time_limit", "safety_abort", "operator_abort"}
)
REPLAY_SPLIT_NAMES: Final = frozenset({"train", "validation", "test"})


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _require_nonempty_string(name: str, value: object) -> None:
    if not _require_string(name, value).strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_timestamp(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number, not bool")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _require_instruction_id(name: str, value: object) -> None:
    instruction_id = _require_string(name, value)
    instruction = INSTRUCTION_SPECS.get(instruction_id)
    if instruction is None:
        raise ValueError(f"Unknown instruction_id {instruction_id!r}")
    if instruction.is_control:
        raise ValueError(f"Control instruction {instruction_id!r} is not valid here")


@dataclass(frozen=True)
class ResetRecordV1:
    """One immutable record of a reset that already happened or failed.

    This record never executes a reset. ``automatic_reset_authorized`` must
    stay false in this version; serialized records can never create live
    permission to move hardware.
    """

    schema_version: str
    reset_id: str
    target_episode_id: str
    reset_reason: str
    requested_monotonic_s: float
    completed_monotonic_s: float | None
    reset_completed: bool
    human_verified: bool
    pre_reset_observation_id: str | None
    post_reset_observation: RLObservationRefV1 | None
    evidence_ref: str
    evidence_sha256: str
    automatic_reset_authorized: bool

    def __post_init__(self) -> None:
        if self.schema_version != RESET_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported reset record schema {self.schema_version!r}; "
                f"expected {RESET_RECORD_SCHEMA_VERSION!r}"
            )
        for name in ("reset_id", "target_episode_id", "evidence_ref"):
            _require_nonempty_string(name, getattr(self, name))

        reset_reason = _require_string("reset_reason", self.reset_reason)
        if reset_reason not in RESET_REASONS:
            raise ValueError(f"reset_reason must be one of {sorted(RESET_REASONS)}")

        requested = _require_timestamp("requested_monotonic_s", self.requested_monotonic_s)
        object.__setattr__(self, "requested_monotonic_s", requested)
        if self.completed_monotonic_s is None:
            completed = None
        else:
            completed = _require_timestamp("completed_monotonic_s", self.completed_monotonic_s)
            object.__setattr__(self, "completed_monotonic_s", completed)

        for name in ("reset_completed", "human_verified", "automatic_reset_authorized"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if self.automatic_reset_authorized:
            raise ValueError("automatic_reset_authorized must be False in this version")

        if self.pre_reset_observation_id is not None:
            _require_nonempty_string("pre_reset_observation_id", self.pre_reset_observation_id)
        if reset_reason != "initial_setup" and self.pre_reset_observation_id is None:
            raise ValueError(f"reset_reason {reset_reason!r} requires a non-empty pre_reset_observation_id")

        if self.post_reset_observation is not None and not isinstance(
            self.post_reset_observation, RLObservationRefV1
        ):
            raise ValueError("post_reset_observation must be an RLObservationRefV1 or None")
        _require_sha256("evidence_sha256", self.evidence_sha256)

        if self.reset_completed:
            if completed is None:
                raise ValueError("a completed reset requires completed_monotonic_s")
            if self.post_reset_observation is None:
                raise ValueError("a completed reset requires post_reset_observation")
            if completed <= requested:
                raise ValueError("completed_monotonic_s must be later than requested_monotonic_s")
            assert self.post_reset_observation is not None
            if self.post_reset_observation.sync_timestamp_s <= completed:
                raise ValueError(
                    "post_reset_observation.sync_timestamp_s must be later than "
                    "completed_monotonic_s"
                )
            if not self.human_verified:
                raise ValueError("a completed reset requires human_verified=True")
        else:
            if completed is not None:
                raise ValueError("a failed reset must leave completed_monotonic_s empty")
            if self.post_reset_observation is not None:
                raise ValueError("a failed reset must leave post_reset_observation empty")
            if self.human_verified:
                raise ValueError("a failed reset requires human_verified=False")

    def to_manifest_record(self) -> dict[str, object]:
        """Return a JSON-serializable deep copy that grants no authorization."""

        record = {
            "schema_version": self.schema_version,
            "reset_id": self.reset_id,
            "target_episode_id": self.target_episode_id,
            "reset_reason": self.reset_reason,
            "requested_monotonic_s": self.requested_monotonic_s,
            "completed_monotonic_s": self.completed_monotonic_s,
            "reset_completed": self.reset_completed,
            "human_verified": self.human_verified,
            "pre_reset_observation_id": self.pre_reset_observation_id,
            "post_reset_observation": (
                None
                if self.post_reset_observation is None
                else self.post_reset_observation.to_manifest_record()
            ),
            "evidence_ref": self.evidence_ref,
            "evidence_sha256": self.evidence_sha256,
            "automatic_reset_authorized": self.automatic_reset_authorized,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_serialization": False,
        }
        return copy.deepcopy(record)


@dataclass(frozen=True)
class RLEpisodeManifestV1:
    """One frozen, finalized episode manifest for offline audit and replay.

    Transitions must form a single chain: each ``next_observation`` must
    equal the next transition's ``observation``, step indices must be exactly
    ``0..N-1``, and only the final transition may terminate the episode.
    Scene ids may change mid-episode; the canonical instruction may not.
    """

    schema_version: str
    episode_id: str
    instruction_id: str
    start_reset: ResetRecordV1
    transitions: tuple[RLTransitionV1, ...]
    termination_reason: str
    finalized: bool

    def __post_init__(self) -> None:
        if self.schema_version != RL_EPISODE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported episode manifest schema {self.schema_version!r}; "
                f"expected {RL_EPISODE_MANIFEST_SCHEMA_VERSION!r}"
            )
        _require_nonempty_string("episode_id", self.episode_id)
        _require_instruction_id("instruction_id", self.instruction_id)

        if not isinstance(self.start_reset, ResetRecordV1):
            raise ValueError("start_reset must be a ResetRecordV1")
        if self.start_reset.target_episode_id != self.episode_id:
            raise ValueError("start_reset.target_episode_id must equal episode_id")
        if not self.start_reset.reset_completed or not self.start_reset.human_verified:
            raise ValueError("start_reset must be completed and human verified")

        if isinstance(self.transitions, (str, bytes)) or not self.transitions:
            raise ValueError("transitions must be a non-empty tuple of RLTransitionV1")
        transitions = tuple(self.transitions)
        if not all(isinstance(t, RLTransitionV1) for t in transitions):
            raise ValueError("transitions must contain only RLTransitionV1")
        object.__setattr__(self, "transitions", transitions)

        if self.start_reset.post_reset_observation != transitions[0].observation:
            raise ValueError(
                "start_reset.post_reset_observation must equal transitions[0].observation"
            )

        termination_reason = _require_string("termination_reason", self.termination_reason)
        if termination_reason not in EPISODE_TERMINATION_REASONS:
            raise ValueError(
                f"termination_reason must be one of {sorted(EPISODE_TERMINATION_REASONS)}"
            )
        if not isinstance(self.finalized, bool):
            raise ValueError("finalized must be bool")
        if not self.finalized:
            raise ValueError("finalized must be True in this version")

        transition_ids: set[str] = set()
        for index, transition in enumerate(transitions):
            if transition.episode_id != self.episode_id:
                raise ValueError(f"transition {transition.transition_id!r} has a foreign episode_id")
            if transition.transition_id in transition_ids:
                raise ValueError(f"duplicate transition_id {transition.transition_id!r}")
            transition_ids.add(transition.transition_id)
            if transition.step_index != index:
                raise ValueError("step_index must be exactly 0..N-1 in order")
            if transition.observation.instruction_id != self.instruction_id:
                raise ValueError("all transitions must use the episode's canonical instruction")
            if index < len(transitions) - 1 and (transition.done or transition.truncated):
                raise ValueError("only the final transition may terminate the episode")
            if index + 1 < len(transitions) and transition.next_observation != transitions[index + 1].observation:
                raise ValueError(
                    f"chain break: transition {transition.transition_id!r} next_observation "
                    f"does not equal the following transition's observation"
                )

        last = transitions[-1]
        if not (last.done or last.truncated):
            raise ValueError("the final transition must end the episode (done or truncated)")
        if termination_reason == "task_success":
            if not last.done or last.reward.task_success != 1:
                raise ValueError("task_success requires final done=True and reward.task_success=1")
        elif termination_reason == "task_failure":
            if not last.done or last.reward.task_success != 0:
                raise ValueError("task_failure requires final done=True and reward.task_success=0")
        else:
            if not last.truncated or last.reward.task_success != 0:
                raise ValueError(
                    f"{termination_reason} requires final truncated=True and reward.task_success=0"
                )
            if termination_reason == "safety_abort" and last.reward.safety_cost <= 0:
                raise ValueError("safety_abort requires final reward.safety_cost>0")

    @property
    def transition_count(self) -> int:
        return len(self.transitions)

    def to_manifest_record(self) -> dict[str, object]:
        """Return a JSON-serializable deep copy that grants no authorization."""

        record = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "instruction_id": self.instruction_id,
            "start_reset": self.start_reset.to_manifest_record(),
            "transitions": [t.to_manifest_record() for t in self.transitions],
            "termination_reason": self.termination_reason,
            "finalized": self.finalized,
            "transition_count": self.transition_count,
            "policy_execution_authorized": False,
            "hardware_access_performed": False,
            "serialized_record_grants_live_authorization": False,
        }
        return copy.deepcopy(record)


@dataclass(frozen=True)
class RLReplayManifestV1:
    """One frozen replay dataset manifest, computed from its episodes.

    This is an audit manifest, not an in-memory replay buffer and not a
    training authorization. Counts and scene ids are derived from the episode
    contents so callers cannot forge them. Because the generic LeRobot RL
    buffer still drops ``truncated`` on the dataset round-trip,
    ``data_ready_for_training`` stays false until
    ``upstream_truncated_roundtrip_verified`` is backed by explicit evidence
    (reference plus SHA-256). Data readiness never grants permission to run
    training.
    """

    schema_version: str
    replay_id: str
    source_dataset_ref: str
    source_dataset_sha256: str
    split_name: str
    episodes: tuple[RLEpisodeManifestV1, ...]
    frozen: bool
    upstream_truncated_roundtrip_verified: bool
    upstream_truncated_roundtrip_evidence_ref: str | None = None
    upstream_truncated_roundtrip_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != RL_REPLAY_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported replay manifest schema {self.schema_version!r}; "
                f"expected {RL_REPLAY_MANIFEST_SCHEMA_VERSION!r}"
            )
        for name in ("replay_id", "source_dataset_ref"):
            _require_nonempty_string(name, getattr(self, name))
        _require_sha256("source_dataset_sha256", self.source_dataset_sha256)

        split_name = _require_string("split_name", self.split_name)
        if split_name not in REPLAY_SPLIT_NAMES:
            raise ValueError(f"split_name must be one of {sorted(REPLAY_SPLIT_NAMES)}")

        if isinstance(self.episodes, (str, bytes)) or not self.episodes:
            raise ValueError("episodes must be a non-empty tuple of RLEpisodeManifestV1")
        episodes = tuple(self.episodes)
        if not all(isinstance(e, RLEpisodeManifestV1) for e in episodes):
            raise ValueError("episodes must contain only RLEpisodeManifestV1")
        object.__setattr__(self, "episodes", episodes)

        for name in ("frozen", "upstream_truncated_roundtrip_verified"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if not self.frozen:
            raise ValueError("frozen must be True in this version")

        has_ref = self.upstream_truncated_roundtrip_evidence_ref is not None
        has_sha = self.upstream_truncated_roundtrip_evidence_sha256 is not None
        if has_ref != has_sha:
            raise ValueError(
                "upstream_truncated_roundtrip evidence ref and sha256 must be provided together"
            )
        if self.upstream_truncated_roundtrip_verified:
            if not (has_ref and has_sha):
                raise ValueError(
                    "a verified truncation round-trip requires both evidence ref and sha256"
                )
        elif has_ref or has_sha:
            raise ValueError(
                "an unverified truncation round-trip must leave evidence ref and sha256 empty"
            )
        if has_ref:
            _require_nonempty_string(
                "upstream_truncated_roundtrip_evidence_ref",
                self.upstream_truncated_roundtrip_evidence_ref,
            )
        if has_sha:
            _require_sha256(
                "upstream_truncated_roundtrip_evidence_sha256",
                self.upstream_truncated_roundtrip_evidence_sha256,
            )

        episode_ids: set[str] = set()
        transition_ids: set[str] = set()
        observation_ids: set[str] = set()
        for episode in episodes:
            if episode.episode_id in episode_ids:
                raise ValueError(f"duplicate episode_id {episode.episode_id!r}")
            episode_ids.add(episode.episode_id)
            for transition in episode.transitions:
                if transition.transition_id in transition_ids:
                    raise ValueError(f"duplicate transition_id {transition.transition_id!r}")
                transition_ids.add(transition.transition_id)
                if transition.observation.observation_id in observation_ids:
                    raise ValueError(f"duplicate observation_id {transition.observation.observation_id!r}")
                observation_ids.add(transition.observation.observation_id)
            terminal_ref = episode.transitions[-1].next_observation
            if terminal_ref.observation_id in observation_ids:
                raise ValueError(f"duplicate observation_id {terminal_ref.observation_id!r}")
            observation_ids.add(terminal_ref.observation_id)

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def transition_count(self) -> int:
        return sum(episode.transition_count for episode in self.episodes)

    @property
    def scene_ids(self) -> tuple[str, ...]:
        scenes = {
            ref.scene_id
            for episode in self.episodes
            for transition in episode.transitions
            for ref in (transition.observation, transition.next_observation)
        }
        return tuple(sorted(scenes))

    @property
    def blockers(self) -> tuple[str, ...]:
        """Return readiness blockers in stable order."""

        if not self.upstream_truncated_roundtrip_verified:
            return ("upstream_truncated_roundtrip_unverified",)
        return ()

    @property
    def data_ready_for_training(self) -> bool:
        """Return whether the replay data contract is evidenced compatible.

        This only audits data-contract compatibility; it never grants
        authorization to run training.
        """

        return self.frozen and not self.blockers

    def to_manifest_record(self) -> dict[str, object]:
        """Return a JSON-serializable deep copy that grants no authorization."""

        record = {
            "schema_version": self.schema_version,
            "replay_id": self.replay_id,
            "source_dataset_ref": self.source_dataset_ref,
            "source_dataset_sha256": self.source_dataset_sha256,
            "split_name": self.split_name,
            "episodes": [e.to_manifest_record() for e in self.episodes],
            "frozen": self.frozen,
            "upstream_truncated_roundtrip_verified": self.upstream_truncated_roundtrip_verified,
            "upstream_truncated_roundtrip_evidence_ref": self.upstream_truncated_roundtrip_evidence_ref,
            "upstream_truncated_roundtrip_evidence_sha256": self.upstream_truncated_roundtrip_evidence_sha256,
            "episode_count": self.episode_count,
            "transition_count": self.transition_count,
            "scene_ids": list(self.scene_ids),
            "data_ready_for_training": self.data_ready_for_training,
            "blockers": list(self.blockers),
            "training_authorized": False,
            "policy_execution_authorized": False,
            "hardware_access_performed": False,
            "serialized_record_grants_live_authorization": False,
        }
        return copy.deepcopy(record)

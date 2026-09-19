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

"""Offline RL contracts for the R0 gate: observation refs, reward, transition, readiness.

These dataclasses are immutable, pure-data audit records. They never hold
device handles, camera objects, RPC clients, numpy image payloads, or
executable callbacks, and they never grant live authorization. Reward fields
keep task reward, safety cost, and human takeover strictly separate so one
signal can never overwrite another.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final

from .contracts import ACTION_SCHEMA_VERSION, INSTRUCTION_SPECS, validate_action_vector
from .lerobot_bridge import CAMERASET_V1_LEROBOT_BRIDGE_VERSION
from .rgb_gate import CAMERA_SET_SCHEMA_VERSION, FROZEN_CAMERA_SET_V1_SHA256

RL_OBSERVATION_REF_SCHEMA_VERSION: Final = "RLObservationRefV1"
REWARD_SCHEMA_VERSION: Final = "RewardSchemaV1"
RL_TRANSITION_SCHEMA_VERSION: Final = "RLTransitionV1"
RL_READINESS_SCHEMA_VERSION: Final = "RLReadinessReportV1"

REWARD_SOURCES: Final = frozenset({"human", "learned"})
TRANSITION_CONTROL_SOURCES: Final = frozenset({"policy", "intervention", "demonstration"})
RL_READINESS_MODES: Final = frozenset({"rabc", "hil_serl_sac", "smolvla_rl"})

# Canonical gate order; blockers are always emitted in this exact order.
_RABC_GATES: Final = (
    "data_split_frozen",
    "reward_schema_frozen",
    "independent_reward_test_set",
    "bc_baseline_available",
)
_HIL_SERL_SAC_EXTRA_GATES: Final = (
    "aubo_env_adapter",
    "explicit_reset",
    "intervention_recording",
    "replay_buffer",
    "actor_learner",
    "thin_safety_gate_integration",
    "sim_or_null_actuator_passed",
)
_SMOLVLA_RL_EXTRA_GATES: Final = (
    "replay_buffer",
    "thin_safety_gate_integration",
    "safety_gate_isolation",
    "chunk_critic",
    "behavior_or_kl_constraint",
    "offline_policy_evaluation",
    "language_counterfactual_tests",
    "checkpoint_provenance",
    "reward_hacking_checks",
)
MODE_REQUIRED_GATES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "rabc": _RABC_GATES,
        "hil_serl_sac": (*_RABC_GATES, *_HIL_SERL_SAC_EXTRA_GATES),
        "smolvla_rl": (*_RABC_GATES, *_SMOLVLA_RL_EXTRA_GATES),
    }
)


def _require_string(name: str, value: object) -> str:
    """Return ``value`` as a string or raise a clear ValueError for any other type."""

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


@dataclass(frozen=True)
class RLObservationRefV1:
    """Reference-and-hash binding to one formal two-RGB CameraSetV1 model input.

    The transition pipeline stores only this identity: observation and content
    hashes, never numpy image payloads. Wrist RGB and depth streams are frozen
    out of CameraSetV1 and therefore cannot appear here.
    """

    schema_version: str
    observation_id: str
    scene_id: str
    instruction_id: str
    camera_set_schema_version: str
    camera_set_sha256: str
    bridge_version: str
    model_input_ref: str
    model_input_sha256: str
    sync_timestamp_s: float
    action_schema_version: str

    def __post_init__(self) -> None:
        if self.schema_version != RL_OBSERVATION_REF_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported observation ref schema {self.schema_version!r}; "
                f"expected {RL_OBSERVATION_REF_SCHEMA_VERSION!r}"
            )
        for name in ("observation_id", "scene_id", "model_input_ref"):
            _require_nonempty_string(name, getattr(self, name))

        instruction_id = _require_string("instruction_id", self.instruction_id)
        instruction = INSTRUCTION_SPECS.get(instruction_id)
        if instruction is None:
            raise ValueError(f"Unknown instruction_id {instruction_id!r}")
        if instruction.is_control:
            raise ValueError("Control instructions cannot anchor RL observation refs")

        if self.camera_set_schema_version != CAMERA_SET_SCHEMA_VERSION:
            raise ValueError(
                f"camera_set_schema_version must be {CAMERA_SET_SCHEMA_VERSION!r}; "
                f"wrist_rgb/depth camera sets are not valid RL inputs"
            )
        if self.camera_set_sha256 != FROZEN_CAMERA_SET_V1_SHA256:
            raise ValueError("camera_set_sha256 must equal FROZEN_CAMERA_SET_V1_SHA256")
        if self.bridge_version != CAMERASET_V1_LEROBOT_BRIDGE_VERSION:
            raise ValueError(
                f"bridge_version must be {CAMERASET_V1_LEROBOT_BRIDGE_VERSION!r}"
            )
        _require_sha256("model_input_sha256", self.model_input_sha256)

        timestamp = self.sync_timestamp_s
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError("sync_timestamp_s must be a finite non-negative number")
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("sync_timestamp_s must be a finite non-negative number")
        object.__setattr__(self, "sync_timestamp_s", float(timestamp))

        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"action_schema_version must be {ACTION_SCHEMA_VERSION!r}"
            )

    def to_manifest_record(self) -> dict[str, object]:
        """Return a JSON-serializable deep copy that grants no authorization."""

        record = copy.deepcopy(asdict(self))
        record["serialized_record_grants_live_authorization"] = False
        return record


@dataclass(frozen=True)
class RewardSchemaV1:
    """One immutable reward record with task, safety, and takeover separated.

    ``task_success`` is the sparse task reward (0 or 1), ``safety_cost`` is a
    separate non-negative penalty signal, and ``intervention`` marks human
    takeover. These fields must never be merged or overwrite each other. The
    reward annotates the observation recorded after the action executed.
    """

    schema_version: str
    observation_id: str
    scene_id: str
    instruction_id: str
    task_success: int
    uncertain: bool
    safety_cost: float
    intervention: bool
    reward_source: str
    reward_model_ref: str | None = None
    reward_model_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != REWARD_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported reward schema {self.schema_version!r}; "
                f"expected {REWARD_SCHEMA_VERSION!r}"
            )
        for name in ("observation_id", "scene_id"):
            _require_nonempty_string(name, getattr(self, name))

        instruction_id = _require_string("instruction_id", self.instruction_id)
        instruction = INSTRUCTION_SPECS.get(instruction_id)
        if instruction is None:
            raise ValueError(f"Unknown instruction_id {instruction_id!r}")
        if instruction.is_control:
            raise ValueError("Control instructions cannot carry deployable task rewards")

        if isinstance(self.task_success, bool) or not isinstance(self.task_success, int):
            raise ValueError("task_success must be the integer 0 or 1, not bool")
        if self.task_success not in (0, 1):
            raise ValueError("task_success must be 0 or 1")
        for name in ("uncertain", "intervention"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if self.uncertain and self.task_success == 1:
            raise ValueError("uncertain=true forbids task_success=1; route it to human review")

        if isinstance(self.safety_cost, bool) or not isinstance(self.safety_cost, (int, float)):
            raise ValueError("safety_cost must be a finite non-negative number, not bool")
        if not math.isfinite(self.safety_cost) or self.safety_cost < 0:
            raise ValueError("safety_cost must be a finite non-negative number")
        object.__setattr__(self, "safety_cost", float(self.safety_cost))

        reward_source = _require_string("reward_source", self.reward_source)
        if reward_source not in REWARD_SOURCES:
            raise ValueError(f"reward_source must be one of {sorted(REWARD_SOURCES)}")
        has_ref = self.reward_model_ref is not None
        has_sha = self.reward_model_sha256 is not None
        if reward_source == "human":
            if has_ref or has_sha:
                raise ValueError("human rewards must leave reward_model_ref and reward_model_sha256 empty")
        elif not (has_ref and has_sha):
            raise ValueError("learned rewards require both reward_model_ref and reward_model_sha256")
        if has_ref:
            _require_nonempty_string("reward_model_ref", self.reward_model_ref)
        if has_sha:
            _require_sha256("reward_model_sha256", self.reward_model_sha256)

    @property
    def requires_human_review(self) -> bool:
        """Return whether this reward lacks enough evidence to be trusted."""

        return self.uncertain

    def to_manifest_record(self) -> dict[str, object]:
        """Return a JSON-serializable deep copy that grants no authorization."""

        record = copy.deepcopy(asdict(self))
        record["serialized_record_grants_live_authorization"] = False
        return record


@dataclass(frozen=True)
class RLTransitionV1:
    """One immutable executed transition for offline RL audit and replay.

    Observations are bound by :class:`RLObservationRefV1` identity, and the
    reward must annotate ``next_observation`` (the observation recorded after
    the action executed): its observation, scene, and instruction ids must
    match ``next_observation``. Scene ids may differ between the current and
    next observation; instruction ids may not. The two observations must also
    have distinct observation ids, and ``next_observation`` must carry a
    strictly later ``sync_timestamp_s``.

    Invariants: ``done`` and ``truncated`` are bool and never both true;
    ``done`` means a terminal task decision was reached, while ``truncated``
    means the episode ended by horizon/reset without a terminal decision.
    For the sparse terminal reward, ``task_success=1`` requires ``done=True``.
    ``control_source='policy'`` requires ``action_gate_passed=True``;
    ``intervention`` and ``demonstration`` may use ``None`` or ``True`` but
    never ``False``. Every source requires ``action_execution_confirmed=True``
    backed by a non-empty ``execution_evidence_ref``, so a rejected,
    unconfirmed, or unattributed candidate action can never be recorded as
    executed.
    """

    schema_version: str
    transition_id: str
    episode_id: str
    step_index: int
    observation: RLObservationRefV1
    action: Sequence[object]
    next_observation: RLObservationRefV1
    reward: RewardSchemaV1
    done: bool
    truncated: bool
    control_source: str
    action_gate_passed: bool | None
    action_execution_confirmed: bool
    execution_evidence_ref: str

    def __post_init__(self) -> None:
        if self.schema_version != RL_TRANSITION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported transition schema {self.schema_version!r}; "
                f"expected {RL_TRANSITION_SCHEMA_VERSION!r}"
            )
        for name in ("transition_id", "episode_id", "execution_evidence_ref"):
            _require_nonempty_string(name, getattr(self, name))

        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise ValueError("step_index must be a non-negative integer, not bool")
        if self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")

        for name in ("observation", "next_observation"):
            if not isinstance(getattr(self, name), RLObservationRefV1):
                raise ValueError(f"{name} must be an RLObservationRefV1")
        if not isinstance(self.reward, RewardSchemaV1):
            raise ValueError("reward must be a RewardSchemaV1")

        if self.observation.instruction_id != self.next_observation.instruction_id:
            raise ValueError("observation and next_observation must share one instruction_id")
        if self.observation.observation_id == self.next_observation.observation_id:
            raise ValueError("observation and next_observation must have distinct observation_ids")
        if self.next_observation.sync_timestamp_s <= self.observation.sync_timestamp_s:
            raise ValueError(
                "next_observation.sync_timestamp_s must be later than observation.sync_timestamp_s"
            )
        if self.reward.observation_id != self.next_observation.observation_id:
            raise ValueError("reward must annotate next_observation: observation_id mismatch")
        if self.reward.scene_id != self.next_observation.scene_id:
            raise ValueError("reward must annotate next_observation: scene_id mismatch")
        if self.reward.instruction_id != self.next_observation.instruction_id:
            raise ValueError("reward must annotate next_observation: instruction_id mismatch")

        action = validate_action_vector(self.action)
        object.__setattr__(self, "action", action)

        for name in ("done", "truncated"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if self.done and self.truncated:
            raise ValueError("done and truncated must not both be true")
        if self.reward.task_success == 1 and not self.done:
            raise ValueError("task_success=1 is a terminal reward and requires done=True")

        control_source = _require_string("control_source", self.control_source)
        if control_source not in TRANSITION_CONTROL_SOURCES:
            raise ValueError(
                f"control_source must be one of {sorted(TRANSITION_CONTROL_SOURCES)}"
            )
        gate = self.action_gate_passed
        if not (gate is None or isinstance(gate, bool)):
            raise ValueError("action_gate_passed must be bool or None")
        if control_source == "policy":
            if gate is not True:
                raise ValueError("policy transitions require action_gate_passed=True")
            if self.reward.intervention:
                raise ValueError("policy transitions must not claim a human intervention reward")
        elif gate is False:
            raise ValueError(
                "a gate-failed candidate action must never be recorded as an executed transition"
            )
        if control_source == "intervention" and not self.reward.intervention:
            raise ValueError("intervention transitions require reward.intervention=True")
        if control_source == "demonstration" and self.reward.intervention:
            raise ValueError("demonstration transitions must not claim intervention")

        if not isinstance(self.action_execution_confirmed, bool):
            raise ValueError("action_execution_confirmed must be bool")
        if not self.action_execution_confirmed:
            raise ValueError("action_execution_confirmed=True is required for every control source")

    def to_manifest_record(self) -> dict[str, object]:
        """Return a JSON-serializable audit record with no image payloads."""

        record = {
            "schema_version": self.schema_version,
            "transition_id": self.transition_id,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "observation": self.observation.to_manifest_record(),
            "action": list(self.action),
            "next_observation": self.next_observation.to_manifest_record(),
            "reward": self.reward.to_manifest_record(),
            "done": self.done,
            "truncated": self.truncated,
            "control_source": self.control_source,
            "action_gate_passed": self.action_gate_passed,
            "action_execution_confirmed": self.action_execution_confirmed,
            "execution_evidence_ref": self.execution_evidence_ref,
            "serialized_record_grants_live_authorization": False,
        }
        return copy.deepcopy(record)


@dataclass(frozen=True)
class RLReadinessReportV1:
    """Offline readiness audit for one RL mode, built only from explicit evidence.

    No field probes hardware. ``blockers`` lists every missing gate in
    canonical order, so reports are diff-stable. Regardless of mode, a true
    ``ready`` never produces real-hardware authorization fields: the manifest
    always pins ``policy_execution_authorized=false`` and
    ``hardware_access_performed=false``.
    """

    schema_version: str
    mode: str
    data_split_frozen: bool
    reward_schema_frozen: bool
    independent_reward_test_set: bool
    bc_baseline_available: bool
    aubo_env_adapter: bool
    explicit_reset: bool
    intervention_recording: bool
    replay_buffer: bool
    actor_learner: bool
    thin_safety_gate_integration: bool
    sim_or_null_actuator_passed: bool
    chunk_critic: bool
    behavior_or_kl_constraint: bool
    offline_policy_evaluation: bool
    language_counterfactual_tests: bool
    safety_gate_isolation: bool
    checkpoint_provenance: bool
    reward_hacking_checks: bool

    def __post_init__(self) -> None:
        if self.schema_version != RL_READINESS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported readiness schema {self.schema_version!r}; "
                f"expected {RL_READINESS_SCHEMA_VERSION!r}"
            )
        mode = _require_string("mode", self.mode)
        if mode not in RL_READINESS_MODES:
            raise ValueError(f"mode must be one of {sorted(RL_READINESS_MODES)}")
        all_gates = {gate for gates in MODE_REQUIRED_GATES.values() for gate in gates}
        for gate in sorted(all_gates):
            if not isinstance(getattr(self, gate), bool):
                raise ValueError(f"{gate} must be an explicit bool")

    @property
    def blockers(self) -> tuple[str, ...]:
        """Return every missing gate in stable canonical order."""

        return tuple(
            gate for gate in MODE_REQUIRED_GATES[self.mode] if not getattr(self, gate)
        )

    @property
    def ready(self) -> bool:
        """Return whether all gates for this mode are evidenced true."""

        return not self.blockers

    def to_manifest_record(self) -> dict[str, object]:
        """Return a JSON-serializable report with fixed no-hardware fields."""

        record = copy.deepcopy(asdict(self))
        record["ready"] = self.ready
        record["blockers"] = list(self.blockers)
        record["policy_execution_authorized"] = False
        record["hardware_access_performed"] = False
        record["serialized_record_grants_live_authorization"] = False
        return record

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

"""Phase B offline VLA runtime: predict, gate, and refuse actuators.

The injected ``policy_forward`` may be a dummy chunk generator in tests or a
SmolVLA callable later. This runtime never sets ``policy_execution_authorized``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Final

from .authorization_contract import EXECUTION_AUTHORIZATION_SCHEMA_VERSION, ExecutionAuthorizationV1
from .contracts import ACTION_SCHEMA_VERSION, INSTRUCTION_SCHEMA_VERSION, INSTRUCTION_SPECS
from .lerobot_bridge import observation_to_lerobot_frame
from .policy_prediction import POLICY_PREDICTION_SCHEMA_VERSION, PolicyPredictionV1
from .thin_safety_gate import GatedActionChunk, ThinSafetyGate

PHASE_B_POLICY_ID: Final = "smolvla-camerasetv2-offline-phase-b"
PolicyForward = Callable[[Mapping[str, object]], Sequence[Sequence[object]]]


class OfflineVLARuntime:
    """Run CameraSetV2 observations through a policy and the thin safety gate."""

    def __init__(
        self,
        policy_forward: PolicyForward,
        *,
        safety_gate: ThinSafetyGate | None = None,
        policy_id: str = PHASE_B_POLICY_ID,
        session_id: str = "phase-b-offline",
        checkpoint_ref: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> None:
        self.policy_forward = policy_forward
        self.safety_gate = safety_gate or ThinSafetyGate()
        self.policy_id = policy_id
        self.session_id = session_id
        self.checkpoint_ref = checkpoint_ref
        self.checkpoint_sha256 = checkpoint_sha256
        self.policy_execution_authorized = False

    def _authorization(
        self,
        *,
        gated: GatedActionChunk,
        observation_id: str,
        action_chunk_id: str,
    ) -> ExecutionAuthorizationV1:
        return ExecutionAuthorizationV1(
            schema_version=EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
            session_id=self.session_id,
            phase_id="phase-b-offline",
            control_source="policy",
            teleop_session_authorized=False,
            policy_execution_authorized=self.policy_execution_authorized,
            action_gate_passed=gated.passed,
            observation_id=observation_id,
            action_chunk_id=action_chunk_id,
            policy_id=self.policy_id,
        )

    def predict_and_gate(
        self,
        payload: Mapping[str, object],
        *,
        now_monotonic_s: float,
        observation_sync_timestamp_s: float,
        chunk_created_monotonic_s: float,
        observation_id: str,
        action_chunk_id: str,
        previous_tcp_m: Sequence[float] | None = None,
        previous_j6_rad: float | None = None,
        ik_checker: Callable[[tuple[float, ...]], bool] | None = None,
    ) -> tuple[dict[str, object], GatedActionChunk, ExecutionAuthorizationV1]:
        frame = observation_to_lerobot_frame(payload)
        predicted = self.policy_forward(frame)
        gated = self.safety_gate.evaluate(
            predicted,
            now_monotonic_s=now_monotonic_s,
            observation_sync_timestamp_s=observation_sync_timestamp_s,
            chunk_created_monotonic_s=chunk_created_monotonic_s,
            previous_tcp_m=previous_tcp_m,
            previous_j6_rad=previous_j6_rad,
            ik_checker=ik_checker,
        )
        authorization = self._authorization(
            gated=gated,
            observation_id=observation_id,
            action_chunk_id=action_chunk_id,
        )
        return frame, gated, authorization

    def predict_and_gate_with_audit(
        self,
        payload: Mapping[str, object],
        *,
        now_monotonic_s: float,
        observation_sync_timestamp_s: float,
        chunk_created_monotonic_s: float,
        observation_id: str,
        action_chunk_id: str,
        previous_tcp_m: Sequence[float] | None = None,
        previous_j6_rad: float | None = None,
        ik_checker: Callable[[tuple[float, ...]], bool] | None = None,
        confidence: float | None = None,
    ) -> tuple[
        dict[str, object],
        PolicyPredictionV1,
        GatedActionChunk,
        ExecutionAuthorizationV1,
    ]:
        """Predict and gate while preserving checkpoint and input provenance."""

        if self.checkpoint_ref is None or self.checkpoint_sha256 is None:
            raise ValueError(
                "Audited prediction requires checkpoint_ref and checkpoint_sha256"
            )
        frame = observation_to_lerobot_frame(payload)
        predicted = self.policy_forward(frame)
        instruction = INSTRUCTION_SPECS[str(frame["instruction_id"])]
        prediction = PolicyPredictionV1(
            schema_version=POLICY_PREDICTION_SCHEMA_VERSION,
            prediction_id=action_chunk_id,
            policy_id=self.policy_id,
            checkpoint_ref=self.checkpoint_ref,
            checkpoint_sha256=self.checkpoint_sha256,
            action_schema_version=ACTION_SCHEMA_VERSION,
            observation_id=observation_id,
            instruction_schema_version=INSTRUCTION_SCHEMA_VERSION,
            instruction_id=instruction.instruction_id,
            instruction_text_sha256=instruction.text_sha256,
            chunk_created_monotonic_s=chunk_created_monotonic_s,
            actions=tuple(tuple(action) for action in predicted),
            confidence=confidence,
        )
        gated = self.safety_gate.evaluate(
            prediction.actions,
            now_monotonic_s=now_monotonic_s,
            observation_sync_timestamp_s=observation_sync_timestamp_s,
            chunk_created_monotonic_s=prediction.chunk_created_monotonic_s,
            previous_tcp_m=previous_tcp_m,
            previous_j6_rad=previous_j6_rad,
            ik_checker=ik_checker,
        )
        authorization = self._authorization(
            gated=gated,
            observation_id=prediction.observation_id,
            action_chunk_id=prediction.prediction_id,
        )
        return frame, prediction, gated, authorization

    def dispatch_to_actuator(
        self, gated: GatedActionChunk, authorization: ExecutionAuthorizationV1
    ) -> None:
        raise PermissionError(
            "Phase B keeps policy_execution_authorized=false and does not connect "
            f"gated chunks to actuators (gate_passed={gated.passed}, "
            f"policy_execution_authorized={authorization.policy_execution_authorized})"
        )

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

from __future__ import annotations

import math

from lerobot.bamboo_sorting.thin_safety_gate import ThinSafetyGate, ThinSafetyGateLimits


def _action(*, x: float = 0.0, y: float = -0.5, z: float = 0.2, j6: float = 0.0, gripper: float = 0.0):
    return (j6, x, y, z, 0.0, 0.0, 0.0, gripper)


def _evaluate(actions, **changes):
    values = {
        "now_monotonic_s": 10.0,
        "observation_sync_timestamp_s": 9.95,
        "chunk_created_monotonic_s": 9.99,
        "previous_tcp_m": (0.0, -0.5, 0.2),
        "previous_j6_rad": 0.0,
    }
    values.update(changes)
    return ThinSafetyGate().evaluate(actions, **values)


def test_in_workspace_chunk_passes_without_rewriting() -> None:
    actions = [_action(), _action(x=0.02)]
    gated = _evaluate(actions)

    assert gated.passed
    assert gated.reasons == ()
    assert gated.actions == tuple(tuple(action) for action in actions)


def test_workspace_violation_rejects_and_keeps_original_action() -> None:
    actions = [_action(x=1.5)]
    gated = _evaluate(actions, previous_tcp_m=(1.48, -0.5, 0.2))

    assert not gated.passed
    assert "workspace:0" in gated.reasons
    assert gated.actions[0][1] == 1.5


def test_step_and_speed_reject_without_clipping() -> None:
    actions = [_action(x=0.4)]
    gated = _evaluate(actions)

    assert not gated.passed
    assert "step:0" in gated.reasons
    assert "speed:0" in gated.reasons
    assert gated.actions[0][1] == 0.4


def test_expired_observation_and_chunk_are_rejected() -> None:
    gated = _evaluate(
        [_action()],
        observation_sync_timestamp_s=9.0,
        chunk_created_monotonic_s=8.0,
    )

    assert not gated.passed
    assert "expired_observation" in gated.reasons
    assert "expired_action_chunk" in gated.reasons


def test_non_finite_and_illegal_gripper_are_schema_failures() -> None:
    gated = _evaluate([(0.0, math.nan, -0.5, 0.2, 0.0, 0.0, 0.0, 50.0)])

    assert not gated.passed
    assert "invalid_action_schema:0" in gated.reasons


def test_ik_failure_rejects_and_does_not_replan() -> None:
    gate = ThinSafetyGate(ThinSafetyGateLimits(require_ik=True))
    gated = gate.evaluate(
        [_action()],
        now_monotonic_s=10.0,
        observation_sync_timestamp_s=9.95,
        chunk_created_monotonic_s=9.99,
        previous_tcp_m=(0.0, -0.5, 0.2),
        previous_j6_rad=0.0,
        ik_checker=lambda _action: False,
    )

    assert not gated.passed
    assert gated.reasons == ("ik:0",)
    assert gated.actions[0] == _action()

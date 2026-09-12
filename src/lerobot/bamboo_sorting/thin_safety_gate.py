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

"""Reject-only thin safety gate for Phase B offline VLA.

This module never rewrites actions. Workspace, step, speed, age, and IK
failures only produce a rejected ``GatedActionChunk``. Passing the gate is
not execution authorization.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from .contracts import ACTION_SCHEMA_VERSION, validate_action_vector

THIN_SAFETY_GATE_SCHEMA_VERSION: Final = "ThinSafetyGateV1"
DEFAULT_WORKSPACE_MIN_M: Final = (-0.8, -1.2, 0.0)
DEFAULT_WORKSPACE_MAX_M: Final = (0.8, 0.0, 0.8)
DEFAULT_MAX_EE_STEP_M: Final = 0.05
DEFAULT_MAX_EE_SPEED_MPS: Final = 0.5
DEFAULT_MAX_J6_STEP_RAD: Final = 0.4
DEFAULT_MAX_OBSERVATION_AGE_MS: Final = 100.0
DEFAULT_MAX_CHUNK_AGE_MS: Final = 1000.0
DEFAULT_CONTROL_FPS: Final = 10.0
_TCP_SLICE: Final = slice(1, 4)
_J6_INDEX: Final = 0


@dataclass(frozen=True)
class ThinSafetyGateLimits:
    """Numeric reject thresholds. These are not clip bounds."""

    workspace_min_m: tuple[float, float, float] = DEFAULT_WORKSPACE_MIN_M
    workspace_max_m: tuple[float, float, float] = DEFAULT_WORKSPACE_MAX_M
    max_ee_step_m: float = DEFAULT_MAX_EE_STEP_M
    max_ee_speed_mps: float = DEFAULT_MAX_EE_SPEED_MPS
    max_j6_step_rad: float = DEFAULT_MAX_J6_STEP_RAD
    max_observation_age_ms: float = DEFAULT_MAX_OBSERVATION_AGE_MS
    max_chunk_age_ms: float = DEFAULT_MAX_CHUNK_AGE_MS
    control_fps: float = DEFAULT_CONTROL_FPS
    require_previous_tcp: bool = True
    require_ik: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_ee_step_m",
            "max_ee_speed_mps",
            "max_j6_step_rad",
            "max_observation_age_ms",
            "max_chunk_age_ms",
            "control_fps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("workspace_min_m", "workspace_max_m"):
            values = getattr(self, name)
            if len(values) != 3 or any(isinstance(item, bool) or not math.isfinite(item) for item in values):
                raise ValueError(f"{name} must contain three finite coordinates")
        if any(lo > hi for lo, hi in zip(self.workspace_min_m, self.workspace_max_m, strict=True)):
            raise ValueError("workspace_min_m must be componentwise <= workspace_max_m")
        for name in ("require_previous_tcp", "require_ik"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")


@dataclass(frozen=True)
class GatedActionChunk:
    """Original action chunk plus a pass/fail audit. Actions are unmodified."""

    schema_version: str
    action_schema_version: str
    actions: tuple[tuple[object, ...], ...]
    passed: bool
    reasons: tuple[str, ...]
    observation_age_ms: float | None
    chunk_age_ms: float | None

    def __post_init__(self) -> None:
        if self.schema_version != THIN_SAFETY_GATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported thin safety schema {self.schema_version!r}; "
                f"expected {THIN_SAFETY_GATE_SCHEMA_VERSION!r}"
            )
        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported action schema {self.action_schema_version!r}; "
                f"expected {ACTION_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be bool")
        if self.passed and self.reasons:
            raise ValueError("A passed gate cannot carry rejection reasons")
        if not self.passed and not self.reasons:
            raise ValueError("A rejected gate must name at least one reason")


class ThinSafetyGate:
    """Evaluate an absolute end-effector action chunk without rewriting it."""

    def __init__(self, limits: ThinSafetyGateLimits | None = None) -> None:
        self.limits = limits or ThinSafetyGateLimits()

    def evaluate(
        self,
        actions: Sequence[Sequence[object]],
        *,
        now_monotonic_s: float,
        observation_sync_timestamp_s: float | None,
        chunk_created_monotonic_s: float,
        previous_tcp_m: Sequence[float] | None = None,
        previous_j6_rad: float | None = None,
        ik_checker: Callable[[tuple[float, ...]], bool] | None = None,
    ) -> GatedActionChunk:
        unmodified = tuple(tuple(action) for action in actions)
        reasons: list[str] = []
        observation_age_ms = _age_ms(
            now_monotonic_s, observation_sync_timestamp_s, "observation_sync_timestamp_s", reasons
        )
        chunk_age_ms = _age_ms(
            now_monotonic_s, chunk_created_monotonic_s, "chunk_created_monotonic_s", reasons
        )
        if (
            observation_age_ms is not None
            and observation_age_ms > self.limits.max_observation_age_ms
        ):
            reasons.append("expired_observation")
        if chunk_age_ms is not None and chunk_age_ms > self.limits.max_chunk_age_ms:
            reasons.append("expired_action_chunk")

        parsed: list[tuple[float, ...]] = []
        if not unmodified:
            reasons.append("empty_action_chunk")
        for index, action in enumerate(unmodified):
            try:
                parsed.append(validate_action_vector(action))
            except ValueError:
                reasons.append(f"invalid_action_schema:{index}")

        dt_s = 1.0 / self.limits.control_fps
        if parsed:
            for index, action in enumerate(parsed):
                tcp = action[_TCP_SLICE]
                if any(
                    value < lo or value > hi
                    for value, lo, hi in zip(tcp, self.limits.workspace_min_m, self.limits.workspace_max_m, strict=True)
                ):
                    reasons.append(f"workspace:{index}")

            previous_tcp = _optional_xyz(previous_tcp_m, "previous_tcp_m", reasons)
            if previous_tcp is None and self.limits.require_previous_tcp:
                reasons.append("missing_previous_tcp")
            previous_j6 = previous_j6_rad
            if previous_j6 is not None and (
                isinstance(previous_j6, bool) or not math.isfinite(previous_j6)
            ):
                reasons.append("invalid_previous_j6")
                previous_j6 = None

            anchors_tcp: list[tuple[float, float, float]] = []
            anchors_j6: list[float] = []
            if previous_tcp is not None:
                anchors_tcp.append(previous_tcp)
            if previous_j6 is not None:
                anchors_j6.append(float(previous_j6))
            anchors_tcp.extend(action[_TCP_SLICE] for action in parsed)
            anchors_j6.extend(action[_J6_INDEX] for action in parsed)
            tcp_offset = 0 if previous_tcp is None else 1
            j6_offset = 0 if previous_j6 is None else 1
            for index in range(tcp_offset, len(anchors_tcp)):
                step_m = _distance(anchors_tcp[index - 1], anchors_tcp[index])
                if step_m > self.limits.max_ee_step_m:
                    reasons.append(f"step:{index - tcp_offset}")
                if step_m / dt_s > self.limits.max_ee_speed_mps:
                    reasons.append(f"speed:{index - tcp_offset}")
            for index in range(j6_offset, len(anchors_j6)):
                if abs(anchors_j6[index] - anchors_j6[index - 1]) > self.limits.max_j6_step_rad:
                    reasons.append(f"j6_step:{index - j6_offset}")

            if self.limits.require_ik and ik_checker is None:
                reasons.append("missing_ik_checker")
            elif ik_checker is not None:
                for index, action in enumerate(parsed):
                    try:
                        solvable = ik_checker(action)
                    except Exception:
                        reasons.append(f"ik:{index}")
                        continue
                    if solvable is not True:
                        reasons.append(f"ik:{index}")

        unique_reasons = tuple(dict.fromkeys(reasons))
        return GatedActionChunk(
            schema_version=THIN_SAFETY_GATE_SCHEMA_VERSION,
            action_schema_version=ACTION_SCHEMA_VERSION,
            actions=unmodified,
            passed=not unique_reasons,
            reasons=unique_reasons,
            observation_age_ms=observation_age_ms,
            chunk_age_ms=chunk_age_ms,
        )


def _age_ms(
    now_monotonic_s: float,
    timestamp_s: float | None,
    name: str,
    reasons: list[str],
) -> float | None:
    if isinstance(now_monotonic_s, bool) or not math.isfinite(now_monotonic_s):
        reasons.append("invalid_now")
        return None
    if timestamp_s is None:
        reasons.append(f"missing_{name}")
        return None
    if isinstance(timestamp_s, bool) or not math.isfinite(timestamp_s):
        reasons.append(f"invalid_{name}")
        return None
    age_s = now_monotonic_s - timestamp_s
    if age_s < -1e-9:
        reasons.append(f"future_{name}")
        return age_s * 1000.0
    return age_s * 1000.0


def _optional_xyz(
    values: Sequence[float] | None, name: str, reasons: list[str]
) -> tuple[float, float, float] | None:
    if values is None:
        return None
    if len(values) != 3 or any(isinstance(item, bool) or not math.isfinite(item) for item in values):
        reasons.append(f"invalid_{name}")
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))

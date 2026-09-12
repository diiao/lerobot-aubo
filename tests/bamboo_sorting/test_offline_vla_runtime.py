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

import numpy as np
import pytest

from lerobot.bamboo_sorting.contracts import INSTRUCTION_LANGUAGE, INSTRUCTION_SCHEMA_VERSION, INSTRUCTION_SPECS
from lerobot.bamboo_sorting.offline_vla_runtime import OfflineVLARuntime


def _payload() -> dict[str, object]:
    instruction = INSTRUCTION_SPECS["pick_any_collection"]
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "instruction_schema_version": INSTRUCTION_SCHEMA_VERSION,
        "instruction_id": instruction.instruction_id,
        "instruction_text": instruction.text,
        "instruction_language": INSTRUCTION_LANGUAGE,
        "global_rgb": image,
        "grasp_rgb": image.copy(),
        "observation.state": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0),
        "action": (0.0, 0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0),
    }


def _safe_chunk(_frame):
    return [(0.0, 0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0), (0.0, 0.02, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0)]


def test_offline_runtime_gates_language_observation_and_refuses_actuators() -> None:
    runtime = OfflineVLARuntime(_safe_chunk)
    frame, gated, authorization = runtime.predict_and_gate(
        _payload(),
        now_monotonic_s=10.0,
        observation_sync_timestamp_s=9.95,
        chunk_created_monotonic_s=9.99,
        observation_id="observation-0001",
        action_chunk_id="action-chunk-0001",
        previous_tcp_m=(0.0, -0.5, 0.2),
        previous_j6_rad=0.0,
    )

    assert frame["task"] == "Pick one strip and place it in the collection area."
    assert gated.passed
    assert authorization.action_gate_passed is True
    assert authorization.policy_execution_authorized is False
    assert runtime.policy_execution_authorized is False
    with pytest.raises(PermissionError, match="does not connect"):
        runtime.dispatch_to_actuator(gated, authorization)

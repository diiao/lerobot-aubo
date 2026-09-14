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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.bamboo_sorting.contracts import (
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_SCHEMA_VERSION,
    INSTRUCTION_SPECS,
)
from lerobot.bamboo_sorting.lerobot_bridge import observation_to_lerobot_frame
from lerobot.bamboo_sorting.offline_vla_runtime import OfflineVLARuntime
from lerobot.bamboo_sorting.smolvla_adapter import (
    SmolVLAOfflineForwardAdapter,
    prepare_smolvla_inference_frame,
    validate_smolvla_policy_contract,
)
from lerobot.configs.types import FeatureType, PolicyFeature


def _payload(*, height: int, width: int) -> dict[str, object]:
    instruction = INSTRUCTION_SPECS["pick_any_collection"]
    image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    return {
        "instruction_schema_version": INSTRUCTION_SCHEMA_VERSION,
        "instruction_id": instruction.instruction_id,
        "instruction_text": instruction.text,
        "instruction_language": INSTRUCTION_LANGUAGE,
        "global_rgb": image,
        "grasp_rgb": image.copy(),
        "observation.state": np.asarray(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0),
            dtype=np.float32,
        ),
        "action": (0.0, 0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0),
    }


def test_prepare_smolvla_input_maps_rgb_language_and_state_without_action() -> None:
    payload = _payload(height=4, width=5)
    frame = observation_to_lerobot_frame(payload, height=4, width=5)

    prepared = prepare_smolvla_inference_frame(frame, height=4, width=5)

    assert set(prepared) == {
        "observation.images.global_rgb",
        "observation.images.grasp_rgb",
        "observation.state",
        "task",
    }
    assert prepared["task"] == "Pick one strip and place it in the collection area."
    assert isinstance(prepared["observation.state"], torch.Tensor)
    assert prepared["observation.state"].shape == (13,)
    expected = torch.from_numpy(payload["global_rgb"].transpose(2, 0, 1)).float() / 255.0
    torch.testing.assert_close(prepared["observation.images.global_rgb"], expected)
    assert prepared["observation.images.global_rgb"].shape == (3, 4, 5)
    assert "action" not in prepared


class _Policy:
    def __init__(self, chunk: torch.Tensor) -> None:
        self.chunk = chunk
        self.seen: dict[str, object] | None = None
        self.config = _config()

    def predict_action_chunk(self, batch):
        self.seen = dict(batch)
        return self.chunk


def _identity(value):
    return value


def _config(**input_changes: PolicyFeature) -> SimpleNamespace:
    input_features = {
        "observation.images.global_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 480, 640),
        ),
        "observation.images.grasp_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 480, 640),
        ),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(13,)),
    }
    input_features.update(input_changes)
    return SimpleNamespace(
        input_features=input_features,
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        },
    )


def test_adapter_runs_policy_stack_and_offline_runtime_stays_unauthorized() -> None:
    chunk = torch.tensor(
        [
            [
                [0.0, 0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.02, -0.5, 0.2, 0.0, 0.0, 0.0, 100.0],
            ]
        ],
        dtype=torch.float32,
    )
    policy = _Policy(chunk)
    adapter = SmolVLAOfflineForwardAdapter(policy, _identity, _identity)
    runtime = OfflineVLARuntime(adapter)

    frame, gated, authorization = runtime.predict_and_gate(
        _payload(height=480, width=640),
        now_monotonic_s=10.0,
        observation_sync_timestamp_s=9.95,
        chunk_created_monotonic_s=9.99,
        observation_id="observation-0001",
        action_chunk_id="action-chunk-0001",
        previous_tcp_m=(0.0, -0.5, 0.2),
        previous_j6_rad=0.0,
    )

    assert frame["task"] == "Pick one strip and place it in the collection area."
    assert policy.seen is not None
    assert "action" not in policy.seen
    assert policy.seen["observation.images.global_rgb"].shape == (3, 480, 640)
    assert gated.passed is True
    assert authorization.policy_execution_authorized is False


@pytest.mark.parametrize(
    ("chunk", "message"),
    [
        (torch.zeros(2, 8), "shape"),
        (torch.zeros(1, 0, 8), "non-empty horizon"),
        (torch.zeros(1, 2, 7), "8 actions"),
        (torch.full((1, 2, 8), float("nan")), "NaN or Inf"),
        (np.zeros((1, 2, 8), dtype=np.bool_), "numeric"),
    ],
)
def test_adapter_rejects_invalid_policy_chunks(chunk: object, message: str) -> None:
    policy = _Policy(chunk)  # type: ignore[arg-type]
    frame = observation_to_lerobot_frame(_payload(height=480, width=640))
    adapter = SmolVLAOfflineForwardAdapter(policy, _identity, _identity)

    with pytest.raises(ValueError, match=message):
        adapter(frame)


def test_prepare_rejects_non_bridge_input() -> None:
    with pytest.raises(ValueError, match="CameraSetV1 bridge"):
        prepare_smolvla_inference_frame({}, height=4, width=5)


def test_policy_contract_rejects_missing_frozen_camera() -> None:
    policy = _Policy(torch.zeros(1, 2, 8))
    del policy.config.input_features["observation.images.grasp_rgb"]

    with pytest.raises(ValueError, match="exactly match CameraSetV1"):
        validate_smolvla_policy_contract(policy)


def test_policy_contract_rejects_wrong_action_width() -> None:
    policy = _Policy(torch.zeros(1, 2, 8))
    policy.config.output_features["action"] = PolicyFeature(
        type=FeatureType.ACTION,
        shape=(7,),
    )

    with pytest.raises(ValueError, match=r"shape \(8,\)"):
        validate_smolvla_policy_contract(policy)

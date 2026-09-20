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

"""Offline-only CameraSetV2 adapter for a LeRobot SmolVLA policy stack.

The bridge produces dataset-style HWC uint8 images. SmolVLA inference expects
CHW float tensors before its saved preprocessor adds a batch dimension,
normalizes state, and tokenizes the canonical task text. This module performs
that representation boundary and has no camera, robot, IO, or network access.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Final, Protocol

import numpy as np
import torch

from .contracts import ACTION_FIELD_NAMES, INSTRUCTION_LANGUAGE, validate_instruction_fields
from .lerobot_bridge import (
    ACTION_FEATURE_KEY,
    CURRENT_CAMERASET_LEROBOT_BRIDGE_VERSION,
    IMAGE_FEATURE_KEYS,
    STATE_FEATURE_KEY,
    TASK_FEATURE_KEY,
)
from .observation_contract import OBSERVATION_STATE_FIELD_NAMES

DEFAULT_IMAGE_HEIGHT: Final = 480
DEFAULT_IMAGE_WIDTH: Final = 640


class ChunkPolicy(Protocol):
    """Minimal policy surface used by the offline adapter."""

    config: object

    def predict_action_chunk(self, batch: Mapping[str, object]) -> object: ...


def validate_smolvla_policy_contract(
    policy: ChunkPolicy,
    *,
    height: int = DEFAULT_IMAGE_HEIGHT,
    width: int = DEFAULT_IMAGE_WIDTH,
) -> None:
    """Require the checkpoint feature contract to match frozen CameraSetV2."""

    config = getattr(policy, "config", None)
    input_features = getattr(config, "input_features", None)
    output_features = getattr(config, "output_features", None)
    if not isinstance(input_features, Mapping) or not isinstance(output_features, Mapping):
        raise ValueError("SmolVLA policy config must expose input_features and output_features")

    expected_inputs = {*IMAGE_FEATURE_KEYS, STATE_FEATURE_KEY}
    if set(input_features) != expected_inputs:
        raise ValueError(
            f"SmolVLA input features must exactly match CameraSetV2: {sorted(expected_inputs)}"
        )
    if set(output_features) != {ACTION_FEATURE_KEY}:
        raise ValueError("SmolVLA output features must contain only the 8D action")

    for key in IMAGE_FEATURE_KEYS:
        if tuple(getattr(input_features[key], "shape", ())) != (3, height, width):
            raise ValueError(f"SmolVLA feature {key} must have shape {(3, height, width)}")
    expected_state_shape = (len(OBSERVATION_STATE_FIELD_NAMES),)
    if tuple(getattr(input_features[STATE_FEATURE_KEY], "shape", ())) != expected_state_shape:
        raise ValueError(f"SmolVLA state feature must have shape {expected_state_shape}")
    expected_action_shape = (len(ACTION_FIELD_NAMES),)
    if tuple(getattr(output_features[ACTION_FEATURE_KEY], "shape", ())) != expected_action_shape:
        raise ValueError(f"SmolVLA action feature must have shape {expected_action_shape}")


def prepare_smolvla_inference_frame(
    frame: Mapping[str, object],
    *,
    height: int = DEFAULT_IMAGE_HEIGHT,
    width: int = DEFAULT_IMAGE_WIDTH,
) -> dict[str, object]:
    """Convert one validated bridge frame to an unbatched SmolVLA input.

    The saved SmolVLA preprocessor remains responsible for batching,
    tokenization, device placement, and normalization. The demonstration
    action is deliberately omitted to prevent target leakage at inference.
    """

    if frame.get("bridge_version") != CURRENT_CAMERASET_LEROBOT_BRIDGE_VERSION:
        raise ValueError("SmolVLA input requires a CameraSetV2 bridge frame")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")

    instruction = validate_instruction_fields(
        schema_version=str(frame.get("instruction_schema_version", "")),
        instruction_id=str(frame.get("instruction_id", "")),
        instruction_text=str(frame.get(TASK_FEATURE_KEY, "")),
        instruction_language=INSTRUCTION_LANGUAGE,
    )

    state = frame.get(STATE_FEATURE_KEY)
    if not isinstance(state, np.ndarray):
        raise ValueError(f"{STATE_FEATURE_KEY} must be a numpy array")
    expected_state_shape = (len(OBSERVATION_STATE_FIELD_NAMES),)
    if state.dtype != np.float32 or state.shape != expected_state_shape:
        raise ValueError(
            f"{STATE_FEATURE_KEY} must be float32 with shape {expected_state_shape}"
        )
    if not np.isfinite(state).all():
        raise ValueError(f"{STATE_FEATURE_KEY} must contain only finite values")
    if float(state[-1]) not in (0.0, 100.0):
        raise ValueError("gripper_pos state must be the last commanded 0/100 value")

    result: dict[str, object] = {
        STATE_FEATURE_KEY: torch.from_numpy(np.ascontiguousarray(state)).clone(),
        TASK_FEATURE_KEY: instruction.text,
    }
    for key in IMAGE_FEATURE_KEYS:
        image = frame.get(key)
        if not isinstance(image, np.ndarray):
            raise ValueError(f"{key} must be a numpy array")
        expected_image_shape = (height, width, 3)
        if image.dtype != np.uint8 or image.shape != expected_image_shape:
            raise ValueError(f"{key} must be uint8 RGB with shape {expected_image_shape}")
        chw = np.ascontiguousarray(image.transpose(2, 0, 1))
        result[key] = torch.from_numpy(chw).to(dtype=torch.float32).div(255.0)

    if ACTION_FEATURE_KEY in result:
        raise AssertionError("SmolVLA inference input must never contain demonstration action")
    return result


def _validated_action_chunk(value: object) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise ValueError("SmolVLA action chunk must contain numeric values")
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(
            "SmolVLA action chunk must have shape (1, horizon, action_dim)"
        )
    if array.shape[1] < 1 or array.shape[2] != len(ACTION_FIELD_NAMES):
        raise ValueError(
            f"SmolVLA action chunk must have a non-empty horizon and {len(ACTION_FIELD_NAMES)} actions"
        )
    if not np.isfinite(array).all():
        raise ValueError("SmolVLA action chunk contains NaN or Inf")
    return tuple(tuple(float(item) for item in action) for action in array[0])


class SmolVLAOfflineForwardAdapter:
    """Bind a SmolVLA policy and its saved processors to OfflineVLARuntime."""

    def __init__(
        self,
        policy: ChunkPolicy,
        preprocessor: Callable[[dict[str, object]], Mapping[str, object]],
        postprocessor: Callable[[object], object],
        *,
        height: int = DEFAULT_IMAGE_HEIGHT,
        width: int = DEFAULT_IMAGE_WIDTH,
    ) -> None:
        if not callable(getattr(policy, "predict_action_chunk", None)):
            raise TypeError("policy must provide predict_action_chunk")
        if not callable(preprocessor) or not callable(postprocessor):
            raise TypeError("preprocessor and postprocessor must be callable")
        validate_smolvla_policy_contract(policy, height=height, width=width)
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.height = height
        self.width = width

    def __call__(self, frame: Mapping[str, object]) -> Sequence[Sequence[object]]:
        inputs = prepare_smolvla_inference_frame(
            frame,
            height=self.height,
            width=self.width,
        )
        processed = self.preprocessor(inputs)
        if not isinstance(processed, Mapping):
            raise TypeError("SmolVLA preprocessor must return a mapping")
        with torch.inference_mode():
            predicted = self.policy.predict_action_chunk(processed)
            postprocessed = self.postprocessor(predicted)
        return _validated_action_chunk(postprocessed)

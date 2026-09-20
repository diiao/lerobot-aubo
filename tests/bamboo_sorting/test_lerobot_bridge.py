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

from pathlib import Path

import numpy as np
import pytest

from lerobot.bamboo_sorting.contracts import INSTRUCTION_LANGUAGE, INSTRUCTION_SCHEMA_VERSION, INSTRUCTION_SPECS
from lerobot.bamboo_sorting.lerobot_bridge import (
    CAMERASET_V2_LEROBOT_BRIDGE_VERSION,
    build_camera_set_v2_lerobot_features,
    observation_to_lerobot_frame,
    sha256_file,
    write_scene_manifest,
)


def _payload(**changes: object) -> dict[str, object]:
    instruction = INSTRUCTION_SPECS["pick_any_collection"]
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    values: dict[str, object] = {
        "instruction_schema_version": INSTRUCTION_SCHEMA_VERSION,
        "instruction_id": instruction.instruction_id,
        "instruction_text": instruction.text,
        "instruction_language": INSTRUCTION_LANGUAGE,
        "global_rgb": image,
        "grasp_rgb": image.copy(),
        "observation.state": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0),
        "action": (0.0, 0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0),
    }
    values.update(changes)
    return values


def test_bridge_emits_two_rgb_language_and_action_contract() -> None:
    features = build_camera_set_v2_lerobot_features()
    frame = observation_to_lerobot_frame(_payload())

    assert set(features) >= {
        "observation.images.global_rgb",
        "observation.images.grasp_rgb",
        "observation.state",
        "action",
        "task",
    }
    assert "observation.images.wrist_rgb" not in features
    assert frame["task"] == "Pick one strip and place it in the collection area."
    assert frame["bridge_version"] == CAMERASET_V2_LEROBOT_BRIDGE_VERSION


@pytest.mark.parametrize("forbidden_key", ["handeye", "fixed", "wrist_rgb", "wrist_depth_m"])
def test_bridge_rejects_act_and_depth_keys(forbidden_key: str) -> None:
    payload = _payload()
    payload[forbidden_key] = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="rejects ACT/depth keys"):
        observation_to_lerobot_frame(payload)


def test_scene_manifest_is_immutable(tmp_path: Path) -> None:
    artifact = tmp_path / "frame.bin"
    artifact.write_bytes(b"vla-offline")
    digest = sha256_file(artifact)
    path = write_scene_manifest(tmp_path, {"frame.bin": digest})

    assert path.exists()
    with pytest.raises(FileExistsError):
        write_scene_manifest(tmp_path, {"frame.bin": digest})

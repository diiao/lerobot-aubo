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

"""Current CameraSetV2 mapping regression."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot.bamboo_sorting.c0_smoke_capture import (
    C0_SMOKE_CAMERA_DEVICES,
    C0_SMOKE_IMAGE_KEYS,
    C0_SMOKE_LEGACY_ACT_KEYS,
    frozen_c0_smoke_camera_mapping,
)
from lerobot.bamboo_sorting.rgb_gate import FROZEN_CAMERA_SET_V2_SHA256


def test_camera_keys_and_frozen_device_mapping() -> None:
    mapping = frozen_c0_smoke_camera_mapping()

    assert tuple(mapping) == ("global_rgb", "grasp_rgb")
    assert C0_SMOKE_IMAGE_KEYS == ("global_rgb", "grasp_rgb")
    assert mapping["global_rgb"]["device"] == C0_SMOKE_CAMERA_DEVICES["global_rgb"]
    assert mapping["grasp_rgb"]["device"] == C0_SMOKE_CAMERA_DEVICES["grasp_rgb"]
    assert mapping["global_rgb"]["legacy_act_key"] == "handeye"
    assert mapping["grasp_rgb"]["legacy_act_key"] == "fixed"
    assert C0_SMOKE_LEGACY_ACT_KEYS["grasp_rgb"] == "fixed"
    assert "handeye" not in mapping
    assert "fixed" not in mapping
    assert "wrist_rgb" not in mapping
    assert FROZEN_CAMERA_SET_V2_SHA256 == (
        "20de7adfd6ed9734c3a4329d86ab7e0ad08df9c6758d878d3c1302e2ac6423b4"
    )


@pytest.mark.parametrize("change", ["schema", "device"])
def test_camera_mapping_rejects_changed_configuration(tmp_path: Path, change: str) -> None:
    source = Path(__file__).resolve().parents[2] / "configs/aubo_i10/CameraSetV2.json"
    record = json.loads(source.read_text())
    if change == "schema":
        record["schema_version"] = "CameraSetV1"
    else:
        record["capture_profiles"]["global_rgb"]["device"] = "/dev/incorrect-camera"
    candidate = tmp_path / "CameraSetV2.json"
    candidate.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="schema_version|frozen SHA-256"):
        frozen_c0_smoke_camera_mapping(candidate)

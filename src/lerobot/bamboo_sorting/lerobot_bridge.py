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

"""Versioned CameraSetV1 to LeRobot frame bridge for Phase B.

This mapping is offline-only. It refuses ACT ``handeye``/``fixed`` keys, wrist
RGB, and depth fields so report ACT data cannot enter the VLA contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .contracts import (
    ACTION_FIELD_NAMES,
    ACTION_SCHEMA_VERSION,
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_SCHEMA_VERSION,
    validate_action_vector,
    validate_instruction_fields,
)
from .observation_contract import OBSERVATION_STATE_FIELD_NAMES, OBSERVATION_STATE_FIELD_SPECS
from .rgb_gate import FIXED_RGB_STREAMS

CAMERASET_V1_LEROBOT_BRIDGE_VERSION: Final = "CameraSetV1LeRobotBridgeV1"
FORBIDDEN_VLA_KEYS: Final = frozenset(
    {
        "handeye",
        "fixed",
        "wrist_rgb",
        "wrist_depth_m",
        "wrist_depth_valid",
        "wrist_xyz_m",
        "observation.images.handeye",
        "observation.images.fixed",
        "observation.images.wrist_rgb",
        "observation.images.wrist_depth_m",
    }
)
IMAGE_FEATURE_KEYS: Final = tuple(f"observation.images.{name}" for name in FIXED_RGB_STREAMS)
STATE_FEATURE_KEY: Final = "observation.state"
ACTION_FEATURE_KEY: Final = "action"
TASK_FEATURE_KEY: Final = "task"


def build_cameras_set_v1_lerobot_features(
    *, height: int = 480, width: int = 640
) -> dict[str, dict[str, object]]:
    """Return the frozen two-RGB LeRobot feature contract."""

    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    features: dict[str, dict[str, object]] = {
        key: {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
        for key in IMAGE_FEATURE_KEYS
    }
    features[STATE_FEATURE_KEY] = {
        "dtype": "float32",
        "shape": (len(OBSERVATION_STATE_FIELD_SPECS),),
        "names": list(OBSERVATION_STATE_FIELD_NAMES),
    }
    features[ACTION_FEATURE_KEY] = {
        "dtype": "float32",
        "shape": (len(ACTION_FIELD_NAMES),),
        "names": list(ACTION_FIELD_NAMES),
    }
    features[TASK_FEATURE_KEY] = {"dtype": "string", "shape": (1,)}
    return features


def _reject_forbidden_keys(payload: Mapping[str, object]) -> None:
    present = sorted(FORBIDDEN_VLA_KEYS.intersection(payload))
    if present:
        raise ValueError(f"VLA bridge rejects ACT/depth keys: {present}")


def _validate_rgb(name: str, image: object, *, height: int, width: int) -> NDArray[np.uint8]:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape != (height, width, 3):
        raise ValueError(f"{name} must be uint8 RGB with shape {(height, width, 3)}")
    return image


def _validate_state(state: Sequence[object]) -> tuple[float, ...]:
    if isinstance(state, (str, bytes)) or len(state) != len(OBSERVATION_STATE_FIELD_SPECS):
        raise ValueError(f"{STATE_FEATURE_KEY} requires {len(OBSERVATION_STATE_FIELD_SPECS)} scalars")
    values: list[float] = []
    for field, raw_value in zip(OBSERVATION_STATE_FIELD_SPECS, state, strict=True):
        if isinstance(raw_value, bool):
            raise ValueError(f"{field.name} must be numeric, not bool")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field.name} must be numeric") from exc
        if not np.isfinite(value):
            raise ValueError(f"{field.name} must be finite")
        values.append(value)
    if values[-1] not in (0.0, 100.0):
        raise ValueError("gripper_pos must be the last commanded 0/100 state")
    return tuple(values)


def observation_to_lerobot_frame(
    payload: Mapping[str, object],
    *,
    height: int = 480,
    width: int = 640,
) -> dict[str, object]:
    """Convert a CameraSetV1 sample into a LeRobot frame."""

    _reject_forbidden_keys(payload)
    instruction = validate_instruction_fields(
        schema_version=str(payload.get("instruction_schema_version", INSTRUCTION_SCHEMA_VERSION)),
        instruction_id=str(payload["instruction_id"]),
        instruction_text=str(payload["instruction_text"]),
        instruction_language=str(payload.get("instruction_language", INSTRUCTION_LANGUAGE)),
    )
    action = validate_action_vector(payload[ACTION_FEATURE_KEY])  # type: ignore[arg-type]
    frame: dict[str, object] = {
        "bridge_version": CAMERASET_V1_LEROBOT_BRIDGE_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "instruction_schema_version": INSTRUCTION_SCHEMA_VERSION,
        "instruction_id": instruction.instruction_id,
        TASK_FEATURE_KEY: instruction.text,
        STATE_FEATURE_KEY: np.asarray(_validate_state(payload[STATE_FEATURE_KEY]), dtype=np.float32),  # type: ignore[arg-type]
        ACTION_FEATURE_KEY: np.asarray(action, dtype=np.float32),
    }
    for stream, key in zip(FIXED_RGB_STREAMS, IMAGE_FEATURE_KEYS, strict=True):
        image = payload.get(key, payload.get(stream))
        frame[key] = _validate_rgb(key, image, height=height, width=width)
    return frame


def write_scene_manifest(scene_dir: Path, artifacts: Mapping[str, str]) -> Path:
    """Write a new SHA-256 scene manifest and refuse to overwrite it."""

    scene_dir = Path(scene_dir)
    if not scene_dir.is_dir():
        raise ValueError("scene_dir must exist and be a directory")
    manifest_path = scene_dir / "scene_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    records = []
    for relative_path, digest in artifacts.items():
        if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
            raise ValueError("artifact paths must be relative")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid SHA-256 for {relative_path}")
        records.append({"relative_path": relative_path, "sha256": digest})
    payload = {
        "schema_version": CAMERASET_V1_LEROBOT_BRIDGE_VERSION,
        "scene_id": scene_dir.name,
        "raw_data_immutable": True,
        "artifacts": records,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(encoded)
    return manifest_path


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

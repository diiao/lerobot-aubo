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

"""One-episode C0 smoke capture helpers.

Status: one-episode C0 smoke capture, pending formal final evidence binding.

This module is not the formal ten-episode C0 batch and never grants training
or policy-execution authorization. Importing it does not connect a camera,
AUBO, phone, or dataset, and does not construct ``C0EpisodeManifestV1``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pyarrow.parquet as pq

from .c0_capture_contract import C0_DATASET_FPS, C0_INSTRUCTION_ID, GRIPPER_STATE_SEMANTICS
from .contracts import ACTION_FIELD_NAMES, INSTRUCTION_SPECS
from .lerobot_bridge import FORBIDDEN_VLA_KEYS, IMAGE_FEATURE_KEYS
from .observation_contract import OBSERVATION_STATE_FIELD_NAMES
from .rgb_gate import (
    CAMERA_SET_V2_SCHEMA_VERSION,
    FROZEN_CAMERA_SET_V2_SHA256,
    GLOBAL_RGB,
    GRASP_RGB,
    load_camera_set_v2,
)

C0_SMOKE_STATUS: Final = (
    "one-episode C0 smoke capture, pending formal final evidence binding"
)
C0_SMOKE_NUM_EPISODES: Final = 1
C0_SMOKE_DATASET_FPS: Final = int(C0_DATASET_FPS)
C0_SMOKE_INSTRUCTION = INSTRUCTION_SPECS[C0_INSTRUCTION_ID]
C0_SMOKE_TASK_TEXT: Final = C0_SMOKE_INSTRUCTION.text
C0_SMOKE_IMAGE_KEYS: Final = (GLOBAL_RGB, GRASP_RGB)
C0_SMOKE_IMAGE_FEATURE_KEYS: Final = IMAGE_FEATURE_KEYS
C0_SMOKE_CAMERA_DEVICES: Final = {
    GLOBAL_RGB: (
        "/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH0319_20210712_v102-video-index0"
    ),
    GRASP_RGB: (
        "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
    ),
}
C0_SMOKE_CAMERA_MODELS: Final = {
    GLOBAL_RGB: "GENERAL WEBCAM",
    GRASP_RGB: "Sonix USB2.0_CAM1",
}
C0_SMOKE_LEGACY_ACT_KEYS: Final = {
    GLOBAL_RGB: "handeye",
    GRASP_RGB: "fixed",
}
C0_SMOKE_OPERATOR_CONFIRMATION: Final = "YES"
C0_SMOKE_ANGLE_DATUM: Final = (
    "physical collection-area scale mark; not robot base +X"
)

_FORBIDDEN_IMAGE_TOKENS: Final = ("handeye", "fixed", "wrist", "depth", "mech")


class C0SmokeCaptureError(ValueError):
    """Fail-closed smoke-capture or postflight error."""


def require_placement_reference(value: object) -> str:
    """Require an operator-entered physical scale label. Never default to 0°."""

    if not isinstance(value, str) or not value.strip():
        raise C0SmokeCaptureError(
            "placement reference is required and must be a non-empty operator "
            "entry of the physical scale mark or scale-line name; it is not "
            "inferred from robot base +X and must not be auto-recorded as 0°"
        )
    return value.strip()


def require_new_dataset_root(path: object) -> Path:
    """Return a resolved dataset root that does not already exist."""

    if path is None or (isinstance(path, str) and not path.strip()):
        raise C0SmokeCaptureError("dataset root must be specified explicitly")
    root = Path(path).expanduser().resolve()
    if root.exists():
        raise C0SmokeCaptureError(
            f"dataset root already exists; refuse to overwrite or resume: {root}"
        )
    return root


def frozen_c0_smoke_camera_mapping(
    camera_set_path: Path | None = None,
) -> dict[str, dict[str, object]]:
    """Return CameraSetV2 RGB device mapping for the smoke capture entry."""

    record, digest = load_camera_set_v2(camera_set_path)
    if digest != FROZEN_CAMERA_SET_V2_SHA256:
        raise C0SmokeCaptureError("CameraSetV2 SHA-256 does not match the freeze")
    if record.schema_version != CAMERA_SET_V2_SCHEMA_VERSION:
        raise C0SmokeCaptureError("camera set schema must be CameraSetV2")
    if tuple(record.camera_streams) != C0_SMOKE_IMAGE_KEYS:
        raise C0SmokeCaptureError(
            f"camera streams must be {C0_SMOKE_IMAGE_KEYS}, not {record.camera_streams}"
        )

    mapping: dict[str, dict[str, object]] = {}
    for key in C0_SMOKE_IMAGE_KEYS:
        profile = record.capture_profiles.get(key)
        if not isinstance(profile, Mapping):
            raise C0SmokeCaptureError(f"CameraSetV2 missing capture profile for {key}")
        device = profile.get("device")
        if device != C0_SMOKE_CAMERA_DEVICES[key]:
            raise C0SmokeCaptureError(
                f"{key} device must be {C0_SMOKE_CAMERA_DEVICES[key]!r}, got {device!r}"
            )
        fps = profile.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise C0SmokeCaptureError(f"{key} capture fps is missing")
        mapping[key] = {
            "device": device,
            "fps": float(fps),
            "width": int(profile["width"]),
            "height": int(profile["height"]),
            "fourcc": str(profile["fourcc"]),
            "model": C0_SMOKE_CAMERA_MODELS[key],
            "legacy_act_key": C0_SMOKE_LEGACY_ACT_KEYS[key],
            "physical_role": record.physical_roles[key],
        }
    return mapping


def c0_smoke_robot_observation_features() -> dict[str, type | tuple]:
    """13D robot state plus CameraSetV2 RGB keys for dataset feature assembly."""

    features: dict[str, type | tuple] = {
        name: float for name in OBSERVATION_STATE_FIELD_NAMES
    }
    for key in C0_SMOKE_IMAGE_KEYS:
        features[key] = (480, 640, 3)
    return features


def build_operator_checklist(
    *,
    placement_reference: str,
    dataset_root: Path,
    camera_mapping: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    """Return the console checklist that an on-site operator must confirm."""

    reference = require_placement_reference(placement_reference)
    return (
        C0_SMOKE_STATUS,
        "This is not the formal ten-episode C0 batch.",
        "Training, inference, and policy execution are not authorized.",
        f"Dataset FPS: {C0_SMOKE_DATASET_FPS}",
        f"Episodes planned: {C0_SMOKE_NUM_EPISODES}",
        f"Canonical task: {C0_SMOKE_TASK_TEXT}",
        f"Dataset root: {dataset_root}",
        f"CameraSetV2 SHA-256: {FROZEN_CAMERA_SET_V2_SHA256}",
        (
            f"{GLOBAL_RGB}: {C0_SMOKE_CAMERA_MODELS[GLOBAL_RGB]} "
            f"{camera_mapping[GLOBAL_RGB]['device']} "
            f"(external stationary; legacy ACT key {C0_SMOKE_LEGACY_ACT_KEYS[GLOBAL_RGB]})"
        ),
        (
            f"{GRASP_RGB}: {C0_SMOKE_CAMERA_MODELS[GRASP_RGB]} "
            f"{camera_mapping[GRASP_RGB]['device']} "
            f"(wrist / gripper mount; legacy ACT key {C0_SMOKE_LEGACY_ACT_KEYS[GRASP_RGB]})"
        ),
        "Mech-Eye RGB/depth are excluded from this capture.",
        f"Placement reference: {reference}",
        f"Angle datum: {C0_SMOKE_ANGLE_DATUM}",
        "Confirm: single strip only.",
        "Confirm: physical placement reference is the on-site scale mark or scale-line name.",
        "Confirm: strip long axis follows the named physical scale line.",
        "Confirm: collection area is empty.",
        "Confirm: the robot workspace is clear of people and obstacles.",
        "Confirm: operator is on site.",
        "Confirm: e-stop is available.",
        "Confirm: record exactly 1 episode.",
        "Confirm: new dataset root.",
        f"Type {C0_SMOKE_OPERATOR_CONFIRMATION} to proceed.",
    )


@dataclass(frozen=True)
class C0SmokeShutdownResult:
    dataset_finalize_attempted: bool
    dataset_finalize_succeeded: bool
    errors: tuple[BaseException, ...]


def _attempt_cleanup_step(name: str, fn: Callable[[], Any], errors: list[BaseException]) -> None:
    try:
        fn()
    except Exception as exc:
        logging.error("C0 smoke cleanup failed: %s", name, exc_info=True)
        errors.append(exc)


def _attempt_disconnect(name: str, device: Any, errors: list[BaseException]) -> None:
    if device is None:
        return
    try:
        connected = getattr(device, "is_connected", True)
        if callable(connected):
            connected = connected()
    except Exception as exc:
        logging.error("C0 smoke %s.is_connected failed", name, exc_info=True)
        errors.append(exc)
        connected = True
    if not connected:
        return
    disconnect = getattr(device, "disconnect", None)
    if not callable(disconnect):
        errors.append(C0SmokeCaptureError(f"{name}.disconnect is missing"))
        return
    _attempt_cleanup_step(f"{name}.disconnect", disconnect, errors)


def shutdown_c0_smoke_session(
    *,
    dataset: Any | None,
    listener: Any | None,
    phone: Any | None,
    robot: Any | None,
) -> C0SmokeShutdownResult:
    """Attempt each cleanup independently. Never raises.

    If ``dataset`` was created, ``dataset.finalize()`` is attempted exactly once
    even when listener/phone/robot cleanup fails.
    """

    errors: list[BaseException] = []
    if listener is not None:
        stop = getattr(listener, "stop", None)
        if callable(stop):
            _attempt_cleanup_step("listener.stop", stop, errors)
        else:
            errors.append(C0SmokeCaptureError("listener.stop is missing"))
    _attempt_disconnect("phone", phone, errors)
    _attempt_disconnect("robot", robot, errors)

    finalize_attempted = False
    finalize_succeeded = False
    if dataset is not None:
        finalize = getattr(dataset, "finalize", None)
        finalize_attempted = True
        if not callable(finalize):
            errors.append(C0SmokeCaptureError("dataset.finalize is missing"))
        else:
            try:
                finalize()
                finalize_succeeded = True
            except Exception as exc:
                logging.error("C0 smoke dataset.finalize failed", exc_info=True)
                errors.append(exc)

    return C0SmokeShutdownResult(
        dataset_finalize_attempted=finalize_attempted,
        dataset_finalize_succeeded=finalize_succeeded,
        errors=tuple(errors),
    )


def maybe_run_c0_smoke_postflight(
    *,
    shutdown_result: C0SmokeShutdownResult,
    dataset: Any | None,
    episode_idx: int,
    dataset_root: Path,
    postflight: Callable[[Path], dict[str, object]] | None = None,
) -> dict[str, object]:
    """Run postflight only after a successful finalize of exactly one episode."""

    if shutdown_result.errors:
        raise shutdown_result.errors[0]
    if (
        dataset is None
        or episode_idx != C0_SMOKE_NUM_EPISODES
        or not shutdown_result.dataset_finalize_succeeded
    ):
        raise C0SmokeCaptureError(
            "C0 smoke capture did not save exactly one finalized episode; postflight not claimed"
        )
    runner = run_c0_smoke_postflight if postflight is None else postflight
    return runner(dataset_root)


def require_operator_confirmation(typed: object) -> None:
    if not isinstance(typed, str) or typed.strip() != C0_SMOKE_OPERATOR_CONFIRMATION:
        raise C0SmokeCaptureError(
            f"operator confirmation must be {C0_SMOKE_OPERATOR_CONFIRMATION!r}"
        )


def _as_name_tuple(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    names = tuple(str(item) for item in value)
    return names


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise C0SmokeCaptureError(f"{path} must contain a JSON object")
    return payload


def _require_task_text(value: object, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise C0SmokeCaptureError(
            f"meta/tasks.parquet {source} must be a non-empty task string"
        )
    return value


def _task_texts(dataset_root: Path) -> tuple[str, ...]:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise C0SmokeCaptureError("meta/tasks.parquet is missing")
    table = pq.read_table(tasks_path)
    names = table.column_names
    if "task_index" not in names:
        raise C0SmokeCaptureError("meta/tasks.parquet lacks a task_index column")
    has_v3_text = "__index_level_0__" in names
    has_legacy_text = "task" in names
    if not has_v3_text and not has_legacy_text:
        raise C0SmokeCaptureError("meta/tasks.parquet lacks a task text source")
    if table.num_rows == 0:
        raise C0SmokeCaptureError("meta/tasks.parquet has no task text")

    task_indexes = table.column("task_index").to_pylist()
    v3_values = table.column("__index_level_0__").to_pylist() if has_v3_text else None
    legacy_values = table.column("task").to_pylist() if has_legacy_text else None
    texts: list[str] = []
    index_to_text: dict[int, str] = {}
    text_to_index: dict[str, int] = {}
    for row, task_index in enumerate(task_indexes):
        if isinstance(task_index, bool) or not isinstance(task_index, int):
            raise C0SmokeCaptureError(
                "meta/tasks.parquet task_index must be a Python integer"
            )
        if task_index < 0:
            raise C0SmokeCaptureError(
                "meta/tasks.parquet task_index must be greater than or equal to 0"
            )
        row_texts: list[str] = []
        if v3_values is not None:
            row_texts.append(_require_task_text(v3_values[row], "__index_level_0__"))
        if legacy_values is not None:
            row_texts.append(_require_task_text(legacy_values[row], "task"))
        if len(set(row_texts)) != 1:
            raise C0SmokeCaptureError(
                "meta/tasks.parquet task and __index_level_0__ disagree"
            )
        text = row_texts[0]
        if task_index in index_to_text and index_to_text[task_index] != text:
            raise C0SmokeCaptureError(
                "meta/tasks.parquet has contradictory task_index mapping"
            )
        if task_index in index_to_text:
            raise C0SmokeCaptureError(
                "meta/tasks.parquet has duplicate task_index mapping"
            )
        if text in text_to_index:
            raise C0SmokeCaptureError(
                "meta/tasks.parquet maps the same task text to multiple task_index values"
            )
        index_to_text[task_index] = text
        text_to_index[text] = task_index
        texts.append(text)
    return tuple(texts)


def _episode_frame_count(dataset_root: Path, info: Mapping[str, Any]) -> int:
    episodes_dir = dataset_root / "meta" / "episodes"
    episode_files = sorted(episodes_dir.rglob("*.parquet")) if episodes_dir.is_dir() else []
    if episode_files:
        table = pq.read_table(episode_files[0])
        if "length" in table.column_names:
            lengths = [int(value) for value in table.column("length").to_pylist()]
            if lengths:
                return lengths[0]
    total_frames = info.get("total_frames")
    if isinstance(total_frames, bool) or not isinstance(total_frames, int):
        raise C0SmokeCaptureError("episode frame_count is missing")
    return total_frames


def _video_path(dataset_root: Path, image_key: str) -> Path:
    default = dataset_root / "videos" / image_key / "chunk-000" / "file-000.mp4"
    if default.is_file():
        return default
    video_root = dataset_root / "videos" / image_key
    if not video_root.is_dir():
        raise C0SmokeCaptureError(f"video directory missing for {image_key}")
    matches = sorted(path for path in video_root.rglob("*.mp4") if path.is_file())
    if len(matches) != 1:
        raise C0SmokeCaptureError(f"expected one video file for {image_key}, found {len(matches)}")
    return matches[0]


def _read_video_metadata(path: Path) -> dict[str, object]:
    import av

    with av.open(str(path), "r") as container:
        if not container.streams.video:
            raise C0SmokeCaptureError(f"{path} has no video stream")
        stream = container.streams.video[0]
        return {
            "path": str(path),
            "width": int(stream.width or 0),
            "height": int(stream.height or 0),
            "codec": stream.codec.canonical_name if stream.codec else None,
        }


def run_c0_smoke_postflight(dataset_root: Path) -> dict[str, object]:
    """Read-only checks after the smoke dataset is closed.

    Does not claim physical gripper feedback or grasp success.
    Placement reference is only operator-entered at capture start; this
    function does not read or verify it from dataset metadata.
    """

    root = Path(dataset_root).expanduser().resolve()
    errors: list[str] = []
    if not root.is_dir():
        raise C0SmokeCaptureError(f"dataset directory does not exist: {root}")

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise C0SmokeCaptureError("meta/info.json is missing")
    info = _read_json_object(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise C0SmokeCaptureError("meta/info.json features must be an object")

    total_episodes = info.get("total_episodes")
    if total_episodes != C0_SMOKE_NUM_EPISODES:
        errors.append(f"total_episodes must be {C0_SMOKE_NUM_EPISODES}, got {total_episodes!r}")

    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) != C0_SMOKE_DATASET_FPS:
        errors.append(f"fps must be {C0_SMOKE_DATASET_FPS}, got {fps!r}")

    image_keys = tuple(
        key for key in features if isinstance(key, str) and key.startswith("observation.images.")
    )
    if tuple(image_keys) != C0_SMOKE_IMAGE_FEATURE_KEYS and set(image_keys) != set(
        C0_SMOKE_IMAGE_FEATURE_KEYS
    ):
        errors.append(
            "image fields must be exactly "
            f"{list(C0_SMOKE_IMAGE_FEATURE_KEYS)}, got {list(image_keys)}"
        )
    for key in image_keys:
        if key in FORBIDDEN_VLA_KEYS or any(token in key for token in _FORBIDDEN_IMAGE_TOKENS):
            errors.append(f"forbidden image field: {key}")
    for required in C0_SMOKE_IMAGE_FEATURE_KEYS:
        if required not in features:
            errors.append(f"missing image field: {required}")

    action = features.get("action")
    if not isinstance(action, dict):
        errors.append("action feature is missing")
        action_names: tuple[str, ...] = ()
    else:
        action_names = _as_name_tuple(action.get("names")) or ()
        if action_names != ACTION_FIELD_NAMES:
            errors.append(f"action names must be {list(ACTION_FIELD_NAMES)}, got {list(action_names)}")
        shape = action.get("shape")
        if list(shape or []) != [len(ACTION_FIELD_NAMES)]:
            errors.append(f"action shape must be [{len(ACTION_FIELD_NAMES)}], got {shape!r}")

    state = features.get("observation.state")
    if not isinstance(state, dict):
        errors.append("observation.state feature is missing")
        state_names: tuple[str, ...] = ()
    else:
        state_names = _as_name_tuple(state.get("names")) or ()
        if state_names != OBSERVATION_STATE_FIELD_NAMES:
            errors.append(
                "observation.state names must be "
                f"{list(OBSERVATION_STATE_FIELD_NAMES)}, got {list(state_names)}"
            )
        shape = state.get("shape")
        if list(shape or []) != [len(OBSERVATION_STATE_FIELD_NAMES)]:
            errors.append(
                "observation.state shape must be "
                f"[{len(OBSERVATION_STATE_FIELD_NAMES)}], got {shape!r}"
            )

    try:
        task_texts = _task_texts(root)
    except C0SmokeCaptureError as exc:
        errors.append(str(exc))
        task_texts = ()
    unique_tasks = tuple(dict.fromkeys(task_texts))
    if unique_tasks != (C0_SMOKE_TASK_TEXT,):
        errors.append(
            "task text must equal "
            f"{C0_SMOKE_TASK_TEXT!r}, got {list(unique_tasks)!r}"
        )

    try:
        frame_count = _episode_frame_count(root, info)
    except C0SmokeCaptureError as exc:
        errors.append(str(exc))
        frame_count = 0
    if frame_count <= 0:
        errors.append("episode frame_count must be > 0")

    videos: dict[str, object] = {}
    for image_key in C0_SMOKE_IMAGE_FEATURE_KEYS:
        try:
            video_path = _video_path(root, image_key)
            videos[image_key] = _read_video_metadata(video_path)
        except C0SmokeCaptureError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"cannot read video metadata for {image_key}: {exc}")

    if errors:
        raise C0SmokeCaptureError("; ".join(errors))

    return {
        "status": C0_SMOKE_STATUS,
        "dataset_root": str(root),
        "dataset_root_exists": True,
        "total_episodes": total_episodes,
        "fps": float(fps),
        "task": C0_SMOKE_TASK_TEXT,
        "image_feature_keys": list(C0_SMOKE_IMAGE_FEATURE_KEYS),
        "action_names": list(ACTION_FIELD_NAMES),
        "observation_state_names": list(OBSERVATION_STATE_FIELD_NAMES),
        "episode_frame_count": frame_count,
        "videos": videos,
        "gripper_pos_semantics": GRIPPER_STATE_SEMANTICS,
        "gripper_pos_is_physical_feedback": False,
        "grasp_success_claimed": False,
        "grasp_success_requires_on_site_human_confirmation": True,
        "training_authorized": False,
        "policy_execution_authorized": False,
        "formal_c0_batch": False,
        "formal_final_evidence_bound": False,
    }


def format_c0_smoke_postflight(report: Mapping[str, object]) -> str:
    lines = [
        C0_SMOKE_STATUS,
        f"dataset_root_exists: {report['dataset_root_exists']}",
        f"total_episodes: {report['total_episodes']}",
        f"fps: {report['fps']}",
        f"task: {report['task']}",
        f"image_feature_keys: {report['image_feature_keys']}",
        f"action_names: {report['action_names']}",
        f"observation_state_names: {report['observation_state_names']}",
        f"episode_frame_count: {report['episode_frame_count']}",
        f"videos: {report['videos']}",
        f"gripper_pos_semantics: {report['gripper_pos_semantics']}",
        "gripper_pos is last commanded 0/100 state, not physical gripper feedback.",
        "Grasp success is not claimed; confirm on site.",
        "training_authorized: False",
        "policy_execution_authorized: False",
        "formal_c0_batch: False",
        "formal_final_evidence_bound: False",
    ]
    return "\n".join(lines)

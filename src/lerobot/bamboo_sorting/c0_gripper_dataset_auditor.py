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

"""Offline, read-only auditor for the C0 gripper slice of a LeRobot dataset episode.

This module audits one *explicitly indexed* episode of a local LeRobot dataset
against the single-strip C0 gripper command rules. It reads only
``meta/info.json`` (via ``json``) and the ``action``/``observation.state``/
``episode_index``/``frame_index``/``timestamp`` (and ``index`` when present)
columns of ``data/**/*.parquet`` (via ``pyarrow.parquet``). It never loads or
decodes video, never writes into or modifies the audited dataset directory,
never touches the Hugging Face Hub, the network, threads, sockets, cameras, or
the AUBO controller, and never starts capture, training, or policy execution.

Scope boundary: this is a *dataset slice* auditor, not a complete C0 dataset
auditor, not a controller DO event auditor, and not a physical-grasp verifier.
The recorded ``gripper_pos`` / ``ee.gripper_pos`` values are software command
latches (the last commanded 0/100 state), not vacuum pressure, not physical
jaw position, and not grasp success feedback. No real C0 dataset currently has
a bound ``C0GripperEventV1`` controller-event sidecar, so even a structurally
perfect report keeps ``complete_gripper_audit_ready`` and every
training/execution authorization field fixed at ``False``.

Timing model follows ``record_loop``: the observation is captured before the
action is sent on each control cycle, so for frames that were successfully
saved into the dataset, ``observation[t] == action[t-1]`` for ``t >= 1``.
This module verifies that relation on the saved rows only; it says nothing
about DO writes that were actually executed.

Digest boundary: :class:`C0GripperDatasetSliceDigestV1` is a *gripper slice*
digest. It covers only the gripper columns plus the indexing/timing fields
listed in its payload. It is NOT a full dataset episode digest, it does not
cover video content, and it does not verify
``C0EpisodeManifestV1.dataset_episode_sha256`` — no canonical algorithm for
recomputing that field from a multi-file LeRobot episode exists yet, so this
module never compares against it.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .c0_capture_contract import C0_DATASET_FPS, C0EpisodeManifestV1
from .c0_gripper_contract import (
    GRIPPER_COMMANDED_ON,
    GRIPPER_RELEASED,
    derive_gripper_transition_frames,
)
from .contracts import ACTION_FIELD_NAMES
from .lerobot_bridge import (
    FORBIDDEN_VLA_KEYS,
    IMAGE_FEATURE_KEYS,
    build_cameras_set_v1_lerobot_features,
)
from .observation_contract import OBSERVATION_STATE_FIELD_NAMES

C0_GRIPPER_DATASET_AUDIT_REPORT_SCHEMA_VERSION: Final = "C0GripperDatasetAuditReportV1"
C0_GRIPPER_SLICE_DIGEST_SCHEMA_VERSION: Final = "C0GripperDatasetSliceDigestV1"

ACTION_GRIPPER_FIELD: Final = "ee.gripper_pos"
STATE_GRIPPER_FIELD: Final = "gripper_pos"
STATE_FEATURE_KEY: Final = "observation.state"
ACTION_FEATURE_KEY: Final = "action"

# Half a control cycle: timestamps must reproduce frame_index / fps within
# this absolute tolerance. Deterministic and derived only from the frozen
# 25 Hz dataset profile.
C0_TIMESTAMP_ABS_TOLERANCE_S: Final = 0.5 / C0_DATASET_FPS

# A constant run of the commanded gripper value shorter than this many frames
# is reported as chatter. Two frames is the smallest dwell that is visibly
# distinct from a single-frame pulse at 25 Hz.
C0_GRIPPER_MIN_DWELL_FRAMES: Final = 2

_REQUIRED_DATA_COLUMNS: Final = (
    "action",
    "observation.state",
    "episode_index",
    "frame_index",
    "timestamp",
)
_OPTIONAL_DATA_COLUMNS: Final = ("index",)

_HUMAN_OUTCOME_BLOCKER: Final = "human_outcome_not_success"


def _sorted_unique(messages: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(messages)))


@dataclass(frozen=True)
class C0GripperDatasetSliceDigestV1:
    """Deterministic SHA-256 over the gripper slice of one dataset episode.

    This digest covers only: the digest schema version, ``episode_index``,
    ``fps``, ``frame_count``, the action/state feature names, the sorted
    ``frame_index`` sequence, the ``timestamp`` sequence, and the two gripper
    command columns. Identical inputs produce identical digests; changing any
    gripper value, frame index, or timestamp changes the digest.

    It is NOT a full dataset episode digest: it excludes every non-gripper
    action/state dimension, all video content, and all dataset files outside
    the audited slice, and it never verifies or replaces
    ``C0EpisodeManifestV1.dataset_episode_sha256``.
    """

    schema_version: str
    episode_index: int
    fps: float
    frame_count: int
    action_feature_names: Sequence[str]
    observation_state_feature_names: Sequence[str]
    frame_index: Sequence[int]
    timestamps: Sequence[float]
    action_gripper_values: Sequence[float]
    observation_gripper_values: Sequence[float]

    def __post_init__(self) -> None:
        if self.schema_version != C0_GRIPPER_SLICE_DIGEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_GRIPPER_SLICE_DIGEST_SCHEMA_VERSION!r}"
            )
        if isinstance(self.episode_index, bool) or not isinstance(self.episode_index, int):
            raise ValueError("episode_index must be an integer, not bool")
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise ValueError("fps must be a finite positive number, not bool")
        fps = float(self.fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a finite positive number")
        object.__setattr__(self, "fps", fps)
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int):
            raise ValueError("frame_count must be a positive integer, not bool or float")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be a positive integer")

        for name in ("action_feature_names", "observation_state_feature_names"):
            raw = getattr(self, name)
            if isinstance(raw, (str, bytes)):
                raise ValueError(f"{name} must be a sequence of strings")
            values = tuple(raw)
            if not values or not all(isinstance(item, str) for item in values):
                raise ValueError(f"{name} must be a non-empty sequence of strings")
            object.__setattr__(self, name, values)

        if isinstance(self.frame_index, (str, bytes)):
            raise ValueError("frame_index must be a sequence of integers")
        frame_index: list[int] = []
        for index, item in enumerate(self.frame_index):
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"frame_index[{index}] must be an integer, not bool")
            frame_index.append(item)
        object.__setattr__(self, "frame_index", tuple(frame_index))

        for name in ("timestamps", "action_gripper_values", "observation_gripper_values"):
            raw = getattr(self, name)
            if isinstance(raw, (str, bytes)):
                raise ValueError(f"{name} must be a sequence of finite numbers")
            values: list[float] = []
            for index, item in enumerate(raw):
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise ValueError(f"{name}[{index}] must be a finite number, not bool")
                value = float(item)
                if not math.isfinite(value):
                    raise ValueError(f"{name}[{index}] must be finite")
                values.append(value)
            object.__setattr__(self, name, tuple(values))

        if not (
            len(self.frame_index)
            == len(self.timestamps)
            == len(self.action_gripper_values)
            == len(self.observation_gripper_values)
            == self.frame_count
        ):
            raise ValueError(
                "frame_index, timestamps, gripper columns, and frame_count must agree"
            )

    def payload(self) -> dict[str, object]:
        """Return the canonical JSON payload covered by this digest."""

        return {
            "schema_version": self.schema_version,
            "episode_index": self.episode_index,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "action_feature_names": list(self.action_feature_names),
            "observation_state_feature_names": list(self.observation_state_feature_names),
            "frame_index": list(self.frame_index),
            "timestamps": list(self.timestamps),
            "action_gripper_values": list(self.action_gripper_values),
            "observation_gripper_values": list(self.observation_gripper_values),
        }

    @property
    def sha256(self) -> str:
        """Return the deterministic SHA-256 of the canonical JSON payload."""

        encoded = json.dumps(
            self.payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible record with explicit scope limits."""

        record = {
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "payload": self.payload(),
            "digest_scope": "gripper_slice_only",
            "covers_full_dataset_episode": False,
            "covers_video_content": False,
            "verifies_manifest_dataset_episode_sha256": False,
        }
        return copy.deepcopy(record)


def _require_feature(
    features: dict[str, object], key: str, error_code: str, errors: list[str]
) -> dict[str, object] | None:
    feature = features.get(key)
    if not isinstance(feature, dict):
        errors.append(error_code)
        return None
    return feature


def _expected_features() -> dict[str, object]:
    """Return the frozen CameraSetV1 LeRobot feature schema as parsed from JSON."""

    return json.loads(json.dumps(build_cameras_set_v1_lerobot_features()))


def _audit_info_features(
    info: object, manifest: C0EpisodeManifestV1, errors: list[str]
) -> dict[str, object] | None:
    """Validate meta/info.json content; return the features mapping when usable."""

    if not isinstance(info, dict):
        errors.append("info_json_invalid")
        return None
    features = info.get("features")
    if not isinstance(features, dict):
        errors.append("info_json_invalid")
        return None

    fps = info.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or float(fps) != C0_DATASET_FPS
        or float(fps) != manifest.fps
    ):
        errors.append("fps_mismatch")

    forbidden = sorted(
        key
        for key in features
        if isinstance(key, str)
        and (
            key in FORBIDDEN_VLA_KEYS
            or "handeye" in key
            or "fixed" in key
            or "wrist" in key
            or "depth" in key
        )
    )
    for key in forbidden:
        errors.append(f"forbidden_feature_present:{key}")

    expected = _expected_features()

    # CameraSetV1 is frozen at exactly two RGB streams: any other
    # observation.images.* key is rejected, not only known legacy names.
    unexpected_images = sorted(
        key
        for key in features
        if isinstance(key, str)
        and key.startswith("observation.images.")
        and key not in IMAGE_FEATURE_KEYS
    )
    for key in unexpected_images:
        errors.append(f"unexpected_image_feature:{key}")

    action = _require_feature(features, ACTION_FEATURE_KEY, "action_shape_mismatch", errors)
    state = _require_feature(features, STATE_FEATURE_KEY, "state_shape_mismatch", errors)
    global_rgb = _require_feature(
        features, IMAGE_FEATURE_KEYS[0], "missing_global_rgb_feature", errors
    )
    grasp_rgb = _require_feature(
        features, IMAGE_FEATURE_KEYS[1], "missing_grasp_rgb_feature", errors
    )

    # The image features must match the frozen bridge schema exactly: dtype,
    # shape, and names. Only the schema is checked here; video content is
    # never read or decoded.
    if global_rgb is not None and global_rgb != expected[IMAGE_FEATURE_KEYS[0]]:
        errors.append("global_rgb_feature_mismatch")
    if grasp_rgb is not None and grasp_rgb != expected[IMAGE_FEATURE_KEYS[1]]:
        errors.append("grasp_rgb_feature_mismatch")

    if action is not None:
        if action.get("dtype") != expected[ACTION_FEATURE_KEY]["dtype"]:
            errors.append("action_dtype_mismatch")
        if list(action.get("shape", [])) != [len(ACTION_FIELD_NAMES)]:
            errors.append("action_shape_mismatch")
        names = action.get("names")
        if not isinstance(names, list) or tuple(names) != ACTION_FIELD_NAMES:
            errors.append("action_names_mismatch")
    if state is not None:
        if state.get("dtype") != expected[STATE_FEATURE_KEY]["dtype"]:
            errors.append("state_dtype_mismatch")
        if list(state.get("shape", [])) != [len(OBSERVATION_STATE_FIELD_NAMES)]:
            errors.append("state_shape_mismatch")
        names = state.get("names")
        if not isinstance(names, list) or tuple(names) != OBSERVATION_STATE_FIELD_NAMES:
            errors.append("state_names_mismatch")

    return features


_INTEGER_DATA_COLUMNS: Final = ("episode_index", "frame_index", "index")
_LIST_DATA_COLUMNS: Final = ("action", "observation.state")


def _arrow_column_type_valid(name: str, arrow_type: object) -> bool:
    """Return whether one parquet column's Arrow type is safe and meaningful.

    Index columns must be integer Arrow types so float or string indices can
    never be silently truncated by a NumPy cast. The timestamp must be
    numeric. action/state must be (large or fixed-size) lists of numeric
    values.
    """

    if name in _INTEGER_DATA_COLUMNS:
        return bool(pa.types.is_integer(arrow_type))
    if name == "timestamp":
        return bool(pa.types.is_floating(arrow_type) or pa.types.is_integer(arrow_type))
    if name in _LIST_DATA_COLUMNS:
        if (
            pa.types.is_list(arrow_type)
            or pa.types.is_large_list(arrow_type)
            or pa.types.is_fixed_size_list(arrow_type)
        ):
            value_type = arrow_type.value_type
            return bool(pa.types.is_floating(value_type) or pa.types.is_integer(value_type))
        return False
    return False


def _convert_column(name: str, values: list, errors: list[str]) -> np.ndarray | None:
    """Convert one collected column, turning any failure into an error code."""

    if any(value is None for value in values):
        errors.append(f"invalid_column_value:{name}")
        return None
    dtype = np.int64 if name in _INTEGER_DATA_COLUMNS else np.float64
    try:
        return np.asarray(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"invalid_column_value:{name}")
        return None


def _load_episode_rows(
    dataset_root: Path, episode_index: int, errors: list[str]
) -> dict[str, np.ndarray] | None:
    """Read one episode's required columns from data/**/*.parquet, read-only."""

    data_dir = dataset_root / "data"
    files = sorted(data_dir.rglob("*.parquet")) if data_dir.is_dir() else []
    if not files:
        errors.append("data_parquet_missing")
        return None

    wanted = [*_REQUIRED_DATA_COLUMNS, *_OPTIONAL_DATA_COLUMNS]
    tables = []
    load_failed = False
    for path in files:
        try:
            schema = pq.read_schema(path)
            available = set(schema.names)
            missing = [name for name in _REQUIRED_DATA_COLUMNS if name not in available]
            if missing:
                for name in missing:
                    errors.append(f"missing_required_column:{name}")
                load_failed = True
                continue
            columns = [name for name in wanted if name in available]
            type_failed = False
            for name in columns:
                if not _arrow_column_type_valid(name, schema.field(name).type):
                    errors.append(f"invalid_column_type:{name}")
                    type_failed = True
            if type_failed:
                load_failed = True
                continue
            tables.append(pq.read_table(path, columns=columns, use_threads=False))
        except Exception:
            errors.append("parquet_read_error")
            load_failed = True
    if load_failed or not tables:
        return None

    def _column(name: str) -> list:
        values: list = []
        for table in tables:
            if name in table.column_names:
                values.extend(table.column(name).to_pylist())
        return values

    episode_column = _convert_column("episode_index", _column("episode_index"), errors)
    if episode_column is None:
        return None
    mask = episode_column == episode_index
    if not bool(mask.any()):
        errors.append("episode_index_not_found")
        return None

    action = _convert_column("action", _column("action"), errors)
    state = _convert_column("observation.state", _column("observation.state"), errors)
    frame_index = _convert_column("frame_index", _column("frame_index"), errors)
    timestamp = _convert_column("timestamp", _column("timestamp"), errors)
    if (
        action is None
        or state is None
        or frame_index is None
        or timestamp is None
    ):
        return None

    return {
        "action": action[mask],
        "state": state[mask],
        "frame_index": frame_index[mask],
        "timestamp": timestamp[mask],
    }


def _audit_dataset_episode(
    dataset_root: Path,
    episode_index: int,
    manifest: C0EpisodeManifestV1,
) -> dict[str, object]:
    """Run the full read-only audit; pure with respect to its inputs."""

    errors: list[str] = []
    blockers: list[str] = []
    frame_count: int | None = None
    fps: float | None = None
    slice_sha256: str | None = None
    on_frames: tuple[int, ...] = ()
    off_frames: tuple[int, ...] = ()

    # Defense in depth: the manifest constructor already guarantees these, but
    # the auditor re-checks them so a future schema version cannot silently
    # drop the invariant here.
    if not manifest.finalized:
        errors.append("manifest_not_finalized")
    if not manifest.raw_data_immutable:
        errors.append("manifest_raw_data_mutable")

    if not dataset_root.is_dir():
        errors.append("dataset_root_missing")
        return _result(
            errors,
            blockers,
            frame_count,
            fps,
            slice_sha256,
            on_frames,
            off_frames,
            manifest,
        )

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        errors.append("info_json_missing")
        info = None
    else:
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            errors.append("info_json_invalid")
            info = None

    features: dict[str, object] | None = None
    if info is not None:
        features = _audit_info_features(info, manifest, errors)
        if isinstance(info, dict):
            fps_value = info.get("fps")
            if isinstance(fps_value, (int, float)) and not isinstance(fps_value, bool):
                candidate = float(fps_value)
                # Only a finite, positive fps may be stored or feed the slice
                # digest; NaN/Inf/zero/negative values keep fps_mismatch and
                # leave the report's fps and digest as None.
                if math.isfinite(candidate) and candidate > 0:
                    fps = candidate

    rows = _load_episode_rows(dataset_root, episode_index, errors)
    if rows is None:
        return _result(
            errors,
            blockers,
            frame_count,
            fps,
            slice_sha256,
            on_frames,
            off_frames,
            manifest,
        )

    action = rows["action"]
    state = rows["state"]
    raw_frame_index = rows["frame_index"]
    timestamp = rows["timestamp"]
    frame_count = int(len(raw_frame_index))

    if action.ndim != 2 or action.shape[1] != len(ACTION_FIELD_NAMES):
        errors.append("action_dim_mismatch")
    if state.ndim != 2 or state.shape[1] != len(OBSERVATION_STATE_FIELD_NAMES):
        errors.append("state_dim_mismatch")

    # Deterministic ordering: the auditor sorts by frame_index itself; parquet
    # row order is never trusted.
    order = np.argsort(raw_frame_index, kind="stable")
    frame_index = raw_frame_index[order]
    action = action[order]
    state = state[order]
    timestamp = timestamp[order]

    has_duplicates = bool(len(np.unique(frame_index)) != len(frame_index))
    if has_duplicates:
        errors.append("duplicate_frame_index")
    frames_contiguous = False
    if not has_duplicates:
        if int(frame_index[0]) != 0:
            errors.append("frame_index_nonzero_start")
        expected = np.arange(len(frame_index), dtype=np.int64)
        frames_contiguous = bool(np.array_equal(frame_index, expected))
        if not frames_contiguous and int(frame_index[0]) == 0:
            errors.append("missing_frame_index")

    if frame_count != manifest.frame_count:
        errors.append("frame_count_mismatch")

    finite_timestamps = bool(np.isfinite(timestamp).all())
    if not finite_timestamps:
        errors.append("timestamp_non_finite")
    else:
        if bool((timestamp < 0).any()):
            errors.append("timestamp_negative")
        if len(timestamp) > 1 and bool((np.diff(timestamp) <= 0).any()):
            errors.append("timestamp_not_strictly_increasing")
        if (
            fps is not None
            and "fps_mismatch" not in errors
            and not bool(
                np.all(
                    np.abs(timestamp - frame_index.astype(np.float64) / fps)
                    <= C0_TIMESTAMP_ABS_TOLERANCE_S
                )
            )
        ):
            errors.append("timestamp_frame_mismatch")

    if not bool(np.isfinite(action).all()):
        errors.append("action_non_finite")
    if not bool(np.isfinite(state).all()):
        errors.append("state_non_finite")

    names_valid = (
        features is not None
        and "action_names_mismatch" not in errors
        and "state_names_mismatch" not in errors
        and "action_dim_mismatch" not in errors
        and "state_dim_mismatch" not in errors
    )
    action_gripper: np.ndarray | None = None
    observation_gripper: np.ndarray | None = None
    if names_valid:
        # Gripper columns are located by metadata names, never by a hardcoded
        # last-dimension position.
        action_gripper = action[:, ACTION_FIELD_NAMES.index(ACTION_GRIPPER_FIELD)]
        observation_gripper = state[:, OBSERVATION_STATE_FIELD_NAMES.index(STATE_GRIPPER_FIELD)]

    gripper_binary_valid = False
    if (
        action_gripper is not None
        and observation_gripper is not None
        and bool(np.isfinite(action_gripper).all())
        and bool(np.isfinite(observation_gripper).all())
    ):
        invalid = (action_gripper != GRIPPER_RELEASED) & (
            action_gripper != GRIPPER_COMMANDED_ON
        )
        invalid |= (observation_gripper != GRIPPER_RELEASED) & (
            observation_gripper != GRIPPER_COMMANDED_ON
        )
        if bool(invalid.any()):
            errors.append("gripper_value_invalid")
        else:
            gripper_binary_valid = True

    if gripper_binary_valid:
        assert action_gripper is not None and observation_gripper is not None
        action_values = tuple(float(value) for value in action_gripper)
        observation_values = tuple(float(value) for value in observation_gripper)

        if observation_values[0] != GRIPPER_RELEASED:
            errors.append("observation_start_not_released")
        if action_values[0] != GRIPPER_RELEASED:
            errors.append("action_not_start_released")
        if action_values[-1] != GRIPPER_RELEASED:
            errors.append("action_not_end_released")

        # The latch relation only has a frame-correspondence meaning when the
        # saved frames are exactly 0..N-1; gaps or duplicates already carry
        # their own error codes.
        if frames_contiguous:
            for index in range(1, len(action_values)):
                if observation_values[index] != action_values[index - 1]:
                    errors.append("observation_action_latch_mismatch")
                    break

            on_frames, off_frames = derive_gripper_transition_frames(
                action_values, observation_values
            )
            if not on_frames:
                errors.append("missing_command_on")
            elif len(on_frames) > 1:
                errors.append("multiple_command_on")
            if not off_frames:
                errors.append("missing_command_off")
            elif len(off_frames) > 1:
                errors.append("multiple_command_off")
            if (
                len(on_frames) == 1
                and len(off_frames) == 1
                and off_frames[0] < on_frames[0]
            ):
                errors.append("command_off_before_command_on")

            # Chatter: any constant run of the commanded value shorter than
            # the minimum dwell. This catches single-frame pulses even when
            # the on/off counts are otherwise exactly one each.
            run_lengths: list[int] = []
            previous_value: float | None = None
            for value in action_values:
                if previous_value is not None and value == previous_value:
                    run_lengths[-1] += 1
                else:
                    run_lengths.append(1)
                previous_value = value
            if any(length < C0_GRIPPER_MIN_DWELL_FRAMES for length in run_lengths):
                errors.append("gripper_chatter")

    # The slice digest is a content identifier, not a validity claim: it is
    # computed whenever the sorted slice and its digest inputs are well defined
    # (unique frames, finite timestamps, located and finite gripper columns),
    # even if the content itself fails audit rules.
    if (
        not has_duplicates
        and finite_timestamps
        and action_gripper is not None
        and observation_gripper is not None
        and bool(np.isfinite(action_gripper).all())
        and bool(np.isfinite(observation_gripper).all())
        and fps is not None
        and frame_count is not None
    ):
        digest = C0GripperDatasetSliceDigestV1(
            schema_version=C0_GRIPPER_SLICE_DIGEST_SCHEMA_VERSION,
            episode_index=episode_index,
            fps=fps,
            frame_count=frame_count,
            action_feature_names=ACTION_FIELD_NAMES,
            observation_state_feature_names=OBSERVATION_STATE_FIELD_NAMES,
            frame_index=tuple(int(value) for value in frame_index),
            timestamps=tuple(float(value) for value in timestamp),
            action_gripper_values=tuple(float(value) for value in action_gripper),
            observation_gripper_values=tuple(float(value) for value in observation_gripper),
        )
        slice_sha256 = digest.sha256

    return _result(
        errors,
        blockers,
        frame_count,
        fps,
        slice_sha256,
        on_frames,
        off_frames,
        manifest,
    )


def _result(
    errors: list[str],
    blockers: list[str],
    frame_count: int | None,
    fps: float | None,
    slice_sha256: str | None,
    on_frames: tuple[int, ...],
    off_frames: tuple[int, ...],
    manifest: C0EpisodeManifestV1,
) -> dict[str, object]:
    if manifest.human_outcome != "success":
        blockers.append(_HUMAN_OUTCOME_BLOCKER)
    return {
        "errors": _sorted_unique(errors),
        "blockers": _sorted_unique(blockers),
        "frame_count": frame_count,
        "fps": fps,
        "gripper_slice_sha256": slice_sha256,
        "derived_command_on_frames": on_frames,
        "derived_command_off_frames": off_frames,
    }


@dataclass(frozen=True)
class C0GripperDatasetAuditReportV1:
    """Deterministic read-only audit report for one dataset episode gripper slice.

    The constructor accepts only the schema version, the dataset root, the
    explicit episode index, and the episode manifest; it then performs the
    entire read-only audit itself. Errors, blockers, derived transition
    frames, counts, the slice digest, and the structural-validity flag are all
    computed internally and can never be supplied or forged by a caller.

    ``dataset_slice_structurally_valid`` means only that the saved
    action/state gripper slice satisfies the C0 structural and timing rules.
    ``complete_gripper_audit_ready`` stays ``False`` in this schema version
    because no controller-event evidence is bound to any real dataset yet.
    This report never authorizes capture, training, or policy execution and
    never proves a physical grasp.

    Equality compares the full audit snapshot (derived results included), not
    just the input coordinates, so two reports over the same path compare
    unequal when the underlying dataset content changed between audits.
    """

    schema_version: str
    dataset_root: str
    episode_index: int
    episode_manifest: C0EpisodeManifestV1
    _errors: tuple[str, ...] = field(init=False, repr=False)
    _blockers: tuple[str, ...] = field(init=False, repr=False)
    _frame_count: int | None = field(init=False, repr=False)
    _fps: float | None = field(init=False, repr=False)
    _gripper_slice_sha256: str | None = field(init=False, repr=False)
    _derived_command_on_frames: tuple[int, ...] = field(init=False, repr=False)
    _derived_command_off_frames: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != C0_GRIPPER_DATASET_AUDIT_REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {C0_GRIPPER_DATASET_AUDIT_REPORT_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.dataset_root, str) or not self.dataset_root:
            raise ValueError("dataset_root must be a non-empty string path")
        if isinstance(self.episode_index, bool) or not isinstance(self.episode_index, int):
            raise ValueError("episode_index must be an integer, not bool")
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if not isinstance(self.episode_manifest, C0EpisodeManifestV1):
            raise ValueError("episode_manifest must be a C0EpisodeManifestV1")

        outcome = _audit_dataset_episode(
            Path(self.dataset_root), self.episode_index, self.episode_manifest
        )
        object.__setattr__(self, "_errors", outcome["errors"])
        object.__setattr__(self, "_blockers", outcome["blockers"])
        object.__setattr__(self, "_frame_count", outcome["frame_count"])
        object.__setattr__(self, "_fps", outcome["fps"])
        object.__setattr__(self, "_gripper_slice_sha256", outcome["gripper_slice_sha256"])
        object.__setattr__(
            self, "_derived_command_on_frames", outcome["derived_command_on_frames"]
        )
        object.__setattr__(
            self, "_derived_command_off_frames", outcome["derived_command_off_frames"]
        )

    @property
    def episode_id(self) -> str:
        return self.episode_manifest.episode_id

    @property
    def frame_count(self) -> int | None:
        return self._frame_count

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def gripper_slice_sha256(self) -> str | None:
        return self._gripper_slice_sha256

    @property
    def derived_command_on_frames(self) -> tuple[int, ...]:
        return self._derived_command_on_frames

    @property
    def derived_command_off_frames(self) -> tuple[int, ...]:
        return self._derived_command_off_frames

    @property
    def errors(self) -> tuple[str, ...]:
        """Audit errors in stable sorted order."""

        return self._errors

    @property
    def blockers(self) -> tuple[str, ...]:
        """Review blockers in stable sorted order."""

        return self._blockers

    @property
    def dataset_slice_structurally_valid(self) -> bool:
        """Return structural validity of the saved gripper slice only."""

        return not self._errors

    @property
    def complete_gripper_audit_ready(self) -> bool:
        """Always False: no controller-event evidence is bound yet."""

        return False

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached JSON-compatible report that grants no authorization."""

        record = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "dataset_root": self.dataset_root,
            "frame_count": self._frame_count,
            "fps": self._fps,
            "human_outcome": self.episode_manifest.human_outcome,
            "gripper_slice_sha256": self._gripper_slice_sha256,
            "gripper_slice_digest_schema_version": (
                C0_GRIPPER_SLICE_DIGEST_SCHEMA_VERSION
                if self._gripper_slice_sha256 is not None
                else None
            ),
            "gripper_slice_digest_scope": (
                "gripper_slice_only_not_full_episode_no_video"
                if self._gripper_slice_sha256 is not None
                else None
            ),
            "derived_command_on_frames": list(self._derived_command_on_frames),
            "derived_command_off_frames": list(self._derived_command_off_frames),
            "errors": list(self._errors),
            "blockers": list(self._blockers),
            "dataset_slice_structurally_valid": self.dataset_slice_structurally_valid,
            "complete_gripper_audit_ready": False,
            "controller_event_evidence_bound": False,
            "controller_output_verified": False,
            "dataset_episode_sha256_match": None,
            "manifest_dataset_episode_sha256_verified": False,
            "video_content_verified": False,
            "physical_gripper_feedback_available": False,
            "physical_grasp_success_proven": False,
            "training_authorized": False,
            "policy_execution_authorized": False,
            "serialized_record_grants_live_authorization": False,
            "hardware_access_performed_by_audit": False,
            "dataset_files_modified_by_audit": False,
        }
        return copy.deepcopy(record)


def audit_c0_gripper_dataset_episode(
    dataset_root: str | Path,
    episode_index: int,
    episode_manifest: C0EpisodeManifestV1,
) -> C0GripperDatasetAuditReportV1:
    """Audit the gripper slice of one explicitly indexed local dataset episode.

    The caller must pass ``episode_index`` explicitly; it is never guessed
    from an ``episode_id`` string. The read is offline and read-only: only
    ``meta/info.json`` and the required ``data/**/*.parquet`` columns are
    opened, video files are never loaded or decoded, and nothing is written.

    A structurally valid result proves only the software command sequence
    saved in the dataset. It does not prove that DO writes executed, that the
    physical gripper moved, or that a grasp succeeded, and it never authorizes
    training or policy execution.
    """

    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise TypeError("episode_index must be an integer, not bool")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    if not isinstance(episode_manifest, C0EpisodeManifestV1):
        raise TypeError("episode_manifest must be a C0EpisodeManifestV1")
    if not isinstance(dataset_root, (str, Path)):
        raise TypeError("dataset_root must be a string or Path")
    return C0GripperDatasetAuditReportV1(
        schema_version=C0_GRIPPER_DATASET_AUDIT_REPORT_SCHEMA_VERSION,
        dataset_root=str(dataset_root),
        episode_index=episode_index,
        episode_manifest=episode_manifest,
    )

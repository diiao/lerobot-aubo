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

"""Build a hardware-free review pack for the ten formal C0 episodes.

This module reads existing parquet, video, and capture-sidecar evidence. It
does not import robot/camera drivers, open devices or sockets, start threads,
run a VLM, train a policy, or create a final C0 episode/batch manifest.

``human_outcome`` in every derived review record intentionally starts as
``pending``. The pre-existing sidecar value is preserved separately as
``source_human_outcome`` and is never treated as physical-grasp proof.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .c0_gripper_contract import derive_gripper_transition_frames

C0_REVIEW_EPISODE_SCHEMA_VERSION: Final = "C0PostCaptureReviewEpisodeV1"
C0_REVIEW_MANIFEST_SCHEMA_VERSION: Final = "C0PostCaptureReviewManifestV1"
C0_REVIEW_DATASET_NAME: Final = "c0_formal_single_strip_20260921_full"
C0_REVIEW_EXPECTED_EPISODES: Final = 10
C0_REVIEW_IMAGE_KEYS: Final = (
    "observation.images.global_rgb",
    "observation.images.grasp_rgb",
)
C0_REVIEW_TASK_TEXT: Final = "Pick one strip and place it in the collection area."
C0_REVIEW_FRAME_LABELS: Final = (
    "start",
    "gripper_on_before",
    "gripper_on",
    "gripper_on_after",
    "gripper_on_hold_mid",
    "gripper_off_before",
    "gripper_off",
    "gripper_off_after",
    "end",
)


class C0PostCaptureReviewError(RuntimeError):
    """Raised when source evidence is incomplete, contradictory, or unsafe."""


@dataclass(frozen=True)
class C0ReviewSourceV1:
    """Explicit aggregate-to-source mapping; no source is auto-discovered."""

    aggregate_episode_index: int
    source_dataset_ref: str
    source_episode_index: int
    capture_sidecar_ref: str
    formal_episode_index: int


C0_REVIEW_SOURCES: Final = (
    C0ReviewSourceV1(
        0,
        "datasets/c0_formal_single_strip_20260921_run01",
        0,
        "artifacts/c0_capture/c0_formal_single_strip_20260921_run01/episode-00-capture.json",
        0,
    ),
    C0ReviewSourceV1(
        1,
        "datasets/c0_formal_single_strip_20260921_run02",
        0,
        "artifacts/c0_capture/c0_formal_single_strip_20260921_run02/episode-00-capture.json",
        1,
    ),
    *tuple(
        C0ReviewSourceV1(
            local_index + 2,
            "datasets/c0_formal_single_strip_20260921_run04",
            local_index,
            (
                "artifacts/c0_capture/c0_formal_single_strip_20260921_run04/"
                f"episode-{local_index:02d}-capture.json"
            ),
            local_index + 2,
        )
        for local_index in range(8)
    ),
)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def sha256_file(path: Path) -> str:
    """Hash one local file without changing it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class C0ReviewEpisodeV1:
    """One pending human-review record with immutable evidence references."""

    aggregate_episode_index: int
    source_dataset_ref: str
    source_episode_index: int
    formal_episode_index: int
    episode_id: str
    scene_id: str
    split_name: str
    frame_count: int
    fps: float
    gripper_on_frame: int
    gripper_off_frame: int
    selected_frames: Sequence[tuple[str, int]]
    keyframe_files: Sequence[tuple[str, str, str]]
    capture_sidecar_ref: str
    capture_sidecar_sha256: str
    source_human_outcome: str
    compatibility_notes: Sequence[str] = ()
    schema_version: str = C0_REVIEW_EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != C0_REVIEW_EPISODE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {C0_REVIEW_EPISODE_SCHEMA_VERSION!r}")
        for name in (
            "aggregate_episode_index",
            "source_episode_index",
            "formal_episode_index",
            "frame_count",
            "gripper_on_frame",
            "gripper_off_frame",
        ):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if not isinstance(self.fps, (int, float)) or isinstance(self.fps, bool) or float(self.fps) != 25.0:
            raise ValueError("fps must be 25")
        object.__setattr__(self, "fps", float(self.fps))
        if not self.gripper_on_frame < self.gripper_off_frame < self.frame_count:
            raise ValueError("gripper transition frames must be ordered and inside the episode")
        for name in (
            "source_dataset_ref",
            "episode_id",
            "scene_id",
            "capture_sidecar_ref",
        ):
            _require_nonempty_string(name, getattr(self, name))
        if self.split_name not in {"train", "validation"}:
            raise ValueError("split_name must be train or validation")
        if self.source_human_outcome not in {"success", "failure", "uncertain"}:
            raise ValueError("source_human_outcome must be success, failure, or uncertain")
        _require_sha256("capture_sidecar_sha256", self.capture_sidecar_sha256)

        selected_frames = tuple(self.selected_frames)
        if tuple(label for label, _ in selected_frames) != C0_REVIEW_FRAME_LABELS:
            raise ValueError("selected_frames must contain the canonical review labels in order")
        for label, index in selected_frames:
            _require_nonempty_string("selected frame label", label)
            if not _is_int(index) or not 0 <= index < self.frame_count:
                raise ValueError(f"selected frame {label!r} is outside the episode")
        object.__setattr__(self, "selected_frames", selected_frames)

        keyframe_files = tuple(self.keyframe_files)
        if tuple(item[0] for item in keyframe_files) != C0_REVIEW_IMAGE_KEYS:
            raise ValueError("keyframe_files must contain global_rgb and grasp_rgb in order")
        for image_key, ref, digest in keyframe_files:
            _require_nonempty_string("image_key", image_key)
            _require_nonempty_string("keyframe ref", ref)
            _require_sha256("keyframe sha256", digest)
        object.__setattr__(self, "keyframe_files", keyframe_files)

        notes = tuple(self.compatibility_notes)
        if not all(isinstance(note, str) and note.strip() for note in notes):
            raise ValueError("compatibility_notes must contain non-empty strings")
        object.__setattr__(self, "compatibility_notes", notes)

    def to_manifest_record(self) -> dict[str, object]:
        """Return detached JSON data; review status never grants authorization."""

        selected = {label: index for label, index in self.selected_frames}
        keyframes = {
            image_key: {"ref": ref, "sha256": digest}
            for image_key, ref, digest in self.keyframe_files
        }
        record = {
            "schema_version": self.schema_version,
            "aggregate_episode_index": self.aggregate_episode_index,
            "source_dataset_ref": self.source_dataset_ref,
            "source_episode_index": self.source_episode_index,
            "formal_episode_index": self.formal_episode_index,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "split_name": self.split_name,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "gripper_on_frame": self.gripper_on_frame,
            "gripper_off_frame": self.gripper_off_frame,
            "selected_frames": selected,
            "selected_frame_indices": list(selected.values()),
            "keyframe_files": keyframes,
            "capture_sidecar_ref": self.capture_sidecar_ref,
            "capture_sidecar_sha256": self.capture_sidecar_sha256,
            "source_human_outcome": self.source_human_outcome,
            "human_outcome": "pending",
            "human_note": "",
            "vlm_shadow_status": "not_run",
            "final_manifest_bound": False,
            "training_authorized": False,
            "policy_execution_authorized": False,
            "hardware_access_performed": False,
            "gripper_evidence_semantics": "software_command_and_controller_output_not_physical_grasp_proof",
            "compatibility_notes": list(self.compatibility_notes),
        }
        return copy.deepcopy(record)


@dataclass(frozen=True)
class C0ReviewManifestV1:
    """Validated collection of exactly ten pending C0 review records."""

    dataset_ref: str
    aggregate_dataset_splits: Mapping[str, str]
    episodes: Sequence[C0ReviewEpisodeV1]
    schema_version: str = C0_REVIEW_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != C0_REVIEW_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {C0_REVIEW_MANIFEST_SCHEMA_VERSION!r}")
        _require_nonempty_string("dataset_ref", self.dataset_ref)
        splits = dict(self.aggregate_dataset_splits)
        if splits != {"train": "0:10"}:
            raise ValueError("aggregate_dataset_splits must truthfully preserve {'train': '0:10'}")
        object.__setattr__(self, "aggregate_dataset_splits", splits)

        episodes = tuple(self.episodes)
        if len(episodes) != C0_REVIEW_EXPECTED_EPISODES:
            raise ValueError("review manifest must contain exactly ten episodes")
        if not all(isinstance(item, C0ReviewEpisodeV1) for item in episodes):
            raise ValueError("episodes must contain only C0ReviewEpisodeV1 records")
        if [item.aggregate_episode_index for item in episodes] != list(range(10)):
            raise ValueError("aggregate episode indices must be 0..9")
        if [item.formal_episode_index for item in episodes] != list(range(10)):
            raise ValueError("formal episode indices must be 0..9")
        if [item.split_name for item in episodes] != [*(["train"] * 8), "validation", "validation"]:
            raise ValueError("review split must be eight train followed by two validation episodes")
        if len({item.episode_id for item in episodes}) != 10:
            raise ValueError("episode_id values must be unique")
        if len({item.scene_id for item in episodes}) != 10:
            raise ValueError("scene_id values must be unique")
        object.__setattr__(self, "episodes", episodes)

    def to_manifest_record(self) -> dict[str, object]:
        """Return a detached review manifest, not a final C0 batch manifest."""

        record = {
            "schema_version": self.schema_version,
            "dataset_ref": self.dataset_ref,
            "episode_count": len(self.episodes),
            "review_split_source": "capture_sidecar_episode_plan",
            "review_splits": {"train": list(range(8)), "validation": [8, 9]},
            "aggregate_dataset_splits": dict(self.aggregate_dataset_splits),
            "aggregate_split_mismatch_acknowledged": True,
            "episodes": [item.to_manifest_record() for item in self.episodes],
            "human_outcome_status": "pending_for_all_episodes",
            "vlm_shadow_status": "not_run",
            "canonical_full_episode_digest_bound": False,
            "final_c0_episode_manifests_generated": False,
            "final_c0_batch_manifest_generated": False,
            "training_authorized": False,
            "inference_authorized": False,
            "policy_execution_authorized": False,
            "hardware_access_performed": False,
        }
        return copy.deepcopy(record)


def select_review_frames(
    frame_count: int,
    gripper_on_frame: int,
    gripper_off_frame: int,
) -> tuple[tuple[str, int], ...]:
    """Choose canonical review frames, clipping every neighbor to valid bounds."""

    if not _is_int(frame_count) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    if not _is_int(gripper_on_frame) or not _is_int(gripper_off_frame):
        raise ValueError("gripper transition frames must be integers")
    if not 0 <= gripper_on_frame < gripper_off_frame < frame_count:
        raise ValueError("gripper transition frames must be ordered and inside the episode")

    def clip(index: int) -> int:
        return min(max(index, 0), frame_count - 1)

    values = (
        0,
        clip(gripper_on_frame - 1),
        gripper_on_frame,
        clip(gripper_on_frame + 1),
        gripper_on_frame + (gripper_off_frame - gripper_on_frame) // 2,
        clip(gripper_off_frame - 1),
        gripper_off_frame,
        clip(gripper_off_frame + 1),
        frame_count - 1,
    )
    return tuple(zip(C0_REVIEW_FRAME_LABELS, values, strict=True))


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C0PostCaptureReviewError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise C0PostCaptureReviewError(f"JSON root must be an object: {path}")
    return value


def _episode_metadata(dataset_root: Path) -> dict[int, dict[str, object]]:
    import pyarrow.parquet as pq

    files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise C0PostCaptureReviewError(f"episode metadata missing: {dataset_root}")
    rows: dict[int, dict[str, object]] = {}
    for path in files:
        for row in pq.read_table(path, use_threads=False).to_pylist():
            index = row.get("episode_index")
            if not _is_int(index) or index in rows:
                raise C0PostCaptureReviewError(f"invalid or duplicate episode_index in {path}")
            rows[index] = row
    return rows


def _episode_rows(dataset_root: Path, episode_index: int) -> dict[str, object]:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = ("action", "observation.state", "timestamp", "frame_index", "episode_index")
    tables = []
    for path in sorted((dataset_root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=list(columns), use_threads=False)
        tables.append(table)
    if not tables:
        raise C0PostCaptureReviewError(f"data parquet missing: {dataset_root}")
    table = pa.concat_tables(tables)
    episode_values = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    mask = episode_values == episode_index
    if not bool(mask.any()):
        raise C0PostCaptureReviewError(f"episode {episode_index} missing in {dataset_root}")
    result: dict[str, object] = {}
    for name in columns[:-1]:
        dtype = np.int64 if name == "frame_index" else np.float64
        result[name] = np.asarray(table.column(name).to_pylist(), dtype=dtype)[mask]
    order = np.argsort(result["frame_index"], kind="stable")
    return {name: values[order] for name, values in result.items()}


def _validate_source_slice(aggregate_rows: Mapping[str, object], source_rows: Mapping[str, object]) -> None:
    import numpy as np

    for name in ("action", "observation.state", "timestamp", "frame_index"):
        if not np.array_equal(aggregate_rows[name], source_rows[name]):
            raise C0PostCaptureReviewError(f"aggregate/source episode mismatch in {name}")


def _validate_sidecar(
    sidecar: Mapping[str, object], source: C0ReviewSourceV1, frame_count: int
) -> tuple[str, str, str, tuple[str, ...]]:
    plan = sidecar.get("episode_plan")
    evidence = sidecar.get("frame_evidence")
    if not isinstance(plan, dict) or not isinstance(evidence, dict):
        raise C0PostCaptureReviewError("capture sidecar lacks episode_plan or frame_evidence")
    timestamp_journal = evidence.get("timestamp_journal")
    gripper_journal = evidence.get("gripper_journal")
    if not isinstance(timestamp_journal, dict) or not isinstance(gripper_journal, dict):
        raise C0PostCaptureReviewError("capture sidecar lacks timestamp/gripper journal")
    if timestamp_journal.get("frame_count") != frame_count:
        raise C0PostCaptureReviewError("sidecar frame_count does not match dataset episode length")
    if sidecar.get("dataset_episode_durably_saved") is not True:
        raise C0PostCaptureReviewError("dataset episode is not marked durably saved")

    episode_id = _require_nonempty_string("episode_id", plan.get("episode_id"))
    scene_id = _require_nonempty_string("scene_id", plan.get("scene_id"))
    split_name = _require_nonempty_string("split_name", plan.get("split_name"))
    expected_split = "train" if source.formal_episode_index < 8 else "validation"
    if split_name != expected_split:
        raise C0PostCaptureReviewError("capture sidecar split does not match the formal 8/2 rule")
    if plan.get("episode_index") != source.source_episode_index:
        raise C0PostCaptureReviewError("capture sidecar source episode index mismatch")
    if float(plan.get("fps", -1)) != 25.0:
        raise C0PostCaptureReviewError("capture sidecar fps must be 25")
    if plan.get("instruction_text") != C0_REVIEW_TASK_TEXT:
        raise C0PostCaptureReviewError("capture sidecar task text mismatch")

    notes: list[str] = []
    declared_formal_index = plan.get("formal_episode_index")
    if source.formal_episode_index == 0 and declared_formal_index is None:
        notes.append("run01 formal_episode_index absent; explicit batch mapping assigns 0")
        if sidecar.get("status") != "dataset_saved_pending_capture_sidecar":
            raise C0PostCaptureReviewError("run01 compatibility status is unexpected")
        notes.append(
            "run01 legacy status dataset_saved_pending_capture_sidecar accepted because durable=true"
        )
    elif declared_formal_index != source.formal_episode_index:
        raise C0PostCaptureReviewError("capture sidecar formal_episode_index mismatch")

    source_outcome = sidecar.get("human_outcome")
    if source_outcome not in {"success", "failure", "uncertain"}:
        raise C0PostCaptureReviewError("capture sidecar human_outcome is invalid")
    if source.formal_episode_index == 0 and source_outcome == "success":
        notes.append("source sidecar says success; derived review remains pending until visual confirmation")
    return episode_id, scene_id, split_name, tuple(notes)


def _validate_controller_attempts(
    sidecar: Mapping[str, object], gripper_on_frame: int, gripper_off_frame: int
) -> None:
    evidence = sidecar["frame_evidence"]
    assert isinstance(evidence, dict)
    journal = evidence["gripper_journal"]
    assert isinstance(journal, dict)
    attempts = journal.get("controller_attempts")
    if not isinstance(attempts, list) or len(attempts) != 2:
        raise C0PostCaptureReviewError("controller evidence must contain exactly two attempts")
    expected = (("command_on", gripper_on_frame), ("command_off", gripper_off_frame))
    actual: list[tuple[object, object]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict) or not isinstance(attempt.get("event"), dict):
            raise C0PostCaptureReviewError("controller attempt event is missing")
        event = attempt["event"]
        if (
            event.get("do_api_success") is not True
            or event.get("controller_output_matches_requested") is not True
        ):
            raise C0PostCaptureReviewError("controller attempt did not complete with matching output")
        actual.append((event.get("event_type"), event.get("frame_index")))
    if tuple(actual) != expected:
        raise C0PostCaptureReviewError("controller event order/frame does not match dataset transitions")


def _video_path(dataset_root: Path, episode: Mapping[str, object], image_key: str) -> Path:
    chunk = episode.get(f"videos/{image_key}/chunk_index")
    file_index = episode.get(f"videos/{image_key}/file_index")
    if not _is_int(chunk) or not _is_int(file_index):
        raise C0PostCaptureReviewError(f"video mapping missing for {image_key}")
    path = dataset_root / "videos" / image_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
    if not path.is_file():
        raise C0PostCaptureReviewError(f"video file missing: {path}")
    return path


def _decode_video_frames(video_path: Path, query_timestamps: Sequence[float], fps: float):
    import av

    requested = tuple(float(value) for value in query_timestamps)
    nearest: list[tuple[float, object] | None] = [None] * len(requested)
    with av.open(str(video_path), mode="r") as container:
        if not container.streams.video:
            raise C0PostCaptureReviewError(f"video stream missing: {video_path}")
        stream = container.streams.video[0]
        for ordinal, frame in enumerate(container.decode(stream)):
            timestamp = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None and frame.time_base is not None
                else ordinal / fps
            )
            for query_index, query in enumerate(requested):
                distance = abs(timestamp - query)
                current = nearest[query_index]
                if current is None or distance < current[0]:
                    nearest[query_index] = (distance, frame.to_image())
            if timestamp > max(requested) + 1.0 / fps:
                break
    tolerance = 0.5 / fps + 1e-6
    if any(item is None or item[0] > tolerance for item in nearest):
        raise C0PostCaptureReviewError(f"video frame timestamp exceeds tolerance: {video_path}")
    return [item[1] for item in nearest if item is not None]


def _write_contact_sheet(
    path: Path,
    frames: Sequence[object],
    selected_frames: Sequence[tuple[str, int]],
    image_key: str,
) -> None:
    from PIL import Image, ImageDraw

    if len(frames) != len(selected_frames):
        raise C0PostCaptureReviewError("decoded frame count does not match selected frame count")
    cell_width, image_height, label_height = 320, 240, 28
    sheet = Image.new("RGB", (cell_width * 3, (image_height + label_height) * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (frame, (label, frame_index)) in enumerate(zip(frames, selected_frames, strict=True)):
        if not isinstance(frame, Image.Image):
            raise C0PostCaptureReviewError("decoded frame is not a PIL image")
        x = (index % 3) * cell_width
        y = (index // 3) * (image_height + label_height)
        resized = frame.convert("RGB").resize((cell_width, image_height), Image.Resampling.LANCZOS)
        sheet.paste(resized, (x, y + label_height))
        draw.text((x + 6, y + 7), f"{label} | frame {frame_index}", fill="black")
    draw.rectangle((0, 0, sheet.width - 1, sheet.height - 1), outline="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", optimize=True)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _review_checklist(manifest: C0ReviewManifestV1) -> str:
    lines = [
        "# C0 Human Review Checklist",
        "",
        "Inspect both contact sheets for every episode. DO/controller evidence only proves "
        "the commanded output; it does not prove that a strip was physically grasped.",
        "",
        "Choose exactly one outcome per episode and add a short note. Keep `uncertain` "
        "when the visual evidence is insufficient.",
        "",
        "| Episode | Split | ON | OFF | Source outcome | Human outcome | Note |",
        "|---:|---|---:|---:|---|---|---|",
    ]
    for episode in manifest.episodes:
        lines.append(
            f"| {episode.aggregate_episode_index} | {episode.split_name} | "
            f"{episode.gripper_on_frame} | {episode.gripper_off_frame} | "
            f"{episode.source_human_outcome} | ☐ success ☐ failure ☐ uncertain | |"
        )
    lines.extend(
        [
            "",
            "## Unresolved blockers",
            "",
            "- Human outcomes remain pending until this checklist is completed.",
            "- VLM shadow review has not been run.",
            "- Canonical full LeRobot episode digests are not bound.",
            "- Final C0EpisodeManifestV1 and C0BatchManifestV1 are not generated.",
            "- Training, inference, and policy execution remain unauthorized.",
            "",
            "The aggregate LeRobot metadata says `train: 0:10`; this review pack uses the "
            "capture-sidecar 8 train / 2 validation plan and records that distinction explicitly.",
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest_sha256(output_root: Path) -> None:
    """Hash every regular output file except MANIFEST.sha256 itself."""

    manifest_path = output_root / "MANIFEST.sha256"
    files = sorted(path for path in output_root.rglob("*") if path.is_file() and path != manifest_path)
    lines = [f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}" for path in files]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_manifest_sha256(output_root: Path) -> bool:
    """Recompute the evidence-pack checksums without changing any file."""

    manifest_path = output_root / "MANIFEST.sha256"
    if not manifest_path.is_file():
        return False
    seen: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if "  " not in line:
            return False
        expected, relative = line.split("  ", 1)
        if relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            return False
        seen.add(relative)
        path = output_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            return False
    expected_files = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    return seen == expected_files


def build_c0_review_pack(
    *, repo_root: Path, dataset_root: Path, output_root: Path
) -> C0ReviewManifestV1:
    """Validate the ten episodes and write a new, non-overwriting review pack."""

    repo_root = repo_root.resolve()
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"review output already exists: {output_root}")
    if not dataset_root.is_dir():
        raise C0PostCaptureReviewError(f"dataset root missing: {dataset_root}")
    for protected in (dataset_root, *(repo_root / source.source_dataset_ref for source in C0_REVIEW_SOURCES)):
        protected = protected.resolve()
        if output_root == protected or protected in output_root.parents:
            raise C0PostCaptureReviewError("review output must not be inside a source dataset")

    info = _read_json_object(dataset_root / "meta" / "info.json")
    if info.get("robot_type") != "aubo_i10" or info.get("fps") != 25:
        raise C0PostCaptureReviewError("aggregate robot_type/fps mismatch")
    if info.get("total_episodes") != 10 or info.get("total_frames") != 8324:
        raise C0PostCaptureReviewError("aggregate episode/frame totals mismatch")
    if info.get("splits") != {"train": "0:10"}:
        raise C0PostCaptureReviewError("aggregate split metadata changed; review policy must be reconsidered")
    features = info.get("features")
    if not isinstance(features, dict) or tuple(
        key for key in C0_REVIEW_IMAGE_KEYS if key in features
    ) != C0_REVIEW_IMAGE_KEYS:
        raise C0PostCaptureReviewError("aggregate dataset lacks the two frozen RGB streams")

    aggregate_meta = _episode_metadata(dataset_root)
    if sorted(aggregate_meta) != list(range(10)):
        raise C0PostCaptureReviewError("aggregate episode metadata must be exactly 0..9")

    prepared: list[dict[str, object]] = []
    episode_ids: set[str] = set()
    scene_ids: set[str] = set()
    for source in C0_REVIEW_SOURCES:
        source_root = (repo_root / source.source_dataset_ref).resolve()
        sidecar_path = (repo_root / source.capture_sidecar_ref).resolve()
        if not source_root.is_dir() or not sidecar_path.is_file():
            raise C0PostCaptureReviewError("explicit source dataset or sidecar is missing")
        source_meta = _episode_metadata(source_root)
        if source.source_episode_index not in source_meta:
            raise C0PostCaptureReviewError("explicit source episode is missing")
        aggregate_episode = aggregate_meta[source.aggregate_episode_index]
        source_episode = source_meta[source.source_episode_index]
        frame_count = aggregate_episode.get("length")
        if not _is_int(frame_count) or frame_count != source_episode.get("length"):
            raise C0PostCaptureReviewError("aggregate/source episode length mismatch")

        aggregate_rows = _episode_rows(dataset_root, source.aggregate_episode_index)
        source_rows = _episode_rows(source_root, source.source_episode_index)
        _validate_source_slice(aggregate_rows, source_rows)
        if len(aggregate_rows["frame_index"]) != frame_count:
            raise C0PostCaptureReviewError("episode metadata/data row count mismatch")

        sidecar = _read_json_object(sidecar_path)
        episode_id, scene_id, split_name, notes = _validate_sidecar(sidecar, source, frame_count)
        if episode_id in episode_ids or scene_id in scene_ids:
            raise C0PostCaptureReviewError("episode_id and scene_id must be unique")
        episode_ids.add(episode_id)
        scene_ids.add(scene_id)

        action_names = (
            features.get("action", {}).get("names")
            if isinstance(features.get("action"), dict)
            else None
        )
        state_feature = features.get("observation.state")
        state_names = state_feature.get("names") if isinstance(state_feature, dict) else None
        if not isinstance(action_names, list) or not isinstance(state_names, list):
            raise C0PostCaptureReviewError("action/state feature names are missing")
        try:
            action_gripper_index = action_names.index("ee.gripper_pos")
            state_gripper_index = state_names.index("gripper_pos")
        except ValueError as exc:
            raise C0PostCaptureReviewError("gripper feature name is missing") from exc
        action_gripper = aggregate_rows["action"][:, action_gripper_index]
        state_gripper = aggregate_rows["observation.state"][:, state_gripper_index]
        if not set(action_gripper).issubset({0.0, 100.0}) or not set(state_gripper).issubset({0.0, 100.0}):
            raise C0PostCaptureReviewError("gripper values must be binary 0/100")
        on_frames, off_frames = derive_gripper_transition_frames(action_gripper, state_gripper)
        if len(on_frames) != 1 or len(off_frames) != 1:
            raise C0PostCaptureReviewError("episode must contain exactly one ON and one OFF transition")
        gripper_on_frame, gripper_off_frame = on_frames[0], off_frames[0]
        _validate_controller_attempts(sidecar, gripper_on_frame, gripper_off_frame)
        selected_frames = select_review_frames(frame_count, gripper_on_frame, gripper_off_frame)
        prepared.append(
            {
                "source": source,
                "episode_meta": aggregate_episode,
                "frame_count": frame_count,
                "episode_id": episode_id,
                "scene_id": scene_id,
                "split_name": split_name,
                "notes": notes,
                "sidecar": sidecar,
                "sidecar_sha256": sha256_file(sidecar_path),
                "gripper_on_frame": gripper_on_frame,
                "gripper_off_frame": gripper_off_frame,
                "selected_frames": selected_frames,
            }
        )

    if sum(int(item["frame_count"]) for item in prepared) != 8324:
        raise C0PostCaptureReviewError("prepared episode frame total is not 8324")

    output_root.mkdir(parents=True, exist_ok=False)
    records: list[C0ReviewEpisodeV1] = []
    for item in prepared:
        source = item["source"]
        assert isinstance(source, C0ReviewSourceV1)
        episode_meta = item["episode_meta"]
        selected_frames = item["selected_frames"]
        assert isinstance(episode_meta, dict) and isinstance(selected_frames, tuple)
        output_episode_dir = output_root / "episodes" / f"episode-{source.aggregate_episode_index:02d}"
        keyframe_files: list[tuple[str, str, str]] = []
        indices = [index for _, index in selected_frames]
        for image_key in C0_REVIEW_IMAGE_KEYS:
            from_timestamp = episode_meta.get(f"videos/{image_key}/from_timestamp")
            if not isinstance(from_timestamp, (int, float)) or isinstance(from_timestamp, bool):
                raise C0PostCaptureReviewError(f"video from_timestamp missing for {image_key}")
            timestamps = [float(from_timestamp) + index / 25.0 for index in indices]
            frames = _decode_video_frames(
                _video_path(dataset_root, episode_meta, image_key), timestamps, 25.0
            )
            short_name = image_key.removeprefix("observation.images.")
            image_path = output_episode_dir / f"{short_name}_contact_sheet.png"
            _write_contact_sheet(image_path, frames, selected_frames, image_key)
            image_ref = image_path.relative_to(output_root).as_posix()
            keyframe_files.append((image_key, image_ref, sha256_file(image_path)))

        sidecar = item["sidecar"]
        assert isinstance(sidecar, dict)
        record = C0ReviewEpisodeV1(
            aggregate_episode_index=source.aggregate_episode_index,
            source_dataset_ref=source.source_dataset_ref,
            source_episode_index=source.source_episode_index,
            formal_episode_index=source.formal_episode_index,
            episode_id=str(item["episode_id"]),
            scene_id=str(item["scene_id"]),
            split_name=str(item["split_name"]),
            frame_count=int(item["frame_count"]),
            fps=25.0,
            gripper_on_frame=int(item["gripper_on_frame"]),
            gripper_off_frame=int(item["gripper_off_frame"]),
            selected_frames=selected_frames,
            keyframe_files=tuple(keyframe_files),
            capture_sidecar_ref=source.capture_sidecar_ref,
            capture_sidecar_sha256=str(item["sidecar_sha256"]),
            source_human_outcome=str(sidecar["human_outcome"]),
            compatibility_notes=item["notes"],
        )
        records.append(record)
        _write_json(output_episode_dir / "review.json", record.to_manifest_record())

    manifest = C0ReviewManifestV1(
        dataset_ref=f"datasets/{C0_REVIEW_DATASET_NAME}",
        aggregate_dataset_splits=info["splits"],
        episodes=tuple(records),
    )
    _write_json(output_root / "review_manifest.json", manifest.to_manifest_record())
    (output_root / "REVIEW_CHECKLIST.md").write_text(_review_checklist(manifest), encoding="utf-8")
    write_manifest_sha256(output_root)
    if not verify_manifest_sha256(output_root):
        raise C0PostCaptureReviewError("generated MANIFEST.sha256 did not verify")
    return manifest

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

"""Offline C0 episode binding digests and pre-training readiness gate.

The episode digest binds the canonical tabular slice and every physical file
that contains the episode: metadata, parquet, and both MP4 containers. Shared
containers are deliberately hashed in full, so changing an adjacent episode
invalidates every affected binding. This conservative over-coverage avoids
pretending that byte ranges in inter-frame-compressed video are independent.

Importing this module has no hardware, network, model, training, or thread
side effects. The readiness report never grants training or policy execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .c0_post_capture_review import sha256_file, verify_manifest_sha256, write_manifest_sha256
from .contracts import INSTRUCTION_SPECS
from .rgb_gate import FROZEN_CAMERA_SET_V2_SHA256

C0_EPISODE_BINDING_DIGEST_SCHEMA_VERSION: Final = "C0DatasetEpisodeBindingDigestV1"
C0_PRETRAINING_READINESS_SCHEMA_VERSION: Final = "C0PretrainingReadinessV1"
C0_DATASET_NAME: Final = "c0_formal_single_strip_20260921_full"
C0_DATASET_FPS: Final = 25.0
C0_EPISODE_COUNT: Final = 10
C0_TOTAL_FRAMES: Final = 8324
C0_IMAGE_KEYS: Final = (
    "observation.images.global_rgb",
    "observation.images.grasp_rgb",
)


class C0PretrainingGateError(RuntimeError):
    """Raised when C0 evidence cannot be safely frozen or assessed."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise C0PretrainingGateError("digest input is not finite pure JSON data") from exc
    return encoded.encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C0PretrainingGateError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise C0PretrainingGateError(f"JSON root must be an object: {path}")
    return value


@dataclass(frozen=True)
class C0ArtifactFileDigestV1:
    """Hash binding to one physical file that contains episode evidence."""

    role: str
    ref: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("role must be a non-empty string")
        if not isinstance(self.ref, str) or not self.ref or Path(self.ref).is_absolute():
            raise ValueError("ref must be a non-empty relative path")
        if ".." in Path(self.ref).parts:
            raise ValueError("ref must not escape the dataset root")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise ValueError("size_bytes must be a positive integer")
        if not _valid_sha256(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")

    def to_record(self) -> dict[str, object]:
        return {
            "role": self.role,
            "ref": self.ref,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class C0VideoSegmentBindingV1:
    """Episode time range within one fully hashed MP4 container."""

    image_key: str
    file_ref: str
    file_sha256: str
    from_timestamp_s: float
    to_timestamp_s: float

    def __post_init__(self) -> None:
        if self.image_key not in C0_IMAGE_KEYS:
            raise ValueError(f"image_key must be one of {C0_IMAGE_KEYS}")
        if (
            not isinstance(self.file_ref, str)
            or not self.file_ref
            or Path(self.file_ref).is_absolute()
            or ".." in Path(self.file_ref).parts
        ):
            raise ValueError("file_ref must be a relative path inside the dataset root")
        if not _valid_sha256(self.file_sha256):
            raise ValueError("file_sha256 must be a lowercase SHA-256")
        for name in ("from_timestamp_s", "to_timestamp_s"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite numeric")
        if not 0 <= float(self.from_timestamp_s) < float(self.to_timestamp_s):
            raise ValueError("video segment timestamps must be positive and ordered")

    def to_record(self) -> dict[str, object]:
        return {
            "image_key": self.image_key,
            "file_ref": self.file_ref,
            "file_sha256": self.file_sha256,
            "from_timestamp_s": float(self.from_timestamp_s),
            "to_timestamp_s": float(self.to_timestamp_s),
        }


@dataclass(frozen=True)
class C0DatasetEpisodeBindingDigestV1:
    """Canonical digest for one logical episode and all containing files."""

    dataset_ref: str
    episode_index: int
    frame_count: int
    fps: float
    task_texts: Sequence[str]
    tabular_slice_sha256: str
    episode_metadata_sha256: str
    artifact_files: Sequence[C0ArtifactFileDigestV1]
    video_segments: Sequence[C0VideoSegmentBindingV1]
    schema_version: str = C0_EPISODE_BINDING_DIGEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != C0_EPISODE_BINDING_DIGEST_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {C0_EPISODE_BINDING_DIGEST_SCHEMA_VERSION!r}")
        if not isinstance(self.dataset_ref, str) or not self.dataset_ref:
            raise ValueError("dataset_ref must be a non-empty string")
        for name in ("episode_index", "frame_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.frame_count <= 0
            or isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or float(self.fps) != C0_DATASET_FPS
        ):
            raise ValueError("frame_count must be positive and fps must be 25")
        object.__setattr__(self, "fps", float(self.fps))
        tasks = tuple(self.task_texts)
        if tasks != (INSTRUCTION_SPECS["pick_any_collection"].text,):
            raise ValueError("task_texts must contain only the canonical C0 instruction")
        object.__setattr__(self, "task_texts", tasks)
        for name in ("tabular_slice_sha256", "episode_metadata_sha256"):
            if not _valid_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        artifacts = tuple(self.artifact_files)
        if not artifacts or not all(isinstance(item, C0ArtifactFileDigestV1) for item in artifacts):
            raise ValueError("artifact_files must contain file digest records")
        if [item.ref for item in artifacts] != sorted(item.ref for item in artifacts):
            raise ValueError("artifact_files must be sorted by ref")
        object.__setattr__(self, "artifact_files", artifacts)
        videos = tuple(self.video_segments)
        if tuple(item.image_key for item in videos) != C0_IMAGE_KEYS:
            raise ValueError("video_segments must contain the two frozen streams in order")
        object.__setattr__(self, "video_segments", videos)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_ref": self.dataset_ref,
            "episode_index": self.episode_index,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "task_texts": list(self.task_texts),
            "tabular_slice_sha256": self.tabular_slice_sha256,
            "episode_metadata_sha256": self.episode_metadata_sha256,
            "artifact_files": [item.to_record() for item in self.artifact_files],
            "video_segments": [item.to_record() for item in self.video_segments],
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.payload())

    def to_manifest_record(self) -> dict[str, object]:
        record = {
            **self.payload(),
            "sha256": self.sha256,
            "digest_scope": "episode_slice_plus_full_hashes_of_containing_files",
            "covers_full_tabular_episode": True,
            "covers_video_content": True,
            "video_binding_method": "full_mp4_sha256_plus_episode_time_range",
            "shared_file_overcoverage": True,
            "physical_grasp_success_proven": False,
            "training_authorized": False,
            "policy_execution_authorized": False,
            "hardware_access_performed": False,
        }
        return copy.deepcopy(record)


def _episode_metadata(dataset_root: Path, episode_index: int) -> tuple[dict[str, object], Path]:
    import pyarrow.parquet as pq

    matches: list[tuple[dict[str, object], Path]] = []
    for path in sorted((dataset_root / "meta" / "episodes").rglob("*.parquet")):
        for row in pq.read_table(path, use_threads=False).to_pylist():
            if row.get("episode_index") == episode_index:
                matches.append((row, path))
    if len(matches) != 1:
        raise C0PretrainingGateError(f"episode {episode_index} metadata match count is {len(matches)}")
    return matches[0]


def _episode_tabular_slice(
    data_path: Path, episode_index: int, frame_count: int
) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(data_path, use_threads=False)
    if "episode_index" not in table.column_names or "frame_index" not in table.column_names:
        raise C0PretrainingGateError("data parquet lacks episode_index or frame_index")
    filtered = table.filter(pc.equal(table["episode_index"], pa.scalar(episode_index)))
    if filtered.num_rows != frame_count:
        raise C0PretrainingGateError("data slice row count does not match episode metadata")
    frame_indices = filtered["frame_index"].to_pylist()
    order = sorted(range(len(frame_indices)), key=frame_indices.__getitem__)
    sorted_table = filtered.take(pa.array(order, type=pa.int64()))
    sorted_indices = sorted_table["frame_index"].to_pylist()
    if sorted_indices != list(range(frame_count)):
        raise C0PretrainingGateError("episode frame_index must be exactly 0..frame_count-1")
    return {
        "column_names": sorted_table.column_names,
        "rows": sorted_table.to_pylist(),
    }


def compute_c0_dataset_episode_digest(
    dataset_root: Path,
    episode_index: int,
    *,
    dataset_ref: str = f"datasets/{C0_DATASET_NAME}",
) -> C0DatasetEpisodeBindingDigestV1:
    """Compute one read-only, fail-closed C0 episode binding digest."""

    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise C0PretrainingGateError(f"dataset root missing: {dataset_root}")
    if (
        isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or not 0 <= episode_index < C0_EPISODE_COUNT
    ):
        raise ValueError("episode_index must be an integer in 0..9")
    info_path = dataset_root / "meta" / "info.json"
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    stats_path = dataset_root / "meta" / "stats.json"
    info = _read_json_object(info_path)
    if (
        info.get("robot_type") != "aubo_i10"
        or info.get("fps") != 25
        or info.get("total_episodes") != C0_EPISODE_COUNT
        or info.get("total_frames") != C0_TOTAL_FRAMES
    ):
        raise C0PretrainingGateError("dataset identity/totals do not match formal C0")
    episode, episode_meta_path = _episode_metadata(dataset_root, episode_index)
    frame_count = episode.get("length")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise C0PretrainingGateError("episode length must be a positive integer")
    tasks = episode.get("tasks")
    if tasks != [INSTRUCTION_SPECS["pick_any_collection"].text]:
        raise C0PretrainingGateError("episode task text does not match the C0 instruction")

    data_chunk = episode.get("data/chunk_index")
    data_file = episode.get("data/file_index")
    if (
        isinstance(data_chunk, bool)
        or not isinstance(data_chunk, int)
        or isinstance(data_file, bool)
        or not isinstance(data_file, int)
    ):
        raise C0PretrainingGateError("episode data file mapping is missing")
    data_path = dataset_root / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet"
    required_files = [info_path, tasks_path, stats_path, episode_meta_path, data_path]
    video_ranges: list[tuple[str, Path, float, float]] = []
    for image_key in C0_IMAGE_KEYS:
        chunk = episode.get(f"videos/{image_key}/chunk_index")
        file_index = episode.get(f"videos/{image_key}/file_index")
        start = episode.get(f"videos/{image_key}/from_timestamp")
        end = episode.get(f"videos/{image_key}/to_timestamp")
        if (
            isinstance(chunk, bool)
            or not isinstance(chunk, int)
            or isinstance(file_index, bool)
            or not isinstance(file_index, int)
        ):
            raise C0PretrainingGateError(f"video mapping missing for {image_key}")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or not math.isfinite(float(start))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not math.isfinite(float(end))
        ):
            raise C0PretrainingGateError(f"video timestamps missing or invalid for {image_key}")
        video_path = (
            dataset_root
            / "videos"
            / image_key
            / f"chunk-{chunk:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        required_files.append(video_path)
        video_ranges.append((image_key, video_path, float(start), float(end)))

    artifacts: list[C0ArtifactFileDigestV1] = []
    roles = {
        info_path: "dataset_info",
        tasks_path: "task_table",
        stats_path: "dataset_stats",
        episode_meta_path: "episode_metadata_container",
        data_path: "episode_data_container",
    }
    for path in sorted(set(required_files), key=lambda item: item.relative_to(dataset_root).as_posix()):
        if not path.is_file():
            raise C0PretrainingGateError(f"required episode artifact missing: {path}")
        ref = path.relative_to(dataset_root).as_posix()
        role = roles.get(path, "video_container")
        artifacts.append(
            C0ArtifactFileDigestV1(
                role=role,
                ref=ref,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    hashes_by_ref = {item.ref: item.sha256 for item in artifacts}
    video_segments = [
        C0VideoSegmentBindingV1(
            image_key=image_key,
            file_ref=video_path.relative_to(dataset_root).as_posix(),
            file_sha256=hashes_by_ref[video_path.relative_to(dataset_root).as_posix()],
            from_timestamp_s=start,
            to_timestamp_s=end,
        )
        for image_key, video_path, start, end in video_ranges
    ]

    tabular_slice = _episode_tabular_slice(data_path, episode_index, frame_count)
    return C0DatasetEpisodeBindingDigestV1(
        dataset_ref=dataset_ref,
        episode_index=episode_index,
        frame_count=frame_count,
        fps=C0_DATASET_FPS,
        task_texts=tuple(tasks),
        tabular_slice_sha256=_sha256_json(tabular_slice),
        episode_metadata_sha256=_sha256_json(episode),
        artifact_files=tuple(artifacts),
        video_segments=tuple(video_segments),
    )


def build_c0_pretraining_pack(
    *,
    dataset_root: Path,
    review_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Freeze C0 digests and emit the truthful next-stage readiness decision."""

    dataset_root = dataset_root.resolve()
    review_root = review_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"pretraining output already exists: {output_root}")
    if not verify_manifest_sha256(review_root):
        raise C0PretrainingGateError("review MANIFEST.sha256 does not verify")
    review = _read_json_object(review_root / "review_manifest.json")
    episodes = review.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 10:
        raise C0PretrainingGateError("review manifest must contain ten episodes")
    if any(not isinstance(item, dict) or item.get("human_outcome") != "success" for item in episodes):
        raise C0PretrainingGateError("all ten human outcomes must be confirmed success")

    digests = [compute_c0_dataset_episode_digest(dataset_root, index) for index in range(10)]
    output_root.mkdir(parents=True, exist_ok=False)
    digest_refs: list[dict[str, object]] = []
    for digest in digests:
        path = output_root / "episodes" / f"episode-{digest.episode_index:02d}-digest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                digest.to_manifest_record(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        digest_refs.append(
            {
                "episode_index": digest.episode_index,
                "ref": path.relative_to(output_root).as_posix(),
                "evidence_file_sha256": sha256_file(path),
                "dataset_episode_sha256": digest.sha256,
            }
        )

    blockers = [
        "c1_single_strip_angle_coverage_incomplete",
        "c1_two_strip_dataset_missing",
        "vlm_shadow_model_prompt_revision_not_frozen",
        "vlm_shadow_not_run",
        "final_c0_episode_and_batch_manifests_not_generated",
        "aggregate_native_split_is_train_0_10",
        "smolvla_base_weights_not_available_in_local_cache",
    ]
    report: dict[str, object] = {
        "schema_version": C0_PRETRAINING_READINESS_SCHEMA_VERSION,
        "dataset_ref": f"datasets/{C0_DATASET_NAME}",
        "camera_set_v2_sha256": FROZEN_CAMERA_SET_V2_SHA256,
        "c0_episode_count": 10,
        "c0_total_frames": C0_TOTAL_FRAMES,
        "c0_human_success_count": 10,
        "c0_review_manifest_ref": review_root.as_posix(),
        "c0_review_manifest_sha256": sha256_file(review_root / "review_manifest.json"),
        "episode_digest_schema_version": C0_EPISODE_BINDING_DIGEST_SCHEMA_VERSION,
        "episode_digests": digest_refs,
        "canonical_episode_bindings_ready": True,
        "c0_capture_and_human_review_passed": True,
        "go_for_c1_collection": True,
        "training_ready": False,
        "training_decision": "NO_GO",
        "training_blockers": blockers,
        "collection_decision": "GO_C1_ONLY",
        "collection_plan": {
            "existing_single_strip_90_degree_episodes": 10,
            "nominal_route_single_strip_target": 40,
            "nominal_route_two_strip_target": 40,
            "route_conflict": (
                "ten existing 90-degree episodes cannot fit eight angle bins times five within forty"
            ),
            "recommended_single_strip_target": 45,
            "recommended_additional_single_strip_episodes": 35,
            "recommended_single_strip_allocation": (
                "five new episodes at each of the seven non-90-degree centers"
            ),
            "additional_two_strip_episodes": 40,
            "total_additional_episodes_before_first_smolvla_smoke_run": 75,
            "c2_c3_collection_before_first_smoke_run": False,
        },
        "vlm_shadow_status": "blocked_no_frozen_model_prompt_or_local_weights",
        "final_c0_manifests_generated": False,
        "training_authorized": False,
        "inference_authorized": False,
        "policy_execution_authorized": False,
        "hardware_access_performed": False,
    }
    report_path = output_root / "pretraining_readiness.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    readme = (
        "# C0 pre-training gate\n\n"
        "C0 capture and operator-confirmed review pass. The decision is GO only for bounded C1 "
        "collection and NO-GO for training. See `pretraining_readiness.json` for blockers.\n\n"
        "The recommended first-training dataset is 45 single-strip episodes (the existing ten at "
        "90 degrees plus five at each other angle center) and 40 two-strip episodes. C2/C3 data are "
        "not required before the first fixed SmolVLA smoke run.\n"
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    write_manifest_sha256(output_root)
    if not verify_manifest_sha256(output_root):
        raise C0PretrainingGateError("generated pretraining MANIFEST.sha256 did not verify")
    return copy.deepcopy(report)

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

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from lerobot.bamboo_sorting.c0_post_capture_review import (
    C0_REVIEW_EPISODE_SCHEMA_VERSION,
    C0_REVIEW_FRAME_LABELS,
    C0_REVIEW_IMAGE_KEYS,
    C0_REVIEW_MANIFEST_SCHEMA_VERSION,
    C0_REVIEW_SOURCES,
    C0ReviewEpisodeV1,
    C0ReviewManifestV1,
    build_c0_review_pack,
    select_review_frames,
    verify_manifest_sha256,
    write_manifest_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "examples" / "phone_to_auboi10" / "build_c0_review_pack.py"
ZERO_SHA256 = "0" * 64


def _episode(index: int, *, source_outcome: str = "uncertain") -> C0ReviewEpisodeV1:
    split = "train" if index < 8 else "validation"
    source = C0_REVIEW_SOURCES[index]
    return C0ReviewEpisodeV1(
        aggregate_episode_index=index,
        source_dataset_ref=source.source_dataset_ref,
        source_episode_index=source.source_episode_index,
        formal_episode_index=index,
        episode_id=f"episode-{index}",
        scene_id=f"scene-{index}",
        split_name=split,
        frame_count=100,
        fps=25.0,
        gripper_on_frame=20,
        gripper_off_frame=70,
        selected_frames=select_review_frames(100, 20, 70),
        keyframe_files=tuple(
            (key, f"episodes/episode-{index:02d}/{key.rsplit('.', 1)[-1]}.png", ZERO_SHA256)
            for key in C0_REVIEW_IMAGE_KEYS
        ),
        capture_sidecar_ref=source.capture_sidecar_ref,
        capture_sidecar_sha256=ZERO_SHA256,
        source_human_outcome=source_outcome,
    )


def _manifest() -> C0ReviewManifestV1:
    episodes = [_episode(0, source_outcome="success")]
    episodes.extend(_episode(index) for index in range(1, 10))
    return C0ReviewManifestV1(
        dataset_ref="datasets/c0_formal_single_strip_20260921_full",
        aggregate_dataset_splits={"train": "0:10"},
        episodes=episodes,
    )


def test_explicit_source_mapping_excludes_run03_and_smoke() -> None:
    assert [item.aggregate_episode_index for item in C0_REVIEW_SOURCES] == list(range(10))
    assert [item.formal_episode_index for item in C0_REVIEW_SOURCES] == list(range(10))
    refs = " ".join(item.source_dataset_ref for item in C0_REVIEW_SOURCES)
    assert "run03" not in refs
    assert "smoke" not in refs
    assert [item.source_episode_index for item in C0_REVIEW_SOURCES] == [0, 0, *range(8)]


def test_select_review_frames_uses_canonical_order_and_safe_clipping() -> None:
    selected = select_review_frames(10, 0, 8)
    assert tuple(label for label, _ in selected) == C0_REVIEW_FRAME_LABELS
    assert tuple(index for _, index in selected) == (0, 0, 0, 1, 4, 7, 8, 9, 9)
    assert all(0 <= index < 10 for _, index in selected)


@pytest.mark.parametrize(
    ("frame_count", "on_frame", "off_frame"),
    [(0, 0, 1), (10, -1, 5), (10, 5, 5), (10, 7, 6), (10, 1, 10)],
)
def test_select_review_frames_rejects_invalid_boundaries(
    frame_count: int, on_frame: int, off_frame: int
) -> None:
    with pytest.raises(ValueError):
        select_review_frames(frame_count, on_frame, off_frame)


def test_episode_record_preserves_source_outcome_but_review_stays_pending() -> None:
    record = _episode(0, source_outcome="success").to_manifest_record()
    assert record["schema_version"] == C0_REVIEW_EPISODE_SCHEMA_VERSION
    assert record["source_human_outcome"] == "success"
    assert record["human_outcome"] == "pending"
    assert record["vlm_shadow_status"] == "not_run"
    assert record["final_manifest_bound"] is False
    assert record["training_authorized"] is False
    assert record["policy_execution_authorized"] is False
    assert record["hardware_access_performed"] is False
    assert record["selected_frame_indices"] == [index for _, index in _episode(0).selected_frames]


def test_episode_manifest_record_is_deep_copy_isolated() -> None:
    episode = _episode(0)
    pristine = episode.to_manifest_record()
    mutated = episode.to_manifest_record()
    mutated["selected_frames"]["start"] = 99
    mutated["keyframe_files"][C0_REVIEW_IMAGE_KEYS[0]]["sha256"] = "f" * 64
    assert episode.to_manifest_record() == pristine


def test_review_manifest_validates_ten_unique_episodes_and_split() -> None:
    manifest = _manifest()
    record = manifest.to_manifest_record()
    assert record["schema_version"] == C0_REVIEW_MANIFEST_SCHEMA_VERSION
    assert record["episode_count"] == 10
    assert record["review_splits"] == {"train": list(range(8)), "validation": [8, 9]}
    assert record["aggregate_dataset_splits"] == {"train": "0:10"}
    assert record["final_c0_batch_manifest_generated"] is False
    assert record["inference_authorized"] is False


@pytest.mark.parametrize(
    "episodes",
    [
        [_episode(index) for index in range(9)],
        [*[_episode(index) for index in range(9)], replace(_episode(9), scene_id="scene-0")],
        [*[_episode(index) for index in range(8)], replace(_episode(8), split_name="train"), _episode(9)],
    ],
)
def test_review_manifest_rejects_count_identity_or_split_errors(
    episodes: list[C0ReviewEpisodeV1],
) -> None:
    with pytest.raises(ValueError):
        C0ReviewManifestV1(
            dataset_ref="datasets/c0_formal_single_strip_20260921_full",
            aggregate_dataset_splits={"train": "0:10"},
            episodes=episodes,
        )


def test_review_manifest_record_is_deep_copy_isolated() -> None:
    manifest = _manifest()
    pristine = manifest.to_manifest_record()
    mutated = manifest.to_manifest_record()
    mutated["episodes"][0]["compatibility_notes"].append("changed")
    mutated["review_splits"]["train"].append(99)
    assert manifest.to_manifest_record() == pristine


def test_manifest_sha256_is_recomputable_and_detects_changes(tmp_path: Path) -> None:
    (tmp_path / "episodes").mkdir()
    (tmp_path / "review_manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "episodes" / "review.json").write_text('{"human_outcome":"pending"}\n', encoding="utf-8")
    write_manifest_sha256(tmp_path)
    assert verify_manifest_sha256(tmp_path)

    (tmp_path / "episodes" / "review.json").write_text('{"human_outcome":"success"}\n', encoding="utf-8")
    assert not verify_manifest_sha256(tmp_path)


def test_builder_refuses_existing_output_before_reading_sources(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        build_c0_review_pack(
            repo_root=tmp_path,
            dataset_root=tmp_path / "missing-dataset",
            output_root=output,
        )
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_imports_do_not_open_sockets_start_threads_or_load_hardware_modules() -> None:
    probe = r"""
import json
import os
import sys
import threading

import lerobot.bamboo_sorting.c0_post_capture_review  # noqa: F401

forbidden = [name for name in sys.modules if name.split('.')[0] in {'cv2', 'pyaubo_sdk'}]
threads = [thread.name for thread in threading.enumerate() if thread is not threading.main_thread()]
sockets = []
for fd in os.listdir('/proc/self/fd'):
    path = f'/proc/self/fd/{fd}'
    if os.path.islink(path) and 'socket' in os.readlink(path):
        sockets.append(fd)
print(json.dumps({'forbidden': forbidden, 'threads': threads, 'sockets': sockets}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload == {"forbidden": [], "threads": [], "sockets": []}


def test_build_script_import_is_hardware_free() -> None:
    probe = f"""
import importlib.util
import json
import sys
import threading

spec = importlib.util.spec_from_file_location('build_c0_review_pack_under_test', {str(BUILD_SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
forbidden = [name for name in sys.modules if name.split('.')[0] in {{'cv2', 'pyaubo_sdk'}}]
threads = [thread.name for thread in threading.enumerate() if thread is not threading.main_thread()]
print(json.dumps({{'forbidden': forbidden, 'threads': threads}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"forbidden": [], "threads": []}

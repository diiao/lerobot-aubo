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
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from lerobot.bamboo_sorting.c0_pretraining_gate import (
    C0_IMAGE_KEYS,
    C0ArtifactFileDigestV1,
    C0DatasetEpisodeBindingDigestV1,
    C0VideoSegmentBindingV1,
    build_c0_pretraining_pack,
    compute_c0_dataset_episode_digest,
)
from lerobot.bamboo_sorting.contracts import INSTRUCTION_SPECS

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "examples" / "phone_to_auboi10" / "build_c0_pretraining_pack.py"
FORMAL_DATASET = REPO_ROOT / "datasets" / "c0_formal_single_strip_20260921_full"
ZERO_SHA256 = "0" * 64


def _digest() -> C0DatasetEpisodeBindingDigestV1:
    videos = tuple(
        C0VideoSegmentBindingV1(
            image_key=image_key,
            file_ref=f"videos/{image_key}/chunk-000/file-000.mp4",
            file_sha256=ZERO_SHA256,
            from_timestamp_s=0.0,
            to_timestamp_s=1.0,
        )
        for image_key in C0_IMAGE_KEYS
    )
    return C0DatasetEpisodeBindingDigestV1(
        dataset_ref="datasets/c0_formal_single_strip_20260921_full",
        episode_index=0,
        frame_count=25,
        fps=25,
        task_texts=(INSTRUCTION_SPECS["pick_any_collection"].text,),
        tabular_slice_sha256=ZERO_SHA256,
        episode_metadata_sha256=ZERO_SHA256,
        artifact_files=(
            C0ArtifactFileDigestV1(
                role="dataset_info",
                ref="meta/info.json",
                size_bytes=1,
                sha256=ZERO_SHA256,
            ),
        ),
        video_segments=videos,
    )


def test_episode_binding_digest_is_deterministic_and_deep_copy_isolated() -> None:
    digest = _digest()
    assert digest.sha256 == _digest().sha256
    pristine = digest.to_manifest_record()
    mutated = digest.to_manifest_record()
    mutated["artifact_files"][0]["sha256"] = "f" * 64
    assert digest.to_manifest_record() == pristine

    changed = replace(digest, tabular_slice_sha256="f" * 64)
    assert changed.sha256 != digest.sha256


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"file_ref": "../escape.mp4"}, "relative path"),
        ({"from_timestamp_s": float("nan")}, "finite numeric"),
        ({"to_timestamp_s": 0.0}, "positive and ordered"),
    ],
)
def test_video_binding_rejects_unsafe_or_invalid_ranges(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "image_key": C0_IMAGE_KEYS[0],
        "file_ref": "videos/stream/chunk-000/file-000.mp4",
        "file_sha256": ZERO_SHA256,
        "from_timestamp_s": 0.0,
        "to_timestamp_s": 1.0,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        C0VideoSegmentBindingV1(**values)


def test_builder_refuses_existing_output_before_reading_inputs(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        build_c0_pretraining_pack(
            dataset_root=tmp_path / "missing-dataset",
            review_root=tmp_path / "missing-review",
            output_root=output,
        )
    assert marker.read_text(encoding="utf-8") == "preserve"


@pytest.mark.skipif(not FORMAL_DATASET.is_dir(), reason="formal local C0 dataset is not present")
def test_formal_episode_zero_digest_is_recomputable() -> None:
    first = compute_c0_dataset_episode_digest(FORMAL_DATASET, 0)
    second = compute_c0_dataset_episode_digest(FORMAL_DATASET, 0)
    assert first.sha256 == second.sha256
    assert first.frame_count == 793
    assert tuple(segment.image_key for segment in first.video_segments) == C0_IMAGE_KEYS
    assert {artifact.role for artifact in first.artifact_files} >= {
        "dataset_info",
        "task_table",
        "dataset_stats",
        "episode_metadata_container",
        "episode_data_container",
        "video_container",
    }


def test_imports_do_not_open_sockets_start_threads_or_load_hardware_modules() -> None:
    probe = r"""
import json
import os
import sys
import threading

import lerobot.bamboo_sorting.c0_pretraining_gate  # noqa: F401

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
    assert json.loads(result.stdout) == {"forbidden": [], "threads": [], "sockets": []}


def test_build_script_import_is_hardware_free() -> None:
    probe = f"""
import importlib.util
import json
import sys
import threading

spec = importlib.util.spec_from_file_location('build_c0_pretraining_pack_under_test', {str(BUILD_SCRIPT)!r})
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

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

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.bamboo_sorting.c0_smoke_capture import (
    C0_SMOKE_CAMERA_DEVICES,
    C0_SMOKE_DATASET_FPS,
    C0_SMOKE_IMAGE_FEATURE_KEYS,
    C0_SMOKE_IMAGE_KEYS,
    C0_SMOKE_LEGACY_ACT_KEYS,
    C0_SMOKE_NUM_EPISODES,
    C0_SMOKE_STATUS,
    C0_SMOKE_TASK_TEXT,
    C0SmokeCaptureError,
    build_operator_checklist,
    c0_smoke_robot_observation_features,
    frozen_c0_smoke_camera_mapping,
    maybe_run_c0_smoke_postflight,
    require_new_dataset_root,
    require_operator_confirmation,
    require_placement_reference,
    run_c0_smoke_postflight,
    shutdown_c0_smoke_session,
)
from lerobot.bamboo_sorting.contracts import ACTION_FIELD_NAMES
from lerobot.bamboo_sorting.observation_contract import OBSERVATION_STATE_FIELD_NAMES
from lerobot.bamboo_sorting.rgb_gate import FROZEN_CAMERA_SET_V2_SHA256

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_PY = REPO_ROOT / "examples" / "phone_to_auboi10" / "record.py"
SMOKE_PY = REPO_ROOT / "examples" / "phone_to_auboi10" / "record_c0_smoke.py"

_IMPORT_PROBE = r"""
import json
import os
import sys
import threading

import lerobot.bamboo_sorting.c0_smoke_capture  # noqa: F401

forbidden_modules = [name for name in sys.modules if name.split(".")[0] in {"cv2", "pyaubo_sdk"}]
extra_threads = [
    thread.name for thread in threading.enumerate() if thread is not threading.main_thread()
]
sockets = []
for fd in os.listdir("/proc/self/fd"):
    path = f"/proc/self/fd/{fd}"
    if os.path.islink(path) and "socket" in os.readlink(path):
        sockets.append(fd)
print(
    json.dumps(
        {
            "forbidden_modules": forbidden_modules,
            "extra_threads": extra_threads,
            "sockets": sockets,
        }
    )
)
"""

_SMOKE_SCRIPT_IMPORT_PROBE = r"""
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path

script = Path(%r)
spec = importlib.util.spec_from_file_location("record_c0_smoke_under_test", script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

forbidden_modules = [name for name in sys.modules if name.split(".")[0] in {"cv2", "pyaubo_sdk"}]
print(json.dumps({"forbidden_modules": forbidden_modules, "status": module.C0_SMOKE_STATUS}))
""" % str(SMOKE_PY)


def _load_smoke_script():
    spec = importlib.util.spec_from_file_location("record_c0_smoke_under_test", SMOKE_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_color_mp4(path: Path, n_frames: int = 3, fps: int = 25) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = 640
    stream.height = 480
    stream.pix_fmt = "yuv420p"
    for index in range(n_frames):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :, 0] = (index * 3) % 255
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _features(
    *,
    image_keys: tuple[str, ...] = C0_SMOKE_IMAGE_FEATURE_KEYS,
    action_names: tuple[str, ...] = ACTION_FIELD_NAMES,
    state_names: tuple[str, ...] = OBSERVATION_STATE_FIELD_NAMES,
) -> dict[str, object]:
    features: dict[str, object] = {
        "action": {
            "dtype": "float32",
            "shape": [len(action_names)],
            "names": list(action_names),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [len(state_names)],
            "names": list(state_names),
        },
    }
    for key in image_keys:
        features[key] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        }
    return features


def _write_smoke_dataset(
    root: Path,
    *,
    fps: float = C0_SMOKE_DATASET_FPS,
    total_episodes: int = 1,
    total_frames: int = 8,
    task: str = C0_SMOKE_TASK_TEXT,
    features: dict[str, object] | None = None,
    write_videos: bool = True,
    image_keys: tuple[str, ...] = C0_SMOKE_IMAGE_FEATURE_KEYS,
) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "features": features if features is not None else _features(image_keys=image_keys),
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_tasks_parquet(
        root,
        {
            "task_index": [0],
            "__index_level_0__": [task],
        },
    )
    episodes_dir = meta / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0], type=pa.int64()),
                "length": pa.array([total_frames], type=pa.int64()),
                "tasks": pa.array([[task]]),
            }
        ),
        episodes_dir / "file-000.parquet",
    )
    if write_videos:
        for key in image_keys:
            _write_color_mp4(root / "videos" / key / "chunk-000" / "file-000.mp4")


def _write_tasks_parquet(
    root: Path,
    columns: dict[str, list[object] | pa.Array],
) -> None:
    arrays: dict[str, pa.Array] = {}
    for name, values in columns.items():
        if isinstance(values, pa.Array):
            arrays[name] = values
            continue
        if name == "task_index":
            arrays[name] = pa.array(values, type=pa.int64())
        else:
            arrays[name] = pa.array(values, type=pa.string())
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), meta / "tasks.parquet")


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
    features = c0_smoke_robot_observation_features()
    assert "global_rgb" in features
    assert "grasp_rgb" in features
    assert "handeye" not in features
    assert "fixed" not in features


def test_canonical_task_text() -> None:
    assert C0_SMOKE_TASK_TEXT == "Pick one strip and place it in the collection area."


def test_default_single_episode_and_25hz() -> None:
    assert C0_SMOKE_NUM_EPISODES == 1
    assert C0_SMOKE_DATASET_FPS == 25
    assert C0_SMOKE_STATUS == (
        "one-episode C0 smoke capture, pending formal final evidence binding"
    )


def test_empty_placement_reference_is_rejected() -> None:
    for value in ("", "   ", None, 0):
        with pytest.raises(C0SmokeCaptureError, match="placement reference"):
            require_placement_reference(value)
    with pytest.raises(C0SmokeCaptureError, match="0°"):
        require_placement_reference("")
    assert require_placement_reference(" scale-line-A ") == "scale-line-A"


def test_existing_dataset_root_is_rejected(tmp_path: Path) -> None:
    existing = tmp_path / "already"
    existing.mkdir()
    with pytest.raises(C0SmokeCaptureError, match="already exists"):
        require_new_dataset_root(existing)
    target = tmp_path / "fresh"
    resolved = require_new_dataset_root(target)
    assert resolved == target.resolve()
    assert not resolved.exists()


def test_postflight_rejects_legacy_handeye_fixed_fields(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    _write_smoke_dataset(
        root,
        image_keys=(
            "observation.images.handeye",
            "observation.images.fixed",
        ),
    )
    with pytest.raises(C0SmokeCaptureError, match="handeye|fixed"):
        run_c0_smoke_postflight(root)


def test_postflight_rejects_wrong_fps_episodes_and_contracts(tmp_path: Path) -> None:
    fps_root = tmp_path / "bad_fps"
    _write_smoke_dataset(fps_root, fps=30)
    with pytest.raises(C0SmokeCaptureError, match="fps"):
        run_c0_smoke_postflight(fps_root)

    count_root = tmp_path / "bad_count"
    _write_smoke_dataset(count_root, total_episodes=10)
    with pytest.raises(C0SmokeCaptureError, match="total_episodes"):
        run_c0_smoke_postflight(count_root)

    action_root = tmp_path / "bad_action"
    _write_smoke_dataset(
        action_root,
        features=_features(action_names=ACTION_FIELD_NAMES[1:]),
    )
    with pytest.raises(C0SmokeCaptureError, match="action"):
        run_c0_smoke_postflight(action_root)

    state_root = tmp_path / "bad_state"
    _write_smoke_dataset(
        state_root,
        features=_features(state_names=OBSERVATION_STATE_FIELD_NAMES[:-1]),
    )
    with pytest.raises(C0SmokeCaptureError, match="observation.state"):
        run_c0_smoke_postflight(state_root)


def test_postflight_accepts_one_episode_c0_contract(tmp_path: Path) -> None:
    root = tmp_path / "ok"
    _write_smoke_dataset(root)
    report = run_c0_smoke_postflight(root)

    assert report["total_episodes"] == 1
    assert report["fps"] == 25.0
    assert report["task"] == C0_SMOKE_TASK_TEXT
    assert report["image_feature_keys"] == list(C0_SMOKE_IMAGE_FEATURE_KEYS)
    assert report["action_names"] == list(ACTION_FIELD_NAMES)
    assert report["observation_state_names"] == list(OBSERVATION_STATE_FIELD_NAMES)
    assert report["episode_frame_count"] > 0
    assert report["gripper_pos_is_physical_feedback"] is False
    assert report["grasp_success_claimed"] is False
    assert report["training_authorized"] is False
    assert report["formal_c0_batch"] is False
    assert report["formal_final_evidence_bound"] is False


def test_postflight_accepts_lerobot_v3_tasks_schema(tmp_path: Path) -> None:
    root = tmp_path / "v3"
    _write_smoke_dataset(root)
    report = run_c0_smoke_postflight(root)
    assert report["task"] == C0_SMOKE_TASK_TEXT


def test_postflight_rejects_wrong_v3_task_text(tmp_path: Path) -> None:
    root = tmp_path / "wrong_v3"
    _write_smoke_dataset(root)
    _write_tasks_parquet(
        root,
        {
            "task_index": [0],
            "__index_level_0__": ["Place the long strip in zone A."],
        },
    )
    with pytest.raises(C0SmokeCaptureError, match="task text must equal"):
        run_c0_smoke_postflight(root)


@pytest.mark.parametrize("value", ["", None])
def test_postflight_rejects_empty_or_null_v3_task_text(tmp_path: Path, value: str | None) -> None:
    root = tmp_path / "empty_v3"
    _write_smoke_dataset(root)
    _write_tasks_parquet(
        root,
        {
            "task_index": [0],
            "__index_level_0__": [value],
        },
    )
    with pytest.raises(C0SmokeCaptureError, match="__index_level_0__"):
        run_c0_smoke_postflight(root)


def test_postflight_rejects_missing_task_text_source(tmp_path: Path) -> None:
    root = tmp_path / "no_text"
    _write_smoke_dataset(root)
    _write_tasks_parquet(root, {"task_index": [0]})
    with pytest.raises(C0SmokeCaptureError, match="task text source"):
        run_c0_smoke_postflight(root)


def test_postflight_rejects_missing_task_index(tmp_path: Path) -> None:
    root = tmp_path / "no_task_index"
    _write_smoke_dataset(root)
    _write_tasks_parquet(root, {"__index_level_0__": [C0_SMOKE_TASK_TEXT]})
    with pytest.raises(C0SmokeCaptureError, match="task_index column"):
        run_c0_smoke_postflight(root)


@pytest.mark.parametrize(
    ("task_indexes", "message"),
    [
        (pa.array([None], type=pa.int64()), "Python integer"),
        (pa.array([-1], type=pa.int64()), "greater than or equal to 0"),
        (pa.array([True], type=pa.bool_()), "Python integer"),
    ],
)
def test_postflight_rejects_invalid_task_index(
    tmp_path: Path,
    task_indexes: pa.Array,
    message: str,
) -> None:
    root = tmp_path / "invalid_task_index"
    _write_smoke_dataset(root)
    _write_tasks_parquet(
        root,
        {
            "task_index": task_indexes,
            "__index_level_0__": [C0_SMOKE_TASK_TEXT],
        },
    )
    with pytest.raises(C0SmokeCaptureError, match=message):
        run_c0_smoke_postflight(root)


def test_postflight_rejects_duplicate_task_index_with_same_text(tmp_path: Path) -> None:
    root = tmp_path / "duplicate_same"
    _write_smoke_dataset(root)
    _write_tasks_parquet(
        root,
        {
            "task_index": [0, 0],
            "__index_level_0__": [C0_SMOKE_TASK_TEXT, C0_SMOKE_TASK_TEXT],
        },
    )
    with pytest.raises(C0SmokeCaptureError, match="duplicate task_index"):
        run_c0_smoke_postflight(root)


def test_postflight_rejects_duplicate_task_index_with_different_text(tmp_path: Path) -> None:
    root = tmp_path / "duplicate_different"
    _write_smoke_dataset(root)
    _write_tasks_parquet(
        root,
        {
            "task_index": [0, 0],
            "__index_level_0__": [
                C0_SMOKE_TASK_TEXT,
                "Place the long strip in zone A.",
            ],
        },
    )
    with pytest.raises(C0SmokeCaptureError, match="contradictory task_index"):
        run_c0_smoke_postflight(root)


def test_postflight_rejects_same_text_for_different_task_indexes(tmp_path: Path) -> None:
    root = tmp_path / "same_text_different_indexes"
    _write_smoke_dataset(root)
    _write_tasks_parquet(
        root,
        {
            "task_index": [0, 1],
            "__index_level_0__": [C0_SMOKE_TASK_TEXT, C0_SMOKE_TASK_TEXT],
        },
    )
    with pytest.raises(C0SmokeCaptureError, match="same task text"):
        run_c0_smoke_postflight(root)


def test_postflight_accepts_legacy_task_column(tmp_path: Path) -> None:
    root = tmp_path / "legacy_task"
    _write_smoke_dataset(root)
    _write_tasks_parquet(
        root,
        {
            "task_index": [0],
            "task": [C0_SMOKE_TASK_TEXT],
        },
    )
    report = run_c0_smoke_postflight(root)
    assert report["task"] == C0_SMOKE_TASK_TEXT


def test_postflight_accepts_matching_v3_and_legacy_task_columns(tmp_path: Path) -> None:
    root = tmp_path / "both_match"
    _write_smoke_dataset(root)
    _write_tasks_parquet(
        root,
        {
            "task_index": [0],
            "__index_level_0__": [C0_SMOKE_TASK_TEXT],
            "task": [C0_SMOKE_TASK_TEXT],
        },
    )
    report = run_c0_smoke_postflight(root)
    assert report["task"] == C0_SMOKE_TASK_TEXT


def test_postflight_rejects_contradictory_v3_and_legacy_task_columns(tmp_path: Path) -> None:
    root = tmp_path / "both_conflict"
    _write_smoke_dataset(root)
    _write_tasks_parquet(
        root,
        {
            "task_index": [0],
            "__index_level_0__": [C0_SMOKE_TASK_TEXT],
            "task": ["Place the long strip in zone A."],
        },
    )
    with pytest.raises(C0SmokeCaptureError, match="disagree"):
        run_c0_smoke_postflight(root)


def test_importing_c0_smoke_module_does_not_touch_hardware() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    state = json.loads(result.stdout.strip().splitlines()[-1])
    assert state["forbidden_modules"] == []
    assert state["extra_threads"] == []
    assert state["sockets"] == []


def test_importing_record_c0_smoke_script_does_not_connect_hardware() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SMOKE_SCRIPT_IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    state = json.loads(result.stdout.strip().splitlines()[-1])
    assert state["forbidden_modules"] == []
    assert state["status"] == C0_SMOKE_STATUS


def test_historical_record_py_behavior_is_unchanged() -> None:
    source = RECORD_PY.read_text(encoding="utf-8")
    assert 'TASK_DESCRIPTION = "抓取竹条"' in source
    assert 'NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "4"))' in source
    assert '"handeye": OpenCVCameraConfig(' in source
    assert '"fixed": OpenCVCameraConfig(' in source
    assert "global_rgb" not in source
    assert "grasp_rgb" not in source
    assert "C0_SMOKE" not in source


def test_smoke_cli_rejects_empty_placement_and_existing_root(tmp_path: Path) -> None:
    smoke = _load_smoke_script()
    with pytest.raises(C0SmokeCaptureError, match="placement reference"):
        smoke.parse_c0_smoke_args(
            ["--dataset-root", str(tmp_path / "n"), "--placement-reference", ""]
        )
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(C0SmokeCaptureError, match="already exists"):
        smoke.parse_c0_smoke_args(
            [
                "--dataset-root",
                str(existing),
                "--placement-reference",
                "scale-line-A",
            ]
        )


def test_operator_checklist_uses_physical_scale_not_base_x() -> None:
    mapping = frozen_c0_smoke_camera_mapping()
    dataset_root = Path("/tmp/c0-smoke-does-not-exist")
    lines = build_operator_checklist(
        placement_reference="scale-line-A",
        dataset_root=dataset_root,
        camera_mapping=mapping,
    )
    text = "\n".join(lines)
    assert "operator is on site" in text
    assert "e-stop is available" in text
    assert "the robot workspace is clear of people and obstacles" in text
    assert "collection area is empty" in text
    assert "single strip" in text
    assert "physical placement reference" in text
    assert "exactly 1 episode" in text
    assert "new dataset root" in text
    assert "scale-line-A" in text
    assert str(dataset_root) in text
    assert "Confirm: collection area is empty." in lines
    assert "Confirm: the robot workspace is clear of people and obstacles." in lines
    assert lines.count("Confirm: the robot workspace is clear of people and obstacles.") == 1
    assert "base +X" in text
    assert "GENERAL WEBCAM" in text
    assert "Sonix USB2.0_CAM1" in text
    with pytest.raises(C0SmokeCaptureError):
        require_operator_confirmation("yes")
    require_operator_confirmation("YES")


class _FakeDevice:
    def __init__(self, *, connected: bool = True, error: Exception | None = None) -> None:
        self.is_connected = connected
        self.calls: list[str] = []
        self._error = error

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        if self._error is not None:
            raise self._error


class _FakeListener:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[str] = []
        self._error = error

    def stop(self) -> None:
        self.calls.append("stop")
        if self._error is not None:
            raise self._error


class _FakeDataset:
    def __init__(self, error: Exception | None = None) -> None:
        self.finalize_calls = 0
        self._error = error

    def finalize(self) -> None:
        self.finalize_calls += 1
        if self._error is not None:
            raise self._error


def test_phone_disconnect_error_still_finalizes_and_skips_postflight(tmp_path: Path) -> None:
    dataset = _FakeDataset()
    phone = _FakeDevice(error=RuntimeError("phone disconnect failed"))
    robot = _FakeDevice()
    listener = _FakeListener()
    result = shutdown_c0_smoke_session(
        dataset=dataset, listener=listener, phone=phone, robot=robot
    )
    assert listener.calls == ["stop"]
    assert phone.calls == ["disconnect"]
    assert robot.calls == ["disconnect"]
    assert dataset.finalize_calls == 1
    assert result.dataset_finalize_attempted is True
    assert result.dataset_finalize_succeeded is True
    assert len(result.errors) == 1
    postflight_calls: list[Path] = []
    with pytest.raises(RuntimeError, match="phone disconnect failed"):
        maybe_run_c0_smoke_postflight(
            shutdown_result=result,
            dataset=dataset,
            episode_idx=1,
            dataset_root=tmp_path,
            postflight=lambda root: postflight_calls.append(root) or {},
        )
    assert postflight_calls == []


def test_robot_disconnect_error_still_finalizes_and_skips_postflight(tmp_path: Path) -> None:
    dataset = _FakeDataset()
    phone = _FakeDevice()
    robot = _FakeDevice(error=RuntimeError("robot disconnect failed"))
    result = shutdown_c0_smoke_session(
        dataset=dataset, listener=None, phone=phone, robot=robot
    )
    assert phone.calls == ["disconnect"]
    assert robot.calls == ["disconnect"]
    assert dataset.finalize_calls == 1
    assert result.dataset_finalize_succeeded is True
    postflight_calls: list[Path] = []
    with pytest.raises(RuntimeError, match="robot disconnect failed"):
        maybe_run_c0_smoke_postflight(
            shutdown_result=result,
            dataset=dataset,
            episode_idx=1,
            dataset_root=tmp_path,
            postflight=lambda root: postflight_calls.append(root) or {},
        )
    assert postflight_calls == []


def test_finalize_error_does_not_run_postflight(tmp_path: Path) -> None:
    dataset = _FakeDataset(error=RuntimeError("finalize failed"))
    phone = _FakeDevice()
    robot = _FakeDevice()
    result = shutdown_c0_smoke_session(
        dataset=dataset, listener=None, phone=phone, robot=robot
    )
    assert dataset.finalize_calls == 1
    assert result.dataset_finalize_attempted is True
    assert result.dataset_finalize_succeeded is False
    postflight_calls: list[Path] = []
    with pytest.raises(RuntimeError, match="finalize failed"):
        maybe_run_c0_smoke_postflight(
            shutdown_result=result,
            dataset=dataset,
            episode_idx=1,
            dataset_root=tmp_path,
            postflight=lambda root: postflight_calls.append(root) or {},
        )
    assert postflight_calls == []
    assert result.dataset_finalize_attempted is True

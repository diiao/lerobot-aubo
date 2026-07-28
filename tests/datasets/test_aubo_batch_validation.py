import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _load_aggregate_module():
    script = (
        Path(__file__).parents[2]
        / "examples"
        / "phone_to_auboi10"
        / "aggregate.py"
    )
    spec = importlib.util.spec_from_file_location("aubo_aggregate", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_dataset_root(tmp_path: Path, *, fps: int = 25, gripper_value: float | None = None):
    root = tmp_path / "batch"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    action_names = [
        "ee.j6_target",
        "ee.x",
        "ee.y",
        "ee.z",
        "ee.wx",
        "ee.wy",
        "ee.wz",
        "ee.gripper_pos",
    ]
    info = {
        "fps": fps,
        "total_episodes": 1,
        "total_frames": 20,
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [8],
                "names": action_names,
            },
            "observation.images.handeye": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
            "observation.images.fixed": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))

    actions = np.zeros((20, 8), dtype=np.float32)
    actions[:, 1] = np.arange(20) * 0.001
    actions[:, 7] = np.r_[np.zeros(10), np.full(10, 100.0)]
    if gripper_value is not None:
        actions[5, 7] = gripper_value

    table = pa.table(
        {
            "action": pa.array(
                actions.tolist(),
                type=pa.list_(pa.float32(), len(action_names)),
            ),
            "episode_index": pa.array(np.zeros(20, dtype=np.int64)),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    return root


def test_valid_batch_metadata_and_episode_actions(tmp_path):
    aggregate = _load_aggregate_module()
    root = _make_dataset_root(tmp_path)

    info = aggregate.validate_metadata(root)
    aggregate.validate_gripper_labels(root)
    aggregate.validate_episode_actions(root, info)


def test_old_30_fps_batch_is_rejected(tmp_path):
    aggregate = _load_aggregate_module()
    root = _make_dataset_root(tmp_path, fps=30)

    with pytest.raises(RuntimeError, match="25 FPS"):
        aggregate.validate_metadata(root)


def test_legacy_neutral_gripper_label_is_rejected(tmp_path):
    aggregate = _load_aggregate_module()
    root = _make_dataset_root(tmp_path, gripper_value=50.0)

    with pytest.raises(RuntimeError, match="非 0/100"):
        aggregate.validate_gripper_labels(root)

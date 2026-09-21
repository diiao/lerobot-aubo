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


def _make_dataset_root(
    tmp_path: Path,
    *,
    fps: int = 25,
    gripper_value: float | None = None,
    camera_keys: tuple[str, str] = (
        "observation.images.handeye",
        "observation.images.fixed",
    ),
):
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
            camera_keys[0]: {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
            camera_keys[1]: {
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


def test_c0_profile_accepts_frozen_camera_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_PROFILE", "c0")
    aggregate = _load_aggregate_module()
    root = _make_dataset_root(
        tmp_path,
        camera_keys=(
            "observation.images.global_rgb",
            "observation.images.grasp_rgb",
        ),
    )

    assert aggregate.DATASET_PROFILE == "c0"
    aggregate.validate_metadata(root)


def test_expected_total_episodes_is_explicit_and_positive(monkeypatch):
    aggregate = _load_aggregate_module()
    monkeypatch.setenv("EXPECTED_TOTAL_EPISODES", "10")
    assert aggregate.expected_total_episodes() == 10

    monkeypatch.setenv("EXPECTED_TOTAL_EPISODES", "0")
    with pytest.raises(ValueError, match="positive integer"):
        aggregate.expected_total_episodes()


def test_configured_episode_splits_supports_ten_train_two_validation(monkeypatch):
    aggregate = _load_aggregate_module()
    monkeypatch.setenv("TRAIN_EPISODE_COUNT", "10")
    assert aggregate.configured_episode_splits(12) == {
        "train": "0:10",
        "validation": "10:12",
    }


@pytest.mark.parametrize("value", ["no", "0", "12", "13"])
def test_configured_episode_splits_rejects_invalid_or_empty_validation(
    monkeypatch, value
):
    aggregate = _load_aggregate_module()
    monkeypatch.setenv("TRAIN_EPISODE_COUNT", value)
    with pytest.raises(ValueError, match="TRAIN_EPISODE_COUNT"):
        aggregate.configured_episode_splits(12)


def test_write_aggregate_splits_changes_only_output_info(tmp_path):
    aggregate = _load_aggregate_module()
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "meta").mkdir(parents=True)
    (output / "meta").mkdir(parents=True)
    source_info = {"total_episodes": 10, "splits": {"train": "0:10"}}
    output_info = {"total_episodes": 12, "splits": {"train": "0:12"}}
    (source / "meta" / "info.json").write_text(json.dumps(source_info))
    (output / "meta" / "info.json").write_text(json.dumps(output_info))

    aggregate.write_aggregate_splits(
        output, {"train": "0:10", "validation": "10:12"}
    )

    assert json.loads((source / "meta" / "info.json").read_text()) == source_info
    assert json.loads((output / "meta" / "info.json").read_text())["splits"] == {
        "train": "0:10",
        "validation": "10:12",
    }


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


def test_aggregation_paths_refuse_existing_output_without_deleting_it(tmp_path):
    aggregate = _load_aggregate_module()
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "aggregate"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("do not delete", encoding="utf-8")

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        aggregate.validate_aggregation_paths([source.resolve()], output.resolve())

    assert marker.read_text(encoding="utf-8") == "do not delete"


def test_aggregation_paths_refuse_duplicate_sources(tmp_path):
    aggregate = _load_aggregate_module()
    source = (tmp_path / "source").resolve()
    output = (tmp_path / "aggregate").resolve()

    with pytest.raises(ValueError, match="重复目录"):
        aggregate.validate_aggregation_paths([source, source], output)


def test_aggregation_paths_refuse_protected_original_name(tmp_path):
    aggregate = _load_aggregate_module()
    source = (tmp_path / "source").resolve()
    output = (tmp_path / "bamboo_act_report_full").resolve()

    with pytest.raises(ValueError, match="受保护"):
        aggregate.validate_aggregation_paths([source], output)

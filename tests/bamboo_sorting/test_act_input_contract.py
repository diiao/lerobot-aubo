from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lerobot.bamboo_sorting.act_input_contract import (
    DROP_GRIPPER_STATE_VARIANT,
    FULL_STATE_VARIANT,
    FULL_ZERO_GRIPPER_STATE_VARIANT,
    STATE_KEY,
    adapt_features_and_stats,
    aggregate_selected_episode_stats,
    apply_state_input_contract,
    build_state_input_contract,
    load_episode_split,
    load_state_input_contract,
    write_state_input_contract,
)


def _features() -> dict:
    return {
        STATE_KEY: {
            "dtype": "float32",
            "shape": [4],
            "names": ["J1", "ee.x", "ee.wz", "gripper_pos"],
        },
        "action": {
            "dtype": "float32",
            "shape": [2],
            "names": ["ee.x", "ee.gripper_pos"],
        },
    }


def _stats() -> dict:
    return {
        STATE_KEY: {
            "mean": np.array([1.0, 2.0, 3.0, 50.0]),
            "std": torch.tensor([0.1, 0.2, 0.3, 50.0]),
            "count": np.array([100]),
        },
        "action": {"mean": [1.0, 50.0], "std": [0.1, 50.0]},
    }


def test_drop_gripper_changes_only_state_input() -> None:
    features = _features()
    stats = _stats()
    contract = build_state_input_contract(features, DROP_GRIPPER_STATE_VARIANT)
    adapted_features, adapted_stats = adapt_features_and_stats(features, stats, contract)
    original = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    batch = {STATE_KEY: original, "action": torch.tensor([[4.0, 100.0]])}
    adapted_batch = apply_state_input_contract(batch, contract)

    assert contract.removed_name == "gripper_pos"
    assert contract.removed_index == 3
    assert adapted_features[STATE_KEY]["shape"] == [3]
    assert adapted_features[STATE_KEY]["names"] == ["J1", "ee.x", "ee.wz"]
    assert adapted_stats[STATE_KEY]["mean"].tolist() == [1.0, 2.0, 3.0]
    assert adapted_stats[STATE_KEY]["std"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert adapted_stats[STATE_KEY]["count"].tolist() == [100]
    assert adapted_batch[STATE_KEY].tolist() == [[1.0, 2.0, 3.0]]
    assert adapted_batch["action"].tolist() == [[4.0, 100.0]]
    assert original.tolist() == [[1.0, 2.0, 3.0, 100.0]]
    assert features[STATE_KEY]["shape"] == [4]


def test_full_contract_validates_width_without_mutating_batch() -> None:
    contract = build_state_input_contract(_features(), FULL_STATE_VARIANT)
    state = np.arange(4, dtype=np.float32)
    result = apply_state_input_contract({STATE_KEY: state}, contract)
    assert result[STATE_KEY] is state
    with pytest.raises(ValueError, match="width must be 4"):
        apply_state_input_contract({STATE_KEY: state[:3]}, contract)


def test_full_zero_gripper_contract_preserves_width_and_masks_without_mutating() -> None:
    contract = build_state_input_contract(_features(), FULL_ZERO_GRIPPER_STATE_VARIANT)
    original = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    result = apply_state_input_contract({STATE_KEY: original}, contract)

    assert contract.source_width == 4
    assert contract.model_width == 4
    assert contract.overridden_name == "gripper_pos"
    assert contract.overridden_index == 3
    assert contract.overridden_value == 0.0
    assert result[STATE_KEY].tolist() == [[1.0, 2.0, 3.0, 0.0]]
    assert original.tolist() == [[1.0, 2.0, 3.0, 100.0]]


def test_drop_contract_rejects_ambiguous_or_wrong_width_state() -> None:
    features = _features()
    features[STATE_KEY]["names"][-1] = "not_a_gripper"
    with pytest.raises(ValueError, match="exactly one gripper_pos"):
        build_state_input_contract(features, DROP_GRIPPER_STATE_VARIANT)

    contract = build_state_input_contract(_features(), DROP_GRIPPER_STATE_VARIANT)
    with pytest.raises(ValueError, match="end in width 4"):
        apply_state_input_contract({STATE_KEY: torch.zeros(2, 3)}, contract)


def test_episode_split_is_bound_to_dataset_hash_and_disjoint(tmp_path: Path) -> None:
    info_path = tmp_path / "info.json"
    info_path.write_text('{"dataset": "v1"}\n', encoding="utf-8")
    digest = hashlib.sha256(info_path.read_bytes()).hexdigest()
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": "AuboActEpisodeSplitV1",
                "dataset_info_sha256": digest,
                "train_episodes": [0, 1],
                "val_episodes": [2],
            }
        ),
        encoding="utf-8",
    )

    split = load_episode_split(split_path, dataset_info_path=info_path, total_episodes=3)
    assert split.train_episodes == (0, 1)
    assert split.val_episodes == (2,)

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["val_episodes"] = [1]
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_episode_split(split_path, dataset_info_path=info_path, total_episodes=3)


def test_episode_split_rejects_wrong_dataset_snapshot(tmp_path: Path) -> None:
    info_path = tmp_path / "info.json"
    info_path.write_text("current", encoding="utf-8")
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": "AuboActEpisodeSplitV1",
                "dataset_info_sha256": "0" * 64,
                "train_episodes": [0],
                "val_episodes": [1],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        load_episode_split(split_path, dataset_info_path=info_path, total_episodes=2)


@pytest.mark.parametrize(
    "variant",
    [FULL_STATE_VARIANT, FULL_ZERO_GRIPPER_STATE_VARIANT, DROP_GRIPPER_STATE_VARIANT],
)
def test_checkpoint_contract_round_trip(tmp_path: Path, variant: str) -> None:
    contract = build_state_input_contract(_features(), variant)
    destination = write_state_input_contract(tmp_path, contract)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    expected_schema = (
        "AuboActStateInputContractV2"
        if variant == FULL_ZERO_GRIPPER_STATE_VARIANT
        else "AuboActStateInputContractV1"
    )
    assert payload["schema_version"] == expected_schema
    assert destination.name == "act_state_input_contract.json"
    assert load_state_input_contract(tmp_path) == contract


def test_checkpoint_contract_loads_existing_v1_full_model(tmp_path: Path) -> None:
    payload = {
        "schema_version": "AuboActStateInputContractV1",
        "variant": "full",
        "state_key": STATE_KEY,
        "source_names": ["J1", "ee.x", "ee.wz", "gripper_pos"],
        "model_names": ["J1", "ee.x", "ee.wz", "gripper_pos"],
        "removed_name": None,
        "removed_index": None,
    }
    (tmp_path / "act_state_input_contract.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    contract = load_state_input_contract(tmp_path)

    assert contract.variant == FULL_STATE_VARIANT
    assert contract.overridden_name is None


def test_checkpoint_contract_rejects_inconsistent_drop(tmp_path: Path) -> None:
    contract = build_state_input_contract(_features(), DROP_GRIPPER_STATE_VARIANT)
    destination = write_state_input_contract(tmp_path, contract)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["removed_index"] = 0
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="internally inconsistent"):
        load_state_input_contract(tmp_path)


def test_aggregate_selected_episode_stats_uses_only_requested_rows(tmp_path: Path) -> None:
    episode_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    episode_dir.mkdir(parents=True)
    rows = []
    for episode_index, mean, count in ((0, 1.0, 2), (1, 5.0, 2), (2, 100.0, 10)):
        rows.append(
            {
                "episode_index": episode_index,
                "stats/observation.state/min": np.array([mean - 1.0]),
                "stats/observation.state/max": np.array([mean + 1.0]),
                "stats/observation.state/mean": np.array([mean]),
                "stats/observation.state/std": np.array([1.0]),
                "stats/observation.state/count": np.array([count]),
            }
        )
    pd.DataFrame(rows).to_parquet(episode_dir / "file-000.parquet")
    features = {
        "observation.state": {"dtype": "float32", "shape": [1], "names": ["J1"]},
    }
    stats = aggregate_selected_episode_stats(tmp_path, [0, 1], features)
    assert stats["observation.state"]["mean"].tolist() == [3.0]
    assert stats["observation.state"]["count"].tolist() == [4]
    assert stats["observation.state"]["max"].tolist() == [6.0]


def test_aggregate_selected_episode_stats_rejects_missing_episode(tmp_path: Path) -> None:
    episode_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    episode_dir.mkdir(parents=True)
    pd.DataFrame({"episode_index": [0]}).to_parquet(episode_dir / "file-000.parquet")
    with pytest.raises(ValueError, match="missing requested indices"):
        aggregate_selected_episode_stats(tmp_path, [1], {})

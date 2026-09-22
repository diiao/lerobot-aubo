import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.bamboo_sorting.joint_training import prepare_training_data, prediction_metrics


def _entry():
    path = Path(__file__).parents[2] / "examples/phone_to_auboi10/train_joint_smolvla.py"
    spec = importlib.util.spec_from_file_location("joint_training_entry_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_never_imports_model_or_hardware_and_writes_nothing(tmp_path, monkeypatch):
    import builtins
    entry = _entry()
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith(("torch", "transformers", "lerobot")):
            raise AssertionError(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert entry.main(["--output", str(tmp_path / "plan")]) == 0
    assert not (tmp_path / "plan").exists()


@pytest.mark.parametrize("args", [["--stage", "train"], ["--stage", "smoke"], ["--steps", "0"],
                                  ["--stage", "evaluate"], ["--batch-size", "0"]])
def test_incomplete_or_invalid_invocations_are_refused(args):
    with pytest.raises(SystemExit):
        _entry().parse_args(args)


def test_refuses_output_inside_input(tmp_path):
    with pytest.raises(SystemExit):
        _entry().parse_args(["--stage", "smoke", "--base-path", str(tmp_path),
            "--vlm-path", str(tmp_path / "vlm"), "--output", str(tmp_path / "output")])


def test_train_statistics_exclude_failed_and_validation_episodes(tmp_path, monkeypatch):
    train, val = tmp_path / "train", tmp_path / "validation"
    for root in (train, val):
        (root / "data/chunk-000").mkdir(parents=True)
    good = [[0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 100]]
    bad = [[999, 999, 999, 999, 999, 999, 0]]
    pq.write_table(pa.table({"episode_index": [0, 0, 1], "action": good + bad,
                            "observation.state": good + bad}), train / "data/chunk-000/file-000.parquet")
    def report(root, evidence, **kwargs):
        split = "train" if root == train else "validation"
        return {"source_dataset_root": str(root), "gripper_quality_passed": True, "supervised_candidate_episodes": [0],
                "capture_complete": True, "episodes": [
                    {"episode_index": 0, "split": split, "scene_id": split + "-scene", "frames": 2},
                    {"episode_index": 1, "split": split, "scene_id": "failed-scene", "frames": 1}]}
    monkeypatch.setattr("lerobot.bamboo_sorting.joint_training.audit_joint_dataset", report)
    result = prepare_training_data(train, tmp_path / "te", val, tmp_path / "ve")
    assert result["train"]["episodes"] == [0]
    assert result["train_only_stats"]["action"]["mean"] == [1, 1, 1, 1, 1, 1, 50]
    def overlap(root, evidence, **kwargs):
        record = report(root, evidence)
        record["episodes"][0]["scene_id"] = "same-scene"
        return record
    monkeypatch.setattr("lerobot.bamboo_sorting.joint_training.audit_joint_dataset", overlap)
    with pytest.raises(ValueError, match="scene leakage"):
        prepare_training_data(train, tmp_path / "te", val, tmp_path / "ve")


def test_metrics_expose_copy_state_failure_despite_high_accuracy():
    states = np.zeros((150, 7))
    target = states.copy()
    target[50:100, 6] = 100
    states[1:, 6] = target[:-1, 6]
    result = prediction_metrics(states, target, states.copy())
    assert result["gripper_accuracy"] > .98
    assert result["activate_recall"] == result["release_recall"] == 0
    assert result["hold_false_switch_rate"] == 0
    perfect = prediction_metrics(states, target, target)
    assert perfect["activate_recall"] == perfect["release_recall"] == 1


def test_manifest_pools_selected_frames_and_rejects_duplicate_sources(tmp_path, monkeypatch):
    import json
    from lerobot.bamboo_sorting.joint_training import prepare_training_manifest, read_data_manifest
    arrays = [np.array([[0] * 7, [2] * 6 + [100]], dtype=float),
              np.array([[10] * 7, [20] * 7, [30] * 7, [40] * 6 + [100]], dtype=float)]
    def prepare(root, evidence, validation, ve, **kwargs):
        idx = int(Path(root).name[-1]);a = arrays[idx]
        stats = {k: {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
                     "min": a.min(0).tolist(), "max": a.max(0).tolist()}
                 for k in ("action", "observation.state")}
        return {"train": {"frames": len(a), "scenes": [str(idx)], "root": root},
                "validation": {"root": validation}, "train_only_stats": stats}
    monkeypatch.setattr("lerobot.bamboo_sorting.joint_training.prepare_training_data", prepare)
    value = {"schema_version": "AuboJointTrainingSourcesV1", "train": [
        {"root": "train0", "evidence_root": "ev0"}, {"root": "train1", "evidence_root": "ev1"}],
        "validation": {"root": "val", "evidence_root": "ve"}}
    path = tmp_path / "sources.json";path.write_text(json.dumps(value))
    result = prepare_training_manifest(path);combined = np.concatenate(arrays)
    assert result["train"]["frames"] == 6
    np.testing.assert_allclose(result["train_only_stats"]["action"]["mean"], combined.mean(0))
    np.testing.assert_allclose(result["train_only_stats"]["action"]["std"], combined.std(0))
    value["train"].append(value["train"][0]);path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="duplicate"):
        read_data_manifest(path)


def test_concat_preserves_source_local_action_chunks(monkeypatch):
    import torch
    from lerobot.bamboo_sorting.joint_training import load_prepared_dataset
    class Source(torch.utils.data.Dataset):
        def __init__(self, _, *, root, episodes, delta_timestamps, video_backend):
            self.value = int(root)
            assert len(delta_timestamps["action"]) == 50
        def __len__(self): return 2
        def __getitem__(self, i):
            return {"action": torch.full((50, 7), self.value),
                    "action_is_pad": torch.arange(50) >= 2-i}
    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", Source)
    ds = load_prepared_dataset({"frames": 4, "sources": [
        {"root": str(i), "episodes": [0], "frames": 2} for i in (1, 9)]})
    assert ds[1]["action"].eq(1).all() and ds[1]["action_is_pad"].sum() == 49
    assert ds[2]["action"].eq(9).all()


@pytest.mark.parametrize("multiple_sources", [False, True])
def test_training_loop_checkpoint_and_evaluation_with_toy_policy(tmp_path, monkeypatch, multiple_sources):
    """Exercise orchestration without installing dependencies or training SmolVLA."""
    import json
    import sys
    from types import SimpleNamespace
    import torch
    from lerobot.bamboo_sorting.smolvla_joint_adapter import make_joint_smolvla_config
    from tests.processor.test_smolvla_processor import MockTokenizerProcessorStep
    from tests.bamboo_sorting.test_joint_gripper_quality import _stats

    base, vlm = tmp_path / "base", tmp_path / "vlm"
    base.mkdir()
    vlm.mkdir()
    make_joint_smolvla_config().save_pretrained(base)
    class ToyPolicy(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.value = torch.nn.Parameter(torch.zeros(7))

        @classmethod
        def from_pretrained(cls, path, *, config, **kwargs):
            model = cls(config)
            if (Path(path) / "toy.pt").exists():
                model.load_state_dict(torch.load(Path(path) / "toy.pt", weights_only=True))
            return model

        def save_pretrained(self, path):
            path.mkdir()
            self.config.save_pretrained(path)
            torch.save(self.state_dict(), path / "toy.pt")

        def forward(self, batch, reduction="mean"):
            loss = (self.value - batch["action"]).square().mean((1, 2))
            return (loss if reduction == "none" else loss.mean()), {}

        def predict_action_chunk(self, batch):
            return self.value.expand(len(batch["observation.state"]), 50, 7)

    class ToyDataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs):
            assert kwargs["episodes"] == [0]

        def __len__(self):
            return 3

        def __getitem__(self, index):
            from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_TASK
            state = torch.zeros(7)
            state[6] = [0, 0, 100][index]
            action = torch.zeros(50, 7)
            action[:, 6] = [0, 100, 0][index]
            return {"observation.state": state, "action": action, "task": JOINT_TASK,
                    "episode_index": 0, "frame_index": index,
                    "action_is_pad": torch.zeros(50, dtype=torch.bool),
                    **{f"observation.images.{name}": torch.zeros(3, 480, 640)
                       for name in ("global_rgb", "grasp_rgb")}}

    monkeypatch.setitem(sys.modules, "lerobot.policies.smolvla.modeling_smolvla", SimpleNamespace(SmolVLAPolicy=ToyPolicy))
    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", ToyDataset)
    monkeypatch.setattr("lerobot.policies.smolvla.processor_smolvla.TokenizerProcessorStep", MockTokenizerProcessorStep)
    import importlib.metadata
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "mock" if name == "transformers" else version(name))
    prepared = {"train_only_stats": _stats(), **{key: {"root": str(tmp_path / key), "episodes": [0], "frames": 3}
                for key in ("train", "validation")}}
    if multiple_sources:
        prepared["train"] = {"sources": [prepared["train"],
            {"root": str(tmp_path / "train2"), "episodes": [0], "frames": 3}], "frames": 6}
    monkeypatch.setattr("lerobot.bamboo_sorting.joint_training.prepare_training_data", lambda *args, **kwargs: prepared)
    data_args = ["--train-root", str(tmp_path / "train"), "--train-evidence", str(tmp_path / "te"),
                 "--validation-root", str(tmp_path / "validation"), "--validation-evidence", str(tmp_path / "ve")]
    output = tmp_path / "output"
    entry = _entry()
    assert entry.main(["--stage", "train", "--steps", "2", "--batch-size", "2",
        "--base-path", str(base), "--vlm-path", str(vlm), "--output", str(output), *data_args]) == 0
    assert json.loads((output / "reload_check.json").read_text())["identical_prediction"]
    assert len((output / "training.jsonl").read_text().splitlines()) == 2
    assert json.loads((output / "validation.json").read_text())["activate_samples"] == 1
    assert entry.main(["--stage", "evaluate", "--checkpoint", str(output / "final"),
                       "--output", str(tmp_path / "evaluation"), *data_args]) == 0
    assert json.loads((tmp_path / "evaluation/complete.json").read_text())["completed"]

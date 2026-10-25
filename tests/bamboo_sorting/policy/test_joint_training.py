import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.bamboo_sorting.policy.joint_training import prepare_training_data, prediction_metrics
from tests.bamboo_sorting.paths import REPO_ROOT


def _entry():
    path = REPO_ROOT / "examples/phone_to_auboi10/train_joint_smolvla.py"
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


def test_manifest_refuses_output_inside_second_validation_source(tmp_path, capsys):
    import json
    value = {"schema_version": "AuboJointTrainingSourcesV1",
             "train": [{"root": "train", "evidence_root": "train-evidence"}],
             "validation": [{"root": "validation1", "evidence_root": "validation-evidence1"},
                            {"root": "validation2", "evidence_root": "validation-evidence2"}]}
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps(value))
    with pytest.raises(SystemExit):
        _entry().parse_args(["--stage", "preflight", "--data-manifest", str(manifest),
            "--output", str(tmp_path / "validation2" / "output")])
    assert "output must be independent" in capsys.readouterr().err


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
    monkeypatch.setattr("lerobot.bamboo_sorting.policy.joint_training.audit_joint_dataset", report)
    result = prepare_training_data(train, tmp_path / "te", val, tmp_path / "ve")
    assert result["train"]["episodes"] == [0]
    assert result["train_only_stats"]["action"]["mean"] == [1, 1, 1, 1, 1, 1, 50]
    def overlap(root, evidence, **kwargs):
        record = report(root, evidence)
        record["episodes"][0]["scene_id"] = "same-scene"
        return record
    monkeypatch.setattr("lerobot.bamboo_sorting.policy.joint_training.audit_joint_dataset", overlap)
    with pytest.raises(ValueError, match="scene leakage"):
        prepare_training_data(train, tmp_path / "te", val, tmp_path / "ve")


def test_failed_step_still_groups_entire_pile_for_split_leakage(tmp_path, monkeypatch):
    def audit(root, evidence, **kwargs):
        split = Path(root).name
        episodes = [dict(episode_index=0, scene_id='p-001' if split == 'validation' else 'p-002',
                         split=split, frames=2, pick_step=1)]
        if split == 'train':
            episodes.append(dict(episode_index=1, scene_id='p-001', split=split, frames=2, pick_step=2))
        return dict(source_dataset_root=str(root), gripper_quality_passed=True,
                    supervised_candidate_episodes=[0], capture_complete=True, episodes=episodes)
    monkeypatch.setattr('lerobot.bamboo_sorting.policy.joint_training.audit_joint_dataset', audit)
    with pytest.raises(ValueError, match='scene leakage'):
        prepare_training_data(tmp_path/'train', tmp_path/'te', tmp_path/'validation', tmp_path/'ve')


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


def test_original_trainer_reports_native_image_preprocessing_requirement(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.bamboo_sorting.policy.joint_training.audit_joint_dataset", lambda *a, **kw: {
        "image_shapes": {"global_rgb": [1080, 1920, 3], "grasp_rgb": [480, 640, 3]}})
    with pytest.raises(ValueError, match="configure native-resolution model preprocessing"):
        prepare_training_data(tmp_path / "train", tmp_path / "train-evidence",
                              tmp_path / "validation", tmp_path / "validation-evidence")


def test_manifest_pools_selected_frames_and_rejects_duplicate_sources(tmp_path, monkeypatch):
    import json
    from lerobot.bamboo_sorting.policy.joint_training import prepare_training_manifest, read_data_manifest
    arrays = [np.array([[0] * 7, [2] * 6 + [100]], dtype=float),
              np.array([[10] * 7, [20] * 7, [30] * 7, [40] * 6 + [100]], dtype=float)]
    def prepare(root, evidence, validation, ve, **kwargs):
        idx = int(Path(root).name[-1]);a = arrays[idx]
        validation_name = Path(validation).name
        stats = {k: {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
                     "min": a.min(0).tolist(), "max": a.max(0).tolist()}
                 for k in ("action", "observation.state")}
        return {"train": {"frames": len(a), "scenes": [str(idx)], "root": root},
                "validation": {"root": validation, "frames": 3 if validation_name == "val" else 2,
                    "scenes": ["1" if validation_name == "val-overlap" else validation_name],
                    "capture_complete": True}, "train_only_stats": stats}
    monkeypatch.setattr("lerobot.bamboo_sorting.policy.joint_training.prepare_training_data", prepare)
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
    value["train"].pop()
    value["validation"] = [{"root": "val", "evidence_root": "ve"},
                           {"root": "val2", "evidence_root": "ve2"}]
    path.write_text(json.dumps(value))
    result = prepare_training_manifest(path)
    assert result["validation"]["frames"] == 5
    assert [Path(item["root"]).name for item in result["validation"]["sources"]] == ["val", "val2"]
    value["validation"][1] = value["validation"][0]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="duplicate"):
        read_data_manifest(path)
    value["validation"][1] = {"root": "val-overlap", "evidence_root": "ve2"}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="scene leakage"):
        prepare_training_manifest(path)


def test_concat_preserves_source_local_action_chunks(monkeypatch):
    import torch
    from lerobot.bamboo_sorting.policy.joint_training import load_prepared_dataset
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


def test_preclose_weights_follow_future_expert_frame_and_source(tmp_path):
    import json
    import torch
    from lerobot.bamboo_sorting.policy.joint_training import PrecloseWeightedDataset
    original = {"episode_index": 0, "frame_index": 5, "action": torch.arange(350).reshape(50, 7),
                "action_is_pad": torch.arange(50) >= 45}
    datasets = []
    for name, close in (("source_a", 12), ("source_b", 23)):
        annotation = tmp_path/f"{name}.json"
        annotation.write_text(json.dumps({"episodes": [{"episode_index": 0, "close_frame": close}]}))
        datasets.append(PrecloseWeightedDataset([original], dict(root=name,episodes=[0],target_annotations=str(annotation)),
            weight=3.,frames_before_close=10))
    a, b = datasets[0][0], datasets[1][0]
    assert torch.where(a["action_loss_weights"] == 3)[0].tolist() == list(range(8))
    assert torch.where(b["action_loss_weights"] == 3)[0].tolist() == list(range(8, 19))
    assert a["training_sample_key"] == "source_a/0/5" and b["training_sample_key"] == "source_b/0/5"
    assert "action_loss_weights" not in original
    assert a["action"] is original["action"] and a["action_is_pad"] is original["action_is_pad"]


@pytest.mark.parametrize("multiple_sources,target_conditioned,objective", [
    (False, False, None), (True, False, None), (False, True, None),
    (True, True, "preclose"), (True, True, "holding"), (True, True, "gripper"),
    (True, True, "phase"), (True, True, "fit"),
])
def test_training_loop_checkpoint_and_evaluation_with_toy_policy(tmp_path, monkeypatch, multiple_sources, target_conditioned, objective):
    """Exercise orchestration without installing dependencies or training SmolVLA."""
    import json
    import sys
    from types import SimpleNamespace
    import torch
    from lerobot.bamboo_sorting.policy.smolvla_joint_adapter import make_joint_smolvla_config
    from tests.bamboo_sorting.capture_fakes import stats as _stats
    from tests.mocks.mock_tokenizer import MockTokenizerProcessorStep

    base, vlm = tmp_path / "base", tmp_path / "vlm"
    n_frames = 4 if objective == "phase" else 3
    batch_size = 8 if objective == "phase" else 2
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

        def forward(self, batch, reduction="mean", action_loss_weights=None, *, holding_pair_mask=None,
                    holding_loss_coefficient=0., gripper_loss_weights=None):
            if objective in ("phase", "fit"):
                assert action_loss_weights is holding_pair_mask is gripper_loss_weights is None
                assert holding_loss_coefficient == 0
            image_key = "target_global_rgb" if target_conditioned else "global_rgb"
            assert f"observation.images.{image_key}" in batch
            assert ("observation.images.global_rgb" in batch) != target_conditioned
            from lerobot.policies.smolvla.losses import masked_action_loss, holding_action_loss
            errors = (self.value - batch["action"]).square()
            uniform = errors.mean((1, 2))
            loss = masked_action_loss(errors, 7, batch["action_is_pad"], timestep_weights=action_loss_weights,
                                      gripper_weights=gripper_loss_weights)
            info = {"uniform_valid_loss":float(uniform.mean().detach())}
            if holding_loss_coefficient:
                aux = holding_action_loss(self.value.expand_as(batch["action"]), batch["action"],
                                           holding_pair_mask, batch["action_is_pad"])
                loss = loss + holding_loss_coefficient * aux
                info.update(holding_loss=float(aux.mean().detach()),
                            scaled_holding_loss=float((holding_loss_coefficient * aux).mean().detach()))
            if gripper_loss_weights is not None:
                info["gripper_weighted_loss"] = float(masked_action_loss(errors[..., 6:7], 1,
                    batch["action_is_pad"], timestep_weights=gripper_loss_weights).mean().detach())
            if objective == "phase":
                info.pop("uniform_valid_loss")  # real base objective does not add this auxiliary log key
            return (loss if reduction == "none" else loss.mean()), info

        def predict_action_chunk(self, batch):
            return self.value.expand(len(batch["observation.state"]), 50, 7)

    class ToyDataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs):
            assert kwargs["episodes"] == [0]
            self.root = kwargs["root"]
            self.hf_dataset = {"episode_index": [0]*n_frames, "frame_index": list(range(n_frames))}

        def __len__(self):
            return n_frames

        def __getitem__(self, index):
            from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_TASK
            state = torch.zeros(7)
            state[6] = [0, 0, 100, 0][index]
            action = torch.zeros(50, 7)
            action[:, 6] = [0, 100, 0, 0][index]
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
    prepared = {"train_only_stats": _stats(), **{key: {"root": str(tmp_path / key), "episodes": [0], "frames": n_frames}
                for key in ("train", "validation")}}
    if target_conditioned:
        import cv2
        mask_path = tmp_path / "mask.png"
        mask = np.zeros((480, 640), np.uint8)
        mask[10:20, 10:80] = 255
        cv2.imwrite(str(mask_path), mask)
        for record in (prepared["train"], prepared["validation"]):
            record["target_frames"] = [{"episode_index": 0, "frame_index": i,
                "frame_id": f"0/global_rgb/{i}", "target_id": "strip_a", "transport": i >= 2,
                "annotation": None if i >= 2 else {"schema": "aubo_joint_target_frame",
                    "frame_id": f"0/global_rgb/{i}", "target_id": "strip_a", "reviewed": True,
                    "status": "visible", "mask_path": str(mask_path)}} for i in range(n_frames)]
    if multiple_sources:
        prepared["train"] = {"sources": [prepared["train"],
            {**prepared["train"], "root": str(tmp_path / "train2")}], "frames": 2*n_frames}
        prepared["validation"] = {"sources": [prepared["validation"],
            {**prepared["validation"], "root": str(tmp_path / "validation2")}], "frames": 2*n_frames}
    training_extra = []
    if objective == "fit":
        selection = tmp_path/"fit_observations.json"
        selection.write_text(json.dumps({"sample_keys": ["train2/0/1", "train/0/2"]}))
        training_extra = ["--fit-observations", str(selection), "--learning-rate", "1e-5"]
    elif objective is not None:
        annotation = tmp_path/"close_frames.json"
        annotation.write_text(json.dumps({"episodes":[{"episode_index":0,"close_frame":1,
                                                       "reviewed":True,"scene_id":"synthetic"}]}))
        for source in prepared["train"]["sources"]:
            source["target_annotations"] = str(annotation)
            if objective in ("holding", "gripper", "phase"):
                data = Path(source["root"]) / "data/chunk-000"
                data.mkdir(parents=True)
                pq.write_table(pa.table({"episode_index":[0]*n_frames,"frame_index":list(range(n_frames)),
                    "action":[[0.]*7,[0.]*6+[100.],[0.]*7,[0.]*7][:n_frames],
                    "observation.state":[[0.]*7,[0.]*7,[0.]*6+[100.],[0.]*7][:n_frames]}), data/"file-000.parquet")
        training_extra = {"preclose":["--preclose-loss-weight","3"],
                          "holding":["--holding-loss-coefficient","0.1"],
                          "gripper":["--gripper-close-loss-weight","3"],
                          "phase":["--phase-balanced-sampling"]}[objective] + ["--learning-rate","1e-5"]
    monkeypatch.setattr("lerobot.bamboo_sorting.policy.joint_training.prepare_training_data", lambda *args, **kwargs: prepared)
    data_args = ["--train-root", str(tmp_path / "train"), "--train-evidence", str(tmp_path / "te"),
                 "--validation-root", str(tmp_path / "validation"), "--validation-evidence", str(tmp_path / "ve")]
    if target_conditioned:
        manifest = tmp_path / "sources.json"
        manifest.write_text(json.dumps({"schema_version": "AuboJointTrainingSourcesV1",
            "train": [{"root": "train", "evidence_root": "te"}],
            "validation": {"root": "validation", "evidence_root": "ve"}}))
        monkeypatch.setattr("lerobot.bamboo_sorting.policy.joint_training.prepare_training_manifest",
                            lambda *args, **kwargs: prepared)
        data_args = ["--target-conditioned", "--data-manifest", str(manifest)]
    output = tmp_path / "output"
    entry = _entry()
    assert entry.main(["--stage", "train", "--steps", "2", "--batch-size", str(batch_size),
        "--base-path", str(base), "--vlm-path", str(vlm), "--output", str(output), *data_args, *training_extra]) == 0
    assert json.loads((output / "reload_check.json").read_text())["identical_prediction"]
    from lerobot.bamboo_sorting.aubo_joint_contract import joint_contract_record
    from lerobot.bamboo_sorting.policy.joint_target_policy import target_policy_contract_record
    assert json.loads((output / "final/aubo_joint_contract.json").read_text()) == (
        target_policy_contract_record() if target_conditioned else joint_contract_record())
    assert len((output / "training.jsonl").read_text().splitlines()) == 2
    if training_extra:
        assert json.loads((output/"final/config.json").read_text())["optimizer_lr"] == 1e-5
        logs = [json.loads(line) for line in (output/"training.jsonl").read_text().splitlines()]
        assert all(len(row["sample_keys"]) == batch_size for row in logs)
        if objective == "fit":
            selection = json.loads((output/"fit_observations.json").read_text())
            assert selection["source_indices"] == [4, 2]
            assert selection["source_frames"] == 6 and selection["selected_frames"] == 2
            assert all(row["sample_keys"] == ["train2/0/1", "train/0/2"] for row in logs)
            assert json.loads((output/"final/fit_observations.json").read_text()) == selection
            assert json.loads((output/"complete.json").read_text())["diagnostic_only"]
        elif objective == "phase":
            sampling = json.loads((output/"sampling_plan.json").read_text())
            assert [r["sample_keys"] for r in logs] == [b["sample_keys"] for b in sampling["batches"]]
            assert all(r["loss"] == r["uniform_valid_loss"] for r in logs)
            assert sampling["assigned_group_counts"]["late_hold"] == 4
        else:
            counter = {"preclose":"weighted_valid_timesteps", "holding":"holding_valid_pairs",
                       "gripper":"gripper_weighted_valid_timesteps"}[objective]
            assert sum(row[counter] for row in logs) > 0
        if objective in ("holding", "gripper", "phase"):
            assert len(json.loads((output/"training_supervision.json").read_text())["sources"]) == 2
    assert json.loads((output / "validation.json").read_text())["activate_samples"] == (2 if multiple_sources else 1)
    prediction_sources = np.load(output / "heldout_predictions.npz")["source_index"]
    assert prediction_sources.tolist() == ([0]*n_frames+[1]*n_frames if multiple_sources else [0]*n_frames)
    by_source = json.loads((output / "validation_by_source.json").read_text())["sources"]
    assert [item["frames"] for item in by_source] == ([n_frames, n_frames] if multiple_sources else [n_frames])
    assert [item["activate_samples"] for item in by_source] == ([1, 1] if multiple_sources else [1])
    assert entry.main(["--stage", "evaluate", "--checkpoint", str(output / "final"),
                       "--output", str(tmp_path / "evaluation"), *data_args]) == 0
    assert json.loads((tmp_path / "evaluation/complete.json").read_text())["completed"]
    assert json.loads((tmp_path / "evaluation/complete.json").read_text())["diagnostic_only"] == (objective == "fit")


def test_hold_close_labels_stop_at_departure_and_release_and_bind_source_episode(tmp_path):
    import json
    import torch
    from lerobot.bamboo_sorting.policy.joint_training import HoldCloseDataset, expert_hold_close_labels
    actions = np.zeros((8, 7))
    actions[:, 0] = [1, .004, 0, .002, 1, 0, 0, 0]  # later return must not restart hold
    actions[2:6, 6] = 100
    states = np.vstack([actions[0], actions[:-1]])
    labels = expert_hold_close_labels(actions, states, 2)
    assert (labels["hold_start"], labels["hold_end"], labels["close_window_end"]) == (1, 3, 4)
    with pytest.raises(ValueError, match="first close"):
        expert_hold_close_labels(actions, states, 3)
    records = []
    for name, shift in (("a", 0), ("b", 1)):
        root = tmp_path / name
        data = root / "data/chunk-000"
        data.mkdir(parents=True)
        extended = np.vstack([np.repeat(actions[:1], shift, axis=0), actions])
        extended_states = np.vstack([extended[0], extended[:-1]])
        # A separate one-row closure ends at the release, not close+2.
        short = np.zeros((3, 7)); short[1, 6] = 100
        short_states = np.vstack([short[0], short[:-1]])
        pq.write_table(pa.table({"episode_index":[0]*len(extended)+[1]*3,
            "frame_index":list(range(len(extended)))+list(range(3)),
            "action":np.vstack([extended, short]).tolist(),
            "observation.state":np.vstack([extended_states, short_states]).tolist()}), data / "file-000.parquet")
        annotation = root / "annotations.json"
        annotation.write_text(json.dumps({"episodes":[
            {"episode_index":ep,"reviewed":True,"scene_id":name,"close_frame":close}
            for ep,close in [(0,2+shift),(1,1)]]}))
        records.append(dict(root=str(root),target_annotations=str(annotation),episodes=[0,1],frames=len(extended)+3))
    def raw(ep, frame, valid):
        return {"episode_index":ep,"frame_index":frame,"action":torch.zeros(7,7),
                "action_is_pad":torch.arange(7)>=valid,"observation.state":torch.zeros(7)}
    items = [raw(0,0,7), raw(0,6,2), raw(1,0,3)]
    rng = torch.get_rng_state().clone()
    ds = [HoldCloseDataset(items,r,holding=True,holding_tolerance_deg=.01,gripper_weight=3.,close_frames=3)
          for r in records]
    assert torch.where(ds[0][0]["holding_pair_mask"])[0].tolist() == [1,2]
    assert torch.where(ds[1][0]["holding_pair_mask"])[0].tolist() == [2,3]
    assert torch.where(ds[0][0]["gripper_loss_weights"] == 3)[0].tolist() == [2,3,4]
    assert torch.where(ds[0][2]["gripper_loss_weights"] == 3)[0].tolist() == [1]
    assert not ds[0][1]["holding_pair_mask"].any() and ds[0][1]["gripper_loss_weights"].eq(1).all()
    assert ds[0][0]["action"] is items[0]["action"] and ds[0][0]["observation.state"] is items[0]["observation.state"]
    assert "holding_pair_mask" not in items[0] and torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("options", [
    ["--holding-loss-coefficient","0.1"], ["--gripper-close-loss-weight","3"],
    ["--phase-balanced-sampling"],
    ["--fit-observations", "not_loaded_in_plan.json"],
])
def test_new_candidate_plan_is_model_free_and_rejects_combination(tmp_path, options):
    entry = _entry()
    assert entry.main(["--target-conditioned",*options,"--output",str(tmp_path/"unused")]) == 0
    assert not (tmp_path/"unused").exists()
    with pytest.raises(SystemExit):
        entry.parse_args(["--target-conditioned",*options,"--preclose-loss-weight","3"])
    with pytest.raises(SystemExit):
        entry.parse_args(options)


def test_prepared_data_reuse_checks_manifest_order_and_keeps_stats(tmp_path):
    import json
    from lerobot.bamboo_sorting.aubo_joint_contract import joint_contract_record
    from lerobot.bamboo_sorting.policy.joint_training import read_prepared_training_data, read_data_manifest
    from tests.bamboo_sorting.capture_fakes import stats
    manifest = tmp_path / "sources.json"
    sources = {"train":[{"root":"train","evidence_root":"te"}],
               "validation":[{"root":"validation","evidence_root":"ve"}]}
    manifest.write_text(json.dumps({"schema_version":"AuboJointTrainingSourcesV1",**sources}))
    for p in ("train","te","validation","ve"):
        (tmp_path/p).mkdir()
    resolved = read_data_manifest(manifest)
    value = {"contract":joint_contract_record(),"train_only_stats":stats(),
             **{k:{"sources":[{**resolved[k][0],"episodes":[0],"frames":3}],"frames":3} for k in sources}}
    saved = tmp_path / "data_preflight.json"
    saved.write_text(json.dumps(value))
    assert read_prepared_training_data(saved, manifest) == value
    value["train"]["sources"][0]["source_dataset_root"] = str(tmp_path/"different")
    saved.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source/order"):
        read_prepared_training_data(saved, manifest)


def test_phase_sampler_preserves_identity_and_covers_each_event_subgroup_without_rng_side_effects():
    from collections import Counter
    import torch
    from lerobot.bamboo_sorting.policy.joint_training import PhaseBalancedBatchSampler
    # Nonconsecutive episode IDs, several sources, and reversed loaded-row order.
    labels = {f"{source}/{ep}": dict(frames=30, hold_start=3, close_frame=4, hold_end=10,
                                    first_release_frame=20)
              for source in ("a", "b") for ep in (2, 7, 9)}
    keys = [f"{demo}/{f}" for demo in labels for f in range(30)][::-1]
    numpy_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
    sampler = PhaseBalancedBatchSampler(keys, labels, steps=180, seed=42)
    assert sampler.report == PhaseBalancedBatchSampler(keys, labels, steps=180, seed=42).report
    assert sampler.batches != PhaseBalancedBatchSampler(keys, labels, steps=180, seed=43).batches
    assert torch.equal(torch_state, torch.get_rng_state())
    for before, after in zip(numpy_state, np.random.get_state(), strict=True):
        np.testing.assert_array_equal(before, after)
    assert len(sampler) == 180 and all(len(batch) == 8 for batch in sampler)
    assert sampler.report["current_event_demonstrations"] == {"close": 6, "release": 6}
    assert sampler.report["assigned_group_counts"] == dict(uniform=720, late_hold=360,
        close_before=60, close_event=60, close_after=60, release_before=60, release_event=60, release_after=60)
    uniform_indices = []
    for batch in sampler.report["batches"]:
        assert batch["sample_keys"] == [keys[i] for i in batch["indices"]]
        for index, key, group in zip(batch["indices"], batch["sample_keys"], batch["groups"], strict=True):
            demo, frame = key.rsplit("/", 1); frame = int(frame)
            if group == "uniform":
                uniform_indices.append(index)
            else:
                bounds = {"late_hold": (0, 10), "close_before": (0, 3), "close_event": (4, 4),
                          "close_after": (5, 14), "release_before": (10, 19),
                          "release_event": (20, 20), "release_after": (21, 29)}
                assert bounds[group][0] <= frame <= bounds[group][1]
    assert set(Counter(uniform_indices).values()) == {4}
    for demo in labels:
        assigned = sampler.report["by_demonstration"][demo]["assigned_groups"]
        assert all(assigned[f"{event}_{subgroup}"] == 10 for event in ("close", "release")
                   for subgroup in ("before", "event", "after"))
    with pytest.raises(ValueError, match="complete episode indices"):
        PhaseBalancedBatchSampler(keys[:-1], labels, steps=1, seed=42)
    labels["a/2"]["first_release_frame"] = None
    with pytest.raises(ValueError, match="ordered hold/close/release"):
        PhaseBalancedBatchSampler(keys, labels, steps=1, seed=42)


def test_phase_sampling_cli_requires_frozen_batch_allocation():
    with pytest.raises(SystemExit):
        _entry().parse_args(["--target-conditioned", "--phase-balanced-sampling", "--batch-size", "4"])


def test_fixed_observation_identity_lookup_and_unchanged_sample():
    from torch.utils.data import ConcatDataset
    from lerobot.bamboo_sorting.policy.joint_training import FixedObservationDataset

    class Native:
        def __init__(self, source):
            self.root = Path(source)
            self.hf_dataset = dict(episode_index=[5, 2], frame_index=[17, 9])
            self.items = [dict(episode_index=ep, frame_index=frame, action=object(), action_is_pad=object())
                          for ep, frame in zip([5, 2], [17, 9])]
        def __len__(self):
            return len(self.items)
        def __getitem__(self, index):
            return self.items[index]

    a, b = Native("source_a"), Native("source_b")
    combined = ConcatDataset([a, b])
    subset = FixedObservationDataset(combined, ["source_b/2/9", "source_a/5/17"])
    assert subset.indices == [3, 0]
    assert subset[0]["action"] is b.items[1]["action"]
    assert subset[0]["action_is_pad"] is b.items[1]["action_is_pad"]
    assert "training_sample_key" not in b.items[1]
    for invalid in ([], ["source_a/5/17"]*2, ["validation/5/17"]):
        with pytest.raises(ValueError):
            FixedObservationDataset(combined, invalid)
    b.items[1]["frame_index"] = 10
    with pytest.raises(ValueError, match="frozen identity"):
        subset[0]

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
    assert "demo_scope" not in result["train"]  # Legacy checkpoint provenance remains comparable.
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


def test_whole_scene_holdout_reuses_original_capture_and_train_only_stats(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root, evidence = tmp_path / 'capture', tmp_path / 'evidence'
    (root / 'data/chunk-000').mkdir(parents=True)
    values = [[0] * 7, [2] * 6 + [100], [900] * 7, [901] * 6 + [100]]
    pq.write_table(pa.table({'episode_index': [0, 0, 1, 1], 'action': values,
        'observation.state': values}), root / 'data/chunk-000/file-000.parquet')
    calls = []
    def audit(*args, **kwargs):
        calls.append(args)
        return dict(source_dataset_root=str(root), gripper_quality_passed=True,
            supervised_candidate_episodes=[0, 1], capture_complete=True, episodes=[
                dict(episode_index=i, split='train', scene_id=scene, frames=2,
                     pick_step=1, target_id='s-001', available_target_count=1)
                for i, scene in enumerate(['keep', 'hold'])])
    monkeypatch.setattr('lerobot.bamboo_sorting.policy.joint_training.audit_joint_dataset', audit)
    args = (root, evidence, root, evidence)
    result = prepare_training_data(*args, train_scenes=['keep'], validation_scenes=['hold'])
    assert result['train']['episodes'] == [0] and result['validation']['episodes'] == [1]
    assert result['validation']['effective_split'] == 'validation'
    assert result['validation']['capture_splits'] == ['train']
    assert result['train']['stack_summary']['scenes'] == 1
    assert result['train_only_stats']['action']['mean'] == [1] * 6 + [50]
    assert len(calls) == 1
    with pytest.raises(ValueError, match='independent roots'):
        prepare_training_data(*args, train_scenes=['keep'], validation_scenes=['keep'])
    with pytest.raises(ValueError, match='unknown scenes'):
        prepare_training_data(*args, train_scenes=['keep'], validation_scenes=['unknown'])


def test_manifest_shared_source_requires_disjoint_explicit_scene_selections(tmp_path):
    import json
    from lerobot.bamboo_sorting.policy.joint_training import read_data_manifest
    path = tmp_path / 'sources.json'
    item = dict(root='capture', evidence_root='evidence')
    value = dict(schema_version='AuboJointTrainingSourcesV1',
        train=[{**item, 'selection_scenes': ['a']}],
        validation=[{**item, 'selection_scenes': ['b']}])
    path.write_text(json.dumps(value))
    assert read_data_manifest(path)['validation'][0]['selection_scenes'] == ['b']
    value['validation'][0]['selection_scenes'] = ['a']
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='overlapping'):
        read_data_manifest(path)
    value['validation'][0].pop('selection_scenes')
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='duplicate'):
        read_data_manifest(path)


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
    value["train"][1]["demo_scope"] = "local_alignment"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="differs from audited capture"):
        prepare_training_manifest(path)
    value["train"][1].pop("demo_scope")
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


@pytest.mark.parametrize("multiple_sources,target_conditioned,outline_width,continuation,balanced", [
    (False, False, 2, False, False), (True, False, 2, False, False),
    (False, True, 2, False, False), (True, True, 1, False, False),
    (True, True, 2, True, True), (True, True, 2, False, True),
])
def test_training_loop_checkpoint_and_evaluation_with_toy_policy(tmp_path, monkeypatch, multiple_sources, target_conditioned, outline_width, continuation, balanced):
    """Exercise orchestration without installing dependencies or training SmolVLA."""
    import json
    import sys
    from types import SimpleNamespace
    import torch
    from lerobot.bamboo_sorting.policy.smolvla_joint_adapter import make_joint_smolvla_config
    from tests.bamboo_sorting.capture_fakes import stats as _stats
    from tests.mocks.mock_tokenizer import MockTokenizerProcessorStep

    base, vlm = tmp_path / "base", tmp_path / "vlm"
    n_frames, batch_size = 3, 2
    base.mkdir()
    vlm.mkdir()
    base_config = make_joint_smolvla_config(target_conditioned=target_conditioned if continuation else False)
    base_config.save_pretrained(base)
    class ToyPolicy(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.value = torch.nn.Parameter(torch.ones(7))

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
            image_key = "target_global_rgb" if target_conditioned else "global_rgb"
            assert f"observation.images.{image_key}" in batch
            assert ("observation.images.global_rgb" in batch) != target_conditioned
            from lerobot.policies.smolvla.losses import masked_action_loss
            loss = masked_action_loss((self.value-batch["action"]).square(), 7, batch["action_is_pad"])
            return (loss if reduction == "none" else loss.mean()), {}

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
    if continuation:
        from lerobot.bamboo_sorting.policy.smolvla_joint_adapter import make_joint_smolvla_processors
        from lerobot.bamboo_sorting.policy.joint_target_policy import target_policy_contract_record
        pre, post = make_joint_smolvla_processors(base_config, _stats(), target_conditioned=True)
        pre.save_pretrained(base)
        post.save_pretrained(base)
        (base / "aubo_joint_contract.json").write_text(json.dumps(target_policy_contract_record()))
    if balanced:
        for split in ("train", "validation"):
            prepared[split]["sources"][1]["demo_scope"] = "local_alignment"
        # Fresh training must use the new 60/40 statistics; continuation
        # must still encode/decode with the old saved 80/40 statistics.
        for key in ("action", "observation.state"):
            prepared["train_only_stats"][key]["mean"][6] = 60
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
        data_args = ["--target-conditioned", "--data-manifest", str(manifest),
                     "--target-outline-width-px", str(outline_width)]
    output = tmp_path / "output"
    entry = _entry()
    selection = balanced and not continuation
    if selection:
        # Video/episode chronology is tested against real Parquet separately.
        # This toy plan exercises the complete train -> select -> replace ->
        # strict reload -> best/final validation path without a real policy.
        def toy_monitor(record, **kwargs):
            rows = [dict(dataset_index=si*n_frames+i, source_index=si, episode_index=0,
                         frame_index=i, scene_id=f"toy-{si}", demo_scope=s.get("demo_scope", "full"),
                         phase=("preclose", "transport", "release")[i])
                    for si, s in enumerate(record["sources"]) for i in range(n_frames)]
            return dict(indices=list(range(len(rows))), observations=rows, full_observations=rows,
                        frames=len(rows), episodes=2, scenes=["toy-0", "toy-1"], points_per_phase=8)
        monkeypatch.setattr("lerobot.bamboo_sorting.policy.joint_checkpoint_selection.build_monitor_plan", toy_monitor)
    sampling_args = ["--local-demo-fraction", "0.5"] if balanced else []
    normalization_args = ["--normalization-source", "base_checkpoint"] if continuation else []
    checkpoint_args = ["--validation-every", "1"] if selection else ["--checkpoint-steps", "1"]
    assert entry.main(["--stage", "train", "--steps", "2", "--batch-size", str(batch_size),
        *checkpoint_args, "--base-path", str(base), "--vlm-path", str(vlm),
        "--output", str(output), *data_args, *sampling_args, *normalization_args]) == 0
    assert json.loads((output / "reload_check.json").read_text())["identical_prediction"]
    from lerobot.bamboo_sorting.aubo_joint_contract import joint_contract_record
    from lerobot.bamboo_sorting.policy.joint_target_policy import target_policy_contract_record
    assert json.loads((output / "final/aubo_joint_contract.json").read_text()) == (
        target_policy_contract_record(outline_width_px=outline_width) if target_conditioned else joint_contract_record())
    assert len((output / "training.jsonl").read_text().splitlines()) == 2
    intermediate = output / ("checkpoints/best_step_000002" if selection else "checkpoints/step_000001")
    assert json.loads((intermediate / "training_progress.json").read_text())["optimizer_step"] == (2 if selection else 1)
    assert (intermediate / "aubo_joint_contract.json").read_bytes() == (output / "final/aubo_joint_contract.json").read_bytes()
    same = torch.equal(torch.load(intermediate / "toy.pt", weights_only=True)["value"],
                       torch.load(output / "final/toy.pt", weights_only=True)["value"])
    assert same == selection
    if selection:
        assert not (output / "checkpoints/best_step_000001").exists()
        assert json.loads((intermediate / "reload_check.json").read_text())["state_dict_equal"]
        history = [json.loads(line) for line in (output / "validation_history.jsonl").read_text().splitlines()]
        assert len(history) == 2 and all(h["comparison"]["improved"] for h in history)
        assert json.loads((output / "best_checkpoint.json").read_text())["optimizer_steps"] == 2
        assert (output / "full_validation/best.json").is_file() and (output / "full_validation/final.json").is_file()
    assert json.loads((output / "validation.json").read_text())["activate_samples"] == (2 if multiple_sources else 1)
    prediction_sources = np.load(output / "heldout_predictions.npz")["source_index"]
    assert prediction_sources.tolist() == ([0]*n_frames+[1]*n_frames if multiple_sources else [0]*n_frames)
    by_source = json.loads((output / "validation_by_source.json").read_text())["sources"]
    assert [item["frames"] for item in by_source] == ([n_frames, n_frames] if multiple_sources else [n_frames])
    assert [item["activate_samples"] for item in by_source] == ([1, 1] if multiple_sources else [1])
    if balanced:
        assert json.loads((output/"train_only_stats.json").read_text())["action"]["mean"][6] == 60
        for folder in (output, intermediate, output/"final"):
            normalization = json.loads((folder/"normalization.json").read_text())
            assert normalization["source"] == ("base_checkpoint" if continuation else "train_data")
            assert normalization["stats"]["action"]["mean"][6] == (80 if continuation else 60)
            sampling = json.loads((folder/"training_sampling.json").read_text())
            assert sampling["expected_group_fractions"] == {"full": .5, "local_alignment": .5}
        groups = json.loads((output/"validation_by_demo_scope.json").read_text())["groups"]
        assert {k: v["frames"] for k, v in groups.items()} == {"full": 3, "local_alignment": 3}
    assert entry.main(["--stage", "evaluate", "--checkpoint", str(output / "final"),
                       "--output", str(tmp_path / "evaluation"), *data_args]) == 0
    assert json.loads((tmp_path / "evaluation/complete.json").read_text())["completed"]


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
    # A formerly audited directory cannot silently change its scene selection.
    sources['train'][0]['selection_scenes'] = ['keep']
    manifest.write_text(json.dumps({'schema_version':'AuboJointTrainingSourcesV1',**sources}))
    with pytest.raises(ValueError, match='scene selection mismatch'):
        read_prepared_training_data(saved, manifest)
    record = value['train']['sources'][0]
    record.update(selection_scenes=['keep'], effective_split='validation')
    saved.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='effective split mismatch'):
        read_prepared_training_data(saved, manifest)
    record['effective_split'] = 'train'
    saved.write_text(json.dumps(value))
    assert read_prepared_training_data(saved, manifest) == value
    sources['train'][0].pop('selection_scenes')
    record.pop('selection_scenes'); record.pop('effective_split')
    saved.write_text(json.dumps(value))
    sources["train"][0]["demo_scope"] = "local_alignment"
    manifest.write_text(json.dumps({"schema_version":"AuboJointTrainingSourcesV1",**sources}))
    with pytest.raises(ValueError, match="demo_scope mismatch"):
        read_prepared_training_data(saved, manifest)
    sources["train"][0].pop("demo_scope")
    manifest.write_text(json.dumps({"schema_version":"AuboJointTrainingSourcesV1",**sources}))
    value["train"]["sources"][0]["source_dataset_root"] = str(tmp_path/"different")
    saved.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source/order"):
        read_prepared_training_data(saved, manifest)


def test_thin_prepared_data_requires_explicit_width_and_accepts_original_source_audit(tmp_path):
    import json
    from lerobot.bamboo_sorting.policy.joint_target_policy import target_policy_contract_record
    from lerobot.bamboo_sorting.policy.joint_training import read_prepared_training_data, read_data_manifest
    from tests.bamboo_sorting.capture_fakes import stats
    sources = {k:[{"root":k,"evidence_root":k+"_evidence","target_annotations":k+".json"}]
               for k in ("train","validation")}
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"schema_version":"AuboJointTrainingSourcesV1",**sources}))
    for split in sources:
        (tmp_path / split).mkdir()
        (tmp_path / (split+"_evidence")).mkdir()
        (tmp_path / (split+".json")).write_text("{}")
    resolved = read_data_manifest(manifest)
    value = {"contract":target_policy_contract_record(outline_width_px=1),"train_only_stats":stats(),
        **{k:{"sources":[{**resolved[k][0],"episodes":[0],"frames":1,"target_frames":[{}]}],"frames":1}
           for k in sources}}
    saved = tmp_path / "data_preflight.json"
    saved.write_text(json.dumps(value))
    kwargs = dict(target_conditioned=True,target_outline_width_px=1)
    assert read_prepared_training_data(saved,manifest,**kwargs)==value
    with pytest.raises(ValueError,match="contract"):
        read_prepared_training_data(saved,manifest,target_conditioned=True)
    value["contract"] = target_policy_contract_record()
    saved.write_text(json.dumps(value))
    assert read_prepared_training_data(saved,manifest,**kwargs)==value


def test_balanced_sampling_is_by_scope_not_folder_or_episode_length():
    from lerobot.bamboo_sorting.policy.joint_training import make_training_sampler, training_sampling_plan
    record = {"sources": [{"frames": 100}, {"frames": 900},
                          {"frames": 50, "demo_scope": "local_alignment"}], "frames": 1050}
    assert make_training_sampler(record, None, seed=42) is None
    assert training_sampling_plan(record)["expected_group_fractions"]["local_alignment"] == pytest.approx(50/1050)
    sampler = make_training_sampler(record, .5, seed=42)
    assert sampler.weights[:1000].sum().item() == pytest.approx(.5)
    assert sampler.weights[1000:].sum().item() == pytest.approx(.5)
    assert sampler.weights[0] == sampler.weights[999]
    first = list(sampler)
    assert first == list(make_training_sampler(record, .5, seed=42))
    assert first != list(sampler)  # Reusing the loader advances, not reseeds, each pass.
    assert abs(sum(i >= 1000 for i in first)/len(first) - .5) < .05
    with pytest.raises(ValueError, match="both full and local_alignment"):
        training_sampling_plan({"frames": 100}, .5)


@pytest.mark.parametrize("options", [["--local-demo-fraction", "0"], ["--local-demo-fraction", "nan"],
                                     ["--local-demo-fraction", ".5", "--stage", "evaluate"],
                                     ["--normalization-source", "base_checkpoint", "--stage", "smoke"]])
def test_invalid_sampling_and_normalization_stages_are_refused(options):
    with pytest.raises(SystemExit):
        _entry().parse_args(["--data-manifest", "sources.json", *options])


@pytest.mark.parametrize("normalization", ["train_data", "base_checkpoint"])
def test_balanced_sampling_supports_fresh_training_and_continuation(normalization):
    args = _entry().parse_args(["--data-manifest", "sources.json", "--local-demo-fraction", ".5",
                                "--normalization-source", normalization])
    assert args.local_demo_fraction == .5 and args.normalization_source == normalization
    with pytest.raises(SystemExit):
        _entry().parse_args(["--local-demo-fraction", ".5", "--normalization-source", normalization])


@pytest.mark.parametrize("options", [["--validation-every", "-1"],
    ["--validation-every", "10000"],
    ["--validation-every", "10000", "--target-conditioned", "--data-manifest", "sources.json", "--checkpoint-steps", "20000", "--steps", "60000"],
    ["--validation-seeds", "42", "42"], ["--validation-points-per-phase", "1"]])
def test_invalid_periodic_checkpoint_selection_is_refused(options):
    with pytest.raises(SystemExit):
        _entry().parse_args(options)


def test_saved_normalization_is_preserved_and_mismatched_inverse_is_rejected(tmp_path, monkeypatch):
    import json
    import torch
    from lerobot.bamboo_sorting.policy.smolvla_joint_adapter import (
        make_joint_smolvla_config, make_joint_smolvla_processors, load_joint_smolvla_processors,
    )
    from lerobot.bamboo_sorting.aubo_joint_contract import joint_contract_record, JOINT_TASK
    from tests.bamboo_sorting.capture_fakes import stats
    from tests.mocks.mock_tokenizer import MockTokenizerProcessorStep
    monkeypatch.setattr("lerobot.policies.smolvla.processor_smolvla.TokenizerProcessorStep", MockTokenizerProcessorStep)
    cfg = make_joint_smolvla_config()
    pre, post = make_joint_smolvla_processors(cfg, stats())
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    contract = joint_contract_record()
    (tmp_path/"aubo_joint_contract.json").write_text(json.dumps(contract))
    loaded_pre, loaded_post, effective = load_joint_smolvla_processors(tmp_path, cfg, contract=contract)
    raw = {"observation.state": torch.zeros(7), "action": torch.zeros(50, 7), "task": JOINT_TASK}
    assert effective == stats()
    torch.testing.assert_close(loaded_pre(raw)["action"], pre(raw)["action"], rtol=0, atol=0)
    torch.testing.assert_close(loaded_post(loaded_pre(raw)["action"]), raw["action"], rtol=0, atol=0)
    inconsistent = stats()
    inconsistent["action"]["mean"][6] = 70
    _, wrong_post = make_joint_smolvla_processors(cfg, inconsistent)
    wrong_post.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="inverse disagree"):
        load_joint_smolvla_processors(tmp_path, cfg, contract=contract)
    with pytest.raises(ValueError, match="contract differs"):
        load_joint_smolvla_processors(tmp_path, cfg, contract={})

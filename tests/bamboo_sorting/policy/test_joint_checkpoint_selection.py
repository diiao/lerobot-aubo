import json
from pathlib import Path
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.bamboo_sorting.policy.joint_checkpoint_selection import (
    SELECTION_RULE, build_monitor_plan, collect_action_chunks, evaluate_monitor, publish_best_checkpoint,
    score_action_chunks, selection_comparison,
)


def samples(groups=None):
    groups = groups or [("full", "a", 1, 1.), ("local_alignment", "b", 1, 2.)]
    observations, states, targets, predictions = [], [], [], []
    for source, (scope, scene, count, error) in enumerate(groups):
        for phase in ("preclose", "transport", "release"):
            for i in range(count):
                state = np.zeros(7); target = np.zeros((50, 7))
                if phase == "preclose":
                    target[2:, 6] = 100
                else:
                    state[6] = 100; target[:2, 6] = 100
                prediction = target.copy(); prediction[:, :6] += error
                observations.append(dict(dataset_index=len(observations), source_index=source,
                    episode_index=0, frame_index=len(observations), scene_id=scene, demo_scope=scope, phase=phase))
                states.append(state); targets.append(target); predictions.append(prediction)
    return observations, np.array(states), np.array(targets), np.array(predictions), np.zeros((len(states), 50), bool)


def candidate(primary=1., guard=.1):
    return dict(primary=primary, primary_name=SELECTION_RULE["primary"], seeds=[42, 43, 44], coverage=[("a", 0)],
                guards={"preclose_joint_mae_deg": guard, "close_timing_steps": 0., "release_timing_steps": 0.})


def test_macro_balancing_does_not_favor_long_easy_scenes_or_the_larger_scope():
    values = samples([("full", "long", 100, 0.), ("full", "hard", 1, 10.),
                      ("local_alignment", "local", 1, 20.)])
    metrics = score_action_chunks(*values)
    assert metrics["primary"] == pytest.approx(((0+10)/2+20)/2)
    assert metrics["guards"]["full_precontact_joint_mae_deg"] == 5
    assert metrics["guards"]["local_alignment_precontact_joint_mae_deg"] == 20
    assert metrics["guards"]["worst_scene_preclose_joint_mae_deg"] == 20


def test_future_rows_detect_bad_chunks_hidden_by_the_first_row_and_ignore_padding():
    observations, states, targets, predictions, padding = samples()
    predictions[:, 0, :6] = targets[:, 0, :6]
    predictions[:, 1:20, :6] = targets[:, 1:20, :6]+2
    predictions[:, 40:, :6] = 1e6; padding[:, 40:] = True
    assert score_action_chunks(observations, states, targets, predictions, padding)["primary"] == pytest.approx(1.9)
    padding[:] = True
    with pytest.raises(ValueError, match="coverage"):
        score_action_chunks(observations, states, targets, predictions, padding)


def test_missing_gripper_transitions_are_penalized_instead_of_reporting_zero_timing_error():
    observations, states, targets, predictions, padding = samples()
    predictions[:, :, 6] = 0
    result = score_action_chunks(observations, states, targets, predictions, padding)
    assert result["guards"]["close_timing_steps"] == 50
    assert result["diagnostics"]["close_events"]["missing"] > 0


def test_better_primary_must_pass_guards_and_regressions_cannot_accumulate():
    best = candidate(); best["guard_reference"] = dict(best["guards"])
    improved = candidate(.9, .14)
    assert selection_comparison(improved, best)["improved"]
    improved["guard_reference"] = best["guard_reference"]
    drifting = candidate(.8, .19)
    result = selection_comparison(drifting, improved)
    assert not result["improved"] and result["blocked_guards"] == ["preclose_joint_mae_deg"]
    bad_timing = candidate(.9); bad_timing["guards"]["release_timing_steps"] = 1.01
    assert selection_comparison(bad_timing, best)["blocked_guards"] == ["release_timing_steps"]
    assert not selection_comparison(candidate(), best)["improved"]
    improved["validation_loss"] = 1e20
    improved["close_event_errors_per_window"] = improved["release_event_errors_per_window"] = 1000
    assert selection_comparison(improved, best)["improved"]
    for key, value in (("seeds", [1]), ("coverage", []), ("primary", float("nan"))):
        bad = {**improved, key: value}
        with pytest.raises(ValueError):
            selection_comparison(bad, best)


def test_real_parquet_monitor_covers_all_scenes_phases_and_event_boundaries(tmp_path):
    root = tmp_path / "data"; (root / "data/chunk-000").mkdir(parents=True)
    actions = np.zeros((180, 7)); actions[80:150, 6] = 100
    states = actions.copy(); states[:, 6] = np.r_[0., actions[:-1, 6]]
    states[10:, 0] = np.arange(170)*.3+.3
    values = {"episode_index": [0]*180, "frame_index": list(range(180)),
              "observation.state": states.tolist(), "action": actions.tolist()}
    pq.write_table(pa.table(values), root / "data/chunk-000/file-000.parquet")
    source = dict(root=str(root), episodes=[0], frames=180, target_episode_metadata=[dict(episode_index=0, scene_id="a")],
                  target_frames=[dict(episode_index=0, frame_index=i) for i in range(180)])
    record = dict(sources=[source], frames=180, scenes=["a"])
    plan = build_monitor_plan(record)
    assert plan == build_monitor_plan(record)
    assert plan["scenes"] == ["a"] and plan["episodes"] == 1 and plan["frames"] < 180
    assert {r["phase"] for r in plan["observations"]} == set(("initial_hold", "onset", "approach", "preclose", "transport", "release"))
    assert {61, 70, 79, 80, 81, 90, 131, 140, 149, 150, 151, 160} <= set(plan["indices"])
    values["action"] = np.zeros((180, 7)).tolist()
    pq.write_table(pa.table(values), root / "data/chunk-000/file-000.parquet")
    with pytest.raises(ValueError, match="one expert close"):
        build_monitor_plan(record)


def test_monitor_preserves_training_rng_mode_and_queues_and_repeats_paired_noise():
    observations, states, targets, _, padding = samples()
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self._queues = {"old": [1]}
        def forward(self, batch, **kwargs):
            random.random(); np.random.rand()
            return torch.rand(len(batch["episode_index"])), {}
        def predict_action_chunk(self, batch):
            self.eval(); self._queues = {"changed": True}
            return torch.randn(len(batch["observation.state"]), 50, 7)
    class Data(torch.utils.data.Dataset):
        def __len__(self):return len(observations)
        def __getitem__(self, i):
            row = observations[i]
            return {"observation.state": torch.tensor(states[i], dtype=torch.float32),
                    "action": torch.tensor(targets[i], dtype=torch.float32), "action_is_pad": torch.tensor(padding[i]),
                    "episode_index": row["episode_index"], "frame_index": row["frame_index"], "task": "test"}
    policy = Policy().train()
    torch.manual_seed(18); random.seed(18); np.random.seed(18)
    torch_state, py_state, np_state = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    first = evaluate_monitor(policy, lambda x:x, lambda x:x, Data(), observations, batch_size=2, seeds=[42,43,44])
    second = evaluate_monitor(policy, lambda x:x, lambda x:x, Data(), observations, batch_size=2, seeds=[42,43,44])
    assert first == second and policy.training and policy._queues == {"old": [1]}
    assert torch.equal(torch_state, torch.get_rng_state()) and py_state == random.getstate()
    np.testing.assert_equal(np_state, np.random.get_state())
    assert first["seeds"] == [42,43,44] and first["primary_seed_std"] > 0


def test_compact_collection_owns_storage_and_partitioned_rng_matches_one_pass():
    import gc
    import weakref
    observations, states, targets, _, padding = samples()
    references = []

    class Policy(torch.nn.Module):
        def forward(self, batch, **kwargs):
            references.append(weakref.ref(batch["action"]))
            return torch.rand(len(batch["episode_index"])), {}

        def predict_action_chunk(self, batch):
            return torch.randn(len(batch["observation.state"]), 50, 7)

    class Data(torch.utils.data.Dataset):
        def __len__(self):
            return len(observations)

        def __getitem__(self, i):
            row = observations[i]
            return {"observation.state": torch.tensor(states[i], dtype=torch.float32),
                    "action": torch.tensor(targets[i], dtype=torch.float32),
                    "action_is_pad": torch.tensor(padding[i]), "episode_index": row["episode_index"],
                    "frame_index": row["frame_index"], "task": "test"}

    policy, data = Policy(), Data()
    torch.manual_seed(42)
    full = collect_action_chunks(policy, lambda x:x, lambda x:x, data, observations,
                                 batch_size=2, seed=42)
    gc.collect()
    assert all(a.flags.owndata for a in full)
    assert all(ref() is None for ref in references)
    torch.manual_seed(42)
    first = collect_action_chunks(policy, lambda x:x, lambda x:x, data, observations[:4],
                                  batch_size=2, seed=42)
    continuation = torch.get_rng_state()
    # Model construction in a new process consumes randomness before restoration.
    torch.rand(123)
    torch.set_rng_state(continuation)
    second = collect_action_chunks(policy, lambda x:x, lambda x:x, data, observations[4:],
                                   batch_size=2, seed=42)
    for expected, left, right in zip(full, first, second, strict=True):
        np.testing.assert_array_equal(expected, np.concatenate([left, right]))
    bad = [dict(r) for r in observations]; bad[0]["frame_index"] = -1
    with pytest.raises(ValueError, match="identity/order"):
        collect_action_chunks(policy, lambda x:x, lambda x:x, data, bad, batch_size=2, seed=42)


def writer(path, step):
    path.mkdir(); (path / "model.safetensors").write_bytes(str(step).encode())


def verifier(path):
    assert (path / "model.safetensors").read_bytes()
    return dict(strict_reload=True, state_dict_equal=True)


def test_new_best_is_verified_before_old_best_deleted_and_final_is_preserved(tmp_path):
    output = tmp_path / "run"; output.mkdir()
    final = output / "final"; final.mkdir(); (final / "model.safetensors").write_bytes(b"final")
    first = publish_best_checkpoint(output, 10, candidate(), None, writer, verifier)
    def checked(path):
        assert first.is_dir()
        assert json.loads((output / "best_checkpoint.json").read_text())["optimizer_steps"] == 10
        return verifier(path)
    second = publish_best_checkpoint(output, 20, candidate(.9), first, writer, checked)
    assert not first.exists() and second.is_dir() and (final / "model.safetensors").read_bytes() == b"final"
    assert json.loads((output / "best_checkpoint.json").read_text())["optimizer_steps"] == 20
    assert [p.name for p in (output / "checkpoints").iterdir()] == ["best_step_000020"]


@pytest.mark.parametrize("failure", ["save", "reload"])
def test_failed_candidate_save_or_reload_keeps_previous_best_and_pointer(tmp_path, failure):
    output = tmp_path / "run"; output.mkdir()
    first = publish_best_checkpoint(output, 10, candidate(), None, writer, verifier)
    before = (output / "best_checkpoint.json").read_bytes()
    def broken_save(path, step):
        writer(path, step); raise OSError("disk full")
    def broken_reload(path):return dict(strict_reload=True, state_dict_equal=False)
    with pytest.raises((OSError, ValueError)):
        publish_best_checkpoint(output, 20, candidate(.9), first,
                                broken_save if failure == "save" else writer,
                                broken_reload if failure == "reload" else verifier)
    assert first.is_dir() and (output / "best_checkpoint.json").read_bytes() == before


def test_retention_rejects_unowned_and_symlinked_checkpoints(tmp_path):
    output = tmp_path / "run"; output.mkdir(); (output / "checkpoints").mkdir()
    protected = tmp_path / "historic"; protected.mkdir(); (protected / "keep").write_text("original")
    link = output / "checkpoints/best_step_000010"; link.symlink_to(protected, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        publish_best_checkpoint(output, 20, candidate(), link, writer, verifier)
    assert (protected / "keep").read_text() == "original"
    other = output / "checkpoints/best_step_000011"; other.mkdir()
    (other / "checkpoint_owner.json").write_text(json.dumps(dict(output="another run", kind="best", step=11)))
    with pytest.raises(ValueError, match="ownership"):
        publish_best_checkpoint(output, 20, candidate(), other, writer, verifier)

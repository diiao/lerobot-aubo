import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import ConcatDataset, Subset

from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_TASK
from lerobot.bamboo_sorting.policy import pi05_joint_training as training
from tests.bamboo_sorting.paths import REPO_ROOT


def test_fresh_training_preserves_official_fp32_base_values(tmp_path, monkeypatch):
    from safetensors.torch import save_model

    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    group = policy.model.paligemma_with_expert = torch.nn.Module()
    group.paligemma = torch.nn.Linear(2, 2).requires_grad_(False)
    group.gemma_expert = torch.nn.Linear(2, 2)
    original = torch.full_like(group.gemma_expert.weight, .10001)
    with torch.no_grad():
        group.gemma_expert.weight.copy_(original)
    base = tmp_path / "base"
    base.mkdir()
    save_model(policy.model, base / "model.safetensors")
    policy.bfloat16()
    assert not torch.equal(group.gemma_expert.weight.float(), original)

    class Initialized(Exception):
        pass

    def inspect_initialized_policy():
        assert group.gemma_expert.weight.dtype == torch.float32
        torch.testing.assert_close(group.gemma_expert.weight, original, rtol=0, atol=0)
        assert group.paligemma.weight.dtype == torch.bfloat16
        raise Initialized

    monkeypatch.setitem(sys.modules, "lerobot.policies.pi05.modeling_pi05",
                        SimpleNamespace(PI05Policy=lambda config: policy))
    monkeypatch.setattr(policy, "train", inspect_initialized_policy)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(training, "make_joint_pi05_processors", lambda *args: (None, None))
    monkeypatch.setattr(training, "load_prepared_dataset", lambda value: [0])
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    (preflight / "train_quantile_stats.json").write_text("{}")
    args = SimpleNamespace(stage="train", precision="fp32-expert", seed=42,
                           output=tmp_path / "output", preflight=preflight, base_path=base)
    with pytest.raises(Initialized):
        training.run(args, {"storage": {"fits": True}}, {"train": {}, "validation": {}},
                     {"tokenizer_path": "unused"}, SimpleNamespace())


def test_probe_does_not_update_weights_but_train_uses_same_gradient_path(monkeypatch):
    policy = torch.nn.Linear(1, 1, bias=False)
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    policy.register_parameter("frozen", frozen)
    with torch.no_grad():
        policy.weight.fill_(1)
    monkeypatch.setattr(training, "joint_pi05_loss", lambda model, batch: model(batch["x"]).square().mean())
    raw = {"task": [JOINT_TASK], "x": torch.ones(1, 1)}
    initial = policy.weight.detach().clone()
    probe = training.backward_batch(policy, lambda value: value, raw, clip_norm=1)
    assert probe["optimizer_steps"] == 0
    torch.testing.assert_close(policy.weight, initial, rtol=0, atol=0)
    assert all(p.grad is None for p in policy.parameters())
    optimizer = torch.optim.AdamW([policy.weight], lr=.01, foreach=False)
    update = training.backward_batch(policy, lambda value: value, raw, clip_norm=1, optimizer=optimizer)
    assert update["optimizer_steps"] == 1 and policy.weight.item() < initial.item()
    assert policy.frozen.item() == 1 and policy.frozen.grad is None


def test_nonfinite_batch_stops_before_optimizer_update(monkeypatch):
    policy = torch.nn.Linear(1, 1)
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.01)
    monkeypatch.setattr(training, "joint_pi05_loss", lambda model, batch: model.weight.sum() * float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        training.backward_batch(policy, lambda value: value, {"task": [JOINT_TASK]}, clip_norm=1, optimizer=optimizer)
    assert not optimizer.state
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_storage_budget_reserves_all_full_checkpoints_and_optimizer():
    smoke = {"checkpoint_bytes": 7473096344, "trainable_parameters": 693422112}
    one, two = (training.storage_budget(smoke, n) for n in (1, 2))
    assert two["required_free_bytes"] - one["required_free_bytes"] >= smoke["checkpoint_bytes"]
    assert two["optimizer_bytes_upper_bound"] >= smoke["trainable_parameters"] * 8
    assert 23 * training.GIB < two["required_free_bytes"] < 27 * training.GIB
    corrected = training.storage_budget(smoke, 2, fp32_expert=True)
    assert corrected['checkpoint_bytes_budget']-two['checkpoint_bytes_budget'] == smoke['trainable_parameters']*2
    bounded = training.storage_budget(smoke, 1, fp32_expert=True, save_optimizer=False)
    assert bounded['optimizer_bytes_upper_bound'] == 0


def test_outputs_never_overlap_or_overwrite_sources(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    for output in (source, source / "run", tmp_path):
        with pytest.raises(ValueError):
            training.require_new_output(output, [source])
    training.require_new_output(tmp_path / "new-run", [source])
    assert not (tmp_path / "new-run").exists()


def test_checkpoint_tokenizer_path_survives_atomic_rename(tmp_path, monkeypatch):
    source = tmp_path / "tokenizer-source"
    source.mkdir()
    (source / "tokenizer_config.json").write_text('{}')
    class Processor:
        def __init__(self, path, filename):
            self.steps = [SimpleNamespace(tokenizer_name=str(path))]
            self.filename = filename
        def save_pretrained(self, path):
            (path / self.filename).write_text(json.dumps({"tokenizer": self.steps[0].tokenizer_name}))
    monkeypatch.setattr(training, "make_joint_pi05_processors",
                        lambda config, stats, path: (Processor(path,"pre.json"), Processor(path,"post.json")))
    monkeypatch.setattr(training.shutil, "disk_usage", lambda path: SimpleNamespace(free=100*training.GIB))
    model = SimpleNamespace(save_pretrained=lambda path: (path / "model.safetensors").write_bytes(b"test"),
                            aubo_joint_precision=training.FP32_EXPERT_PRECISION)
    destination = tmp_path / "checkpoint"
    plan = {"storage":{"checkpoint_bytes_budget":8*training.GIB},"preflight":"preflight"}
    provenance = {"tokenizer_path":str(source),"contract":{},"data_manifest":"manifest","prepared_data":"prepared"}
    training._save_checkpoint(model, destination, None, {}, provenance, 15, plan)
    assert not (tmp_path / "checkpoint.incomplete").exists()
    assert Path(json.loads((destination / "pre.json").read_text())["tokenizer"]).is_dir()
    assert json.loads((destination / "training_progress.json").read_text())["optimizer_steps"] == 15
    assert json.loads((destination / training.PRECISION_FILE).read_text()) == training.FP32_EXPERT_PRECISION
    with pytest.raises(FileExistsError):
        training._save_checkpoint(model, destination, None, {}, provenance, 15, plan)


def test_entry_defaults_to_plan_and_rejects_invalid_checkpoint_budget():
    path = REPO_ROOT / "examples/phone_to_auboi10/train_joint_pi05.py"
    spec = importlib.util.spec_from_file_location("pi05_training_entry_test", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    assert entry.parse_args([]).stage == "plan"
    assert not entry.parse_args([]).early_stopping
    assert entry.parse_args(['--stage','validate']).steps == 128
    assert entry.parse_args([]).precision == 'fp32-expert'
    assert entry.parse_args([]).batch_size == 16
    assert entry.parse_args(['--stage', 'validate']).batch_size == 8
    for args in (["--batch-size", "0"], ["--learning-rate", "nan"], ["--checkpoint-steps", "30000"],
                 ["--checkpoint-steps", "100", "100"], ["--eval-interval", "0"],
                 ["--eval-frames-per-episode", "1"], ['--stage','validate','--steps','257']):
        with pytest.raises(SystemExit):
            entry.parse_args(args)


def test_monitor_samples_each_episode_and_close_boundary_without_changing_data():
    frames = [{"episode_index": e, "frame_index": f, "transport": f >= 4}
              for e in (0, 1) for f in range(9)]
    source = {"episodes": [0, 1], "frames": 18, "target_frames": frames}
    record = {"sources": [source, source], "frames": 36}
    before = json.dumps(record)
    indices = training.monitoring_indices(record, 2)
    assert indices == [offset + f for offset in (0, 9, 18, 27) for f in (0, 2, 3, 4, 5, 6, 8)]
    assert json.dumps(record) == before


def test_early_stop_requires_persistent_regression_in_both_metrics_after_minimum():
    best = {"normalized_action_mae": 1., "validation_loss": 1.}
    worse = {"normalized_action_mae": 1.3, "validation_loss": 1.3}
    assert training.validation_decision(worse, best, 0, 4000)[1:3] == (0, False)
    count = 0
    for step in (5000, 6000, 7000):
        _, count, stop, improved = training.validation_decision(worse, best, count, step)
        assert stop == (step == 7000) and not improved
    assert training.validation_decision(worse, best, 2, 7000, early_stopping=False)[1:3] == (3, False)
    one_metric = {"normalized_action_mae": 1.3, "validation_loss": .8}
    assert training.validation_decision(one_metric, best, 2, 7000)[1:3] == (0, False)
    better = {"normalized_action_mae": .9, "validation_loss": .9}
    assert training.validation_decision(better, best, 2, 7000) == (better, 0, False, True)


def test_evaluation_preserves_training_rng_mode_and_subset_source_identity(tmp_path, monkeypatch):
    class Policy(torch.nn.Module):
        def predict_action_chunk(self, batch):
            assert not self.training
            return batch["observation.state"][:, None, :] + torch.rand(len(batch["task"]), 50, 7)
    def sample(i):
        return {"observation.state": torch.full((7,), float(i)), "action": torch.zeros(50, 7),
                "action_is_pad": torch.zeros(50, dtype=torch.bool), "task": JOINT_TASK,
                "episode_index": 0, "frame_index": i}
    dataset = ConcatDataset([[sample(i) for i in range(6)], [sample(i) for i in range(7)]])
    dataset.aubo_evaluation_events = {(i, 0): dict(onset=0, close=5, release=100) for i in range(2)}
    subset = Subset(dataset, [0, 2, 6, 9, 12])
    monkeypatch.setattr(training, "joint_pi05_loss", lambda policy, batch: torch.rand(()))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    policy = Policy().train()
    state = torch.random.get_rng_state().clone()
    geometry = tmp_path/'geometry.json'
    geometry.write_text(json.dumps(dict(purpose='offline evaluation only', tool_offset_m=[1, 0, 0],
        joints=[dict(origin=training.np.eye(4).tolist(), axis=[0, 1, 0]) for _ in range(6)])))
    args = SimpleNamespace(seed=42, batch_size=2, evaluation_geometry=geometry)
    stats = {"action": {"q01": [0]*7, "q99": [100]*7}}
    first = training._evaluate(policy, lambda x: x, lambda x: x, subset, args, stats, output=tmp_path / "first")
    assert policy.training and torch.equal(torch.random.get_rng_state(), state)
    second = training._evaluate(policy, lambda x: x, lambda x: x, subset, args, stats, output=tmp_path / "second")
    assert first == second and first["frames"] == 5
    saved = training.np.load(tmp_path / "first/heldout_predictions.npz")
    assert saved["source_index"].tolist() == [0, 0, 1, 1, 1]
    assert saved["frame_index"].tolist() == [0, 2, 0, 3, 6]
    assert saved['action_chunks'].shape == (5,50,7)
    assert first['execution_window_valid_steps'] == 60
    assert first['offline_selection'] is not None
    preclose = [r for r in first['episode_phase_metrics'] if r['phase'] == 'preclose']
    assert {r['source_index'] for r in preclose} == {0, 1}
    assert first['phase_metrics']['preclose']['episodes'] == 2


def test_model_selection_scores_the_future_execution_window():
    best = dict(normalized_action_mae=.01,execution_window_normalized_action_mae=.2,validation_loss=.1)
    improved = dict(normalized_action_mae=.02,execution_window_normalized_action_mae=.1,validation_loss=.1)
    assert training.validation_decision(improved,best,0,1000)[3]


def test_tcp_selection_rejects_gripper_masking_and_guard_regressions():
    def metrics(tcp, joint, grip):
        return dict(validation_loss=1., execution_window_normalized_action_mae=grip,
                    offline_selection=dict(primary=tcp, primary_name='TCP',
                        available_precontact_phases=['preclose'], guards=dict(joints=joint, close_failure=0.)))
    best = metrics(8.38, .407, .04073)
    # The real failure mode: combined action score improves while TCP gets worse.
    assert not training.validation_decision(metrics(9.12, .426, .03678), best, 0, 1000)[3]
    # Better TCP alone cannot hide an excessive joint regression.
    blocked = training.selection_comparison(metrics(7., .6, .03), best)
    assert not blocked['improved'] and blocked['blocked_guards'] == ['joints']
    assert training.validation_decision(metrics(7., .4, .06), best, 0, 1000)[3]
    false_close = metrics(7., .4, .03)
    false_close['offline_selection']['guards']['close_failure'] = .1
    assert training.selection_comparison(false_close, best)['blocked_guards'] == ['close_failure']
    with pytest.raises(ValueError, match='coverage'):
        training.selection_comparison({'offline_selection': None}, None)
    for step in (5000, 6000, 7000):
        worse = metrics(12., .6, .02)
        worse['validation_loss'] = 2.
        assert not training.validation_decision(worse, best, 3, step, early_stopping=False)[2]


def test_event_counts_are_diagnostic_without_weakening_movement_or_timing_guards():
    def metrics(position, **updates):
        guards = dict(joints=.5, close_event_errors_per_window=0., close_timing_with_misses_steps=8.,
                      release_timing_with_misses_steps=8., release_event_errors_per_window=.074)
        guards.update(updates)
        return dict(validation_loss=1., offline_selection=dict(primary=position, primary_name='TCP',
            available_precontact_phases=['preclose'], guards=guards))

    best = metrics(15.8)
    candidate = metrics(11.1, close_event_errors_per_window=.0222, release_event_errors_per_window=.296)
    decision = training.selection_comparison(candidate, best)
    assert decision['improved'] and not decision['blocked_guards']
    assert decision['diagnostic_only_guards'] == ['close_event_errors_per_window', 'release_event_errors_per_window']
    assert candidate['offline_selection']['guards']['close_event_errors_per_window'] == .0222
    assert candidate['offline_selection']['guards']['release_event_errors_per_window'] == .296
    for key, value in (('joints', .7), ('close_timing_with_misses_steps', 10.),
                       ('release_timing_with_misses_steps', 10.)):
        blocked = training.selection_comparison(metrics(11.1, **{key: value}), best)
        assert not blocked['improved'] and blocked['blocked_guards'] == [key]
    assert not training.selection_comparison(metrics(16., release_event_errors_per_window=0.), best)['improved']


def test_formal_plan_and_training_require_offline_geometry_before_loading_any_model():
    for stage in ('plan', 'train'):
        with pytest.raises(ValueError, match='evaluation-geometry'):
            training.build_plan(SimpleNamespace(stage=stage, evaluation_geometry=None))


def test_small_guard_regressions_cannot_compound_past_the_initial_budget():
    def metrics(position, pose):
        return dict(validation_loss=1., offline_selection=dict(primary=position, primary_name='TCP',
            available_precontact_phases=['preclose'], guards=dict(pose=pose)))
    baseline = metrics(10., 1.)
    best, _, _, improved = training.validation_decision(metrics(9., 1.19), baseline, 0, 1000)
    assert improved
    # 1.4 < 1.19*1.2, but 1.4 exceeds the original 1.0*1.2 budget.
    assert not training.validation_decision(metrics(8., 1.4), best, 0, 2000)[3]


def test_monitor_includes_every_phase_and_overlapping_plan_observations():
    frames = [dict(episode_index=0, frame_index=f, transport=f >= 70) for f in range(120)]
    source = dict(episodes=[0], frames=120, target_frames=frames)
    events = {(0, 0): dict(onset=5, close=70, release=105)}
    indices = training.monitoring_indices(source, 12, events=events)
    labels = [training.evaluation_phase(f, events[(0, 0)]) for f in indices]
    assert set(labels) == set(training.EVALUATION_PHASES)
    assert all(labels.count(p) >= 2 for p in training.EVALUATION_PHASES)
    assert any(f+8 in indices and training.evaluation_phase(f, events[(0, 0)]) ==
               training.evaluation_phase(f+8, events[(0, 0)]) for f in indices)


def test_expert_events_use_accepted_action_instead_of_later_observed_gripper_state():
    states = training.np.zeros((8, 7)); actions = states.copy()
    states[2:, 0] = .3
    actions[3:6, 6] = 100
    states[4:7, 6] = 100  # Observation reflects the previous accepted command.
    assert training.episode_event_frames(states, actions, training.np.arange(10, 18)) == dict(onset=12, close=13, release=16)


@pytest.mark.parametrize('inconsistent_seed, corrupt_reload', [(False, False), (True, False), (False, True)])
def test_both_saved_candidates_are_reloaded_and_fully_reviewed_with_matched_seeds(tmp_path, monkeypatch, inconsistent_seed, corrupt_reload):
    best = tmp_path/'checkpoints/best_step_001000'
    training.write_json(tmp_path/'best_checkpoint.json', {'optimizer_steps': 1000})
    identities = dict(source_index=training.np.array([0]), episode_index=training.np.array([0]),
                      frame_index=training.np.array([25]), action_chunks=training.np.zeros((1, 50, 7)))
    expected = tmp_path/'validation/step_001000'
    expected.mkdir(parents=True)
    training.np.savez(expected/'heldout_predictions.npz', **identities)
    loaded, calls = [], []
    def load(path):
        loaded.append(path)
        return SimpleNamespace(label='best' if path == best else 'final'), lambda x: x, lambda x: x
    monkeypatch.setattr(training, '_load_saved_policy', load)
    monkeypatch.setattr(training, 'predict_joint_pi05_chunk', lambda *args: torch.zeros(1, 50, 7))
    validation, monitor = list(range(13)), list(range(3))
    def evaluate(policy, pre, post, dataset, args, stats, *, output):
        calls.append((policy.label, args.seed, len(dataset)))
        output.mkdir(parents=True, exist_ok=True)
        saved = {key: value.copy() for key, value in identities.items()}
        if corrupt_reload and output.name == 'best_reload_monitor':
            saved['action_chunks'][0, 0, 0] = 1
        training.np.savez(output/'heldout_predictions.npz', **saved)
        score = 2. if policy.label == 'best' else 1.
        if inconsistent_seed and policy.label == 'final' and args.seed == 43:
            score = 3.
        return dict(offline_selection=dict(primary=score, primary_name='TCP', guards={'pose': 1.},
                    available_precontact_phases=['preclose']))
    monkeypatch.setattr(training, '_evaluate', evaluate)
    args = SimpleNamespace(seed=42, output=tmp_path)
    if corrupt_reload:
        with pytest.raises(AssertionError):
            training._review_saved_checkpoints(args, {}, validation, monitor, best,
                {'action': torch.zeros(1, 50, 7)}, torch.zeros(1, 50, 7))
        assert not (tmp_path/'full_validation/comparison.json').exists()
        return
    report = training._review_saved_checkpoints(args, {}, validation, monitor, best,
        {'action': torch.zeros(1, 50, 7)}, torch.zeros(1, 50, 7))
    assert loaded == [tmp_path/'final', best]
    assert [c for c in calls if c[2] == 13] == [(label, seed, 13) for label in ('final', 'best') for seed in (42, 43, 44)]
    assert ('best', 42, 3) in calls
    assert report['consistent_offline_preference'] == ('inconclusive' if inconsistent_seed else 'final')
    assert json.loads((tmp_path/'reload_check.json').read_text())['same_seed_equal']
    assert not report['automatic_deployment'] and not report['physical_grasp_verified']


def test_precision_subset_contains_onsets_and_contact_neighbors_in_training_only():
    from lerobot.bamboo_sorting.policy.pi05_joint_validation import precision_training_indices
    source = dict(episodes=[0],frames=90,target_frames=[dict(episode_index=0,frame_index=i,transport=i>=50) for i in range(90)])
    record = dict(sources=[source,source,source],frames=270)
    indices = precision_training_indices(record,{(i,0):5 for i in range(3)})
    assert indices == [offset+i for offset in (0,90,180) for i in (5,25,30,40,45,49,75,80)]


def test_best_replacement_keeps_previous_until_new_save_succeeds(tmp_path, monkeypatch):
    parent = tmp_path / "checkpoints"
    parent.mkdir()
    previous, destination = parent / "best_step_000000", parent / "best_step_001000"
    previous.mkdir()
    protected = parent / "step_000500"
    protected.mkdir()
    def failing_save(*args):
        raise OSError("disk write failed")
    monkeypatch.setattr(training, "_save_checkpoint", failing_save)
    with pytest.raises(OSError):
        training._save_best(None, destination, previous, None, {}, {}, 1000, {}, {})
    assert previous.is_dir() and protected.is_dir()
    monkeypatch.setattr(training, "_save_checkpoint", lambda policy, path, *args: path.mkdir())
    training._save_best(None, destination, previous, None, {}, {}, 1000, {}, {})
    assert destination.is_dir() and protected.is_dir() and not previous.exists()
    assert json.loads((tmp_path / "best_checkpoint.json").read_text())["path"] == str(destination)

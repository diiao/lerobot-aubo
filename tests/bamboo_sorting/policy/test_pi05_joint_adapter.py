from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.bamboo_sorting.policy.pi05_joint_adapter import (
    FP32_EXPERT_PRECISION, PRECISION_FILE, configure_joint_pi05_precision,
    PI05JointTargetOfflineAdapter, load_joint_pi05_weights, make_joint_pi05_config, masked_joint_loss, train_quantile_stats,
)


def test_pi05_runtime_preserves_joint_actions_and_validates_phase_before_policy():
    from lerobot.bamboo_sorting.aubo_joint_contract import JOINT_TASK
    from lerobot.bamboo_sorting.policy.joint_target_policy import target_policy_contract_record
    observed = []
    actions = torch.arange(350, dtype=torch.float32).reshape(1, 50, 7) / 10
    actions[:, :25, 6] = 49.9
    actions[:, 25:, 6] = 50.0
    def predict(batch):
        observed.append(batch)
        return actions.clone()
    policy = SimpleNamespace(config=make_joint_pi05_config(), predict_action_chunk=predict)
    adapter = PI05JointTargetOfflineAdapter(policy, lambda x:x, lambda x:x, contract=target_policy_contract_record())
    frame = {'observation.state':np.zeros(7,dtype=np.float32), 'task':JOINT_TASK,
             'target_contract':target_policy_contract_record(),
             'target_input':dict(target_id='B',frame_id='recorded/1',phase='approach'),
             **{f'observation.images.{key}':np.full((480,640,3),128,np.uint8) for key in adapter.image_keys}}
    decoded = adapter(frame)
    np.testing.assert_array_equal(decoded[:,:6], actions[0,:,:6])
    np.testing.assert_array_equal(adapter.raw_actions, actions[0])
    assert np.all(decoded[:25,6] == 0) and np.all(decoded[25:,6] == 100)
    assert set(observed[0]) == {'observation.state','task',*policy.config.image_features}
    assert observed[0]['observation.images.target_global_rgb'].shape == (3,480,640)
    assert observed[0]['observation.images.target_global_rgb'][0,0,0] == pytest.approx(128/255)
    with pytest.raises(ValueError, match='explicit phase'):
        adapter({**frame, 'target_input':dict(target_id='B',frame_id='recorded/1',phase='predicted_close')})
    assert len(observed) == 1 and adapter.raw_actions is None
    policy.config.dtype = 'float32'
    with pytest.raises(ValueError, match='PI05 inference configuration'):
        PI05JointTargetOfflineAdapter(policy, lambda x:x, lambda x:x, contract=target_policy_contract_record())


def test_joint_config_preserves_units_and_limits_training_to_expert():
    config = make_joint_pi05_config()
    assert config.input_features["observation.state"].shape == (7,)
    assert config.output_features["action"].shape == (7,)
    assert set(config.image_features) == {"observation.images.target_global_rgb", "observation.images.grasp_rgb"}
    assert config.chunk_size == 50 and config.n_action_steps == 1
    assert config.train_expert_only and config.freeze_vision_encoder and config.gradient_checkpointing
    assert not config.push_to_hub and not config.compile_model


@pytest.mark.parametrize('kind', ['pi05', 'smolvla'])
def test_stdio_registers_checkpoint_config_in_fresh_process(tmp_path, kind):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from lerobot.bamboo_sorting.policy.smolvla_joint_adapter import make_joint_smolvla_config
    config = make_joint_pi05_config() if kind == 'pi05' else make_joint_smolvla_config(target_conditioned=True)
    config.save_pretrained(tmp_path)
    entry = Path(__file__).resolve().parents[3] / 'examples/phone_to_auboi10/joint_inference_stdio.py'
    code = "import runpy,sys; m=runpy.run_path(sys.argv[1]); print(m['load_checkpoint_config'](sys.argv[2]).type)"
    result = subprocess.run([sys.executable,'-c',code,str(entry),str(tmp_path)],
        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':''},
        capture_output=True,text=True,check=True,timeout=30)
    assert result.stdout.strip() == kind


def test_loss_excludes_tail_and_padded_coordinates_from_gradients():
    losses = torch.arange(2 * 3 * 32, dtype=torch.float32).reshape(2, 3, 32).requires_grad_()
    padding = torch.tensor([[False, False, True], [False, True, True]])
    loss = masked_joint_loss(losses, padding)
    assert torch.equal(loss, torch.cat([losses[0, :2, :7].flatten(), losses[1, 0, :7]]).mean())
    loss.backward()
    assert torch.count_nonzero(losses.grad[..., 7:]) == 0
    assert torch.count_nonzero(losses.grad[padding]) == 0
    torch.testing.assert_close(losses.grad[..., :7][~padding], torch.full((3, 7), 1/21))


@pytest.mark.parametrize("padding", [torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, 3), torch.zeros(2, 2, dtype=torch.bool)])
def test_loss_rejects_empty_or_incorrect_padding(padding):
    with pytest.raises(ValueError):
        masked_joint_loss(torch.ones(2, 3, 32), padding)


def _prepared(tmp_path):
    root = tmp_path / "training"
    (root / "data/chunk-000").mkdir(parents=True)
    # Selected episode varies in all seven coordinates; excluded episode is an extreme outlier.
    good = np.arange(100, dtype=float)[:, None] + np.arange(7)[None, :]
    values = np.concatenate([good, np.full((1, 7), 99999.0)])
    pq.write_table(pa.table({"episode_index": [0]*100 + [1],
                            "observation.state": values.tolist(), "action": values.tolist()}),
                   root / "data/chunk-000/file-000.parquet")
    stats = {key: {"mean": good.mean(0).tolist(), "std": good.std(0).tolist(),
                   "min": good.min(0).tolist(), "max": good.max(0).tolist()}
             for key in ("action", "observation.state")}
    return {"train": {"sources": [{"root": str(root), "episodes": [0], "frames": 100}],
                      "frames": 100, "scenes": ["p-001"]},
            # This nonexistent validation path must never be read to calculate training stats.
            "validation": {"root": str(tmp_path/"do-not-read-validation"), "scenes": ["p-101"]},
            "train_only_stats": stats}, good


def test_quantiles_exclude_failed_episodes_and_never_read_validation(tmp_path):
    prepared, good = _prepared(tmp_path)
    stats = train_quantile_stats(prepared)
    for key in ("action", "observation.state"):
        np.testing.assert_allclose(stats[key]["q01"], np.quantile(good, .01, axis=0))
        np.testing.assert_allclose(stats[key]["q99"], np.quantile(good, .99, axis=0))


def test_quantiles_refuse_scene_leakage_and_stale_statistics(tmp_path):
    prepared, _ = _prepared(tmp_path)
    prepared["validation"]["scenes"] = ["p-001"]
    with pytest.raises(ValueError, match="scene leakage"):
        train_quantile_stats(prepared)
    prepared["validation"]["scenes"] = ["p-101"]
    prepared["train_only_stats"]["action"]["mean"][0] += 1
    with pytest.raises(ValueError, match="statistics changed"):
        train_quantile_stats(prepared)


@pytest.mark.parametrize("wrapped", [False, True])
def test_strict_checkpoint_loader_accepts_only_matching_weights(tmp_path, wrapped):
    from safetensors.torch import save_model, save_file

    policy = torch.nn.Module()
    policy.model = torch.nn.Linear(7, 7)
    source = policy if wrapped else policy.model
    path = tmp_path / "model.safetensors"
    save_model(source, path)
    expected = policy.model.weight.detach().clone()
    with torch.no_grad():
        policy.model.weight.zero_()
    load_joint_pi05_weights(policy, path)
    torch.testing.assert_close(policy.model.weight, expected, rtol=0, atol=0)
    save_file({"weight": expected}, path)
    with pytest.raises(RuntimeError, match="Missing"):
        load_joint_pi05_weights(policy, path)


def test_strict_loader_does_not_fall_back_when_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_joint_pi05_weights(SimpleNamespace(model=None), tmp_path / "missing.safetensors")


def test_pi05_nonsquare_rgb_padding_stays_inside_model_range():
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    stub = torch.nn.Module()
    stub.register_parameter("device_anchor", torch.nn.Parameter(torch.zeros(())))
    stub.config = make_joint_pi05_config()
    batch = {key: torch.full((2, 3, 480, 640), .5) for key in stub.config.image_features}
    images, masks = PI05Policy._preprocess_images(stub, batch)
    for image, mask in zip(images, masks, strict=True):
        assert image.shape == (2, 3, 224, 224)
        assert torch.all(image[:, :, :28] == -1)  # Black top border, not -3.
        assert torch.all(image[:, :, 196:] == -1)
        assert torch.all(image[:, :, 28:196] == 0)  # Unchanged mid-gray content.
        assert mask.all()


def _tiny_precision_policy():
    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    group = policy.model.paligemma_with_expert = torch.nn.Module()
    group.paligemma = torch.nn.Linear(2, 2).bfloat16().requires_grad_(False)
    group.gemma_expert = torch.nn.Module()
    group.gemma_expert.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.q_proj = torch.nn.Linear(2, 2).bfloat16()
    group.gemma_expert.model.layers = torch.nn.ModuleList([layer])
    return policy


def test_precision_contract_preserves_small_updates_and_strict_fp32_reload(tmp_path):
    import json
    from safetensors.torch import save_model

    policy = _tiny_precision_policy()
    frozen = policy.model.paligemma_with_expert.paligemma.weight.detach().clone()
    configure_joint_pi05_precision(policy)
    expert = policy.model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight
    with torch.no_grad():
        expert.fill_(.02001953125)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=2.5e-6, foreach=False)
    expert.grad = torch.full_like(expert, .1)
    before = expert.detach().clone()
    optimizer.step()
    assert torch.all(expert != before) and expert.dtype == torch.float32
    assert optimizer.state[expert]['exp_avg'].dtype == torch.float32
    torch.testing.assert_close(policy.model.paligemma_with_expert.paligemma.weight, frozen, rtol=0, atol=0)
    path = tmp_path/'model.safetensors'
    save_model(policy, path)
    reloaded = _tiny_precision_policy().requires_grad_(False)
    with pytest.raises(ValueError, match='missing its precision contract'):
        load_joint_pi05_weights(reloaded, path)
    (tmp_path/PRECISION_FILE).write_text(json.dumps(FP32_EXPERT_PRECISION))
    load_joint_pi05_weights(reloaded, path)
    actual = reloaded.model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expert, rtol=0, atol=0)
    assert reloaded.aubo_joint_precision == FP32_EXPERT_PRECISION
    (tmp_path/PRECISION_FILE).write_text('{}')
    with pytest.raises(ValueError, match='unsupported AUBO PI05 precision'):
        load_joint_pi05_weights(reloaded, path)

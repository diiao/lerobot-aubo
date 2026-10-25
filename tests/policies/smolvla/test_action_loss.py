import pytest
import torch

from lerobot.policies.smolvla.losses import holding_action_loss, masked_action_loss


def test_gripper_weighting_preserves_arm_gradients_and_supervises_release():
    errors = torch.arange(2 * 5 * 32, dtype=torch.float64).reshape(2, 5, 32).requires_grad_()
    pad = torch.tensor([[False, False, False, False, True], [False, False, True, True, True]])
    weights = torch.tensor([[1., 3., 3., 1., 999.], [1., 3., 1., 1., 1.]])
    uniform = masked_action_loss(errors, 7, pad)
    weighted = masked_action_loss(errors, 7, pad, gripper_weights=weights)
    original_grad = torch.autograd.grad(uniform.sum(), errors, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(weighted.sum(), errors)[0]
    torch.testing.assert_close(actual_grad[..., :6], original_grad[..., :6], rtol=0, atol=0)
    assert torch.count_nonzero(actual_grad[pad]) == torch.count_nonzero(actual_grad[..., 7:]) == 0
    torch.testing.assert_close(actual_grad[0, :4, 6], weights[0, :4].double() / (8 * 7))
    assert actual_grad[0, 3, 6] > 0  # unweighted release still supervised
    torch.testing.assert_close(masked_action_loss(errors, 7, pad, gripper_weights=torch.ones(2, 5)),
                               uniform, rtol=0, atol=0)
    with pytest.raises(ValueError, match="choose gripper"):
        masked_action_loss(errors, 7, pad, gripper_weights=weights, timestep_weights=weights)


def test_holding_loss_separates_constant_offset_from_drift_and_masks_unused_rows():
    expert = torch.zeros(2, 5, 32, dtype=torch.float64)
    expert[:, :, :6] = torch.arange(5)[:, None] * .001  # preserve expert micro-motion
    pad = torch.tensor([[False, False, False, False, True], [False, False, False, True, True]])
    pairs = torch.tensor([[True, True, False, True], [False, False, False, True]])
    constant = expert + 2
    torch.testing.assert_close(holding_action_loss(constant, expert, pairs, pad), torch.zeros(2, dtype=torch.float64), atol=1e-25, rtol=0)
    assert masked_action_loss((constant - expert).square(), 7, pad).min() > 0
    drift = expert.clone()
    drift[0, :, :6] += torch.arange(5)[:, None] * .5
    drift[pad] = float("nan")
    drift[..., 6:] = float("nan")  # gripper and padded dimensions have no auxiliary gradient
    drift.requires_grad_()
    result = holding_action_loss(drift, expert, pairs, pad)
    torch.testing.assert_close(result, torch.tensor([.125, 0.], dtype=torch.float64))
    result.sum().backward()
    assert torch.isfinite(drift.grad).all()
    assert torch.count_nonzero(drift.grad[pad]) == torch.count_nonzero(drift.grad[..., 6:]) == 0
    assert torch.count_nonzero(drift.grad[0, 3]) == torch.count_nonzero(drift.grad[1]) == 0


@pytest.mark.parametrize("flow_time", [0., .1, .5, 1.])
def test_reconstructed_hold_gradient_scales_with_noise_time_and_stays_bounded(flow_time):
    generator = torch.Generator().manual_seed(7)
    expert = torch.randn(2, 50, 32, generator=generator, dtype=torch.float64)
    noise = torch.randn(2, 50, 32, generator=generator, dtype=torch.float64)
    target = noise - expert
    velocity = (target + torch.randn(2, 50, 32, generator=generator, dtype=torch.float64)).requires_grad_()
    pairs = torch.ones(2, 49, dtype=torch.bool)
    pairs[1, 1:] = False  # a sparse hold does not amplify gradients
    estimate = flow_time * noise + (1-flow_time) * expert - flow_time * velocity
    base = masked_action_loss((velocity - target).square(), 7).mean()
    auxiliary = .1 * holding_action_loss(estimate, expert, pairs).mean()
    base_grad = torch.autograd.grad(base, velocity, retain_graph=True)[0]
    aux_grad = torch.autograd.grad(auxiliary, velocity)[0]
    bound = 4 * .1 * 7 / 6 * flow_time**2
    assert aux_grad.norm() <= (bound + 1e-12) * base_grad.norm()
    assert torch.count_nonzero(aux_grad[..., 6:]) == 0
    perfect = flow_time * noise + (1-flow_time) * expert - flow_time * target
    torch.testing.assert_close(holding_action_loss(perfect, expert, pairs), torch.zeros(2, dtype=torch.float64), atol=1e-28, rtol=0)


def _production_forward(class_name):
    """Execute the actual forward method with fake tensors, without transformers/model downloads."""
    import ast
    import math
    from pathlib import Path
    path = Path(__file__).resolve().parents[3] / "src/lerobot/policies/smolvla/modeling_smolvla.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    namespace = dict(torch=torch, Tensor=torch.Tensor, math=math, F=torch.nn.functional,
                     masked_action_loss=masked_action_loss, holding_action_loss=holding_action_loss,
                     OBS_STATE="observation.state", ACTION="action",
                     OBS_LANGUAGE_TOKENS="observation.language.tokens",
                     OBS_LANGUAGE_ATTENTION_MASK="observation.language.attention_mask",
                     make_att_2d_masks=lambda pad, att: pad[:, :, None] & pad[:, None, :])
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["forward"]


def test_actual_training_forward_wires_clean_estimate_without_changing_default_or_rng():
    from types import SimpleNamespace
    flow_forward = _production_forward("VLAFlowMatching")
    policy_forward = _production_forward("SmolVLAPolicy")
    actions = torch.zeros(1, 4, 32)
    noise = torch.ones_like(actions)
    time = torch.tensor([.5])
    velocity = torch.arange(4).float()[None, :, None].expand_as(actions).clone().requires_grad_()
    embeds = lambda *a, **kw: (torch.zeros(1, 4, 32), torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.bool))
    flow = SimpleNamespace(config=SimpleNamespace(chunk_size=4), embed_prefix=embeds, embed_suffix=embeds,
        vlm_with_expert=SimpleNamespace(forward=lambda **kw: ((None, velocity), None)), action_out_proj=lambda x: x)
    fake = SimpleNamespace(config=SimpleNamespace(adapt_to_pi_aloha=False, action_feature=SimpleNamespace(shape=(7,)),
        action_representation="absolute", execution_loss_fraction=None, n_action_steps=1),
        prepare_images=lambda b: (None, None), prepare_state=lambda b: None, prepare_action=lambda b: actions,
        model=SimpleNamespace(forward=lambda *a, **kw: flow_forward(flow, *a, **kw)))
    batch = {"observation.language.tokens": None, "observation.language.attention_mask": None,
             "action_is_pad": torch.tensor([[False, False, False, True]])}
    rng = torch.get_rng_state().clone()
    base, _ = policy_forward(fake, batch, noise=noise, time=time)
    disabled, _ = policy_forward(fake, batch, noise=noise, time=time, holding_loss_coefficient=0.)
    torch.testing.assert_close(base, disabled, rtol=0, atol=0)
    actual, info = policy_forward(fake, batch, noise=noise, time=time, holding_loss_coefficient=.1,
                                  holding_pair_mask=torch.ones(1, 3, dtype=torch.bool), reduction="none")
    torch.testing.assert_close(actual, base[None] + .1 * torch.tensor([2 * .25 / 3]))
    assert info["holding_loss"] == pytest.approx(2 * .25 / 3)
    actual.sum().backward()
    assert torch.count_nonzero(velocity.grad[:, 3]) == torch.count_nonzero(velocity.grad[..., 7:]) == 0
    closed, info = policy_forward(fake, batch, noise=noise, time=time, gripper_loss_weights=torch.tensor([[1., 3., 3., 1.]]))
    expected = masked_action_loss((noise-actions-velocity).square(), 7, batch["action_is_pad"],
                                  gripper_weights=torch.tensor([[1., 3., 3., 1.]])).mean()
    torch.testing.assert_close(closed, expected)
    assert "gripper_weighted_loss" in info and torch.equal(rng, torch.get_rng_state())


def test_unused_coordinates_have_no_loss_or_gradient():
    losses = torch.full((2, 3, 32), 1000.0)
    losses[..., :8] = 2.0
    losses.requires_grad_()
    result = masked_action_loss(losses, 8)
    torch.testing.assert_close(result, torch.tensor([2.0, 2.0]))
    result.sum().backward()
    assert torch.count_nonzero(losses.grad[..., 8:]) == 0
    torch.testing.assert_close(losses.grad[..., :8], torch.full((2, 3, 8), 1 / 24))


def test_terminal_padding_does_not_dilute_mean_or_backpropagate():
    losses = torch.full((2, 4, 8), 999.0)
    losses[0, :1] = 3; losses[1, :3] = 7
    losses.requires_grad_()
    pad = torch.tensor([[False, True, True, True], [False, False, False, True]])
    result = masked_action_loss(losses, 8, pad)
    torch.testing.assert_close(result, torch.tensor([3.0, 7.0]))
    result.sum().backward()
    assert torch.count_nonzero(losses.grad[pad]) == 0


def test_real_dimensions_preserve_unpadded_loss():
    losses = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)
    torch.testing.assert_close(masked_action_loss(losses, 8), losses.mean((1, 2)))


def test_expert_timestep_weights_have_expected_loss_and_gradient():
    losses = torch.full((1, 4, 32), float("nan"))
    losses[0, :3, :7] = torch.tensor([2., 4., 8.])[:, None]
    losses.requires_grad_()
    pad = torch.tensor([[False, False, False, True]])
    weights = torch.tensor([[1., 3., 1., 999.]])
    result = masked_action_loss(losses, 7, pad, timestep_weights=weights)
    torch.testing.assert_close(result, torch.tensor([4.4]))
    result.sum().backward()
    torch.testing.assert_close(losses.grad[0, :3, :7], torch.tensor([1., 3., 1.])[:, None].expand(3, 7)/35)
    assert torch.count_nonzero(losses.grad[0, 3]) == 0
    assert torch.count_nonzero(losses.grad[..., 7:]) == 0
    torch.testing.assert_close(masked_action_loss(losses, 7, pad, timestep_weights=torch.ones(1, 4)),
                               masked_action_loss(losses, 7, pad), rtol=0, atol=0)


@pytest.mark.parametrize('pad', [torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, 3), torch.zeros(2, 4, dtype=torch.bool)])
def test_invalid_or_empty_temporal_masks_fail(pad):
    with pytest.raises(ValueError):
        masked_action_loss(torch.zeros(2, 3, 32), 8, pad)


def test_execution_weight_balances_groups_and_preserves_future_supervision():
    losses = torch.zeros(1, 50, 32, requires_grad=True)
    result = masked_action_loss(losses, 8, execution_horizon=1, execution_loss_fraction=.5)
    result.sum().backward()
    torch.testing.assert_close(losses.grad[0, 0, :8].sum(), torch.tensor(.5))
    torch.testing.assert_close(losses.grad[0, 1:, :8].sum(), torch.tensor(.5))
    assert torch.all(losses.grad[0, 1:, :8] > 0)
    assert torch.count_nonzero(losses.grad[..., 8:]) == 0


def test_terminal_padding_renormalizes_each_sample_and_excludes_nonfinite_padding():
    losses = torch.full((3, 4, 32), float('nan'))
    losses[0, :1, :8] = 2
    losses[1, :1, :8] = 2
    losses[1, 1:3, :8] = 6
    losses[2, 1:3, :8] = 10
    losses.requires_grad_()
    pad = torch.tensor([[False, True, True, True], [False, False, False, True], [True, False, False, True]])
    result = masked_action_loss(losses, 8, pad, execution_horizon=1, execution_loss_fraction=.5)
    torch.testing.assert_close(result, torch.tensor([2., 4., 10.]))
    result.sum().backward()
    assert torch.isfinite(losses.grad).all()
    assert torch.count_nonzero(losses.grad[pad]) == 0
    assert torch.count_nonzero(losses.grad[..., 8:]) == 0


def test_full_execution_horizon_and_disabled_weight_preserve_original_objective():
    losses = torch.arange(192, dtype=torch.float32).reshape(2, 3, 32)
    pad = torch.tensor([[False, False, True], [False, False, False]])
    expected = masked_action_loss(losses, 8, pad)
    torch.testing.assert_close(masked_action_loss(losses, 8, pad, execution_horizon=3, execution_loss_fraction=.5), expected)
    torch.testing.assert_close(masked_action_loss(losses, 8, pad, execution_horizon=1), expected, rtol=0, atol=0)


@pytest.mark.parametrize('fraction', [True, 0, 1, -.1, float('nan'), float('inf'), '0.5'])
def test_invalid_execution_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        masked_action_loss(torch.zeros(1, 50, 32), 8, execution_horizon=1, execution_loss_fraction=fraction)


@pytest.mark.parametrize('horizon', [None, True, 0, 51, 1.5])
def test_invalid_execution_horizon_is_rejected(horizon):
    with pytest.raises(ValueError):
        masked_action_loss(torch.zeros(1, 50, 32), 8, execution_horizon=horizon, execution_loss_fraction=.5)


def test_execution_loss_configuration_roundtrip(tmp_path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    assert SmolVLAConfig().execution_loss_fraction is None
    config = SmolVLAConfig(n_action_steps=1, execution_loss_fraction=.5)
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.execution_loss_fraction == .5 and restored.n_action_steps == 1


@pytest.mark.parametrize('fraction', [True, 0, 1, float('nan'), float('inf')])
def test_configuration_rejects_invalid_execution_fraction(fraction):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    with pytest.raises(ValueError):
        SmolVLAConfig(n_action_steps=1, execution_loss_fraction=fraction)


def test_policy_forward_uses_execution_weight_and_preserves_padding_alias():
    pytest.importorskip("transformers")
    from types import SimpleNamespace
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    losses = torch.full((1, 4, 32), 999.)
    losses[:, 0, :8] = 2
    losses[:, 1, :8] = 6
    losses[:, 2, :8] = 10
    losses.requires_grad_()
    fake = SimpleNamespace(
        config=SimpleNamespace(adapt_to_pi_aloha=False, action_feature=SimpleNamespace(shape=(8,)),
            n_action_steps=1, execution_loss_fraction=.5),
        prepare_images=lambda batch: (None, None),
        prepare_state=lambda batch: None,
        prepare_action=lambda batch: None,
        model=SimpleNamespace(forward=lambda *args: losses),
    )
    batch = {"observation.language.tokens": None, "observation.language.attention_mask": None,
        "actions_id_pad": torch.tensor([[False, False, False, True]])}
    per_sample, _ = SmolVLAPolicy.forward(fake, batch, reduction="none")
    torch.testing.assert_close(per_sample, torch.tensor([5.]))
    per_sample.sum().backward()
    assert torch.count_nonzero(losses.grad[:, 3]) == 0
    assert torch.count_nonzero(losses.grad[..., 8:]) == 0
    fake.config.execution_loss_fraction = None
    uniform, _ = SmolVLAPolicy.forward(fake, batch)
    torch.testing.assert_close(uniform, torch.tensor(6.))
    weighted, info = SmolVLAPolicy.forward(fake, batch, action_loss_weights=torch.tensor([[1., 3., 2., 1.]]))
    torch.testing.assert_close(weighted, torch.tensor(40/6))
    assert info["uniform_valid_loss"] == 6.
    batch["action_is_pad"] = torch.tensor([[False, True, True, True]])
    with pytest.raises(ValueError, match="Conflicting"):
        SmolVLAPolicy.forward(fake, batch)

import pytest
import torch

from lerobot.policies.smolvla.losses import masked_action_loss


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
    batch["action_is_pad"] = torch.tensor([[False, True, True, True]])
    with pytest.raises(ValueError, match="Conflicting"):
        SmolVLAPolicy.forward(fake, batch)

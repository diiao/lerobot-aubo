import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy, masked_action_l1_loss
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def test_act_vae_forward_in_eval_mode_computes_validation_loss():
    """Validation batches include actions, so VAE loss must work in eval mode."""
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        chunk_size=2,
        n_action_steps=1,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=4,
        n_vae_encoder_layers=1,
    )
    policy = ACTPolicy(config).eval()
    batch = {
        OBS_STATE: torch.zeros((2, 3)),
        OBS_ENV_STATE: torch.zeros((2, 2)),
        ACTION: torch.zeros((2, 2, 2)),
        "action_is_pad": torch.zeros((2, 2), dtype=torch.bool),
    }

    loss, loss_dict = policy.forward(batch)

    assert torch.isfinite(loss)
    assert "kld_loss" in loss_dict


def test_act_vae_select_action_accepts_none_action_placeholder():
    """Inference pre-processors can preserve ACTION with a None placeholder."""
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        chunk_size=2,
        n_action_steps=1,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=4,
        n_vae_encoder_layers=1,
    )
    policy = ACTPolicy(config).eval()
    batch = {
        OBS_STATE: torch.zeros((1, 3)),
        OBS_ENV_STATE: torch.zeros((1, 2)),
        ACTION: None,
    }

    action = policy.select_action(batch)

    assert action.shape == (1, 2)
    assert torch.isfinite(action).all()


def test_masked_action_l1_loss_without_weights_matches_original_act_objective():
    target = torch.tensor([[[1.0, 3.0], [9.0, 9.0]]])
    prediction = torch.zeros_like(target)
    is_pad = torch.tensor([[False, True]])

    weighted, unweighted = masked_action_l1_loss(target, prediction, is_pad)
    original = ((target - prediction).abs() * (~is_pad).unsqueeze(-1)).mean()

    torch.testing.assert_close(weighted, original)
    torch.testing.assert_close(unweighted, original)


def test_masked_action_l1_loss_normalizes_transition_weight_and_ignores_padding():
    target = torch.tensor([[[1.0, 4.0], [100.0, 100.0]]])
    prediction = torch.zeros_like(target)
    is_pad = torch.tensor([[False, True]])
    weights = torch.tensor([[[1.0, 3.0], [1.0, 99.0]]])

    weighted, unweighted = masked_action_l1_loss(
        target, prediction, is_pad, weights
    )

    # Valid weights [1, 3] are normalized to sum to two. The padded row has no effect.
    assert weighted.item() == 1.625
    assert unweighted.item() == 1.25


def test_masked_action_l1_loss_all_padding_is_finite_zero():
    target = torch.ones((1, 2, 2))
    prediction = torch.zeros_like(target)
    is_pad = torch.ones((1, 2), dtype=torch.bool)
    weights = torch.full_like(target, 20.0)

    weighted, unweighted = masked_action_l1_loss(
        target, prediction, is_pad, weights
    )

    assert weighted.item() == 0.0
    assert unweighted.item() == 0.0

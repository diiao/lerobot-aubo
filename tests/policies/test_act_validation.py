import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
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

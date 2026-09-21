import pytest
import torch

from lerobot.policies.smolvla.action_padding import restore_action_padding


@pytest.mark.parametrize("noise_time", [1.0, 0.6, 0.1, 0.0])
def test_padding_follows_training_interpolation_without_changing_real_actions(noise_time):
    noise = torch.arange(64, dtype=torch.float32).reshape(1, 2, 32) / 10
    latent = torch.full_like(noise, 37.)
    original = latent.clone()
    result = restore_action_padding(latent, noise, noise_time, 8)
    torch.testing.assert_close(result[..., :8], original[..., :8], rtol=0, atol=0)
    torch.testing.assert_close(result[..., 8:], noise_time * noise[..., 8:], rtol=0, atol=0)
    torch.testing.assert_close(latent, original, rtol=0, atol=0)


def test_multistep_integration_keeps_padding_on_manifold_despite_arbitrary_unused_velocity():
    noise = torch.ones(2, 4, 32)
    latent = noise.clone()
    dt = -.1
    for step in range(10):
        velocity = torch.full_like(latent, 1000.)
        velocity[..., :8] = 2.
        latent = restore_action_padding(latent + dt * velocity, noise, max(0., 1-(step+1)*.1), 8)
        torch.testing.assert_close(latent[..., :8], torch.full((2, 4, 8), 1-2*(step+1)*.1))
    assert torch.count_nonzero(latent[..., 8:]) == 0


def test_full_action_width_is_unchanged():
    latent = torch.randn(1, 2, 8)
    assert restore_action_padding(latent, torch.zeros_like(latent), 0., 8) is latent


@pytest.mark.parametrize("time", [float('nan'), float('inf'), -0.1, 1.1, True])
def test_invalid_time_is_rejected(time):
    with pytest.raises(ValueError):
        restore_action_padding(torch.zeros(1, 2, 32), torch.zeros(1, 2, 32), time, 8)


def test_actual_sampler_restores_padding_before_each_denoising_step():
    pytest.importorskip("transformers")
    from types import SimpleNamespace

    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    noise = torch.ones(1, 3, 6)
    seen = []

    def denoise_step(*, x_t, timestep, **kwargs):
        torch.testing.assert_close(x_t[..., 2:], timestep[:, None, None] * noise[..., 2:])
        seen.append(x_t.clone())
        velocity = torch.full_like(x_t, 1000.0)
        velocity[..., :2] = 2.0
        return velocity

    fake = SimpleNamespace(
        config=SimpleNamespace(num_steps=10, action_feature=SimpleNamespace(shape=(2,)), use_cache=True),
        embed_prefix=lambda *args, **kwargs: (
            torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool)
        ),
        vlm_with_expert=SimpleNamespace(forward=lambda **kwargs: (None, None)),
        _rtc_enabled=lambda: False,
        rtc_processor=None,
        denoise_step=denoise_step,
    )
    result = VLAFlowMatching.sample_actions(fake, None, None, None, None, torch.zeros(1, 2), noise=noise)
    assert len(seen) == 10
    torch.testing.assert_close(result[..., :2], -torch.ones(1, 3, 2))
    assert torch.count_nonzero(result[..., 2:]) == 0
    assert torch.equal(noise, torch.ones_like(noise))

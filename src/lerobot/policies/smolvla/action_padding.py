"""Keep unused action latents on the same known path used during training."""

import math

from torch import Tensor


def restore_action_padding(
    latent: Tensor, initial_noise: Tensor, noise_time: float, action_dim: int
) -> Tensor:
    """Restore only padded coordinates to t * noise; preserve real actions.

    Flow training uses x_t = t * noise + (1-t) * padded_action. Padded action
    targets are exactly zero. Once excluded from the loss, these coordinates
    must follow their known path rather than an unsupervised model velocity.
    The final padded coordinates are therefore exactly zero at t=0.
    """
    if latent.ndim != 3 or latent.shape != initial_noise.shape:
        raise ValueError("latent and initial_noise must have matching (batch, time, action) shapes")
    if isinstance(action_dim, bool) or not isinstance(action_dim, int) or not 0 < action_dim <= latent.shape[-1]:
        raise ValueError("action_dim must select the real action coordinates")
    if isinstance(noise_time, bool) or not math.isfinite(noise_time) or not 0 <= noise_time <= 1:
        raise ValueError("noise_time must be finite and between zero and one")
    if action_dim == latent.shape[-1]:
        return latent
    restored = latent.clone()
    restored[..., action_dim:] = noise_time * initial_noise[..., action_dim:]
    return restored

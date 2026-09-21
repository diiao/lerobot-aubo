"""Loss reduction over real action coordinates and non-padded timesteps."""

import math

import torch
from torch import Tensor


def masked_action_loss(
    losses: Tensor,
    action_dim: int,
    action_is_pad: Tensor | None = None,
    *,
    execution_horizon: int | None = None,
    execution_loss_fraction: float | None = None,
) -> Tensor:
    """Reduce real coordinates, optionally balancing execution and future groups.

    A fraction of None preserves the uniform valid-timestep objective. When
    enabled, each group is averaged over its own valid entries. If a sample
    has only one nonempty group, that group receives the whole weight.
    """
    if losses.ndim != 3:
        raise ValueError("losses must have shape (batch, time, padded_action_dim)")
    if isinstance(action_dim, bool) or not isinstance(action_dim, int) or not 0 < action_dim <= losses.shape[-1]:
        raise ValueError("action_dim must select real action coordinates")
    real = losses[..., :action_dim]
    if action_is_pad is None and execution_loss_fraction is None:
        return real.mean(dim=(1, 2))
    if action_is_pad is None:
        valid = torch.ones(losses.shape[:2], dtype=torch.bool, device=losses.device)
    elif action_is_pad.dtype != torch.bool or action_is_pad.shape != losses.shape[:2]:
        raise ValueError("action_is_pad must be a boolean (batch, time) mask")
    else:
        valid = ~action_is_pad.to(device=losses.device)
    count = valid.sum(dim=1)
    if (count == 0).any():
        raise ValueError("each sample must contain a non-padded action")
    masked = real.masked_fill(~valid[..., None], 0)
    if execution_loss_fraction is None:
        return masked.sum(dim=(1, 2)) / (count * action_dim)
    if (
        isinstance(execution_loss_fraction, bool)
        or not isinstance(execution_loss_fraction, (int, float))
        or not math.isfinite(execution_loss_fraction)
        or not 0 < execution_loss_fraction < 1
    ):
        raise ValueError("execution_loss_fraction must be finite and strictly between zero and one")
    if (
        isinstance(execution_horizon, bool)
        or not isinstance(execution_horizon, int)
        or not 1 <= execution_horizon <= losses.shape[1]
    ):
        raise ValueError("execution_horizon must select a nonempty prefix of the action chunk")
    execution_count = valid[:, :execution_horizon].sum(dim=1)
    future_count = valid[:, execution_horizon:].sum(dim=1)
    execution_mean = masked[:, :execution_horizon].sum(dim=(1, 2)) / (
        execution_count.clamp_min(1) * action_dim
    )
    future_mean = masked[:, execution_horizon:].sum(dim=(1, 2)) / (
        future_count.clamp_min(1) * action_dim
    )
    fraction = torch.full_like(execution_mean, execution_loss_fraction)
    fraction = torch.where(future_count == 0, 1.0, fraction)
    fraction = torch.where(execution_count == 0, 0.0, fraction)
    return fraction * execution_mean + (1 - fraction) * future_mean

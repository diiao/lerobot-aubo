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
    timestep_weights: Tensor | None = None,
    gripper_weights: Tensor | None = None,
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
    if gripper_weights is not None:
        if timestep_weights is not None or execution_loss_fraction is not None:
            raise ValueError("choose gripper weights or another weighting objective")
        # Keep every arm coordinate's coefficient exactly as in the base loss.
        # Redistribute only the last real coordinate's fixed 1/action_dim share.
        base = masked_action_loss(losses, action_dim, action_is_pad)
        gripper = real[..., -1:]
        weighted = masked_action_loss(gripper, 1, action_is_pad, timestep_weights=gripper_weights)
        uniform = masked_action_loss(gripper, 1, action_is_pad)
        return base + (weighted - uniform) / action_dim
    if action_is_pad is None and execution_loss_fraction is None and timestep_weights is None:
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
    if timestep_weights is not None:
        if execution_loss_fraction is not None:
            raise ValueError("choose timestep weights or execution-group weighting")
        if timestep_weights.shape != losses.shape[:2]:
            raise ValueError("timestep_weights must have shape (batch, time)")
        weights = timestep_weights.to(device=losses.device, dtype=losses.dtype)
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("timestep_weights must be finite and positive")
        weights = weights.masked_fill(~valid, 0)
        return (masked * weights[..., None]).sum(dim=(1, 2)) / (weights.sum(dim=1) * action_dim)
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


def holding_action_loss(
    estimated_actions: Tensor,
    expert_actions: Tensor,
    holding_pair_mask: Tensor,
    action_is_pad: Tensor | None = None,
    *,
    joint_dim: int = 6,
) -> Tensor:
    """Match adjacent clean-action increments in normalized action coordinates.

    Pair k joins rows k and k+1. Labels must come from the same expert episode.
    Divide by joint_dim * valid ROW count, not the number of selected pairs:
    sparse hold windows must not amplify their gradients. Empty windows give
    zero; gripper, padded coordinates and edges touching padding give no loss.
    The absolute-action flow objective remains necessary for constant offsets.
    """
    if estimated_actions.ndim != 3 or estimated_actions.shape != expert_actions.shape:
        raise ValueError("estimated and expert actions must have matching (batch, time, dim) shapes")
    if not 0 < joint_dim <= estimated_actions.shape[-1]:
        raise ValueError("joint_dim must select real joint coordinates")
    shape = estimated_actions.shape[:2]
    if holding_pair_mask.dtype != torch.bool or holding_pair_mask.shape != (shape[0], shape[1] - 1):
        raise ValueError("holding_pair_mask must be boolean (batch, time-1)")
    if action_is_pad is None:
        valid = torch.ones(shape, dtype=torch.bool, device=estimated_actions.device)
    elif action_is_pad.dtype != torch.bool or action_is_pad.shape != shape:
        raise ValueError("action_is_pad must be a boolean (batch, time) mask")
    else:
        valid = ~action_is_pad.to(estimated_actions.device)
    count = valid.sum(1)
    if (count == 0).any():
        raise ValueError("each sample must contain a non-padded action")
    pairs = holding_pair_mask.to(estimated_actions.device) & valid[:, :-1] & valid[:, 1:]
    error = estimated_actions[..., :joint_dim] - expert_actions[..., :joint_dim]
    # Mask before squaring so even nonfinite unused rows cannot enter gradients.
    increments = (error[:, 1:] - error[:, :-1]).masked_fill(~pairs[..., None], 0)
    return increments.square().sum((1, 2)) / (count * joint_dim)

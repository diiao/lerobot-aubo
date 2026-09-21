"""C0-only internal XYZ representation. External actions remain absolute 8D.

No state reconstruction from normalized values, mutable origin cache, or future
state is used. The origin travels with each observation through preprocessing.
"""

import numpy as np
import torch

RAW_TCP_KEY = "observation.smolvla_raw_tcp"
RELATIVE_XYZ = "relative_tcp_xyz"


def shift_xyz(actions: torch.Tensor, origin: torch.Tensor, *, inverse: bool = False):
    if actions.ndim != 3 or actions.shape[-1] != 8:
        raise ValueError("C0 XYZ conversion requires actions [B, T, 8]")
    if origin.shape != (actions.shape[0], 3) or not torch.isfinite(origin).all():
        raise ValueError("C0 XYZ conversion requires a finite raw TCP [B, 3]")
    result = actions.clone()
    result[..., 1:4] += (1 if inverse else -1) * origin.to(actions)[:, None, :]
    return result


def encode_relative_actions(actions, origin, stats):
    relative = shift_xyz(actions, origin)
    mean, std = (actions.new_tensor(stats[k]) for k in ("mean", "std"))
    return (relative - mean) / (std + 1e-8)


def decode_relative_actions(actions, origin, stats):
    mean, std = (actions.new_tensor(stats[k]) for k in ("mean", "std"))
    return shift_xyz(actions * std + mean, origin, inverse=True)


def compute_relative_xyz_stats(states, actions, episodes, frames, *, train_episodes, chunk_size):
    """XYZ: every valid (anchor t, offset k) pair, counted once, train only.

    This matches dataset offsets 0..chunk_size-1 and action_is_pad exclusions.
    Other action dimensions and state retain the original per-frame statistics.
    All moments are population moments, computed float64 then saved float32.
    """
    states, actions = np.asarray(states), np.asarray(actions)
    episodes, frames = np.asarray(episodes), np.asarray(frames)
    if states.shape != (len(episodes), 13) or actions.shape != (len(episodes), 8):
        raise ValueError("Expected C0 13D state and 8D action")
    if chunk_size < 1 or not train_episodes or len(set(train_episodes)) != len(train_episodes):
        raise ValueError("Invalid chunk size or training episodes")
    selected = np.isin(episodes, train_episodes)
    xyz = []
    for episode in train_episodes:
        rows = np.flatnonzero(episodes == episode)
        if not len(rows) or not np.array_equal(frames[rows], np.arange(len(rows))):
            raise ValueError("Training episodes must have contiguous ordered frame indices")
        s, a = states[rows], actions[rows]
        for offset in range(min(chunk_size, len(rows))):
            # Subtract in float32, exactly as the training tensors do.
            xyz.append(a[offset:, 1:4].astype(np.float32) - s[:len(rows)-offset, 6:9].astype(np.float32))
    xyz = np.concatenate(xyz).astype(np.float64)

    def moments(values):
        values = values.astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite training data")
        return {k: v.astype(np.float32) for k, v in {
            "mean": values.mean(0), "std": values.std(0),
            "min": values.min(0), "max": values.max(0),
        }.items()}

    stats = {"observation.state": moments(states[selected]), "action": moments(actions[selected])}
    for key, value in moments(xyz).items():
        stats["action"][key][1:4] = value
    return stats, {"train_episodes": list(train_episodes), "train_frames": int(selected.sum()),
                   "chunk_size": chunk_size, "valid_xyz_pairs": len(xyz),
                   "xyz_measure": "each valid anchor/offset pair once; no padded or validation rows",
                   "other_dimensions_measure": "original training-frame population moments"}

"""Offline suction-label checks; consistency is distinct from learnability."""

import numpy as np

from .aubo_joint_contract import JOINT_DIM, GRIPPER_INDEX


def gripper_episode_quality(states, actions):
    states, actions = np.asarray(states), np.asarray(actions)
    if (states.ndim != 2 or states.shape[1] != JOINT_DIM or actions.shape != states.shape
            or not len(states) or not np.isfinite(states).all() or not np.isfinite(actions).all()):
        raise ValueError("joint states/actions must be finite nonempty [frames, 7]")
    current, target = states[:, GRIPPER_INDEX], actions[:, GRIPPER_INDEX]
    if not np.isin(current, [0, 100]).all() or not np.isin(target, [0, 100]).all():
        raise ValueError("suction labels must be persistent 0/100 targets")
    if not np.array_equal(current[1:], target[:-1]):
        raise ValueError("suction history does not equal preceding command")
    on = np.flatnonzero((current == 0) & (target == 100))
    off = np.flatnonzero((current == 100) & (target == 0))
    complete = bool(current[0] == 0 and target[-1] == 0 and len(on) and len(off)
                    and on[0] < off[-1])
    issues = []
    if not complete:
        issues.append("missing_recorded_pick_and_release_cycle")
    if not np.any(current == 100):
        issues.append("missing_observed_suction_on_state")
    return {"on_frames": int(np.count_nonzero(target == 100)),
            "off_frames": int(np.count_nonzero(target == 0)),
            "activate_frames": on.tolist(), "release_frames": off.tolist(),
            "copy_current_state_accuracy": float(np.mean(current == target)),
            "complete_pick_release_cycle": complete,
            "quality_issues": issues}


def require_joint_normalization_stats(stats):
    """Reject wrong-width, missing, or constant suction statistics before use.

    This validates values, not provenance: training-only episode selection must
    be established by the caller and stored with the checkpoint.
    """
    for key in ("observation.state", "action"):
        feature = stats.get(key, {})
        arrays = {}
        for name in ("mean", "std", "min", "max"):
            value = np.asarray(feature.get(name), dtype=np.float64)
            if value.shape != (JOINT_DIM,) or not np.isfinite(value).all():
                raise ValueError(f"{key}.{name} must contain seven finite statistics")
            arrays[name] = value
        mean, std, lo, hi = (arrays[k] for k in ("mean", "std", "min", "max"))
        if (np.any(std < 0) or np.any(lo > hi) or np.any(mean < lo - 1e-5)
                or np.any(mean > hi + 1e-5)):
            raise ValueError(f"{key} has invalid normalization statistics")
        if lo[GRIPPER_INDEX] != 0 or hi[GRIPPER_INDEX] != 100 or std[GRIPPER_INDEX] <= 0 or not 0 < mean[GRIPPER_INDEX] < 100:
            raise ValueError(f"{key} suction statistics must cover both 0 and 100")

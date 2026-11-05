"""Demonstration scope shared by capture, audit and offline training."""

from .aubo_joint_contract import JOINT_DIM, GRIPPER_INDEX, finite_vector

DEMO_SCOPES = ("full", "local_alignment")
START_POSE_GROUPS = ("baseline", "position", "angle", "combined")


def demo_metadata(metadata, *, require_initial_state=False):
    """Old captures without this field are full demonstrations, never relabeled."""
    scope = metadata.get("demo_scope", "full")
    if scope not in DEMO_SCOPES:
        raise ValueError("unknown demo_scope")
    result = {"demo_scope": scope}
    group = metadata.get("start_pose_group")
    if scope == "full":
        if group is not None:
            raise ValueError("start_pose_group is only valid for local_alignment")
        return result
    # New captures omit the retired grouping; retain historical labels unchanged.
    if group is not None:
        if group not in START_POSE_GROUPS:
            raise ValueError("invalid historical start_pose_group")
        result["start_pose_group"] = group
    if require_initial_state or "initial_state" in metadata:
        state = list(finite_vector(metadata.get("initial_state", []), JOINT_DIM, "initial_state"))
        if state[GRIPPER_INDEX] != 0:
            raise ValueError("local_alignment must begin with an open gripper command (0)")
        result["initial_state"] = state
    return result

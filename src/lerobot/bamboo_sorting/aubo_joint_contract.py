"""AUBO i10 joint-space learning contract; no hardware or model imports.

Angles are degrees at the dataset boundary, matching AuboI10Robot J1..J6.
Only the driver converts them to SDK radians. Suction is commanded 0/100,
never physical vacuum feedback. Historical C0 13D/8D datasets stay unchanged.
"""

import math
from collections.abc import Mapping, Sequence
from numbers import Real

JOINT_SCHEMA_VERSION = "AuboI10JointLegacyTeleopV2"
JOINT_NAMES = tuple(f"J{i}" for i in range(1, 7))
LEARNED_JOINT_NAMES = JOINT_NAMES
JOINT_FIELDS = (*LEARNED_JOINT_NAMES, "gripper_pos")
JOINT_DIM = len(JOINT_FIELDS)
GRIPPER_INDEX = JOINT_FIELDS.index("gripper_pos")
JOINT_IMAGE_KEYS = ("global_rgb", "grasp_rgb")
JOINT_TASK = "Pick one strip and place it in the collection area."
JOINT_FPS = 25


def finite_vector(values: Sequence, size: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != size:
        raise ValueError(f"{name} must contain {size} values")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in values):
        raise ValueError(f"{name} must contain finite numbers, not bool")
    return tuple(float(v) for v in values)


def joint_command(action: Mapping) -> dict[str, float]:
    """Reject Cartesian, partial, ambiguous and non-binary commands."""
    if not isinstance(action, Mapping) or set(action) != set(JOINT_FIELDS):
        raise ValueError(f"joint action must contain exactly {JOINT_FIELDS}")
    values = finite_vector([action[k] for k in JOINT_FIELDS], len(JOINT_FIELDS), "joint action")
    if values[-1] not in (0.0, 100.0):
        raise ValueError("gripper_pos must be commanded 0 or 100")
    return dict(zip(JOINT_FIELDS, values, strict=True))


def expand_joint_command(action: Mapping) -> dict[str, float]:
    """All six joint targets are already explicit in the legacy-teleop contract."""
    return joint_command(action)


def joint_observation_features() -> dict:
    return {**dict.fromkeys(JOINT_FIELDS, float), **dict.fromkeys(JOINT_IMAGE_KEYS, (480, 640, 3))}


def joint_dataset_features() -> dict:
    vector = {"dtype": "float32", "shape": (JOINT_DIM,), "names": list(JOINT_FIELDS)}
    return {
        "observation.state": dict(vector),
        "action": dict(vector),
        **{f"observation.images.{key}": {"dtype": "video", "shape": (480, 640, 3),
           "names": ["height", "width", "channels"]} for key in JOINT_IMAGE_KEYS},
    }


def joint_contract_record() -> dict:
    return {
        "schema_version": JOINT_SCHEMA_VERSION,
        "robot_type": "aubo_i10",
        "state_names": list(JOINT_FIELDS),
        "action_names": list(JOINT_FIELDS),
        "joint_unit": "deg",
        "fixed_joint_targets_deg": {},
        "teleoperation_profile": "legacy_abs_j6yaw",
        "action_representation": "absolute_joint_positions",
        "action_source": "accepted_servoJoint_target",
        "gripper_values": {"off": 0, "on": 100},
        "physical_gripper_feedback_available": False,
        "fps": JOINT_FPS,
        "task": JOINT_TASK,
        "policy_execution_authorized": False,
    }


def require_joint_dataset(features: Mapping, contract: Mapping, fps: float) -> None:
    expected = joint_contract_record()
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("dataset must carry the exact AuboI10JointLegacyTeleopV2 contract")
    if fps != JOINT_FPS:
        raise ValueError("joint dataset must be 25 Hz")
    for key in ("observation.state", "action"):
        feature = features.get(key, {})
        if (tuple(feature.get("shape", ())) != (JOINT_DIM,) or feature.get("dtype") != "float32"
                or tuple(feature.get("names", ())) != JOINT_FIELDS):
            raise ValueError(f"{key} must have ordered AUBO 7D full-joint joint fields")
    images = {key for key in features if key.startswith("observation.images.")}
    if images != {f"observation.images.{key}" for key in JOINT_IMAGE_KEYS}:
        raise ValueError("joint dataset requires exactly the CameraSetV2 RGB streams")
    for key in images:
        if tuple(features[key].get("shape", ())) != (480, 640, 3) or features[key].get("dtype") != "video":
            raise ValueError(f"invalid RGB video feature: {key}")


def require_joint_target(target, current, lower, upper, max_step_deg: float) -> None:
    """Numerical joint guard, not a collision or tool-workspace certificate."""
    target = finite_vector(target, 6, "joint target")
    current = finite_vector(current, 6, "joint state")
    lower = finite_vector(lower, 6, "joint lower limits")
    upper = finite_vector(upper, 6, "joint upper limits")
    if isinstance(max_step_deg, bool) or not math.isfinite(max_step_deg) or max_step_deg <= 0:
        raise ValueError("max_step_deg must be finite and positive")
    for name, q, now, lo, hi in zip(JOINT_NAMES, target, current, lower, upper, strict=True):
        if lo >= hi or not lo <= now <= hi or not lo <= q <= hi:
            raise ValueError(f"{name} outside controller joint limits")
        # Do not wrap/clamp: a 360-degree branch jump is a rejected command.
        if abs(q - now) > max_step_deg:
            raise ValueError(f"{name} joint target exceeds {max_step_deg} deg tracking distance")

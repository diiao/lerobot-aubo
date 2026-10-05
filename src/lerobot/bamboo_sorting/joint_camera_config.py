"""Selectable camera configuration for new joint recordings and camera checks.

Loading configuration never opens devices. Existing policy execution retains
its frozen CameraSetV2 mapping until a new-view policy is prepared.
"""

import json
import math
from pathlib import Path

from .aubo_joint_contract import JOINT_IMAGE_KEYS
from .c0_smoke_capture import frozen_c0_smoke_camera_mapping


REPO = Path(__file__).resolve().parents[3]
CAMERA_SET_PATHS = {
    "original-global": REPO / "configs/aubo_i10/CameraSetV2.json",
    "wide-global": REPO / "configs/aubo_i10/camera_set_wide_global.json",
}
DEFAULT_CAPTURE_CAMERA_SET = "original-global"


def load_joint_camera_configuration(camera_set=DEFAULT_CAPTURE_CAMERA_SET):
    """Return the selected mapping and a snapshot to save with new data."""
    if camera_set not in CAMERA_SET_PATHS:
        raise ValueError(f"unknown camera set: {camera_set}")
    path = CAMERA_SET_PATHS[camera_set]
    record = json.loads(path.read_text())
    if camera_set == "original-global":
        mapping = frozen_c0_smoke_camera_mapping(path)
    else:
        if record.get("schema_version") != "AuboCameraSet":
            raise ValueError("expected AuboCameraSet")
        if tuple(record.get("camera_streams", [])) != JOINT_IMAGE_KEYS:
            raise ValueError("joint capture requires global_rgb and grasp_rgb")
        profiles = record.get("capture_profiles", {})
        if set(profiles) != set(JOINT_IMAGE_KEYS):
            raise ValueError("joint capture requires exactly two capture profiles")
        mapping = {}
        for name in JOINT_IMAGE_KEYS:
            profile = profiles[name]
            if any(type(profile.get(key)) is not int or profile[key] <= 0 for key in ("width", "height")):
                raise ValueError("camera width and height must be positive integers")
            if profile.get("model_color_layout") != "RGB_uint8":
                raise ValueError("joint capture requires RGB_uint8")
            fps = profile.get("fps")
            if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
                raise ValueError("camera fps must be finite and positive")
            device = profile.get("device")
            if not isinstance(device, str) or not device.startswith("/dev/"):
                raise ValueError("camera device must be an explicit /dev/ path")
            fourcc = profile.get("fourcc")
            if not isinstance(fourcc, str) or len(fourcc) != 4:
                raise ValueError("camera fourcc must have four characters")
            mapping[name] = {**profile, "physical_role": record["physical_roles"][name]}
        if mapping["global_rgb"]["device"] == mapping["grasp_rgb"]["device"]:
            raise ValueError("global and wrist cameras must use different devices")
    return {"camera_set": camera_set, "camera_set_path": str(path),
            "camera_set_record": record, "camera_mapping": mapping}

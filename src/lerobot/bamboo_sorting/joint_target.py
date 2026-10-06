"""Offline target-image preparation, shared by future demonstration and inference adapters.

No tracking, model loading or robot access. A frame-local mask never establishes
physical graspability or target identity by itself.
"""

import copy

import cv2
import numpy as np

from .aubo_joint_contract import joint_contract_record

TARGET_FRAME_SCHEMA = "aubo_joint_target_frame"
TARGET_IMAGE_KEY = "target_global_rgb"


def target_contract_record():
    contract = joint_contract_record()
    contract.update(
        schema_version="AuboI10JointTargetOutline",
        image_keys=[TARGET_IMAGE_KEY, "grasp_rgb"],
        target_conditioning={
            "source": "current global RGB and same-frame visible target mask",
            "encoding": "inner mask boundary, 2 pixels, RGB magenta [255,0,255]",
            "identity": "stable target_id within one pick; instance IDs are frame-local",
            "unavailable_target": "reject input; never reuse previous mask or choose a replacement",
        },
    )
    return contract


def selected_instance_mask(instance_map, segments, instance_id):
    """Select an explicitly chosen instance; pixel ID is not a class ID."""
    if (not isinstance(instance_map, np.ndarray) or instance_map.shape != (480, 640)
            or not np.issubdtype(instance_map.dtype, np.integer)):
        raise ValueError("instance map must be integer [480,640]")
    ids = [s["id"] for s in segments]
    if len(ids) != len(set(ids)) or isinstance(instance_id, bool) or instance_id <= 0:
        raise ValueError("invalid instance IDs")
    if instance_id not in ids:
        raise ValueError("selected instance is absent from segments")
    mask = instance_map == instance_id
    if not mask.any():
        raise ValueError("selected instance has no visible pixels")
    return mask


def compare_target_instances(instance_map, segments, reference):
    """Oracle shape comparison for one reviewed target, not runtime selection.

    Ignore class when finding the best overlap, but preserve the predicted class.
    Other objects are not exhaustively annotated, so this cannot report FP/F1.
    """
    if reference.dtype != np.bool_ or reference.shape != instance_map.shape or not reference.any():
        raise ValueError("nonempty reviewed target mask required")
    candidates = []
    for segment in segments:
        mask = instance_map == segment["mask_id"]
        if not mask.any():
            continue
        intersection = int(np.count_nonzero(mask & reference))
        union = int(np.count_nonzero(mask | reference))
        candidates.append({**segment, "iou": intersection / union,
                           "recall": intersection / int(reference.sum()),
                           "precision": intersection / int(mask.sum())})
    best = max(candidates, key=lambda row: row["iou"], default=None)
    return {"selection": "best overlap using human reference, not autonomous identity association",
            "best": best, "candidates": candidates}


def prepare_target_images(images, visible_mask, annotation, *, frame_id, target_id, allow_draft=False):
    """Build a distinct two-image input; legacy models must not consume it.

    frame_id must identify the RGB frame used to produce the mask. target_id is
    fixed for a pick, supplied by the caller, not inferred from the instance ID.
    allow_draft is only for human preview, never training or execution.
    """
    if not frame_id or not target_id:
        raise ValueError("frame and target identity required")
    if annotation.get("schema") != TARGET_FRAME_SCHEMA:
        raise ValueError("target annotation schema mismatch")
    if annotation.get("frame_id") != frame_id:
        raise ValueError("mask belongs to another frame")
    if annotation.get("target_id") != target_id:
        raise ValueError("selected target identity changed")
    if annotation.get("status") != "visible":
        raise ValueError("target unavailable: do not reuse a previous mask")
    if annotation.get("reviewed") is not True and not allow_draft:
        raise ValueError("target annotation requires review")
    for name in ("global_rgb", "grasp_rgb"):
        image = images.get(name)
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (480, 640, 3):
            raise ValueError(f"invalid RGB image: {name}")
    if (not isinstance(visible_mask, np.ndarray) or visible_mask.dtype != np.bool_
            or visible_mask.shape != (480, 640) or not visible_mask.any()):
        raise ValueError("visible target mask must be nonempty bool [480,640]")
    mask = visible_mask.astype(np.uint8)
    interior = cv2.erode(mask, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    boundary = visible_mask & (interior == 0)
    marked = images["global_rgb"].copy()
    marked[boundary] = (255, 0, 255)
    return {
        "contract": target_contract_record(),
        "images": {TARGET_IMAGE_KEY: marked, "grasp_rgb": images["grasp_rgb"].copy()},
        "target": copy.deepcopy(annotation),
        "preview_only": annotation.get("reviewed") is not True,
    }

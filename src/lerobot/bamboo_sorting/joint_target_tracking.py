"""Offline optical-flow baseline. Propagated masks are drafts, never verified labels."""

import cv2
import numpy as np


class TargetMaskFlow:
    """Propagate only from an explicit seed; after loss, require another seed.

    Photometric and forward/backward checks are diagnostic heuristics, not an
    occlusion detector or a calibrated confidence score. Newly revealed regions
    cannot be recovered reliably by warping an old visible mask.
    """

    def __init__(self):
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.gray = self.mask = None
        self.seed_area = 0

    def seed(self, rgb, mask):
        if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2] or not mask.any():
            raise ValueError("seed requires a nonempty boolean mask matching RGB")
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.mask = mask.copy()
        self.seed_area = int(mask.sum())

    def step(self, rgb):
        if self.mask is None:
            return None, {"status": "lost", "reason": "explicit_seed_required"}
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if gray.shape != self.gray.shape:
            raise ValueError("frame resolution changed")
        backward = self.flow.calc(gray, self.gray, None)
        forward = self.flow.calc(self.gray, gray, None)
        yy, xx = np.indices(gray.shape, dtype=np.float32)
        mx, my = xx + backward[..., 0], yy + backward[..., 1]
        candidate = cv2.remap(self.mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST).astype(bool)
        reverse = cv2.remap(forward, mx, my, cv2.INTER_LINEAR)
        cycle_error = np.linalg.norm(backward + reverse, axis=2)
        old_gray = cv2.remap(self.gray, mx, my, cv2.INTER_LINEAR)
        residual = np.abs(gray.astype(np.float32) - old_gray.astype(np.float32))
        valid = (cycle_error <= 2.0) & (residual <= 35)
        retained = candidate & valid
        fraction = float(retained.sum() / max(1, candidate.sum()))
        area_ratio = float(retained.sum() / self.seed_area)
        report = {"retained_fraction": fraction, "area_vs_seed": area_ratio}
        if fraction < 0.65 or not 0.35 <= area_ratio <= 2.0:
            self.gray = self.mask = None
            return None, {**report, "status": "lost", "reason": "flow_consistency_or_area"}
        self.gray, self.mask = gray, retained
        return retained.copy(), {**report, "status": "draft", "reviewed": False}


def mask_overlap(candidate, reference):
    """Count a lost track as empty for checkpoint coverage, with explicit status."""
    if candidate is None:
        return {"iou": 0.0, "recall": 0.0, "track_available": False}
    intersection = np.count_nonzero(candidate & reference)
    union = np.count_nonzero(candidate | reference)
    return {"iou": float(intersection / union) if union else 0.0,
            "recall": float(intersection / max(1, np.count_nonzero(reference))), "track_available": True}

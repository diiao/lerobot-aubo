"""Offline bamboo strip geometry from one RGB handeye frame.

This module does not import robot control, cameras, or action senders.
Pixel detections are not robot-base coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

ALGORITHM_VERSION = "BambooGeometryColorBaselineV1"
MIN_AREA_PX = 800
MAX_AREA_PX = 18000
MIN_ASPECT = 4.0
MIN_LENGTH_PX = 80.0
MIN_CONFIDENCE = 0.45


@dataclass(frozen=True)
class BambooGeometryDetection:
    valid: bool
    p1_uv: tuple[float, float] | None
    p2_uv: tuple[float, float] | None
    center_uv: tuple[float, float] | None
    theta_rad: float | None
    cos2theta: float | None
    sin2theta: float | None
    confidence: float
    reject_reason: str | None
    mask: np.ndarray | None
    color_space: str
    algorithm_version: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "p1_uv": list(self.p1_uv) if self.p1_uv is not None else None,
            "p2_uv": list(self.p2_uv) if self.p2_uv is not None else None,
            "center_uv": list(self.center_uv) if self.center_uv is not None else None,
            "theta_rad": self.theta_rad,
            "cos2theta": self.cos2theta,
            "sin2theta": self.sin2theta,
            "confidence": self.confidence,
            "reject_reason": self.reject_reason,
            "color_space": self.color_space,
            "algorithm_version": self.algorithm_version,
        }


def _invalid(
    reason: str,
    *,
    color_space: str,
    mask: np.ndarray | None = None,
    confidence: float = 0.0,
) -> BambooGeometryDetection:
    return BambooGeometryDetection(
        valid=False,
        p1_uv=None,
        p2_uv=None,
        center_uv=None,
        theta_rad=None,
        cos2theta=None,
        sin2theta=None,
        confidence=float(confidence),
        reject_reason=reason,
        mask=mask,
        color_space=color_space,
        algorithm_version=ALGORITHM_VERSION,
    )


def _axis_encoding(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
    delta = p2 - p1
    theta = float(np.arctan2(delta[1], delta[0]))
    two = 2.0 * theta
    return theta, float(np.cos(two)), float(np.sin(two))


def geometry_from_endpoints(
    p1_uv: tuple[float, float],
    p2_uv: tuple[float, float],
    image_hw: tuple[int, int],
    *,
    color_space: str = "rgb",
    mask: np.ndarray | None = None,
    confidence: float = 1.0,
) -> BambooGeometryDetection:
    height, width = image_hw
    p1 = np.asarray(p1_uv, dtype=np.float64)
    p2 = np.asarray(p2_uv, dtype=np.float64)
    for name, point in ("p1", p1), ("p2", p2):
        if not np.isfinite(point).all():
            return _invalid("non_finite_endpoint", color_space=color_space, mask=mask)
        if point[0] < 0 or point[1] < 0 or point[0] > width - 1 or point[1] > height - 1:
            return _invalid(f"{name}_out_of_bounds", color_space=color_space, mask=mask)
    length = float(np.linalg.norm(p2 - p1))
    if length < MIN_LENGTH_PX:
        return _invalid("endpoints_too_close", color_space=color_space, mask=mask)
    center = (p1 + p2) / 2.0
    theta, cos2, sin2 = _axis_encoding(p1, p2)
    if confidence < MIN_CONFIDENCE:
        return _invalid(
            "low_confidence",
            color_space=color_space,
            mask=mask,
            confidence=confidence,
        )
    return BambooGeometryDetection(
        valid=True,
        p1_uv=(float(p1[0]), float(p1[1])),
        p2_uv=(float(p2[0]), float(p2[1])),
        center_uv=(float(center[0]), float(center[1])),
        theta_rad=theta,
        cos2theta=cos2,
        sin2theta=sin2,
        confidence=float(confidence),
        reject_reason=None,
        mask=mask,
        color_space=color_space,
        algorithm_version=ALGORITHM_VERSION,
    )


def _wood_mask_rgb(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    wood = (
        (red > 110)
        & (green > 80)
        & (blue < red - 10)
        & (green <= red + 25)
        & ((red - blue) > 25)
        & ((red + green) > 220)
    )
    mesh = (green > red + 8) & (green > blue + 8)
    mask = (wood & ~mesh).astype(np.uint8) * 255
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )


def _component_candidates(mask: np.ndarray) -> list[dict[str, Any]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates: list[dict[str, Any]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            continue
        ys, xs = np.where(labels == index)
        points = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(points)
        (center_x, center_y), (width, height), angle_deg = rect
        long_side = float(max(width, height))
        short_side = float(min(width, height))
        aspect = long_side / max(short_side, 1e-3)
        if aspect < MIN_ASPECT or long_side < MIN_LENGTH_PX:
            continue
        if width >= height:
            axis_rad = np.deg2rad(angle_deg)
            half = width / 2.0
        else:
            axis_rad = np.deg2rad(angle_deg + 90.0)
            half = height / 2.0
        axis = np.array([np.cos(axis_rad), np.sin(axis_rad)], dtype=np.float64)
        center = np.array([center_x, center_y], dtype=np.float64)
        p1 = center - axis * half
        p2 = center + axis * half
        score = area * aspect
        component_mask = (labels == index).astype(np.uint8) * 255
        candidates.append(
            {
                "score": float(score),
                "area": area,
                "aspect": float(aspect),
                "p1": (float(p1[0]), float(p1[1])),
                "p2": (float(p2[0]), float(p2[1])),
                "mask": component_mask,
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def detect_bamboo_geometry(image: np.ndarray, *, color_space: str = "rgb") -> BambooGeometryDetection:
    if color_space != "rgb":
        return _invalid(f"unsupported_color_space:{color_space}", color_space=color_space)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8:
        raise ValueError("image must be uint8 RGB")
    mask = _wood_mask_rgb(rgb)
    if int(mask.sum() / 255) < MIN_AREA_PX:
        return _invalid("empty_mask", color_space=color_space, mask=mask)
    candidates = _component_candidates(mask)
    if not candidates:
        return _invalid("no_elongated_component", color_space=color_space, mask=mask)
    if len(candidates) >= 2 and candidates[1]["score"] > 0.55 * candidates[0]["score"]:
        return _invalid("multiple_strips", color_space=color_space, mask=mask, confidence=0.2)
    best = candidates[0]
    height, width = rgb.shape[:2]
    p1 = (
        float(np.clip(best["p1"][0], 0, width - 1)),
        float(np.clip(best["p1"][1], 0, height - 1)),
    )
    p2 = (
        float(np.clip(best["p2"][0], 0, width - 1)),
        float(np.clip(best["p2"][1], 0, height - 1)),
    )
    aspect_term = min((best["aspect"] - MIN_ASPECT) / 8.0, 1.0)
    area_term = min(best["area"] / 8000.0, 1.0)
    confidence = 0.35 + 0.4 * max(aspect_term, 0.0) + 0.25 * area_term
    return geometry_from_endpoints(
        p1,
        p2,
        rgb.shape[:2],
        color_space=color_space,
        mask=best["mask"],
        confidence=float(confidence),
    )


def render_geometry_overlay(rgb: np.ndarray, detection: BambooGeometryDetection) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    if detection.mask is not None:
        green = np.zeros_like(overlay)
        green[:, :, 1] = 180
        mask = detection.mask > 0
        overlay[mask] = cv2.addWeighted(overlay, 0.65, green, 0.35, 0)[mask]
    if not detection.valid:
        cv2.line(overlay, (20, 20), (overlay.shape[1] - 20, overlay.shape[0] - 20), (0, 0, 255), 3)
        cv2.line(overlay, (overlay.shape[1] - 20, 20), (20, overlay.shape[0] - 20), (0, 0, 255), 3)
        text = f"INVALID: {detection.reject_reason}"
        cv2.putText(overlay, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return overlay
    p1 = (int(round(detection.p1_uv[0])), int(round(detection.p1_uv[1])))
    p2 = (int(round(detection.p2_uv[0])), int(round(detection.p2_uv[1])))
    center = (int(round(detection.center_uv[0])), int(round(detection.center_uv[1])))
    cv2.line(overlay, p1, p2, (0, 255, 255), 2)
    cv2.circle(overlay, p1, 6, (255, 0, 0), -1)
    cv2.circle(overlay, p2, 6, (255, 0, 0), -1)
    cv2.putText(overlay, "P1", (p1[0] + 8, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(overlay, "P2", (p2[0] + 8, p2[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.drawMarker(overlay, center, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    cv2.circle(overlay, center, 10, (0, 0, 255), 2)
    cv2.putText(overlay, "C", (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    axis = np.array([np.cos(detection.theta_rad), np.sin(detection.theta_rad)], dtype=np.float64)
    tip = (
        int(round(center[0] + 28 * axis[0])),
        int(round(center[1] + 28 * axis[1])),
    )
    cv2.arrowedLine(overlay, center, tip, (0, 255, 255), 2, tipLength=0.3)
    status = (
        f"{detection.algorithm_version} q={detection.confidence:.2f} "
        f"C=({detection.center_uv[0]:.1f},{detection.center_uv[1]:.1f})"
    )
    cv2.putText(overlay, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2)
    cv2.putText(overlay, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    return overlay

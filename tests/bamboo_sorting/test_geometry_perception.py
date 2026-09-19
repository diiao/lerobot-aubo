import numpy as np

from lerobot.bamboo_sorting.geometry_perception import (
    detect_bamboo_geometry,
    geometry_from_endpoints,
    render_geometry_overlay,
)


def _green_canvas(height: int = 120, width: int = 160) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (20, 150, 30)
    return image


def _paint_stick(image: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    painted = image.copy()
    painted[y0:y1, x0:x1] = (210, 170, 90)
    return painted


def test_endpoint_swap_preserves_center_and_double_angle() -> None:
    image = _green_canvas()
    first = geometry_from_endpoints((20.0, 40.0), (140.0, 80.0), image.shape[:2])
    swapped = geometry_from_endpoints((140.0, 80.0), (20.0, 40.0), image.shape[:2])
    assert first.valid and swapped.valid
    assert first.center_uv == swapped.center_uv
    assert abs(first.cos2theta - swapped.cos2theta) < 1e-9
    assert abs(first.sin2theta - swapped.sin2theta) < 1e-9


def test_out_of_bounds_endpoint_is_invalid() -> None:
    image = _green_canvas()
    detection = geometry_from_endpoints((-1.0, 10.0), (40.0, 10.0), image.shape[:2])
    assert detection.valid is False
    assert detection.reject_reason == "p1_out_of_bounds"


def test_empty_mask_is_invalid() -> None:
    detection = detect_bamboo_geometry(_green_canvas(), color_space="rgb")
    assert detection.valid is False
    assert detection.reject_reason == "empty_mask"


def test_low_confidence_is_invalid() -> None:
    image = _green_canvas()
    detection = geometry_from_endpoints(
        (20.0, 40.0),
        (140.0, 40.0),
        image.shape[:2],
        confidence=0.1,
    )
    assert detection.valid is False
    assert detection.reject_reason == "low_confidence"


def test_bgr_color_space_is_rejected() -> None:
    detection = detect_bamboo_geometry(_green_canvas(), color_space="bgr")
    assert detection.valid is False
    assert detection.reject_reason == "unsupported_color_space:bgr"


def test_synthetic_stick_is_detected_near_center() -> None:
    image = _paint_stick(_green_canvas(240, 320), 110, 126, 40, 280)
    detection = detect_bamboo_geometry(image, color_space="rgb")
    assert detection.valid
    assert abs(detection.center_uv[0] - 159.5) < 8.0
    assert abs(detection.center_uv[1] - 117.5) < 6.0
    overlay = render_geometry_overlay(image, detection)
    assert overlay.shape == (240, 320, 3)

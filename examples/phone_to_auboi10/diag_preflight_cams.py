"""录制前双相机体检。

使用与 record.py 相同的设备、编码和采样策略，同时打开两台相机，检查：
1. 能否连接并持续出帧；
2. 实际 cadence 是否满足 25 Hz 控制循环；
3. 最新帧是否过旧或接近全黑/纯色；
4. 保存现场预览图，供人工确认 handeye/fixed 没有接反。

本脚本不连接机械臂。预览图只写到 /tmp/aubo_camera_preflight。
"""

import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

CONTROL_FPS = 25
MIN_ACCEPTABLE_FPS = 23.0
MAX_FRAME_AGE_MS = 80.0
PREVIEW_DIR = Path("/tmp/aubo_camera_preflight")

CAMERA_CONFIGS = {
    # 该相机驱动只接受 30 FPS，但硬件时间戳实测约 25 FPS。
    "handeye": {
        "device": "/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH0319_20210712_v102-video-index0",
        "capture_fps": 30,
    },
    "fixed": {
        "device": "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
        "capture_fps": 25,
    },
}


def measure_camera(name: str, camera: OpenCVCamera) -> list[str]:
    failures: list[str] = []
    timestamps: list[float] = []
    end = time.perf_counter() + 2.0

    while time.perf_counter() < end:
        with camera.frame_lock:
            capture_time = camera.latest_timestamp
        if capture_time is not None and (
            not timestamps or capture_time != timestamps[-1]
        ):
            timestamps.append(capture_time)
        time.sleep(0.001)

    intervals_ms = [
        (timestamps[index + 1] - timestamps[index]) * 1000
        for index in range(len(timestamps) - 1)
    ]
    actual_fps = 1000.0 / statistics.mean(intervals_ms) if intervals_ms else 0.0

    ages_ms: list[float] = []
    frame = None
    loop_start = time.perf_counter()
    for index in range(CONTROL_FPS):
        target = loop_start + index / CONTROL_FPS
        remaining = target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        frame = camera.read_latest(max_age_ms=500)
        with camera.frame_lock:
            capture_time = camera.latest_timestamp
        ages_ms.append((time.perf_counter() - capture_time) * 1000)

    if frame is None:
        failures.append(f"{name}: 没有获得预览帧")
        return failures

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = PREVIEW_DIR / f"{name}.jpg"
    cv2.imwrite(str(preview_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    frame_mean = float(np.mean(frame))
    frame_std = float(np.std(frame))
    max_age = max(ages_ms)
    print(
        f"  cadence={actual_fps:.1f} fps, "
        f"interval={statistics.mean(intervals_ms):.1f}±"
        f"{statistics.pstdev(intervals_ms):.1f} ms"
        if intervals_ms
        else "  cadence=0 fps"
    )
    print(
        f"  frame age mean={statistics.mean(ages_ms):.1f} ms, "
        f"max={max_age:.1f} ms"
    )
    print(
        f"  image mean={frame_mean:.1f}, std={frame_std:.1f}, "
        f"preview={preview_path}"
    )

    if actual_fps < MIN_ACCEPTABLE_FPS:
        failures.append(
            f"{name}: 实际 {actual_fps:.1f} FPS，低于最低 {MIN_ACCEPTABLE_FPS:.1f} FPS"
        )
    if max_age > MAX_FRAME_AGE_MS:
        failures.append(
            f"{name}: 最大帧年龄 {max_age:.1f} ms，超过 {MAX_FRAME_AGE_MS:.1f} ms"
        )
    if frame_mean < 5.0 or frame_std < 5.0:
        failures.append(
            f"{name}: 图像接近全黑或纯色 (mean={frame_mean:.1f}, std={frame_std:.1f})"
        )

    return failures


def main() -> int:
    cameras: dict[str, OpenCVCamera] = {}
    failures: list[str] = []

    try:
        # 保持两台相机同时运行，测量条件才与 record.py 一致。
        for name, values in CAMERA_CONFIGS.items():
            print(f"\n连接 [{name}] {values['device']}")
            config = OpenCVCameraConfig(
                index_or_path=values["device"],
                width=640,
                height=480,
                fps=values["capture_fps"],
                fourcc="MJPG",
                warmup_s=3,
            )
            camera = OpenCVCamera(config)
            camera.connect(warmup=True)
            cameras[name] = camera

        for name, camera in cameras.items():
            print(f"\n检查 [{name}]")
            failures.extend(measure_camera(name, camera))
    except Exception as exc:
        failures.append(f"相机连接或读取异常: {type(exc).__name__}: {exc}")
    finally:
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()

    if failures:
        print("\n体检失败，不要开始录制：")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\n体检通过。开始录制前请人工打开两张预览图确认相机名称和视野。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

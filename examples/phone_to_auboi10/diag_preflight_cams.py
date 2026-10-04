"""录制前双相机体检。

使用与 record_joint.py 相同的可选相机配置，同时打开两台相机，检查：
1. 能否连接并持续出帧；
2. 实际 cadence 是否满足 25 Hz 控制循环；
3. 最新帧是否过旧或接近全黑/纯色；
4. 保存现场预览图，供人工确认 global_rgb/grasp_rgb 没有接反。

本脚本不连接机械臂。--plan 只显示配置；实际测试默认写入新的临时目录。
"""

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

from lerobot.bamboo_sorting.joint_camera_config import (
    CAMERA_SET_PATHS, DEFAULT_CAPTURE_CAMERA_SET, load_joint_camera_configuration,
)

CONTROL_FPS = 25
MIN_ACCEPTABLE_FPS = 23.0
MAX_FRAME_AGE_MS = 80.0
def measure_camera(name, camera, *, seconds, output) -> list[str]:
    import cv2
    import numpy as np

    failures: list[str] = []
    timestamps: list[float] = []
    end = time.perf_counter() + seconds

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
        frame, capture_time = camera.read_latest_with_timestamp(max_age_ms=500)
        ages_ms.append((time.perf_counter() - capture_time) * 1000)

    if frame is None:
        failures.append(f"{name}: 没有获得预览帧")
        return failures

    preview_path = output / f"{name}.jpg"
    if not cv2.imwrite(str(preview_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to save {preview_path}")

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-set", choices=tuple(CAMERA_SET_PATHS), default=DEFAULT_CAPTURE_CAMERA_SET)
    parser.add_argument("--plan", action="store_true", help="Print configuration without opening devices")
    parser.add_argument("--seconds", type=float, default=float(os.environ.get("CAMERA_TEST_SECONDS", "2")))
    parser.add_argument("--output", type=Path, help="New output directory; default: unique /tmp directory")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be finite and positive")
    configuration = load_joint_camera_configuration(args.camera_set)
    print(json.dumps(configuration, ensure_ascii=False, indent=2))
    if args.plan:
        return 0

    import cv2
    from lerobot.cameras.opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    output = args.output
    if output is None:
        output = Path(tempfile.mkdtemp(prefix=f"aubo_camera_preflight_{args.camera_set}_"))
    else:
        output.mkdir(parents=True, exist_ok=False)
    (output / "camera_configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    cameras = {}
    failures: list[str] = []

    try:
        # 保持两台相机同时运行，测量条件才与 record_joint.py 一致。
        for name, values in configuration["camera_mapping"].items():
            print(f"\n连接 [{name}] {values['device']}")
            config = OpenCVCameraConfig(
                index_or_path=values["device"],
                width=values["width"],
                height=values["height"],
                fps=values["fps"],
                fourcc=values["fourcc"],
                warmup_s=3,
            )
            camera = OpenCVCamera(config)
            cameras[name] = camera
            camera.connect(warmup=True)
            actual_fourcc_code = int(camera.videocapture.get(cv2.CAP_PROP_FOURCC))
            actual_fourcc = "".join(
                chr((actual_fourcc_code >> (8 * index)) & 0xFF)
                for index in range(4)
            )
            print(
                f"  configured={values['fourcc']}@{values['fps']} FPS, "
                f"actual={actual_fourcc}@{camera.videocapture.get(cv2.CAP_PROP_FPS):.1f} FPS"
            )

        for name, camera in cameras.items():
            print(f"\n检查 [{name}]")
            failures.extend(measure_camera(name, camera, seconds=args.seconds, output=output))
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

    print("\n体检通过。开始录制前请人工打开两张预览图确认相机角色和视野。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

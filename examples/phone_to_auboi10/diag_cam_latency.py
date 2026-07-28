"""测录像 pipeline (OpenCVCamera) 的延迟特性。

只测软件侧可量化部分:
  1) CAP_PROP_BUFFERSIZE 能否设成 1
  2) 后台线程的真实取帧节奏 (cadence)
  3) 新数据 25fps 控制循环里,取到的帧有多"旧" (freshness)

注意: age 只反映"后台线程拿到帧 -> 录制循环取帧"的时间差,
**不包含**传感器曝光 + USB 传输的硬件固有延迟(那部分需要画面基准才能测)。
"""
import time
import statistics
import cv2

from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

DEV = "/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH0319_20210712_v102-video-index0"
CAPTURE_FPS = 30
CONTROL_FPS = 25

cfg = OpenCVCameraConfig(
    index_or_path=DEV,
    width=640,
    height=480,
    fps=CAPTURE_FPS,
    fourcc="MJPG",
    warmup_s=8,
)
cam = OpenCVCamera(cfg)
cam.connect(warmup=True)

before = cam.videocapture.get(cv2.CAP_PROP_BUFFERSIZE)
print(f"[buf] CAP_PROP_BUFFERSIZE (默认, 录像实际用的) = {before}")
print(f"      (v4l2 流开启后不能改; 如需 =1 要在 connect 前设, 此处保持默认以测真实录像延迟)")

# --- Test1: 后台线程取帧节奏 ---
print("\n[Test1] 后台线程取帧节奏 (采样 2s)...")
t0 = time.perf_counter()
caps = []
end = t0 + 2.0
while time.perf_counter() < end:
    with cam.frame_lock:
        ct = cam.latest_timestamp
    if ct is not None and (not caps or ct != caps[-1]):
        caps.append(ct)
    time.sleep(0.001)
intervals = [(caps[i + 1] - caps[i]) * 1000 for i in range(len(caps) - 1)]
print(
    f"  2s 内拿到 {len(caps)} 帧 -> {len(caps)/2.0:.1f} fps "
    f"(新数据控制目标 {CONTROL_FPS})"
)
if intervals:
    print(f"  帧间隔 ms: mean={statistics.mean(intervals):.1f} "
          f"min={min(intervals):.1f} max={max(intervals):.1f} "
          f"stdev={statistics.pstdev(intervals):.1f}")

# --- Test2: 录制式 25fps 循环的帧新鲜度 ---
print("\n[Test2] 录制式 25fps 循环, 帧新鲜度 (3s = 75 帧)...")
ages = []
N = CONTROL_FPS * 3
loop_start = time.perf_counter()
for i in range(N):
    target = loop_start + i / CONTROL_FPS
    now = time.perf_counter()
    if now < target:
        time.sleep(target - now)
    frame = cam.read_latest(max_age_ms=2000)
    t = time.perf_counter()
    with cam.frame_lock:
        ct = cam.latest_timestamp
    ages.append((t - ct) * 1000)
print(f"  帧年龄(age) ms: mean={statistics.mean(ages):.1f} "
      f"min={min(ages):.1f} max={max(ages):.1f} "
      f"stdev={statistics.pstdev(ages):.1f}")
print("  (age = 后台取帧 -> 录制循环取帧 的时差; 不含传感器曝光/USB 固有延迟)")

cam.disconnect()
print("\ndone")

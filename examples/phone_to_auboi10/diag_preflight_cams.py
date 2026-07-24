"""录像前体检: 用 record.py 完全相同的配置测两个相机能否正常出帧。
不连机器人, 只测相机。"""
import time
import statistics
import cv2

from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

CAMS = {
    "handeye": "/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH0319_20210712_v102-video-index0",
    "fixed":   "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
}
FPS = 30

for name, dev in CAMS.items():
    print(f"\n{'='*50}\n[{name}]  {dev}")
    cfg = OpenCVCameraConfig(index_or_path=dev, width=640, height=480, fps=FPS, fourcc="MJPG", warmup_s=8)
    cam = OpenCVCamera(cfg)
    try:
        cam.connect(warmup=True)
    except Exception as e:
        print(f"  连接失败: {e}")
        continue
    buf = cam.videocapture.get(cv2.CAP_PROP_BUFFERSIZE)
    print(f"  连接OK, BUFFERSIZE={buf}")

    # 取 1 帧看尺寸
    f = cam.read_latest(max_age_ms=2000)
    print(f"  帧尺寸: {f.shape} dtype={f.dtype}")

    # cadence 2s
    t0 = time.perf_counter(); caps = []
    end = t0 + 2.0
    while time.perf_counter() < end:
        with cam.frame_lock:
            ct = cam.latest_timestamp
        if ct is not None and (not caps or ct != caps[-1]):
            caps.append(ct)
        time.sleep(0.001)
    iv = [(caps[i+1]-caps[i])*1000 for i in range(len(caps)-1)]
    print(f"  cadence: {len(caps)/2.0:.1f} fps, 间隔 mean={statistics.mean(iv):.1f}ms stdev={statistics.pstdev(iv):.1f}ms" if iv else "  cadence: 无帧")

    # 录制式新鲜度 1.5s
    ages = []; N = 45; ls = time.perf_counter()
    for i in range(N):
        tgt = ls + i/FPS
        now = time.perf_counter()
        if now < tgt: time.sleep(tgt-now)
        cam.read_latest(max_age_ms=2000)
        t = time.perf_counter()
        with cam.frame_lock: ct = cam.latest_timestamp
        ages.append((t-ct)*1000)
    print(f"  新鲜度: age mean={statistics.mean(ages):.1f}ms max={max(ages):.1f}ms")
    cam.disconnect()
    print(f"  [{name}] 通过 ✓")
print("\n体检完成")

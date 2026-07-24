#!/usr/bin/env python
"""端到端测试 inference_server.predict() 的启发式 gripper 覆盖。
用合成观测 (真模型 + 假图) 走完整 predict 路径, 验证:
  1. 不崩 (修复 prepare_observation_for_inference 原地重绑导致的 IndexError)
  2. fire 区 state -> a[7]==100
  3. release 区 state (holding) -> a[7]==0
  4. 接近区 state -> a[7]==0
图像用全零 dummy, 启发式只看 state, 图像内容不影响 gripper 判断。
"""
import numpy as np
import cv2

from inference_server import InferenceServer, HEURISTIC_GRIPPER

print(f"HEURISTIC_GRIPPER={HEURISTIC_GRIPPER}")
print("加载 InferenceServer (含模型)...")
srv = InferenceServer()

# dummy 图像 JPEG (480x640x3 全灰), 走 encode_obs 同样的 JPEG bytes 路径
dummy_img = np.full((480, 640, 3), 128, dtype=np.uint8)
ok, jpg = cv2.imencode(".jpg", dummy_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
dummy_jpg = jpg.tobytes()


def make_obs(state_vec):
    return {
        "observation.state": np.array(state_vec, dtype=np.float32),
        "observation.images.handeye": dummy_jpg,
        "observation.images.fixed": dummy_jpg,
        "task": "抓取竹条",
        "robot_type": "aubo_i10",
    }


def grip_of(state_vec):
    a = srv.predict(make_obs(state_vec))
    return float(a[7])


srv.reset()
# 1. 接近中: 高位, 不在 fire 区 -> 0
g = grip_of([0, 0, 0, 0, 0, 0, 0.59, -0.41, 0.30, 0, 0, 0, 0])
print(f"[接近] z=0.30 -> gripper={g:.1f}  (期望 0)")
assert g == 0.0, "接近时应为 0"

# 2. fire 区: z=0.10 @ 抓取点 xy -> 100, latch
g = grip_of([0, 0, 0, 0, 0, 0, 0.59, -0.41, 0.10, 0, 0, 0, 0])
print(f"[fire ] z=0.10 @抓取点 -> gripper={g:.1f}  (期望 100)")
assert g == 100.0, "fire 区应触发 100"

# 3. 持物提起: holding, 离开 fire 区, 高位 -> 100 (latch)
g = grip_of([0, 0, 0, 0, 0, 0, 0.59, -0.41, 0.40, 0, 0, 0, 0])
print(f"[提起 ] holding, z=0.40 -> gripper={g:.1f}  (期望 100, latch 保持)")
assert g == 100.0, "holding 提起应保持 100"

# 4. release 区: x=0.15,y=-0.63,z=0.13 -> 0, unlatch
g = grip_of([0, 0, 0, 0, 0, 0, 0.15, -0.63, 0.13, 0, 0, 0, 0])
print(f"[放   ] release 区 -> gripper={g:.1f}  (期望 0)")
assert g == 0.0, "release 区应释放 0"

# 5. 放完后: holding=False, 回到接近位 -> 0
g = grip_of([0, 0, 0, 0, 0, 0, 0.59, -0.41, 0.30, 0, 0, 0, 0])
print(f"[放后 ] z=0.30 -> gripper={g:.1f}  (期望 0)")
assert g == 0.0, "放完后应为 0"

# 6. 边界: eval 实测最低悬停 z=0.117 @ fire xy -> 应触发 (z<0.125)
srv.reset()
g = grip_of([0, 0, 0, 0, 0, 0, 0.585, -0.378, 0.117, 0, 0, 0, 0])
print(f"[eval悬停] z=0.117 @eval xy -> gripper={g:.1f}  (期望 100, 这是修'从不触发'的关键)")
assert g == 100.0, "eval 悬停高度(0.117)必须触发"

print("\n全部通过 ✓")

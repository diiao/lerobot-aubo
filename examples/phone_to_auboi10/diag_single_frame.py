#!/usr/bin/env python
"""单帧诊断: 把训练触发帧 + 评估 z 最低帧各跑一遍 eval 推理管线, 看 gripper 输出。

- 训练帧 (z=0.12, 训练里 gripper=100) 经 eval 管线后 -> gripper≈100 ?
  - 是 -> 管线正常, 评估帧的图像在模型层面就是 OOD -> 加增广/补数据
  - 否 -> 管线有 bug 破坏了触发信号
"""
import cv2
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

MODEL = "./models/bamboo_shift"
TRAIN_DS = "./datasets/bamboo_full_shift"
EVAL_DS = "./datasets/phone_auboi10_split_eval"

print("加载策略...")
policy = ACTPolicy.from_pretrained(MODEL)
policy.eval()
dev = torch.device(policy.config.device)
meta = LeRobotDatasetMetadata(TRAIN_DS)
pre, post = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=MODEL,
    dataset_stats=meta.stats,
    preprocessor_overrides={"device_processor": {"device": str(dev)}},
)
print(f"  chunk_size={policy.config.chunk_size} n_action_steps={policy.config.n_action_steps}")

print("\n加载数据集...")
dtrain = LeRobotDataset(TRAIN_DS)
try:
    deval = LeRobotDataset(EVAL_DS)
    has_eval = True
except Exception as e:
    print(f"  (评估数据集不在本机, 跳过评估帧测试: {type(e).__name__})")
    has_eval = False


def to_hwc_uint8(img_chw_tensor):
    a = img_chw_tensor.numpy()
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    return np.transpose(a, (1, 2, 0))  # CHW -> HWC


def run_one(label, ds, idx):
    s = ds[idx]
    obs_np = {
        "observation.state": s["observation.state"].numpy().astype(np.float32),
        "observation.images.handeye": to_hwc_uint8(s["observation.images.handeye"]),
        "observation.images.fixed": to_hwc_uint8(s["observation.images.fixed"]),
    }
    # 模拟 eval 的 JPEG 往返 (quality 85, 同 evaluate_split.py)
    for k in ["observation.images.handeye", "observation.images.fixed"]:
        ok, jpg = cv2.imencode(".jpg", obs_np[k], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        obs_np[k] = cv2.imdecode(jpg, cv2.IMREAD_COLOR)

    policy.reset(); pre.reset(); post.reset()
    with torch.inference_mode():
        batch = prepare_observation_for_inference(obs_np, str(dev), "抓取竹条", "aubo_i10")
        batch = pre(batch)
        a = policy.select_action(batch)  # (1, n_action_steps, action_dim) or (1, action_dim)
        a = post(a)
    a = a.squeeze(0).cpu().numpy()
    a = np.atleast_1d(a)
    # 取第一个 action step
    if a.ndim == 2:
        a = a[0]
    st = s["observation.state"].numpy()
    print(f"\n[{label}] idx={idx}")
    print(f"  输入 state: x={st[6]:.4f} y={st[7]:.4f} z={st[8]:.4f} grip_state={st[12]:.1f}")
    print(f"  模型输出: grip_act={a[7]:.2f}  z_act={a[3]:.4f}  x_act={a[1]:.4f}  y_act={a[2]:.4f}")
    print(f"  完整 action: {np.array2string(a, precision=3)}")
    return a[7]


# 训练触发帧 (ep0 首个 gripper_act>=100 的帧, 全局 idx 495, z=0.120)
g_train = run_one("训练触发帧 (训练里 gripper=100)", dtrain, 495)

if has_eval:
    # 评估 ep3 z 最低帧 (全局 idx 6220, z=0.106, eval 里 gripper=-5)
    g_eval = run_one("评估 z 最低帧 (eval 里 gripper=-5)", deval, 6220)
else:
    g_eval = None

print("\n" + "=" * 60)
print(f"训练帧 gripper 输出: {g_train:.2f}  (训练标签=100)")
if g_eval is not None:
    print(f"评估帧 gripper 输出: {g_eval:.2f}  (eval 记录=-5)")
if g_train > 50:
    print("=> 训练帧能触发 (gripper>50): 管线正常")
    if g_eval is not None and g_eval < 50:
        print("   但评估帧不触发: 评估帧图像对模型是 OOD")
        print("   方向: 视觉增广 + 补数据, 不是 z 扰动")
else:
    print("=> 训练帧也不触发: 管线/preprocess 有 bug, 模型根本没收到正确输入")
    print("   方向: 查 preprocess/图像 transform, 不用重训")
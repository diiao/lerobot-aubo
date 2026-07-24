#!/usr/bin/env python
"""记忆 vs 轨迹发散 诊断。

对每个 eval z-min 帧 (机器人到最低点本该触发吸盘但没触发的帧):
  1. 在训练集里按 EE xyz 找最近帧 (整体最近 + 最近"抓取帧" gripper>50)
  2. 算 EE 距离 / 手眼图像素MSE / 固定图像素MSE
  3. 跑模型看两帧的 gripper 输出

判读:
  - EE 距离小 + 手眼MSE小 + 输出差很大  -> 纯记忆 (模型咬训练帧特有的小像素特征)
  - EE 距离大 或 手眼MSE大              -> 轨迹发散 (eval 到的位姿训练里没见过)
  - 固定图MSE 大于手眼                  -> 物体/背景在变 (但用户说没变, 用来复核)
"""
import glob

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

MODEL_DIR = "./models/bamboo_shift_aug"  # 刚训完的, 训练帧能触发 eval 不能
TRAIN_DS = "./datasets/bamboo_full_shift"
EVAL_DS = "./datasets/phone_auboi10_split_eval"

HF_HOME = "/home/rentao/.cache/huggingface/lerobot/datasets"


def to_hwc_uint8(img_chw_tensor):
    a = img_chw_tensor.numpy()
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    return np.transpose(a, (1, 2, 0))


def encode_eval_pipe(s):
    """LeRobotDataset sample -> eval 管线输入 (JPEG 往返, 同 evaluate_split.py)"""
    obs_np = {
        "observation.state": s["observation.state"].numpy().astype(np.float32),
        "observation.images.handeye": to_hwc_uint8(s["observation.images.handeye"]),
        "observation.images.fixed": to_hwc_uint8(s["observation.images.fixed"]),
    }
    for k in ["observation.images.handeye", "observation.images.fixed"]:
        ok, jpg = cv2.imencode(".jpg", obs_np[k], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        obs_np[k] = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    return obs_np


def load_policy_and_procs(ckpt_path):
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.eval()
    dev = torch.device(policy.config.device)
    meta = LeRobotDatasetMetadata(TRAIN_DS)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt_path,
        dataset_stats=meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(dev)}},
    )
    return policy, pre, post, dev


def run_grip(policy, pre, post, dev, obs_np):
    policy.reset(); pre.reset(); post.reset()
    obs = {**obs_np}  # 浅拷贝: prepare_observation_for_inference 会原地重绑 key 成 CUDA tensor
    with torch.inference_mode():
        batch = prepare_observation_for_inference(obs, str(dev), "抓取竹条", "aubo_i10")
        batch = pre(batch)
        a = policy.select_action(batch)
        a = post(a)
    a = a.squeeze(0).cpu().numpy()
    if a.ndim == 2:
        a = a[0]
    return float(a[7])


def read_states(repo_name):
    """读 parquet 所有帧的 state(13) + episode_index, 返回 (states, eps)。"""
    files = sorted(glob.glob(f"{HF_HOME}/{repo_name}/data/chunk-*/*.parquet"))
    states, eps = [], []
    for f in files:
        t = pq.read_table(f)
        states.extend(t["observation.state"].to_pylist())
        eps.extend(t["episode_index"].to_pylist())
    return np.array(states), np.array(eps)


def pixel_stats(a, b):
    """返回 (MSE, 均值差, 像素差std)。均值差大=亮度漂移; std大=结构差异。"""
    def _to_np(x):
        if hasattr(x, "numpy"):
            x = x.numpy()
        return np.asarray(x, dtype=np.float64)
    a = _to_np(a); b = _to_np(b)
    diff = a - b
    return float(np.mean(diff ** 2)), float(np.mean(diff)), float(np.std(diff))


def main():
    print("加载数据集 + 读 parquet state...")
    dtrain = LeRobotDataset(TRAIN_DS)
    deval = LeRobotDataset(EVAL_DS)

    train_states, train_eps = read_states("bamboo_full_shift")
    eval_states, eval_eps = read_states("phone_auboi10_split_eval")
    train_ee = train_states[:, 6:9]          # ee.x/y/z
    train_rot = train_states[:, 9:12]        # ee.wx/wy/wz (rotvec)
    train_grip = train_states[:, 12]         # gripper_pos
    eval_ee = eval_states[:, 6:9]
    eval_rot = eval_states[:, 9:12]

    # eval 每条 ep 的 z 最低帧 (全局 idx)
    eval_zmin = []
    for ep in np.unique(eval_eps):
        mask = eval_eps == ep
        zs = eval_states[mask, 8]
        local = np.argmin(zs)
        gidx = int(np.where(mask)[0][local])
        eval_zmin.append((int(ep), gidx))
    print(f"eval z-min 帧: {eval_zmin}")

    # 训练抓取帧 (gripper>50) 的索引
    grasp_idx = np.where(train_grip > 50)[0]
    print(f"训练帧总数 {len(train_states)}, 其中抓取帧(gripper>50) {len(grasp_idx)}")

    print(f"\n加载模型 {MODEL_DIR} ...")
    policy, pre, post, dev = load_policy_and_procs(MODEL_DIR)

    print("\n" + "=" * 120)
    print("对每个 eval z-min 帧: 找训练集中 EE xyz 最近的【抓取帧 gripper>50】, 对比位姿/像素/模型输出")
    print(f"{'ep':>3}{'xyz距离(m)':>11}{'rot距离':>9}{'手眼MSE':>9}{'手眼均值差':>11}{'手眼std':>9}"
          f"{'固定MSE':>9}{'固定均值差':>11}{'模型eval':>10}{'模型train':>11}{'跨ep对照MSE':>13}")
    print("-" * 120)
    for ep, gidx in eval_zmin:
        s_eval = deval[gidx]
        obs_eval = encode_eval_pipe(s_eval)
        ee_eval = eval_ee[gidx]
        grip_eval_model = run_grip(policy, pre, post, dev, obs_eval)

        # 最近的训练抓取帧 (gripper>50) by xyz
        d_grasp = np.linalg.norm(train_ee[grasp_idx] - ee_eval, axis=1)
        ng = int(grasp_idx[int(np.argmin(d_grasp))])
        s_g = dtrain[ng]
        obs_g = encode_eval_pipe(s_g)
        grip_g_model = run_grip(policy, pre, post, dev, obs_g)
        xyz_d = float(d_grasp.min())
        rot_d = float(np.linalg.norm(train_rot[ng] - eval_rot[gidx]))
        he_mse, he_mean, he_std = pixel_stats(
            obs_eval["observation.images.handeye"], obs_g["observation.images.handeye"])
        fx_mse, fx_mean, fx_std = pixel_stats(
            obs_eval["observation.images.fixed"], obs_g["observation.images.fixed"])

        # 对照: 该训练抓取帧 vs 另一 episode 的最近抓取帧 (训练内部跨 ep 正常变异)
        ng_ep = int(train_eps[ng])
        other_mask = (train_eps[grasp_idx] != ng_ep)
        other_idx = grasp_idx[other_mask]
        d_other = np.linalg.norm(train_ee[other_idx] - train_ee[ng], axis=1)
        ng2 = int(other_idx[int(np.argmin(d_other))])
        s_g2 = dtrain[ng2]
        obs_g2 = encode_eval_pipe(s_g2)
        ctrl_he_mse, _, _ = pixel_stats(
            obs_g["observation.images.handeye"], obs_g2["observation.images.handeye"])

        print(f"{ep:>3}{xyz_d:>11.4f}{rot_d:>9.4f}{he_mse:>9.1f}{he_mean:>11.1f}{he_std:>9.1f}"
              f"{fx_mse:>9.1f}{fx_mean:>11.1f}{grip_eval_model:>10.1f}{grip_g_model:>11.1f}{ctrl_he_mse:>13.1f}")

    print("\n判读:")
    print("  xyz距离 <0.02m + rot距离 <0.1 = eval 到了和训练抓取帧几乎相同的位姿")
    print("  手眼均值差 |大| (>10) 且 std小 = 亮度/曝光漂移 (ColorJitter 应对症, 但要更强)")
    print("  手眼均值差 ~0 且 std大 (>20)   = 结构差异 (腕部朝向/物体位置不同)")
    print("  跨ep对照MSE = 训练内部不同 ep 间的正常像素变异; eval MSE 若远大于它 = eval 有特殊漂移")


if __name__ == "__main__":
    main()
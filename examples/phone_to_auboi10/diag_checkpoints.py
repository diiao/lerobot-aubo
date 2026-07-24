#!/usr/bin/env python
"""多 checkpoint 诊断: 对每个 checkpoint, 把 eval 每条 ep 的 z 最低帧 + 训练触发帧
走 eval 推理管线, 看 gripper 输出。

过拟合的话早期 checkpoint 可能还能触发 (gripper>50)。如果全不触发, 说明不是
过拟合随步数恶化的问题, 要换思路 (视觉增广 / 补数据)。
"""
import cv2
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

MODEL_DIR = "./models/bamboo_shift"
TRAIN_DS = "./datasets/bamboo_full_shift"
EVAL_DS = "./datasets/phone_auboi10_split_eval"
CKPT_STEPS = [5000, 10000, 15000, 20000, 25000]  # + final


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


def run_one(policy, pre, post, dev, obs_np):
    policy.reset(); pre.reset(); post.reset()
    with torch.inference_mode():
        batch = prepare_observation_for_inference(obs_np, str(dev), "抓取竹条", "aubo_i10")
        batch = pre(batch)
        a = policy.select_action(batch)
        a = post(a)
    a = a.squeeze(0).cpu().numpy()
    if a.ndim == 2:
        a = a[0]
    return a


def main():
    print("加载数据集...")
    dtrain = LeRobotDataset(TRAIN_DS)
    deval = LeRobotDataset(EVAL_DS)

    # 找 eval 每条 ep 的 z 最低帧
    import pyarrow.parquet as pq, glob
    efiles = sorted(glob.glob(
        "/home/rentao/.cache/huggingface/lerobot/datasets/phone_auboi10_split_eval/data/chunk-*/*.parquet"))
    es, ee = [], []
    for f in efiles:
        t = pq.read_table(f)
        es.extend(t["observation.state"].to_pylist())
        ee.extend(t["episode_index"].to_pylist())
    es = np.array(es); ee = np.array(ee)
    eval_frames = []  # (label, global_idx)
    for ep in np.unique(ee):
        mask = ee == ep
        zs = es[mask, 8]
        local = np.argmin(zs)
        gidx = int(np.where(mask)[0][local])
        eval_frames.append((f"eval_ep{ep}_zmin", gidx))
    # 训练触发帧 (idx 495, z=0.120, gripper=100)
    eval_frames.append(("train_fire_495", 495))

    print(f"测试帧: {[(l, i) for l, i in eval_frames]}")

    # 各 checkpoint
    ckpts = [(s, f"{MODEL_DIR}/checkpoint_{s}") for s in CKPT_STEPS]
    ckpts.append(("final", MODEL_DIR))

    results = {}  # ckpt_label -> {frame_label: gripper}
    for ck_label, ck_path in ckpts:
        print(f"\n加载 {ck_label}: {ck_path} ...")
        try:
            policy, pre, post, dev = load_policy_and_procs(ck_path)
        except Exception as e:
            print(f"  跳过 ({e})")
            continue
        results[ck_label] = {}
        for fr_label, gidx in eval_frames:
            ds = dtrain if fr_label.startswith("train") else deval
            s = ds[gidx]
            obs_np = encode_eval_pipe(s)
            a = run_one(policy, pre, post, dev, obs_np)
            results[ck_label][fr_label] = (float(a[7]), float(s["observation.state"].numpy()[8]))
            print(f"  {fr_label}: grip_act={a[7]:7.2f}  (z={s['observation.state'].numpy()[8]:.4f})")

    # 汇总表
    print("\n" + "=" * 70)
    print("gripper 输出汇总 (>50=触发吸盘):")
    print(f"  {'checkpoint':<12}", end="")
    for fr_label, _ in eval_frames:
        print(f"  {fr_label:<16}", end="")
    print()
    for ck_label in [s for s, _ in ckpts] + ["final"]:
        if ck_label not in results:
            continue
        print(f"  {str(ck_label):<12}", end="")
        for fr_label, _ in eval_frames:
            v = results[ck_label].get(fr_label)
            if v is None:
                print(f"  {'--':<16}", end="")
            else:
                mark = "OK" if v[0] > 50 else "x "
                print(f"  {mark}{v[0]:>7.1f}    ", end="")
        print()


if __name__ == "__main__":
    main()
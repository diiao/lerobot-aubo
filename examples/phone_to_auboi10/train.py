# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train ACT policy on Aubo I10 phone-teleoperated dataset.

Usage:
    cd examples/phone_to_auboi10
    python train.py

The script loads the dataset recorded by record.py, trains an ACT model,
and saves it to LOCAL_MODEL_PATH for use with evaluate.py.
"""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import init_logging

LOCAL_DATASET_PATH = "./datasets/bamboo_full_shift"
LOCAL_MODEL_PATH = "./models/bamboo_shift_aug"

# --- Training hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
TRAINING_STEPS = 10_000   # 策略2: 从 ckpt_5000 续训, 步数少; 从头训改 30_000
LOG_FREQ = 200
SAVE_FREQ = 2_000         # 续训时存密一点, 便于挑泛化最好的 checkpoint

# ACT hyperparameters (tuned for 30fps Aubo data, small dataset)
CHUNK_SIZE = 10       # 10 steps @ 30fps ≈ 0.33s action horizon
N_ACTION_STEPS = 10   # Execute all predicted actions before re-querying

# --- 图像增广 (治过拟合: 训练帧能触发 gripper=100, eval 帧不触发, 图像泛化差) ---
# LeRobot 内置 ImageTransforms, 在 dataset.__getitem__ 里作用于解码后的原始像素
# (归一化之前), 评估侧 inference_server 不走此路径, 不会泄漏。
IMAGE_AUG = True

# --- 续训: 从已有 checkpoint 加载权重 + 增广再训。None = 从头训 ---
RESUME_FROM = "./models/bamboo_shift/checkpoint_5000"  # 诊断里最不 overfit 的 (ep3=47.4)

# --- 验证监控: 训练中定期跑 eval 帧看 gripper 是否 >50 (触发吸盘) ---
# eval 数据集 (phone_auboi10_split_eval) 已在 GPU 机。监控用的不是训练信号 (无梯度),
# 只是挑 checkpoint 的依据。真正的判据是训练后跑一次真实评估。
VAL_ON_EVAL = True
VAL_EVAL_DATASET = "./datasets/phone_auboi10_split_eval"
VAL_FREQ = 1000
TRAIN_SANITY_IDX = 495  # 训练触发帧 (z=0.120, gripper=100), 监控它应保持 ~100


def build_image_transforms() -> ImageTransforms:
    """构造图像增广。默认配置 (LeRobot 校准过的温和值):
    brightness(0.8-1.2) / contrast(0.8-1.2) / saturation(0.5-1.5) /
    hue(±0.05) / sharpness(0.5-1.5) / affine(±5°, ±5%平移)
    每帧 RandomSubsetApply 抽最多 3 个施加。
    affine 最对症相机 mm 级位置漂移; ColorJitter 对症曝光/白平衡微漂。
    """
    cfg = ImageTransformsConfig(enable=True)
    return ImageTransforms(cfg)


def find_eval_zmin_frames(eval_root: str) -> list[int]:
    """读 eval 数据集 parquet, 找每条 ep 的 state.z 最低帧 (全局 idx)。"""
    import glob
    import numpy as np
    import pyarrow.parquet as pq
    files = sorted(glob.glob(f"{eval_root}/data/chunk-*/*.parquet"))
    if not files:
        return []
    states, eps = [], []
    for f in files:
        t = pq.read_table(f)
        states.extend(t["observation.state"].to_pylist())
        eps.extend(t["episode_index"].to_pylist())
    states = np.array(states)
    eps = np.array(eps)
    indices = []
    for ep in np.unique(eps):
        mask = eps == ep
        zs = states[mask, 8]
        local = int(np.argmin(zs))
        gidx = int(np.where(mask)[0][local])
        indices.append(gidx)
    return indices


def val_check(policy, preprocessor, postprocessor, eval_ds, eval_indices,
              train_ds, train_idx, device) -> None:
    """跑 eval z-min 帧 + 训练触发帧, 打印 gripper 输出 (>50=触发)。
    用训练 preprocessor (训练 stats 归一化), 单样本无 batch 维, preprocessor 自动加。"""
    policy.eval()
    preprocessor.reset()
    postprocessor.reset()
    print("  val gripper:", end="")
    with torch.inference_mode():
        for label, ds, idx in (
            [("train", train_ds, train_idx)] +
            [("eval", eval_ds, i) for i in eval_indices]
        ):
            item = ds[idx]
            batch = {
                "observation.state": item["observation.state"].float(),
                "observation.images.handeye": item["observation.images.handeye"].float(),
                "observation.images.fixed": item["observation.images.fixed"].float(),
            }
            batch = {k: v.to(device) for k, v in batch.items()}
            batch = preprocessor(batch)
            policy.reset()
            a = policy.select_action(batch)
            a = postprocessor(a)
            a = a.squeeze(0).cpu().numpy()
            if a.ndim == 2:
                a = a[0]
            mark = "OK" if float(a[7]) > 50 else "x "
            print(f"  {label}={a[7]:>7.1f}{mark}", end="")
    print()
    policy.train()


def main():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    init_logging(log_file=str(log_dir / "train.log"))

    output_dir = Path(LOCAL_MODEL_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"Training on device: {device}")

    # ------------------------------------------------------------------
    # 1. Load dataset metadata and derive policy input/output features.
    # ------------------------------------------------------------------
    dataset_metadata = LeRobotDatasetMetadata(LOCAL_DATASET_PATH)
    features = dataset_to_policy_features(dataset_metadata.features)

    # ACTION features are the policy outputs; everything else is input.
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    print("Input features:")
    for k, ft in input_features.items():
        print(f"  {k}: type={ft.type.name} shape={ft.shape}")
    print("Output features:")
    for k, ft in output_features.items():
        print(f"  {k}: type={ft.type.name} shape={ft.shape}")

    # ------------------------------------------------------------------
    # 2. Build ACT config and policy. RESUME_FROM 时从 checkpoint 加载权重续训。
    # ------------------------------------------------------------------
    if RESUME_FROM:
        print(f"从 checkpoint 续训: {RESUME_FROM}")
        policy = ACTPolicy.from_pretrained(RESUME_FROM)
    else:
        cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=CHUNK_SIZE,
            n_action_steps=N_ACTION_STEPS,
            # Backbone: pretrained ResNet18 (lighter, faster to train on small datasets)
            vision_backbone="resnet18",
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            # VAE 关闭：单任务 + 夹爪闭合是稀有脉冲，VAE(z=0 推理)会把稀有事件回归到均值；
            # 关掉 VAE 强制策略基于观测做确定性预测，改善伸手精度和阶段切换。
            use_vae=False,
            device=DEVICE,
        )
        policy = ACTPolicy(cfg)

    policy.train()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, dataset_stats=dataset_metadata.stats
    )

    # ------------------------------------------------------------------
    # 3. Build dataloader.
    # ------------------------------------------------------------------
    # ACT needs action_delta_indices frames of future actions for chunking,
    # and observation_delta_indices for multi-step obs (None → current frame only).
    # ACT's observation_delta_indices is None, so per resolve_delta_timestamps
    # only `action` gets delta timestamps (the future action chunk). Observation
    # state and images are read as single current-frame values — do NOT add
    # `observation.state: [0.0]` or image timestamps, that inserts a spurious
    # time dimension (e.g. state becomes (B,1,D)) and breaks the VAE encoder's
    # tensor concat with a mismatched-ndim error.
    delta_timestamps: dict[str, list[float]] = {
        "action": [i / dataset_metadata.fps for i in policy.config.action_delta_indices],
    }

    image_transforms = build_image_transforms() if IMAGE_AUG else None
    if IMAGE_AUG:
        print("图像增广: 已启用 (brightness/contrast/saturation/hue/sharpness/affine)")

    dataset = LeRobotDataset(
        LOCAL_DATASET_PATH,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=True,  # 避免每个 epoch 重启 worker + 重解视频的 ~100s 停顿
    )

    # ------------------------------------------------------------------
    # 4. Optimizer.
    # ------------------------------------------------------------------
    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())

    # ------------------------------------------------------------------
    # 5. Training loop.
    # ------------------------------------------------------------------
    # 验证监控: eval 数据集 + z-min 帧索引
    eval_ds = None
    eval_indices = []
    if VAL_ON_EVAL:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        eval_root = str(HF_LEROBOT_HOME / VAL_EVAL_DATASET)
        try:
            eval_ds = LeRobotDataset(VAL_EVAL_DATASET)
            eval_indices = find_eval_zmin_frames(eval_root)
            print(f"val 监控: {len(eval_indices)} 个 eval z-min 帧 + 训练帧 idx={TRAIN_SANITY_IDX}")
        except Exception as e:
            print(f"val 监控跳过 (eval 数据集不可用: {type(e).__name__}: {e})")
            eval_ds = None

    print(f"Starting training for {TRAINING_STEPS} steps...")
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, output = policy.forward(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_FREQ == 0:
                loss_val = loss.item()
                print(f"step {step:>6d} | loss: {loss_val:.4f}")

            if VAL_ON_EVAL and eval_ds is not None and step % VAL_FREQ == 0:
                try:
                    val_check(policy, preprocessor, postprocessor, eval_ds, eval_indices,
                              dataset, TRAIN_SANITY_IDX, device)
                except Exception as e:
                    # 监控不是训练信号, 任何报错都不能打断训练
                    print(f"  val 监控跳过本轮 ({type(e).__name__}: {e})")

            if step % SAVE_FREQ == 0 and step > 0:
                ckpt_dir = output_dir / f"checkpoint_{step}"
                ckpt_dir.mkdir(exist_ok=True)
                policy.save_pretrained(ckpt_dir)
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)
                print(f"  Saved checkpoint to {ckpt_dir}")

            step += 1
            if step >= TRAINING_STEPS:
                done = True
                break

    # ------------------------------------------------------------------
    # 6. Save final model.
    # ------------------------------------------------------------------
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()

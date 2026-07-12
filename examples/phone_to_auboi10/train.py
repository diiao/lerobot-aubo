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
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import init_logging

LOCAL_DATASET_PATH = "./datasets/phone_auboi10_full_shift"
LOCAL_MODEL_PATH = "./models/phone_auboi10_aug"

# --- Training hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
TRAINING_STEPS = 40_000
LOG_FREQ = 200
SAVE_FREQ = 5_000

# ACT hyperparameters (tuned for 30fps Aubo data, small dataset)
CHUNK_SIZE = 10       # 10 steps @ 30fps ≈ 0.33s action horizon
N_ACTION_STEPS = 10   # Execute all predicted actions before re-querying


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
    # 2. Build ACT config and policy.
    # ------------------------------------------------------------------
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
        cfg, dataset_stats=dataset_metadata.stats
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
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    # 图像增强：训练时对相机图像随机抖动亮度/对比度/饱和度/色调 + 小幅裁剪平移，
    # 让模型对光照变化和物体位置微调鲁棒（评估时画面的轻微差异不再导致预测漂移）。
    # 只在训练 dataset 生效；评估走 inference_server 不经过这里，保持确定性。
    from torchvision.transforms import v2

    image_transforms = v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        v2.RandomResizedCrop(size=(480, 640), scale=(0.95, 1.0), ratio=(0.97, 1.03), antialias=True),
    ])

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
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    # ------------------------------------------------------------------
    # 5. Training loop.
    # ------------------------------------------------------------------
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

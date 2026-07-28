#!/usr/bin/env python

"""Train a pure ACT policy for the Aubo I10 bamboo task.

The default workflow is intentionally conservative:
- train from scratch on the aggregated new-view dataset;
- split validation by complete episodes;
- keep image augmentation disabled for the first reproducible baseline;
- select checkpoints using validation loss instead of a hand-picked frame.
"""

import os
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import init_logging

LOCAL_DATASET_PATH = os.environ.get("DATASET_PATH", "./datasets/bamboo_newview_full")
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "./models/bamboo_newview_act")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
TRAINING_STEPS = int(os.environ.get("TRAINING_STEPS", "30000"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
LOG_FREQ = 200
VAL_FREQ = 1000
SAVE_FREQ = 5000
MAX_VAL_BATCHES = 50
VAL_FRACTION = 0.2

# Standard ACT CVAE objective is the baseline. Set USE_VAE=0 only for a
# controlled comparison trained from scratch with the same episode split.
USE_VAE = os.environ.get("USE_VAE", "1").strip().lower() not in ("", "0", "false", "no")


def validate_dataset_stats(metadata: LeRobotDatasetMetadata) -> None:
    """Fail early on statistics produced by the historical uint8 overflow bug."""
    if int(metadata.fps) != 25:
        raise RuntimeError(
            f"新视角纯 ACT 数据必须是 25 FPS，当前数据集是 {metadata.fps} FPS"
        )

    for key, feature in metadata.features.items():
        if feature["dtype"] not in ("video", "image"):
            continue
        min_std = float(torch.as_tensor(metadata.stats[key]["std"]).min())
        if min_std < 0.02:
            raise RuntimeError(
                f"{key} min std={min_std:.6f} 异常偏小；请用修复后的代码重新聚合/计算统计"
            )

    action_min = torch.as_tensor(metadata.stats["action"]["min"])
    action_max = torch.as_tensor(metadata.stats["action"]["max"])
    action_names = metadata.features["action"]["names"]
    try:
        gripper_index = action_names.index("ee.gripper_pos")
    except ValueError as exc:
        raise RuntimeError("action features 中缺少 ee.gripper_pos") from exc
    if float(action_min[gripper_index]) > 1.0 or float(action_max[gripper_index]) < 99.0:
        raise RuntimeError(
            "夹爪 action 统计未覆盖释放(0)和吸取(100)，请先检查新录制数据"
        )


def split_episodes(total_episodes: int) -> tuple[list[int], list[int]]:
    """Reserve the latest complete episodes for cross-session validation."""
    if total_episodes < 2:
        raise ValueError("至少需要 2 个 episode 才能划分训练集和验证集")

    num_val = max(1, round(total_episodes * VAL_FRACTION))
    num_val = min(num_val, total_episodes - 1)
    split_at = total_episodes - num_val
    return list(range(split_at)), list(range(split_at, total_episodes))


def make_loader(dataset: LeRobotDataset, *, shuffle: bool, device: torch.device):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=shuffle,
        persistent_workers=(NUM_WORKERS > 0),
    )


@torch.inference_mode()
def evaluate_loss(policy, preprocessor, dataloader, max_batches: int) -> float:
    policy.eval()
    preprocessor.reset()
    total = 0.0
    count = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        total += float(loss.item())
        count += 1
    policy.train()
    if count == 0:
        raise RuntimeError("验证集没有可用 batch")
    return total / count


def save_checkpoint(policy, preprocessor, postprocessor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)


def main():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    init_logging(log_file=str(log_dir / "train.log"))

    output_dir = Path(LOCAL_MODEL_PATH)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"模型目录非空: {output_dir}；请为新实验设置新的 MODEL_PATH"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)

    metadata = LeRobotDatasetMetadata(LOCAL_DATASET_PATH)
    validate_dataset_stats(metadata)
    train_episodes, val_episodes = split_episodes(metadata.total_episodes)
    # Express the ACT horizon in physical time, then convert using the dataset
    # FPS. New 25 Hz recordings therefore use a 25-frame (~1 s) chunk and
    # replan after 4 frames (~0.16 s).
    chunk_size = int(os.environ.get("CHUNK_SIZE", str(round(metadata.fps * 1.0))))
    n_action_steps = int(
        os.environ.get("N_ACTION_STEPS", str(max(1, round(metadata.fps / 6))))
    )
    print(f"Training on {device}")
    print(f"dataset={LOCAL_DATASET_PATH}")
    print(f"train episodes={train_episodes}")
    print(f"val episodes={val_episodes}")

    features = dataset_to_policy_features(metadata.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=USE_VAE,
        device=DEVICE,
    )
    policy = ACTPolicy(config).to(device)
    policy.train()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        dataset_stats=metadata.stats,
    )

    delta_timestamps = {
        "action": [i / metadata.fps for i in policy.config.action_delta_indices],
    }
    train_dataset = LeRobotDataset(
        LOCAL_DATASET_PATH,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
    )
    val_dataset = LeRobotDataset(
        LOCAL_DATASET_PATH,
        episodes=val_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
    )
    train_loader = make_loader(train_dataset, shuffle=True, device=device)
    val_loader = make_loader(val_dataset, shuffle=False, device=device)

    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
    best_val = float("inf")
    step = 0

    print(
        f"Starting pure ACT training: steps={TRAINING_STEPS}, "
        f"chunk={chunk_size}, execute={n_action_steps}, use_vae={USE_VAE}"
    )
    while step < TRAINING_STEPS:
        for batch in train_loader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_FREQ == 0:
                print(f"step {step:>6d} | train_loss={loss.item():.5f}")

            if step > 0 and step % VAL_FREQ == 0:
                val_loss = evaluate_loss(policy, preprocessor, val_loader, MAX_VAL_BATCHES)
                print(f"step {step:>6d} | val_loss={val_loss:.5f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(policy, preprocessor, postprocessor, output_dir / "best")
                    print(f"  saved best checkpoint (val_loss={best_val:.5f})")

            if step > 0 and step % SAVE_FREQ == 0:
                checkpoint_dir = output_dir / f"checkpoint_{step}"
                save_checkpoint(policy, preprocessor, postprocessor, checkpoint_dir)
                print(f"  saved {checkpoint_dir}")

            step += 1
            if step >= TRAINING_STEPS:
                break

    final_val = evaluate_loss(policy, preprocessor, val_loader, MAX_VAL_BATCHES)
    print(f"final validation | val_loss={final_val:.5f}")
    if final_val < best_val:
        best_val = final_val
        save_checkpoint(policy, preprocessor, postprocessor, output_dir / "best")
        print(f"  saved best checkpoint (val_loss={best_val:.5f})")

    save_checkpoint(policy, preprocessor, postprocessor, output_dir)
    print(f"Training complete: {output_dir}; best_val={best_val:.5f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

"""Train a pure ACT policy for the Aubo I10 bamboo task.

The default workflow is intentionally conservative:
- train from scratch on the aggregated new-view dataset;
- split validation by complete episodes;
- keep image augmentation disabled for the first reproducible baseline;
- select checkpoints using validation loss instead of a hand-picked frame.
"""

import hashlib
import inspect
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from lerobot.bamboo_sorting.act_input_contract import (
    ActStateInputContract,
    EpisodeSplit,
    adapt_features_and_stats,
    apply_state_input_contract,
    build_state_input_contract,
    load_episode_split,
)
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import init_logging

_DATASET_PATH = Path(os.environ.get("DATASET_PATH", "./datasets/bamboo_newview_full")).expanduser()
if _DATASET_PATH.exists():
    LOCAL_DATASET_ROOT = _DATASET_PATH.resolve()
    LOCAL_DATASET_REPO_ID = LOCAL_DATASET_ROOT.name
else:
    LOCAL_DATASET_ROOT = None
    LOCAL_DATASET_REPO_ID = str(_DATASET_PATH)
LOCAL_DATASET_PATH = str(LOCAL_DATASET_ROOT or _DATASET_PATH)
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "./models/bamboo_newview_act")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
TRAINING_STEPS = int(os.environ.get("TRAINING_STEPS", "30000"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
SEED = int(os.environ.get("SEED", "42"))
VAL_SEED = int(os.environ.get("VAL_SEED", str(SEED + 1)))
LOG_FREQ = int(os.environ.get("LOG_FREQ", "200"))
VAL_FREQ = int(os.environ.get("VAL_FREQ", "5000"))
SAVE_FREQ = int(os.environ.get("SAVE_FREQ", "5000"))
_VAL_MAX_BATCHES = int(os.environ.get("VAL_MAX_BATCHES", "0"))
MAX_VAL_BATCHES = None if _VAL_MAX_BATCHES == 0 else _VAL_MAX_BATCHES
VAL_FRACTION = 0.2
STATE_INPUT_VARIANT = os.environ.get("STATE_INPUT_VARIANT", "full").strip()
_EPISODE_SPLIT_PATH = os.environ.get("EPISODE_SPLIT_PATH", "").strip()
EPISODE_SPLIT_PATH = Path(_EPISODE_SPLIT_PATH).expanduser() if _EPISODE_SPLIT_PATH else None

# Standard ACT CVAE objective is the baseline. Set USE_VAE=0 only for a
# controlled comparison trained from scratch with the same episode split.
USE_VAE = os.environ.get("USE_VAE", "1").strip().lower() not in ("", "0", "false", "no")


def validate_runtime_settings() -> None:
    """Reject invalid experiment settings before allocating the model."""

    for name, value in (
        ("BATCH_SIZE", BATCH_SIZE),
        ("TRAINING_STEPS", TRAINING_STEPS),
        ("LOG_FREQ", LOG_FREQ),
        ("VAL_FREQ", VAL_FREQ),
        ("SAVE_FREQ", SAVE_FREQ),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if NUM_WORKERS < 0:
        raise ValueError("NUM_WORKERS must be a non-negative integer")
    if SEED < 0:
        raise ValueError("SEED must be a non-negative integer")
    if VAL_SEED < 0:
        raise ValueError("VAL_SEED must be a non-negative integer")
    if _VAL_MAX_BATCHES < 0:
        raise ValueError("VAL_MAX_BATCHES must be zero (full validation) or a positive integer")


def seed_everything(seed: int) -> torch.Generator:
    """Seed model initialization, sampling, and DataLoader shuffling."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(_worker_id: int) -> None:
    """Give Python and NumPy deterministic seeds in each DataLoader worker."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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


def make_loader(
    dataset: LeRobotDataset,
    *,
    shuffle: bool,
    device: torch.device,
    generator: torch.Generator | None = None,
):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=shuffle,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        generator=generator,
    )


@torch.inference_mode()
def evaluate_loss(
    policy,
    preprocessor,
    dataloader,
    max_batches: int | None,
    state_contract: ActStateInputContract,
) -> float:
    """Measure held-out loss with dropout disabled and no parameter updates."""

    was_training = policy.training
    policy.eval()
    preprocessor.reset()
    total = 0.0
    count = 0
    try:
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(VAL_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(VAL_SEED)
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                batch = apply_state_input_contract(batch, state_contract)
                batch = preprocessor(batch)
                loss, _ = policy.forward(batch)
                total += float(loss.item())
                count += 1
    finally:
        policy.train(was_training)
    if count == 0:
        raise RuntimeError("验证集没有可用 batch")
    return total / count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    policy,
    preprocessor,
    postprocessor,
    optimizer,
    output_dir: Path,
    *,
    step: int,
    best_val: float,
) -> None:
    """Save inference files plus optimizer and RNG state for recovery."""

    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    torch.save(
        {
            "step": step,
            "best_val": best_val,
            "optimizer_state_dict": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        output_dir / "training_state.pt",
    )


def write_experiment_manifest(
    output_dir: Path,
    *,
    metadata: LeRobotDatasetMetadata,
    train_episodes: list[int],
    val_episodes: list[int],
    chunk_size: int,
    n_action_steps: int,
    state_contract: ActStateInputContract,
    episode_split: EpisodeSplit | None,
) -> None:
    """Record the exact local inputs and settings used by this run."""

    dataset_root = Path(metadata.root)
    act_source = Path(inspect.getfile(ACTPolicy))
    manifest = {
        "schema_version": "AuboActTrainingRunV2",
        "dataset_path": str(dataset_root.resolve()),
        "dataset_repo_id": LOCAL_DATASET_REPO_ID,
        "dataset_total_episodes": metadata.total_episodes,
        "dataset_total_frames": metadata.total_frames,
        "dataset_info_sha256": sha256_file(dataset_root / "meta" / "info.json"),
        "dataset_stats_sha256": sha256_file(dataset_root / "meta" / "stats.json"),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "episode_split_path": str(episode_split.path) if episode_split is not None else None,
        "episode_split_sha256": sha256_file(episode_split.path) if episode_split is not None else None,
        "state_input_contract": state_contract.as_dict(),
        "normalization_stats_scope": "full_dataset_metadata",
        "validation_scope": "full" if MAX_VAL_BATCHES is None else f"first_{MAX_VAL_BATCHES}_batches",
        "validation_mode": "eval",
        "seed": SEED,
        "validation_seed": VAL_SEED,
        "batch_size": BATCH_SIZE,
        "training_steps": TRAINING_STEPS,
        "validation_frequency": VAL_FREQ,
        "save_frequency": SAVE_FREQ,
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
        "use_vae": USE_VAE,
        "train_script_sha256": sha256_file(Path(__file__)),
        "act_source_path": str(act_source),
        "act_source_sha256": sha256_file(act_source),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    validate_runtime_settings()
    train_generator = seed_everything(SEED)
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

    metadata = LeRobotDatasetMetadata(LOCAL_DATASET_REPO_ID, root=LOCAL_DATASET_ROOT)
    validate_dataset_stats(metadata)
    episode_split = None
    if EPISODE_SPLIT_PATH is None:
        train_episodes, val_episodes = split_episodes(metadata.total_episodes)
    else:
        episode_split = load_episode_split(
            EPISODE_SPLIT_PATH,
            dataset_info_path=Path(metadata.root) / "meta" / "info.json",
            total_episodes=metadata.total_episodes,
        )
        train_episodes = list(episode_split.train_episodes)
        val_episodes = list(episode_split.val_episodes)
    state_contract = build_state_input_contract(metadata.features, STATE_INPUT_VARIANT)
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
    print(
        f"state_input_variant={state_contract.variant} "
        f"({state_contract.source_width}D -> {state_contract.model_width}D)"
    )
    print(f"episode_split={episode_split.path if episode_split is not None else 'default_tail_split'}")
    print(f"seed={SEED}")
    print(f"validation_seed={VAL_SEED}")
    print(
        "validation="
        + ("all held-out frames" if MAX_VAL_BATCHES is None else f"first {MAX_VAL_BATCHES} batches")
    )

    model_features, model_stats = adapt_features_and_stats(
        metadata.features,
        metadata.stats,
        state_contract,
    )
    features = dataset_to_policy_features(model_features)
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
        push_to_hub=False,
    )
    policy = ACTPolicy(config).to(device)
    policy.train()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        dataset_stats=model_stats,
    )

    delta_timestamps = {
        "action": [i / metadata.fps for i in policy.config.action_delta_indices],
    }
    train_dataset = LeRobotDataset(
        LOCAL_DATASET_REPO_ID,
        root=LOCAL_DATASET_ROOT,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
    )
    val_dataset = LeRobotDataset(
        LOCAL_DATASET_REPO_ID,
        root=LOCAL_DATASET_ROOT,
        episodes=val_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
    )
    train_loader = make_loader(
        train_dataset,
        shuffle=True,
        device=device,
        generator=train_generator,
    )
    val_loader = make_loader(val_dataset, shuffle=False, device=device)

    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
    best_val = float("inf")
    step = 0
    write_experiment_manifest(
        output_dir,
        metadata=metadata,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        state_contract=state_contract,
        episode_split=episode_split,
    )

    print(
        f"Starting pure ACT training: steps={TRAINING_STEPS}, "
        f"chunk={chunk_size}, execute={n_action_steps}, use_vae={USE_VAE}, "
        f"state={state_contract.variant}"
    )
    while step < TRAINING_STEPS:
        for batch in train_loader:
            batch = apply_state_input_contract(batch, state_contract)
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_FREQ == 0:
                print(f"step {step:>6d} | train_loss={loss.item():.5f}")

            if step > 0 and step % VAL_FREQ == 0:
                val_loss = evaluate_loss(
                    policy,
                    preprocessor,
                    val_loader,
                    MAX_VAL_BATCHES,
                    state_contract,
                )
                print(f"step {step:>6d} | val_loss={val_loss:.5f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        policy,
                        preprocessor,
                        postprocessor,
                        optimizer,
                        output_dir / "best",
                        step=step,
                        best_val=best_val,
                    )
                    print(f"  saved best checkpoint (val_loss={best_val:.5f})")

            if step > 0 and step % SAVE_FREQ == 0:
                checkpoint_dir = output_dir / f"checkpoint_{step}"
                save_checkpoint(
                    policy,
                    preprocessor,
                    postprocessor,
                    optimizer,
                    checkpoint_dir,
                    step=step,
                    best_val=best_val,
                )
                print(f"  saved {checkpoint_dir}")

            step += 1
            if step >= TRAINING_STEPS:
                break

    final_val = evaluate_loss(
        policy,
        preprocessor,
        val_loader,
        MAX_VAL_BATCHES,
        state_contract,
    )
    print(f"final validation | val_loss={final_val:.5f}")
    if final_val < best_val:
        best_val = final_val
        save_checkpoint(
            policy,
            preprocessor,
            postprocessor,
            optimizer,
            output_dir / "best",
            step=TRAINING_STEPS,
            best_val=best_val,
        )
        print(f"  saved best checkpoint (val_loss={best_val:.5f})")

    save_checkpoint(
        policy,
        preprocessor,
        postprocessor,
        optimizer,
        output_dir,
        step=TRAINING_STEPS,
        best_val=best_val,
    )
    print(f"Training complete: {output_dir}; best_val={best_val:.5f}")


if __name__ == "__main__":
    main()

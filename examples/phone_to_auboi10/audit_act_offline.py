#!/usr/bin/env python
"""Offline acceptance audit for an AUBO pure-ACT checkpoint.

This script never connects to cameras, the robot, or the inference TCP server.
It replays saved dataset observations through a checkpoint and compares the
predicted actions with the recorded demonstrator actions.  In particular, it
reports metrics which the scalar validation loss hides:

* per-action-dimension absolute error;
* suction on/off recall and false positives;
* predicted TCP height at each recorded suction-on transition;
* action discontinuities under the deployed 4-step ACT queue behaviour.

Run this beside ``train.py`` on the GPU machine.  It only reads the model and
dataset, then writes a new timestamped report directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION


DEFAULT_DATASET_PATH = os.environ.get("DATASET_PATH", "./datasets/bamboo_newview_full")
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "./models/bamboo_newview_act/best")


def parse_episode_list(value: str | None, total_episodes: int) -> list[int]:
    """Return explicit episodes, or the same final 20% split used by train.py."""
    if value:
        episodes = [int(item.strip()) for item in value.split(",") if item.strip()]
        if not episodes:
            raise ValueError("--episodes 不能为空")
        invalid = [ep for ep in episodes if ep < 0 or ep >= total_episodes]
        if invalid:
            raise ValueError(f"episode 超出范围 [0, {total_episodes - 1}]: {invalid}")
        return episodes

    num_val = max(1, round(total_episodes * 0.2))
    return list(range(total_episodes - num_val, total_episodes))


def select_device(value: str | None) -> torch.device:
    if value:
        device = torch.device(value)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前 Python 环境未检测到可用 GPU")
    return device


def deployed_queue_actions(chunks: np.ndarray, n_action_steps: int) -> np.ndarray:
    """Reproduce ACT's default queue: replan, then execute the first N actions.

    ``chunks[t]`` is the action chunk predicted from observation t.  The current
    inference server asks ACT for an action every control tick, but ACT internally
    performs a model forward only when its N-action queue is empty.  This helper
    reconstructs the exact action sequence exposed to the robot for a saved
    episode, without running any hardware code.
    """
    if chunks.ndim != 3:
        raise ValueError(f"chunks 必须为 (frames, chunk, action_dim)，实际为 {chunks.shape}")
    if n_action_steps < 1:
        raise ValueError("n_action_steps 必须 >= 1")

    frames, chunk_size, _ = chunks.shape
    if n_action_steps > chunk_size:
        raise ValueError("n_action_steps 不能大于 chunk_size")

    executed = np.empty((frames, chunks.shape[-1]), dtype=np.float32)
    for start in range(0, frames, n_action_steps):
        count = min(n_action_steps, frames - start)
        executed[start : start + count] = chunks[start, :count]
    return executed


def binary_metrics(predicted: np.ndarray, target: np.ndarray, threshold: float = 60.0) -> dict[str, Any]:
    """Metrics for the physical suction state, whose labels are 0 or 100."""
    pred_on = predicted > threshold
    target_on = target > threshold
    tp = int(np.logical_and(pred_on, target_on).sum())
    tn = int(np.logical_and(~pred_on, ~target_on).sum())
    fp = int(np.logical_and(pred_on, ~target_on).sum())
    fn = int(np.logical_and(~pred_on, target_on).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "threshold": threshold,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "target_on_rate": float(target_on.mean()),
        "predicted_on_rate": float(pred_on.mean()),
        "predicted_min": float(predicted.min()),
        "predicted_max": float(predicted.max()),
    }


def action_error_metrics(predicted: np.ndarray, target: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    error = np.abs(predicted - target)
    return {
        name: {
            "mae": float(error[:, dim].mean()),
            "p95_abs_error": float(np.quantile(error[:, dim], 0.95)),
            "max_abs_error": float(error[:, dim].max()),
        }
        for dim, name in enumerate(names)
    }


def first_suction_transition(target_gripper: np.ndarray) -> int | None:
    """Find the first release-to-suction transition in one episode."""
    on = target_gripper > 60.0
    transitions = np.flatnonzero(on & np.concatenate(([True], ~on[:-1])))
    return int(transitions[0]) if len(transitions) else None


def episode_metrics(
    episode_index: int,
    frame_index: np.ndarray,
    predicted: np.ndarray,
    target: np.ndarray,
    action_names: list[str],
    fps: int,
) -> dict[str, Any]:
    xyz_indices = [action_names.index(name) for name in ("ee.x", "ee.y", "ee.z")]
    gripper_index = action_names.index("ee.gripper_pos")
    deltas = np.abs(np.diff(predicted[:, xyz_indices], axis=0))
    transition = first_suction_transition(target[:, gripper_index])
    predicted_on = np.flatnonzero(predicted[:, gripper_index] > 60.0)

    record: dict[str, Any] = {
        "episode_index": episode_index,
        "frames": int(len(predicted)),
        "duration_s": float(len(predicted) / fps),
        "predicted_suction_on_frames": int((predicted[:, gripper_index] > 60.0).sum()),
        "first_predicted_suction_on_s": float(predicted_on[0] / fps) if len(predicted_on) else None,
        "max_xyz_step_m": float(deltas.max()) if len(deltas) else 0.0,
        "xyz_steps_over_1cm": int((deltas > 0.01).any(axis=1).sum()) if len(deltas) else 0,
        "xyz_steps_over_3cm": int((deltas > 0.03).any(axis=1).sum()) if len(deltas) else 0,
    }
    if transition is None:
        record.update(
            {
                "first_true_suction_on_s": None,
                "target_pickup_z_m": None,
                "predicted_pickup_z_m": None,
                "pickup_z_error_m": None,
                "predicted_gripper_at_true_pickup": None,
            }
        )
        return record

    z_index = xyz_indices[2]
    record.update(
        {
            "first_true_suction_on_s": float(frame_index[transition] / fps),
            "target_pickup_z_m": float(target[transition, z_index]),
            "predicted_pickup_z_m": float(predicted[transition, z_index]),
            "pickup_z_error_m": float(predicted[transition, z_index] - target[transition, z_index]),
            "predicted_gripper_at_true_pickup": float(predicted[transition, gripper_index]),
        }
    )
    return record


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report_dir(base: str | None) -> Path:
    if base:
        path = Path(base)
        if path.exists():
            raise FileExistsError(f"报告目录已存在，拒绝覆盖: {path}")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("./reports") / f"act_offline_audit_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_inference_observation(
    batch: dict[str, Any],
    input_features: dict[str, Any],
    *,
    state_gripper_index: int | None = None,
) -> dict[str, Any]:
    """Build a non-mutating inference batch, optionally masking suction state.

    The counterfactual mask answers a crucial closed-loop question: can the
    policy initiate suction while the robot still reports ``gripper_pos=0``, or
    does it only echo the state after a demonstrator has already turned suction
    on?  It deliberately changes no camera pixels or pose values.
    """
    raw_observation = {
        key: batch[key].clone() if isinstance(batch[key], torch.Tensor) else batch[key]
        for key in input_features
        if key in batch
    }
    if set(raw_observation) != set(input_features):
        missing_inputs = set(input_features).difference(raw_observation)
        raise RuntimeError(f"数据批次缺少模型输入: {sorted(missing_inputs)}")
    if state_gripper_index is not None:
        raw_observation["observation.state"][:, state_gripper_index] = 0.0
    raw_observation["task"] = batch["task"][0] if "task" in batch else ""
    raw_observation["robot_type"] = "aubo_i10"
    return raw_observation


def audit(args: argparse.Namespace) -> Path:
    dataset_path = Path(args.dataset_path)
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"未找到模型目录: {model_path}")

    metadata = LeRobotDatasetMetadata(dataset_path)
    episodes = parse_episode_list(args.episodes, metadata.total_episodes)
    action_names = list(metadata.features[ACTION]["names"])
    required = {"ee.x", "ee.y", "ee.z", "ee.gripper_pos"}
    missing = required.difference(action_names)
    if missing:
        raise ValueError(f"action features 缺少必要字段: {sorted(missing)}")
    state_names = list(metadata.features["observation.state"]["names"])
    state_gripper_index = state_names.index("gripper_pos") if "gripper_pos" in state_names else None

    device = select_device(args.device)
    policy = ACTPolicy.from_pretrained(model_path).to(device).eval()
    policy.config.device = str(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    delta_timestamps = {"action": [i / metadata.fps for i in policy.config.action_delta_indices]}
    dataset = LeRobotDataset(dataset_path, episodes=episodes, delta_timestamps=delta_timestamps)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            # Normal replay mirrors inference_server.py: only current observation enters ACT.
            processed = preprocessor(make_inference_observation(batch, policy.config.input_features))
            action_chunks = postprocessor(policy.predict_action_chunk(processed)).cpu().numpy()
            forced_release_chunks = None
            if state_gripper_index is not None:
                forced_processed = preprocessor(
                    make_inference_observation(
                        batch,
                        policy.config.input_features,
                        state_gripper_index=state_gripper_index,
                    )
                )
                forced_release_chunks = postprocessor(policy.predict_action_chunk(forced_processed)).cpu().numpy()
            targets = batch[ACTION][:, 0].cpu().numpy()

            for item in range(len(targets)):
                records.append(
                    {
                        "episode_index": int(batch["episode_index"][item]),
                        "frame_index": int(batch["frame_index"][item]),
                        "dataset_index": int(batch["index"][item]),
                        "target": targets[item].astype(np.float32),
                        "chunk": action_chunks[item].astype(np.float32),
                        "forced_release_chunk": (
                            forced_release_chunks[item].astype(np.float32)
                            if forced_release_chunks is not None
                            else None
                        ),
                    }
                )

    if not records:
        raise RuntimeError("没有可审核的帧")

    report_dir = make_report_dir(args.report_dir)
    frame_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    all_predicted: list[np.ndarray] = []
    all_forced_release_predicted: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for episode_index in episodes:
        group = sorted((row for row in records if row["episode_index"] == episode_index), key=lambda row: row["frame_index"])
        if not group:
            raise RuntimeError(f"episode {episode_index} 没有读取到任何帧")
        chunks = np.stack([row["chunk"] for row in group])
        target = np.stack([row["target"] for row in group])
        predicted = deployed_queue_actions(chunks, policy.config.n_action_steps)
        forced_release_predicted = None
        if group[0]["forced_release_chunk"] is not None:
            forced_release_chunks = np.stack([row["forced_release_chunk"] for row in group])
            forced_release_predicted = deployed_queue_actions(forced_release_chunks, policy.config.n_action_steps)
        frames = np.asarray([row["frame_index"] for row in group])
        row = episode_metrics(episode_index, frames, predicted, target, action_names, int(metadata.fps))
        if forced_release_predicted is not None:
            forced_on = np.flatnonzero(forced_release_predicted[:, action_names.index("ee.gripper_pos")] > 60.0)
            row["forced_release_predicted_suction_on_frames"] = int(len(forced_on))
            row["forced_release_first_predicted_suction_on_s"] = (
                float(forced_on[0] / metadata.fps) if len(forced_on) else None
            )
            all_forced_release_predicted.append(forced_release_predicted)
        episode_rows.append(row)
        all_predicted.append(predicted)
        all_targets.append(target)

        for frame_pos, (row, pred, truth) in enumerate(zip(group, predicted, target, strict=True)):
            output: dict[str, Any] = {
                "episode_index": row["episode_index"],
                "frame_index": row["frame_index"],
                "dataset_index": row["dataset_index"],
            }
            for dim, name in enumerate(action_names):
                output[f"target.{name}"] = float(truth[dim])
                output[f"predicted.{name}"] = float(pred[dim])
                if forced_release_predicted is not None:
                    output[f"forced_release_predicted.{name}"] = float(forced_release_predicted[frame_pos, dim])
            frame_rows.append(output)

    predicted = np.concatenate(all_predicted)
    target = np.concatenate(all_targets)
    gripper_index = action_names.index("ee.gripper_pos")
    z_errors = [row["pickup_z_error_m"] for row in episode_rows if row["pickup_z_error_m"] is not None]
    report = {
        "purpose": "Offline pure-ACT acceptance audit; no robot or camera was connected.",
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "device": str(device),
        "fps": int(metadata.fps),
        "episodes": episodes,
        "frames": int(len(target)),
        "deployed_execution": {
            "chunk_size": int(policy.config.chunk_size),
            "n_action_steps": int(policy.config.n_action_steps),
            "description": "Replan every n_action_steps and execute the first actions from each ACT chunk.",
        },
        "per_action_error": action_error_metrics(predicted, target, action_names),
        "suction": binary_metrics(predicted[:, gripper_index], target[:, gripper_index]),
        "forced_release_state_counterfactual": (
            {
                "description": (
                    "Every input observation.state.gripper_pos was forced to 0 before inference. "
                    "This tests whether the policy proactively commands the first suction event."
                ),
                "suction": binary_metrics(
                    np.concatenate(all_forced_release_predicted)[:, gripper_index], target[:, gripper_index]
                ),
            }
            if all_forced_release_predicted
            else {"available": False, "reason": "observation.state has no gripper_pos"}
        ),
        "pickup_height": {
            "episodes_with_true_suction_transition": len(z_errors),
            "mean_signed_error_m": float(np.mean(z_errors)) if z_errors else None,
            "mean_abs_error_m": float(np.mean(np.abs(z_errors))) if z_errors else None,
            "max_abs_error_m": float(np.max(np.abs(z_errors))) if z_errors else None,
        },
        "episode_summary_file": "per_episode.csv",
        "frame_predictions_file": "per_frame_predictions.csv",
    }
    (report_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report_dir / "per_episode.csv", episode_rows)
    write_csv(report_dir / "per_frame_predictions.csv", frame_rows)
    return report_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线审核 AUBO 纯 ACT checkpoint，不连接机器人")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", help="逗号分隔的 episode，例如 24,25,26；默认最后 20 percent")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", help="例如 cuda 或 cpu；默认自动选择")
    parser.add_argument("--report-dir", help="新报告目录；已存在时会拒绝覆盖")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = audit(args)
    print(f"离线审核完成，报告已写入: {report_dir}")


if __name__ == "__main__":
    main()

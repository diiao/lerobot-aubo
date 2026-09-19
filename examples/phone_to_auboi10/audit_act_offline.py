#!/usr/bin/env python
"""Offline acceptance audit for an AUBO pure-ACT checkpoint.

This script never connects to cameras, the robot, or the inference TCP server.
It replays saved dataset observations through a checkpoint and compares the
predicted actions with the recorded demonstrator actions.

Run this beside ``train.py`` on the GPU machine.  It only reads the model and
dataset, then writes a new timestamped report directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.bamboo_sorting.act_input_contract import (
    STATE_INPUT_CONTRACT_FILENAME,
    ActStateInputContract,
    apply_state_input_contract,
    build_state_input_contract,
    load_state_input_contract,
)
from lerobot.bamboo_sorting.contracts import ACTION_FIELD_NAMES
from lerobot.bamboo_sorting.thin_safety_gate import ThinSafetyGate, ThinSafetyGateLimits
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION


DEFAULT_DATASET_PATH = os.environ.get("DATASET_PATH", "./datasets/bamboo_newview_full")
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "./models/bamboo_newview_act/best")
SUCTION_ON_THRESHOLD = 60.0
SUCTION_OFF_THRESHOLD = 20.0
QUEUE_N_ACTION_STEPS = 4
TEMPORAL_ENSEMBLE_COEFF = 0.01
LATENT_MODE_ZERO = "zero"
LATENT_MODE_EPISODE_SAMPLE = "episode_sample"
LATENT_MODES = (LATENT_MODE_ZERO, LATENT_MODE_EPISODE_SAMPLE)
TRANSITION_WINDOW = 10
CONTACT_OFFSETS = (-10, -5, 0, 1, 5, 10)
ROTATION_THRESHOLDS_DEG = (5.0, 10.0, 20.0, 30.0, 60.0, 90.0)
XYZ_THRESHOLDS_M = (0.001, 0.005, 0.008, 0.01, 0.03)
J6_THRESHOLDS_RAD = (0.01, 0.03, 0.05, 0.1, 0.2)
CAMERA_KEYS = ("observation.images.handeye", "observation.images.fixed")

# Current evaluate_split.py defaults.  Keeping them explicit in the report makes
# drift visible; tests compare the replay against the client helpers.
CLIENT_WORKSPACE_MIN_M = (0.0, -0.85, 0.05)
CLIENT_WORKSPACE_MAX_M = (0.67, -0.25, 0.30)
CLIENT_MAX_EE_STEP_M = 0.008
CLIENT_MAX_J6_STEP_RAD = 0.03
CLIENT_MAX_OBSERVATION_AGE_MS = 2000.0
CLIENT_MAX_ACTION_CHUNK_AGE_MS = 2000.0
CLIENT_J6_TARGET_MIN_RAD = -4.47
CLIENT_J6_TARGET_MAX_RAD = -1.39
CLIENT_GRIPPER_CHUNK_CLOSE_SCORE = 20.0
CLIENT_GRIPPER_CHUNK_OPEN_SCORE = 8.0
CLIENT_GRIPPER_CHUNK_HOLD_SCORE = 40.0
CLIENT_MOTION_CHUNK_MIN_Z_DROP_M = 0.02
CLIENT_USE_GRIPPER_CHUNK_LOOKAHEAD = False
CLIENT_USE_MOTION_CHUNK_LOOKAHEAD = False


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def deployed_temporal_ensemble_actions(chunks: np.ndarray, coeff: float) -> np.ndarray:
    """Replay ACT temporal ensembling with the real ACTTemporalEnsembler."""
    if chunks.ndim != 3:
        raise ValueError(f"chunks 必须为 (frames, chunk, action_dim)，实际为 {chunks.shape}")
    frames, chunk_size, action_dim = chunks.shape
    ensembler = ACTTemporalEnsembler(coeff, chunk_size)
    executed = np.empty((frames, action_dim), dtype=np.float32)
    for step in range(frames):
        action = ensembler.update(torch.from_numpy(np.ascontiguousarray(chunks[step : step + 1])))
        executed[step] = action.squeeze(0).detach().cpu().numpy()
    return executed


def deployed_hybrid_actions(
    temporal_ensemble: np.ndarray,
    gripper_source: np.ndarray,
    gripper_index: int,
) -> np.ndarray:
    """Use smoothed motion dimensions with an explicitly selected gripper source."""
    if temporal_ensemble.shape != gripper_source.shape:
        raise ValueError("temporal_ensemble and gripper_source must have identical shapes")
    result = temporal_ensemble.copy()
    result[:, gripper_index] = gripper_source[:, gripper_index]
    return result


def _clip_vector_step(target: np.ndarray, anchor: np.ndarray, limit: float) -> tuple[np.ndarray, bool]:
    """Clip one Cartesian vector toward an anchor, matching evaluate_split.py."""
    delta = target - anchor
    distance = float(np.linalg.norm(delta))
    effective_limit = limit * (1.0 - 1e-6)
    if distance > effective_limit and distance > 0.0:
        return anchor + delta * (effective_limit / distance), True
    return target, False


def replay_current_client_guard(
    selected_actions: np.ndarray,
    raw_chunks: np.ndarray,
    state: np.ndarray,
    *,
    action_names: list[str],
    state_names: list[str],
    fps: int,
    use_gripper_chunk_lookahead: bool = CLIENT_USE_GRIPPER_CHUNK_LOOKAHEAD,
    use_motion_chunk_lookahead: bool = CLIENT_USE_MOTION_CHUNK_LOOKAHEAD,
) -> list[dict[str, Any]]:
    """Replay the current default client transforms and its non-IK safety gate.

    The replay uses saved expert observations as measured anchors.  Timing is set
    to zero age and IK is deliberately disabled because neither can be recovered
    from a dataset frame without a live RPC round trip or a robot-side IK service.
    Every such limitation is recorded in the final report.
    """
    if list(action_names) != list(ACTION_FIELD_NAMES):
        raise ValueError(
            "action names do not match ActionSchemaV1: "
            f"dataset={action_names}, schema={list(ACTION_FIELD_NAMES)}"
        )
    if selected_actions.ndim != 2 or raw_chunks.ndim != 3 or state.ndim != 2:
        raise ValueError("selected_actions/raw_chunks/state dimensions are invalid")
    if not (len(selected_actions) == len(raw_chunks) == len(state)):
        raise ValueError("selected_actions/raw_chunks/state must have equal frame counts")

    action_index = {name: action_names.index(name) for name in action_names}
    state_index = {name: state_names.index(name) for name in state_names}
    tcp_state_indices = [state_index[name] for name in ("ee.x", "ee.y", "ee.z")]
    rot_state_indices = [state_index[name] for name in ("ee.wx", "ee.wy", "ee.wz")]
    gripper_state_index = state_index["gripper_pos"]
    j6_state_index = state_index["J6"]
    gripper_action_index = action_index["ee.gripper_pos"]

    gate = ThinSafetyGate(
        ThinSafetyGateLimits(
            workspace_min_m=CLIENT_WORKSPACE_MIN_M,
            workspace_max_m=CLIENT_WORKSPACE_MAX_M,
            max_ee_step_m=CLIENT_MAX_EE_STEP_M,
            max_ee_speed_mps=CLIENT_MAX_EE_STEP_M * fps,
            max_j6_step_rad=CLIENT_MAX_J6_STEP_RAD,
            max_observation_age_ms=CLIENT_MAX_OBSERVATION_AGE_MS,
            max_chunk_age_ms=CLIENT_MAX_ACTION_CHUNK_AGE_MS,
            control_fps=fps,
            require_previous_tcp=True,
            require_ik=False,
        )
    )
    suction_on = bool(state[0, gripper_state_index] > SUCTION_ON_THRESHOLD)
    locked_rotvec = state[0, rot_state_indices].astype(np.float64)
    processor_last_pos: np.ndarray | None = None
    rows: list[dict[str, Any]] = []

    for frame_pos, selected in enumerate(selected_actions):
        action = {name: float(selected[index]) for index, name in enumerate(action_names)}
        chunk = raw_chunks[frame_pos]
        measured_tcp = state[frame_pos, tcp_state_indices].astype(np.float64)
        measured_j6 = math.radians(float(state[frame_pos, j6_state_index]))

        motion_lookahead = False
        if use_motion_chunk_lookahead and not suction_on:
            min_z_index = int(np.argmin(chunk[:, action_index["ee.z"]]))
            planned = chunk[min_z_index]
            if action["ee.z"] - float(planned[action_index["ee.z"]]) >= CLIENT_MOTION_CHUNK_MIN_Z_DROP_M:
                for name in ("ee.j6_target", "ee.x", "ee.y", "ee.z"):
                    action[name] = float(planned[action_index[name]])
                motion_lookahead = True

        gripper_lookahead_mode = "disabled"
        if use_gripper_chunk_lookahead:
            chunk_max = float(np.max(chunk[:, gripper_action_index]))
            if chunk_max > CLIENT_GRIPPER_CHUNK_CLOSE_SCORE:
                action["ee.gripper_pos"] = 100.0
                gripper_lookahead_mode = "close"
            elif chunk_max < CLIENT_GRIPPER_CHUNK_OPEN_SCORE:
                gripper_lookahead_mode = "pass"
            else:
                action["ee.gripper_pos"] = CLIENT_GRIPPER_CHUNK_HOLD_SCORE
                gripper_lookahead_mode = "hold"

        raw_gripper = float(action["ee.gripper_pos"])
        if raw_gripper > SUCTION_ON_THRESHOLD:
            next_suction_on = True
        elif raw_gripper < SUCTION_OFF_THRESHOLD:
            next_suction_on = False
        else:
            next_suction_on = suction_on
        action["ee.gripper_pos"] = 100.0 if next_suction_on else 0.0

        raw_vector = tuple(float(action[name]) for name in action_names)
        raw_gate = gate.evaluate(
            [raw_vector],
            now_monotonic_s=1.0,
            observation_sync_timestamp_s=1.0,
            chunk_created_monotonic_s=1.0,
            previous_tcp_m=measured_tcp,
            previous_j6_rad=measured_j6,
        )
        raw_xyz_step = math.dist(
            tuple(raw_vector[action_index[name]] for name in ("ee.x", "ee.y", "ee.z")),
            measured_tcp,
        )
        raw_j6_step = abs(float(action["ee.j6_target"]) - measured_j6)

        # Current default HOLD_LOCKED_EE_POSE=1.
        for value, name in zip(locked_rotvec, ("ee.wx", "ee.wy", "ee.wz"), strict=True):
            action[name] = float(value)

        target_tcp = np.asarray([action[name] for name in ("ee.x", "ee.y", "ee.z")], dtype=np.float64)
        target_tcp, measured_xyz_clipped = _clip_vector_step(
            target_tcp, measured_tcp, CLIENT_MAX_EE_STEP_M
        )
        for value, name in zip(target_tcp, ("ee.x", "ee.y", "ee.z"), strict=True):
            action[name] = float(value)
        j6_delta = float(action["ee.j6_target"]) - measured_j6
        measured_j6_clipped = abs(j6_delta) > CLIENT_MAX_J6_STEP_RAD * (1.0 - 1e-6)
        if measured_j6_clipped:
            action["ee.j6_target"] = measured_j6 + math.copysign(
                CLIENT_MAX_J6_STEP_RAD * (1.0 - 1e-6), j6_delta
            )

        # AuboEEBoundsAndSafety runs after the measured-pose limiter.  It clips
        # workspace first, then applies a stateful last-command step limiter.
        bounded_tcp = np.clip(
            np.asarray([action[name] for name in ("ee.x", "ee.y", "ee.z")]),
            CLIENT_WORKSPACE_MIN_M,
            CLIENT_WORKSPACE_MAX_M,
        )
        workspace_clipped = not np.allclose(bounded_tcp, target_tcp, rtol=0.0, atol=0.0)
        if processor_last_pos is None:
            processor_last_pos = measured_tcp.copy()
        processor_delta = bounded_tcp - processor_last_pos
        processor_distance = float(np.linalg.norm(processor_delta))
        processor_step_clipped = processor_distance > CLIENT_MAX_EE_STEP_M
        if processor_step_clipped and processor_distance > 0.0:
            bounded_tcp = processor_last_pos + processor_delta * (
                CLIENT_MAX_EE_STEP_M / processor_distance
            )
        processor_last_pos = bounded_tcp.copy()
        for value, name in zip(bounded_tcp, ("ee.x", "ee.y", "ee.z"), strict=True):
            action[name] = float(value)

        final_vector = tuple(float(action[name]) for name in action_names)
        envelope_rejected = not (
            CLIENT_J6_TARGET_MIN_RAD
            <= float(action["ee.j6_target"])
            <= CLIENT_J6_TARGET_MAX_RAD
        )
        final_gate = gate.evaluate(
            [final_vector],
            now_monotonic_s=1.0,
            observation_sync_timestamp_s=1.0,
            chunk_created_monotonic_s=1.0,
            previous_tcp_m=measured_tcp,
            previous_j6_rad=measured_j6,
        )
        final_reasons = list(final_gate.reasons)
        if envelope_rejected:
            final_reasons.insert(0, "task_j6_envelope")
        final_xyz_step = math.dist(
            tuple(final_vector[action_index[name]] for name in ("ee.x", "ee.y", "ee.z")),
            measured_tcp,
        )
        final_j6_step = abs(float(action["ee.j6_target"]) - measured_j6)
        rows.append(
            {
                "frame_pos": frame_pos,
                "raw_gate_reasons": list(raw_gate.reasons),
                "final_gate_reasons": final_reasons,
                "raw_xyz_step_m": raw_xyz_step,
                "final_xyz_step_m": final_xyz_step,
                "raw_j6_step_rad": raw_j6_step,
                "final_j6_step_rad": final_j6_step,
                "measured_xyz_clipped": measured_xyz_clipped,
                "measured_j6_clipped": measured_j6_clipped,
                "workspace_clipped": workspace_clipped,
                "processor_step_clipped": processor_step_clipped,
                "motion_chunk_lookahead_overrode": motion_lookahead,
                "gripper_chunk_lookahead_mode": gripper_lookahead_mode,
                "suction_on": next_suction_on,
            }
        )
        if not final_reasons:
            suction_on = next_suction_on
    return rows


def summarize_client_guard_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate frame-level client replay evidence without hiding rejections."""
    raw_reasons = Counter(reason for row in rows for reason in row["raw_gate_reasons"])
    final_reasons = Counter(reason for row in rows for reason in row["final_gate_reasons"])
    gripper_modes = Counter(row["gripper_chunk_lookahead_mode"] for row in rows)

    def first_rejection_by_episode(reason_key: str) -> list[dict[str, Any]]:
        first: dict[int, dict[str, Any]] = {}
        for row in rows:
            reasons = row[reason_key]
            episode_index = row.get("episode_index")
            if reasons and episode_index is not None and episode_index not in first:
                first[int(episode_index)] = {
                    "episode_index": int(episode_index),
                    "frame_pos": int(row["frame_pos"]),
                    "reasons": list(reasons),
                }
        return [first[index] for index in sorted(first)]

    return {
        "frames": len(rows),
        "raw_gate_pass_count": sum(not row["raw_gate_reasons"] for row in rows),
        "raw_gate_reject_count": sum(bool(row["raw_gate_reasons"]) for row in rows),
        "raw_gate_reason_counts": dict(sorted(raw_reasons.items())),
        "post_transform_gate_pass_count": sum(not row["final_gate_reasons"] for row in rows),
        "post_transform_gate_reject_count": sum(bool(row["final_gate_reasons"]) for row in rows),
        "post_transform_gate_reason_counts": dict(sorted(final_reasons.items())),
        "first_raw_rejection_by_episode": first_rejection_by_episode("raw_gate_reasons"),
        "first_post_transform_rejection_by_episode": first_rejection_by_episode(
            "final_gate_reasons"
        ),
        "raw_xyz_step_m": distribution_report(
            np.asarray([row["raw_xyz_step_m"] for row in rows]), XYZ_THRESHOLDS_M
        ),
        "post_transform_xyz_step_m": distribution_report(
            np.asarray([row["final_xyz_step_m"] for row in rows]), XYZ_THRESHOLDS_M
        ),
        "raw_j6_step_rad": distribution_report(
            np.asarray([row["raw_j6_step_rad"] for row in rows]), J6_THRESHOLDS_RAD
        ),
        "post_transform_j6_step_rad": distribution_report(
            np.asarray([row["final_j6_step_rad"] for row in rows]), J6_THRESHOLDS_RAD
        ),
        "transform_counts": {
            "measured_xyz_step_clip": sum(row["measured_xyz_clipped"] for row in rows),
            "measured_j6_step_clip": sum(row["measured_j6_clipped"] for row in rows),
            "workspace_clip": sum(row["workspace_clipped"] for row in rows),
            "processor_last_command_step_clip": sum(
                row["processor_step_clipped"] for row in rows
            ),
            "motion_chunk_lookahead_override": sum(
                row["motion_chunk_lookahead_overrode"] for row in rows
            ),
            "gripper_chunk_lookahead_modes": dict(sorted(gripper_modes.items())),
        },
        "limitations": {
            "teacher_forced_observations": True,
            "observation_age_replayed": False,
            "chunk_age_replayed": False,
            "ik_replayed": False,
            "reason": (
                "Saved frames contain neither live RPC latency nor the controller-side "
                "inverse-kinematics service. Timing is evaluated at zero age and the gate "
                "uses require_ik=False; these items remain unresolved live preflight gates."
            ),
        },
    }


def binary_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    predicted_on: np.ndarray | None = None,
    target_on: np.ndarray | None = None,
) -> dict[str, Any]:
    """Metrics for latched suction state while retaining raw prediction range."""
    if predicted_on is None:
        predicted_on = apply_gripper_hysteresis(predicted)
    if target_on is None:
        target_on = apply_gripper_hysteresis(target)
    pred_on = np.asarray(predicted_on, dtype=bool)
    target_on = np.asarray(target_on, dtype=bool)
    if pred_on.shape != target_on.shape or pred_on.shape != np.asarray(predicted).shape:
        raise ValueError("gripper value/state arrays must have identical shapes")
    tp = int(np.logical_and(pred_on, target_on).sum())
    tn = int(np.logical_and(~pred_on, ~target_on).sum())
    fp = int(np.logical_and(pred_on, ~target_on).sum())
    fn = int(np.logical_and(~pred_on, target_on).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "on_threshold": SUCTION_ON_THRESHOLD,
        "off_threshold": SUCTION_OFF_THRESHOLD,
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


def apply_gripper_hysteresis(
    values: np.ndarray,
    *,
    initial_on: bool = False,
    on_threshold: float = SUCTION_ON_THRESHOLD,
    off_threshold: float = SUCTION_OFF_THRESHOLD,
) -> np.ndarray:
    """Replay the client's >60 on, <20 off, middle-band hold state machine."""
    if off_threshold >= on_threshold:
        raise ValueError("off_threshold must be lower than on_threshold")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"gripper values must be one-dimensional; got {array.shape}")
    state = bool(initial_on)
    states = np.empty(array.shape, dtype=bool)
    for index, value in enumerate(array):
        if value > on_threshold:
            state = True
        elif value < off_threshold:
            state = False
        states[index] = state
    return states


def find_threshold_transitions(
    values: np.ndarray,
    *,
    initial_on: bool = False,
) -> list[dict[str, Any]]:
    """Return every latched off→on and on→off transition after frame 0."""
    on = apply_gripper_hysteresis(values, initial_on=initial_on)
    events: list[dict[str, Any]] = []
    on_ordinal = 0
    off_ordinal = 0
    for frame in range(1, len(on)):
        if on[frame] and not on[frame - 1]:
            events.append({"kind": "on", "frame": int(frame), "ordinal": on_ordinal})
            on_ordinal += 1
        elif on[frame - 1] and not on[frame]:
            events.append({"kind": "off", "frame": int(frame), "ordinal": off_ordinal})
            off_ordinal += 1
    return events


def summarize_target_suction_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Inventory target-label transitions; this does not measure model recall."""
    on_events = [event for event in events if event["kind"] == "on"]
    off_events = [event for event in events if event["kind"] == "off"]
    return {
        "target_on_count": len(on_events),
        "target_off_count": len(off_events),
        "on_events": [
            {
                "episode_index": event["episode_index"],
                "frame": event["target_frame"],
                "ordinal": event["ordinal"],
            }
            for event in on_events
        ],
        "off_events": [
            {
                "episode_index": event["episode_index"],
                "frame": event["target_frame"],
                "ordinal": event["ordinal"],
            }
            for event in off_events
        ],
    }


def _desired_on(kind: str) -> bool:
    if kind not in {"on", "off"}:
        raise ValueError(f"未知切换类型: {kind}")
    return kind == "on"


def match_transition_events(
    target_events: list[dict[str, Any]],
    predicted_events: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int], int], dict[str, int]]:
    """Pair same-kind crossings uniquely while ignoring extra predicted crossings.

    The previous ordinal-only pairing shifted every later match after one early
    spurious pulse.  This sequence-alignment objective first maximizes the number
    of unique monotonic matches, then minimizes their total frame distance.
    """
    matched: dict[tuple[str, int], int] = {}
    unmatched_predicted: dict[str, int] = {}
    for kind in ("on", "off"):
        targets = sorted(
            (event for event in target_events if event["kind"] == kind),
            key=lambda event: event["frame"],
        )
        predictions = sorted(
            (event for event in predicted_events if event["kind"] == kind),
            key=lambda event: event["frame"],
        )

        @lru_cache(maxsize=None)
        def solve(
            target_index: int, prediction_index: int
        ) -> tuple[int, int, tuple[tuple[int, int], ...]]:
            if target_index == len(targets) or prediction_index == len(predictions):
                return 0, 0, ()
            tail_matches, tail_cost, tail_pairs = solve(target_index + 1, prediction_index + 1)
            candidates = [
                (
                    tail_matches + 1,
                    tail_cost
                    + abs(
                        int(targets[target_index]["frame"])
                        - int(predictions[prediction_index]["frame"])
                    ),
                    ((target_index, prediction_index),) + tail_pairs,
                ),
                solve(target_index, prediction_index + 1),
                solve(target_index + 1, prediction_index),
            ]
            return max(candidates, key=lambda result: (result[0], -result[1]))

        match_count, _cost, pairs = solve(0, 0)
        for target_index, prediction_index in pairs:
            target = targets[target_index]
            matched[(kind, int(target["ordinal"]))] = int(
                predictions[prediction_index]["frame"]
            )
        unmatched_predicted[kind] = len(predictions) - match_count
    return matched, unmatched_predicted


def transition_timing(
    predicted: np.ndarray,
    event_frame: int,
    kind: str,
    predicted_frame: int | None,
    initial_on: bool = False,
    window: int = TRANSITION_WINDOW,
) -> dict[str, Any]:
    """Compare one target switch against the matching predicted switch."""
    pred_on = apply_gripper_hysteresis(predicted, initial_on=initial_on)
    want_on = _desired_on(kind)
    n = len(pred_on)
    start = max(0, event_frame - window)
    end_before = event_frame
    end_after = min(n, event_frame + window + 1)
    pre = pred_on[start:end_before]
    post = pred_on[event_frame:end_after]
    pre_false_trigger = bool(np.any(pre) if want_on else np.any(~pre)) if len(pre) else False
    if want_on:
        post_still_missed = bool(len(post) == 0 or not np.any(post))
    else:
        post_still_missed = bool(len(post) == 0 or not np.any(~post))
    return {
        "predicted_cross_frame": predicted_frame,
        "timing_error_frames": None if predicted_frame is None else int(predicted_frame - event_frame),
        "pre_window_false_trigger": pre_false_trigger,
        "post_window_still_missed": post_still_missed,
    }


def rotvec_to_matrix(rotvecs: np.ndarray) -> np.ndarray:
    """Rodrigues formula, batched over leading dimensions. Shape (..., 3) -> (..., 3, 3)."""
    rv = np.asarray(rotvecs, dtype=np.float64)
    original = rv.shape[:-1]
    flat = rv.reshape(-1, 3)
    theta = np.linalg.norm(flat, axis=1)
    matrices = np.zeros((flat.shape[0], 3, 3), dtype=np.float64)
    small = theta < 1e-12
    matrices[small] = np.eye(3)
    large = ~small
    if np.any(large):
        th = theta[large][:, None]
        axis = flat[large] / th
        kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
        zeros = np.zeros_like(kx)
        k = np.stack(
            [zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros],
            axis=1,
        ).reshape(-1, 3, 3)
        eye = np.eye(3)[None, :, :]
        sth = np.sin(theta[large])[:, None, None]
        cth = np.cos(theta[large])[:, None, None]
        matrices[large] = eye + sth * k + (1.0 - cth) * (k @ k)
    return matrices.reshape(*original, 3, 3)


def geodesic_angle_deg(rotvecs_a: np.ndarray, rotvecs_b: np.ndarray) -> np.ndarray:
    """Relative rotation angle in degrees; +π and −π about the same axis are equivalent."""
    ra = rotvec_to_matrix(rotvecs_a)
    rb = rotvec_to_matrix(rotvecs_b)
    relative = np.matmul(np.swapaxes(ra, -1, -2), rb)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def distribution_report(values: np.ndarray, thresholds: tuple[float, ...] | list[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None, "exceed": {}}
    exceed = {}
    for threshold in thresholds:
        count = int(np.sum(data > threshold))
        exceed[str(threshold)] = {"count": count, "ratio": float(count / data.size)}
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "p50": float(np.quantile(data, 0.50)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(data.max()),
        "exceed": exceed,
    }


def action_continuity_values(
    actions: np.ndarray,
    action_names: list[str],
) -> dict[str, np.ndarray]:
    """Measure adjacent command changes within one episode, never across resets."""
    if actions.ndim != 2 or actions.shape[1] != len(action_names):
        raise ValueError("actions must have shape (frames, len(action_names))")
    xyz_indices = [action_names.index(name) for name in ("ee.x", "ee.y", "ee.z")]
    rot_indices = [action_names.index(name) for name in ("ee.wx", "ee.wy", "ee.wz")]
    j6_index = action_names.index("ee.j6_target")
    return {
        "xyz_step_l2_m": np.linalg.norm(np.diff(actions[:, xyz_indices], axis=0), axis=1),
        "j6_step_abs_rad": np.abs(np.diff(actions[:, j6_index])),
        "rotation_step_geodesic_deg": geodesic_angle_deg(
            actions[:-1, rot_indices], actions[1:, rot_indices]
        ),
    }


def summarize_transition_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    errors = [row["timing_error_frames"] for row in rows if row["timing_error_frames"] is not None]
    return {
        "count": len(rows),
        "matched_crossings": len(errors),
        "unmatched_crossings": sum(1 for row in rows if row["predicted_cross_frame"] is None),
        "mean_timing_error_frames": float(np.mean(errors)) if errors else None,
        "mean_abs_timing_error_frames": float(np.mean(np.abs(errors))) if errors else None,
        "pre_window_false_trigger_count": int(sum(bool(row["pre_window_false_trigger"]) for row in rows)),
        "post_window_still_missed_count": int(sum(bool(row["post_window_still_missed"]) for row in rows)),
    }


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
    state_contract: ActStateInputContract | None = None,
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
    if state_contract is not None:
        raw_observation = apply_state_input_contract(raw_observation, state_contract)
    if state_gripper_index is not None:
        raw_observation["observation.state"] = raw_observation["observation.state"].clone()
        raw_observation["observation.state"][:, state_gripper_index] = 0.0
    raw_observation["task"] = batch["task"][0] if "task" in batch else ""
    raw_observation["robot_type"] = "aubo_i10"
    return raw_observation


def episode_fixed_latent_batch(
    episode_indices: torch.Tensor,
    latent_dim: int,
    *,
    seed: int,
    device: torch.device,
    cache: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Return one deterministic sampled latent per episode, independent of batching."""

    if episode_indices.ndim != 1:
        raise ValueError("episode_indices must be a one-dimensional batch")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if seed < 0:
        raise ValueError("latent seed must be non-negative")

    rows = []
    for raw_episode_index in episode_indices.detach().cpu().tolist():
        episode_index = int(raw_episode_index)
        if episode_index not in cache:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed * 1_000_003 + episode_index)
            cache[episode_index] = torch.randn(
                latent_dim,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        rows.append(cache[episode_index])
    return torch.stack(rows)


def load_episode_video_meta(dataset_path: Path) -> dict[int, dict[str, Any]]:
    import pyarrow.parquet as pq

    files = sorted((dataset_path / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 episode parquet: {dataset_path / 'meta' / 'episodes'}")
    rows: dict[int, dict[str, Any]] = {}
    for file in files:
        table = pq.read_table(file)
        for record in table.to_pylist():
            rows[int(record["episode_index"])] = record
    return rows


def video_file_for(dataset_path: Path, record: dict[str, Any], camera_key: str) -> Path:
    chunk = int(record[f"videos/{camera_key}/chunk_index"])
    file_index = int(record[f"videos/{camera_key}/file_index"])
    return dataset_path / "videos" / camera_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"


def read_video_frame_bgr(path: Path, timestamp_s: float) -> np.ndarray:
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        seek_s = max(0.0, timestamp_s - 0.08)
        container.seek(int(seek_s * av.time_base))
        best = None
        best_dt = 1e9
        for frame in container.decode(video=0):
            time_s = float(frame.time) if frame.time is not None else 0.0
            delta = abs(time_s - timestamp_s)
            if delta < best_dt:
                best_dt = delta
                best = frame
            if time_s >= timestamp_s + (1.0 / max(float(stream.average_rate or 25), 1.0)):
                break
        if best is None:
            raise RuntimeError(f"无法从 {path} 读取 timestamp={timestamp_s:.4f}s 的帧")
        return best.to_ndarray(format="bgr24")
    finally:
        container.close()


def _annotate_panel(image: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    canvas = image.copy()
    y = 18
    for line in lines:
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return canvas


def render_contact_sheet(panels: list[list[np.ndarray | None]], cell_h: int = 240) -> np.ndarray:
    import cv2

    rows = []
    width = None
    for row in panels:
        resized = []
        for cell in row:
            if cell is None:
                placeholder = np.zeros((cell_h, int(cell_h * 640 / 480), 3), dtype=np.uint8)
                resized.append(placeholder)
            else:
                scale = cell_h / cell.shape[0]
                resized.append(cv2.resize(cell, (int(cell.shape[1] * scale), cell_h)))
            width = resized[-1].shape[1]
        rows.append(np.concatenate(resized, axis=1))
    max_w = max(row.shape[1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
            row = np.concatenate([row, pad], axis=1)
        padded.append(row)
    return np.concatenate(padded, axis=0)


def write_suction_contact_sheets(
    report_dir: Path,
    dataset_path: Path,
    fps: int,
    events: list[dict[str, Any]],
    frame_lookup: dict[tuple[int, int], dict[str, Any]],
) -> list[str]:
    import cv2

    episode_meta = load_episode_video_meta(dataset_path)
    out_dir = report_dir / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=False)
    written: list[str] = []
    for event in events:
        if event["kind"] != "on":
            continue
        episode = int(event["episode_index"])
        target_frame = int(event["target_frame"])
        meta = episode_meta[episode]
        panels: list[list[np.ndarray | None]] = []
        for camera_key in CAMERA_KEYS:
            video_path = video_file_for(dataset_path, meta, camera_key)
            from_ts = float(meta[f"videos/{camera_key}/from_timestamp"])
            row: list[np.ndarray | None] = []
            for offset in CONTACT_OFFSETS:
                frame = target_frame + offset
                info = frame_lookup.get((episode, frame))
                if info is None:
                    row.append(None)
                    continue
                timestamp = from_ts + frame / fps
                image = read_video_frame_bgr(video_path, timestamp)
                tcp = info["tcp_xyz"]
                lines = [
                    f"{camera_key.split('.')[-1]} t{offset:+d}",
                    f"ep={episode} frame={frame}",
                    f"act.g={info['action_gripper']:.1f} obs.g={info['obs_gripper']:.1f}",
                    f"xyz=({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f})",
                ]
                row.append(_annotate_panel(image, lines))
            panels.append(row)
        sheet = render_contact_sheet(panels)
        name = f"on_ep{episode:02d}_frame{target_frame:04d}.jpg"
        dest = out_dir / name
        if not cv2.imwrite(str(dest), sheet):
            raise RuntimeError(f"写入 contact sheet 失败: {dest}")
        written.append(str(dest.relative_to(report_dir)))
    return written


def build_transition_row(
    episode_index: int,
    event: dict[str, Any],
    deployment: str,
    input_mode: str,
    predicted: np.ndarray,
    target: np.ndarray,
    observation_tcp: np.ndarray,
    action_names: list[str],
    predicted_cross_frame: int | None,
    initial_suction_on: bool,
) -> dict[str, Any]:
    z_index = action_names.index("ee.z")
    gripper_index = action_names.index("ee.gripper_pos")
    frame = event["frame"]
    timing = transition_timing(
        predicted[:, gripper_index],
        frame,
        event["kind"],
        predicted_cross_frame,
        initial_on=initial_suction_on,
    )
    return {
        "episode_index": episode_index,
        "kind": event["kind"],
        "ordinal": event["ordinal"],
        "deployment": deployment,
        "input_mode": input_mode,
        "target_frame": frame,
        "target_time_s": None,
        "predicted_cross_frame": timing["predicted_cross_frame"],
        "timing_error_frames": timing["timing_error_frames"],
        "pre_window_false_trigger": timing["pre_window_false_trigger"],
        "post_window_still_missed": timing["post_window_still_missed"],
        "target_gripper": float(target[frame, gripper_index]),
        "predicted_gripper": float(predicted[frame, gripper_index]),
        "target_z_m": float(target[frame, z_index]),
        "predicted_z_m": float(predicted[frame, z_index]),
        "observation_z_m": float(observation_tcp[frame, 2]),
    }


def rotation_and_state_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    observation_tcp: np.ndarray,
    observation_rot: np.ndarray,
    observation_j6_rad: np.ndarray,
    action_names: list[str],
) -> dict[str, Any]:
    xyz_idx = [action_names.index(name) for name in ("ee.x", "ee.y", "ee.z")]
    j6_idx = action_names.index("ee.j6_target")
    rot_idx = [action_names.index(name) for name in ("ee.wx", "ee.wy", "ee.wz")]
    return {
        "target_xyz_vs_observation_l2_m": distribution_report(
            np.linalg.norm(target[:, xyz_idx] - observation_tcp, axis=1), XYZ_THRESHOLDS_M
        ),
        "predicted_xyz_vs_observation_l2_m": distribution_report(
            np.linalg.norm(predicted[:, xyz_idx] - observation_tcp, axis=1), XYZ_THRESHOLDS_M
        ),
        "target_j6_vs_observation_abs_rad": distribution_report(
            np.abs(target[:, j6_idx] - observation_j6_rad), J6_THRESHOLDS_RAD
        ),
        "predicted_j6_vs_observation_abs_rad": distribution_report(
            np.abs(predicted[:, j6_idx] - observation_j6_rad), J6_THRESHOLDS_RAD
        ),
        "geodesic_predicted_vs_target_deg": distribution_report(
            geodesic_angle_deg(predicted[:, rot_idx], target[:, rot_idx]), ROTATION_THRESHOLDS_DEG
        ),
        "geodesic_target_vs_observation_deg": distribution_report(
            geodesic_angle_deg(target[:, rot_idx], observation_rot), ROTATION_THRESHOLDS_DEG
        ),
        "geodesic_predicted_vs_observation_deg": distribution_report(
            geodesic_angle_deg(predicted[:, rot_idx], observation_rot), ROTATION_THRESHOLDS_DEG
        ),
    }


def read_model_feature_shapes(model_path: Path) -> dict[str, Any]:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    state_shape = list(config["input_features"]["observation.state"]["shape"])
    action_shape = list(config["output_features"]["action"]["shape"])
    if state_shape not in ([12], [13]):
        raise ValueError(f"observation.state 期望 [12] 或 [13]，实际 {state_shape}")
    if action_shape != [8]:
        raise ValueError(f"action 期望 [8]，实际 {action_shape}")
    return {
        "observation.state": state_shape,
        "action": action_shape,
        "n_action_steps": config.get("n_action_steps"),
        "chunk_size": config.get("chunk_size"),
        "temporal_ensemble_coeff": config.get("temporal_ensemble_coeff"),
    }


def resolve_local_dataset_path(path_value: str | Path) -> tuple[str, Path]:
    """Resolve a local dataset explicitly so Hugging Face lookup is never attempted."""
    root = Path(path_value).expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"本地 LeRobot 数据集不存在或不完整: {root}")
    return root.name, root


def load_local_dataset_metadata(dataset_path: Path) -> LeRobotDatasetMetadata:
    repo_id, root = resolve_local_dataset_path(dataset_path)
    return LeRobotDatasetMetadata(repo_id=repo_id, root=root)


def load_local_dataset(
    dataset_path: Path,
    *,
    episodes: list[int],
    delta_timestamps: dict[str, list[float]],
) -> LeRobotDataset:
    repo_id, root = resolve_local_dataset_path(dataset_path)
    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
    )


def audit(args: argparse.Namespace) -> Path:
    _dataset_repo_id, dataset_path = resolve_local_dataset_path(args.dataset_path)
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"未找到模型目录: {model_path}")

    feature_shapes = read_model_feature_shapes(model_path)
    evidence = {
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "model_safetensors_sha256": sha256_file(model_path / "model.safetensors"),
        "dataset_info_sha256": sha256_file(dataset_path / "meta" / "info.json"),
        "dataset_stats_sha256": sha256_file(dataset_path / "meta" / "stats.json"),
        "feature_shapes": feature_shapes,
    }

    metadata = load_local_dataset_metadata(dataset_path)
    episodes = parse_episode_list(args.episodes, metadata.total_episodes)
    action_names = list(metadata.features[ACTION]["names"])
    required = {"ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos", "ee.j6_target"}
    missing = required.difference(action_names)
    if missing:
        raise ValueError(f"action features 缺少必要字段: {sorted(missing)}")
    state_names = list(metadata.features["observation.state"]["names"])
    if len(state_names) != 13:
        raise ValueError(f"observation.state 维数不是 13: {state_names}")
    state_gripper_index = state_names.index("gripper_pos") if "gripper_pos" in state_names else None
    tcp_idx = [state_names.index(name) for name in ("ee.x", "ee.y", "ee.z")]
    rot_idx = [state_names.index(name) for name in ("ee.wx", "ee.wy", "ee.wz")]
    j6_state_index = state_names.index("J6")

    contract_path = model_path / STATE_INPUT_CONTRACT_FILENAME
    if contract_path.is_file():
        state_contract = load_state_input_contract(model_path)
        evidence["state_input_contract_sha256"] = sha256_file(contract_path)
    elif feature_shapes["observation.state"] == [13]:
        state_contract = build_state_input_contract(metadata.features, "full")
        evidence["state_input_contract_sha256"] = None
        evidence["state_input_contract_compatibility"] = "legacy_13d_full"
    else:
        raise FileNotFoundError(f"12D checkpoint 缺少必要输入合同: {contract_path}")
    if state_contract.source_names != tuple(state_names):
        raise ValueError("checkpoint 输入合同的 source_names 与数据集 observation.state 不一致")
    if state_contract.model_width != feature_shapes["observation.state"][0]:
        raise ValueError("checkpoint 输入合同的 model_width 与模型 config 不一致")
    model_gripper_index = (
        state_contract.model_names.index("gripper_pos")
        if "gripper_pos" in state_contract.model_names
        else None
    )
    evidence["state_input_contract"] = state_contract.as_dict()

    device = select_device(args.device)
    policy = ACTPolicy.from_pretrained(model_path).to(device).eval()
    policy.config.device = str(device)
    policy.model.sample_latent_at_inference = args.latent_mode == LATENT_MODE_EPISODE_SAMPLE
    latent_cache: dict[int, torch.Tensor] = {}
    evidence["latent_inference"] = {
        "mode": args.latent_mode,
        "seed": args.latent_seed if args.latent_mode == LATENT_MODE_EPISODE_SAMPLE else None,
        "scope": "one_fixed_z_per_episode" if args.latent_mode == LATENT_MODE_EPISODE_SAMPLE else "zero",
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    delta_timestamps = {"action": [i / metadata.fps for i in policy.config.action_delta_indices]}
    dataset = load_local_dataset(
        dataset_path,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            latent_batch = None
            if args.latent_mode == LATENT_MODE_EPISODE_SAMPLE:
                latent_batch = episode_fixed_latent_batch(
                    batch["episode_index"],
                    policy.config.latent_dim,
                    seed=args.latent_seed,
                    device=device,
                    cache=latent_cache,
                )
                policy.model._infer_z = latent_batch
            processed = preprocessor(
                make_inference_observation(
                    batch,
                    policy.config.input_features,
                    state_contract=state_contract,
                )
            )
            action_chunks = postprocessor(policy.predict_action_chunk(processed)).cpu().numpy()
            forced_release_chunks = None
            if model_gripper_index is not None:
                if latent_batch is not None:
                    policy.model._infer_z = latent_batch
                forced_processed = preprocessor(
                    make_inference_observation(
                        batch,
                        policy.config.input_features,
                        state_gripper_index=model_gripper_index,
                        state_contract=state_contract,
                    )
                )
                forced_release_chunks = postprocessor(policy.predict_action_chunk(forced_processed)).cpu().numpy()
            targets = batch[ACTION][:, 0].cpu().numpy()
            states = batch["observation.state"].cpu().numpy()

            for item in range(len(targets)):
                records.append(
                    {
                        "episode_index": int(batch["episode_index"][item]),
                        "frame_index": int(batch["frame_index"][item]),
                        "dataset_index": int(batch["index"][item]),
                        "target": targets[item].astype(np.float32),
                        "state": states[item].astype(np.float32),
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
    transition_rows: list[dict[str, Any]] = []
    stored: dict[str, dict[str, Any]] = {}
    all_target_events: list[dict[str, Any]] = []
    frame_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    client_guard_rows: list[dict[str, Any]] = []
    gripper_index = action_names.index("ee.gripper_pos")

    for episode_index in episodes:
        group = sorted(
            (row for row in records if row["episode_index"] == episode_index),
            key=lambda row: row["frame_index"],
        )
        if not group:
            raise RuntimeError(f"episode {episode_index} 没有读取到任何帧")
        chunks = np.stack([row["chunk"] for row in group])
        target = np.stack([row["target"] for row in group])
        state = np.stack([row["state"] for row in group])
        observation_tcp = state[:, tcp_idx]
        observation_rot = state[:, rot_idx]
        observation_j6_rad = np.deg2rad(state[:, j6_state_index])
        initial_suction_on = bool(state[0, state_gripper_index] > SUCTION_ON_THRESHOLD)
        forced_chunks = None
        if group[0]["forced_release_chunk"] is not None:
            forced_chunks = np.stack([row["forced_release_chunk"] for row in group])
        queue_actions = deployed_queue_actions(chunks, QUEUE_N_ACTION_STEPS)
        ensemble_actions = deployed_temporal_ensemble_actions(
            chunks, TEMPORAL_ENSEMBLE_COEFF
        )
        latest_actions = chunks[:, 0, :]
        predicted_by_mode = {
            "normal": {
                "queue": queue_actions,
                "temporal_ensemble": ensemble_actions,
                "ensemble_queue_gripper": deployed_hybrid_actions(
                    ensemble_actions, queue_actions, gripper_index
                ),
                "ensemble_latest_gripper": deployed_hybrid_actions(
                    ensemble_actions, latest_actions, gripper_index
                ),
            }
        }
        episode_guard_rows = replay_current_client_guard(
            predicted_by_mode["normal"]["ensemble_latest_gripper"],
            chunks,
            state,
            action_names=action_names,
            state_names=state_names,
            fps=int(metadata.fps),
        )
        for row in episode_guard_rows:
            row["episode_index"] = episode_index
        client_guard_rows.extend(episode_guard_rows)
        if forced_chunks is not None:
            forced_queue = deployed_queue_actions(forced_chunks, QUEUE_N_ACTION_STEPS)
            forced_ensemble = deployed_temporal_ensemble_actions(
                forced_chunks, TEMPORAL_ENSEMBLE_COEFF
            )
            predicted_by_mode["forced_zero"] = {
                "queue": forced_queue,
                "temporal_ensemble": forced_ensemble,
                "ensemble_queue_gripper": deployed_hybrid_actions(
                    forced_ensemble, forced_queue, gripper_index
                ),
                "ensemble_latest_gripper": deployed_hybrid_actions(
                    forced_ensemble, forced_chunks[:, 0, :], gripper_index
                ),
            }

        target_on = apply_gripper_hysteresis(
            target[:, gripper_index], initial_on=initial_suction_on
        )
        target_events = find_threshold_transitions(
            target[:, gripper_index], initial_on=initial_suction_on
        )
        for event in target_events:
            all_target_events.append({"episode_index": episode_index, "target_frame": event["frame"], **event})

        for frame_pos, row in enumerate(group):
            frame_lookup[(episode_index, int(row["frame_index"]))] = {
                "action_gripper": float(target[frame_pos, gripper_index]),
                "obs_gripper": float(
                    state[frame_pos, state_gripper_index]
                    if state_gripper_index is not None
                    else 0.0
                ),
                "tcp_xyz": [float(v) for v in observation_tcp[frame_pos]],
            }

        for input_mode, deployed in predicted_by_mode.items():
            for deployment, predicted in deployed.items():
                predicted_on = apply_gripper_hysteresis(
                    predicted[:, gripper_index], initial_on=initial_suction_on
                )
                pred_events = find_threshold_transitions(
                    predicted[:, gripper_index], initial_on=initial_suction_on
                )
                event_matches, unmatched_predicted = match_transition_events(
                    target_events, pred_events
                )
                for event in target_events:
                    row = build_transition_row(
                        episode_index,
                        event,
                        deployment,
                        input_mode,
                        predicted,
                        target,
                        observation_tcp,
                        action_names,
                        event_matches.get((event["kind"], event["ordinal"])),
                        initial_suction_on,
                    )
                    row["target_time_s"] = float(event["frame"] / metadata.fps)
                    if row["predicted_cross_frame"] is not None:
                        row["predicted_cross_time_s"] = float(row["predicted_cross_frame"] / metadata.fps)
                    else:
                        row["predicted_cross_time_s"] = None
                    transition_rows.append(row)
                key = f"{deployment}:{input_mode}"
                slot = stored.setdefault(
                    key,
                    {
                        "predicted": [],
                        "target": [],
                        "tcp": [],
                        "rot": [],
                        "j6": [],
                        "predicted_on": [],
                        "target_on": [],
                        "continuity": {
                            "xyz_step_l2_m": [],
                            "j6_step_abs_rad": [],
                            "rotation_step_geodesic_deg": [],
                        },
                        "predicted_transition_count": {"on": 0, "off": 0},
                        "extra_predicted_transition_count": {"on": 0, "off": 0},
                    },
                )
                slot["predicted"].append(predicted)
                slot["target"].append(target)
                slot["tcp"].append(observation_tcp)
                slot["rot"].append(observation_rot)
                slot["j6"].append(observation_j6_rad)
                slot["predicted_on"].append(predicted_on)
                slot["target_on"].append(target_on)
                continuity = action_continuity_values(predicted, action_names)
                for metric_name, values in continuity.items():
                    slot["continuity"][metric_name].append(values)
                for kind in ("on", "off"):
                    slot["predicted_transition_count"][kind] += sum(
                        event["kind"] == kind for event in pred_events
                    )
                    slot["extra_predicted_transition_count"][kind] += unmatched_predicted[kind]

    comparison_out: dict[str, Any] = {}
    for key, slot in stored.items():
        deployment, input_mode = key.split(":", 1)
        predicted = np.concatenate(slot["predicted"])
        target = np.concatenate(slot["target"])
        comparison_out[key] = {
            "deployment": deployment,
            "input_mode": input_mode,
            "frames": int(len(target)),
            "suction": binary_metrics(
                predicted[:, gripper_index],
                target[:, gripper_index],
                predicted_on=np.concatenate(slot["predicted_on"]),
                target_on=np.concatenate(slot["target_on"]),
            ),
            "per_action_error": action_error_metrics(predicted, target, action_names),
            "command_vs_state": rotation_and_state_metrics(
                predicted,
                target,
                np.concatenate(slot["tcp"]),
                np.concatenate(slot["rot"]),
                np.concatenate(slot["j6"]),
                action_names,
            ),
            "action_continuity": {
                "xyz_step_l2_m": distribution_report(
                    np.concatenate(slot["continuity"]["xyz_step_l2_m"]),
                    XYZ_THRESHOLDS_M,
                ),
                "j6_step_abs_rad": distribution_report(
                    np.concatenate(slot["continuity"]["j6_step_abs_rad"]),
                    J6_THRESHOLDS_RAD,
                ),
                "rotation_step_geodesic_deg": distribution_report(
                    np.concatenate(slot["continuity"]["rotation_step_geodesic_deg"]),
                    ROTATION_THRESHOLDS_DEG,
                ),
            },
            "transitions": {
                "on": {
                    **summarize_transition_rows(
                    [
                        row
                        for row in transition_rows
                        if row["deployment"] == deployment and row["input_mode"] == input_mode and row["kind"] == "on"
                    ]
                    ),
                    "predicted_crossing_count": slot["predicted_transition_count"]["on"],
                    "extra_predicted_crossing_count": slot["extra_predicted_transition_count"]["on"],
                },
                "off": {
                    **summarize_transition_rows(
                    [
                        row
                        for row in transition_rows
                        if row["deployment"] == deployment and row["input_mode"] == input_mode and row["kind"] == "off"
                    ]
                    ),
                    "predicted_crossing_count": slot["predicted_transition_count"]["off"],
                    "extra_predicted_crossing_count": slot["extra_predicted_transition_count"]["off"],
                },
            },
        }

    contact_files = write_suction_contact_sheets(
        report_dir,
        dataset_path,
        int(metadata.fps),
        all_target_events,
        frame_lookup,
    )

    report = {
        "purpose": "Offline pure-ACT causal audit V2; no robot or camera was connected.",
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "device": str(device),
        "fps": int(metadata.fps),
        "episodes": episodes,
        "frames": int(sum(1 for _ in records)),
        "evidence": evidence,
        "deployments": {
            "queue": {
                "chunk_size": int(policy.config.chunk_size),
                "n_action_steps": QUEUE_N_ACTION_STEPS,
                "description": "Replan every n_action_steps and execute the first actions from each ACT chunk.",
            },
            "temporal_ensemble": {
                "chunk_size": int(policy.config.chunk_size),
                "n_action_steps": 1,
                "coeff": TEMPORAL_ENSEMBLE_COEFF,
                "implementation": "lerobot.policies.act.modeling_act.ACTTemporalEnsembler",
            },
            "ensemble_queue_gripper": {
                "motion": "temporal_ensemble",
                "gripper": "queue",
                "description": "Ensemble motion dimensions; preserve the 4-step queue gripper command.",
            },
            "ensemble_latest_gripper": {
                "motion": "temporal_ensemble",
                "gripper": "latest_chunk_step_0",
                "description": "Ensemble motion dimensions; use the newest chunk's current gripper command.",
            },
        },
        "target_suction_events": summarize_target_suction_events(all_target_events),
        "comparison": comparison_out,
        "current_client_guard_replay": {
            "selected_server_deployment": "ensemble_latest_gripper",
            "client_defaults": {
                "hold_locked_ee_pose": True,
                "use_motion_chunk_lookahead": CLIENT_USE_MOTION_CHUNK_LOOKAHEAD,
                "motion_chunk_min_z_drop_m": CLIENT_MOTION_CHUNK_MIN_Z_DROP_M,
                "use_gripper_chunk_lookahead": CLIENT_USE_GRIPPER_CHUNK_LOOKAHEAD,
                "gripper_chunk_close_score": CLIENT_GRIPPER_CHUNK_CLOSE_SCORE,
                "gripper_chunk_open_score": CLIENT_GRIPPER_CHUNK_OPEN_SCORE,
                "workspace_min_m": CLIENT_WORKSPACE_MIN_M,
                "workspace_max_m": CLIENT_WORKSPACE_MAX_M,
                "max_ee_step_m": CLIENT_MAX_EE_STEP_M,
                "max_j6_step_rad": CLIENT_MAX_J6_STEP_RAD,
                "j6_target_min_rad": CLIENT_J6_TARGET_MIN_RAD,
                "j6_target_max_rad": CLIENT_J6_TARGET_MAX_RAD,
                "max_observation_age_ms": CLIENT_MAX_OBSERVATION_AGE_MS,
                "max_action_chunk_age_ms": CLIENT_MAX_ACTION_CHUNK_AGE_MS,
            },
            **summarize_client_guard_replay(client_guard_rows),
        },
        "contact_sheets": contact_files,
        "transition_file": "transitions.csv",
    }
    (report_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report_dir / "transitions.csv", transition_rows)
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
    parser.add_argument(
        "--latent-mode",
        choices=LATENT_MODES,
        default=LATENT_MODE_ZERO,
        help="zero 使用 ACT 原始 z=0；episode_sample 为每个 episode 固定采样一个 z",
    )
    parser.add_argument("--latent-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = audit(args)
    print(f"离线审核完成，报告已写入: {report_dir}")


if __name__ == "__main__":
    main()

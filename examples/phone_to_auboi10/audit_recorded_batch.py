#!/usr/bin/env python
"""Read-only audit of one recorded AUBO teaching batch.

Never connects to cameras, the robot, suction IO, or a GPU.
Never writes into the dataset directory and never aggregates or trains.

``ee.gripper_pos`` / ``observation.state.gripper_pos`` are software command
latches, not vacuum pressure. This script can check command timing only; the
operator must confirm that each episode actually grasped, carried, and placed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

EXPECTED_FPS = 25
EXPECTED_STATE_DIM = 13
EXPECTED_ACTION_DIM = 8
EXPECTED_IMAGE_SHAPE = [480, 640, 3]
EXPECTED_VIDEO_SIZE = (640, 480)
REQUIRED_CAMERAS = (
    "observation.images.handeye",
    "observation.images.fixed",
)
EXPECTED_ACTION_NAMES = (
    "ee.j6_target",
    "ee.x",
    "ee.y",
    "ee.z",
    "ee.wx",
    "ee.wy",
    "ee.wz",
    "ee.gripper_pos",
)
EXPECTED_STATE_NAMES = (
    "J1",
    "J2",
    "J3",
    "J4",
    "J5",
    "J6",
    "ee.x",
    "ee.y",
    "ee.z",
    "ee.wx",
    "ee.wy",
    "ee.wz",
    "gripper_pos",
)
LIFT_RISE_M = 0.005
LIFT_REFERENCE_S = (1.8, 2.1)
LIFT_WARN_S = (1.2, 3.5)
MIN_SUCTION_PERIOD_S = 0.3
MAX_EE_STEP_M = 0.055
PHYSICAL_GRASP_DISCLAIMER = (
    "gripper_pos is a software command latch, not vacuum or grasp feedback. "
    "Offline audit cannot prove the strip was actually held; the operator must "
    "confirm real grasp, carry, and place success."
)


def _load_aggregate():
    script = Path(__file__).resolve().parent / "aggregate.py"
    spec = importlib.util.spec_from_file_location("aubo_aggregate_for_batch_audit", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def column_to_1d(column) -> np.ndarray:
    values = column.to_pylist()
    if not values:
        return np.empty((0,), dtype=np.float64)
    first = values[0]
    if isinstance(first, (list, tuple)):
        return np.asarray([item[0] if item else np.nan for item in values], dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def video_file_for(dataset_root: Path, record: dict[str, Any], camera_key: str) -> Path:
    chunk = int(record[f"videos/{camera_key}/chunk_index"])
    file_index = int(record[f"videos/{camera_key}/file_index"])
    return dataset_root / "videos" / camera_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"


def load_episode_records(dataset_root: Path) -> dict[int, dict[str, Any]]:
    files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 episode parquet: {dataset_root / 'meta' / 'episodes'}")
    rows: dict[int, dict[str, Any]] = {}
    for file in files:
        for record in pq.read_table(file).to_pylist():
            rows[int(record["episode_index"])] = record
    return rows


def load_manifest_batch(manifest_path: Path, dataset_name: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batches = manifest.get("batches", {})
    if dataset_name not in batches:
        raise KeyError(f"manifest 中没有批次 {dataset_name}")
    return batches[dataset_name]


def expected_gripper_switches(strip_count: int) -> int:
    if strip_count == 1:
        return 2
    if strip_count == 2:
        return 4
    raise ValueError(f"strip_count 只能是 1 或 2，实际 {strip_count}")


def gripper_transitions(gripper: np.ndarray) -> tuple[list[int], list[int]]:
    on = np.isclose(gripper, 100.0)
    delta = np.diff(on.astype(np.int8))
    ons = (np.where(delta == 1)[0] + 1).tolist()
    offs = (np.where(delta == -1)[0] + 1).tolist()
    return ons, offs


def suction_periods(ons: list[int], offs: list[int], n_frames: int) -> list[tuple[int, int]]:
    periods: list[tuple[int, int]] = []
    off_iter = list(offs)
    for on in ons:
        matching = [off for off in off_iter if off > on]
        if not matching:
            periods.append((on, n_frames))
            continue
        off = matching[0]
        off_iter.remove(off)
        periods.append((on, off))
    return periods


def lift_delay_s(z: np.ndarray, on_frame: int, fps: int, rise_m: float = LIFT_RISE_M) -> float | None:
    later = np.where(z[on_frame:] >= z[on_frame] + rise_m)[0]
    if len(later) == 0:
        return None
    return float(later[0] / fps)


def decode_video(path: Path) -> dict[str, Any]:
    import av

    container = av.open(str(path))
    try:
        if not container.streams.video:
            raise RuntimeError(f"{path} 没有视频流")
        stream = container.streams.video[0]
        frames = 0
        for _ in container.decode(video=0):
            frames += 1
        return {
            "path": str(path),
            "frames": frames,
            "fps": float(stream.average_rate or 0.0),
            "width": int(stream.width),
            "height": int(stream.height),
        }
    except Exception as exc:
        raise RuntimeError(f"{path} 解码失败: {exc}") from exc
    finally:
        container.close()


def load_episode_arrays(dataset_root: Path, aggregate) -> dict[int, dict[str, np.ndarray]]:
    episodes: dict[int, dict[str, list[np.ndarray]]] = {}
    columns = [
        "action",
        "observation.state",
        "episode_index",
        "frame_index",
        "timestamp",
        "index",
    ]
    for parquet_path in sorted((dataset_root / "data").rglob("*.parquet")):
        table = pq.read_table(parquet_path, columns=columns)
        actions = aggregate.action_array_to_numpy(table.column("action").combine_chunks())
        states = aggregate.action_array_to_numpy(table.column("observation.state").combine_chunks())
        episode_index = column_to_1d(table.column("episode_index")).astype(np.int64)
        frame_index = column_to_1d(table.column("frame_index")).astype(np.int64)
        timestamp = column_to_1d(table.column("timestamp"))
        dataset_index = column_to_1d(table.column("index")).astype(np.int64)
        for episode in np.unique(episode_index):
            mask = episode_index == episode
            slot = episodes.setdefault(
                int(episode),
                {"action": [], "state": [], "frame_index": [], "timestamp": [], "index": []},
            )
            slot["action"].append(actions[mask])
            slot["state"].append(states[mask])
            slot["frame_index"].append(frame_index[mask])
            slot["timestamp"].append(timestamp[mask])
            slot["index"].append(dataset_index[mask])

    out: dict[int, dict[str, np.ndarray]] = {}
    for episode, slot in episodes.items():
        order_parts = []
        for key in ("action", "state", "frame_index", "timestamp", "index"):
            values = np.concatenate(slot[key], axis=0)
            order_parts.append(values)
        frame_index = order_parts[2]
        order = np.argsort(frame_index, kind="stable")
        out[episode] = {
            "action": order_parts[0][order],
            "state": order_parts[1][order],
            "frame_index": order_parts[2][order],
            "timestamp": order_parts[3][order],
            "index": order_parts[4][order],
        }
    return out


def audit_features(info: dict[str, Any], errors: list[str]) -> None:
    if int(info["fps"]) != EXPECTED_FPS:
        errors.append(f"FPS 应为 {EXPECTED_FPS}，实际 {info['fps']}")
    action = info["features"].get("action", {})
    state = info["features"].get("observation.state", {})
    if list(action.get("shape", [])) != [EXPECTED_ACTION_DIM]:
        errors.append(f"action 维数应为 [{EXPECTED_ACTION_DIM}]，实际 {action.get('shape')}")
    if list(state.get("shape", [])) != [EXPECTED_STATE_DIM]:
        errors.append(f"observation.state 维数应为 [{EXPECTED_STATE_DIM}]，实际 {state.get('shape')}")
    if tuple(action.get("names", [])) != EXPECTED_ACTION_NAMES:
        errors.append(f"action names 已变化: {action.get('names')}")
    if tuple(state.get("names", [])) != EXPECTED_STATE_NAMES:
        errors.append(f"observation.state names 已变化: {state.get('names')}")
    for key in REQUIRED_CAMERAS:
        feature = info["features"].get(key)
        if feature is None:
            errors.append(f"缺少相机特征 {key}")
            continue
        if feature.get("dtype") != "video" or list(feature.get("shape", [])) != EXPECTED_IMAGE_SHAPE:
            errors.append(
                f"{key} 应为 video {EXPECTED_IMAGE_SHAPE}，实际为 {feature.get('dtype')} {feature.get('shape')}"
            )


def audit_indices(episode: int, arrays: dict[str, np.ndarray], fps: int, errors: list[str]) -> None:
    frame_index = arrays["frame_index"]
    timestamp = arrays["timestamp"]
    n = len(frame_index)
    if n == 0:
        errors.append(f"episode {episode}: 没有帧")
        return
    expected_frames = np.arange(n, dtype=np.int64)
    if not np.array_equal(frame_index, expected_frames):
        errors.append(f"episode {episode}: frame_index 不是 0..{n - 1} 连续编号")
    expected_ts = expected_frames / fps
    if not np.allclose(timestamp, expected_ts, atol=1.0 / fps + 1e-4):
        errors.append(f"episode {episode}: timestamp 与 frame_index/{fps} 不一致")
    if not np.isfinite(arrays["action"]).all():
        errors.append(f"episode {episode}: action 包含非有限值")
    if not np.isfinite(arrays["state"]).all():
        errors.append(f"episode {episode}: observation.state 包含非有限值")
    if not np.isfinite(timestamp).all():
        errors.append(f"episode {episode}: timestamp 包含非有限值")


def audit_gripper_and_motion(
    episode: int,
    arrays: dict[str, np.ndarray],
    *,
    action_names: list[str],
    state_names: list[str],
    fps: int,
    expected_switches: int,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    gripper = arrays["action"][:, action_names.index("ee.gripper_pos")]
    z = arrays["state"][:, state_names.index("ee.z")]
    xyz = arrays["action"][:, [action_names.index(name) for name in ("ee.x", "ee.y", "ee.z")]]
    unique = sorted({float(value) for value in np.unique(np.round(gripper, 6))})
    invalid = [value for value in unique if not (np.isclose(value, 0.0) or np.isclose(value, 100.0))]
    ons, offs = gripper_transitions(gripper)
    switch_count = len(ons) + len(offs)
    periods = suction_periods(ons, offs, len(gripper))
    lifts = [lift_delay_s(z, on, fps) for on in ons]
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(xyz) > 1 else np.array([], dtype=np.float64)
    max_step = float(steps.max()) if len(steps) else 0.0

    if invalid:
        errors.append(f"episode {episode}: 夹爪不是持续 0/100，出现 {invalid}")
    if switch_count != expected_switches:
        errors.append(
            f"episode {episode}: 夹爪切换 {switch_count} 次，期望 {expected_switches} "
            f"(on={ons}, off={offs})"
        )
    if expected_switches == 2 and (len(ons) != 1 or len(offs) != 1):
        errors.append(f"episode {episode}: 单根必须恰好 1 次开启和 1 次释放")
    if expected_switches == 4 and (len(ons) != 2 or len(offs) != 2):
        errors.append(f"episode {episode}: 双根必须恰好 2 次开启和 2 次释放")
    if not np.isclose(gripper[0], 0.0):
        errors.append(f"episode {episode}: 起始夹爪应为 0，实际 {float(gripper[0])}")
    if not np.isclose(gripper[-1], 0.0):
        errors.append(f"episode {episode}: 结束夹爪应为 0，实际 {float(gripper[-1])}")
    for start, end in periods:
        duration = (end - start) / fps
        if duration < MIN_SUCTION_PERIOD_S:
            errors.append(
                f"episode {episode}: 吸取段 [{start},{end}) 仅 {duration:.2f}s，疑似脉冲或抖动"
            )
    if len(ons) != len(offs):
        errors.append(f"episode {episode}: 开启/释放次数不相等 on={ons} off={offs}")
    for on, delay in zip(ons, lifts, strict=True):
        if delay is None:
            errors.append(f"episode {episode}: 吸盘开启帧 {on} 后末端未上升 {LIFT_RISE_M * 1000:.0f}mm")
        elif delay < LIFT_WARN_S[0] or delay > LIFT_WARN_S[1]:
            warnings.append(
                f"episode {episode}: 开启到上升 {LIFT_RISE_M * 1000:.0f}mm 用时 {delay:.2f}s，"
                f"参考 {LIFT_REFERENCE_S[0]:.1f}–{LIFT_REFERENCE_S[1]:.1f}s"
            )
    if max_step > MAX_EE_STEP_M:
        errors.append(f"episode {episode}: 最大末端目标跳变 {max_step:.4f} m")

    return {
        "episode_index": episode,
        "frames": int(len(gripper)),
        "gripper_unique": unique,
        "gripper_on_frames": ons,
        "gripper_off_frames": offs,
        "gripper_switch_count": switch_count,
        "suction_periods_s": [float((end - start) / fps) for start, end in periods],
        "lift_5mm_s": lifts,
        "max_ee_step_m": max_step,
    }


def audit_videos(
    dataset_root: Path,
    info: dict[str, Any],
    episode_records: dict[int, dict[str, Any]],
    episode_arrays: dict[int, dict[str, np.ndarray]],
    errors: list[str],
) -> list[dict[str, Any]]:
    fps = int(info["fps"])
    decoded_files: dict[str, dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []
    for episode in sorted(episode_arrays):
        record = episode_records.get(episode)
        if record is None:
            errors.append(f"episode {episode}: 缺少 meta/episodes 记录")
            continue
        length = int(record["length"])
        if length != len(episode_arrays[episode]["frame_index"]):
            errors.append(
                f"episode {episode}: meta length={length} 与 parquet 帧数 "
                f"{len(episode_arrays[episode]['frame_index'])} 不一致"
            )
        from_index = int(record["dataset_from_index"])
        to_index = int(record["dataset_to_index"])
        if to_index - from_index != length:
            errors.append(
                f"episode {episode}: dataset_from/to_index 与 length 不一致 "
                f"({from_index},{to_index},{length})"
            )
        for camera_key in REQUIRED_CAMERAS:
            path = video_file_for(dataset_root, record, camera_key)
            if not path.is_file():
                errors.append(f"episode {episode}: 缺少视频 {path}")
                continue
            cache_key = str(path)
            if cache_key not in decoded_files:
                try:
                    decoded_files[cache_key] = decode_video(path)
                except RuntimeError as exc:
                    errors.append(str(exc))
                    continue
            video = decoded_files[cache_key]
            if abs(video["fps"] - fps) > 0.01:
                errors.append(f"{path.name}: 视频 FPS={video['fps']}，期望 {fps}")
            if (video["width"], video["height"]) != EXPECTED_VIDEO_SIZE:
                errors.append(
                    f"{path.name}: 分辨率 {video['width']}x{video['height']}，"
                    f"期望 {EXPECTED_VIDEO_SIZE[0]}x{EXPECTED_VIDEO_SIZE[1]}"
                )
            from_ts = float(record[f"videos/{camera_key}/from_timestamp"])
            to_ts = float(record[f"videos/{camera_key}/to_timestamp"])
            duration = to_ts - from_ts
            expected = length / fps
            if abs(duration - expected) > (1.0 / fps) + 1e-3:
                errors.append(
                    f"episode {episode} {camera_key}: 视频时段 {duration:.4f}s 与 "
                    f"length/fps={expected:.4f}s 不一致"
                )
            reports.append(
                {
                    "episode_index": episode,
                    "camera": camera_key,
                    "path": cache_key,
                    "decoded_file_frames": video["frames"],
                    "from_timestamp": from_ts,
                    "to_timestamp": to_ts,
                }
            )

    for path, video in decoded_files.items():
        used_lengths = [
            int(episode_records[ep]["length"])
            for ep, record in episode_records.items()
            if any(video_file_for(dataset_root, record, cam) == Path(path) for cam in REQUIRED_CAMERAS)
        ]
        if used_lengths and video["frames"] != sum(used_lengths):
            errors.append(
                f"{path}: 解码 {video['frames']} 帧，对应 episode 合计 {sum(used_lengths)} 帧"
            )
    return reports


def audit_batch(
    dataset_path: str | Path,
    manifest_path: str | Path,
    *,
    decode_videos: bool = True,
) -> dict[str, Any]:
    dataset_root = Path(dataset_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"本地数据集不完整: {dataset_root}")

    aggregate = _load_aggregate()
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    batch = load_manifest_batch(manifest_file, dataset_root.name)
    planned = int(batch["planned_success_episodes"])
    strip_count = int(batch["strip_count"])
    expected_switches = expected_gripper_switches(strip_count)
    actual_episodes = int(info["total_episodes"])
    if actual_episodes != planned:
        errors.append(f"episode 数应为 {planned}，实际 {actual_episodes}")

    try:
        aggregate.validate_metadata(dataset_root)
    except RuntimeError as exc:
        errors.append(str(exc))
    audit_features(info, errors)

    episode_arrays = load_episode_arrays(dataset_root, aggregate)
    if sorted(episode_arrays) != list(range(actual_episodes)):
        errors.append(
            f"parquet episode_index 应为 0..{actual_episodes - 1}，实际 {sorted(episode_arrays)}"
        )
    total_frames = sum(len(item["frame_index"]) for item in episode_arrays.values())
    if total_frames != int(info["total_frames"]):
        errors.append(f"info.total_frames={info['total_frames']} 与 parquet 帧数 {total_frames} 不一致")

    action_names = list(info["features"]["action"]["names"])
    state_names = list(info["features"]["observation.state"]["names"])
    episode_reports = []
    for episode in sorted(episode_arrays):
        arrays = episode_arrays[episode]
        audit_indices(episode, arrays, int(info["fps"]), errors)
        episode_reports.append(
            audit_gripper_and_motion(
                episode,
                arrays,
                action_names=action_names,
                state_names=state_names,
                fps=int(info["fps"]),
                expected_switches=expected_switches,
                errors=errors,
                warnings=warnings,
            )
        )

    video_reports: list[dict[str, Any]] = []
    if decode_videos:
        episode_records = load_episode_records(dataset_root)
        video_reports = audit_videos(dataset_root, info, episode_records, episode_arrays, errors)

    report = {
        "purpose": "Read-only recorded-batch audit; no robot or camera was connected.",
        "dataset_path": str(dataset_root),
        "manifest_path": str(manifest_file),
        "batch_name": dataset_root.name,
        "strip_count": strip_count,
        "layout": batch.get("layout"),
        "grasp": batch.get("grasp"),
        "start_mode": batch.get("start_mode", "normal"),
        "planned_success_episodes": planned,
        "actual_episodes": actual_episodes,
        "actual_frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "local_train_episodes": batch.get("local_train_episodes"),
        "local_val_episodes": batch.get("local_val_episodes"),
        "validation_holdout": "last_local_episode_in_this_batch",
        "validation_limitation": (
            "This is a within-batch holdout, not an independent recording-session split."
        ),
        "physical_grasp": {
            "offline_proof": False,
            "disclaimer": PHYSICAL_GRASP_DISCLAIMER,
            "operator_confirmation_required": True,
        },
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "episodes": episode_reports,
        "videos": video_reports,
    }
    return report


def write_report(report: dict[str, Any], report_dir: str | Path | None) -> Path | None:
    if report_dir is None:
        return None
    path = Path(report_dir)
    if path.exists():
        raise FileExistsError(f"报告目录已存在，拒绝覆盖: {path}")
    path.mkdir(parents=True, exist_ok=False)
    (path / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读审核一批已录制示教，不连接硬件")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parent / "datasets" / "bamboo_act_report_manifest.json"),
    )
    parser.add_argument("--report-dir", help="新报告目录；已存在时拒绝覆盖")
    parser.add_argument("--skip-videos", action="store_true", help="单元测试用：跳过视频解码")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_batch(args.dataset_path, args.manifest, decode_videos=not args.skip_videos)
    report_dir = write_report(report, args.report_dir)
    print(json.dumps({k: report[k] for k in ("passed", "errors", "warnings", "actual_episodes", "actual_frames")}, ensure_ascii=False, indent=2))
    if report_dir is not None:
        print(f"报告已写入: {report_dir}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

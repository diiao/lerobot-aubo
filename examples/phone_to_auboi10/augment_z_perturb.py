#!/usr/bin/env python
"""方案 A：z 扰动增强。

问题：评估时 z 峰值中位数 0.254 (训练 0.186)。模型从未见过
"抓取位 + z>0.23" 的状态 (0% 训练样本), OOD 动作 -> 抓取不触发。

方案：对每条原 episode 生成 3 个扰动版本, 把符合条件帧的 state.z 抬高
+0.02/+0.03/+0.04 cm, action 不变。模型学到: "看到 z_state 高 -> 同样
z_target -> 相对下降幅度更大"。视频共享 (4 版本发同一段)。

条件: gripper<30 (OFF 阶段) AND state.x>0.4 AND state.y>-0.55 (近物料且未吸)。
"""

import glob
import json
import os
import shutil
from itertools import groupby

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC_REPO = "./datasets/bamboo_full_shift"
DST_REPO = "./datasets/bamboo_full_shift_aug"
Z_STATE_IDX = 8          # observation.state: [J1..J6, ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, gripper_pos]
GRIPPER_STATE_IDX = 12   # observation.state gripper_pos
DELTAS = [0.02, 0.03, 0.04]  # z_state 抬升量 (米)
ROWS_PER_FILE = 1000     # 与原数据集保持一致


def perturb_episode(rows, delta):
    """对一组 (action, state, ts, idx, tid) 帧应用 z 扰动。"""
    new_states = []
    n_perturbed = 0
    for (_act, state, _ts, _idx, _tid) in rows:
        s = list(state)
        if (s[GRIPPER_STATE_IDX] < 30 and s[6] > 0.4 and s[7] > -0.55):
            s[Z_STATE_IDX] = float(s[Z_STATE_IDX]) + delta
            n_perturbed += 1
        new_states.append(s)
    return new_states, n_perturbed


def main():
    from lerobot.utils.constants import HF_LEROBOT_HOME
    src = HF_LEROBOT_HOME / SRC_REPO
    dst = HF_LEROBOT_HOME / DST_REPO
    print(f"源:   {src}")
    print(f"目标: {dst}")
    print(f"扰动档位: {DELTAS}")

    # 1. 复制数据集
    if dst.exists():
        print(f"删除已有目标: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # 2. 读 data schema
    files = sorted(glob.glob(str(dst / "data" / "chunk-*" / "*.parquet")))
    data_sample = pq.read_table(files[0])
    print(f"data parquet 文件: {len(files)}")

    # 3. 收集所有原始帧
    rows = []
    for f in files:
        t = pq.read_table(f)
        eps = t["episode_index"].to_pylist()
        frs = t["frame_index"].to_pylist()
        acts = t["action"].to_pylist()
        sts = t["observation.state"].to_pylist()
        tss = t["timestamp"].to_pylist()
        idxs = t["index"].to_pylist()
        tids = t["task_index"].to_pylist()
        for ri in range(len(eps)):
            rows.append((eps[ri], frs[ri], acts[ri], sts[ri], tss[ri], idxs[ri], tids[ri]))
    rows.sort(key=lambda x: (x[0], x[1]))
    print(f"原始总帧数: {len(rows)}")

    # 4. 按 orig_ep 分组, 生成 4 版本 (原始 + 3 个 delta)
    new_rows = []
    perturb_counts = {d: 0 for d in DELTAS}
    new_ep_id = 0
    ep_src_map = {}
    ep_offsets = {}
    cum_frames = 0

    for orig_ep, grp in groupby(rows, key=lambda x: x[0]):
        grp_rows = [(r[2], r[3], r[4], r[5], r[6]) for r in grp]

        # 原始版本
        for new_frame, (act, state, ts, idx, tid) in enumerate(grp_rows):
            new_rows.append((new_ep_id, new_frame, act, list(state), ts, idx, tid))
        n = len(grp_rows)
        ep_offsets[new_ep_id] = (cum_frames, cum_frames + n)
        cum_frames += n
        ep_src_map[new_ep_id] = orig_ep
        new_ep_id += 1

        # 扰动版本
        for delta in DELTAS:
            perturbed_states, n_pert = perturb_episode(grp_rows, delta)
            perturb_counts[delta] += n_pert
            for new_frame, (act, _st_orig, ts, idx, tid) in enumerate(grp_rows):
                new_rows.append((new_ep_id, new_frame, act, perturbed_states[new_frame],
                                 ts, idx, tid))
            ep_offsets[new_ep_id] = (cum_frames, cum_frames + n)
            cum_frames += n
            ep_src_map[new_ep_id] = orig_ep
            new_ep_id += 1

    print(f"\n扰动后总帧数: {len(new_rows)}")
    print(f"新 episode 数: {new_ep_id} ({new_ep_id // 4} × 4 版本)")
    for d, c in perturb_counts.items():
        print(f"  delta=+{d:.2f}: 扰动了 {c} 帧")

    # 5. 重写 data parquet
    new_rows.sort(key=lambda x: (x[0], x[1]))
    print("\n重写 data parquet...")
    for f in files:
        os.remove(f)
    chunk_dir = dst / "data" / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    file_count = (len(new_rows) + ROWS_PER_FILE - 1) // ROWS_PER_FILE
    print(f"将写入 {file_count} 个 parquet 文件")
    cols_in_order = data_sample.column_names
    schema = data_sample.schema

    for fi in range(file_count):
        slice_ = new_rows[fi * ROWS_PER_FILE: (fi + 1) * ROWS_PER_FILE]
        if not slice_:
            continue
        d = {
            "action": pa.array([r[2] for r in slice_], type=schema.field("action").type),
            "observation.state": pa.array([r[3] for r in slice_],
                                          type=schema.field("observation.state").type),
            "timestamp": pa.array([r[4] for r in slice_], type=schema.field("timestamp").type),
            "frame_index": pa.array([r[1] for r in slice_], type=schema.field("frame_index").type),
            "episode_index": pa.array([r[0] for r in slice_],
                                       type=schema.field("episode_index").type),
            "index": pa.array([r[5] for r in slice_], type=schema.field("index").type),
            "task_index": pa.array([r[6] for r in slice_], type=schema.field("task_index").type),
        }
        table = pa.table({name: d[name] for name in cols_in_order}, schema=schema)
        out_path = chunk_dir / f"file-{fi:03d}.parquet"
        pq.write_table(table, out_path)
    print(f"  完成 ({file_count} 文件)")

    # 6. 重算 observation.state stats
    print("\n重算 observation.state stats...")
    all_states = []
    for f in sorted(glob.glob(str(chunk_dir / "*.parquet"))):
        t = pq.read_table(f)
        all_states.extend(t["observation.state"].to_pylist())
    S = np.array(all_states, dtype=np.float64)
    state_stats = {
        "min": S.min(axis=0).tolist(),
        "max": S.max(axis=0).tolist(),
        "mean": S.mean(axis=0).tolist(),
        "std": S.std(axis=0).tolist(),
        "count": [int(len(S))],
        "q01": np.quantile(S, 0.01, axis=0).tolist(),
        "q10": np.quantile(S, 0.10, axis=0).tolist(),
        "q50": np.quantile(S, 0.50, axis=0).tolist(),
        "q90": np.quantile(S, 0.90, axis=0).tolist(),
        "q99": np.quantile(S, 0.99, axis=0).tolist(),
    }
    stats_path = dst / "meta" / "stats.json"
    stats = json.load(open(stats_path))
    old_z_mean = stats["observation.state"]["mean"][Z_STATE_IDX]
    old_z_std = stats["observation.state"]["std"][Z_STATE_IDX]
    stats["observation.state"] = state_stats
    json.dump(stats, open(stats_path, "w"), indent=2)
    new_z_mean = state_stats["mean"][Z_STATE_IDX]
    new_z_std = state_stats["std"][Z_STATE_IDX]
    print(f"  state z: mean {old_z_mean:.4f} -> {new_z_mean:.4f}, std {old_z_std:.4f} -> {new_z_std:.4f}")

    # 7. 更新 info.json
    info_path = dst / "meta" / "info.json"
    info = json.load(open(info_path))
    info["total_episodes"] = new_ep_id
    info["total_frames"] = len(new_rows)
    json.dump(info, open(info_path, "w"), indent=2)
    print(f"\ninfo.json: total_episodes {info['total_episodes']}, total_frames {info['total_frames']}")

    # 8. 重写 episodes parquet
    print("\n重写 episodes parquet (扩展为 4x, 视频共享)...")
    src_ep_table = pq.read_table(src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    src_ep_rows = {row["episode_index"]: row for row in src_ep_table.to_pylist()}

    def file_idx_for_global_frame(global_f):
        return int(global_f // ROWS_PER_FILE)

    STATS_KEYS = [
        "action", "observation.state",
        "observation.images.handeye", "observation.images.fixed",
        "timestamp", "frame_index", "episode_index", "index", "task_index",
    ]
    STATS_SUBKEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]

    new_ep_rows = []
    for new_ep in range(new_ep_id):
        src_ep = ep_src_map[new_ep]
        from_idx, to_idx = ep_offsets[new_ep]
        src_row = src_ep_rows[src_ep]
        row = {
            "episode_index": new_ep,
            "tasks": src_row["tasks"],
            "length": to_idx - from_idx,
            "data/chunk_index": 0,
            "data/file_index": file_idx_for_global_frame(from_idx),
            "dataset_from_index": from_idx,
            "dataset_to_index": to_idx,
            "videos/observation.images.handeye/chunk_index":
                src_row["videos/observation.images.handeye/chunk_index"],
            "videos/observation.images.handeye/file_index":
                src_row["videos/observation.images.handeye/file_index"],
            "videos/observation.images.handeye/from_timestamp":
                src_row["videos/observation.images.handeye/from_timestamp"],
            "videos/observation.images.handeye/to_timestamp":
                src_row["videos/observation.images.handeye/to_timestamp"],
            "videos/observation.images.fixed/chunk_index":
                src_row["videos/observation.images.fixed/chunk_index"],
            "videos/observation.images.fixed/file_index":
                src_row["videos/observation.images.fixed/file_index"],
            "videos/observation.images.fixed/from_timestamp":
                src_row["videos/observation.images.fixed/from_timestamp"],
            "videos/observation.images.fixed/to_timestamp":
                src_row["videos/observation.images.fixed/to_timestamp"],
        }
        # 复用 src_ep 的所有 stats/* (state 整体 stats 已在 stats.json 中重算,
        # 这里只是 per-episode stats; 图像与 action 与 src_ep 完全相同)
        for sk in STATS_KEYS:
            for ss in STATS_SUBKEYS:
                key = f"stats/{sk}/{ss}"
                if key in src_row:
                    row[key] = src_row[key]
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        new_ep_rows.append(row)

    col_names = src_ep_table.column_names
    table = pa.table({col: [r[col] for r in new_ep_rows] for col in col_names})
    ep_dir = dst / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    for f in ep_dir.glob("*.parquet"):
        os.remove(f)
    pq.write_table(table, ep_dir / "file-000.parquet")
    print(f"  episodes parquet: {len(new_ep_rows)} episodes")

    print(f"\n✅ 完成: {dst}")


if __name__ == "__main__":
    main()
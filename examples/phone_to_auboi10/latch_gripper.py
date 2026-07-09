#!/usr/bin/env python
"""方案2：对现有数据集的夹爪(吸盘) action 做锁存重建，无需重录。

原始 AuboGripperVelocityToPosition 是三态：100=吸(A键)/0=放(B键)/50=保持。
松手记 50 导致抓取只有 3 帧脉冲，L1 回归被平均掉，策略永远不吸。
锁存：遇 100 持续吸、遇 0 持续放、50 保持上一状态。这样吸合段持续几十帧，L1 学得到。

输入: ./datasets/phone_auboi10_full
输出: ./datasets/phone_auboi10_full_latched  (复制 + 改 parquet action 列 + 重算 action stats)
观测(图像/state)不变，只动 action 的 gripper 维度(index 7)，故不用重算图像 stats。
"""

import glob
import json
import os
import shutil
from itertools import groupby

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.utils.constants import HF_LEROBOT_HOME

SRC_REPO = "./datasets/phone_auboi10_full"
DST_REPO = "./datasets/phone_auboi10_full_latched"
GRIPPER_IDX = 7  # action: [ee.j6_target, ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos]


def main():
    src = HF_LEROBOT_HOME / SRC_REPO
    dst = HF_LEROBOT_HOME / DST_REPO
    print(f"源: {src}")
    print(f"目标: {dst}")

    # 1. 复制数据集（视频/观测原样保留）
    if dst.exists():
        print(f"删除已有目标: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    files = sorted(glob.glob(str(dst / "data" / "chunk-*" / "*.parquet")))
    print(f"parquet 文件数: {len(files)}")

    # 2. 收集所有帧 (episode, frame, action, file_idx, row_idx)，按 episode+frame 排序
    rows = []
    for fi, f in enumerate(files):
        t = pq.read_table(f)
        eps = t["episode_index"].to_pylist()
        frs = t["frame_index"].to_pylist()
        acts = t["action"].to_pylist()
        for ri in range(len(eps)):
            rows.append((eps[ri], frs[ri], acts[ri], fi, ri))
    rows.sort(key=lambda x: (x[0], x[1]))
    print(f"总帧数: {len(rows)}")

    # 3. 按 episode 锁存夹爪
    updates = {}  # (fi, ri) -> new_gripper
    sample_before, sample_after = None, None
    for ep, grp in groupby(rows, key=lambda x: x[0]):
        state = 0.0  # 起始吸盘 off
        grp_rows = list(grp)
        ep_grippers = []
        for (e, fr, act, fi, ri) in grp_rows:
            v = float(act[GRIPPER_IDX])
            if v >= 60:
                state = 100.0
            elif v <= 20:
                state = 0.0
            # else 保持
            updates[(fi, ri)] = state
            ep_grippers.append((v, state))
        # 记录第一条 episode 的锁存前后对比
        if ep == 0:
            sample_before = [g[0] for g in ep_grippers[:40]]
            sample_after = [g[1] for g in ep_grippers[:40]]

    print(f"\nep0 前 40 帧 gripper (锁存前): {[int(x) for x in sample_before]}")
    print(f"ep0 前 40 帧 gripper (锁存后): {[int(x) for x in sample_after]}")

    # 4. 写回每个 parquet（保留原 schema）
    for fi, f in enumerate(files):
        t = pq.read_table(f)
        acts = t["action"].to_pylist()
        new_acts = []
        for ri in range(len(acts)):
            a = list(acts[ri])
            a[GRIPPER_IDX] = updates[(fi, ri)]
            new_acts.append(a)
        new_col = pa.array(new_acts, type=t["action"].type)
        t = t.set_column(t.schema.get_field_index("action"), "action", new_col)
        pq.write_table(t, f)
    print("parquet 已写回")

    # 5. 重算 action stats（只 action；观测不变保留原 stats）
    all_acts = []
    for f in files:
        t = pq.read_table(f)
        for a in t["action"].to_pylist():
            all_acts.append(a)
    A = np.array(all_acts, dtype=np.float64)  # (N, 8)
    action_stats = {
        "min": A.min(axis=0).tolist(),
        "max": A.max(axis=0).tolist(),
        "mean": A.mean(axis=0).tolist(),
        "std": A.std(axis=0).tolist(),
        "count": [int(len(A))],
        "q01": np.quantile(A, 0.01, axis=0).tolist(),
        "q10": np.quantile(A, 0.10, axis=0).tolist(),
        "q50": np.quantile(A, 0.50, axis=0).tolist(),
        "q90": np.quantile(A, 0.90, axis=0).tolist(),
        "q99": np.quantile(A, 0.99, axis=0).tolist(),
    }
    stats_path = dst / "meta" / "stats.json"
    stats = json.load(open(stats_path))
    old_g = stats["action"]["mean"][GRIPPER_IDX]
    stats["action"] = action_stats
    json.dump(stats, open(stats_path, "w"), indent=2)
    print(f"action stats 已重算（gripper mean {old_g:.1f} -> {action_stats['mean'][GRIPPER_IDX]:.1f}, "
          f"std {stats.get('action',{}).get('std',[None]*8)[GRIPPER_IDX]} -> {action_stats['std'][GRIPPER_IDX]:.1f}）")
    print(f"\n完成: {dst}")


if __name__ == "__main__":
    main()

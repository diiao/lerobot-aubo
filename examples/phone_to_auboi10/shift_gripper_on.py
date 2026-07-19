#!/usr/bin/env python
"""方案2进阶：把锁存数据的吸合(ON)段整体前移，让模型学到"伸手末段就开始吸"。

问题：锁存数据里吸合从 frame 532（到苹果位）开始，但模型学到 frame 560 才预测 ON
（晚了~1秒，绑到搬运阶段）。导致评估时机器人到苹果位模型说"还不吸"，鸡生蛋。

修复：把每条 episode 的吸合起始点提前 SHIFT 帧（默认 60=2秒），让吸合从伸手末段
就开始。模型会学到"伸手接近苹果时吸合已开"，评估时伸手过程中吸盘就激活，到苹果位
时已经吸住，打破鸡生蛋。

输入: ./datasets/phone_auboi10_full_latched (锁存后)
输出: ./datasets/phone_auboi10_full_shift  (前移后)
观测不变，只动 action 的 gripper 维度，重算 action stats。
"""

import glob
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.utils.constants import HF_LEROBOT_HOME

SRC_REPO = "./datasets/bamboo_full_latched"
DST_REPO = "./datasets/bamboo_full_shift"
GRIPPER_IDX = 7
SHIFT = 60  # 吸合前移帧数（2秒@30fps）


def main():
    src = HF_LEROBOT_HOME / SRC_REPO
    dst = HF_LEROBOT_HOME / DST_REPO
    print(f"源: {src}\n目标: {dst}\nSHIFT={SHIFT}帧")

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    files = sorted(glob.glob(str(dst / "data" / "chunk-*" / "*.parquet")))

    # 收集所有帧按 episode+frame 排序
    rows = []
    for fi, f in enumerate(files):
        t = pq.read_table(f)
        eps = t["episode_index"].to_pylist()
        frs = t["frame_index"].to_pylist()
        acts = t["action"].to_pylist()
        for ri in range(len(eps)):
            rows.append((eps[ri], frs[ri], acts[ri], fi, ri))
    rows.sort(key=lambda x: (x[0], x[1]))

    # 按 episode 找首个 ON，前移 SHIFT 帧
    from itertools import groupby
    updates = {}  # (fi,ri) -> 100.0 (改为ON)
    shifted_count = 0
    for ep, grp in groupby(rows, key=lambda x: x[0]):
        grp_rows = list(grp)
        # 找首个 ON (gripper>60)
        on_start = None
        for i, (e, fr, act, fi, ri) in enumerate(grp_rows):
            if float(act[GRIPPER_IDX]) > 60:
                on_start = i
                break
        if on_start is None:
            continue  # 本 episode 无吸合
        new_on = max(0, on_start - SHIFT)
        # 把 [new_on, on_start) 的帧改成 ON
        for i in range(new_on, on_start):
            _, _, _, fi, ri = grp_rows[i]
            updates[(fi, ri)] = 100.0
        shifted_count += on_start - new_on

    print(f"共前移 {shifted_count} 帧为 ON")

    # 样例：ep10 前后对比
    for ep, grp in groupby(rows, key=lambda x: x[0]):
        if ep == 10:
            g = list(grp)
            on_idx = next((i for i, r in enumerate(g) if float(r[2][GRIPPER_IDX]) > 60), None)
            print(f"ep10: 原ON起始 frame_idx={on_idx}, 前移后起始={max(0,on_idx-SHIFT)}")
            break

    # 写回 parquet
    for fi, f in enumerate(files):
        t = pq.read_table(f)
        acts = t["action"].to_pylist()
        new_acts = []
        for ri in range(len(acts)):
            a = list(acts[ri])
            if (fi, ri) in updates:
                a[GRIPPER_IDX] = 100.0
            new_acts.append(a)
        new_col = pa.array(new_acts, type=t["action"].type)
        t = t.set_column(t.schema.get_field_index("action"), "action", new_col)
        pq.write_table(t, f)

    # 重算 action stats
    all_acts = []
    for f in files:
        for a in pq.read_table(f)["action"].to_pylist():
            all_acts.append(a)
    A = np.array(all_acts, dtype=np.float64)
    stats = json.load(open(dst / "meta" / "stats.json"))
    stats["action"] = {
        "min": A.min(axis=0).tolist(), "max": A.max(axis=0).tolist(),
        "mean": A.mean(axis=0).tolist(), "std": A.std(axis=0).tolist(),
        "count": [int(len(A))],
        "q01": np.quantile(A, 0.01, axis=0).tolist(), "q10": np.quantile(A, 0.10, axis=0).tolist(),
        "q50": np.quantile(A, 0.50, axis=0).tolist(), "q90": np.quantile(A, 0.90, axis=0).tolist(),
        "q99": np.quantile(A, 0.99, axis=0).tolist(),
    }
    json.dump(stats, open(dst / "meta" / "stats.json", "w"), indent=2)
    print(f"action stats 重算: gripper mean={stats['action']['mean'][GRIPPER_IDX]:.1f} std={stats['action']['std'][GRIPPER_IDX]:.1f}")
    print(f"完成: {dst}")


if __name__ == "__main__":
    main()

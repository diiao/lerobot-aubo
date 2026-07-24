#!/usr/bin/env python
"""验证启发式夹爪状态机的阈值是否正确。

把 inference_server._heuristic_gripper 的状态机逻辑原样跑在训练集每条 ep 的 state
序列上, 对比启发式 fire(0->100)/release(100->0) 帧索引 vs 训练真实 gripper 转换帧索引。
应分别在 ±20 帧 (~0.7s @30fps) 内匹配。

无真机、无模型, 纯 state 序列 + 阈值判断。
"""
import glob

import numpy as np
import pyarrow.parquet as pq

# 与 inference_server.py 完全一致的阈值
FIRE_XY = (0.592, -0.407)
FIRE_XY_R = 0.07
FIRE_Z_MAX = 0.125
RELEASE_X_MAX = 0.30
RELEASE_Y_MAX = -0.50
RELEASE_Z_MAX = 0.17

HF_HOME = "/home/rentao/.cache/huggingface/lerobot/datasets"


def heuristic_gripper(state, holding):
    """复刻 InferenceServer._heuristic_gripper, 返回 (gripper, holding_new)。"""
    x, y, z = float(state[6]), float(state[7]), float(state[8])
    in_fire = (z < FIRE_Z_MAX and abs(x - FIRE_XY[0]) < FIRE_XY_R and abs(y - FIRE_XY[1]) < FIRE_XY_R)
    in_release = (z < RELEASE_Z_MAX and x < RELEASE_X_MAX and y < RELEASE_Y_MAX)
    if not holding and in_fire:
        return 100.0, True
    if holding and in_release:
        return 0.0, False
    if holding:
        return 100.0, True
    return 0.0, False


def main():
    fs = sorted(glob.glob(f"{HF_HOME}/bamboo_full_shift/data/chunk-*/*.parquet"))
    states, eps = [], []
    for f in fs:
        t = pq.read_table(f)
        states.extend(t["observation.state"].to_pylist())
        eps.extend(t["episode_index"].to_pylist())
    states = np.array(states)
    eps = np.array(eps)
    grip = states[:, 12]

    print(f"训练 {len(np.unique(eps))} 条 ep, {len(states)} 帧")
    print(f"阈值: FIRE xy{FIRE_XY} r{FIRE_XY_R} z<{FIRE_Z_MAX} | "
          f"RELEASE x<{RELEASE_X_MAX} y<{RELEASE_Y_MAX} z<{RELEASE_Z_MAX}")
    print("\n  ep  真实fire  启发fire  Δfire   真实rel  启发rel  Δrel   末态grip")
    print("  " + "-" * 78)
    fire_errs, rel_errs = [], []
    n_match = n_rel_match = 0
    for ep in np.unique(eps):
        m = eps == ep
        g = grip[m]
        ep_states = states[m]
        # 真实转换帧
        gt_fire = next((i for i in range(1, len(g)) if g[i - 1] < 50 and g[i] >= 50), None)
        gt_rel = next((i for i in range(1, len(g)) if g[i - 1] >= 50 and g[i] < 50), None)
        # 启发式跑一遍
        holding = False
        h_fire = h_rel = None
        prev = 0.0
        for i in range(len(ep_states)):
            gv, holding = heuristic_gripper(ep_states[i], holding)
            if h_fire is None and prev < 50 and gv >= 50:
                h_fire = i
            if h_fire is not None and h_rel is None and prev >= 50 and gv < 50:
                h_rel = i
            prev = gv
        df = (h_fire - gt_fire) if (h_fire and gt_fire) is not None else "-"
        dr = (h_rel - gt_rel) if (h_rel and gt_rel) is not None else "-"
        if isinstance(df, int):
            fire_errs.append(df); n_match += 1
        if isinstance(dr, int):
            rel_errs.append(dr); n_rel_match += 1
        print(f"  {ep:>3}  {str(gt_fire):>8}  {str(h_fire):>8}  {str(df):>6}  "
              f"{str(gt_rel):>8}  {str(h_rel):>8}  {str(dr):>6}  {prev:>7.0f}")

    print("\n汇总:")
    if fire_errs:
        print(f"  fire 帧误差: 匹配 {n_match}/{len(np.unique(eps))}, "
              f"Δ均值 {np.mean(fire_errs):+.1f}帧, |Δ|最大 {np.max(np.abs(fire_errs))}帧")
    if rel_errs:
        print(f"  rel  帧误差: 匹配 {n_rel_match}/{len(np.unique(eps))}, "
              f"Δ均值 {np.mean(rel_errs):+.1f}帧, |Δ|最大 {np.max(np.abs(rel_errs))}帧")
    print("\n判读: |Δ|<=20 帧 (~0.7s) 为可接受; 末态grip 应为 0 (放完)")


if __name__ == "__main__":
    main()

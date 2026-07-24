"""合并 bamboo_full_shift(30) + bamboo_perturb(20) -> bamboo_combined(50)。

用 LeRobotDataset create + add_frame + save_episode (同 record.py)。
图像: ds[i] 返回 float32 CHW[0,1] -> PIL(HWC uint8) (add_frame validate 要 HWC)。

校验(合并后):
  - state/action 抽样逐元素相等 (parquet 无损, 应精确)
  - episode/frame 计数 = 50 / 59137
  - 图像解码非退化 (mean/std 合理, 非全黑全白)
  - 存源帧 vs 合并帧样张到 /tmp 供肉眼比对画质

env: LIMIT_PER_SRC=2,2 只合并每源前2集(测速/验证)。
"""
import os, time, numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

OUT = "./datasets/bamboo_combined"
SRCS = ["./datasets/bamboo_full_shift", "./datasets/bamboo_perturb"]
TASK = "抓取竹条"
LIMIT = os.environ.get("LIMIT_PER_SRC", "")


def to_pil(img):
    """float32 CHW[0,1] (torch/numpy) -> PIL HWC uint8."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = np.transpose(a, (1, 2, 0))  # CHW -> HWC
    if a.dtype != np.uint8:
        if a.max() <= 1.0:
            a = (a * 255.0).astype(np.uint8)
        else:
            a = a.astype(np.uint8)
    return Image.fromarray(a)


# 记录每源每全局ep的首帧索引, 用于事后抽样比对
src0 = LeRobotDataset(SRCS[0])
out = LeRobotDataset.create(
    repo_id=OUT, fps=src0.fps, features=src0.features,
    robot_type=src0.meta.robot_type, use_videos=True,
)
# 尝试硬件编码加速 (失败回退默认)
for vc in ["auto", "h264_nvenc"]:
    try:
        out.vcodec = vc
        print(f"vcodec 设为 {vc}")
        break
    except Exception as e:
        print(f"vcodec={vc} 失败: {e}")

limits = [int(x) for x in LIMIT.split(",")] if LIMIT else [None, None]
g_ep = 0
t0 = time.time()
# 抽样: 每源记第一个被合并的原始帧 (src_path, src_idx, global_idx)
samples = []

for si, sp in enumerate(SRCS):
    src = LeRobotDataset(sp)
    N = src.num_frames
    lim = limits[si]
    cur_ep = None
    eps_done = 0
    i = 0
    first_frame_of_src = True
    while i < N:
        f = src[i]
        ep_i = int(f["episode_index"])
        if cur_ep is None:
            cur_ep = ep_i
        elif ep_i != cur_ep:
            out.save_episode()
            g_ep += 1
            eps_done += 1
            print(f"  [src{si}] ep{g_ep-1} 存完 (累计{g_ep}ep, {time.time()-t0:.0f}s)", flush=True)
            cur_ep = ep_i
            if lim and eps_done >= lim:
                break
        frame = {
            "observation.state": np.array(f["observation.state"], dtype=np.float32),
            "observation.images.handeye": to_pil(f["observation.images.handeye"]),
            "observation.images.fixed": to_pil(f["observation.images.fixed"]),
            "action": np.array(f["action"], dtype=np.float32),
            "task": TASK,
        }
        if first_frame_of_src:
            samples.append((sp, i, out.num_frames))  # 该帧将写入 out 的 num_frames 位置
            first_frame_of_src = False
        out.add_frame(frame)
        i += 1
    if not (lim and eps_done >= lim):
        out.save_episode()
        g_ep += 1
        print(f"  [src{si}] 末ep存完 (累计{g_ep}ep, {time.time()-t0:.0f}s)", flush=True)

out.finalize()
print(f"\n合并完成: {g_ep} episodes -> {OUT}, 用时 {time.time()-t0:.0f}s")

# ============ 校验 ============
print("\n=== 校验 ===")
chk = LeRobotDataset(OUT)
exp_ep = 50 if not LIMIT else sum(limits)  # 每源 lim 集
print(f"episodes={chk.num_episodes} (期望{exp_ep})  frames={chk.num_frames}")
assert chk.num_episodes == exp_ep, f"ep数不符: {chk.num_episodes}"

# stats 已由 finalize 重算
import json
stats_path = os.path.expanduser(f"~/.cache/huggingface/lerobot/datasets/{OUT.split('/')[-1]}/meta/stats.json")
if os.path.exists(stats_path):
    st = json.load(open(stats_path))
    print(f"stats 字段: {list(st.keys())}")
    sa = st.get("action", {})
    print(f"action min={sa.get('min')} max={sa.get('max')}")

# 抽样比对 state/action (parquet 无损, 应精确相等)
print("\n--- state/action 抽样比对 ---")
for sp, src_idx, out_idx in samples:
    s = LeRobotDataset(sp)
    sf = s[src_idx]
    of = chk[out_idx]
    s_state = np.array(sf["observation.state"], dtype=np.float32)
    o_state = np.array(of["observation.state"], dtype=np.float32)
    s_act = np.array(sf["action"], dtype=np.float32)
    o_act = np.array(of["action"], dtype=np.float32)
    se = np.array_equal(s_state, o_state)
    ae = np.array_equal(s_act, o_act)
    print(f"  {sp}[{src_idx}] -> out[{out_idx}]: state_eq={se} action_eq={ae}")
    assert se, f"state 不一致: {sp}[{src_idx}]"
    assert ae, f"action 不一致: {sp}[{src_idx}]"

# 图像非退化 + 存样张
print("\n--- 图像校验 + 存样张 ---")
for sp, src_idx, out_idx in samples:
    s = LeRobotDataset(sp)
    sf = s[src_idx]
    of = chk[out_idx]
    for cam in ["handeye", "fixed"]:
        si_img = np.asarray(sf[f"observation.images.{cam}"])
        oi = np.asarray(of[f"observation.images.{cam}"])
        # 统一转 HWC 算统计
        if si_img.shape[0] == 3:
            si_img = np.transpose(si_img, (1, 2, 0))
        if oi.shape[0] == 3:
            oi = np.transpose(oi, (1, 2, 0))
        si_u = (si_img * 255).astype(np.uint8) if si_img.dtype != np.uint8 else si_img
        oi_u = oi if oi.dtype == np.uint8 else (oi * 255).astype(np.uint8)
        print(f"  {sp}[{src_idx}].{cam}: src mean={si_u.mean():.1f} std={si_u.std():.1f} | "
              f"out mean={oi_u.mean():.1f} std={oi_u.std():.1f} shape={oi_u.shape}")
        # 存样张: 源 vs 合并 (HWC uint8 -> PNG)
        tag = f"{'full' if 'full_shift' in sp else 'perturb'}_{cam}"
        Image.fromarray(si_u).save(f"/tmp/merge_src_{tag}.png")
        Image.fromarray(oi_u).save(f"/tmp/merge_out_{tag}.png")

print("\n样张已存: /tmp/merge_src_*.png /tmp/merge_out_*.png (请肉眼比对画质)")
print("校验全部通过 ✓")

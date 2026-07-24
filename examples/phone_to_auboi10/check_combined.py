"""全面校验 bamboo_combined 是否训练可用 (快速版: episode_index 走 parquet, 不解码视频)。
重点:
  1. episode_index 连续 0..49, 每集帧数与源逐集匹配
  2. 边界处图像-action 对齐 (last src0 frame @36621, first src1 frame @36622)
  3. 跨数据集抽样: state/action 与源精确一致, 图像解码非退化 (只解码6帧)
  4. 50 个视频文件全部 ffprobe 可解
  5. stats.json 覆盖 action/state 全维
  6. info.json features 与训练期望一致 (action 8, state 13, 2 cam)
"""
import os, sys, json, collections, subprocess
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

COMB = "./datasets/bamboo_combined"
SRC0 = "./datasets/bamboo_full_shift"
SRC1 = "./datasets/bamboo_perturb"
OK = True
def fail(msg):
    global OK; OK = False; print(f"  FAIL {msg}", flush=True)
def ok(msg):
    print(f"  OK  {msg}", flush=True)

print("loading datasets...", flush=True)
comb = LeRobotDataset(COMB)
s0 = LeRobotDataset(SRC0)
s1 = LeRobotDataset(SRC1)
print(f"combined: {comb.num_episodes}ep / {comb.num_frames}f | src0: {s0.num_episodes}ep/{s0.num_frames}f | src1: {s1.num_episodes}ep/{s1.num_frames}f", flush=True)

# 用 parquet 直接读 episode_index (不解码视频)
def ep_counts_from_parquet(name):
    import pyarrow.parquet as pq, glob
    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/datasets/{name}/data")
    files = sorted(glob.glob(f"{root}/chunk-*/*.parquet"))
    cnt = collections.Counter()
    for fp in files:
        t = pq.read_table(fp, columns=["episode_index"])
        for v in t.column("episode_index").to_pylist():
            cnt[int(v)] += 1
    return cnt

print("\n[1] episode_index 连续性 + 每集帧数 vs 源", flush=True)
comb_cnt = ep_counts_from_parquet("bamboo_combined")
s0_cnt = ep_counts_from_parquet("bamboo_full_shift")
s1_cnt = ep_counts_from_parquet("bamboo_perturb")
if sorted(comb_cnt.keys()) == list(range(50)):
    ok(f"episode_index 连续 0..49")
else:
    miss = sorted(set(range(50)) - set(comb_cnt.keys()))
    extra = sorted(set(comb_cnt.keys()) - set(range(50)))
    fail(f"episode_index 不连续, 缺{miss} 多{extra}")
# 源每集帧数顺序: src0 ep0..29 -> comb ep0..29; src1 ep0..19 -> comb ep30..49
src_lens = [s0_cnt[k] for k in range(s0.num_episodes)] + [s1_cnt[k] for k in range(s1.num_episodes)]
comb_lens = [comb_cnt[k] for k in range(50)]
mismatch = [(i, src_lens[i], comb_lens[i]) for i in range(50) if src_lens[i] != comb_lens[i]]
if not mismatch:
    ok(f"50 集帧数全部与源匹配 (src0:{sum(s0_cnt.values())} src1:{sum(s1_cnt.values())} comb:{sum(comb_cnt.values())})")
else:
    fail(f"帧数不匹配 (ep, src, comb): {mismatch[:5]}")

# 2 & 3. 跨数据集抽样比对 (只解码6帧)
print("\n[2] 边界 + 抽样: state/action 精确一致, 图像非退化", flush=True)
def img_stat(img):
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] == 3:
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255).astype(np.uint8)
    return float(a.mean()), float(a.std()), a.shape
samples = [
    (s0, 0, 0, "src0 首"),
    (s0, s0.num_frames - 1, s0.num_frames - 1, "src0 末(边界前)"),
    (s1, 0, s0.num_frames, "src1 首(边界后)"),
    (s1, s1.num_frames - 1, comb.num_frames - 1, "src1 末(comb末)"),
    (s0, s0.num_frames // 2, s0.num_frames // 2, "src0 中"),
    (s1, s1.num_frames // 2, s0.num_frames + s1.num_frames // 2, "src1 中"),
]
for sd, si, ci, desc in samples:
    sf, cf = sd[si], comb[ci]
    se = np.array_equal(np.array(sf["observation.state"], dtype=np.float32),
                        np.array(cf["observation.state"], dtype=np.float32))
    ae = np.array_equal(np.array(sf["action"], dtype=np.float32),
                        np.array(cf["action"], dtype=np.float32))
    bad_img = False
    for cam in ["handeye", "fixed"]:
        sm, ss, shp = img_stat(sf[f"observation.images.{cam}"])
        cm, cs, chp = img_stat(cf[f"observation.images.{cam}"])
        if abs(sm - cm) > 10 or cs < 5 or cm < 5 or cm > 250:
            bad_img = True
            fail(f"{desc} {cam}: src m{sm:.0f}/s{ss:.0f} comb m{cm:.0f}/s{cs:.0f} 退化?")
    if se and ae and not bad_img:
        ok(f"{desc} [{si}->{ci}]: state/action 精确一致, 图像OK")
    else:
        if not se: fail(f"{desc} state 不一致")
        if not ae: fail(f"{desc} action 不一致")

# 4. 视频文件 ffprobe 可解
print("\n[3] 视频文件可解 (ffprobe)", flush=True)
vroot = os.path.expanduser("~/.cache/huggingface/lerobot/datasets/bamboo_combined/videos")
bad_videos = []
total_vids = 0
for cam in ["handeye", "fixed"]:
    d = os.path.join(vroot, f"observation.images.{cam}", "chunk-000")
    if not os.path.isdir(d):
        fail(f"视频目录不存在: {d}"); continue
    files = sorted(os.listdir(d))
    total_vids += len(files)
    for fn in files:
        p = os.path.join(d, fn)
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_frames",
            "-of", "default=nw=1", p], capture_output=True, text=True)
        if r.returncode != 0 or "width=640" not in r.stdout or "height=480" not in r.stdout:
            bad_videos.append((cam, fn, r.stdout.strip()[:80]))
if not bad_videos and total_vids == 100:
    ok(f"100 个视频文件全部 640x480 可解 (50ep x 2cam)")
else:
    fail(f"视频问题: total={total_vids}, bad={bad_videos[:3]}")

# 5. stats.json 覆盖
print("\n[4] stats.json 覆盖 action/state 全维", flush=True)
stats_path = os.path.expanduser("~/.cache/huggingface/lerobot/datasets/bamboo_combined/meta/stats.json")
st = json.load(open(stats_path))
for feat in ["action", "observation.state"]:
    if feat not in st:
        fail(f"stats 缺 {feat}"); continue
    keys = set(st[feat].keys())
    if keys >= {"min", "max", "mean", "std"}:
        n = len(st[feat]["min"])
        ok(f"{feat}: {sorted(keys)} 维度={n}")
    else:
        fail(f"{feat} stats 字段不全: {keys}")

# 6. info.json features
print("\n[5] info.json features 与训练期望一致", flush=True)
info = json.load(open(os.path.expanduser("~/.cache/huggingface/lerobot/datasets/bamboo_combined/meta/info.json")))
feats = info.get("features", {})
def dim(feat):
    shp = feats[feat]["shape"]
    return shp[0] if isinstance(shp, list) else shp
for name, exp in [("action", 8), ("observation.state", 13)]:
    if name in feats and dim(name) == exp:
        ok(f"{name} dim={exp}")
    else:
        got = dim(name) if name in feats else "MISSING"
        fail(f"{name} 期望{exp} 实际{got}")
for cam in ["observation.images.handeye", "observation.images.fixed"]:
    if cam in feats and feats[cam].get("dtype") == "video":
        ok(f"{cam} dtype=video")
    else:
        fail(f"{cam} 不是 video: {feats.get(cam)}")
print(f"  fps={info.get('fps')} num_episodes={info.get('total_episodes')} total_frames={info.get('total_frames')}", flush=True)

print("\n" + ("===== 全部检查通过 OK 可以训练 =====" if OK else "===== 存在问题, 请先修复 ====="), flush=True)

# 远程训练流程：本地采集 → GPU 机训练

本地工作站无 GPU，只负责采集（手机遥操 + 相机 + 机器人）。训练在 GPU 机 `192.168.31.9`（RTX 4090 D）上进行。本文档记录把本地数据集和代码搬到 GPU 机并启动训练的完整流程。

## 机器角色

| 机器 | IP | 角色 |
|------|-----|------|
| 本地工作站 | 192.168.31.222 | 采集（手机遥操 + 相机 + 机器人） |
| GPU 机 | 192.168.31.9 | 训练（SSH 免密，用户 rentao） |
| 机器人 Aubo I10 | 192.168.31.200:30004 | LAN 两端可达 |

## 0. 前置条件

- 本地已用 `record.py` 采集数据集（默认 `./datasets/phone_auboi10`）。
- GPU 机已装 `v4l-utils`、`uv`、`ffmpeg`（`sudo apt install -y v4l-utils ffmpeg`，uv 在 `~/.local/bin/uv` 或 `/snap/bin/uv`）。
- 本地能 `ssh 192.168.31.9` 免密登录。

## 1. 找到本地数据集（重要坑点）

`record.py` 里 `LeRobotDataset.create(repo_id="./datasets/phone_auboi10")` **不会**把数据写到这个相对路径目录——它把 `phone_auboi10` 当数据集名，存进了 HuggingFace 缓存：

```
~/.cache/huggingface/lerobot/datasets/phone_auboi10/
├── meta/info.json            # episodes 数 / frames / features 定义
├── data/chunk-000/*.parquet  # 数值数据
└── videos/observation.images.{handeye,fixed}/chunk-000/*.mp4
```

`train.py` 里 `LeRobotDatasetMetadata("./datasets/phone_auboi10")` 能解析到它，是因为 LeRobot 用 repo_id 去 HF 缓存里找。所以训练机也得让这个缓存目录存在。

确认本地数据集内容：

```bash
cat ~/.cache/huggingface/lerobot/datasets/phone_auboi10/meta/info.json | head -40
```

## 2. 同步数据集到 GPU 机

把整个数据集缓存目录 rsync 到 GPU 机的同一缓存路径：

```bash
ssh 192.168.31.9 'mkdir -p ~/.cache/huggingface/lerobot/datasets'
rsync -av ~/.cache/huggingface/lerobot/datasets/phone_auboi10/ \
  192.168.31.9:~/.cache/huggingface/lerobot/datasets/phone_auboi10/
```

验证 GPU 机能加载：

```bash
ssh 192.168.31.9 'cd ~/program/lerobot-aubo/examples/phone_to_auboi10 && \
  ../../.venv/bin/python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
m = LeRobotDatasetMetadata(\"./datasets/phone_auboi10\")
print(\"episodes\", m.total_episodes, \"frames\", m.total_frames, \"fps\", m.fps)
print(\"features:\", list(m.features.keys()))
"'
```

## 3. 同步代码

GPU 机上 `git pull` 会因 github SSL 证书验证失败而报错，所以**直接 rsync 仓库**（排除 `.git/.venv/datasets` 等）：

```bash
cd /home/rentao/program/lerobot-aubo
rsync -av --delete \
  --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
  --exclude='datasets' --exclude='models' --exclude='logs' \
  --exclude='*.deb' --exclude='.claude' --exclude='.pytest_cache' \
  ./ 192.168.31.9:~/program/lerobot-aubo/
```

只改了单个文件时也可以直接 scp：

```bash
scp examples/phone_to_auboi10/train.py \
  192.168.31.9:~/program/lerobot-aubo/examples/phone_to_auboi10/train.py
```

## 4. 准备 GPU 机环境（首次）

参考 [[env-setup-network-uv]]。TUNA 镜像可直连；PyPI 上的 `torch` wheel 自带 CUDA，无需额外 pytorch 索引。

```bash
ssh 192.168.31.9 'cd ~/program/lerobot-aubo && \
  export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple && \
  uv venv --python 3.10 .venv && \
  uv pip install --index-strategy unsafe-best-match -e ".[phone]" pyaubo_sdk torch torchvision'
```

验证 GPU torch + ACT 可导入：

```bash
ssh 192.168.31.9 'cd ~/program/lerobot-aubo && .venv/bin/python -c "
import torch
print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0))
from lerobot.policies.act.modeling_act import ACTPolicy
print(\"ACT OK\")
"'
```

期望：`torch 2.10.0+cu128 | cuda True | NVIDIA GeForce RTX 4090 D`。

## 5. 启动训练

SSH 后台启动（nohup 脱离终端，输出重定向到日志文件）：

```bash
ssh 192.168.31.9 'cd ~/program/lerobot-aubo/examples/phone_to_auboi10 && \
  mkdir -p logs && rm -f logs/train_trial.log && \
  nohup ../../.venv/bin/python train.py > logs/train_trial.log 2>&1 &'
```

> 注意：`nohup ... &` over ssh 有时不会立即返回，命令可能挂起直到超时。这不影响训练——进程已在 GPU 机上 detach 启动。可另开一条 ssh 用 `pgrep -af train.py` 确认。

## 6. 监控训练

本地查看远程日志（loss 每 200 步打印一次）：

```bash
ssh 192.168.31.9 'tail -f ~/program/lerobot-aubo/examples/phone_to_auboi10/logs/train_trial.log' \
  | grep --line-buffered -E "step +[0-9]+ \| loss|Traceback|Error|Saved checkpoint|Training complete|CUDA out of memory"
```

或只看一眼最新 loss：

```bash
ssh 192.168.31.9 'grep "step" ~/program/lerobot-aubo/examples/phone_to_auboi10/logs/train_trial.log | tail -5'
```

checkpoint 每 5000 步落一个到 `examples/phone_to_auboi10/models/phone_auboi10/checkpoint_<step>/`，最终模型在同目录。

## 7. 停止训练

```bash
ssh 192.168.31.9 'pkill -9 -f train.py; sleep 1; pgrep -af train.py | grep -v pgrep || echo "all stopped"'
```

## 8. 已知坑：delta_timestamps

`train.py` 必须只给 `action` 加 delta_timestamps，**不要**给 `observation.state` 和图像加 `[0.0]`：

```python
# 正确
delta_timestamps = {
    "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
}
```

ACT 的 `observation_delta_indices = None`，标准流程 `resolve_delta_timestamps` 只给 action 加。若给观测加 `[0.0]`，会塞进一个多余的时间维（state 变 `(B,1,D)`），VAE encoder 里 `torch.cat` 会报 `Tensors must have same number of dimensions: got 3 and 4`。

## 9. 推理（评估）走 usbip

评估时相机经 usbip 从本地转发到 GPU 机，evaluate.py 整条跑在 GPU 机上。见仓库记忆 [[usbip-camera-forwarding]] 与 `evaluate.py` 顶部注释。

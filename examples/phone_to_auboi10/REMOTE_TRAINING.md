# 远程训练流程：本地采集 → GPU 机训练

> 录制和聚合步骤见 [README.md](README.md)。本文从
> `bamboo_newview_full` 已在本地生成开始。

本地工作站无 GPU，只负责采集（手机遥操 + 相机 + 机器人）。训练在 GPU 机 `192.168.31.9`（RTX 4090 D）上进行。本文档记录把本地数据集和代码搬到 GPU 机并启动训练的完整流程。

## 机器角色

| 机器 | IP | 角色 |
|------|-----|------|
| 本地工作站 | 192.168.31.222 | 采集（手机遥操 + 相机 + 机器人） |
| GPU 机 | 192.168.31.9 | 训练（SSH 免密，用户 rentao） |
| 机器人 Aubo I10 | 192.168.31.200:30004 | LAN 两端可达 |

## 0. 前置条件

- 本地已用 `aggregate.py` 生成 `./datasets/bamboo_newview_full`。
- GPU 机已装 `v4l-utils`、`uv`、`ffmpeg`（`sudo apt install -y v4l-utils ffmpeg`，uv 在 `~/.local/bin/uv` 或 `/snap/bin/uv`）。
- 本地能 `ssh 192.168.31.9` 免密登录。

## 1. 找到本地数据集（重要坑点）

LeRobot 会把 `./datasets/bamboo_newview_full` 解析到 HuggingFace 缓存：

```
~/.cache/huggingface/lerobot/datasets/bamboo_newview_full/
├── meta/info.json            # episodes 数 / frames / features 定义
├── data/chunk-000/*.parquet  # 数值数据
└── videos/observation.images.{handeye,fixed}/chunk-000/*.mp4
```

训练机也必须让这个缓存目录存在。

确认本地数据集内容：

```bash
cat ~/.cache/huggingface/lerobot/datasets/bamboo_newview_full/meta/info.json | head -40
```

## 2. 同步数据集到 GPU 机

把整个数据集缓存目录 rsync 到 GPU 机的同一缓存路径：

```bash
ssh 192.168.31.9 'mkdir -p ~/.cache/huggingface/lerobot/datasets'
rsync -av ~/.cache/huggingface/lerobot/datasets/bamboo_newview_full/ \
  192.168.31.9:~/.cache/huggingface/lerobot/datasets/bamboo_newview_full/
```

验证 GPU 机能加载：

```bash
ssh 192.168.31.9 'cd ~/program/lerobot-aubo/examples/phone_to_auboi10 && \
  ../../.venv/bin/python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
m = LeRobotDatasetMetadata(\"./datasets/bamboo_newview_full\")
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
  | grep --line-buffered -E "train_loss|val_loss|Traceback|Error|saved|Training complete|CUDA out of memory"
```

或只看一眼最新 loss：

```bash
ssh 192.168.31.9 'grep "step" ~/program/lerobot-aubo/examples/phone_to_auboi10/logs/train_trial.log | tail -5'
```

checkpoint 每 5000 步保存到 `models/bamboo_newview_act/checkpoint_<step>/`；
验证损失最低的模型保存在 `models/bamboo_newview_act/best/`。

## 7. 停止训练

```bash
ssh 192.168.31.9 'pkill -9 -f train.py; sleep 1; pgrep -af train.py | grep -v pgrep || echo "all stopped"'
```

## 8. 已知坑：delta_timestamps

`train.py` 必须只给 `action` 加 delta_timestamps，**不要**给 `observation.state` 和图像加 `[0.0]`：

```python
# 正确
delta_timestamps = {
    "action": [i / metadata.fps for i in policy.config.action_delta_indices],
}
```

ACT 的 `observation_delta_indices = None`，标准流程 `resolve_delta_timestamps` 只给 action 加。若给观测加 `[0.0]`，会塞进一个多余的时间维（state 变 `(B,1,D)`），VAE encoder 里 `torch.cat` 会报 `Tensors must have same number of dimensions: got 3 and 4`。

## 9. 拆分式纯 ACT 推理

GPU 机启动：

```bash
MODEL_PATH=./models/bamboo_newview_act/best python inference_server.py
```

本地工作站运行：

```bash
DATASET_PATH=./datasets/bamboo_newview_full python evaluate_split.py
```

相机和机器人留在本地，服务端只执行 ACT 前向推理。

# Phone to Aubo I10 — pure ACT workflow

本目录用于手机遥操作 AUBO I10、分批录制新视角数据、训练和拆分式纯 ACT 推理。

## 运行机器

- 工作站：连接机器人和相机，负责录制数据以及运行 `evaluate_split.py`。
- GPU 机：Tailscale 设备名 `510`，IP `100.88.143.45`，SSH 用户 `rentao`。
- GPU 机的 Linux 主机名是 `WP`，配备 NVIDIA GeForce RTX 4090 D。

从工作站连接 GPU 机：

```bash
ssh rentao@100.88.143.45
```

## 相机约定

- `handeye`：当前重新调整的眼在手外相机。
- `fixed`：另一视角相机。
- 录制、训练、推理期间，两台物理相机与以上键名的对应关系必须保持不变。

## 1. 录制前检查

以下工作站命令均从本目录执行：

```bash
cd /home/rentao/program/lerobot-aubo/examples/phone_to_auboi10
../../.venv/bin/python diag_preflight_cams.py
../../.venv/bin/python diag_cam_latency.py
```

`diag_preflight_cams.py` 会同时打开两台相机，并将预览图写到
`/tmp/aubo_camera_preflight/`。必须人工确认：

- `handeye.jpg` 是眼在手外的全局视角；
- `fixed.jpg` 是预期的第二视角；
- 竹条、吸盘接触区和放置区在任务关键阶段不会被遮挡。

机械臂上电后先运行只读连通检查：

```bash
../../.venv/bin/python read_pose.py
```

只有相机体检和 `read_pose.py` 都成功后才开始录制。

## 2. 分批录制

先只录 3 条试验数据，不要一开始录完整批次：

```bash
DATASET_PATH=./datasets/bamboo_newview_s01 \
NUM_EPISODES=3 \
../../.venv/bin/python record.py
```

录制完成后先运行 `aggregate.py` 的硬性验收，再逐条可视化：

```bash
../../.venv/bin/python aggregate.py

../../.venv/bin/lerobot-dataset-viz \
  --repo-id ./datasets/bamboo_newview_s01 \
  --episode-index 0
```

将 `--episode-index` 改成 `1`、`2`，检查所有试验 episode。确认动作、两路视频和
吸盘标签都正确后，再以新名称分批录制，每批建议 5～10 条：

```bash
DATASET_PATH=./datasets/bamboo_newview_s02 \
NUM_EPISODES=8 \
../../.venv/bin/python record.py
```

同一个 `bamboo_newview_sXX` 名称不要重复使用；每批使用新编号，避免覆盖或混入半成品。
录制动作中的吸盘目标是持续的 `0/100` 状态，不需要再执行夹爪锁存或标签前移脚本。

### 数据变化原则

- 相机位置、焦距、曝光、`handeye/fixed` 键名保持不变。
- 竹条位置和朝向做小范围、可复现的变化，不要每条完全相同。
- 首个基线可以保持机器人起始关节位一致，先隔离“物体变化”这一因素。
- 只保留完整、成功、动作连贯的抓取和放置；明显犹豫、碰撞、抓空应立即重录。
- episode 开始后尽快操作，完成放置和释放后立即结束，减少无意义静止帧。
- 3 条试录通过后，先累计至少 20～30 条训练首个基线；条件允许时逐步增加到
  40～60 条，分批录制和聚合即可。

## 3. 聚合

`aggregate.py`默认自动发现所有 `bamboo_newview_sXX`：

```bash
../../.venv/bin/python aggregate.py
```

输出为 `./datasets/bamboo_newview_full`。也可以显式指定：

```bash
DATASET_SOURCES=./datasets/bamboo_newview_s01,./datasets/bamboo_newview_s02 \
OUTPUT_DATASET_PATH=./datasets/bamboo_newview_full \
../../.venv/bin/python aggregate.py
```

聚合后会检查图像标准差，并确认夹爪标签只有持续状态 `0/100` 且两种状态都存在；
还会逐条报告时长、运动占比、吸取时长和最大位置跳变。发现非 25 FPS、缺失相机、
旧的 `50` 中立命令、非法数值或异常统计时会拒绝进入训练。

## 4. 同步到 GPU 机

GPU 机原仓库有未提交实验文件，不要执行 `reset --hard` 或直接覆盖。先在 GPU 机建立
隔离工作树：

```bash
ssh rentao@100.88.143.45
cd /home/rentao/program/lerobot-aubo
git fetch ssh://git@ssh.github.com:443/diiao/lerobot-aubo.git pyc
git worktree add /home/rentao/program/lerobot-aubo-pure-act FETCH_HEAD
exit
```

然后在工作站同步聚合数据集：

```bash
rsync -av --info=progress2 \
  ~/.cache/huggingface/lerobot/datasets/bamboo_newview_full/ \
  rentao@100.88.143.45:~/.cache/huggingface/lerobot/datasets/bamboo_newview_full/
```

## 5. GPU 训练

GPU 机已安装 `tmux`，且 ResNet18 权重已缓存。进入独立会话：

```bash
ssh rentao@100.88.143.45
tmux new -s aubo-act

cd /home/rentao/program/lerobot-aubo-pure-act/examples/phone_to_auboi10
export PYTHONPATH=/home/rentao/program/lerobot-aubo-pure-act/src
mkdir -p logs

HF_HUB_OFFLINE=1 \
DATASET_PATH=./datasets/bamboo_newview_full \
MODEL_PATH=./models/bamboo_newview_act_run01 \
TRAINING_STEPS=30000 \
NUM_WORKERS=8 \
/home/rentao/program/lerobot-aubo/.venv/bin/python train.py \
2>&1 | tee logs/train_run01.log
```

默认设置：

- 数据集：`bamboo_newview_full`
- 模型：`models/bamboo_newview_act`
- 从头训练，不加载旧视角 checkpoint
- 按完整 episode 保留最后 20% 作为验证集
- 新数据以 25 Hz 录制；ACT 默认使用约 1 秒预测窗口（25 帧）
- 每次执行约 0.16 秒（25 Hz 时为 4 步）后重新规划
- 首个基线不使用图像增强

可通过环境变量覆盖数据集、模型路径和训练步数。

按 `Ctrl-b`、再按 `d` 可退出 tmux 但保持训练运行；重新查看：

```bash
tmux attach -t aubo-act
```

GPU 系统盘当前剩余空间约 37 GB。每次实验使用独立 `MODEL_PATH`，训练前后用
`df -h /home/rentao` 检查空间，不要无限保留重复 checkpoint。

## 6. 纯 ACT 推理

GPU 机 `510`：

```bash
cd /home/rentao/program/lerobot-aubo-pure-act/examples/phone_to_auboi10
export PYTHONPATH=/home/rentao/program/lerobot-aubo-pure-act/src
MODEL_PATH=./models/bamboo_newview_act_run01/best \
/home/rentao/program/lerobot-aubo/.venv/bin/python inference_server.py
```

机器人端：

```bash
DATASET_PATH=./datasets/bamboo_newview_full \
EVAL_DATASET_PATH=./datasets/bamboo_newview_eval_run01 \
../../.venv/bin/python evaluate_split.py
```

服务端不会根据固定 xyz 阈值覆盖轨迹或夹爪；机器人端仅保留工作空间、单步位移限制和控制模式注入。

## 辅助工具

- `teleoperate.py`：手动遥操作检查。
- `move_to_start.py`：移动到统一起始姿态。
- `read_pose.py`：只读当前 TCP 位姿。
- `test_io.py`：确认吸盘 IO。
- `test_servo.py`：诊断伺服接口。

旧视角、启发式夹爪、脚本化下降/提起、不兼容当前绝对末端动作的旧回放脚本及旧数据专项诊断已从主流程移除；改造前版本保存在 Git 提交 `c4dfa2c`。

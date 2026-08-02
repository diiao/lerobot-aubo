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

### 常规录制与推理复用的统一起始位姿

以下为当前标准起始关节位，单位均为**度**，顺序为 `J1`～`J6`：

```text
[-65.29, -5.88, 113.77, 31.07, 90.88, -185.32]
```

`record.py`、`evaluate_split.py` 和 `move_to_start.py` 必须使用同一组值；它对应当前
30 条基线的正常任务起点（TCP 约为 `x=0.11, y=-0.72, z=0.15 m`）。后续常规推理先通过
`r` 或 `move_to_start.py` 回到此姿态，再开始 episode。不要把它改成恢复示教的悬停起点：
恢复示教有意从偏离状态开始，需单独记录和标注。

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

默认启用 ACT temporal ensembling（系数 `0.01`），它会融合相邻动作块以减少块边界跳变；
如需做旧队列模式对照实验，显式设置 `TEMPORAL_ENSEMBLE_COEFF=0`。推理客户端的通用
单帧末端位移限幅默认为 `8 mm`，可通过 `MAX_EE_STEP_M` 在 `1–50 mm` 范围内调整。两者
都是通用控制平滑/安全机制，不会根据竹条位置覆盖模型动作。

### 恢复示教（闭环偏离数据）

若模型悬在竹条上方、未下降或未吸取，不要在 `evaluate_split.py` 的同一轮中混入手机操控。
应安全结束推理、保持或手动准备该偏离姿态，然后以新的 `bamboo_newview_sXX_recovery`
数据集运行 `record.py`，从该姿态开始用手机完整示范“下降、吸取、抬升、放置、释放”。
这些 episode 与原始成功示教分开审核、聚合到新的训练集，并使用新的模型目录重训；它们让
ACT 在下次遇到偏离状态时学习恢复，而不是只能依赖理想轨迹。

## 7. 当前基线与下一轮恢复训练（2026-08-02）

当前 `bamboo_newview_full` 已由 `s01`～`s04` 聚合完成，共 **30 episodes / 26,939
frames**。GPU 机上的首个 30 条模型为 `bamboo_newview_act_run05/best`，训练 30,000
steps，最佳保留集 loss 为 `0.05918`。这只是第一条端到端基线，不能据此宣称真实抓取已
成功。

离线审计显示：模型在专家示教轨迹上能正确下降和开吸盘，但真实闭环推理一旦悬在高处、
错过下降阶段，就无法回到训练中常见的正确状态，继而持续输出释放和悬停动作。下列恢复
示教用于补足这一类“模型已偏离”的状态；它不是在现有推理轮中插入人工动作。

### 7.1 录制恢复示教

先完成相机预检和 `read_pose.py`。机械臂、急停和工作区必须由现场操作者确认安全。

```bash
cd /home/rentao/program/lerobot-aubo/examples/phone_to_auboi10

DATASET_PATH=./datasets/bamboo_newview_s05_recovery \
NUM_EPISODES=20 \
../../.venv/bin/python record.py
```

每条 episode 的人工操作顺序：

1. 让机械臂处于模型常见的失败状态：吸盘已释放、末端在竹条上方约 6～10 cm；如果刚刚
   结束失败推理，可安全结束该轮后保持该姿态，或由操作者手动准备相近姿态。
2. `record.py` 出现“按 -> 开始录制”后，**不要按 `r`**；`r` 会回到正常起始位，丢失恢复
   场景。按右箭头开始录制。
3. 在该悬停位置保留约 0.5～1 秒，然后使用手机连续示范：下降、吸取、确认吸住、抬升、
   搬运、放置、释放。
4. 释放后立即按右箭头结束。不要只录下降/抓取，也不要在抓住后停下；应录完整的任务后半段。
5. 明显碰撞、抓空、犹豫过久或相机断流的 episode 用左箭头重录，不进入训练数据。

录完后逐条可视化并审核。恢复轨迹的起始位姿不同是预期现象；相机映射、分辨率、25 FPS、
action 定义和吸盘持续 `0/100` 标签必须与 `s01`～`s04` 一致。

### 7.2 新聚合、训练和离线验收

`s05_recovery` 带后缀，`aggregate.py` 不会自动发现它。因此显式列出所有来源，并写到新的
输出目录，保留原 `bamboo_newview_full` 作为可回退基线：

```bash
cd /home/rentao/program/lerobot-aubo/examples/phone_to_auboi10

DATASET_SOURCES=./datasets/bamboo_newview_s01,./datasets/bamboo_newview_s02,./datasets/bamboo_newview_s03,./datasets/bamboo_newview_s04,./datasets/bamboo_newview_s05_recovery \
OUTPUT_DATASET_PATH=./datasets/bamboo_newview_full_recovery_v1 \
../../.venv/bin/python aggregate.py
```

审核通过后同步这个新目录到 GPU 机，并使用新的模型目录（例如
`bamboo_newview_act_run06_recovery`）从头训练；不要覆盖 `run05`。训练结束后必须先运行
`audit_act_offline.py`，确认保留 episode 的吸盘召回、抓取高度误差和动作跳变，再决定是否
进行真实机械臂推理。

恢复训练的验收重点不是只看总 loss：

- 吸盘首次开启时，预测值必须跨过执行阈值 `60`；
- 抓取点的预测高度要接近示教高度，误差应以厘米以下为目标；
- 正常保留轨迹与恢复起始轨迹都不能出现大量超过 1 cm 的单帧跳变；
- 真实推理时若再次偏离，应能下降并完成吸取，而不是持续悬停。

## 辅助工具

- `teleoperate.py`：手动遥操作检查。
- `move_to_start.py`：移动到统一起始姿态。
- `read_pose.py`：只读当前 TCP 位姿。
- `test_io.py`：确认吸盘 IO。
- `test_servo.py`：诊断伺服接口。

旧视角、启发式夹爪、脚本化下降/提起、不兼容当前绝对末端动作的旧回放脚本及旧数据专项诊断已从主流程移除；改造前版本保存在 Git 提交 `c4dfa2c`。

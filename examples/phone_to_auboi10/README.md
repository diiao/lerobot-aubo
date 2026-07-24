# Phone to Aubo I10 — pure ACT workflow

本目录用于手机遥操作 AUBO I10、分批录制新视角数据、训练和拆分式纯 ACT 推理。

## 相机约定

- `handeye`：当前重新调整的眼在手外相机。
- `fixed`：另一视角相机。
- 录制、训练、推理期间，两台物理相机与以上键名的对应关系必须保持不变。

## 1. 录制前检查

```bash
python diag_preflight_cams.py
python diag_cam_latency.py
```

确认相机位置后，可用 `check_handeye_align.py` 对比参考图和现场图。

## 2. 分批录制

每批使用独立数据集名称，避免覆盖已有数据：

```bash
DATASET_PATH=./datasets/bamboo_newview_s01 NUM_EPISODES=10 python record.py
DATASET_PATH=./datasets/bamboo_newview_s02 NUM_EPISODES=10 python record.py
```

录制动作中的吸盘目标是持续的 `0/100` 状态，不需要再执行夹爪锁存或标签前移脚本。

## 3. 聚合

`aggregate.py`默认自动发现所有 `bamboo_newview_sXX`：

```bash
python aggregate.py
```

输出为 `./datasets/bamboo_newview_full`。也可以显式指定：

```bash
DATASET_SOURCES=./datasets/bamboo_newview_s01,./datasets/bamboo_newview_s02 \
OUTPUT_DATASET_PATH=./datasets/bamboo_newview_full \
python aggregate.py
```

聚合后会检查图像标准差，并确认夹爪标签只有持续状态 `0/100` 且两种状态都存在；
发现旧的 `50` 中立命令或异常统计时会拒绝进入训练。

## 4. 训练

```bash
python train.py
```

默认设置：

- 数据集：`bamboo_newview_full`
- 模型：`models/bamboo_newview_act`
- 从头训练，不加载旧视角 checkpoint
- 按完整 episode 保留最后 20% 作为验证集
- `chunk_size=30`，每次执行 5 步后重新规划
- 首个基线不使用图像增强

可通过环境变量覆盖数据集、模型路径和训练步数。

## 5. 纯 ACT 推理

GPU端：

```bash
MODEL_PATH=./models/bamboo_newview_act/best \
python inference_server.py
```

机器人端：

```bash
DATASET_PATH=./datasets/bamboo_newview_full \
python evaluate_split.py
```

服务端不会根据固定 xyz 阈值覆盖轨迹或夹爪；机器人端仅保留工作空间、单步位移限制和控制模式注入。

## 辅助工具

- `teleoperate.py`：手动遥操作检查。
- `move_to_start.py`：移动到统一起始姿态。
- `read_pose.py`：只读当前 TCP 位姿。
- `test_io.py`：确认吸盘 IO。
- `test_servo.py`：诊断伺服接口。

旧视角、启发式夹爪、脚本化下降/提起、不兼容当前绝对末端动作的旧回放脚本及旧数据专项诊断已从主流程移除；改造前版本保存在 Git 提交 `c4dfa2c`。

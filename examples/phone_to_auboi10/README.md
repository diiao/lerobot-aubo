# Phone to Aubo I10

这个目录包含用于通过手机遥操作 Aubo I10 机械臂、记录数据、回放数据和评估策略的脚本。

## 文件说明

- `teleoperate.py` - 基础手机遥操作脚本（已跑通）
- `record.py` - 记录数据集脚本
- `replay.py` - 回放已记录数据脚本
- `evaluate.py` - 评估训练好的策略脚本
- `REMOTE_TRAINING.md` - 远程训练流程（本地采集 → GPU 机训练）

## 使用流程

### 1. 遥操作测试
首先确保 `teleoperate.py` 可以正常运行：
```bash
python teleoperate.py
```

### 2. 记录数据集
使用 `record.py` 记录演示数据：
```bash
python record.py
```

该脚本会：
- 连接手机和 Aubo I10 机械臂
- 连接两个相机（handeye 和 fixed）
- 记录 NUM_EPISODES 个 episode（默认 3 个）
- 每个 episode 时长 EPISODE_TIME_SEC 秒（默认 60 秒）
- 数据保存到 LOCAL_DATASET_PATH（默认 ./datasets/phone_auboi10）

### 3. 回放数据
使用 `replay.py` 回放已记录的数据：
```bash
python replay.py
```

该脚本会：
- 加载记录的数据集
- 回放指定的 episode（默认第 0 个）
- 在 Aubo I10 上复现演示动作

### 4. 训练 ACT 策略
记录数据后，可以参照 `examples/tutorial/act/act_training_example.py` 训练 ACT 策略。

主要修改点：
- 将 `dataset_id` 改为你记录的数据集路径
- 确保 input_features 和 output_features 与 Aubo I10 的观测/动作空间匹配

### 5. 评估策略
训练好策略后，使用 `evaluate.py` 进行评估：
```bash
python evaluate.py
```

## 配置说明

### 相机配置
在 `record.py` 中修改相机配置：
```python
camera_config = {
    "handeye": OpenCVCameraConfig(index_or_path="/dev/video0", width=640, height=480, fps=FPS),
    "fixed": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=FPS),
}
```

### 机械臂 IP
在 `src/lerobot/robots/aubo_i10/aubo_i10.py` 中修改：
```python
self.robot_ip = "192.168.31.200"
self.robot_port = 30004
```

### 记录参数
在 `record.py` 顶部修改：
```python
NUM_EPISODES = 3          # 记录的 episode 数量
FPS = 30                   # 采样频率
EPISODE_TIME_SEC = 60      # 每个 episode 的时长（秒）
RESET_TIME_SEC = 30        # 重置环境的时长（秒）
TASK_DESCRIPTION = "My task description"  # 任务描述
LOCAL_DATASET_PATH = "./datasets/phone_auboi10"  # 数据集保存路径
```

## 数据格式

记录的数据集使用 LeRobotDataset 格式，包含：
- 观测数据：关节角度 (J1-J6)、末端位姿 (ee.x/y/z/wx/wy/wz)、夹爪位置、相机图像
- 动作数据：末端位姿增量、夹爪位置
- 视频文件：相机录制的 MP4 视频

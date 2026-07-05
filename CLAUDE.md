# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modified version of **LeRobot** - Hugging Face's open-source robotics library - specifically customized for the **Aubo I10 robot arm**. It provides a hardware-agnostic, Python-native interface for robot control, data collection, policy training, and deployment.

**Current Branch:** `rt`

## Key Architecture

### Aubo I10 Robot Implementation

The Aubo I10 robot implementation is in `src/lerobot/robots/aubo_i10/`:

- **`aubo_i10.py`**: Main `AuboI10Robot` class with two control modes:
  - **Joint angle control**: Directly control J1-J6 joint angles (in degrees)
  - **End-effector pose control**: Uses Aubo's servoCartesian interface for real-time control
  - Supports soft gripper control via digital IO pins 4 and 5
  - Default robot IP: `192.168.31.200:30004`

- **`config_aubo_i10.py`**: `AuboI10Config` dataclass for robot configuration including camera setup

- **`robot_processor.py`**: Processor steps for action conversion:
  - `PhoneEEToAuboEE`: Converts phone delta commands to absolute EE pose
  - `AuboGripperVelocityToPosition`: Converts gripper velocity to position
  - `AuboEEBoundsAndSafety`: Clips EE pose to safe bounds
  - `AuboEEToEEDelta`: Converts absolute EE pose to delta for control

### Example Workflows

The main example scripts are in `examples/phone_to_auboi10/`:

- **`teleoperate.py`**: Basic phone teleoperation
- **`record.py`**: Record datasets with phone teleoperation
- **`replay.py`**: Replay recorded data
- **`evaluate.py`**: Evaluate trained policies

## Development Commands

### Installation

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev,test]"

# Install with Aubo I10 and phone teleop support
pip install -e ".[phone]"
```

### Code Quality

The project uses:
- **ruff** for linting and formatting
- **mypy** for type checking (gradually enabled)
- **pre-commit** hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Run ruff
ruff check .
ruff format .
```

### Testing

```bash
# Run all tests
pytest -sv ./tests

# Run a specific test file
pytest -sv tests/test_specific_feature.py

# Run with coverage
pytest --cov=lerobot tests/
```

Note: Test artifacts require git-lfs: `git lfs install && git lfs pull`

### CLI Commands

LeRobot provides these CLI entry points (from pyproject.toml):

- `lerobot-record` - Record data from real robots
- `lerobot-train` - Train policies on datasets
- `lerobot-eval` - Evaluate policies
- `lerobot-teleoperate` - Control robots manually
- `lerobot-find-cameras` - Find and configure cameras
- `lerobot-replay` - Replay recorded data
- `lerobot-info` - Show library info
- And more...

## Important Patterns

### Robot Processor Pipeline

The code uses `RobotProcessorPipeline` to chain transformation steps. For recording:

```python
phone_to_robot_ee_pose_processor = RobotProcessorPipeline(
    steps=[
        MapPhoneActionToRobotAction(),
        PhoneEEToAuboEE(),
        AuboEEBoundsAndSafety(),
        AuboGripperVelocityToPosition(),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)
```

### Observation/Action Features

- **Observation features**: J1-J6 (joint angles), ee.x/ee.y/ee.z/ee.wx/ee.wy/ee.wz (EE pose), gripper_pos, camera images
- **Action features**: ee.x/ee.y/ee.z/ee.wx/ee.wy/ee.wz (EE pose delta), ee.gripper_pos

### Camera Support

The project supports multiple camera types including OpenCV, Intel RealSense, and Mech-Mind. Cameras are configured via `CameraConfig` and created with `make_cameras_from_configs()`.

## Key Files

| Path | Purpose |
|------|---------|
| `src/lerobot/robots/aubo_i10/aubo_i10.py` | Aubo I10 robot implementation |
| `src/lerobot/robots/aubo_i10/robot_processor.py` | Action processing pipelines |
| `examples/phone_to_auboi10/record.py` | Dataset recording example |
| `pyproject.toml` | Project dependencies and config |

## Recent Development Focus

Recent commits (on branch `rt`) have focused on:
- Camera configuration and debugging
- Robot processor improvements
- Policy evaluation on Aubo I10

## 摄像头调试约定

调试摄像头/视频时,不要用 Read 工具去查看抓取到的图片或视频帧文件。改用以下纯文本手段判断是否正常出帧:
- `ffprobe` 的文本输出(分辨率、帧率、编码、帧数)
- `ffmpeg` / 抓帧命令的退出码(`$?`)
- 保存下来的帧文件大小(`ls -l` 的字节数,非零即说明有数据)

如果确实需要根据图像内容来做判断(比如确认画面方向、花屏、曝光),不要自己 Read,而是把图保存到一个固定路径(例如 `/tmp/cam_check_<cam>.png`)并明确告诉我路径,让我自己打开看。只告诉我路径和判断要点,不要把图像内容读进上下文。

## 当前模型限制

当前 Claude Code 接入的模型**只支持文字输入,不支持图片输入**。如果向模型传入图片(例如用 Read 工具读取图片文件、粘贴图片到对话、附截图等),会卡住或报错。所有涉及图像的工作必须走上面的纯文本/保存路径流程,由用户自行查看图像内容。

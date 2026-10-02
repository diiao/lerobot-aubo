# AUBO i10：七维 SmolVLA 竹条抓放

当前主线使用双 RGB 和六关节＋吸盘状态，由 SmolVLA 预测七维绝对关节动作。
从仓库根目录 `/home/rentao/program/lerobot-aubo` 运行下列说明中的命令。

| 工作 | 入口 | 说明 |
| --- | --- | --- |
| 手机遥操采集 | `record_joint.py` | [采集说明](JOINT_CAPTURE.md) |
| 数据审计、微调、离线评价 | `audit_joint_dataset.py`、`train_joint_smolvla.py` | [训练说明](JOINT_TRAINING.md) |
| 归位、持续抓放 | `run_joint_trial.py --stage home`、`run_joint_smooth.py` | [运行说明](JOINT_TRIAL.md) |
| 模型身份、实机结果、历史回溯 | — | [证据索引](EVIDENCE.md) |
| 两三根静态层序评价 | `evaluate_layer_order.py` | [研究范围与标注格式](../../docs/AUBO_I10_TOP_LAYER_VLM_RL_ROUTE_2026-09-26.md) |

## 当前契约

- `CameraSetV2`：`global_rgb` 为固定全局视角，`grasp_rgb` 为腕部视角，两路 RGB 为 640×480。
- 状态和动作均为 `[J1,J2,J3,J4,J5,J6,gripper_pos]`。关节使用度；进入机器人 SDK 时转换为弧度。J5 参与模型预测。
- 吸盘指令为 0（释放）或 100（吸附）；不是物理吸附传感器。
- 固定任务文字：`Pick one strip and place it in the collection area.`。
- 当前模型：`mixed-both-orders`，单根 60＋上45°下90° 50＋上90°下45° 50，共 160 条训练、22 条验证。

## 当前能力与限制

两种熟悉双根摆放各有一次两根抓放成功反馈。单根抓放已有成功反馈，但混训入口仍要求两个
吸附→释放周期，可能在单根放下后再执行一轮。周期数不能由命令独立指定，也没有自动清空判断。
现有结果不能当作未知位置或三根以上堆叠的成功率。

最新现场模式为 `--model mixed-both-orders --approach-age-aligned`：吸盘指令为释放时按观测年龄选择完整动作行，
等待闭合期间继续单请求推理；吸附后使用首步预测。具体命令与停止行为见运行说明。

## 开发与诊断

日常检查使用 `.venv/bin/python`。计划模式不连接硬件、不加载模型；相机、机械臂、IO、运动及
实际训练由现场操作者单独启动。模型输出不加人为接触高度或固定姿态补偿。

当前采集仍复用旧命名的 `record.py`、`record_c0_batch.py` 和部分 `c0_*` 模块，
不得按“旧文件名”整批删除。ACT 和旧 TCP 路线的历史说明在证据索引中，不作为当前命令入口。

软件回归可使用：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/bamboo_sorting/test_joint_smooth.py \
  tests/bamboo_sorting/test_joint_trial.py \
  tests/bamboo_sorting/test_joint_training.py \
  tests/bamboo_sorting/test_layer_order_eval.py
git diff --check
```

修改采集、驱动或视频链路时，另运行对应测试。假设备测试通过不代表实机验收。

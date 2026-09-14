# Handoff for Codex: 12D drop_gripper 真机推理（live02–live12）

接手时仓库工作树干净。本轮在 `lerobot-aubo` 工作站完成；GPU 为 Tailscale `100.88.143.45`（`ssh gpu`）。

## 任务目标

新训 12D `drop_gripper` ACT 真机评估。起点：

- live02：`ThinSafetyGate ik:0`（J5 连续性 14.346° vs 0.25 rad）
- 后续：下降/吸盘/放置、DO 回读、夹爪抖动、90°/135° 悬停

## 模型 / 服务

- 客户端：`examples/phone_to_auboi10/evaluate_split.py`
- GPU 推理：`/tmp/lerobot_act_server_4ac1090/inference_server.py`（从本仓库拷贝后重启）
- `MODEL_PATH=/home/rentao/program/lerobot-aubo-act-ab-v1/examples/phone_to_auboi10/models/bamboo_act_single_strip_ab_v1_drop_gripper_12d_seed42/best`
- `PYTHONPATH=/home/rentao/program/lerobot-aubo-act-ab-v1/src`
- `TEMPORAL_ENSEMBLE_COEFF=0.01`，`ENSEMBLE_GRIPPER_MODE=latest`
- 训练切分：`configs/aubo_i10/act_full65_drop_gripper_12d_v1.json`（s01–s14，含 0°–157.5° 单根）
- 示教数据：`datasets/bamboo_act_report_full`

## 真机结果摘要

| run | 摆放 | 结果 |
|---|---|---|
| live02 | 12D 首次 | ~14s `ik:0`，rotvec 过 180°，J5 超 0.022° |
| live03 | 姿态限幅 0.03 rad | 满 60s，悬停 `[0.57,-0.41,0.20]`，未下降 |
| live04 | 姿态限幅 0.10 rad | 同悬停；后半段限幅未再裁 |
| live05 | `HOLD_LOCKED_EE_POSE` | **降到 z≈0.088**，夹爪全程 0；chunk 末 50–58；到位后 `servoJoint -13` |
| live06 | chunk 夹爪 >50 → 100 | 到位吸；DO 写成功、立刻回读仍是放 → 整轮中止 |
| live07 | 同左 | chunk_max 最高 36.5 <50，未吸；再次 `-13` |
| live08 | **45°** | **整轮成功**；抓/放时夹爪连切两次 |
| live09 | 90° | 悬停 `[0.58,-0.53,0.22]`，J6≈-163°，z 从未 <0.147 |
| live10 | 90° + motion chunk | GPU 新 trace 生效；**chunk 内 z 最多低 2.7mm**，lookahead 0 次 |
| live11 | 135° | 用户报告同样悬停 |
| live12 | 135°，120s | 用户报告同样悬停 |

45° 对上示教 `s04`（抓取 J6≈-96°）。90° 示教是 `s01`（J6≈-146°、z≈0.095、y≈-0.36），live09/10 没有走到该抓取位。

## 改了哪些文件

未提交。回滚：

```bash
git checkout -- \
  examples/phone_to_auboi10/evaluate_split.py \
  examples/phone_to_auboi10/inference_server.py \
  src/lerobot/robots/aubo_i10/aubo_i10.py \
  tests/bamboo_sorting/test_evaluate_split_safety.py \
  tests/robots/test_aubo_gripper_io.py
```

### `examples/phone_to_auboi10/evaluate_split.py`

- `HOLD_LOCKED_EE_POSE=1`（默认）：整轮发送开场实测 rotvec。示教 `ee.wx/wy/wz` 是锁存姿态，J6 单独偏航；跟模型漂移 rotvec 并朝含 J6 的实测 TCP 限姿态会打架。这是 live05 能下降的关键。
- `MAX_EE_ROT_STEP_RAD=0.03`：测地线限幅 + rotvec 2π 对齐。`HOLD` 开启时**不再**朝实测 TCP 限姿态。
- IK 诊断写入 trace：`errno`、解、各关节差。
- 夹爪 chunk lookahead：`max(chunk)>20` 吸，`<8` 放，中间 40 走进迟滞保持带（修 live08 连切）。
- 运动 chunk lookahead：吸盘未开且 chunk 最低 z 比当前低 ≥2cm 则改用该步 xyz/J6。live10 证明 90° 时 chunk 没有下降。
- 未放宽 `MAX_IK_JOINT_STEP_RAD`，未关 IK，未改 ACT 权重。

### `examples/phone_to_auboi10/inference_server.py`

- trace 增加 `predicted_chunk_denormalized_action`（整段 chunk）。GPU `/tmp/lerobot_act_server_4ac1090/` 已拷贝并重启（pid 当时 50673，端口 5555）。

### `src/lerobot/robots/aubo_i10/aubo_i10.py`

- DO 回读最多 8 次、间隔 25ms（live06 写成功但首读仍是旧值）。
- `abs_j6yaw` 的 `servoJoint ret=-13`：重开伺服再试。

### 测试

- `tests/bamboo_sorting/test_evaluate_split_safety.py`
- `tests/robots/test_aubo_gripper_io.py`
- 最近一次：相关测试通过。

### 未改

机器人层其它控制、ACT 网络、夹爪 60/20 迟滞公式、已有 parquet 实验数据。`configs/aubo_i10/act_full65_drop_gripper_12d_v1.json` 本轮未改（原先就未跟踪）。

## 结论（给后续）

1. **姿态锁 + 夹爪 chunk 迟滞** 足以让 **45°** 闭环走完抓放（live08）。
2. **90°/135° 悬停不是安全门问题。** live10：模型 25 步 chunk 的 z 几乎等于当前高度（最大再低 2.7mm）。推理再挖 chunk / 放宽 IK / 加长 120s **都变不出下降**。
3. 数据里**有**这些角度（s01=90°，s07=135°），但是教师强制训练 ≠ 该相机画面下的闭环。走到的悬停位姿在示教里不是抓取位。
4. 不要再放宽 `MAX_IK_JOINT_STEP_RAD`、不要关 IK、不要按竹条坐标写死下降。

## 不录恢复示教的话还能做什么

按有效性排序：

1. **换/重训一个在多角度闭环更好的 checkpoint**（同一批 65 ep 加长训练、换 seed、或 13D 全状态）。这是不采新轨迹时唯一可能改变「chunk 里有没有下降」的办法。
2. **`TEMPORAL_ENSEMBLE_COEFF=0` 重开 GPU 再测 45° 对照 + 90°。** live10 已表明 raw chunk 也没有下降， succeess 概率低，但没做过 A/B。
3. **不要**再做客户端 motion/夹爪 lookahead 阈值空转；90° 的瓶颈在策略输出。
4. **不要**写「悬停 N 秒就向下 8mm」这类任务启发式（违反现有安全约定，且 90° 时 J6/xy 也不对）。

若政策允许采数：从 live09/10 悬停位做 3–5 条手机恢复示教（下降-吸-放）再训，这是 README 里原定的偏离恢复路径，也是闭环 90°/135° 最对症的办法。

## 真机注意

- 新评估必须新的 `EVAL_DATASET_PATH`。
- 自动归位默认未授权；示教器归位到 `[-65.29, -5.88, 113.77, 31.07, 90.88, -185.32]°`。
- 改 `inference_server.py` 后要同步到 GPU `/tmp/lerobot_act_server_4ac1090/` 并重启，保持原 `MODEL_PATH` / `PYTHONPATH`。

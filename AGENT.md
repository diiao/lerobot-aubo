# AUBO + LeRobot 项目背景与当前状态

> 更新时间：2026-08-02
>
> 用途：帮助后续维护者快速理解项目背景、当前目标、系统架构、工程约束和阶段性数据状态。

## 文档定位

- 本文记录“为什么这样做、当前做到哪里、哪些约束不能破坏”。
- 具体录制、聚合、远程训练和推理命令以
  [`examples/phone_to_auboi10/README.md`](examples/phone_to_auboi10/README.md)
  为准。
- 工作站与 GPU 机的连接和数据同步约束以
  [`examples/phone_to_auboi10/REMOTE_TRAINING.md`](examples/phone_to_auboi10/REMOTE_TRAINING.md)
  为准。
- `CLAUDE.md` 描述了项目的基础架构，但其中 `rt` 分支和“模型不能查看图片”等内容
  属于旧工具状态，不能覆盖本文记录的当前项目状态。
- 代码是行为的最终依据。文档与代码冲突时，应先只读检查代码和 Git 历史，不要直接按
  旧文档修改硬件控制逻辑。

## 项目背景

本仓库基于 Hugging Face 的 LeRobot 修改，目标是在 LeRobot 的机器人、数据集和策略框架
内支持 AUBO i10 机械臂。当前具体任务是：

1. 使用 Android 手机遥操作 AUBO i10。
2. 录制机械臂状态、绝对末端目标、吸盘状态和两路相机视频。
3. 将分批数据聚合成 LeRobot 数据集。
4. 在 GPU 机上从头训练 ACT（Action Chunking with Transformers）策略。
5. 由 ACT 根据实时观测直接预测动作，控制机械臂完成“抓取竹条”任务。

ACT 的核心思想是一次预测一段连续动作，而不是只预测下一帧。当前实现以约 1 秒的
25 帧动作块训练，推理时执行约 4 帧（0.16 秒）后重新观察并规划。

## 当前工作阶段

- 当前开发分支是 `pyc`。
- 眼在手外相机已经重新调整位置，因此旧视角数据和旧 checkpoint 不应混入当前基线。
- `bamboo_newview_s01`～`s04` 已完成审核并聚合为 30 条首个基线；run05 已完成训练与一次
  真实推理验证。当前阶段是针对闭环偏离补录恢复示教，再建立独立的 run06 对照实验。
- run05 只验证了数据、训练、网络推理和机器人控制的完整链路，不应被描述为已经具有稳定
  抓取成功率或强泛化能力。
- 最终推理目标是纯 ACT：模型输出不得被固定 xyz 阈值、脚本化下降/提起、启发式夹爪
  或其他任务专用规则覆盖。机器人端只保留通用安全边界、单步位移限制和控制模式注入。

### 2026-07-28 Git 状态快照

- 本地 `pyc` 相对本地记录的 `origin/pyc` 显示领先 6 个提交；这只是本地 Git 状态，
  未联网复核远端。
- `examples/phone_to_auboi10/record.py` 和
  `examples/phone_to_auboi10/diag_preflight_cams.py` 有未提交修改。
- 本文 `AGENT.md` 是新增文件。
- 在提交、推送或同步 GPU 机前必须重新运行 `git status`，不要假设工作树干净。

## 系统组成与网络拓扑

| 设备 | 地址 | 主要职责 |
|---|---|---|
| 工作站 | 局域网 `192.168.31.222`；Tailscale `100.93.231.40` | 连接相机和机械臂；录制；运行推理客户端 |
| GPU 机 `510` | Tailscale `100.88.143.45`；主机名 `WP` | RTX 4090 D；ACT 训练与推理服务端 |
| AUBO i10 | `192.168.31.200:30004` | 仅由工作站通过局域网控制 |

工作站与 GPU 机通过 Tailscale 通信。拆分式推理时，GPU 机运行
`inference_server.py`，工作站运行 `evaluate_split.py`，默认连接
`100.88.143.45:5555`。相机和机械臂不直接连接 GPU 机。

## 核心代码架构

### AUBO i10 机器人实现

主要代码位于 `src/lerobot/robots/aubo_i10/`：

| 文件 | 作用 |
|---|---|
| `aubo_i10.py` | 机器人连接、状态读取、关节/末端控制、`servoCartesian`、吸盘 IO 和相机观测 |
| `config_aubo_i10.py` | `AuboI10Config`，包含相机配置与控制频率 |
| `robot_processor.py` | 手机动作转换、末端位姿、安全边界、吸盘锁存和推理动作处理 |

机器人支持两类主要控制输入：

- 关节角控制：J1～J6，单位为度。
- 末端位姿控制：`ee.x/y/z/wx/wy/wz`，位置单位为米，姿态使用旋转向量。

当前竹条数据采用绝对末端目标，不是旧版的增量 action。AUBO 的
`servoCartesian` 对旋转向量的等价表示和连续性比较敏感，不能随意删除姿态对齐、
竖直姿态锁定或安全处理逻辑。

### 录制数据流

```text
Android 手机
  -> MapPhoneActionToRobotAction
  -> AuboLockVerticalYaw
  -> PhoneEEToAuboEE（速度输入转换为绝对末端目标）
  -> AuboEEBoundsAndSafety
  -> AuboGripperVelocityToPosition（持续的 0/100 吸盘状态）
  -> AUBO i10 执行并写入 LeRobotDataset
```

相机、机器人状态和最终发送给机器人的绝对目标同步写入数据集。夹爪必须记录持续状态，
不能退回旧的单帧按键脉冲或 `50` 中立值，否则 ACT 很难学到吸取和释放时序。

### 当前数据特征

- `observation.state`：13 维，包括 J1～J6、末端 xyz/旋转向量和 `gripper_pos`。
- `action`：8 维，包括 `ee.j6_target`、末端 xyz/旋转向量和
  `ee.gripper_pos`。
- `observation.images.handeye`：480×640×3 视频。
- `observation.images.fixed`：480×640×3 视频。
- 数据集控制频率：25 FPS。

### 常规录制与推理统一起始位姿

当前标准起始关节角（单位：度，顺序 `J1`～`J6`）为：

```text
[-65.29, -5.88, 113.77, 31.07, 90.88, -185.32]
```

该值在 `record.py`、`evaluate_split.py` 和 `move_to_start.py` 中必须保持一致；它是当前
30 条基线的正常任务起点，TCP 约为 `(0.11, -0.72, 0.15) m`。常规录制/推理使用 `r` 或
`move_to_start.py` 回到此位姿。恢复示教的悬停起点是有意不同的数据状态，不能覆盖或替换
该标准起点。

### ACT 训练与推理

- `train.py` 使用纯 ACT，视觉骨干为带 ImageNet 预训练权重的 ResNet18。
- 默认从头训练 30,000 steps，不加载旧视角 checkpoint。
- 默认保留最后 20% 完整 episode 作为验证集。
- 默认 `chunk_size=25`，`n_action_steps=4`，首个基线不使用图像增强。
- 标准 ACT CVAE 是当前基线；修改 `USE_VAE`、动作块长度或执行步数时必须建立独立实验，
  不能同时改变多项后只凭一次推理下结论。
- `inference_server.py` 默认使用低延迟 action queue；只有显式配置
  `TEMPORAL_ENSEMBLE_COEFF` 才启用 temporal ensemble。
- `evaluate_split.py` 只加入通用安全边界和 `abs_j6yaw` 控制模式，不包含任务位置启发式。

## 关键工作流文件

| 文件 | 作用 |
|---|---|
| `examples/phone_to_auboi10/diag_preflight_cams.py` | 双相机录制前检查与预览 |
| `examples/phone_to_auboi10/diag_cam_latency.py` | 双相机帧率和延迟检查 |
| `examples/phone_to_auboi10/read_pose.py` | 机械臂只读连通检查 |
| `examples/phone_to_auboi10/move_to_start.py` | 移动到统一起始关节位 |
| `examples/phone_to_auboi10/teleoperate.py` | 手机遥操作检查 |
| `examples/phone_to_auboi10/record.py` | 分批录制数据 |
| `examples/phone_to_auboi10/aggregate.py` | 校验并聚合 `bamboo_newview_sXX` |
| `examples/phone_to_auboi10/train.py` | 训练纯 ACT |
| `examples/phone_to_auboi10/inference_server.py` | GPU 机纯 ACT 推理服务 |
| `examples/phone_to_auboi10/evaluate_split.py` | 工作站推理客户端和评估数据记录 |
| `examples/phone_to_auboi10/evaluate.py` | 模型和机器人位于同机时的直接评估 |
| `examples/phone_to_auboi10/test_io.py` | 吸盘 IO 诊断 |
| `examples/phone_to_auboi10/test_servo.py` | `servoCartesian` 诊断 |

## 操作与安全约束

- 录制、聚合、训练和推理任务不得由助手自动启动；应先给出命令，由用户在终端手动执行。
- 数据可以分批录制，审核通过后再聚合训练。
- 未确认机械臂已上电、急停可用、工作区清空和人员处于安全位置前，不得执行运动命令。
- 优先运行相机预检和只读 `read_pose.py`；诊断问题时不要随机执行运动脚本。
- `move_to_start.py`、回放和推理都会造成真实机械臂运动，必须由操作者明确启动并现场监护。
- 不得删除安全边界、最大单步位移限制或控制模式处理来“改善推理效果”。
- 相机断流时必须停止当前 episode、清空本轮缓存并完整重录，不能在同一 episode 中途续接。
- 不要自动删除或覆盖数据集、模型、评估结果和 GPU 机上的未提交文件。
- 每次新录制使用新的 `bamboo_newview_sXX`；每次训练使用新的 `MODEL_PATH`；每次评估使用
  新的 `EVAL_DATASET_PATH`。
- 修改代码前先检查工作树和影响范围。禁止使用 `git reset --hard` 清理工作站或 GPU 机。
- 数据聚合会重建目标 `bamboo_newview_full`，执行前必须确认源批次列表和目标路径。

## 相机定义

- `handeye`：当前的眼相机，物理安装方式为眼在手外。不要根据变量名误判成眼在手上。
- `fixed`：另一台近景相机。
- 当前数据集配置为两路 640×480 视频、25 FPS。
- `fixed` 曾偶发读取失败或产生损坏的 MJPEG 帧。USB 线、接头和转接头已重新插拔，之后完成本批录制且没有再次报错，但后续录制前仍应执行双相机预检。
- 不要擅自把 `fixed` 改成 20 FPS；当前保持 MJPG 25 FPS。
- 两台相机均使用 `/dev/v4l/by-id/` 稳定路径，不能改回易重编号的 `/dev/videoN`。
- `handeye` 驱动配置为 MJPG 30 FPS，但硬件实测约 25 FPS；数据集控制频率仍为 25 FPS。
- `handeye`/`fixed` 键名与物理相机的对应关系必须在录制、训练、推理全过程保持一致。
- 相机位置、方向、焦距或曝光改变后，旧数据通常不能直接与新数据混合；必须先比较画面分布
  并重新建立基线。

## 开发与验证约定

- 使用项目虚拟环境 `.venv`，优先通过仓库内现有依赖完成任务，不随意修改系统 Python、
  CUDA、ROS 或全局环境。
- 修改前先阅读相关文件，确认数据流和影响范围；优先小范围、可回滚的增量修改。
- Python 代码使用 Ruff 进行检查和格式化；通用逻辑修改后应运行相关单元测试。
- 相机、机械臂、吸盘和真实运动测试不属于普通单元测试，不能在无人监护时自动执行。
- 诊断应遵循“现象 -> 假设 -> 只读验证 -> 最小修复 -> 再验证”，不要同时调整相机、
  数据、ACT 参数和控制器。
- 不要因训练 loss 下降就声称任务成功。必须结合独立验证 loss、未参与训练的位置和真实
  机器人推理结果判断。
- 旧视角、启发式夹爪、脚本化下降/提起和不兼容当前绝对末端 action 的旧逻辑不属于当前
  主流程；改造前基线可通过 Git 提交 `c4dfa2c` 回溯，不要随意重新接回。

## 远程训练注意事项

- GPU 机原仓库包含未提交实验文件和旧模型，应使用
  `/home/rentao/program/lerobot-aubo-pure-act` 隔离工作树。
- 禁止在 GPU 机执行 `git reset --hard`，也不要用 `rsync --delete` 覆盖其原仓库。
- 只同步审核并聚合后的 `bamboo_newview_full`，不要把旧视角数据混入。
- 每次实验使用独立模型目录，例如 `bamboo_newview_act_run01`、`run02`，避免 checkpoint
  相互覆盖。
- GPU 系统盘空间有限；训练前后应检查磁盘空间，不要无限保留重复 checkpoint。
- 代码同步前先确认 `pyc` 分支已经提交并推送；GPU 机通过隔离工作树获取对应提交。
- 长时间训练放在 `tmux` 中运行，但启动命令仍由用户手动执行。

## 已录制数据

数据集位于：

```text
/home/rentao/.cache/huggingface/lerobot/datasets/
```

截至 2026-08-02：

| 数据集 | Episodes | Frames | 时长 |
|---|---:|---:|---:|
| `bamboo_newview_s01` | 10 | 8,885 | 355.40 秒 |
| `bamboo_newview_s02` | 1 | 832 | 33.28 秒 |
| `bamboo_newview_s03` | 9 | 7,848 | 313.92 秒 |
| `bamboo_newview_s04` | 10 | 9,374 | 374.96 秒 |
| `bamboo_newview_full` | 30 | 26,939 | 1,077.56 秒（17.96 分钟） |

`bamboo_newview_full` 是当前 30 条基线聚合结果。下一轮恢复示教应使用新的
`bamboo_newview_s05_recovery`，并聚合到新的输出目录，不能覆盖该基线。

## 数据审核结果

### 结构与兼容性

- 三批数据均为 AUBO i10、25 FPS。
- `action` 为 8 维，`observation.state` 为 13 维。
- `handeye` 和 `fixed` 均为 480×640×3 视频特征。
- 三批数据的 FPS、机器人类型和 feature 定义一致，可以聚合。
- `s03` 包含连续的 episode 0～8，没有遗留的临时图片。

### Action 与夹具标签

- 所有 episode 的 action 均未发现 NaN 或 Inf。
- 每条轨迹均包含吸盘释放值 `0` 和吸取值 `100`。
- 每条轨迹的吸取状态约持续 13～15 秒。
- `s03` 有效运动比例约为 52%～61%。
- 最大单帧末端目标位移约为 4～5 mm，未发现异常跳变。
- 三批数据的机械臂起始位姿一致，这是录制前统一复位的预期结果。

### 视频完整性

- `s03` 两路相机的视频总帧数均与 7,848 条数据帧一致。
- 所有相关视频均已完整解码，未发现解码错误。
- 抽检未发现黑帧或超过 0.5 秒的明显冻结。
- 本批录制中的重录轮次没有混入最终保存的数据。
- `handeye` 眼相机能够看到机械臂、竹条和目标区域。
- `fixed` 近景画面能够持续看到夹具与竹条的接触过程。

### 数据多样性

`s03` 轨迹结束位置的范围约为：

- X：3.4 cm
- Y：5.5 cm
- Z：1.6 cm

轨迹不是完全机械复制，但从眼相机抽样画面看，竹条初始摆放的变化仍然偏小。

## 推理效果差时的排查顺序

纯 ACT 推理不符合预期不一定只是模型“没学会”，应按以下顺序隔离问题：

1. **输入一致性**：确认推理时 `handeye/fixed` 没有接反，分辨率、25 FPS、相机位置、
   画面方向和训练时一致。
2. **数据与标签**：确认 action 是当前绝对末端定义，图像、状态、动作时间对齐，吸盘只有
   持续的 `0/100`，没有黑帧、断流拼接、抓空或失败示教。
3. **训练行为**：同时查看训练 loss 和按完整 episode 划分的验证 loss。训练 loss 很低但
   验证 loss 不降，通常表示过拟合或数据变化不足。
4. **预处理与后处理**：确认推理加载了训练 checkpoint 自带的 processor 和数据统计，
   action 维度、名称、归一化和反归一化没有错位。
5. **时序配置**：确认训练与推理使用相同的 25 Hz 物理时间定义，检查 action queue、
   `chunk_size`、`n_action_steps` 和策略 reset 时机。
6. **网络与实时性**：拆分推理时记录 GPU 推理耗时、Tailscale 往返延迟和工作站实际控制
   周期；延迟抖动会让正确动作在错误时刻执行。
7. **机器人执行层**：检查 `servoCartesian` 返回值、伺服模式、旋转向量连续性、工作空间
   裁剪和最大步长限制，区分“模型输出错误”与“机器人没有正确执行输出”。
8. **外部因素**：检查吸盘真空、竹条表面与重量、机械臂复位误差、相机曝光、光照、遮挡、
   桌面和目标位置变化。

评估泛化时，应保留少量未用于训练的竹条位置与角度作为测试条件。不能一边把同一位置的
全部数据用于训练，一边用该位置的成功率证明模型具有泛化能力。

## 当前结论

- 本节是 20 条阶段的历史结论。`s04` 已补录 10 条，当前 30 条基线和真实推理结论以紧随其后的
  “2026-08-02 当前状态” 为准。

## 下一步

历史步骤已完成；当前下一步见下方“2026-08-02 当前状态与恢复训练计划”。

## 2026-08-02 当前状态与恢复训练计划

### 已完成工作

- `bamboo_newview_s01`～`s04` 已聚合为 `bamboo_newview_full`：30 episodes、26,939 frames、
  25 FPS；其中 `s04` 为新增的 10 条。
- GPU 隔离工作树
  `/home/rentao/program/lerobot-aubo-pure-act` 已完成纯 ACT 训练：
  `models/bamboo_newview_act_run05/best`，30,000 steps，最佳保留集 loss `0.05918`，最终
  validation loss `0.06058`。
- 首次真实推理保存为工作站数据集 `bamboo_newview_eval_run05_trial02`：1 episode、1,493
  frames、约 60 秒；`trial01` 是早期推理异常留下的空 episode，不能作为模型质量依据。
- 已修复 ACT 推理中 `action=None` 会误进入 VAE 编码器的问题；GPU 服务端的真实推理能够加载
  `run05/best` 并返回动作。
- 已新增 `examples/phone_to_auboi10/audit_act_offline.py`。它不连接硬件，模拟当前 ACT
  action queue，报告逐维误差、吸盘开关、抓取高度和跳变；GPU 上已生成 run05 验证集、反事实和
  `trial02` 回放报告。

### 已证实的推理结论

- 相机键名、视野、25 FPS、网络路径、模型目录和机器人执行链路均没有发现能解释失败的错配；
  `trial02` 离线回放会复现当时的模型输出。
- 在 6 条保留示教（episodes 24～29，5,678 frames）上，run05 的吸盘 precision `98.60%`、
  recall `98.64%`，抓取高度平均绝对误差约 `4.9 mm`。这些数值只表示模型在**专家已处于正确
  轨迹的观测**上的表现。
- 在真实 `trial02` 中，模型没有开启吸盘，且到抓取 XY 区时末端高度约 `0.1735 m`，而训练中
  首次吸取高度为 `0.0891–0.1001 m`。模型偏离后见到未覆盖的高位悬停状态，持续输出释放；这是
  行为克隆常见的闭环/分布偏移失败，不能仅由总 validation loss 发现。
- `trial02` 有 72 次单帧位置跳变超过 1 cm、6 次超过 3 cm；专家保留轨迹没有超过 3 cm 的
  跳变。机器人基本跟随模型命令，因此顿挫主要来自模型动作块的非连续输出，而不是底层伺服失效。

### 已完成但尚未在真实推理生效的软件改进

- `inference_server.py` 默认启用 ACT temporal ensembling，系数为 `0.01`；显式设置
  `TEMPORAL_ENSEMBLE_COEFF=0` 可做旧 action-queue 对照。
- `evaluate_split.py` 的通用单帧末端位移安全限幅默认收紧为 `MAX_EE_STEP_M=0.008`（8 mm），
  合法范围为 1～50 mm。它不使用竹条位置规则，只限制异常大跳变。
- 新服务端文件已同步到隔离 GPU 工作树，但当前运行中的服务未重启；下一次现场确认安全、明确
  重启服务后才会启用时间集成。不得为了平滑而删除工作空间、单步限幅或控制模式注入。

### 下一阶段：恢复示教与独立重训

目标不是在推理 episode 中把手机动作混入模型输出，而是建立独立恢复示教：从“吸盘释放、末端
悬在竹条上方约 6～10 cm”的偏离状态出发，手机连续完成下降、吸取、抬升、搬运、放置、释放。

1. 现场操作者完成相机预检、`read_pose.py`、急停和工作区检查。不要在无人看护时执行任何运动。
2. 以新名称 `bamboo_newview_s05_recovery` 录约 20 条。`record.py` 等待开始时不要按 `r`，
   以保留悬停恢复初态；开始后停 0.5～1 秒再连续完成完整任务后半段。
3. 逐条审核视频、吸盘标签和动作连续性。碰撞、抓空、相机断流、长时间犹豫的 episode 必须重录。
4. 用 `DATASET_SOURCES` 显式聚合 `s01`～`s04` 加 `s05_recovery`，输出为新的
   `bamboo_newview_full_recovery_v1`；不要覆盖 `bamboo_newview_full`。后缀 `_recovery` 不符合
   自动发现的 `sNN` 命名规则。
5. 在 GPU 隔离工作树将新数据同步后，从头训练新目录
   `bamboo_newview_act_run06_recovery`，保留 `run05` 用于对照。
6. 训练完成先运行 `audit_act_offline.py`，以吸盘首次开启、抓取高度、单帧跳变和恢复初态表现
   作为准入项；通过后再由现场操作者进行一次低风险真实推理测试。

具体命令、恢复录制逐步操作和聚合命令维护在
`examples/phone_to_auboi10/README.md` 的“当前基线与下一轮恢复训练”章节。

## 审计边界

已完成数据审核、30 条聚合、GPU run05 训练、真实推理回放和离线模型审计；所有真实机械臂
运动仍由现场操作者手动启动和监护。本文件记录这些阶段性结论，不替代每次操作前的相机预检、
只读位姿检查和安全确认。

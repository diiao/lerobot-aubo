# AUBO 主线清理记录

日期：2026-10-02。用户授权整理当前文档、退役独立旧功能及合并等效检查。
清理前完整代码与文档基线为 `69435fe`；文档提交为 `cc1194d`，旧功能退役提交为 `22b0062`。
原始数据、模型、配置哈希证据、录像与实验目录不删除，未访问远端或硬件。

## 文档

当前入口统一到 [README](../examples/phone_to_auboi10/README.md) 和采集/训练/运行三页。
六份被替代的交接或规划原文从基线提交查询；[证据索引](../examples/phone_to_auboi10/EVIDENCE.md)保留路径和最新结果。
根目录 AGENT/CLAUDE 与上游 README 指向当前入口，不再复制旧 ACT 操作流程。

验证：修改页面的本地链接通过；采集、训练、持续抓放的 plan 模式通过，未创建采集目录。

## 旧功能删除范围

删除8个入口、20个专属模块、24个专属测试文件。已核对全部跟踪文件的调用、配置和命令引用，
也检查了测试中的字符串动态导入。没有本机 Python 进程正在运行待删除入口。

### 入口（examples/phone_to_auboi10）

- `audit_act_offline.py`
- `evaluate.py`
- `evaluate_split.py`
- `inference_server.py`
- `train.py`
- `build_c0_pretraining_pack.py`
- `build_c0_review_pack.py`
- `geometry_preview.py`

### 专属模块（src/lerobot/bamboo_sorting）

- 旧ACT：`act_input_contract.py`。
- 旧C0离线审计与整理：`c0_gripper_auditor.py`、`c0_gripper_controller_binding.py`、`c0_gripper_dataset_auditor.py`、`c0_gripper_recording_session.py`、`c0_gripper_save_finalizer.py`、`c0_post_capture_review.py`、`c0_pretraining_gate.py`。
- 旧深度与几何预览：`depth_gate.py`、`depth_gate_analysis.py`、`geometry_perception.py`。
- 旧TCP推理、安全与原始清单：`authorization_contract.py`、`capture_manifest.py`、`offline_vla_runtime.py`、`policy_prediction.py`、`smolvla_action_decoder.py`、`smolvla_adapter.py`、`thin_safety_gate.py`。
- 旧RL契约：`rl_contract.py`、`rl_episode_contract.py`。

这些模块的测试文件 `tests/bamboo_sorting/test_<模块名>.py` 一并删除；另删除四个旧ACT入口测试：
`test_audit_act_offline.py`、`test_evaluate_split_safety.py`、`test_inference_server_contract.py`、`test_train_gripper_keyframes.py`。
它们验证退役接口，而非当前关节策略。通用 LeRobot ACT 策略和其他机器人功能不变。

根目录 `BAMBOO_GEOMETRY_ASSISTED_ACT_PLAN.md` 是已退役几何预览的旧ACT规划，随该组退出工作树；
原文同样保留在 `69435fe`，不再留下指向已删入口的现行计划。

包 `bamboo_sorting/__init__.py` 的旧批量导出同步移除。当前代码均显式导入所需子模块，
全仓没有从包根导入这些符号的调用；不再为一次关节模块导入连带加载旧推理栈。
保留的契约测试去掉对已退役 auditor 的导入断言。

## 保护与测试逐项判定

| 检查 | 防止的错误 | 等效性判断与处理 |
| --- | --- | --- |
| 完整动作块的形状、有限数值、原始/解码值及首步一致性 | 推理响应缺行、含NaN、吸盘离散化错误或首步与动作块矛盾 | `InferencePipe.predict` 与控制循环对同一个响应重复调用同一校验。集中到管道接收边界，返回规范化后的动作块；完整块分支内重复的首步有限值检查也由该校验覆盖。非完整块分支保留原检查。 |
| 三份模块导入隔离测试 | 纯数据模块意外引入硬件/模型依赖、后台线程、套接字或工作目录文件 | 原检查逻辑相同，合并为三个参数化用例；分别导入 contract、journal、sidecar。移除伪造父包，连真实 `__init__` 一起检查，仍保留三个模块的覆盖。 |
| 预测过期与相机时间差 | 使用旧观测预测的动作，或混用不同步画面 | 保留。它核验观测时间，不能由当前机械臂状态检查代替。 |
| 发送前状态新鲜度 | FK等计算耗时后仍依据旧关节状态判断跟踪误差 | 保留。它核验即将发送时的状态时间；混训路径最多刷新一次，刷新后仍过期则停止。 |
| 无新预测看门狗、伺服发送间隔与吸盘切换到位期限 | 分别防止推理断供、伺服断流、切换目标一直不到位 | 保留。计时起点、被保护对象和触发条件不同，不按“超时”名称合并。 |
| 目标关节/工作区限制、轨迹速度/加速度与跟踪误差 | 分别防止目标不可执行、中间轨迹过快、指令与实测关节偏差过大 | 保留。模型目标通过限制不代表每个伺服中间点都能安全发送。 |
| 伺服所有权、吸盘DO读回、异常停止 | 接管其他控制器、IO指令状态不符、异常后继续发送 | 保留。旧TCP安全模块的退役不影响当前关节执行检查。 |

动作块校验实现及其单元测试不变；管道拒绝第10行解码不一致的既有测试仍保留。
合并没有修改模型动作取步、关节轨迹、周期计数、IO时机或任一时间阈值。

## 明确保留

- `record_joint.py` → `record_c0_batch.py` → `record.py`：当前录制复用新鲜观测读取和原手机归位。
- `aubo_joint_capture.py` → `c0_batch_capture`、`c0_gripper_capture_journal` 等：当前帧标签和指令侧录。
- `run_joint_trial.py` → `c0_smoke_capture.frozen_c0_smoke_camera_mapping`：当前相机映射。
- `contracts`、`observation_contract`、`lerobot_bridge`、`rgb_gate`：第一轮曾整体保留；第二轮细查定义后退役前三者，`rgb_gate` 保留配置读取，见下文。
- `run_joint_live.py`、`run_joint_trial.py`：仍提供任务输入、推理管道、设备读取、归位及诊断。
- `layer_order_eval.py` 和测试：下一阶段两三根静态层序评价。
- 三份 `configs/aubo_i10/act_*.json`：历史划分证据，数据清单和保留的批次审计测试仍引用。
- CameraSetV1/V2及勘误、ACT批次清单、审计摘要与全部原始实验产物。

残留旧命名不等于独立旧功能；本轮不为去除前缀而重构共享采集/驱动。

## 验证与回滚

清理前专项基线：1138项通过、1项跳过，1项数据往返测试因沙箱缓存锁只读失败；
该项在获准访问本地缓存后单独重跑通过。未发生相机、AUBO或远端模型调用。

退役后回归：699项通过、6项跳过，覆盖保留的竹条模块、AUBO驱动、处理器、录制观察器、批次审计和SmolVLA动作损失。
测试使用假设备；缓存变量只作用于该次测试进程，路径在 `/tmp`，不修改系统环境。

```bash
PYTHONPATH=src HF_DATASETS_CACHE=/tmp/aubo-cleanup-dataset-cache-20261002 .venv/bin/python -m pytest -q \
  tests/bamboo_sorting \
  tests/robots/test_aubo_gripper_io.py tests/robots/test_aubo_i10_fail_fast.py \
  tests/processor/test_aubo_gripper_and_safety.py tests/processor/test_aubo_lock_vertical_yaw.py \
  tests/processor/test_smolvla_processor.py tests/scripts/test_lerobot_record_observer.py \
  tests/datasets/test_aubo_batch_validation.py tests/datasets/test_audit_recorded_batch.py \
  tests/policies/smolvla/test_action_padding.py tests/policies/smolvla/test_action_loss.py \
  --tb=short --show-capture=no
```

删除后再次通过采集、训练和运行的plan模式；采集目录未创建。
全仓剩余Python文件语法检查及已删模块的导入引用检查通过，`git diff --check` 通过。
重复检查合并后，以下专项回归304项全部通过；其中三个导入用例覆盖真实包初始化。
完整动作块运行计划与合并前逐字段相同，包含所有现有速度、时效和周期参数。

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/bamboo_sorting/test_joint_trial.py tests/bamboo_sorting/test_joint_smooth.py \
  tests/bamboo_sorting/test_c0_gripper_contract.py \
  tests/bamboo_sorting/test_c0_gripper_capture_journal.py \
  tests/bamboo_sorting/test_c0_gripper_controller_sidecar.py \
  --tb=short --show-capture=no
```

以上是软件回归结果，未执行实机验收，也未重新训练或推理。

Git删除记录即精确清单，可用 `git show 69435fe:<原路径>` 只读回溯。
撤销清理时按相反顺序 revert 对应提交，并重新运行保留链路的回归；不使用 reset --hard。
Git不能恢复未跟踪或被忽略的原始数据。

## 第二轮：逐目录与定义检查

第二轮基线为 `1d384f1`。检查顶层目录用途、跟踪文件、导入及字符串命令引用，
继续删除藏在共享模块内部的旧定义；不以“测试多”或文件名含 `c0` 作为删除理由。

### 目录判定

| 目录或根文件 | 检查结论与处理 |
| --- | --- |
| `src/lerobot/bamboo_sorting/` | 删除旧末端/深度契约，精简共享模块；具体定义见下表。当前关节采集、训练、运行和层序评价保留。 |
| `src/lerobot/` 其他目录 | 数据集、处理器、策略工厂和机器人注册存在跨模块依赖；保留通用框架，未做全框架裁剪。 |
| `examples/phone_to_auboi10/` | 删除3个旧入口，`record_c0_batch.py` 只保留当前录制调用的两个辅助函数。当前入口及现场诊断工具保留。 |
| `examples/` 其他目录 | 是其他硬件/数据工具的独立示例；SO101 目录仍有遥操、标定及标定结果，不因当前使用手机就当作废文件。 |
| `tests/` | 删除退役定义的专属用例，保留当前行为检查；其他 LeRobot 组件仍有实现，对应测试保留。 |
| `docs/` | 首页改为项目导航；历史实验留证据入口，`source/` 保留与本仓库版本对应的上游技术参考。 |
| `configs/` | 两代相机配置、勘误和3份历史ACT划分配置仍有当前或证据用途，保持原样。 |
| `datasets/` | 15个真实录制目录，保留。 |
| `artifacts/` | 41个实验产物目录，保留；不是Git可恢复的普通代码。 |
| `log/`、`logs/` | 旧录制、运行与审计日志，共约24 MB；保留证据，不按扩展名删。 |
| `benchmarks/` | 视频编解码对比工具，有独立说明和依赖选项；保留，正常抓放不自动运行。 |
| `docker/` | 被 Makefile 和上游 CI 引用的环境构建文件，保留。 |
| `.github/` | 上游CI、文档与发布工作流；多项明确限制为 `huggingface/lerobot`，不属于本地伺服检查。保留。 |
| `.kilo/` | 移除未使用的 `worktrees/dot-record` 副本，约12 MB；保留工具配置和agent定义。 |
| `.pytest_cache/`、`.ruff_cache/` | 可再生的测试/格式检查缓存，回归结束后清除。它们不是测试源码，也不是实验数据。 |
| `.venv/` | 当前Python环境，保留，不卸载或重装依赖。 |
| `.vscode/`、`.claude/` | 本地编辑器权限/工具配置与未入Git的计划，保留；不把项目清理扩大为工具配置重置。 |
| `.agents/`、`.codex/`、`.aws/` | 当前为空目录，没有需要删除的代码；不改工具或凭据目录。 |
| `.git/` | 保留版本历史和回滚点；仅正常提交和注销已清洁旧工作树。 |
| `media/` | 4张上游README宣传图仅被旧根README本地引用，导航替换后删除。上游文档中的同名logo使用远程URL。 |
| 根目录文档和构建文件 | 根README改为AUBO入口；保留许可证、贡献/安全说明、包配置、依赖文件、Makefile及清单，未更改安装环境。 |

旧工作树删除前：HEAD为 `b90443083de877e9d5aaba247796cec265e1e721`，是当前基线的祖先；
无已修改、未跟踪或忽略文件，无工作目录位于其中的进程。可在需要时用
`git worktree add --detach .kilo/worktrees/dot-record b904430` 重建，不需要复制备份。

### 删除与保护判定

| 内容 | 原来防止或支持什么 | 处理与等效检查 |
| --- | --- | --- |
| `record_c0_smoke.py`、旧批次CLI | 旧13维状态/8维动作的单条试录、固定10条正式录制及8/2划分 | 该数据路线已退役。删旧入口和固定数量、放置参考确认、旧后检；保留 `read_fresh_c0_observation`、`_load_record_helpers`。当前七维录制使用自己的场景、结果与保存流程。 |
| `phase_a2_rgb_gate.py`、`freeze_camera_set_v1.py` | 早期三路RGB/Mech-Eye选择及一次性配置固化 | 相机选择已完成，删除旧评估/写入入口和专属测试；CameraSetV1/V2原文件、哈希及只读加载仍保留。 |
| `contracts.py`、`observation_contract.py`、`lerobot_bridge.py` | 旧任务语言注册、末端动作、13维状态、三相机/深度封装 | 核对后无当前调用，删除三个模块及其三个测试文件。七维契约仍由 `aubo_joint_contract` 检查；原0/100常量移入仍使用它的吸盘事件模块，数值不变。 |
| C0 episode/batch manifest、整段gripper trace/binding/finalized sidecar | 固定批次、旧数据契约一致性、旧整段摘要与绑定审计 | 调用链已退役，删除类、校验、辅助定义和专属测试。当前每帧时间戳、吸盘事件、指令attempt及其数据格式保持原样。 |
| 无调用的 `require_fixed_j5` 及两个常量 | 旧固定J5方案的角度约束 | 当前J5由模型预测，没有调用；删除死代码及测试文件中的闲置导入。当前关节范围/跟踪误差检查保留。 |
| 相机映射中的文件哈希及版本检查 | 拒绝被改动或错误版本的相机配置 | `load_camera_set_v2` 返回前已对同一个文件完成这两项，删除映射层重复检查。流角色、设备映射与帧率检查保留；两项负例验证错误版本及换设备后仍拒绝。 |
| 当前吸盘事件/attempt的一致性、摘要与拷贝检查 | 保存的数据标签与已发送指令不符，或采集后可变字典被改写 | 保留；构造时与序列化时面对的状态可能不同，不能视为同一次重复检查。 |
| 预测时效、发送前状态、轨迹边界与异常停止 | 分别约束观测、机器人实测状态、每步运动及故障退出 | 本轮未修改相关实现及阈值。 |

### 测试数字是什么意思

清理前 `tests/bamboo_sorting/` 有325个测试函数，pytest展开参数后收集566个用例。
例如同一函数测试3个字段、每个字段4种非法输入，会计作12项；这是输入组合，不是12个新功能或12次真机动作。
反复出现“几百项通过”是重复运行仓库已有检查，不代表每轮新增了几百项测试。

本轮删除205项退役功能用例，新增2项相机配置拒绝回归，竹条专项剩363项。
连同相关驱动、处理器、录制和数据边界检查，本轮一次回归结果为 **496通过、6跳过**；
4项需要CUDA/多GPU，2项需要本机未安装的transformers。未为本次清理安装依赖。
剩余测试不全部属于运动安全：还包括字段顺序、度/弧度转换、吸盘标签、录制结束、保存失败、训练划分与层序标注。

测试数量不能衡量实机成功率，也不能单独证明没有退化。验证使用既有假设备、合成数据和本机缓存，未连接硬件或远端。
沿用上文完整回归命令；后续只需按改动范围运行相应测试，不为提高通过数量重复执行。

最终核对：556个保留Python文件语法及竹条模块导入检查通过；修改的文档本地链接有效；
采集、训练、运行三个plan输出与清理前完全相同。`git diff --check` 通过。
当前录制/运行入口、驱动、吸盘采集journal、运行时限与相机配置原文件均无修改。
本轮可从 `1d384f1` 查阅所有被删的跟踪文件；缓存可重新生成，旧工作树按上文命令重建。

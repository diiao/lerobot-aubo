# AUBO 主线清理记录

日期：2026-10-02。用户授权整理当前文档、退役独立旧功能及合并等效检查。
清理前完整代码与文档基线为 `69435fe`；第一轮文档提交为 `cc1194d`。
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

## 明确保留

- `record_joint.py` → `record_c0_batch.py` → `record.py`：当前录制复用新鲜观测读取和原手机归位。
- `aubo_joint_capture.py` → `c0_batch_capture`、`c0_gripper_capture_journal` 等：当前帧标签和指令侧录。
- `run_joint_trial.py` → `c0_smoke_capture.frozen_c0_smoke_camera_mapping`：当前相机映射。
- `contracts`、`observation_contract`、`lerobot_bridge`、`rgb_gate`：上述共享链路的依赖。
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
以上是软件回归结果，未执行实机验收，也未重新训练或推理。

Git删除记录即精确清单，可用 `git show 69435fe:<原路径>` 只读回溯。
撤销清理时按相反顺序 revert 对应提交，并重新运行保留链路的回归；不使用 reset --hard。
Git不能恢复未跟踪或被忽略的原始数据。

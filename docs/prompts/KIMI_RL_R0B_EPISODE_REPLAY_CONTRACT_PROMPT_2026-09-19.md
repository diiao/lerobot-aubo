# 给 Kimi 的代码任务提示词：RL R0B Episode、Reset 与 Replay 契约

请把下面整段内容原样交给 Kimi。Kimi 完成后不要让它提交 Git；把补丁、测试输出和最终说明交回 Codex
审查。

---

你正在维护仓库 `/home/rentao/program/lerobot-aubo`，当前分支应为
`research/depth-vla-phase-a`，起始 HEAD 应包含：

```text
f5aa19f feat(rl): add offline R0 contracts and roadmap
```

本任务只实现 **RL R0B 的离线 episode、reset 和 replay 审计契约**。不训练模型、不创建 Gym/AUBO
环境、不连接相机或 AUBO、不移动机械臂、不执行夹爪 IO、不调用 actor/learner、不运行真实 reset，也不修改
LeRobot 的通用 `src/lerobot/rl/buffer.py`。

开始前必须阅读：

1. `AGENTS.md`（如果存在）和 `AGENT.md`；
2. `docs/AUBO_I10_SMOLVLA_RL_EXTENSION_PLAN_2026-09-19.md`；
3. `src/lerobot/bamboo_sorting/rl_contract.py`；
4. `src/lerobot/bamboo_sorting/authorization_contract.py`；
5. `src/lerobot/rl/buffer.py` 中 done/truncated、dataset 转换和 episode boundary 相关代码；
6. `tests/bamboo_sorting/test_rl_contract.py`。

先运行 `git status --short --branch`。工作区可能包含用户未提交改动，禁止删除、回滚、覆盖或格式化无关
文件。只允许：

- 新增 `src/lerobot/bamboo_sorting/rl_episode_contract.py`；
- 新增 `tests/bamboo_sorting/test_rl_episode_contract.py`；
- 最小修改 `src/lerobot/bamboo_sorting/__init__.py`，只导出本任务新增类型。

## 1. `ResetRecordV1`

实现 `frozen=True` 的纯数据记录，用来描述已经发生或失败的 reset，不得执行 reset。

至少包含：

- `schema_version`、`reset_id`、`target_episode_id`；
- `reset_reason`，只允许 `initial_setup`、`after_done`、`after_truncated`、
  `operator_requested`、`safety_abort_recovery`；
- `requested_monotonic_s`、`completed_monotonic_s: float | None`；
- `reset_completed: bool`、`human_verified: bool`；
- `pre_reset_observation_id: str | None`；
- `post_reset_observation: RLObservationRefV1 | None`；
- `evidence_ref`、`evidence_sha256`；
- `automatic_reset_authorized: bool`，本版本必须严格为 `False`，传入 `True` 必须拒绝。

不变式：

- 时间戳必须是有限非负数并拒绝 bool；
- reset 成功时必须有 `completed_monotonic_s`、`post_reset_observation`，完成时间必须晚于请求时间，且
  `human_verified=True`；
- reset 失败时完成时间和 post-reset observation 必须为空，`human_verified=False`；
- `initial_setup` 允许 `pre_reset_observation_id=None`，其他 reason 必须提供非空 pre-reset ID；
- evidence ref 非空，SHA-256 为 64 位小写十六进制；
- manifest 固定 `serialized_record_grants_live_authorization=false`、
  `hardware_access_performed_by_serialization=false`。

## 2. `RLEpisodeManifestV1`

实现一个冻结、已经结束的 episode 清单。至少包含：

- `schema_version`、`episode_id`、`instruction_id`；
- `start_reset: ResetRecordV1`；
- 非空 `transitions: tuple[RLTransitionV1, ...]`；
- `termination_reason`，只允许 `task_success`、`task_failure`、`time_limit`、
  `safety_abort`、`operator_abort`；
- `finalized: bool`，本版本只接受 `True`。

必须验证：

- start reset 已成功、人工确认，且 `target_episode_id == episode_id`；
- 所有 transition 的 `episode_id` 一致，`transition_id` 不重复；
- `step_index` 必须严格为 `0..N-1`，不得跳号、重复或乱序；
- 所有 transition 使用同一个 canonical instruction；
- 相邻 transition 必须首尾相接：前一条 `next_observation` 必须完整等于后一条 `observation`；
- 只有最后一条允许 `done=True` 或 `truncated=True`，最后一条必须且只能有一个为真；
- `task_success`：末步 `done=True` 且 `reward.task_success=1`；
- `task_failure`：末步 `done=True` 且 `reward.task_success=0`；
- `time_limit`、`safety_abort`、`operator_abort`：末步 `truncated=True` 且成功奖励为 0；
- `safety_abort` 还要求末步 `safety_cost>0`；
- manifest 不能产生硬件或策略执行授权。

不要规定场景 ID 必须始终相同；项目规则允许移动木/竹条后生成新的 `scene_id`。

## 3. `RLReplayManifestV1`

实现一个冻结 replay 数据集清单，不实现内存 replay buffer。至少包含：

- `schema_version`、`replay_id`；
- `source_dataset_ref`、`source_dataset_sha256`；
- `split_name`，只允许 `train`、`validation`、`test`；
- 非空 `episodes: tuple[RLEpisodeManifestV1, ...]`；
- `frozen: bool`，本版本只接受 `True`；
- `upstream_truncated_roundtrip_verified: bool`。

必须验证：

- episode ID 全局唯一；
- transition ID 全局唯一；
- 不同 episode 的 observation ID 不得重复；
- `episode_count`、`transition_count`、`scene_ids` 由内容计算，调用方不得伪造计数；
- `ready_for_training` 只有在 frozen 且
  `upstream_truncated_roundtrip_verified=True` 时才为真；
- 当前 `src/lerobot/rl/buffer.py` 仍有 truncation 未完整 round-trip 的代码路径，因此测试中的默认现实案例必须
  把该字段设为 `False`，并得到 blocker `upstream_truncated_roundtrip_unverified`；不得修改通用 replay buffer，
  也不得虚构已经通过；
- manifest 固定无真机授权字段，并返回 JSON 可序列化的深拷贝。

## 4. 通用验证与测试

- 所有 dataclass 使用 `frozen=True`；标识符必须为非空字符串；枚举字段先检查字符串类型，list/dict 等输入
  必须得到清晰 `ValueError`；数值拒绝 bool、NaN、Inf；SHA-256 严格验证；
- 复用现有 `RLObservationRefV1`、`RLTransitionV1`、canonical instruction contract，不复制已有定义；
- 不保存 numpy 图像、设备句柄、相机对象、RPC client 或回调；
- `to_manifest_record()` 修改返回值不得影响原对象；
- 测试至少覆盖 reset 成功/失败、非法时间顺序、自动 reset 授权拒绝、episode 链断裂、step 跳号、提前终止、
  五种 termination reason、重复 episode/transition/observation ID、replay 计数、truncation blocker、JSON 深拷贝；
- 增加 import 探针，证明导入新模块不会导入 `cv2`/`pyaubo_sdk`、创建 socket 或额外线程。

## 5. 工程边界

- 使用 `apply_patch` 修改文件；
- 不增加依赖，不修改 `pyproject.toml` 或 lockfile；
- 不修改 `rl_contract.py`、`authorization_contract.py`、`src/lerobot/rl/buffer.py`、机器人、相机、训练或执行代码；
- 不创建真实训练配置，不写任何真机许可开关；
- 不做 Git commit。

完成后运行：

```bash
.venv/bin/python -m pytest -q tests/bamboo_sorting/test_rl_episode_contract.py
.venv/bin/python -m pytest -q \
  tests/bamboo_sorting/test_authorization_contract.py \
  tests/bamboo_sorting/test_rl_contract.py \
  tests/bamboo_sorting/test_rl_episode_contract.py
.venv/bin/python -m pytest -q tests/bamboo_sorting
.venv/bin/python -m ruff check \
  src/lerobot/bamboo_sorting/rl_episode_contract.py \
  tests/bamboo_sorting/test_rl_episode_contract.py \
  src/lerobot/bamboo_sorting/__init__.py
git diff --check
git status --short
```

如果 venv 没有 ruff，可以使用已有缓存的 `uvx ruff`，但不得安装新依赖或联网。

最终说明必须列出：复用的现有契约、实际修改文件、完整测试结果、已知 blocker，以及未进行硬件、训练、
网络或 Git commit。不得把 `upstream_truncated_roundtrip_verified=False` 描述成“无未解决问题”。

---

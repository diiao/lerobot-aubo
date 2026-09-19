# 给 Kimi 的代码任务提示词：RL R0 离线契约与门禁

请把下面整段内容原样交给 Kimi。Kimi 完成后不要让它提交 Git；把补丁、测试输出和最终说明交回 Codex
审查。

---

你正在维护仓库 `/home/rentao/program/lerobot-aubo`，分支应为
`research/depth-vla-phase-a`。本任务只实现 **RL R0 离线契约与审计门禁**，不训练模型、不连接相机、
不连接 AUBO、不发送动作、不执行夹爪 IO、不运行 RL actor/learner，也不修改系统环境。

开始前必须阅读：

1. `AGENTS.md`（如果存在）和 `AGENT.md`；
2. `docs/AUBO_I10_DEPTH_ENHANCED_VLA_RESEARCH_PLAN_2026-09-08.md`；
3. `docs/AUBO_I10_SMOLVLA_RL_EXTENSION_PLAN_2026-09-19.md`；
4. `src/lerobot/bamboo_sorting/contracts.py`；
5. `src/lerobot/bamboo_sorting/authorization_contract.py`；
6. `src/lerobot/bamboo_sorting/policy_prediction.py`；
7. `src/lerobot/bamboo_sorting/thin_safety_gate.py`；
8. `src/lerobot/bamboo_sorting/offline_vla_runtime.py`。

先运行 `git status --short --branch`。工作区可能包含用户未提交改动，禁止删除、回滚、覆盖或顺手格式化
无关文件。只允许新增或修改以下文件：

- `src/lerobot/bamboo_sorting/rl_contract.py`（新增）；
- `tests/bamboo_sorting/test_rl_contract.py`（新增）；
- `src/lerobot/bamboo_sorting/__init__.py`（仅为导出新增类型所需的最小修改）。

实现目标：

1. 定义不可变、纯离线、JSON 可序列化的 `RewardSchemaV1`：
   - `task_success` 只能是 `0` 或 `1`；
   - `uncertain=true` 时禁止 `task_success=1`；
   - `safety_cost` 必须是有限非负数；
   - `intervention` 为 bool；
   - 明确区分 task reward、safety cost、人工接管，不得互相覆盖；
   - 包含 `observation_id`、`scene_id`、`instruction_id`、`reward_source`、
     `reward_model_ref`/`reward_model_sha256`（人工奖励时两者必须同时为空，学习奖励时必须同时存在）；
   - 学习奖励的 SHA-256 必须是 64 位小写十六进制；
   - 必须复用现有 canonical instruction contract，不复制一套指令表。
2. 定义 `RLTransitionV1`：
   - 绑定当前 observation、实际执行的 8D `ActionSchemaV1`、next observation、reward、`done`、
     `truncated`、`control_source`、`action_gate_passed` 和 `action_execution_confirmed`；
   - `control_source` 只能是 `policy`、`intervention` 或 `demonstration`；
   - `action_gate_passed` 类型为 `bool | None`：策略动作必须为 `True`，人工接管/历史示教可以为 `None`，
     但不得把 `False` 的候选动作写成已执行 transition；
   - 所有来源都必须 `action_execution_confirmed=True`，否则拒绝构造 transition；
   - `done` 与 `truncated` 必须是 bool，且给出明确的不变式；
   - 结构中不得出现设备句柄、相机对象、RPC client 或可执行回调。
3. 定义 `RLReadinessReportV1`：
   - 支持 `rabc`、`hil_serl_sac`、`smolvla_rl` 三种 mode；
   - 输入为显式布尔证据字段，不得探测硬件；
   - 输出 `ready` 和稳定排序的 `blockers`；
   - `rabc` 至少要求冻结数据 split、冻结 reward schema、独立 reward 测试集和普通 BC baseline；
   - `hil_serl_sac` 还要求 AUBO env adapter、reset、intervention、replay、actor–learner、ThinSafetyGate
     集成和仿真/空执行器通过；
   - `smolvla_rl` 还要求 chunk critic、behavior/KL constraint、offline policy evaluation、语言反事实测试；
   - 无论 mode，为真都不得产生任何真机授权字段；manifest 中固定
     `policy_execution_authorized=false`、`hardware_access_performed=false`。
4. 所有 dataclass 必须 `frozen=True`，严格拒绝 bool 伪装成数值、NaN/Inf、空标识符和未知 mode。
5. 提供 `to_manifest_record()`，返回可由 `json.dumps(..., allow_nan=False)` 序列化的深拷贝；调用方修改
   返回值不得改变原对象。

测试要求：

- 正常人工奖励、正常学习奖励；
- uncertain 正奖励拒绝；
- 非法 SHA、NaN/负 safety cost、bool 数值拒绝；
- policy gate fail、任意来源 execution unconfirmed transition 拒绝，并覆盖人工接管/示教的 `None` gate；
- 三种 mode 的最小通过案例；
- 每个缺失 gate 都出现在 blockers；
- blocker 顺序稳定；
- manifest JSON 可序列化且修改返回值不影响原对象；
- 证明模块 import 不会导入 AUBO SDK、OpenCV 相机或启动网络/线程。

工程约束：

- 使用 `apply_patch` 修改文件；
- 不增加依赖，不修改 `pyproject.toml`/lockfile；
- 不修改 ACT、SmolVLA、SAC、机器人、相机或执行代码；
- 不创建真实训练配置，不写任何“临时允许真机”的开关；
- 英文代码标识符，必要注释可用英文；最终说明用中文；
- 不提交 Git。

完成后依次运行：

```bash
.venv/bin/python -m pytest -q tests/bamboo_sorting/test_rl_contract.py
.venv/bin/python -m pytest -q \
  tests/bamboo_sorting/test_authorization_contract.py \
  tests/bamboo_sorting/test_policy_prediction.py \
  tests/bamboo_sorting/test_offline_vla_runtime.py \
  tests/bamboo_sorting/test_rl_contract.py
.venv/bin/python -m ruff check \
  src/lerobot/bamboo_sorting/rl_contract.py \
  tests/bamboo_sorting/test_rl_contract.py \
  src/lerobot/bamboo_sorting/__init__.py
git diff --check
git status --short
```

最终交付：

1. 先解释现有契约如何被复用；
2. 列出实际修改文件；
3. 粘贴完整测试结果；
4. 明确说明没有进行硬件、网络、训练或 Git commit；
5. 报告任何未解决问题，不得用“应该正常”代替验证。

---

# AUBO i10 SmolVLA 奖励学习与 RL 扩展计划

> 版本：`RLExtensionGateV1`
>
> 日期：2026-09-19
>
> 状态：计划已建立；不构成训练、设备连接、真机 rollout、自动 reset、机械臂运动或夹爪 IO 授权。

## 1. 结论

当前课题可以采用强化学习，但推荐顺序不是“从零让机械臂试错”，而是：

```text
语言条件 SmolVLA 示范学习基线
-> 奖励/进度模型 shadow 验证
-> RA-BC 奖励对齐模仿学习
-> 仿真或离线 RL
-> 可选的人机协同真机 RL
-> 可选的 SmolVLA-RL 研究扩展
```

原因是当前任务使用真实 AUBO i10、绝对末端动作和气动夹爪。在线探索的错误动作会产生硬件、工件和
人员风险，而且真实 reset 成本高。已有示教应先用于建立可靠的 SmolVLA 初始策略和奖励模型，再讨论
RL 带来的增量价值。

## 2. 三种容易混淆的方法

| 方法 | 是否属于 RL | 当前代码状态 | 本课题角色 |
|---|---|---|---|
| SARM + RA-BC | 否；仍是加权行为克隆 | SmolVLA 已支持逐样本 loss，训练脚本已有 RA-BC 权重接口 | 近期首选奖励学习消融 |
| HIL-SERL/SAC | 是；off-policy actor–critic 在线学习 | LeRobot 有 replay、actor–learner、奖励分类器和人工接管实现 | 后期独立 RL 对照 |
| SmolVLA-RL | 是；直接对语言条件动作块策略做强化微调 | 当前没有完整开箱即用通路 | 可选研究扩展 |

Hugging Face 官方 HIL-SERL 指南说明，其流程由少量示教、奖励分类器、SAC actor–learner 和人工接管组成：
<https://huggingface.co/docs/lerobot/hilserl>。官方 SARM 文档把 RA-BC 定义为根据预测任务进展对行为克隆
样本加权：<https://huggingface.co/docs/lerobot/sarm>。LeRobot 仍在建设通用 VLA-RL 微调接口，因此不能
把现有 HIL-SERL 等同于 SmolVLA-RL：<https://github.com/huggingface/lerobot/issues/3076>。

## 3. 推荐路线

### 3.1 近期：SARM + RA-BC

使用 C0–C3 的语言条件示教训练阶段/进度奖励模型。推荐先采用可审计的阶段标签：

```text
observe -> approach -> grasp -> lift -> transfer -> place -> verify
```

奖励模型先以 shadow 方式运行，只生成离线进度和置信度，不控制机器人、不终止 episode。验证通过后，
计算每个训练样本跨一个 action chunk 的进度增量，用它加权 SmolVLA 的逐样本损失。普通 BC 与 RA-BC
必须使用相同初始化、数据、split、训练步数和随机种子。

当前仓库的 `SARMConfig` 只有一个 `image_key`，并非现成的两路 RGB 融合奖励模型。首轮必须在开发集上
冻结使用 `global_rgb` 或 `grasp_rgb`，并在论文中如实说明；若以后扩展到多视角 SARM，必须建立新版本
和受控消融，不能仅改配置名称后沿用单视角结论。

### 3.2 后期：HIL-SERL/SAC 独立 RL 对照

LeRobot 的 HIL-SERL 是独立 SAC 策略，不继承 SmolVLA 的语言动作块结构。若实施，需要新增：

- AUBO Gymnasium 环境和显式 reset；
- AUBO 绝对 8D 动作与 HIL-SERL 增量末端动作之间的版本化适配；
- `global_rgb`、`grasp_rgb`、13D 状态和夹爪离散动作的 observation/action space；
- 手机或键盘人工接管语义，接管事件与实际下发动作的可审计记录；
- reward、done、truncated、safety cost 和安全门拒绝的严格区分；
- actor–learner 断连、旧参数、旧观测和 reset 失败时的 fail-closed 行为。

完成仿真和空执行器验收前，不得让 RL actor 连接真实 AUBO。

### 3.3 可选：SmolVLA-RL

该方向最符合“具身 VLA 主线”，但工程和研究难度最高。最低需要：

- 针对 action chunk 的 critic/value 定义；
- 语言、两路 RGB、本体状态与 8D 动作块一致的 replay schema；
- 防止策略偏离示教分布的行为约束或 KL 约束；
- offline policy evaluation、反事实语言测试和 reward hacking 检查；
- 策略更新、checkpoint、奖励模型和数据版本的完整 provenance；
- `ThinSafetyGate` 独立于 reward/critic，且不被训练过程修改。

在上述接口存在之前，只能写成“计划探索 SmolVLA 的 RL 微调”，不能写成“已经使用 RL 训练 SmolVLA”。

## 4. RewardSchemaV1 原则

首版任务奖励建议保持稀疏且可验证：

- `task_success=1`：木/竹条被抓起并稳定放入指令指定区域，经独立视觉和人工标签确认；
- `task_success=0`：未完成、抓空、掉落、放错区域或超时；
- `uncertain=true`：证据不足，不作为正奖励，进入人工复核；
- `safety_cost>0`：越界、超步长/速度、过期、非有限、IK 失败、安全门拒绝、人工急停等；
- `intervention=true`：操作者接管，记录接管前策略动作、最终执行动作和持续区间。

任务奖励与 safety cost 必须是不同字段。禁止用“接近木条”“吸盘打开”“到达低高度”等容易被策略
投机的单一内部信号直接判定成功。任何学习型奖励都必须在独立人工标注测试集上验证，并在正式测试前
冻结模型、阈值和裁剪方式。

## 5. 阶段门

| 门 | 必须满足 | 失败处理 |
|---|---|---|
| R0 契约 | reward/done/reset/intervention/replay/authorization 均版本化 | 不写训练代码 |
| R1 奖励 shadow | 独立人工标签集、阈值冻结、误报与校准可接受 | 保持人工奖励 |
| R2 RA-BC | 权重分布合理、关键阶段未被清零、离线验证优于普通 BC | 回退普通 BC |
| R3 仿真/空执行器 | actor–learner、接管、reset、安全拒绝和断连均 fail-closed | 禁止真机 RL |
| R4 真机 HIL-RL | 单独现场授权、低速、短时、人工可立即接管、全量日志 | 任一异常立即撤权 |
| R5 SmolVLA-RL | critic/replay/约束/OPE/语言反事实/安全隔离均通过 | 保留为未实现研究方向 |

## 6. 与主线的关系

- C0/C1 SmolVLA 数据采集与行为克隆基线优先；RL 不得提前抢占数据和硬件资源；
- ACT 已闭环，不因引入 RL 而恢复 ACT 训练；
- HIL-SERL/SAC 若实施，只是独立 RL 对照，不取代 SmolVLA；
- 只有直接更新 SmolVLA 且保留语言输入的方案才能称为 `SmolVLA-RL`；
- 没有严格同条件对照时，不得声称 RL 优于普通 SmolVLA；
- 所有真机 RL 结果仍需人工任务真值和 Wilson/配对置信区间，不能只报告 return 曲线。

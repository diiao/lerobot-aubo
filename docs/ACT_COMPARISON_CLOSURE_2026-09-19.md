# ACT 对照实验闭环决策记录

> 决策版本：`ACTComparisonClosureV1`
>
> 决策日期：2026-09-19
>
> 适用项目：AUBO i10 木/竹条抓放研究
>
> 本文是研究路线决策记录，不构成采集、训练、策略执行、机械臂运动或夹爪 IO 授权。

## 1. 决策

ACT 前期实验对照正式结束。后续不再追加 ACT 的双根训练、恢复示教、超参数搜索或真机评估，研究资源
回归语言条件的多视 RGB SmolVLA 主线。

原有 ACT 数据集、checkpoint、代码、日志、审计报告和真机记录全部保留，不删除、不覆盖。它们的角色
从“待继续优化的候选主线”变更为“已完成的历史系统对照与有效负结果”。

## 2. 支持该决策的证据

前期工作已经覆盖：

- 从 30/60 条早期示教到 65 条多角度数据，再扩展到 137 条示教的多轮数据建设；
- 13D 全状态、夹爪状态干预及 12D `drop_gripper` 等输入对照；
- teacher-forced 离线审计、动作块/夹爪时序检查和独立安全门诊断；
- 受控真机闭环中既出现过已见条件下的成功，也反复出现换角度后悬停、无下降动作、夹爪时序不稳和
  偏离示教状态后难以恢复等问题。

这些证据说明，问题不能简单归因于相机断流、第一步安全门拒绝或单一夹爪阈值。ACT 能在有限条件下
完成任务，但对当前课题要求的无序摆放、遮挡、角度变化和闭环分布偏移缺乏稳定鲁棒性。继续增加同类
示教或部署端启发式，会显著增加数据和调参成本，也会削弱“模型直接产生任务动作”的实验边界。

## 3. 允许与禁止的论文表述

允许表述：

> 在本课题当前任务定义、数据规模、两路视觉、本体状态、绝对末端动作和受控部署条件下，ACT 能在
> 部分已见场景完成抓放，但对场景变化和闭环偏离状态敏感，未达到后续复杂无序抓取研究所需的稳定
> 泛化能力，因此不再作为主线。

禁止表述：

- “ACT 模型普遍不适合机器人抓取”或“已经证明 ACT 无效”；
- 把单次成功写成稳定成功率；
- 把 teacher-forced 离线指标写成闭环性能；
- 在没有同数据、同场景、同动作契约的配对实验时，声称 SmolVLA 统计显著优于 ACT；
- 把人为加入的下降、选点或夹爪启发式成功归因于 ACT 本身。

## 4. 对 SmolVLA 主线的影响

后续唯一主线为：

```text
CameraSetV2 {global_rgb, grasp_rgb}
+ canonical English instruction_text
+ 13D proprioception
-> SmolVLA action chunk
-> reject-only ThinSafetyGate
-> AUBO execution under separate authorization
```

SmolVLA 微调仍属于示范学习，不是“抓对给奖励、抓错不给分”的强化学习流程。正式示教应以高质量成功
轨迹为主，并显式保存自然发生的失败、人工干预和结果标签；除非以后单独建立奖励模型或离线强化学习
协议，否则不得把失败轨迹简单解释成零奖励样本混入训练。

接下来的顺序是：

1. 完成真实 SmolVLA checkpoint 的 Phase B 离线前向、动作契约、语言输入和安全门审计；
2. 保持 `policy_execution_authorized=false`，确认离线链路不能触达执行器；
3. 由现场操作者另行明确授权后，执行 C0 的 10 条单根语言条件试采；
4. C0 审计通过后，才进入 C1 数据补齐和短步数 SmolVLA smoke run。

## 5. 证据索引

- `docs/robot_arm_technical_documentation.md`：run06 数据、离线审计和单次端到端成功的结论边界；
- `examples/phone_to_auboi10/HANDOFF_12D_POSE_IK_FIX.md`：12D ACT 多角度真机闭环结果与失败定位；
- `examples/phone_to_auboi10/datasets/bamboo_act_report_manifest.json`：扩展示教批次与 137 条聚合数据清单；
- `examples/phone_to_auboi10/reports/`：各扩展批次的只读审计摘要；
- Git 提交 `4ac1090`、`3de0e26`：夹爪时序修复、ACT 扩展检查点和 SmolVLA 适配起点。

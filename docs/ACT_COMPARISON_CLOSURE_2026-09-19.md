# ACT 对照实验结论（历史）

决策日期：2026-09-19。当前操作入口见[七维 SmolVLA 工作流](../examples/phone_to_auboi10/README.md)。
本文保留对照结论，不提供 ACT 或旧 C0 的执行命令。

## 已得结论

ACT 前期对照覆盖了早期30/60条、多角度65条和扩展137条示教，以及完整状态、移除夹爪状态、
动作块和闭环恢复等试验。在部分已见场景成功，但在角度变化和闭环偏离后仍出现悬停、
无下降动作或夹爪时序不稳，因此不再作为后续主线。

该结论只适用于本课题当时的数据、观测、动作契约和部署条件；不表示 ACT 普遍无效，
也不构成 SmolVLA 在同数据对照中统计显著更优的证据。
离线误差、指令周期和单次成功不能替代真实泛化成功率。

## 回溯

2026-10-02 用户授权清理已停用入口及专属代码、测试和重复文档。
原始数据、模型、审计摘要、录像不在本轮删除范围内；历史代码和完整决策原文从清理前提交
`69435fe` 查阅，详见[证据与历史索引](../examples/phone_to_auboi10/EVIDENCE.md)。

- [run06 历史技术记录](robot_arm_technical_documentation.md)：离线审计及一次现场成功。
- `git show 69435fe:examples/phone_to_auboi10/HANDOFF_12D_POSE_IK_FIX.md`：12D 实机失败定位。
- `examples/phone_to_auboi10/datasets/bamboo_act_report_manifest.json`：扩展批次来源。
- `examples/phone_to_auboi10/reports/`：保留的只读审计摘要。
- Git `4ac1090`、`3de0e26`：历史夹爪时序和适配起点。

旧决策中的13维状态、8维TCP动作及 ThinSafetyGate 不属于当前七维关节执行链路。

# 证据与历史索引

更新：2026-10-02。操作命令以 [README](README.md) 及其三个工作流页面为准。
本文只索引证据，不把历史部署记录当作远端当前状态。

## 当前双摆放模型

| 项目 | 记录 |
| --- | --- |
| 选项 | `mixed-both-orders` |
| 契约 | CameraSetV2，双 RGB，7维绝对关节＋吸盘，J5参与预测 |
| 训练 / 验证 | 单根60/12，上45°下90°50/5，上90°下45°50/5；合计160/22 |
| 训练参数 | 30000步，batch=8，seed=42 |
| GPU 项目 | `/home/rentao/program/lerobot-aubo-smolvla-joint-both-orders-20260928` |
| 检查点（相对 GPU 项目） | `outputs/joint_mixed_single60_top45_50_top90_50_30k_run01/final` |
| 权重 SHA-256 | `c24b61b8d902d395c88890a3243879507d75333d5deb0bc99c13572ecddbfc3b` |
| CameraSetV2 SHA-256 | `20de7adfd6ed9734c3a4329d86ab7e0ad08df9c6758d878d3c1302e2ac6423b4` |
| 完整动作块远端脚本 | `artifacts/joint_age_aligned_approach_20260928/joint_inference_stdio.py` |
| 该脚本部署时 SHA-256 | `eea4a8fcd910632ea382bac57ba55cd5b8550a4eb0842df1f24e77db8ee082c8` |

本机产物位于 `artifacts/aubo_joint_both_orders_20260928/`：

- [sources.json](../../artifacts/aubo_joint_both_orders_20260928/sources.json)：本机11个训练来源、4个验证来源。
- [离线验收](../../artifacts/aubo_joint_both_orders_20260928/offline_evaluation_run02/RESULTS.md)：对齐、指标复算、权重一致性；不证明物理成功或泛化。
- [最新三轮结果](../../artifacts/aubo_joint_both_orders_20260928/three_scene_review_20260928/RESULTS.md)：场景标签、指令周期和录像取帧。
- [闭合前反馈修正](../../artifacts/aubo_joint_both_orders_20260928/preio_replan_review_20260928/RESULTS.md)：旧版本回归失败与修正证据。
- `age_aligned_approach_candidate_20260928/`：动作取步方案与部署记录。
- `onsite_second_grasp_review_20260928/`、`second_contact_candidate_20260928/`：旧第二根失败诊断。后者的高度补偿补丁已被拒绝，不能重新应用。
- `full_chunk_replay_run01/`：保存观测的离线推理，已有结果无需因文档清理重跑。

## 最新三轮现场记录

以下目录均在 `artifacts/`，虽然文件名都有 both_top90，场景以操作者登记为准：

| 目录 | 实际场景与现场反馈 |
| --- | --- |
| `joint_preclose_feedback_both_top90_20260928_181639_883317523` | 上90°下45°，两根抓放成功 |
| `joint_preclose_feedback_both_top90_20260928_182412_271500264` | 上45°下90°，两根抓放成功 |
| `joint_preclose_feedback_both_top90_20260928_182818_033249696` | 单根90°抓放成功，但继续第二轮 |

单根约74.7秒首次释放、140.2秒再次吸附、224.1秒再次释放后退出。
入口固定两周期，没有硬编码第二轮轨迹；这些结果不是未见位置或普遍层序理解的成功率。

## 历史原文与证据

清理前代码和文档基线为 `69435fe`。下列已替代文档从日常目录移除，原文完整保留于该提交：

- `AGENT.md`、`CLAUDE.md`、本目录旧 README：历史 ACT 操作与工具背景。
- `examples/phone_to_auboi10/HANDOFF_12D_POSE_IK_FIX.md`：12维 ACT 实机失败定位。
- `examples/phone_to_auboi10/JOINT_CHAIN_REVIEW.md`：固定 J5 / 早期七维准备。
- `examples/phone_to_auboi10/REMOTE_TRAINING.md`：旧 ACT 远端入口。
- `docs/AUBO_I10_DEPTH_ENHANCED_VLA_RESEARCH_PLAN_2026-09-08.md`：深度和旧 C0 路线的决策过程。
- `docs/AUBO_I10_SMOLVLA_RL_EXTENSION_PLAN_2026-09-19.md`：旧8维 TCP 契约的 RL 规划。
- `docs/SMOLVLA_BAMBOO_EXPERIMENT_REPORT_2026-09-23.md`：60条单根、20k训练及早期现场汇报。
- 本目录旧 `JOINT_TRIAL.md`：逐次时序调试、首步诊断及归位变更过程。

只读查阅示例：

```bash
git show 69435fe:examples/phone_to_auboi10/JOINT_TRIAL.md
git show 69435fe:docs/SMOLVLA_BAMBOO_EXPERIMENT_REPORT_2026-09-23.md
```

仍保留的历史索引：

- [ACT 对照结论](../../docs/ACT_COMPARISON_CLOSURE_2026-09-19.md)。
- [run06 历史技术记录](../../docs/robot_arm_technical_documentation.md)。
- [2026-09-27 本机数据删除记录](../../docs/AUBO_I10_LOCAL_DATA_CLEANUP_2026-09-27.md)：不是当前160/22数据清单。
- `artifacts/run06_recovery/` 及其 `MANIFEST.sha256`：历史备份位置，本次未重新核验内容。
- `examples/phone_to_auboi10/datasets/bamboo_act_report_manifest.json`、
  `examples/phone_to_auboi10/reports/`：历史批次与审计摘要，保留原样。

Git 仅恢复已跟踪的代码和文档。`artifacts/`、数据、模型和录像多数被忽略，须另行备份；
已于历史清理中删除的原始数据不能通过该提交恢复。

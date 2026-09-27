# AUBO i10 本地数据清理与现行数据索引

日期：2026-09-27。本文记录本机清理后的文件状态；旧实验文档中的路径和“当前”表述应按其原始日期阅读。

## 现行七维关节混训数据

当前单根＋双根 SmolVLA 使用双路 RGB、七维状态与七维绝对关节动作，J5 由模型预测。实际数据来源以
[`sources.json`](../artifacts/aubo_joint_two_strip_top45_pilot_20260927/sources.json) 为准，而不是按目录名称推断条数。

| 用途 | 本机 `datasets/` 目录 | 实际 episode |
| --- | --- | ---: |
| 单根训练 | `aubo_joint_legacy_train_0deg_10`、`aubo_joint_legacy_train_0deg_part2_6`、`aubo_joint_legacy_train_45deg_10`、`aubo_joint_legacy_train_45deg_part2_5`、`aubo_joint_legacy_train_67p5deg_10`、`aubo_joint_legacy_train_90deg_10`、`aubo_joint_legacy_train_135deg_10`、`aubo_joint_legacy_train_157p5deg_10` | 60 |
| 单根验证 | `aubo_joint_legacy_validation_12` | 12 |
| 双根训练 | `aubo_joint_two_strip_top45_train30_20260926`、`aubo_joint_two_strip_top45_train21_20260926` | 29＋21＝50 |
| 双根验证 | `aubo_joint_two_strip_top45_validation5_20260926`、`aubo_joint_two_strip_top45_validation2_20260927` | 3＋2＝5 |

上述 13 个数据目录及清单指向的同名 `artifacts/` 证据目录，在本次清理后均仍存在。本次没有修改
数据、清单、模型或远端训练机。

## 已删除的本机目录

按操作者授权，本次从 `datasets/` 删除了以下三组，共 9 个目录，清理前占用约 0.9 GiB：

| 组别 | 已删除目录 | 与现行混训的关系 |
| --- | --- | --- |
| 旧 C0 原始数据 | `c0_smoke_single_strip_20260920T231125`、`c0_formal_single_strip_20260921_run01`、`c0_formal_single_strip_20260921_run02`、`c0_formal_single_strip_20260921_run04`、`c0_pilot_single_strip_20260921_run05`、`c0_pilot_single_strip_20260921_12` | 旧 13D 状态／8D TCP 动作路线；不在现行清单 |
| 早期关节试采 | `aubo_joint_j5_smoke_01`、`aubo_joint_legacy_smoke_001` | 不在现行清单；前者属于早期固定 J5 试验 |
| 层序静态试标 | `bamboo_layer_static` | 两张单路全景试标图及人工记录；未进入当前双路层序评价集 |

此前同日已按授权删除本机 `artifacts/c0_*` 的 13 个旧 C0 实验目录。旧 C0 规划文档中的本地数据、
哈希清单和实验产物路径因此只保留历史描述，不能再用这些路径复核原始文件。静态层序评价代码仍在，
但本机已没有上述两张试标图；其删除不等于完成层序验证。

`datasets/` 和 `artifacts/` 均被 Git 忽略。此提交仅记录文档状态，不能通过 Git 恢复已删除的原始文件。
后续如需清理现行混训数据，应先重新核对 `sources.json` 和对应证据目录。

# 当前训练与离线评价

入口为 `train_joint_smolvla.py`，使用 [七维关节契约](README.md#当前契约)。
当前 `mixed-both-orders` 已完成 160 条训练、22 条验证、30000 步微调和离线验收，
不再处于“等待首次七维训练”阶段。模型和原始验收路径见 [证据索引](EVIDENCE.md)。

## 阶段与默认值

| `--stage` | 行为 |
| --- | --- |
| `plan`（默认） | 只显示参数，不加载模型、不写结果 |
| `preflight` | 审计来源、筛选成功示范、核对划分、计算训练统计及文件哈希 |
| `smoke` | 本地真实模型的合成数据前向、反向和严格保存重载；零优化器更新 |
| `train` | 数据预检后训练，保存最终模型，重载检查并做离线评价 |
| `evaluate` | 加载已有检查点及其处理器，在相同来源上评价 |

默认 `device=cpu`、`steps=10000`、`batch-size=8`、`seed=42`；
当前模型使用显式 `--steps 30000 --device cuda`。模型一次预测 50 步，每步 7 维。
入口只保存最终检查点，没有中途恢复训练接口。训练输出必须是全新目录。

所有模型阶段从本地缓存加载，进程内关闭 Hub/WandB 上传及遥测。
`preflight` 不加载模型，`smoke/train/evaluate` 会加载；真实训练和新增模型推理由操作者明确启动。

## 数据来源

当前本机清单：`artifacts/aubo_joint_both_orders_20260928/sources.json`。
GPU 副本清单：同目录下的 `sources_remote.json`。

`AuboJointTrainingSourcesV1` 包含 train 来源列表，以及一个或多个 validation 来源。
每项包含 `root`、`evidence_root`、`source_dataset_root`：
前两项定位当前文件，最后一项保留采集时的绝对路径身份。相对路径按清单所在目录解析。
复制数据时连同证据搬运，保持原始 JSON 不变；核对两端预检的内容哈希。

成功示范按结果和夹爪标签筛选；归一化只使用选中的训练帧。
每个来源先构造不跨条目的动作块，再组合训练。场景编号隔离不能替代人工检查物理摆放是否重复。

本机只看方案：

```bash
.venv/bin/python examples/phone_to_auboi10/train_joint_smolvla.py \
  --stage plan --steps 30000 --batch-size 8 --seed 42 \
  --data-manifest artifacts/aubo_joint_both_orders_20260928/sources.json
```

本机数据预检（输出须全新）：

```bash
.venv/bin/python examples/phone_to_auboi10/train_joint_smolvla.py \
  --stage preflight \
  --data-manifest artifacts/aubo_joint_both_orders_20260928/sources.json \
  --output artifacts/joint_preflight_run02
```

## GPU 训练模板

已记录的 GPU 项目根目录是
`/home/rentao/program/lerobot-aubo-smolvla-joint-both-orders-20260928`，
SSH 别名为 `gpu`；这些是部署记录，运行前重新确认文件和 GPU 占用。
不要重复执行历史 `run_training.sh train`，它绑定已经存在的 run01 输出。

在 GPU 项目根目录，核对本地模型缓存后使用下列模板。当前写为 plan，不启动训练：

```bash
PYTHONPATH=src \
/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv/bin/python \
  examples/phone_to_auboi10/train_joint_smolvla.py \
  --stage plan --device cuda --steps 30000 --batch-size 8 --seed 42 \
  --data-manifest artifacts/aubo_joint_both_orders_20260928/sources_remote.json \
  --base-path /path/to/local/smolvla_base \
  --vlm-path /path/to/local/SmolVLM2-500M-Video-Instruct \
  --output outputs/joint_both_orders_run02
```

替换两个模型缓存占位路径及输出目录；确认 GPU 无冲突任务、来源清单正确且已授权训练后，
才把 `--stage plan` 改为 `--stage train`。`PYTHONPATH` 只影响本条进程，不修改系统环境。
实际数据和任务预算变化时，显式记录新的步数，不能把模板当作新实验已批准的预算。

## 结果与评价

输出包含数据预检、训练统计、训练日志、`final/`、重载检查及逐帧预测。
`final/` 保存权重、配置、输入/输出处理器和契约。
`evaluate` 必须指定 `--checkpoint <final目录>`、相同数据来源和新的 `--output`，
不需要重新指定基座和 VLM 路径。

查看六关节误差和夹爪事件时序，并与“复制当前夹爪状态”基线比较。
离线评价使用保存的专家观测及动作块首步，不能当作实机闭环成功率。

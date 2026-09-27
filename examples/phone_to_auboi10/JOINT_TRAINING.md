# 当前状态：七维关节训练准备，等待正式训练授权

**2026-09-22 后续更新：用户选择恢复旧手机遥操，当前代码契约变更为 `AuboI10JointLegacyTeleopV2`。状态/动作均为 `J1,J2,J3,J4,J5,J6,gripper_pos` 七维，J5 不再固定。**

已采集 60 条训练示范（47059 帧）、12 条验证示范（9316 帧）。用户确认验证时重新摆放并改变位置；训练场景 `001` 至 `006`，验证场景 `007`。独立录制不等于已经证明模型能适应任意新位置。

七维真实 SmolVLA 已在 GPU 机的 CPU 上完成合成数据前向、反向和严格保存重载：loss=0.5898761153，吸盘输出层梯度范数=1.0577632，重载预测逐值一致，形状 `[1,50,7]`，优化器更新为 0。正式训练尚未启动。下面固定 J5 的部署及验证记录仍属于历史六维版本。

## 本轮具体入口

- 新部署：`/home/rentao/program/lerobot-aubo-smolvla-joint-legacy-v2-20260922`。
- 复用 Python：`/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv/bin/python`，不安装依赖。
- 工作站准备证据：`artifacts/aubo_joint_legacy_preparation_20260922/`。
- `sources.json` 使用 `AuboJointTrainingSourcesV1`：`train` 为来源列表，`validation` 可为一个来源对象或来源列表。每个来源含 `root`、`evidence_root`、`source_dataset_root`；前两项相对于清单目录解析，最后一项保留采集时的绝对路径。多个验证来源分别保留于预检结果，并在 `heldout_predictions.npz` 的 `source_index` 和 `validation_by_source.json` 中区分。
- 训练入口增加 `--data-manifest`，与原来的独立目录参数互斥；原单目录用法保留。数据各自构造 50 步动作块后再拼接，不重写原始条目或证据。
- 归一化只使用选中的成功训练帧，按帧数合并统计；工作站已与全部原始训练行直接计算的均值、标准差对比一致。

GPU 机只查看方案（不加载模型、不使用 GPU 计算）：

```bash
cd /home/rentao/program/lerobot-aubo-smolvla-joint-legacy-v2-20260922
bash artifacts/aubo_joint_legacy_preparation_20260922/run_training.sh plan
```

得到用户明确训练授权后，才将最后的 `plan` 改为 `train`。脚本会先检查是否已有 GPU 计算进程，发现占用就退出。参数为 CUDA、batch=8、steps=10000、seed=42，输出为全新 `outputs/joint_legacy_v2_run01`。只有最终检查点，无中途续训能力；本轮不自动调整参数或恢复旧训练。

当前采集说明见 [JOINT_CAPTURE.md](JOINT_CAPTURE.md)。可离线运行 `.venv/bin/python examples/phone_to_auboi10/train_joint_smolvla.py --stage plan` 查看七维方案。50 步动作块、`n_action_steps=1`、成功示范筛选、场景隔离及仅训练集统计的要求保持。

---

以下为历史六维准备记录，保留原始结果：

# 固定第五轴：SmolVLA 训练准备与使用

2026-09-22：GPU 机现有环境已只读复核，无需因为工作站缺少 `transformers` 而重装。工作站负责采集；GPU 机负责后续训练和离线预测。本次未开始录制，也未进行真实模型优化器更新。

## 已验证的环境

- SSH 别名 `gpu`，主机名 `WP`，NVIDIA GeForce RTX 4090 D。
- 已部署 Python：`/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv/bin/python`。
- PyTorch 2.10.0、transformers 4.57.1、accelerate 1.14.0、safetensors 0.8.0、torchcodec 0.10.0。
- 官方 SmolVLA 基座缓存：`/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.hf-cache/hub/models--lerobot--smolvla_base/snapshots/d9f33c94a60fb382c90dea2164c96845bd955e28`。
- VLM 配置和分词器缓存：`/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.hf-cache/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467`。

GPU 上有其他任务。准备测试使用该机 CPU、4 个线程、合成图像及动作，不占用 GPU 做模型计算，不更改共享虚拟环境。

## 新入口

`train_joint_smolvla.py` 与旧 ACT `train.py` 独立，有五种模式：

| 模式 | 行为 |
| --- | --- |
| `plan`，默认 | 显示方案；不加载模型、不写文件、不连接设备 |
| `preflight` | 审计两个数据集，筛选成功示范，检查场景隔离，计算训练统计并记录文件 SHA-256 |
| `smoke` | 使用缓存的真实模型和合成数据做前向/反向及保存重载；零优化器更新 |
| `train` | 重新执行数据检查，再按显式配置训练、保存最终模型、重载验证及保留集评价 |
| `evaluate` | 加载本入口保存的检查点和处理器，对同一组数据做离线评价；不执行机器人动作 |

训练默认参数为 seed=42、batch=8、steps=10000、50 帧预测块、每次取第一步、绝对关节目标。步数是可修改的初始配置，不是已证明最优的参数。训练采用基座的 AdamW 参数，预热后余弦衰减；仅保存最终检查点，不自动搜索参数、挑选最佳模型或恢复旧任务。训练中断时保留日志，但当前没有中途续训入口。

默认设备是 CPU；正式训练必须显式给 `--device cuda`。所有模型阶段强制离线加载，本进程禁用 WandB、Hub 上传及遥测。配置、统计、数据划分、旁路证据哈希和处理器随结果保存。失败不会写入 `complete.json`。

## 录制前后如何衔接

录制前可以在工作站运行以下命令检查方案，**不会开始训练**：

```bash
.venv/bin/python examples/phone_to_auboi10/train_joint_smolvla.py --stage plan
```

后续录制时使用 [JOINT_CAPTURE.md](JOINT_CAPTURE.md) 的新入口，先一条，再成批。至少安排 50 条以上训练示范，另用独立场景录验证示范。每条成功示范须在录制中完成吸附和释放。失败数据保留，但不作为本轮监督训练答案。

训练前检查示例，目录名应替换为之后真实录制的目录：

```bash
.venv/bin/python examples/phone_to_auboi10/train_joint_smolvla.py \
  --stage preflight \
  --train-root datasets/aubo_joint_train_01 \
  --train-evidence artifacts/aubo_joint_train_01 \
  --validation-root datasets/aubo_joint_validation_01 \
  --validation-evidence artifacts/aubo_joint_validation_01 \
  --output artifacts/aubo_joint_preflight_01
```

输出目录必须全新。检查只选 `single_success` 且吸盘标签完整的 episode（示范条目），要求训练条目标记 `train`、验证条目标记 `validation`，并拒绝重复场景编号。归一化按**选中的成功训练示范**重新计算，既不混入失败数据，也不使用验证集统计。

复制到 GPU 机后，同时保留数据目录和证据目录；不要改写原始 JSON 中的路径。若目录变化，用 `--train-source-root` 和 `--validation-source-root` 显式填写工作站录制时的绝对数据路径。工具使用声明的原始身份核对所有证据，再对复制后的文件计算哈希。原始身份声明不代替文件哈希校验，搬运时仍应对比两机 `data_preflight.json` 中的哈希。

## 后续 GPU 训练命令模板

以下是**录制及数据检查完成之后**才使用的模板，本次没有执行。GPU 机独立代码目录为 `/home/rentao/program/lerobot-aubo-smolvla-joint-j5-v1`，复用上面已有虚拟环境。源目录参数假定工作站数据位于当前仓库根目录；实际名称变化时同步修改。

```bash
cd /home/rentao/program/lerobot-aubo-smolvla-joint-j5-v1
PYTHONPATH="$PWD/src" HF_DATASETS_CACHE="$PWD/.dataset-cache" \
  /home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv/bin/python \
  examples/phone_to_auboi10/train_joint_smolvla.py \
  --stage train --device cuda --steps 10000 --batch-size 8 \
  --train-root datasets/aubo_joint_train_01 \
  --train-evidence evidence/aubo_joint_train_01 \
  --train-source-root /home/rentao/program/lerobot-aubo/datasets/aubo_joint_train_01 \
  --validation-root datasets/aubo_joint_validation_01 \
  --validation-evidence evidence/aubo_joint_validation_01 \
  --validation-source-root /home/rentao/program/lerobot-aubo/datasets/aubo_joint_validation_01 \
  --base-path /home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.hf-cache/hub/models--lerobot--smolvla_base/snapshots/d9f33c94a60fb382c90dea2164c96845bd955e28 \
  --vlm-path /home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.hf-cache/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --output outputs/joint_j5_run01
```

`PYTHONPATH` 和缓存变量只影响这条命令，不修改系统或共享环境。输出包括 `plan.json`、`environment.json`、`data_preflight.json`、`train_only_stats.json`、`training.jsonl`、`training_state.pt`、`final/`、`reload_check.json`、`heldout_predictions.npz` 和 `validation.json`。`final/` 保存模型、两个处理器和数据契约。

再次离线评价时使用相同四个数据/证据目录及原始目录映射，将阶段改成 `--stage evaluate`，设置 `--checkpoint outputs/joint_j5_run01/final` 和全新的输出目录；不需要重新指定基座或 VLM 路径。

## 如何判断吸盘表现

`validation.json` 分别报告五个关节的角度误差、吸盘原始分数误差、开启召回率、释放召回率、保持期间误切换率，以及复制当前状态的基线准确率。`heldout_predictions.npz` 保留逐帧原始预测、标签、状态及条目/帧索引，可进一步看切换提前/滞后时间。

评价使用保存的专家观测、每次预测动作块的第一步，不能当真机闭环成功率。不会自动将数值指标解释为允许机械臂执行；新关节在线执行接口仍要在后续上机前单独验证。

## 准备验证结果

真实 SmolVLA CPU 冒烟通过：loss=0.7692441940，吸盘输出层梯度范数=4.12253046，优化器更新次数为 0；严格重载后预测逐值一致，形状为 `[1,50,6]`。本地留档：`artifacts/aubo_joint_preparation_20260922/`。这证明代码、现有模型和处理器可以衔接，不证明真实竹条任务已经学会。

另有小型替身模型测试覆盖训练循环、优化器更新、检查点保存/重载与独立评价；这些更新仅在测试替身上执行。数据搬运、场景泄漏、成功示范筛选和训练统计也有离线回归测试。

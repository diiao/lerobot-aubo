# 当前采集：双 RGB＋七维关节示教

入口为 `record_joint.py`。模型的数据契约见 [README](README.md#当前契约)。
采集复用手机遥操：手机控制平移和 J6，内部逆运动学生成六关节目标，吸盘由操作者切换。
记录的 action 是控制器接受的目标，不能用实测关节角替代；TCP 不进入模型状态或动作。

## 先查看计划

在仓库根目录运行。示例路径须选用尚不存在的新目录：

```bash
.venv/bin/python examples/phone_to_auboi10/record_joint.py \
  --dataset-root datasets/joint_train_run02 \
  --evidence-root artifacts/joint_train_run02 \
  --num-episodes 1 --split train
```

省略 `--record` 只打印计划，不连接设备、不创建目录。数据与证据目录必须互相独立且均不存在。
正式采集由操作者在确认工作区和归位路径后，在同一命令末尾加 `--record`。

## 现场操作

1. 填写场景编号和物理摆放参考。同一摆放保持同一编号，不能跨训练、验证或测试。
2. 按 `r` 归位，摆好竹条，再按右箭头开始录制。
3. 手机完成接近、吸附、提起、搬运及释放；必须在录制结束前释放。
4. 右箭头结束，左箭头重录，Esc 停止；填写单根成功、抓空、多抓、滑落、受阻或不确定。
5. 保留失败和中断记录，人工核查视频后再决定哪些条目用于监督训练。

录制与控制目标频率为 25 Hz。填表时快捷键监听暂停，避免与文字输入冲突。
归位沿用 `NORMAL_START_DEG`；录制退出先停伺服，再整理视频。Esc 不是硬件急停，
也不能中断正在等待完成的归位运动。

结果标签沿用 `single_success` 等已有定义；双根历史示范的完整周期和物理结果仍需结合视频审查。
后续多根/指定目标数据的标签和任务终止方式需要另行设计，不能把既有标签直接解释为整堆清空。

## 录后审计

```bash
.venv/bin/python examples/phone_to_auboi10/audit_joint_dataset.py \
  --dataset-root datasets/joint_train_run02 \
  --evidence-root artifacts/joint_train_run02 --require-gripper-quality
```

审计核对七维契约、控制器指令、时间戳、保存一致性及吸盘标签。
`audit_passed` 表示记录一致；`gripper_quality_passed` 表示标签包含所需开关变化，
两者都不能证明物理抓住竹条。还需查看双路视频，核实目标、单根分离和放置。

批量采集时修改 `--num-episodes` 并使用新目录。验证集使用 `--split validation`；
测试集使用 `--split test`，按独立物理摆放划分，而非把相邻帧拆到不同集合。
现行训练入口使用 train/validation 来源，单独测试集不自动进入训练。

## 数据交接

每个 `datasets/<name>` 与对应 `artifacts/<name>` 一起保留。不要重写原始证据中的源路径。
组合不同批次使用显式来源清单，见 [训练说明](JOINT_TRAINING.md)；
当前 160/22 数据来源见 [证据索引](EVIDENCE.md)。

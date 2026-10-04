# 当前采集：双 RGB＋七维关节示教

入口为 `record_joint.py`。模型的数据契约见 [README](README.md#当前契约)。
采集复用手机遥操：手机控制平移和 J6，内部逆运动学生成六关节目标，吸盘由操作者切换。
记录的 action 是控制器接受的目标，不能用实测关节角替代；TCP 不进入模型状态或动作。

## 相机配置与体检

本节统一维护相机更换的配置、测试证据、回退与后续操作，其他项目入口仅链接到这里。

2026-10-03 起，新采集默认使用 `--camera-set wide-global`：新 `2MP USB Camera` 全局相机＋原 Sonix 腕部相机，
读取 [camera_set_wide_global.json](../../configs/aubo_i10/camera_set_wide_global.json)。
全局按1920×1080原图采集保存，腕部保持640×480；MJPG，全局30 FPS、腕部25 FPS。
两路分辨率可以不同，采集入口按各自配置创建视频数据字段，不缩小全局原图。
配置状态 `mounting_pending_review` 表示支架未固定、最终视野待复核；配置已接入采集和体检入口。
首轮三根研究继续使用现有支撑物和工作区。

旧全局相机保留原位置和角度。接回原相机后，采集／体检加 `--camera-set original-global` 即可恢复旧配置。
旧模型执行入口仍使用原 `CameraSetV2.json` 及冻结校验，不随新采集配置切换。
换机前代码与文档回退点为 `37fc88c`；通常只需切回相机选项，无需覆盖工作树或删除新配置。

### 已完成的测试

2026-10-03 使用项目 `OpenCVCamera` 驱动，仅打开新全局相机（序列号 `04434000_P120800_SN0002`），
每种模式预热3秒后测量：

| 图像模式 | 测量时长 | 实测帧率 | 消费端读取错误 |
| --- | --- | --- | --- |
| MJPG 640×480，请求30 FPS | 15秒 | 30.01 FPS | 0 |
| MJPG 1920×1080，请求30 FPS | 8秒 | 30.00 FPS | 0 |

测试正常退出，相机已关闭；未打开腕部相机、连接机械臂、发送IO或运行策略。
这只证明短时单相机出帧，尚未验证双相机并发、最终视野、层序可判定性或新视角下的策略效果。
临时样图主要覆盖机械臂和墙面，不能作为正式料堆视野。
配置接入和不同尺寸数据处理另通过52项软件测试，不替代上述现场复核。

本地证据（`artifacts/` 不随 Git 提交）：
[测试报告](../../artifacts/new_global_camera_20261003_nkaq9n8h/report.json)、
[测试脚本](../../artifacts/new_global_camera_20261003_nkaq9n8h/probe.py)、
[640×480样图](../../artifacts/new_global_camera_20261003_nkaq9n8h/global_rgb_640x480.png)、
[1920×1080样图](../../artifacts/new_global_camera_20261003_nkaq9n8h/global_rgb_1920x1080.png)。

### 支架到位后的复核

先固定视角，检查完整料堆、交叉处细节、收集区及机械臂遮挡，再进行双相机体检。

只显示新相机配置，不打开设备、不写文件：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/diag_preflight_cams.py --camera-set wide-global --plan
```

支架固定后，由操作者启动双相机体检（会打开两台相机，但不连接机械臂）：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/diag_preflight_cams.py --camera-set wide-global --seconds 15
```

预览图和配置快照写入新建的 `/tmp/aubo_camera_preflight_wide-global_*` 目录。查看打印的实际路径，
核对全局图能覆盖料堆及收集区、交叉处清楚，腕部图没有接反。
可用 `--output <新目录>` 保留结果；原 `CAMERA_TEST_SECONDS` 环境变量仍可作为测量时长默认值。

## 先查看计划

在仓库根目录运行。示例路径须选用尚不存在的新目录：

```bash
.venv/bin/python examples/phone_to_auboi10/record_joint.py \
  --camera-set wide-global \
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
新采集在证据 `plan.json` 和数据 `meta/camera_configuration.json` 中保存选中的相机配置快照。
新旧视角使用不同数据目录；本次未更改训练清单，不自动将新相机数据并入现有160/22数据。
审计按已保存的相机尺寸核对视频字段。旧训练入口及旧模型仍要求双640×480；新原图进入动作训练时，
需要另行接入与新模型一致的等比例缩放／补边预处理，不能靠拉伸或覆盖原视频绕过尺寸要求。
组合不同批次使用显式来源清单，见 [训练说明](JOINT_TRAINING.md)；
当前 160/22 数据来源见 [证据索引](EVIDENCE.md)。

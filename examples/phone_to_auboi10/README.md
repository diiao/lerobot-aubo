# AUBO i10 项目手册

更新：2026-10-06。本文集中维护当前状态、操作方法和后续计划；实验指标与原始路径只维护在
[证据索引](EVIDENCE.md)。所有命令从仓库根目录 `/home/rentao/program/lerobot-aubo` 执行。

## 目录

- [当前状态与系统边界](#当前状态与系统边界)
- [相机配置](#相机配置)
- [静态分割：采集与人工复核](#静态分割采集与人工复核)
- [静态分割：数据、训练与独立评价](#静态分割数据训练与独立评价)
- [动作模型：手机遥操采集](#动作模型手机遥操采集)
- [动作模型：训练与离线评价](#动作模型训练与离线评价)
- [动作模型：现场运行](#动作模型现场运行)
- [后续研究与待办](#后续研究与待办)
- [同场景A／B成对示教方案](#同场景ab成对示教方案)
- [一般堆叠的下探高度修正方案](#一般堆叠的下探高度修正方案)
- [指定目标输入：离线原型](#指定目标输入离线原型)
- [代码导航与维护](#代码导航与维护)

## 当前状态与系统边界

当前有两条独立流程。实例分割指为每根木条识别可见像素区域；动作模型根据图像和机器人状态预测运动。

| 流程 | 输入 → 输出 | 当前状态 |
| --- | --- | --- |
| 静态 Mask2Former 分割 | 固定全局 RGB → 可见实例掩码、`top_strip` / `covered_strip` | 首轮训练、验证轮廓复核、047～051新摆放独立测试均完成；发现被压条误判上层，详见[结果](EVIDENCE.md#分割模型与独立测试)。 |
| 七维 SmolVLA 抓放 | 全局＋腕部 RGB、七维状态、任务文字 → 七维绝对关节动作 | `mixed-both-orders` 已训练；熟悉长条摆放及当前40 cm单根均有成功反馈，见[实机证据](EVIDENCE.md#40-cm单根试抓)；支持指定命令周期数。 |
| 感知与执行闭环 | 指定上层目标 → 单次抓放 → 重新观察 | 尚未接通；目标条件化、物理结果判断及自动清空均未验证。 |

静态研究与近期目标动作试采使用 **400×25×8 mm** 短条；旧160条动作训练示教使用60 cm长条。
当前目标服从实验沿用40 cm、原机位和双640×480，不同时切换长度与相机；60 cm及混合长度留待后续明确验证。
不能把短条分割成绩外推为短条抓取或任意无序堆叠成功率。特殊结构、VLM（视觉语言模型）、RL（强化学习）暂缓。

动作契约为 `[J1,J2,J3,J4,J5,J6,gripper_pos]`，关节单位为度，进入机器人 SDK 时转换为弧度；J5由模型预测。
`CameraSetV2` 的 `global_rgb` 为固定全局相机，`grasp_rgb` 为腕部相机，均为640×480。
末端为 Airtac HFKL20 气动平行二指夹爪：0松开、100夹紧；这只是命令，不是持物传感器。
旧字段 `suction_*` 与历史“吸盘/吸附”均指夹爪开合；DO2夹紧、DO3松开。
固定任务文字为 `Pick one strip and place it in the collection area.`。

## 相机配置

采集与体检默认 `--camera-set original-global`：旧 GENERAL 全局相机＋Sonix 腕部相机，
配置为 [CameraSetV2.json](../../configs/aubo_i10/CameraSetV2.json)。当前研究保持原机位与支撑物。
`--camera-set wide-global` 保留为可选项，读取 [camera_set_wide_global.json](../../configs/aubo_i10/camera_set_wide_global.json)：
新全局1920×1080、MJPG 30 FPS，腕部640×480、25 FPS。支架及双路并发待复核，暂缓使用。
新全局原图不缩小；新旧视角分开保存。旧动作训练与执行仍要求冻结的双640×480配置，不能靠拉伸绕过契约。

以下仅查看配置，不打开设备：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/diag_preflight_cams.py --camera-set original-global --plan
```

经相机授权后，操作者可将 `--plan` 换为 `--seconds 15` 做双相机体检；它不连接机械臂。
新相机恢复使用前先固定视角、确认料堆与收集区完整、交叉处清楚，再显式使用 `wide-global` 体检。
默认结果在新建的 `/tmp/aubo_camera_preflight_<配置名>_*`，也可指定 `--output <新目录>`。
旧相机保持原位置；切回 `original-global` 即可恢复采集配置，无需覆盖工作区。历史测试见[证据索引](EVIDENCE.md#相机与环境记录)。

## 静态分割：采集与人工复核

### 采集规则与编号

原图集中于 `artifacts/placement_sequences/pile_XXX/`；人工下载标注在 `pile-json--/`；
[人工复核入口](../../artifacts/placement_sequences/review/index.html) 是唯一人工标注页面入口，每组只维护 `review/pile_XXX/` 一份结果。
当前正式训练／验证为007～046，独立测试047～051已经完成，不能重复拍到这些目录。
每次只安排后续5组；安排前检查原始目录、统一review及下载JSON的组号，并核对脚本参数，不提前假定后续编号可用。

1. 每组清空后独立重摆，拍空场景及依次加入1、2、3根，共4张图；保持普通逐层结构，自然变化位置、方向、交叉点。
2. 逐根从上方加入；旧条不能移动或明显倾斜；相机、承托面保持不动。手退出、场景静止再拍，整根及端部余量留在原图内。
3. 每五组集中检查照片，由操作者确认上述现场条件。照片不能完全证明旧条未移动；条件不成立时不能直接传播遮挡真值。
4. 同一摆放的阶段图、衍生图、关联对照保持同一 split（数据集合）。测试组在采集前指定，不按结果重新划分。

以下为**参数模板**，须先填经核对的新组号和集合；省略 `--record` 只看计划、不创建目录：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/capture_placement_sequence.py \
  --camera-set original-global --output "artifacts/placement_sequences/<新组号>" \
  --placement-id "<新组号>" --split "<train或validation或test>" \
  --strips 3 --roi 280 200 640 480 --min-chroma-change 8
```

拍摄由操作者经授权后添加 `--record`：回车保存各阶段，`q` / Ctrl+C提前结束并保留照片。
入口只打开全局相机，不连接腕部相机或机械臂。ROI（候选提取区域）及色度阈值仅用于差分草稿，模型使用完整640×480原始RGB。
默认RGB差分阈值25、候选最小面积80；同色交叉和阴影可能导致草稿断裂，不能将差分候选直接当真值。
曝光／白平衡尝试锁定并记录返回值，退出仅恢复成功写入的控制项；`camera_control_lock_unverified` 不代表锁定成功，需检查亮度漂移。
原目录保留 `plan.json`、`sequence.json`、`frame_*.png`、候选掩码与标签，已有目录拒绝覆盖。

### 人工修边与导入

每一步只画本步新增木条的多边形轮廓。页面拖点修边，Shift＋点击加点、右键删点；
检查后点“确认当前轮廓”，修改会取消该根确认；下载前核对确认数量，保留下载JSON。
画面截断必须标记，不能推测画外端点；首轮完整轮廓数据排除有截断物体的整帧，保留排除原因。
先确认真值再看预测，避免按模型输出修改答案。

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/review_placement_sequence.py \
  --source "artifacts/placement_sequences/<组号>" \
  --polygons "pile-json--/<组号>_polygons.json"
```

省略 `--output` 更新该组统一复核目录及索引：先在临时位置生成，失败保留原结果，成功后替换同组派生结果。
原照片、下载JSON不变，不建多个最终review目录。确认“从上方加入且旧条静止”后，工具按后放轮廓扣除先放条的被遮部分；
可见区域断为两段仍是同一根实例。`training_ready=false`、`graspable=null` 保留原始含义：复核并不证明已训练或物理可夹。

## 静态分割：数据、训练与独立评价

### 数据与模型输入

| 集合 | 组号与选择清单 | 组数／图数／可见实例 | 现有导出目录 |
| --- | --- | --- | --- |
| train | 007～046中除下列validation；[选择清单](../../configs/aubo_i10/short_strip_segmentation.json) | 30／120／180 | `datasets/short_strip_segmentation/` |
| validation | 011、016、021、026、031、036、041、044、045、046 | 10／40／60 | 同上 |
| test | 047～051；[选择清单](../../configs/aubo_i10/short_strip_segmentation_test.json) | 5／20／30 | `datasets/short_strip_segmentation_test/` |

每组包含1张空场景；三集合没有组级交叉。现有数据已经导出，不重复导出，不为提高指标重划分。
`images/` 是无描边RGB；`instances/` 是16位实例PNG，0为背景、正整数为本帧实例ID；
`dataset.json` 的 `segments` 单独解释类别：0=`top_strip`，1=`covered_strip`。类别0不等于PNG背景0。
只标可见像素，完全不可见实例不构造前景；空图是全零掩码＋空实例列表，不等同于缺少标注。
轮廓、放入顺序与组号是监督和对应信息，不能泄漏到模型输入。

`export_strip_segmentation.py --selection <清单> --output <新目录>` 导出已复核数据；
仅测试清单需显式 `--test-only`，默认仍要求train＋validation。导出会审计确认、尺寸、像素ID、面积及组级划分，拒绝覆盖。

### 训练及环境

`train_strip_segmentation.py` 默认 `--stage preflight`，只审计数据；
真实训练需显式 `--stage train --base-path <本地基座> --output <新目录> --device cuda` 并另行授权。
入口不隐式下载或续训。当前首轮已完成，参数、模型位置及加载修正见[证据索引](EVIDENCE.md#分割模型与独立测试)。

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/train_strip_segmentation.py \
  --stage preflight --dataset datasets/short_strip_segmentation
```

依赖清单为 `strip_segmentation_requirements.txt`。本机补充依赖位于临时目录 `/tmp/aubo-segmentation-deps/`，
GPU补充依赖位于实验目录 `deps/`；通过单条命令的 `PYTHONPATH` 使用，不改系统环境。
临时目录可能消失，应先核对再使用；旧GPU虚拟环境仍在复用，不能按旧实验名称删除。
训练完成以 `complete.json` 和best重载评价为准，tmux会话退出不等于失败。

### 独立评价与预测复核

独立测试需冻结模型、处理器、阈值和后处理，先确认真值，再运行 `evaluate_strip_segmentation.py`。
它要求已完成训练的 `best/`、测试清单与原训练清单，检查组交叉及模型契约；只推理、不优化、不保存新权重。
以下为将来新测试的参数模板，现有047～051结果无需重跑：

```bash
PYTHONPATH=deps:src "<复用Python>" examples/phone_to_auboi10/evaluate_strip_segmentation.py \
  --dataset "<新测试数据目录>" --training-manifest "<原训练dataset.json>" \
  --model-path "<已完成训练的best目录>" --output "<新结果目录>" --device cuda
```

模板中的 `deps` 相对于执行目录；当前GPU路径见证据索引，独立测试子目录运行时使用 `../deps:src`。
结果目录须与数据、训练输出及原清单目录相互独立。固定置信度0.5、mask阈值0.5、后处理重叠阈值0.8、batch=2；不按测试结果扫阈值。
IoU（交并比）是预测与真值重叠面积除以合并面积；按同类别一对一匹配统计TP正确匹配、FP错误预测、FN漏配及F1综合指标。
报告IoU≥0.5与≥0.75、逐实例IoU、空图误检、具体失败图；F1不是抓取成功率，组内多帧不是多个独立摆放。

现有页面直接打开，无需网络、服务器或重新推理：

- [验证集预测复核](../../artifacts/short_strip_training/result/prediction_review.html)：40图／10组。
- [独立测试预测复核](../../artifacts/short_strip_training/independent_test/result/prediction_review.html)：20图／5组。

页面支持按组、原图／真值／预测／误差、放大、描边／填充、最低IoU及类别／数量异常。
预测PNG必须由报告的 `segments` 解释类别，像素ID不是类别ID；页面对齐颜色不表示跨帧追踪。
检查交叉处错误补全、同根分段是否同ID、邻条混入、端部漏分及空图碎片。
`review_strip_predictions.py --dataset <数据> --result <结果> --split validation或test`
利用现有掩码生成 `prediction_review.html` 和 `contour_review.json`，不加载模型；已存在时拒绝覆盖。
同一实验只维护固定页面，不因一次分析创建多个版本目录。

## 动作模型：手机遥操采集

`record_joint.py` 复用手机遥操：手机控制平移和J6，逆运动学生成六关节目标；夹爪由操作者切换。
action记录控制器接受的目标，不能替换成实测角度；TCP（工具中心点）不进入七维模型状态或动作。
以下仅查看计划，输出目录须均不存在且互相独立：

```bash
.venv/bin/python examples/phone_to_auboi10/record_joint.py \
  --camera-set original-global --dataset-root datasets/joint_train_run02 \
  --evidence-root artifacts/joint_train_run02 --num-episodes 1 --split train
```

正式采集由操作者经授权后添加 `--record`。填写场景及摆放编号，`r`归位，右箭头开始／结束，左箭头重录，Esc停止。
手机完成接近、夹紧、提起、搬运、释放，结束前须释放并填写实际结果；保留失败与中断记录。
录制和控制目标25 Hz；填表时暂停快捷键监听。归位使用 `NORMAL_START_DEG`；退出先停伺服再整理视频。
Esc不是硬件急停，不能中断等待完成的归位。既有 `single_success` 标签不自动表示双根或整堆成功。

```bash
.venv/bin/python examples/phone_to_auboi10/audit_joint_dataset.py \
  --dataset-root datasets/joint_train_run02 \
  --evidence-root artifacts/joint_train_run02 --require-gripper-quality
```

审计通过表示契约、时戳、存储及开合标签一致，还需看双路视频确认实际抓放。
增加条目用 `--num-episodes`；验证／测试分别用 `--split validation` / `test`，按独立摆放划分。
数据目录与对应证据一起保留，不重写源路径；相机快照在证据 `plan.json` 及数据 `meta/camera_configuration.json`。
现有动作训练不自动读取静态分割数据或新相机数据；多来源组合使用下节的显式清单。

## 动作模型：训练与离线评价

入口为 `train_joint_smolvla.py`，使用 [七维关节契约](#当前状态与系统边界)。
当前 `mixed-both-orders` 已完成微调和离线验收。模型、训练参数和原始验收路径见 [证据索引](EVIDENCE.md#当前双摆放模型)。

### 阶段与默认值

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

### 数据来源

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

### GPU 训练模板

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

### 结果与评价

输出包含数据预检、训练统计、训练日志、`final/`、重载检查及逐帧预测。
`final/` 保存权重、配置、输入/输出处理器和契约。
`evaluate` 必须指定 `--checkpoint <final目录>`、相同数据来源和新的 `--output`，
不需要重新指定基座和 VLM 路径。

查看六关节误差和夹爪事件时序，并与“复制当前夹爪状态”基线比较。
离线评价使用保存的专家观测及动作块首步，不能当作实机闭环成功率。

## 动作模型：现场运行

当前入口为 `run_joint_smooth.py --model mixed-both-orders --approach-age-aligned`。
运行前核对模型部署、CameraSetV2 和 GPU 占用；全部实机命令由现场操作者手动执行。
当前代码不检测整堆清空。`--cycles N` 指定夹紧→松开指令周期数；省略时保留旧行为：混训两次，旧单根一次。
单根试验使用 `--cycles 1`；只改变结束条件，不改变模型、动作选择、运动限制或总时限。

### 1. 查看计划

在仓库根目录运行，不连接相机、机械臂或远端模型：

```bash
.venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
  --model mixed-both-orders --approach-age-aligned
```

核对 `requested_suction_cycles=2`、
`pending_suction_policy=continue_inference_while_open` 和模型哈希。
当前模型身份、远端脚本和验收证据见 [证据索引](EVIDENCE.md)。

### 2. 归位

确认未持物、返回路径安全，由现场操作者执行：

```bash
.venv/bin/python examples/phone_to_auboi10/run_joint_trial.py --stage home --onsite-confirmed
```

已读到释放状态时直接归位；否则发送释放指令，核对读回后归位。
无需额外释放参数或第二次 Enter。默认不创建日志目录，显式 `--output <新目录>` 才记录。
该阶段不打开相机、不加载模型。释放或归位失败后不要继续抓取。

### 3. 双根抓放

上90°下45°示例：

```bash
.venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
  --model mixed-both-orders --approach-age-aligned --execute \
  --output "artifacts/joint_both_top90_$(date +%Y%m%d_%H%M%S_%N)"
```

上45°下90°保持参数相同，将输出前缀改为 `joint_both_top45_`。
启动后输入完整任务文字 `Pick one strip and place it in the collection area.`，
待模型和设备就绪，按现场提示确认开始。本地入口通过 SSH 启动远端服务，无需另开服务进程。

起点要求夹爪松开，各关节距 `START_DEG` 不超过 1.5°。
若数字输出均低、状态未知，入口保留一次“未持物”确认再释放；
这与独立 home 的无二次确认流程不同。

#### 单次抓放

先查看计划（不连接设备），确认 `requested_suction_cycles=1`：

```bash
.venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
  --model mixed-both-orders --approach-age-aligned --cycles 1
```

现场操作者准备好后，在同一命令末尾加 `--execute --output <新目录>`。
一次闭合后再松开即结束，随后关闭伺服并保存最终状态；不会额外执行第二次夹紧。
这不确认抓到一根或成功放置，也不自动退让、归位或启动下一轮。
下一次仍须满足原来的起点要求；尚未接通任意结束姿态到下一次抓取的闭环。
总时限仍按模型保留（当前 mixed-both-orders 为360秒），增加周期数不会自动延长。
当前代码不提供 `--second-grasp-z-mm` 等高度补偿参数；新讨论的[下探高度修正](#一般堆叠的下探高度修正方案)尚未实现。

#### 当前实验：40 cm单根基线

顺序已确认：先检验现有动作模型对40 cm短条的适应性，再建立指定目标接口，最后验证普通堆叠的感知与执行。
本轮只试一根400×25×8 mm木条，不调用分割模型、不描边输入、不改模型权重或夹爪分类阈值。
目的在于区分“动作模型能否抓短条”和“分割是否选对上层”，不能把这次结果记为分割引导抓取。
当前已有两次正常完成、经用户确认的40 cm单根抓放成功；最新第五次正常结束，全部原始记录与核对结论统一见
[实机证据](EVIDENCE.md#40-cm单根试抓)。已进入第二步，完成[指定目标输入的离线原型](#指定目标输入离线原型)；尚不能据此估计泛化成功率。
成功结束不自动返回起点，重新试验前仍须先确认归位路径并归位。
当前按用户要求使用显式`--latency-tolerant`试验选项，统一放宽相关预测时间预算，参数见下表。
这允许使用更旧且双路时间错位更大的画面，未证明实机可靠性；超出新预算仍停止，不能保证不再报错。

2026-10-06准备检查：上述单周期计划通过；GPU上的动作模型文件、配置、年龄对齐推理脚本及复用Python存在，
当时GPU占用687 MiB、无计算进程。主机设备目录显示GENERAL全局相机映射video2、Sonix腕部相机映射video0，
与CameraSetV2的稳定设备路径一致。沙箱看不到设备节点，不能据此断言相机未连接。
此次未打开相机、连接机械臂或加载模型，实际图像和机器人状态仍由现场启动核对；运行时保留原有权重身份检查。

1. 确认未持物、归位路径可通行，按上节归位命令执行，确认成功后再摆木条。
2. 保持旧全局与腕部机位、原动作工作区和收集区；只放一根短条，先采用熟悉单根摆放的位置与朝向，端部完整入镜。
   若支撑布局与动作训练时不同，单独记录此变化，不把结果只归因于长度变化。
3. 现场操作者执行下列命令，输入本手册的固定任务文字，设备就绪后按提示确认开始。
4. 一次完成或中断后先查看结果，不自动连续重试。记录接近是否对准、是否夹住、能否提起和送达、是否滑落、停止原因。
   原始日志及双路录像位于命令指定目录；真实结果由操作者反馈，不回填成未经核验的成功。

```bash
.venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
  --model mixed-both-orders --approach-age-aligned --cycles 1 --execute \
  --latency-tolerant \
  --output "artifacts/joint_short_strip_single_$(date +%Y%m%d_%H%M%S_%N)"
```

轨迹明显异常时由操作者停止，Ctrl+C结束伺服但不自动松开或归位；紧急情况使用现场急停。
失败后结合录像与trace定位；本节单根基线保持无高度修正，后续下探修正实验另记，避免混淆两者结果。通过一次仅支持继续少量重复验证，不代表成功率已验证。

### 4. 当前动作与保护

动作块是模型一次预测的未来 50 步完整七维动作。夹爪指令为松开时，按观测年龄选取整行，
遇到首次预测闭合边界时不越过它；等待到位期间继续单个在途观测/推理，新结果可更新或取消闭合。
夹紧后沿用首步预测；IO 切换后废弃旧动作块并丢弃切换前观测的在途结果。
当前实现没有接触高度修正、关节偏置、脚本化下降或提起；规划中的下探高度修正尚未接入。

| 当前 mixed-both-orders 限制 | 默认 | `--latency-tolerant`试验 |
| --- | --- | --- |
| 推理响应等待 | 2 s | 5 s |
| 预测观测年龄 | 500 ms | 750 ms |
| 无新鲜预测停止 | 600 ms | 1000 ms |
| 本地命令目标频率 | 25 Hz | 不变 |
| 关节参考速度 / 加速度 | 5°/s / 20°/s² | 不变 |
| TCP（工具中心点）参考速度 | 0.05 m/s | 不变 |
| 伺服跟踪误差 | 3° | 不变 |
| 双相机时间差 | 100 ms | 300 ms |
| 采集时帧年龄 | 250 ms | 400 ms |
| 发送时状态年龄 | 100 ms | 不变 |
| 控制间隔 | 350 ms | 不变 |
| 待夹爪切换目标期限 / 总时限 | 1.5 s / 360 s | 不变 |

试验选项仅用于`mixed-both-orders`，取消该参数恢复默认。`--inference-timeout-s`可单独显式指定响应等待，
优先于表中默认值，不改变观测有效期；模型加载等待仍为120 s。所有实际参数写入本轮`plan.json`。

关节限位来自控制器，当前与目标 TCP 检查工作区边界。
发送前状态过期时最多刷新一次并重新检查；仍过期则停止。
工作区和限速不是全机械臂碰撞检测，日志中的限速是参考命令约束。

### 5. 停止与证据

Ctrl+C、异常及超时进入停止伺服流程，不自动释放或归位。
模型正常释放达到指定周期数后结束。若停止时持物，先由操作者安全处置，再准备下一轮。

每轮保存 `plan.json`、`trace.jsonl`、`commands.jsonl`、`complete.json`、
`final_state.json`、双路 `videos/*.mp4` 及帧时间侧录。
`requested_suction_cycles` / `completed_suction_cycles` 仅为指令计数；
现场另记首抓是否上层、是否只带走一根、放置及停止结果。

### 兼容入口

`legacy-single` 绑定旧单根模型，最多120秒、默认一个周期；
`mixed-double` 绑定旧单根＋一种双根模型，最多360秒、默认两个周期。
二者均不是当前160条模型。

`run_joint_trial.py` 的 shadow/single-step/short-loop 和 `run_joint_live.py`
保留作既有诊断，并向持续入口提供共享实现；它们的旧首步时限不等同于本页持续伺服限制。
其历史操作和逐次故障分析统一从 [证据索引](EVIDENCE.md) 回溯。

## 后续研究与待办

用户已确认按“已完成的[40 cm单根基线](#当前实验40-cm单根基线)与目标数据接口 → 同场景A／B成对示教 → 目标服从实机对照 → 普通堆叠联合验证 → 清堆 → 长短与语言分拣”推进。
总体闭环保持“观察 → 选择可取目标 → 实时全局图描边 → VLA单次抓放 → 结果确认 → 重新观察”。
用户已同意将一般堆叠的下探高度修正纳入方案，仅调整下探抓取段，其他动作保持原策略；具体范围与状态见下方专节，不改为脚本抓放。

**当前主任务：准备并试采同场景双可取目标的A／B示教，尽早验证目标服从。**目标服从指只改变指定对象，实际抓走的对象也随之改变。
首批5条单根示教及目标数据接口已验收，具体数字与复核取舍见[证据索引](EVIDENCE.md#指定目标输入离线原型)；保留作为流程与基础抓放数据，不要求重录或逐帧消除已接受的遮挡漏分。
单根场景即使忽略描边也可能抓对，因此不再以追加普通单根数量作为当前主线，也不等失败后才安排A／B对照。
尚无目标条件模型权重，也未接入实机执行循环；数据接口验收不等于模型会选择目标。
按用户要求，冻结现有跟踪／重新确认诊断，不再补30秒后的帧、调整关联条件或扩展诊断页面；成果保留，
全程搬运轮廓跟踪不再作为示教采集和动作模型适配的前置条件。

| 接下来按顺序完成 | 具体交付与完成依据 |
| --- | --- |
| 1. 完成一对A／B流程试采 | 同一摆放两根都能取，分别示教抓A和抓B；检查目标身份、当前帧轮廓与动作对应。先由人选目标，避免把感知误选混入执行能力评价。操作见下方方案。 |
| 2. 建立成对训练与独立验证数据 | 以不同物理摆放为组，同组A／B及重复录制放入同一split；另设新摆放验证。根据首对采集与标注结果确定规模，一对仅验证流程，不作为充分训练数据。 |
| 3. 微调、接入并检验目标服从 | 使用目标条件模型；离线同一观察只换目标，检查动作响应，再做同摆放恢复后的A／B实机对照。分别记录抓对对象、恰好一根、送达；离线动作变化不是物理成功。保留旧模型自选目标作为对照。 |
| 4. 普通堆叠与清堆 | 目标服从成立后，从双根交叉推进到3～5根；优先复用旧双根示教，仅补不足场景，并研究下方限定的下探高度修正。接入感知与结果判定，下指空间由操作者摆放保证，不以自动下指判断作为前置门槛。每次只抓放一根并重新观察；若模型忽略目标，先修数据或条件化接口。 |
| 5. 长短与语言分拣 | 清堆基本稳定后引入长／短×放置区A／B的四种示教组合；同一目标只换目的地指令，检验语言是否改变放置行为。这里的A／B是放置区，不是前期两根目标的身份。 |

首版接口已实现“接近／夹取阶段标记目标，首次已接受的夹爪闭合命令之后使用原始双RGB完成搬运”，
`target_global_rgb`仍为同一个处理后图像字段，腕部图与七维状态／动作保持原定义。
这使本次指定对象在抓取前表达清楚，不要求夹持搬运全过程都有完整分割轮廓；训练与运行必须共用相同的阶段切换规则。
闭合动作对应的输入仍有标记；全局相机来源时间严格晚于闭合命令接受时间才切换，释放后本轮也不重新描边。
软件接口通过检查不代表模型已学会目标；闭合命令只是输入阶段边界，不证明实际抓住，真实结果仍需人工记录。
同场景A／B是当前必做检验。沿用现有描边表达进行实验，不把其他论文的视觉提示收益直接视为本实现的结论；没有第三张图不等于分割与传输没有延迟。

### 同场景A／B成对示教方案

首对先用两根互不遮挡、均有下指空间的40 cm竹条；相机、支撑物、收集区及固定任务文字保持现状。
先抓A，放回并恢复两根的起始摆放，再从相同机器人起始状态示教抓B。对照起始全局图检查恢复情况；不能用“抓走A后只剩B”的录像充当成对选择。
采用可复现的现有摆放参考，目标ID与实体对应保持不变；若无法恢复，应另记场景，不把明显不同画面称为同场景对照。

| 现有采集字段 | 第一条：抓A | 第二条：恢复后抓B |
| --- | --- | --- |
| `scene_id` | `target_pair_001` | `target_pair_001` |
| `placement_reference` | 填写两根实际位置、角度及可复现参考 | 使用同一摆放参考，核对起始图 |
| `target_id` | `strip_001` | `strip_002` |
| `target_description` | A的实际位置、朝向 | B的实际位置、朝向 |
| `split` | `train` | `train` |

这些字段已有采集支持，不新增格式。目标描述按现场填写，不能只写“A／B”或复制不符合实际的位置。
后续摆放改变两根的位置与朝向，避免某个目标总对应固定抓取位置；A先／B先可交替。同组恢复录制仍属于同一split，验证和最终测试使用未参与对应开发的新摆放。

第一对的预览命令如下；默认只打印计划，不创建目录、不连接设备。两个路径必须尚不存在，不覆盖首批5条：

```bash
.venv/bin/python examples/phone_to_auboi10/record_joint.py \
  --target-demonstrations --camera-set original-global \
  --dataset-root datasets/joint_target_pair_pilot \
  --evidence-root artifacts/joint_target_pair_pilot --num-episodes 2 --split train
```

现场采集仍按[采集章节](#动作模型手机遥操采集)由操作者添加`--record`执行，每条只示教一次夹紧→释放。
采后首先核对两次起始摆放、所选身份与实际抓走对象，再做接近阶段标注。多实例不能直接复用首批“每帧唯一候选”的假设，须把每帧所选掩码对应到该条示教的目标。
首对通过后再确定成对采集数量、独立验证场景和训练清单；不从同一条轨迹拆帧凑验证集。真实训练和实机策略执行沿用各自的明确授权与现场操作入口。

| 事项 | 下一步与完成依据 |
| --- | --- |
| 感知识别 | 保留047～051首轮测试原结果；分析051第二阶段误分类，补充普通结构的独立开发数据。若据此调数据、阈值或模型，泛化成绩需另批未参与开发的测试。 |
| 目标条件化 | 主任务按上表及A／B方案推进；旧跟踪诊断冻结，不再优先扩充普通单根数据。 |
| 可下指区段 | 当前实验由操作者在摆放时保证下指空间，不开发自动区段判断，不把尺寸测量或该模块作为推进门槛；实验结论限定于这一场景条件。 |
| 物理结果反馈 | 磁性开关及控制柜DI（数字输入）接入待现场确认，可并行准备；早期A／B试验先人工判定，不等待自动成败系统。开关未闭合到底不独立证明夹住一根，需结合图像。 |
| 连续逐层抓放 | 独立记录抓对、恰好一根、送达、邻条扰动和耗时。允许抓取引起下层移动，不以零扰动判成功；移动不直接判失败，带起多根或影响继续作业等结果另记，抓后按新画面重新选择。预先固定清空／连续失败3次／超时的停止规则；按堆统计，测试规模和运行预算另定。 |
| 后续对照与纠错 | 保留同数据无描边VLA对照；脚本抓取仅作后续独立比较，不进入主线。保存完整运行及接管边界，再选择纠错片段训练；数据达到条件后安排数量泛化及消融。 |

`top_strip`不直接等于物理可夹；现有下层误判继续作为感知问题处理，与遮挡后的局部漏分分开判断。
旧双根数据已有两种上下摆放的抓放经验，见[已有模型与数据](EVIDENCE.md#当前双摆放模型)。复用优先从完整的第一次抓放片段开始，补当前帧目标轮廓；原始数据保留，以派生片段和来源映射接入。
旧示教含连续两次抓放，且没有目标元数据；当前目标接口首次闭合后本轮不再描边，不能把整条旧示教直接当作单目标样本。复用需适配片段边界、动作块与目标侧录，保持原训练／验证分组；它不能替代双可取目标A／B对照。
逐根摆放的差分标签仍须复核阴影、曝光与旧条被碰动的影响，历史完整形状不替代当前可见轮廓。
保留现有Mask2Former基线，不因斜条矩形框会重叠就断言其他实例分割方法不可用。局部深度仅在交叉处确有判断困难时再评估；论文贡献和泛化结论须由上述实验建立。

### 一般堆叠的下探高度修正方案

**状态：用户已确认纳入研究规划，尚未实现、训练或实机验证。** 先处理厚度8 mm、层高可以按整数层近似的一般堆叠；倾斜、架空及不规则支撑等特殊情形暂不展开，也不作为当前推进门槛。
每次仍选择一根目标并执行一次抓放，抓后重新观察。下指空间由操作者摆放保证，允许下层移动；本方案不要求零扰动。

- **高度基准**：以同一支撑面上单根竹条的抓取高度为基准，目标比该基准高`n`层时，参考高度增量为`n × 8 mm`；若目标是自下而上第`k`层，则`n = k - 1`。例如第二层为+8 mm、第三层为+16 mm，正方向为世界／基座竖直向上。
- **修改范围**：只作用于末端向下探至夹紧前的抓取段，使目标夹取高度上移；不是让夹爪额外向下8 mm。不平移整条轨迹，不修改抓取段以外的接近、夹爪开合逻辑、提起、搬运、收集区放置及返回动作，也不添加姿态修正。实际运动可能因新的抓取终点而变化，“其他动作不变”指不对那些阶段施加人工补偿。
- **层数来源**：`n`表示相对单根基准的额外层数，不直接取画面里的竹条总数。初期按已知摆放人工登记目标层数；现有`top_strip / covered_strip`只提供类别，自动层数估计尚未实现。
- **实现接口**：当前策略输出七维绝对关节动作，需将下探段关节目标通过正运动学转换为末端位姿，在竖直方向施加修正，再通过逆运动学转换回关节目标；保持该段原有横向位置和姿态。下探段边界、修正启停及与相邻阶段的连续衔接在实现时明确，不能仅凭夹爪张开就把所有动作都判为下探。
- **参考高度与模型预测**：`n × 8 mm`定义相对单根基准的高度差，不意味着不检查就叠加到任意模型预测上。实现时核对模型已预测的抓取高度，避免对已学到的层高重复修正；保留原始预测、目标层数、实际修正量及生效区间记录。
- **验证与评价**：先在已知层数的双根／三层一般摆放中，对照无修正与仅下探段修正；检查修正只在指定阶段生效，再记录抓对目标、是否夹住、是否只带一根及送达。结果标记为“VLA＋下探高度修正”，不混记为纯模型成绩。现有关节、工作区、速度与时效限制继续适用。

本节替代旧规划中对任何高度修正的一概排除，但不恢复历史的第二根补丁。当前提交只保存方案，现有实机命令和控制行为保持不变；具体实现和现场验证尚待后续开展。

### 指定目标输入：离线原型

本节记录已实现原型及其局限，当前开发顺序以上方主任务表为准；下面的诊断结果不是继续完善全程跟踪的任务清单。

本阶段采用当前全局图副本描出目标，保留腕部RGB和七维动作的方案；原始图不覆盖。
目标条件化先解决“本次要抓哪根”的输入表达，再用与该目标一致的示教训练模型。
第三张 `goal_image`（固定目标参考图）只是历史候选，本原型不使用它。

已实现 [joint_target.py](../../src/lerobot/bamboo_sorting/joint_target.py)：
显式选择实例 → 当前帧可见掩码 → `prepare_target_images` → `target_global_rgb`＋`grasp_rgb`。
实例PNG的像素ID只用于查找选中实例，不能当类别或跨帧身份。
`target_id`固定表示一次抓放的对象，`frame_id`约束轮廓对应的当前全局帧；这两个字段由调用者提供，接口检查一致性本身不证明跟踪正确。
掩码内侧2像素用RGB紫色`[255,0,255]`描边，被夹爪遮断的可见区域分别描边，不补全不可见部分。
接近阶段目标完全遮挡或丢失时拒绝生成输入；控制器如何处置仍待运行接入，尚未实现自动重识别。搬运阶段不要求掩码。
人工草稿仅在`allow_draft=True`时允许预览，并返回`preview_only=true`；正常入口要求`reviewed=true`。

关键帧复核原型仍保留`AuboI10JointTargetOutline`；实际训练／数值推理使用独立契约
`AuboI10JointTargetApproach`，图像键为`target_global_rgb`、`grasp_rgb`。
[joint_target_policy.py](../../src/lerobot/bamboo_sorting/joint_target_policy.py)提供共享阶段规则、
`JointTargetDataset`数据视图、`TargetInputSession`单轮输入准备和`SmolVLAJointTargetOfflineAdapter`数值推理适配。
数据视图保留原始视频、状态、动作及50步动作序列的尾部填充；新旧适配器拒绝对方的检查点契约。

`train_joint_smolvla.py --target-conditioned --data-manifest ...`启用该路径；清单每个来源需有
`target_annotations`，指向JSON侧录文件（路径相对清单）。默认`--stage plan`不加载模型；正式预检查仍检查原始采集证据、成功示教及场景划分。
侧录文件格式如下，`episodes`仅列需训练／验证的示教，`frames`必须覆盖全部接近阶段帧，不能用4张关键帧代替；
掩码是原分辨率二值可见区域PNG，路径相对侧录文件，原始实例PNG须先选中具体实例。

```json
{"schema":"aubo_joint_target_annotations","source_dataset_root":"原始采集的绝对路径",
 "episodes":[{"episode_index":0,"scene_id":"target_pilot_001","target_id":"strip_001","reviewed":true,
 "frames":[{"frame_index":0,"schema":"aubo_joint_target_frame","frame_id":"0/global_rgb/0",
 "target_id":"strip_001","status":"visible","reviewed":true,"mask_path":"masks/episode-0000/frame-000000.png"}]}]}
```

上例只展示格式，省略帧不能通过预检查；`reviewed=true`必须来自实际复核。
推理调用方先用`TargetInputSession.prepare`处理原图；仅在实际控制命令成功后调用`accept_command`，
不能使用预测动作提前切换。`joint_inference_stdio.py --target-conditioned`接收上述已处理图像及
`target_contract`、`target_input`（目标ID、帧ID、阶段）；默认旧推理协议保持不变。
当前`run_joint_smooth.py`尚未接入目标选择和这一会话，不能加个选项就用旧权重进行目标抓取。

**首批5条试采已完成，当前无需重录。** 数据位于`datasets/joint_target_pilot`，证据位于`artifacts/joint_target_pilot`；
录制固定复核入口为[review.html](../../artifacts/joint_target_pilot/review.html)，检查数字与边界见[证据索引](EVIDENCE.md#指定目标输入离线原型)。
原场景编号实际填写为`001`～`005`，现场描述为`1`～`5`；原记录保留，不自动改成此前建议的编号或补造描述。
五条全局起始画面均为单根、摆放朝向有变化，当前目标可辨；后续采集请填写明确的位置／方向描述，场景编号避免与历史数据混用。
目标ID均为`strip_001`，仅在各自一次抓放中解释身份。五条均为`train`流程试采，不能随后称为独立测试。

以下为本批原命令，仅供回溯；两个目录现在均已存在，入口会拒绝重复执行，不要删除数据来重跑：

```bash
.venv/bin/python examples/phone_to_auboi10/record_joint.py \
  --target-demonstrations --camera-set original-global \
  --dataset-root datasets/joint_target_pilot \
  --evidence-root artifacts/joint_target_pilot --num-episodes 5 --split train
```

本批已由操作者加`--record`完成采集；后续新批次另用新路径，操作见[采集章节](#动作模型手机遥操采集)。
本批接近阶段准备已完成：[固定复核页](../../artifacts/joint_target_pilot/review.html)已增加按episode的全帧浏览，
支持连续播放、逐帧／滑条定位、闭合前两秒跳转、原图／轮廓对照、原尺寸放大、异常记录及多块可见区域的多边形修改。
原有录制关键帧和检查入口保留。浏览器直接打开此HTML即可；相邻`target_annotations/`目录须保留。

所有标注派生产物集中在`artifacts/joint_target_pilot/target_annotations/`：
`annotations.json`覆盖1748个接近帧，`images/`保存按原视频时间提取的640×480 RGB PNG，
`inference_manifest.json`是已有冻结分割入口的输入。帧范围由时间侧录和已接受闭合命令重新计算，
每张图的解码时间与数据索引已核对，不以本批帧数作通用硬编码；1845帧搬运未进入标注清单。
原视频、动作数据和采集元数据未改写。**用户已完成本批复核，1748帧均已记录`reviewed=true`，接受用于流程试采；不是精确像素真值。**
每帧均有唯一预测实例，整帧掩码缺失0；但抽查发现局部漏分和遮挡处误分，不能把文件覆盖率当作轮廓正确率。
用户反馈“复核完毕,我认为遮挡后漏分是正常情况不必多虑”。按对本批整体复核的反馈推进流程试采，原掩码不改写；
9个已知误差保存在`accepted_annotation_limitations`和复核历史中，页面可定位查看，不再作为待修阻塞项。
其中不可见区域无需补全，仍可见的部分漏标与夹爪误标属于已知分割噪声；其对模型的影响尚未验证。
这些是抽查发现，不是全部错误清单。复核原话、范围与接受含义保存在`human_review`及`review_decisions.json`。

后续流程由[prepare_joint_target_review.py](prepare_joint_target_review.py)维护：

1. 本地`--stage prepare`已执行，拒绝覆盖已有标注目录，不要重跑。
2. 本批已获用户授权，完成一次1748帧冻结Mask2Former推理并回传；`inference_request.json`保存授权范围和执行状态，
   `inference.log`与`predictions/`保存执行及加载证据。远端采用本批新目录，未覆盖原实验。不要重新推理或重跑准备。
3. 预测已导入本目录`masks/`，下列命令仅供回溯，重复执行会拒绝覆盖现有掩码。只有唯一候选才生成未复核二值掩码；
   实例像素ID通过`segments`解释，多个候选／空检测分别记为歧义／缺失，不自动合并或沿用上一帧。
   同一实例内部的多块可见区域全部保留；唯一候选仍不证明目标身份正确。

```bash
.venv/bin/python examples/phone_to_auboi10/prepare_joint_target_review.py \
  --stage import-predictions \
  --predictions artifacts/joint_target_pilot/target_annotations/predictions/predictions.json
```

4. 本批已通过用户对话反馈完成复核并写入侧录。后续若需修正，可在固定页面播放、定位、修正后导出`joint_target_pilot_review_decisions.json`。
   浏览和播放不会自动确认；只有操作者实际检查完整区间后，才点“明确确认这段已有轮廓”。
   缺失、异常或有未栅格化修改的帧不能确认，侧录中已记录的问题同样阻止区间确认。导出文件不会自动写回训练侧录。
   将实际导出路径传给`--stage import-review --decisions <导出JSON路径>`，修改后的掩码仍保持草稿，
   需重新打开页面检查紫色内侧2像素轮廓后再确认。确认问题已修好时，用“记录当前异常”清空该问题文字，再明确确认。
   多边形替换当前帧的全部可见区域，须画全遮挡两侧；不传播到相邻帧。
   原掩码保存在该帧`previous_masks`的PNG字节记录，复核操作保存在`review_history`，修订号防止旧导出覆盖新修改。
5. `--stage render`仅从现有侧录更新同一个页面和派生统计。新增`--stage verify`读取实际`LeRobotDataset`，
   经`prepare_target_source`与`JointTargetDataset`检查完整5条图像身份、阶段、状态／动作／50步填充和训练／运行标记一致性。
   本批3593帧已全部通过，见[dataset_check.json](../../artifacts/joint_target_pilot/target_annotations/dataset_check.json)，无具体疑点无需重复全量读取。
   `target_source_ready=true`表示本批数据接口可用；因独立验证数据和正式训练授权尚未具备，`target_training_ready=false`保留。
   当前不创建虚假的validation，不训练。检查只读取本地数据，不加载动作模型或连接硬件。

以下为已冻结的旧录像诊断；其固定入口是[旧指定目标输入复核页](../../artifacts/joint_target_review/index.html)，与上述本批示教分开。
[annotations.json](../../artifacts/joint_target_review/annotations.json)保存来源录像、帧号、时间记录、固定目标ID和4帧已确认人工轮廓，保留初稿来源及纠错记录；
[review_joint_target.py](review_joint_target.py)只读取已保存录像，生成内嵌图片的HTML，不运行模型或连接硬件；默认拒绝覆盖，`--update`显式更新同一入口。
页面可切换抓前、夹持、搬运、收集处关键帧，比较原图、目标图和腕部图。
用户已确认4帧轮廓正确；这仅确认所展示关键帧，中间帧仍未人工确认，本次自主执行录像也未作为专家示教。

已用[track_joint_target.py](track_joint_target.py)对保存录像完成光流基线诊断。
光流是由相邻图像估计像素移动；实现为`joint_target_tracking.py`，只用本机OpenCV，不调用神经网络。
它逐帧传播可见掩码，光流往返误差、灰度差或面积变化过大时停止；这些是未校准的诊断条件，不证明身份或遮挡判断正确。
40、180、222秒均先比较传播结果，再用已确认人工轮廓重新初始化；这是有人工辅助的开发诊断，不能当独立测试或自主连续跟踪。
同一页面的“失败案例：旧光流回放”默认折叠，保留历史证据，不作为当前跟踪方案。
绿色是人工关键帧、橙色是未复核草稿；`lost`后直到下个人工关键帧不再传播。
结果表明丢失检测还会漏掉漂移到背景的情况，此基线不能生成训练标签，详见[证据](EVIDENCE.md#指定目标输入离线原型)。

已按用户确认的下一步，用冻结分割模型检查0、25、27.84、40、80、180、222秒，仍使用原尺寸与原阈值。
[predict_strip_images.py](predict_strip_images.py)默认只检查输入并打印方案，`--execute`才加载模型推理，不创建优化器或保存权重。
诊断输入与结果在原复核目录的`segmentation/`内，页面新增“冻结分割模型”对照区；这是录像帧诊断，不重导出训练数据。
只有已确认4帧计算轮廓重合度；用真值寻找最佳重合实例仅为离线分析，不是实际目标选择算法，也不计算全场景FP／F1。
结果支持继续尝试逐帧分割定位，但遮挡侧存在整段漏分，不能直接生成完整训练标签；具体数值和模型位置见[证据](EVIDENCE.md#指定目标输入离线原型)。

跨帧关联原型已实现于`joint_target_association.py`，由`associate_joint_target.py`读取已保存分割结果，
结合预测中心、长轴方向、面积及形状重合匹配已选对象；匹配有歧义、缺帧或明显不相符时记录持续失效。
形状比较支持平移和旋转对齐，但这些变换只用于打分，输出始终是当前帧的新掩码，不传播或补全旧轮廓。
`TargetReconfirmation`在空检测时进入`unobserved`（暂时未观测），最多保留最后接受状态之后5帧的关联记忆；
候选重现先进入`confirming`（待确认），连续两张不同来源相机图像通过原目标与候选轨迹的一致性检查后，才输出`reconfirmed`。
重新确认比正常关联更严格：中心相对原目标≤30 px、长轴差≤15°、面积比0.5～2、形状重合≥0.5、综合分≥0.6。
重复来源帧不增加确认次数，确认中再次漏检会清空候选确认次数；窗口到期、歧义或不匹配进入`lost`，必须显式重新选择。
`unobserved`、`confirming`和`lost`均不输出目标掩码；旧几何状态只保留在内部用于比较，不涂到当前画面。
这只是受约束的几何重新确认，不是经过多目标验证的身份识别；实例编号相同或固定`target_id`不能证明身份正确。
当前固定条件写入原复核目录`association/reconfirmation_plan.json`，代码入口拒绝覆盖既有结果；
旧的仅平移比较结果保留作诊断对照，数值门限未放宽。网页最前面的“本次24～30秒关联回放”为当前结果。
该片段共151帧，复用25秒和27.84秒两帧，仅补149帧冻结模型推理；结果与局限统一见[证据](EVIDENCE.md#指定目标输入离线原型)。

本轮完全复用151帧既有预测实现并检查“未观测→重新确认”：29.80秒漏检、29.84秒待确认、29.88秒恢复关联；
29.96秒再次漏检，30秒仅有第一张确认图，片段结束仍保持待确认，没有强行恢复或回填中间帧。
上述诊断现已冻结，不继续补帧或要求整段跟踪通过。单根录像不足以验证相邻多目标竞争，
这些已查看场景的开发结果不用于声称独立测试成功率。
后续按主任务表推进。`record_joint.py --target-demonstrations`已补充所选对象ID及现场描述，
仍复用`JointCaptureSession`的原始双RGB、七维状态／动作、帧索引和时间侧录；逐帧目标标注待采后离线复核。
`run_joint_trial.py`的`InferencePipe`与`joint_inference_stdio.py`是后续运行接入位置，目前保持原行为。
训练与运行必须复用同一标记方式，保留原图和动作标签；新的示教、训练及实机实验分别明确授权后执行。
不能把抓A的示教改标为目标B，也不能只改提示词或临时涂色后声称旧模型会服从指定目标。

单根或只有一个可取上层的成功不能证明目标服从；当前优先执行[同场景A／B成对方案](#同场景ab成对示教方案)。
当前下指空间由现场摆放保证；指定局部下指区段不列为本阶段必做任务。

双可取目标已纳入当前实验；一根压两根、拥挤间隙等复杂堆叠及交换上下的关联对照暂缓。
关联对照须同split；不把旧150组预算或历史研究建议当成本轮采集命令。
斜视RGB不能证明三维无碰撞或统一像素／毫米比例；局部深度、SAM修边、VLM/RL均未接入当前流程。

## 代码导航与维护

| 任务 | 入口与实现 |
| --- | --- |
| 静态拍照／人工复核 | `capture_placement_sequence.py`、`review_placement_sequence.py`；`src/lerobot/bamboo_sorting/placement_sequence.py`、`placement_review.py`、`placement_review.html` |
| 分割导出／训练／评价 | `export_strip_segmentation.py`、`train_strip_segmentation.py`、`evaluate_strip_segmentation.py`、`review_strip_predictions.py`；`strip_segmentation_data.py`、`strip_segmentation_training.py` |
| 动作采集／训练／运行 | `record_joint.py`、`audit_joint_dataset.py`、`train_joint_smolvla.py`、`run_joint_smooth.py`；`run_joint_trial.py`提供归位与共享实现 |
| 指定目标离线接口／复核 | `joint_target.py`、`joint_target_tracking.py`；`review_joint_target.py`、`track_joint_target.py`，后者仅是未通过的光流诊断基线 |
| 指定目标训练／数值推理 | `joint_target_policy.py`；训练和stdio推理入口的`--target-conditioned`，采集入口的`--target-demonstrations` |
| 冻结分割模型录像帧诊断 | `predict_strip_images.py`，默认预检查，`--execute`才运行；预测与人工轮廓由原目标复核页对照 |
| 新分割掩码跨帧关联 | `joint_target_association.py`、`associate_joint_target.py`；当前为离线几何关联原型，结果未接入动作模型 |
| 机械臂驱动 | `src/lerobot/robots/aubo_i10/` |
| 历史层序评价 | `evaluate_layer_order.py` / `layer_order_eval.py`，只比较旧双图成对关系，不加载识别模型；不能代替当前单图实例分割评价。格式及旧方案从[历史索引](EVIDENCE.md#文档整理与历史原文)回溯。 |

日常使用项目 `.venv/bin/python`，按改动选相关 `tests/bamboo_sorting/` 测试，再运行 `git diff --check`。
分割相关合成测试需要补充依赖，当前可用 `PYTHONPATH=/tmp/aubo-segmentation-deps:src`；测试通过不代表物理成功。
纯文档整理只检查内容、相对链接和差异，不重跑训练、推理或无关测试。
采集仍复用 `record.py`、`record_c0_batch.py` 和部分 `c0_*`，不能按旧名称删除。

后续维护遵循 [AGENT.md](../../AGENT.md#文档维护规则)：新增工作先更新本文对应章节，结果只在证据索引维护。
按主题保留当前结论，旧讨论通过Git回溯；不逐轮追加长流水账，不生成日期版计划、交接、总结或重复review目录。
相机、机械臂、训练、删除需各自授权；软件改动不自动部署或推送。高度修正仅按[已确认范围](#一般堆叠的下探高度修正方案)推进，不扩展为强制姿态或脚本任务轨迹。

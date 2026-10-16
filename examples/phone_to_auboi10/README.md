# AUBO i10 项目手册

更新：2026-10-16。本文集中维护当前状态、操作方法和后续计划；实验指标与原始路径只维护在
[证据索引](EVIDENCE.md)。所有命令从仓库根目录 `/home/rentao/program/lerobot-aubo` 执行。

## 目录

- [当前状态与系统边界](#当前状态与系统边界)
- [三根堆叠目标示教方案](#三根堆叠目标示教方案)：当前采集、标注与试采主线
- [代码导航与维护](#代码导航与维护)：按任务找入口，区分共享模块与历史工具
- [相机配置](#相机配置)
- 静态分割：[采集与人工复核](#静态分割采集与人工复核)、[数据、训练与独立评价](#静态分割数据训练与独立评价)
- 动作模型：[手机遥操采集](#动作模型手机遥操采集)、[训练与离线评价](#动作模型训练与离线评价)、[现场运行](#动作模型现场运行)
- [目标条件接口约定](#目标条件接口约定)：标注格式、训练／运行共用规则
- [斜条J6欠转的腕部角度诊断](#斜条j6欠转的腕部角度诊断)：只待现有人工标注收束，不阻塞堆叠试采
- [后续研究与待办](#后续研究与待办)
- [历史流程入口](#历史流程入口)：已结束的A／B示教、单根试采与跟踪诊断

先按当前任务进入对应章节。完成记录、历史参数和逐批结果统一查 [EVIDENCE.md](EVIDENCE.md)，不把旧批次的命令当作新实验授权。

## 当前状态与系统边界

当前有两条独立流程。实例分割指为每根木条识别可见像素区域；动作模型根据图像和机器人状态预测运动。

| 流程 | 输入 → 输出 | 当前状态 |
| --- | --- | --- |
| 静态 Mask2Former 分割 | 固定全局 RGB → 可见实例掩码、`top_strip` / `covered_strip` | 首轮训练、验证轮廓复核、047～051新摆放独立测试均完成；发现被压条误判上层，详见[结果](EVIDENCE.md#分割模型与独立测试)。 |
| 七维 SmolVLA 抓放 | 全局＋腕部 RGB、七维状态、任务文字 → 七维绝对关节动作 | `mixed-both-orders` 已训练；熟悉长条摆放及当前40 cm单根均有成功反馈，见[实机证据](EVIDENCE.md#40-cm单根试抓)；支持指定命令周期数。 |
| 感知与执行闭环 | 指定目标 → 单次抓放 → 重新观察 | 目标条件模型`target-pair`已接入`run_joint_smooth.py`并开展监督下单次试验；双目标前景分离后新摆放A／B身份基本保持，遗留问题是斜放目标的抓取姿态：闭合J6随倾角系统性欠转，横向对中不稳定。物理结果自动判断及自动清空未实现，见[新摆放复核](EVIDENCE.md#重新摆放后的指定ab抓取)。A／B实验已阶段性结束，转入[三根堆叠](#三根堆叠目标示教方案)。 |

静态研究与近期目标动作试采使用 **400×25×8 mm** 短条；旧160条动作训练示教使用60 cm长条。
当前目标服从实验沿用40 cm、原机位和双640×480，不同时切换长度与相机；60 cm及混合长度留待后续明确验证。
不能把短条分割成绩外推为短条抓取或任意无序堆叠成功率。特殊结构、VLM（视觉语言模型）、RL（强化学习）暂缓。

动作契约为 `[J1,J2,J3,J4,J5,J6,gripper_pos]`，关节单位为度，进入机器人 SDK 时转换为弧度；J5由模型预测。
`CameraSetV2` 的 `global_rgb` 为固定全局相机，`grasp_rgb` 为腕部相机，均为640×480。
末端为 Airtac HFKL20 气动平行二指夹爪：0松开、100夹紧；这只是命令，不是持物传感器。
旧字段 `suction_*` 与历史“吸盘/吸附”均指夹爪开合；DO2夹紧、DO3松开。
固定任务文字为 `Pick one strip and place it in the collection area.`。

## 三根堆叠目标示教方案

**状态：2026-10-16完成本地软件准备，尚未采集或训练三根堆叠数据，未部署新远端服务。** 软件验证只用假设备和合成数据；复核页尚未完成实际浏览器交互验收。采集、推理、训练及实机运行仍需各自的现场授权，现场命令由操作者执行。
原“下探高度修正”方案已取消（见[分阶段路线](#分阶段路线)），堆叠层高由模型从示教中学习，本方案不加任何高度、姿态或轨迹补偿。

**场景规格**

- 3根400×25×8 mm竹条，在右侧堆放区支撑台上交叉堆叠（如一根压两根、两两交叉），两两交角约不超过30°；每根待抓上层条由操作者摆放时保证下指空间。
- 摆放类型要保证有足够多“至少两根可选上层条”的步骤：例如两根压一根，第一步有两个可选目标；一根压两根，第一步只有一个可选目标，第二步才可能有两个。各类型数量在试采后按此平衡，并在记录中登记每步可选上层条数。
- 相机（`original-global`，双640×480）、机器人起点、左侧放置区及固定任务文字保持不变，不同时切换长度或机位。
- 各堆之间改变交叉位置、朝向和上下关系，不按毫米网格摆放；最终测试堆事先指定，不参与开发和调参。

**采集方式：现场连续清堆，每抓一根存为一条示教**

1. 摆好一堆，按`r`归位；操作者选定当前一根上层条作为目标，按`→`开始，手机示教抓起、送到放置区、松开，按`→`结束并如实填写结果。
2. 不恢复摆放，在抓走后的实际剩余状态上继续下一条，仍从同一起点开始；邻条被碰动也保持原样。3根抓完即本堆结束，一堆产出3条。
3. 每条只有一次夹紧→释放，与当前目标接口“首次闭合后不再描边”及推理时“每抓一根后重新观察”一致。旧双根示教一条含两次抓放，不能直接当作单目标样本。
4. 抓错、带起多根、滑落等如实记录，不改标成目标示教；之后是否继续本堆由操作者判断并记录。
5. 先抓哪根在各堆之间轮换，避免模型学成固定“抓最左”“抓最上”等捷径。

**元数据与划分**

- `scene_id`为堆编号，同一堆各条共用：训练用`p-001`起，验证用`p-101`起，不与历史`t-*`混用。`target_id`为该堆内竹条编号`s-001`～`s-003`；不同竹条不共用ID，若失败后再试同一根仍沿用其ID。
- 新增`--stack-demonstrations`，自动启用目标示教；每条询问`pick_step`（本堆保存的抓取尝试序号1～6）和`available_target_count`（当时可选且有下指空间的上层条数1～3）。失败也占一次已保存尝试，未保存的重录不占；同一会话内同堆序号递增，允许不完整堆，不从序号推算剩余根数。每堆最多保存6次尝试，为失败后人工重试留出余量；可选条数仍最多3根，不随尝试上限增加。自动重试属于后续阶段。文字描述字段仍留空，旧数据无需补填新字段。
- 结果除原`human_outcome`外保存`target_outcome`：`target_matched`、`held`、`single_strip`、`delivered`、`neighbor_disturbed`，分别为抓对、夹住、单根、送达和邻条扰动，取`true/false/null`。前四项明确为真、原结果为`single_success`且恰好一个完整闭合／释放周期，才成为成功目标示教候选；未知或失败保留在原记录中，不自动改标成功。
- 训练集和验证集按整堆划分，同一堆各步不得拆到两边；泄漏检查包含被排除的失败步骤所在堆，避免只检查成功样本而漏掉同堆关联。

**目标轮廓标注（离线）**

- 现有按左右矩形区域筛选前景的办法只适用于两根分离，交叉堆叠不适用。复核页新增首帧种子模式，显示全部分割候选；可点选候选或画多段可见区域多边形，为目标和其余可见竹条分别指定身份。点选候选与多边形两种方式对每根互斥；合并实例不能直接同时分给两根。
- 后续接近帧用冻结Mask2Former的当前帧分割，加多根身份分配生成草稿。现有`PairedStripSeparation`限定两根，按像素到中心线的距离分配归属；交叉处两根中心线都很近，距离无法判断上下遮挡，因此不能只把数量从2改成N。推广时以当前帧分割实例和人工可见区域种子为主、中心线为辅；输出只含当前帧前景，不补遮挡，不沿用旧掩码。“只用当前帧前景”不保证像素分给了正确竹条，交叉处的归属须在复核中重点检查。
- 交叉处一个实例覆盖两根时，须拆分或登记为已知误差；草稿经用户复核接受后才进入训练，沿用`reviewed`与`accepted_annotation_limitations`约定。
- 实时入口`run_joint_smooth.py`的目标输入同样推广到N根种子，每次抓放前重新拍照圈选；初期人工选目标，之后可由`top_strip`候选加规则选择。

新类`MultiStripSeparation`保留有主要种子支持的当前实例，合并实例按可见区域距离拆分，中心线距离只作次要依据；等距歧义像素不分配，拆分处标记复核问题，不用拆分结果更新身份参考。小运动关联候选范围为8像素，主支持判定为至少占种子重叠的80%且覆盖当前实例50%；这些是草稿关联启发式，不是已验证的识别精度或物理安全阈值。只在合成图上验证，真实遮挡、明显位移、颜色混入和分割合并仍可能失败；旧`PairedStripSeparation`及其默认行为保留。

**规模与训练**

- 先试采2～3堆（约6～9条），走通“连续清堆采集→多实例标注→数据接口验收”，再定正式规模（初步约20～30堆训练、约60～90条，另加独立验证堆）。
- 训练起点待试采后决定：在`target-pair`检查点上用堆叠示教微调，或合并成对与堆叠数据从基座训练。同数据无描边基线同步规划，用于核心证据“模型服从描边”。

**评价**：每次指定目标，分别记录抓对、夹住、恰好一根、送达及邻条扰动，以及当时可选上层条数k。目标服从只统计k≥2的步骤；1／k仅是“均匀随机选择”的参考水平，主要对照是同样步骤下无描边基线的实际选择分布。k=1的步骤只用于抓放能力统计。中止轮记为未知，不剔除。

**软件入口与使用顺序**

以下预览总计6次保存尝试的采集计划，不创建目录、不连接设备；它约对应无失败时的2堆，失败也计入总数，不保证清完2堆。每堆`pick_step`上限6与整批`--num-episodes`是不同限制。现场另行授权后由操作者添加`--record`，每条仍按`r`归位、`→`开始／结束。

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/record_joint.py \
  --stack-demonstrations --camera-set original-global --split train --num-episodes 6 \
  --dataset-root datasets/joint_stack_pilot --evidence-root artifacts/joint_stack_pilot
```

采后始终显式传入同批`--dataset-root`和`--evidence-root`，依次调用`prepare_joint_target_review.py`：

1. `--stage capture-check`：新批次生成`raw_audit.json`、`capture_check.json`和固定`review.html`；完整解码双路视频并核对帧时间，保存少量关键帧。已有检查或复核产物时拒绝覆盖。
2. `--stage prepare`：提取接近阶段当前RGB，准备原有推理清单；不运行分割模型。获得另行授权后才对该清单运行冻结分割。
3. `--stage import-predictions --predictions /本批预测报告.json`：保存本批当前实例图和首帧候选；堆叠不得使用`--single-strip-scenes`。
4. 打开本批`review.html`，在每条首帧为全部可见竹条点选／画种子并明确确认，导出JSON；`--stage import-seeds --decisions /导出的记录.json`导入。
5. `--stage separate`生成全部接近帧草稿，离线与实时调用同一分离器。不覆盖已有掩码；发现误差用页面逐帧修改，或“登记可接受的已知误差”，再`--stage import-review --decisions /导出的记录.json`。修改后必须查看重新生成的轮廓，明确确认范围后再次导入；浏览页面不会自动复核。
6. `--stage verify`只对通过原始审计的成功目标示教做真实LeRobot/PyAV接口验收，报告排除的失败／未知episode；检查原始状态、动作、50步动作块尾部填充及训练／运行阶段一致。数据就绪仍不等于训练或物理成功。

实时软件新增`--multi-target-separation`，与旧`--paired-target-separation`互斥；沿用`--model target-pair`的目标输入和单次抓放契约，支持1／2／3根剩余条的圈选。模型仍是既有检查点，**不是已训练的堆叠模型**；新服务路径为远端既有项目下`artifacts/joint_target_live_service/multi_foreground/`，本轮只准备路径和协议，未部署或启动。真实试验前须核对服务、模型与新现场圈选，不能直接把本地软件完成视为可运行部署。

合成页面、检查结果及回滚位置统一见[软件准备证据](EVIDENCE.md#三根堆叠软件准备)。[腕部诊断工具](#斜条j6欠转的腕部角度诊断)独立于堆叠采集流程。

## 代码导航与维护

命令入口在 `examples/phone_to_auboi10/`，任务实现集中在 `src/lerobot/bamboo_sorting/`；机器人驱动在 `src/lerobot/robots/aubo_i10/`。先按下表定位，不按文件名猜测当前方案。

### 当前任务入口

| 要做的事 | 入口 | 主要实现／操作说明 |
| --- | --- | --- |
| 三根堆叠目标示教 | [record_joint.py](record_joint.py) 的 `--stack-demonstrations` | `aubo_joint_capture.py`、`joint_stack_metadata.py`；[堆叠流程](#三根堆叠目标示教方案) |
| 采后检查、种子和逐帧轮廓复核 | [prepare_joint_target_review.py](prepare_joint_target_review.py)、[joint_target_review.html](joint_target_review.html) | `capture-check → prepare → import-predictions → import-seeds → separate → import-review → verify`；分割推理另行授权 |
| 审计原始关节数据 | [audit_joint_dataset.py](audit_joint_dataset.py) | `aubo_joint_audit.py`、`joint_gripper_quality.py`；成功候选筛选不代替物理结果确认 |
| 多实例分离与目标输入 | [joint_target_separation.py](../../src/lerobot/bamboo_sorting/joint_target_separation.py)、[joint_target_policy.py](../../src/lerobot/bamboo_sorting/joint_target_policy.py) | `MultiStripSeparation`由离线与实时共用；[接口约定](#目标条件接口约定) |
| 动作训练／离线评价 | [train_joint_smolvla.py](train_joint_smolvla.py)、[diagnose_joint_target_policy.py](diagnose_joint_target_policy.py) | `joint_training.py`、`smolvla_joint_adapter.py`；[训练阶段](#阶段与默认值)，已有诊断不重复运行 |
| 目标圈选与现场运行 | [run_joint_target_shadow.py](run_joint_target_shadow.py)、[joint_target_shadow_review.html](joint_target_shadow_review.html)、[run_joint_smooth.py](run_joint_smooth.py) | `joint_target_live_setup.py`、`joint_target_runtime.py`、`joint_smooth.py`；[运行说明](#目标条件模型单次真机试验)，新堆叠服务尚未部署 |
| 静态分割采集／人工修边 | [capture_placement_sequence.py](capture_placement_sequence.py)、[review_placement_sequence.py](review_placement_sequence.py) | `placement_sequence.py`、`placement_review.py`／`.html`；[采集与复核](#静态分割采集与人工复核) |
| 分割数据／训练／预测评价 | [export_strip_segmentation.py](export_strip_segmentation.py)、[train_strip_segmentation.py](train_strip_segmentation.py)、[predict_strip_images.py](predict_strip_images.py)、[evaluate_strip_segmentation.py](evaluate_strip_segmentation.py)、[review_strip_predictions.py](review_strip_predictions.py) | `strip_segmentation_data.py`、`strip_segmentation_training.py`；[分割流程](#静态分割数据训练与独立评价) |
| 腕部20帧人工结果导入 | [review_joint_wrist_distribution.py](review_joint_wrist_distribution.py) 的 `--stage errors` | [人工复核流程](#斜条j6欠转的腕部角度诊断)；`diagnose_joint_wrist_angle.py`保留已完成测量的复现入口 |

### 共享依赖与现场辅助

这些文件仍被当前代码调用，旧名称不表示可删除。`examples/`中的脚本也并非全都适合直接运行。

| 文件／模块 | 当前用途 |
| --- | --- |
| `record_c0_batch.py → record.py` | `record_joint.py`复用相机新鲜度检查和手机归位辅助；旧C0批次CLI已退役 |
| `c0_batch_capture.py`、`c0_gripper_*` | 关节采集的帧时间、夹爪命令与保存证据；不是当前8维末端动作方案 |
| `joint_camera_config.py → c0_smoke_capture.py` | 当前相机配置与冻结映射；配置文件仍在 `configs/aubo_i10/` |
| `run_joint_trial.py` | `run_joint_smooth.py`、目标影子入口等复用的 `InferencePipe`、`ReadOnlyStation`，以及[归位入口](#2-归位) |
| `run_joint_live.py` | 保留停稳后单步观察的兼容入口；持续伺服入口还复用其任务文字读取函数 |
| `joint_inference_stdio.py` | 远端模型数值推理协议；由运行链调用，不是本地机械臂控制入口 |
| `aubo_start_poses.py`、`move_to_start.py`、`release_joint_gripper.py`、`teleoperate.py` | 起点定义、移动、单独松爪和手机遥操；涉及现场动作，须按相应授权执行 |
| `diag_preflight_cams.py`、`diag_cam_latency.py`、`read_pose.py` | 相机体检／延迟、机器人位姿读取；读位姿也需要设备连接 |

### 历史与专项工具

| 文件 | 状态与适用范围 |
| --- | --- |
| `review_joint_target.py` | 保存录像的关键帧与已有诊断结果展示；不能生成已接受训练标签 |
| `associate_joint_target.py` | 保存分割结果的离线关联诊断；底层 `joint_target_association.py` 仍被实时路径使用，不能一起移除 |

旧ACT／C0聚合审计、双图层序评价、光流诊断和一次性现场探测入口已退役；删除范围、目录保留理由及源码回溯见[EVIDENCE](EVIDENCE.md#目录与退役代码清理)。历史记录中的旧命令不再是当前可用入口。

软件验证按实际改动选择 `tests/bamboo_sorting/` 及相关驱动用例，使用项目 `.venv/bin/python`。纯文档整理只检查差异、链接与回溯位置，不重跑训练、推理或无关测试。分割测试的额外依赖路径以本机实际存在的环境为准，不把历史 `/tmp` 路径当长期配置。

`datasets/`、`examples/phone_to_auboi10/datasets/`、`artifacts/`、`pile-json--/`中的数据和人工结果保持原位。源码与测试按完整调用链维护；测试已按用户确认的AUBO范围裁剪，无关框架功能的实现仍保留，但不再拥有原完整测试覆盖。目录名含 `test`、`images` 或 `c0` 不能作为删除依据。文档只维护本手册和[EVIDENCE](EVIDENCE.md)，不新增逐轮计划或总结。维护约定见[AGENT.md](../../AGENT.md#文档维护规则)。

### 测试目录入口

| 修改范围 | 优先检查的测试 |
| --- | --- |
| AUBO采集、夹爪、目标输入、堆叠、分割、腕部复核 | `tests/bamboo_sorting/`中对应文件；其中`c0_*`仍覆盖当前共享时间戳、夹爪和相机契约 |
| AUBO驱动与手机动作处理 | `tests/robots/test_aubo_*.py`、`tests/processor/test_aubo_*.py` |
| SmolVLA损失、动作补齐与处理器兼容性 | `tests/policies/smolvla/`、`tests/processor/test_smolvla_processor.py`；`test_relative_xyz.py`含旧表示兼容检查，不代表当前采用相对TCP动作 |
| 通用录制、数据集、视频与处理器 | `tests/scripts/`、`tests/datasets/`、`tests/processor/`、`tests/test_control_robot.py` |
| 公共配置、优化器、工具与测试辅助 | `tests/configs/`、`tests/optim/`、`tests/utils/`、`tests/fixtures/`、`tests/mocks/`；`tests/motors/test_motors_bus.py`保留通用总线与假设备支持 |
| 相机或模型集成 | `tests/cameras/test_opencv.py`、`tests/training/`、`tests/test_cli_peft.py`等需先检查具体用例的设备／下载／训练条件，不是默认现场检查命令 |

本地共有81个`test_*.py`文件；不是所有保留测试都适合直接执行。只检查加载与用例发现可运行 `PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 .venv/bin/python -m pytest --collect-only -p no:cacheprovider -q tests`，它不执行测试函数，不等于测试通过。不要以裸跑全部`pytest tests`代替按改动选择验证。删除范围及回溯见[测试裁剪记录](EVIDENCE.md#按aubo用途裁剪测试)。

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

本节1～5小节以自选目标的 `run_joint_smooth.py --model mixed-both-orders --approach-age-aligned` 为例，下表限制也是该模型的数值；
已有目标条件运行入口使用 `--model target-pair`，其参数与命令只维护在[目标条件模型单次真机试验](#目标条件模型单次真机试验)。
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
当前代码不提供 `--second-grasp-z-mm` 等高度补偿参数；下探高度修正已从研究计划中取消，见[分阶段路线](#分阶段路线)。

<a id="当前实验40-cm单根基线"></a>

#### 40 cm单根基线（历史）

该基线已完成，原现场反馈、参数与录像见[实机证据](EVIDENCE.md#40-cm单根试抓)。它检验旧模型对短条的适应性，不证明目标服从或堆叠抓取。当前工作转入[三根堆叠试采](#三根堆叠目标示教方案)，不再把单根旧命令作为当前任务。

### 4. 当前动作与保护

动作块是模型一次预测的未来 50 步完整七维动作。夹爪指令为松开时，按观测年龄选取整行，
遇到首次预测闭合边界时不越过它；等待到位期间继续单个在途观测/推理，新结果可更新或取消闭合。
夹紧后沿用首步预测；IO 切换后废弃旧动作块并丢弃切换前观测的在途结果。
当前实现没有接触高度修正、关节偏置、脚本化下降或提起，研究计划也不再引入下探高度修正。

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

### 当前模型的现场影子对照

这是保留的辅助入口：`run_joint_target_shadow.py`支持`plan/capture/predict`，没有伺服或夹爪写入。`capture`连接设备获取静止快照，`predict`加载模型计算保存观察上的动作，分别需要相应授权；都不等于抓放验证。
既有目标互换对照已完成，快照和预测证据见[成对模型记录](EVIDENCE.md#成对训练数据采后检查)。当前不重跑旧场景；新现场圈选须绑定新的snapshot ID，不能使用旧照片执行动作。

### 目标条件模型单次真机试验

`run_joint_smooth.py --model target-pair`保留既有目标条件模型的单次抓放入口；A／B现场实验已经阶段性结束，历史结果见[证据索引](EVIDENCE.md#目标条件模型最新ab现场结果与b失败复核)。新堆叠服务未部署、模型未训练，不能直接当成堆叠部署入口。

| 选项 | 行为与边界 |
| --- | --- |
| 不加分离选项 | 按最高分选择当前分割候选；实时位置／方向／面积／重合度／分数拒绝门限已取消，评分不证明身份正确，完全无候选时停止 |
| `--paired-target-separation` | 已有双目标前景分离路径，限两根；保留历史部署 |
| `--multi-target-separation` | 与双目标选项互斥，使用全部可见种子和当前实例；仅完成本地软件准备，服务路径见[堆叠章节](#三根堆叠目标示教方案) |
| `--transport-age-aligned` | 搬运按观察年龄取完整动作行，不跨过首次释放边界，到位后才松爪；省略时保留搬运首步行为 |
| `--latency-tolerant` | `target-pair`预测年龄与无新预测看门狗均放宽至1.5秒，通信等待5秒；相机帧龄250 ms／时差100 ms保持，不能与`mixed-both-orders`的表格混用 |

以下仅为**已有双目标路径的计划模板**，用本次真实快照、导出和目标ID替换占位符；不加`--execute`不连接设备或加载模型：

```bash
.venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
  --model target-pair --target-snapshot "/本次快照目录" \
  --target-selections "/本次实际导出.json" --target-id "本次目标ID" \
  --paired-target-separation --transport-age-aligned --latency-tolerant
```

现场执行须另行授权，由操作者添加`--execute --output "/新的记录目录"`。起点各关节误差须≤1.5°、夹爪松开；需要归位时先按[归位流程](#2-归位)执行。保持圈选时的相机和竹条摆放，改变后重新拍照圈选；现场任务文字使用本手册固定值。
目标入口限定一个夹紧→释放周期、最多120秒，速度／加速度为5°/s、20°/s²，工具速度0.05 m/s。默认预测年龄0.5秒、无新预测看门狗0.6秒。接近时按观察年龄取整行动作，不跨过最早闭合边界；实际闭合成功后按相机来源时间切换搬运输入，并丢弃闭合前在途预测。
只输出本帧前景；自动掩码不标成人工复核，少量或错误前景仍可能进入模型。Ctrl+C／异常／超时停止伺服，不自动释放或归位；命令周期完成不等于物理成功。完整历史命令与被取消的位置／角度对照从[整理前检查点](EVIDENCE.md#文档整理与历史原文)回溯。

## 目标条件接口约定

首版接口已实现“接近／夹取阶段标记目标，首次已接受的夹爪闭合命令之后使用原始双RGB完成搬运”，
`target_global_rgb`仍为同一个处理后图像字段，腕部图与七维状态／动作保持原定义。
这使本次指定对象在抓取前表达清楚，不要求夹持搬运全过程都有完整分割轮廓；训练与运行必须共用相同的阶段切换规则。
闭合动作对应的输入仍有标记；全局相机来源时间严格晚于闭合命令接受时间才切换，释放后本轮也不重新描边。
软件接口通过检查不代表模型已学会目标；闭合命令只是输入阶段边界，不证明实际抓住，真实结果仍需人工记录。
（历史计划）同场景A／B曾是当时的必做检验；2026-10-15起A／B实验结束，当前主线见[三根堆叠目标示教方案](#三根堆叠目标示教方案)。沿用现有描边表达进行实验，不把其他论文的视觉提示收益直接视为本实现的结论；没有第三张图不等于分割与传输没有延迟。

已实现 [joint_target.py](../../src/lerobot/bamboo_sorting/joint_target.py)：
显式选择实例 → 当前帧可见掩码 → `prepare_target_images` → `target_global_rgb`＋`grasp_rgb`。
实例PNG的像素ID只用于查找选中实例，不能当类别或跨帧身份。
`target_id`固定表示一次抓放的对象，`frame_id`约束轮廓对应的当前全局帧；这两个字段由调用者提供，接口检查一致性本身不证明跟踪正确。
掩码内侧2像素用RGB紫色`[255,0,255]`描边，被夹爪遮断的可见区域分别描边，不补全不可见部分。
接近阶段目标无当前可见掩码时拒绝生成输入，实时分离路径停止本轮；尚未实现自动重识别。搬运阶段不要求掩码。
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

上例只展示基本格式，省略帧不能通过预检查；堆叠标注由准备工具带入`pick_step`、`available_target_count`和人工结果，须与原采集记录一致。`reviewed=true`必须来自实际复核。
推理调用方先用`TargetInputSession.prepare`处理原图；仅在实际控制命令成功后调用`accept_command`，
不能使用预测动作提前切换。`joint_inference_stdio.py --target-conditioned`接收上述已处理图像及
`target_contract`、`target_input`（目标ID、帧ID、阶段）；默认旧推理协议保持不变。
`run_joint_smooth.py --model target-pair`已接入实时目标选择和这一会话（见[单次真机试验](#目标条件模型单次真机试验)）；不能加个选项就用旧权重进行目标抓取。`run_joint_target_shadow.py`提供静止观察的目标A/B预测，复用目标会话及显式目标协议；不支持伺服、夹爪写入或完整抓放。

## 斜条J6欠转的腕部角度诊断

**状态：完整序列分布与跳变敏感性检查已完成草稿，现只待20帧人工两点标注及误差导入。未做回归，不作为堆叠试采前置条件。** 现有测量使用固定夹指遮罩和主轴跨度检查，保留修正后的完整序列；不回退到旧颜色单筛选报告。

从[分布入口](../../artifacts/joint_target_live_preparation/ab_outcome_review/wrist_angle/distribution_review/review.html)查看已有结果，在[20帧人工标注页](../../artifacts/joint_target_live_preparation/ab_outcome_review/wrist_angle/distribution_review/manual_axes.html)先看原图，逐帧填写可标注性、遮挡和阶段；可标注时沿竹条长轴点两个相距至少80像素的点，不可标注时写原因，完整导出全部20帧。助手不能代替操作者填写人工结果。

拿到真实导出后，先确认下面目录尚无`measurement_error.json`，再替换实际路径导入；输出使用独占创建，已有人工结果不得覆盖：

```bash
.venv/bin/python examples/phone_to_auboi10/review_joint_wrist_distribution.py \
  --stage errors \
  --output artifacts/joint_target_live_preparation/ab_outcome_review/wrist_angle/distribution_review \
  --annotations "/实际导出的文件路径.json"
```

报告可配对帧的绝对角度差中位数、最大值、人工不可标注数及程序漏测情况。这是程序与一名人工参考的差异，包含双方误差，不是校准真值；样本最大差异不是全数据误差上界。无向轴按180°周期计算差异。

现有643帧来自6条专家示教／19轮实机，程序可测44／138帧，其中15轮实机有可测角度；无效帧保留在时间轴，不能当作没有该状态。独立单位是示教条数／实机轮数，各轮控制配置不同。完整统计、夹指修正及历史验证范围见[腕部证据](EVIDENCE.md#腕部离线诊断与人工复核)，不从帧数推算实验样本量。

[跳变敏感性检查](../../artifacts/joint_target_live_preparation/ab_outcome_review/wrist_angle/distribution_review/jump_sensitivity.html)已完成：同来源相邻程序可测帧间隔≤1.1秒、无向轴角差≥10°，允许跨无效帧；共17对、27个端点。暂时排除两端只是统计敏感性对照，未标为确定错误，原测量不改。数值重叠仍存在，不证明共同状态；按跳变排除可能引入选择偏差。同目录`jump_sensitivity.py`拒绝覆盖已有结果，不为证明运行过而重跑。

全部角度称为“相对当前图像参考线的角度”。不以关节5°/s直接判断图像跳变测错，不假设夹紧后竹条必然摆正，不用闭合后中位角自动重设零点；不扩展相机标定、自动纠偏或新回归工程。
复核后若目标身份、下探阶段、人工误差及共同状态仍不足，报告：“当前经复核的可测样本不足以支持纠偏斜率比较；早期下探阶段还存在观测缺口，尚不能确认训练覆盖不足或分布偏移。”当前人工结果尚未导入，不能提前写成经复核结论。

`diagnose_joint_wrist_angle.py`及原来源配置保留为已有测量的复现入口；旧取帧／回归方法说明从[整理前原文](EVIDENCE.md#文档整理与历史原文)回溯，不作为当前待办。

## 后续研究与待办

本节集中维护课题的总体技术路线、当前位置和各阶段约束；具体实验方案与命令在下列子节，结果只记入[证据索引](EVIDENCE.md)。

### 研究问题与总体架构

工业背景是竹条初加工：目前由工人从散堆中逐根取竹条送入加工机，本课题用机械臂抓取替代人工。场景为竹条散放于右侧堆放区台面，两两交角约不超过30°，逐层清空并放入左侧放置区；放置精度和节拍不作要求。
课题定位为具身智能（VLA）方法的探索性工作，不与工业上已较成熟的无序抓取方案比拼性能；“方向空白”的表述须在写作前经文献调研收窄并给出依据。

课题要回答：竹条处于其他竹条之间或之上时，机器人能否按指定抓起“那一根”，并逐根清空整堆。拆成三项须分别由实验建立的能力：
**感知**（每根在哪、哪些可见区域属于它、哪根在上层）、**目标条件动作**（指定哪根抓哪根，位置和角度变化下稳定夹取）、
**连续作业**（抓走一根后面对邻条移动或遮挡变化，重新观察、重新选择直到清空）。模块组合本身不构成已验证的创新，不能宣称任意堆叠泛化、稳定目标服从或自动清堆已经实现。

```text
固定全局相机观察
  → ① 感知：Mask2Former可见实例＋top_strip／covered_strip
  → ② 选择：从上层候选挑一根（初期规则或人工，后期可学习）
  → ③ 动作：当前全局图描出目标可见轮廓 → 目标条件SmolVLA输出七维关节动作，单次抓放
  → ④ 确认：抓对、恰好一根、送达（初期人工判定）
  → 重新观察，不沿用旧判断
```

分工理由：“选哪根”和“怎么抓”分开后，失败可定位到看错、选错或抓歪；目标只通过当前图像描边表达，七维动作契约、示教流程和训练入口可复用；每次抓放后重新观察，剩余场景仍是同类问题，邻条移动也不会沿用失效的初始顺序。

### 分阶段路线

| 阶段 | 目标 | 核心技术 | 通过依据 |
| --- | --- | --- | --- |
| 1. 指定目标抓取（**已阶段性结束**） | 两根分离摆放时，指定哪根抓哪根 | 描边式目标条件化、双目标前景分离、按观察年龄取步 | 已有约2组近似配对显示换描边后实际抓取对象随之改变，但无无描边基线，只算初步迹象；斜条J6欠转遗留，改由离线诊断处理 |
| 2. 三根堆叠（**当前**） | 3根交叉堆叠时，指定一根上层条并抓起 | 连续清堆采集、每抓一根一条示教；多根目标身份；修正被压条误判上层；堆叠示教微调 | 每次指定目标，分别报告抓对、夹住、恰好一根、送达；目标服从只统计至少两根可选上层条的步骤，并与无描边基线对照 |
| 3. 连续清堆（最终演示） | 逐根清空，失败能停 | 失败自动检测与重试（磁性开关／DI结合图像）、任意结束姿态衔接下一轮、清空／连续失败3次／超时停止 | 按堆统计整堆完成率、每步成功率、邻条扰动和耗时 |
| 4. 长短与语言分拣 | 语言指定放置区 | 长／短×放置区A／B四种示教组合，同一目标只换目的地指令 | 放置行为随指令改变；此处A／B是放置区，不是目标身份 |

2026-10-15起取消原“下探高度修正”阶段：按层高`n×8 mm`在下探段做FK→竖直修正→IK的方案属于脚本混合，不符合纯具身智能方向；堆叠层高由模型从示教中学习，原方案可从Git历史回溯。
A／B成对实验按用户决定不再继续，原[位置／角度分离对照](#目标条件模型单次真机试验)取消；目标服从改在堆叠实验中逐次统计，但只有“当时至少两根可选上层条”的步骤才计入服从证据，并须与同数据无描边基线对照；只有一个可选目标的步骤只验证抓放能力。
VLM选目标、RL、深度相机、架空及不规则支撑暂缓；不为当前问题先换整套架构，前三阶段稳定后再判断是否需要。

### 当前位置与下一步

当前工作顺序为[堆叠种子页面验收与试采](#三根堆叠目标示教方案)。已有成对模型和训练数据继续保留，状态与逐批证据见[成对数据记录](EVIDENCE.md#成对训练数据采后检查)；不再补录普通单根或重开A／B位置／角度对照。
遗留腕部诊断按[现有人工复核流程](#斜条j6欠转的腕部角度诊断)收束。表示能力不足、覆盖不足、闭环分布偏移及时序等仍是候选解释，不据当前分布草稿直接改模型或启动重训。

### 研究贡献候选与评价原则

作为探索性工作，不要求很高的成功率，但必须拿出两项核心证据：

- **模型确实服从描边**：同一画面换一根目标描边，实际抓走的竹条随之改变。为此必须保留“同数据、无描边”VLA基线，这是全文最核心的证据。
- **失败能说清原因**：例如斜条J6欠转，定量分析VLA在细长物体上的角度纠偏局限，比只报告成功率更有分量。

以下贡献须由实验建立：

1. **细长物体的描边式目标条件化**：与原自选目标模型、无描边、只换文字等对照，证明实际抓走对象随描边改变。
2. **遮挡下的目标身份维持**：从双根分离推广到三根堆叠，量化身份错误对动作的影响。
3. **感知与动作分离、每步重新观察的逐层清堆框架**：失败可归因到具体环节，按堆报告完成率。
4. **失败自动检测与重试**：磁性开关／DI结合图像判断是否夹住，失败后重新观察、重新抓取；以有／无重试的整堆完成率对照，同时保障最终演示。
5. **显式目标几何条件（候选）**：是否需要取决于腕部角度诊断，作用由消融检验。

人工接管纠偏示教属于已有的DAgger类方法，只作为提升数据的工程手段，不列为创新点。评价原则：保留同数据无描边VLA及旧模型自选目标作为基线，脚本抓取仅作独立比较、不进入主线；
物理结果以现场确认和录像为准，IO读回与命令周期不算成功；中止轮记为未知，不挑选剔除；测试摆放事先指定、不参与调参，才能称为泛化结果。

### 各阶段待办与约束

| 事项 | 下一步与完成依据 |
| --- | --- |
| 感知识别 | 保留047～051首轮测试原结果；分析051第二阶段误分类，补充普通结构的独立开发数据。若据此调数据、阈值或模型，泛化成绩需另批未参与开发的测试。 |
| 目标条件化 | 按[三根堆叠目标示教方案](#三根堆叠目标示教方案)推进；A／B实验结束，旧跟踪诊断冻结，不再扩充单根或两根分离数据。 |
| 可下指区段 | 当前实验由操作者在摆放时保证下指空间，不开发自动区段判断，不把尺寸测量或该模块作为推进门槛；实验结论限定于这一场景条件。 |
| 物理结果反馈 | 磁性开关及控制柜DI（数字输入）接入待现场确认，与堆叠采集并行准备，是贡献4及最终演示的前提；堆叠试验先人工判定，不等待自动成败系统。开关未闭合到底不独立证明夹住一根，需结合图像。 |
| 连续逐层抓放 | 独立记录抓对、恰好一根、送达、邻条扰动和耗时。允许抓取引起下层移动，不以零扰动判成功；移动不直接判失败，带起多根或影响继续作业等结果另记，抓后按新画面重新选择。预先固定清空／连续失败3次／超时的停止规则；按堆统计，测试规模和运行预算另定。 |
| 后续对照与纠错 | 保留同数据无描边VLA对照；脚本抓取仅作后续独立比较，不进入主线。保存完整运行及接管边界，再选择纠错片段训练；数据达到条件后安排数量泛化及消融。 |

`top_strip`不直接等于物理可夹；现有下层误判继续作为感知问题处理，与遮挡后的局部漏分分开判断。
旧双根数据已有两种上下摆放的抓放经验，见[已有模型与数据](EVIDENCE.md#当前双摆放模型)。复用优先从完整的第一次抓放片段开始，补当前帧目标轮廓；原始数据保留，以派生片段和来源映射接入。
旧示教含连续两次抓放，且没有目标元数据；当前目标接口首次闭合后本轮不再描边，不能把整条旧示教直接当作单目标样本。复用需适配片段边界、动作块与目标侧录，保持原训练／验证分组；它不能替代双可取目标A／B对照。
逐根摆放的差分标签仍须复核阴影、曝光与旧条被碰动的影响，历史完整形状不替代当前可见轮廓。
保留现有Mask2Former基线，不因斜条矩形框会重叠就断言其他实例分割方法不可用。局部深度仅在交叉处确有判断困难时再评估；论文贡献和泛化结论须由上述实验建立。

## 历史流程入口

这些记录用于理解已有数据和设计来源，不是当前实验任务。逐批结果只在EVIDENCE维护；整理前完整操作说明保存在`d13870a`，回溯命令见[文档历史](EVIDENCE.md#文档整理与历史原文)。

### 同场景A／B成对示教方案

该阶段已结束：78条示教完成目标轮廓接受和数据接口验收，训练66条、验证12条，目标条件模型完成30000步训练。近似配对和额外示教的区分、原始／逻辑场景编号、已接受误差及中断尝试均保留，不能把它们重新解释为严格同观察对照或新的堆叠数据。

| 要回溯的内容 | 唯一证据入口 |
| --- | --- |
| 首对示教与现场复核 | [首对A／B采后检查](EVIDENCE.md#首对ab示教采后检查) |
| 各批录制、轮廓接受、训练与离线评价 | [成对训练数据记录](EVIDENCE.md#成对训练数据采后检查) |
| 统一录制清单与训练来源 | [录制清单](../../artifacts/joint_target_pair_train_t029_s002_t033/training_capture_inventory.json)、[来源清单](../../artifacts/joint_target_pair_train_t029_s002_t033/target_training_sources.json) |
| 目标条件模型现场A／B及失败复核 | [现场证据](EVIDENCE.md#目标条件模型最新ab现场结果与b失败复核) |
| 搬运时序修正／新摆放斜条问题 | [搬运停滞](EVIDENCE.md#b夹住后搬运停滞)、[重新摆放复核](EVIDENCE.md#重新摆放后的指定ab抓取) |

### 指定目标输入：离线原型

首批5条单根试采和旧录像诊断已经完成／冻结。当前目标图像契约见[接口约定](#目标条件接口约定)，当前采集标注见[堆叠流程](#三根堆叠目标示教方案)。
旧关键帧、光流、冻结分割及重新确认对照见[离线原型证据](EVIDENCE.md#指定目标输入离线原型)和[旧录像诊断](EVIDENCE.md#旧录像目标输入诊断)。单根成功不证明目标服从；旧光流基线不能生成训练标签，全程搬运轮廓跟踪不作为当前前置条件。

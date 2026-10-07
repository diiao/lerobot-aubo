# 证据与历史索引

更新：2026-10-06。操作方法与后续计划统一见[项目手册](README.md)；本文只维护结果摘要、原始证据路径和历史索引。
部署记录是当时快照，不代表远端当前状态。

2026-10-02文档与旧功能的删除范围、保留理由及验证见[清理记录](../../docs/AUBO_CLEANUP_2026-10-02.md)。

末端实为气动二指夹爪；本页链接的历史记录、实验产物和代码字段中的“吸盘/吸附/suction”均指夹爪开合指令。

## 目录

- [动作模型与部署](#当前双摆放模型)
- [现场抓放记录](#最新三轮现场记录)
- [40 cm单根试抓](#40-cm单根试抓)
- [指定目标输入离线原型](#指定目标输入离线原型)
- [分割训练、轮廓复核与独立测试](#分割模型与独立测试)
- [相机与环境记录](#相机与环境记录)
- [文档整理与历史原文](#文档整理与历史原文)
- [更早历史与证据](#历史原文与证据)

## 当前双摆放模型

| 项目 | 记录 |
| --- | --- |
| 选项 | `mixed-both-orders` |
| 契约 | CameraSetV2，双 RGB，7维绝对关节＋夹爪，J5参与预测 |
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
- `onsite_second_grasp_review_20260928/`、`second_contact_candidate_20260928/`：旧第二根失败诊断；后者的高度补偿补丁当时被拒绝，未作为当前运行实现。后续新讨论的[一般堆叠下探高度修正](README.md#一般堆叠的下探高度修正方案)尚未实现，不代表恢复该旧补丁或已有验证结果。
- `full_chunk_replay_run01/`：保存观测的离线推理，已有结果无需因文档清理重跑。

## 最新三轮现场记录

以下目录均在 `artifacts/`，虽然文件名都有 both_top90，场景以操作者登记为准：

| 目录 | 实际场景与现场反馈 |
| --- | --- |
| `joint_preclose_feedback_both_top90_20260928_181639_883317523` | 上90°下45°，两根抓放成功 |
| `joint_preclose_feedback_both_top90_20260928_182412_271500264` | 上45°下90°，两根抓放成功 |
| `joint_preclose_feedback_both_top90_20260928_182818_033249696` | 单根90°抓放成功，但继续第二轮 |

单根约74.7秒首次释放、140.2秒再次夹紧、224.1秒再次释放后退出。
当时入口固定两周期；当前已支持 `--cycles`。没有硬编码第二轮轨迹；这些结果不是未见位置或普遍层序理解的成功率。

## 40 cm单根试抓

当前已有两次用户确认成功；最新第五次正常完成一个周期。第三、第五次支持进入目标接口开发，但运行时长差异明显，不能估计泛化成功率。

**第五次：正常完成，用户再次确认成功。**原始目录：
[joint_short_strip_single_20261006_013058_523320596](../../artifacts/joint_short_strip_single_20261006_013058_523320596/)。
仍使用现有动作模型及`--cycles 1 --latency-tolerant`，未接入分割或目标描边。
`complete.json`记录`model_suction_cycle_completed`、`failure=null`，5543帧命令、1368次预测、完成1周期；
双路各5572帧、约222.9秒，录像完整、无队列丢帧；全局／腕部重复来源帧178／212，读取缺失2／1次。
命令记录约27.7秒首次夹紧；抽查全局0、40、180、222秒及腕部最终帧，右侧斜放短条随夹爪搬运到左侧收集处，结束夹爪松开。
最终状态静止、伺服关闭、夹爪指令0；仍非起点，不自动归位。
用户反馈“这次机械臂同样抓取成功了”，结合录像记为本次现场成功，原始`physical_grasp_success=null`保持不改。
本轮约223秒，第三次约68秒；摆放朝向不同，尚不能确定时长差异原因，也不能认为放宽时间预算已消除延迟问题。

**第三次正常完成一个周期，用户现场确认成功。**原始目录：
[joint_short_strip_single_20261006_012053_948834880](../../artifacts/joint_short_strip_single_20261006_012053_948834880/)。
使用现有`mixed-both-orders`、年龄对齐取步、`--cycles 1 --latency-tolerant`，模型权重未改变，也未接入分割目标。

| 核对项 | 本轮证据 |
| --- | --- |
| 程序完成 | `reason=model_suction_cycle_completed`、`failure=null`；要求1周期、完成1周期；1685帧伺服命令、415次预测。 |
| 时长与开合 | 双路录像各1702帧，约68.1秒；夹紧、松开各一次，DO写入及读回均成功且符合请求。 |
| 录像复核 | 抽查双路0、20.4、28、45、63、68秒：右侧支撑上的目标进入夹爪，随夹爪搬运到左侧收集处，结束时夹爪松开。 |
| 现场物理结果 | 用户明确反馈“这次机械臂成功完成了一轮抓取”。结合录像，记为本次40 cm单根抓放成功；原始`physical_grasp_success=null`保持不改。 |
| 退出状态 | `is_steady=true`、`servo_enabled=false`、夹爪指令0；最终关节不是起点，下一轮前仍须归位。 |
| 推理与通信 | 模型耗时中位99.2 ms；请求往返中位132.7 ms、最大165.6 ms；收到预测时观测年龄中位190.3 ms、最大243.8 ms。 |
| 控制记录 | 已发送命令间隔最大76.2 ms；命令与实测关节最大差0.957°；记录的年龄对齐选步年龄最大404.2 ms。 |
| 录像完整性 | 两路均完整、无队列丢帧或读取缺失；全局／腕部分别重复来源帧51／56，按25 Hz写入侧录。 |

结论：现有动作模型在这一40 cm单根摆放上完成抓放，支持继续验证短条适应性。
这是一次现场成功，不能估计泛化成功率，也不是分割引导或普通堆叠上层抓取的验证。
本轮已接收预测和记录的取步年龄都未达到原500 ms门槛，不能据此认定放宽参数是成功的唯一原因，
也不能认定前两次延迟故障已根治。现场结果与日志核对完成；没有重新推理、训练或启动硬件。

**第一次：首条推理超时。**

原始目录：[joint_short_strip_single_20261006_010934_964085716](../../artifacts/joint_short_strip_single_20261006_010934_964085716/)。
用户报告异常停止；`error_traceback.txt`定位到`InferencePipe.predict → receive(2)`，等待首条推理响应超过2秒。
这是模型推理管道的超时，不是机械臂RPC的300 ms超时；模型已通过ready阶段，不能解释为模型加载失败。
`complete.json`记录`sent_frames=0`、`predictions=0`、完成周期0；`commands.jsonl`与`trace.jsonl`为空。
退出快照显示起始关节附近、夹爪指令0、`is_steady=true`、`servo_enabled=false`。
两路录像各52帧正常保存。此次未进入实际伺服抓取，物理成功仍为null，不计为短条抓取能力失败。

远端日志仅有模型加载信息，无异常堆栈；检查时GPU无计算进程。历史2026-09-28正常运行的首条请求往返约0.17秒、
模型路径约0.10～0.12秒；这些是历史对照，不能据此确定本次原因。
已准备保存录像的离线诊断，但用户选择先放宽等待重试，未运行该离线推理；原始失败请求图像未单独保存，录像回放只是近似输入。
据此增加显式`--inference-timeout-s`，默认保留2秒，第二次使用5秒；模型启动等待仍为120秒。

**第二次：滚动使用旧动作块时过期。**原始目录：
[joint_short_strip_single_20261006_011503_428586227](../../artifacts/joint_short_strip_single_20261006_011503_428586227/)。
已发送142帧、接收35次预测、未完成开合周期。异常栈定位到年龄对齐动作块的滚动取步检查，
`expired prediction: age=513.3 ms, limit=500 ms`，不是`receive`响应等待超时。
35次响应收到时观测年龄169～231 ms；模型计算路径中位98.9 ms、请求往返中位137.7 ms、最大166.4 ms。
已发送命令间隔最大51.2 ms，跟踪误差最大0.726°；这些数值不包含抛错后未发送的那一拍。
最后接受的第34号预测在191 ms时收到，已执行至约273 ms的动作行，后来复核时过期；
日志不足以确定中间延迟发生在状态读取、FK调用、调度还是其他环节，不能断言GPU变慢或首次超时已修复。
退出快照为静止、伺服关闭、夹爪指令0，J1约-55.42°，已离开起点；重试前必须重新核对路径并归位。

用户要求放宽相关限制，现增加显式`--latency-tolerant`供本次试验：参数统一见[手册](README.md#4-当前动作与保护)。
只调整预测时间预算，保留关节、工作区、限速、跟踪误差、控制间隔及发送时状态检查；无硬件测试145项通过，
覆盖默认值不变、新预算实际传入、延迟响应接受、持续缺失预测仍停止、伺服退出和原有动作限制。
之后第三次的实机结果见本节开头；软件测试仍不等于物理安全或抓取成功证明。

**第四次：相机帧年龄过期。**原始目录：
[joint_short_strip_single_20261006_012606_859991495](../../artifacts/joint_short_strip_single_20261006_012606_859991495/)。
运行约25.4秒，发送618帧、接收154次预测、完成周期0；在`ReadOnlyStation.observe`读取全局图时，
`OpenCVCamera.read_latest_with_timestamp`检测到帧年龄252.7 ms超过250 ms，尚未读取本次腕部图。
退出为静止、伺服关闭、夹爪指令0；TCP约(0.533,-0.358,0.095)米，已离开起点，下一轮先由操作者核对归位路径。
录像各635帧，完整保存；全局／腕部读取缺失2／1次。侧录显示约24.96～25.12秒两路重复同一来源帧，
腕部约25.16秒恢复，全局约25.20秒恢复；只说明短时没有新帧，不能据此确定USB、线程调度或其他根因。
据用户要求，本次在已有`--latency-tolerant`中配套放宽取帧年龄和双路时间差，具体值统一见手册；
默认模式、总预测有效期及运动边界不变。147项相关软件测试通过，包括双路延迟输入经实际入口接收、
新预算之外继续拒绝和默认值保留；更新后的第五次结果见本节开头。相机恢复不表示第四次已完成抓取，原始记录不改写。

## 指定目标输入离线原型

目标条件软件接口已推进到`AuboI10JointTargetApproach`：训练数据视图、数值推理适配、
检查点契约及采集目标元数据已接入；操作与标注格式只维护在[手册](README.md#指定目标输入离线原型)。
187项相关软件用例通过，涵盖接近／搬运切换、相机旧帧不提前切换、随机读取、动作尾部填充、新旧契约隔离，
以及假模型保存／重载／评价和假设备采集；缓存目录权限导致的一项失败单独重跑后通过。
未调用真实模型、训练或硬件。完整原始轨迹核对见[interface_check.json](../../artifacts/joint_target_review/interface_check.json)：
旧60 cm验证示教`aubo_joint_legacy_validation_12`的episode 0共770帧，双路视频逐帧解码时间与Parquet一致，
状态／已接受动作与侧录一致；按新规则为358帧接近、412帧搬运。
这条旧示教没有目标ID和目标掩码，`target_training_ready=false`，未改标、未导出；
该旧示教核对仅作基础数据链路证据；新40 cm试采已完成接近标注复核和整条样本验收，见下文。

首批`joint_target_pilot`已录制完成，5/5保存并成功完成数据整理，无重录／未完成条目，人工结果均为`single_success`。
[capture_check.json](../../artifacts/joint_target_pilot/capture_check.json)与同目录[固定复核页](../../artifacts/joint_target_pilot/review.html)
保存本批检查：原始契约、状态、已接受动作、夹爪开合侧录一致；全局／腕部视频各3593帧均完整解码，逐帧时间与Parquet对应。
双RGB为640×480，25 Hz，视频总时长143.72秒；相机间时差最大34.70 ms，无倒退或重复来源时戳。
每条抽查6张全局图及3张腕部图，均与从右侧单根夹取、搬运、落在左侧收集区的人工结果一致；不是全程逐帧视觉验收或模型成功率测试。

| 人工条号（episode_index） | 帧数 | 时长/秒 | 接近／需标注帧 | 搬运／免标注帧 | 闭合／释放帧号 |
| --- | --- | --- | --- | --- | --- |
| 1（0） | 762 | 30.48 | 359 | 403 | 358 / 650 |
| 2（1） | 751 | 30.04 | 372 | 379 | 371 / 635 |
| 3（2） | 726 | 29.04 | 346 | 380 | 345 / 626 |
| 4（3） | 719 | 28.76 | 361 | 358 | 360 / 612 |
| 5（4） | 635 | 25.40 | 310 | 325 | 309 / 536 |

五条均只有一次闭合／释放周期，均为监督训练候选。目标ID均为`strip_001`，场景原编号`001`～`005`、
现场描述`1`～`5`保留；描述不足以单独辨认摆放，当前单根场景由关键帧辅助定位，后续应填写明确位置／方向。
接近轮廓已由用户复核接受用于流程试采，1845帧搬运保持原图。数据接口已通过验收；独立验证集未建立，`target_training_ready=false`。
本批新增[标注准备检查](../../artifacts/joint_target_pilot/target_annotations/preparation_check.json)：
从原始时间侧录重新算出359／372／346／361／310帧接近阶段，共1748帧；依据episode元数据的视频起始时间与数据索引提取RGB PNG，
1748张解码时间全部对应，未将条内帧号误当合并MP4帧号。所有帧清单、图片及后续掩码集中在同一`target_annotations/`目录。
`annotations.json`当前1748项均为`reviewed=true`，已有二值掩码1748、整帧缺失0；原始采集的`pending_offline_review`作为采集时历史记录保持不变，当前复核状态由标注侧录保存。

[原图抽查记录](../../artifacts/joint_target_pilot/target_annotations/visual_check.json)覆盖五条各自起始帧和闭合观察帧，共10张。
五条起始均为右侧单根、朝向不同；闭合附近夹爪／线缆遮挡局部，尤其episode 2的frame 345在夹爪两侧仍有可见木条，
后续轮廓应保留两侧可见区域且不补全遮挡。该原图抽查发生在推理前，记录与下述预测轮廓抽查分开保存。

固定`review.html`保留原录制证据，增加连续帧与遮挡修正工具。
[浏览器检查](../../artifacts/joint_target_pilot/target_annotations/browser_check.json)保存本机Chrome的file入口检查；准备阶段检查末接近帧、播放、缺失拒绝确认及多边形编辑，
导入后检查5条闭合帧的轮廓加载、描边开关、问题快捷定位和已记录问题阻止区间确认；未导入测试标签。
[5项针对性合成检查](../../artifacts/joint_target_pilot/target_annotations/software_check.json)通过：断开可见区域保留、歧义／缺失不强行合并、修改保留原掩码且旧修订拒绝导入、错帧预测拒绝写入，以及省略问题字段不能绕过已有待修问题。
这些是复核工具检查，不是目标模型或物理成功验证。

[推理执行记录](../../artifacts/joint_target_pilot/target_annotations/inference_request.json)：用户明确授权后，核对`gpu`的冻结最佳模型（best epoch 15）、Python与CUDA，
在`/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/artifacts/joint_target_pilot/target_annotations/`新目录上传输入和独立脚本，完成一次推理。
[predictions.json](../../artifacts/joint_target_pilot/target_annotations/predictions/predictions.json)记录1748帧、61.55秒、原640×480、batch size 2、置信度与掩码阈值均0.5，optimizer steps为0。
加载记录的missing／unexpected／mismatched／error项均为空。每帧唯一实例，全部预测为`top_strip`；这是预测类别，不证明可夹或轮廓正确。

[预测统计](../../artifacts/joint_target_pilot/target_annotations/prediction_check.json)中，五条多块可见区域帧数为2／4／53／47／4；
面积较前帧变化超过20%的提示位于episode 3的290／296与episode 4的245／256／257。面积提示只用于定位，不自动判错。
助手查看35个不同帧的原图与轮廓（拼图在同目录`inspection/`），发现并记录9个待修帧：

| episode_index | 帧号 | 图像依据与问题 |
| --- | --- | --- |
| 2 | 295、335、345 | 样本保留夹爪／线缆两侧断开的可见区域；仍未人工确认像素边界。 |
| 3 | 290、296、297、310 | 夹爪后方木条仍可见但漏分；289与295曾保留该段，说明邻帧间有不稳定变化。 |
| 4 | 245、255、257、258 | 夹爪左上方短段仍可见，但轮廓只保留近侧主体。 |
| 4 | 256 | 轮廓跨过遮挡并包含夹爪区域，需要修正可见边界。 |

整帧缺失0不代表局部漏分0；35帧抽查不替代1748帧人工复核，以上也不是全部错误清单。
用户随后明确反馈“复核完毕,我认为遮挡后漏分是正常情况不必多虑”。依此记录本批整体复核完成并接受用于流程试采，
全部1748帧及5条episode的`reviewed`设为true；预测和二值掩码保留原样，9个问题从待修提示转为`accepted_annotation_limitations`，原问题文字及处理含义保存在复核历史。
这表示接受当前标签的工程取舍，不表示已修正误差、逐像素完美或真实抓取模型成功；已接受的遮挡漏分不作为继续逐帧修补的待办。

[完整数据接口验收](../../artifacts/joint_target_pilot/target_annotations/dataset_check.json)通过：`prepare_target_source`覆盖5条3593帧，
实际`LeRobotDataset`（PyAV解码）与`JointTargetDataset`逐帧核对；1748个接近帧的解码RGB与标注来源PNG完全一致，1845帧搬运按阶段使用原图。
全部3593帧的七维状态、50步绝对关节动作序列及尾部填充保持原值，训练数据视图与`TargetInputSession`的图像、因果阶段切换逐帧一致。
五条各49个尾部动作序列需要填充，均通过；释放后未重新进入接近阶段。该验收没有加载策略或预测动作。
标注侧录记为`target_source_ready=true`；完整训练流程仍缺独立验证数据，未创建替代验证集，未启动训练。
本次没有训练、相机或机械臂连接，没有改写／重新导出原始视频与动作数据，也未创建验证集。
这5条均为单根，没有竞争目标，证明的是流程可用，不证明模型听从指定对象；后续按[同场景A／B成对示教方案](README.md#同场景ab成对示教方案)检验目标服从，保留本批作为基础数据。

A／B采集前离线准备已检查：`record_joint.py`的两条`train`预览成功，双640×480、七维状态／动作及目标元数据开关一致，未创建采集目录或连接设备。
标注准备工具已去除默认唯一候选自动选中，避免双目标场景漏检一根后误选另一根；复核页增加目标描述与摆放参考，远端路径提示按当前批次生成。
`tests/bamboo_sorting/test_joint_target_review_preparation.py`的6项合成测试通过，包含“仅检出另一根时不能自动作为目标掩码”；`git diff --check`通过。
本次未运行模型、训练或浏览器交互验收，未修改首批原数据和标注；A／B实际采集、所选目标关联与轮廓复核仍待现场数据，操作统一见上述方案。

2026-10-06以第五次成功录像制作[固定复核页](../../artifacts/joint_target_review/index.html)，
来源与轮廓集中在同目录[annotations.json](../../artifacts/joint_target_review/annotations.json)。
0、40、180、222秒共4个关键帧，由助手绘制可见轮廓草稿；用户最终反馈“可以，这次标注正确，进行下一步”，
现已记录`reviewed=true`及确认来源，仅适用于这4帧可见轮廓，不是分割预测，也不确认中间帧。
初次草稿漏掉180秒和222秒夹爪后方的可见木条，用户复核指出后，两帧均补入被线缆隔开的两块可见区域；
这两帧各有3块可见区域，均属同一个`strip_001`，夹爪与线缆遮挡处不补线，修正后用户已确认。
同时放大核对0秒和40秒原图，未发现整段可见木条漏标。原复核页面已更新，原图及腕部图保留。
这段自主执行录像未转换成专家示教，也未生成可训练动作数据。
接口和后续接入统一见[手册](README.md#指定目标输入离线原型)。

**连续光流基线未通过。**同目录保存[tracking.json](../../artifacts/joint_target_review/tracking.json)、
[tracking_frames.jsonl](../../artifacts/joint_target_review/tracking_frames.jsonl)及[tracking.mp4](../../artifacts/joint_target_review/tracking.mp4)，由原复核页统一展示。
处理全段5572帧、25 fps：4帧人工初始化，5264帧输出未复核草稿，304帧标为丢失；这些计数不是成功率。
从0秒初始化后，在27.84秒（第696帧）因传播一致性／面积检查失效而停止，直到40秒人工重新初始化。

| 人工检查点 | 重新初始化前的传播结果 | 与确认轮廓IoU |
| --- | --- | --- |
| 40秒 | 已丢失，按空掩码评价 | 0 |
| 180秒 | 仍输出草稿，但漂到背景 | 0 |
| 222秒 | 仍输出草稿，但漂到背景 | 0 |

目视抽查27、27.84、80、179.96、200、221.96秒：夹持附近出现快速移动／模糊；后续橙色轮廓留在背景，未随木条正确移动。
这说明该光流基线及其一致性检查不足，不能把“仍有输出”当作目标身份正确，也不能用它生成训练标签。
用户再次通过截图指出木条移动后橙色轮廓仍留在背景；此回放已归入页面默认折叠的失败案例区，保留原始输出。
每个人工检查点先评价再重新初始化；结果仅为开发诊断，未训练或调参。下一步安排见手册。

**冻结分割模型的7帧诊断已完成。**在用户确认继续后，使用原第15轮`best`对0、25、27.84、40、80、180、222秒原始RGB推理，
分辨率640×480，无拉伸／裁剪，置信度0.5、掩码阈值0.5、重叠面积阈值0.8，batch=2，0次优化。
模型精确加载，无missing／unexpected／mismatched keys；模型加载后7帧处理约0.49秒。
GPU独立目录为`/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/target_video_diagnostic/`，未覆盖训练源文件或权重。
本机[输入清单](../../artifacts/joint_target_review/segmentation/manifest.json)记录原视频帧号及来源时间；
[预测报告](../../artifacts/joint_target_review/segmentation/result/predictions.json)与7张16位实例PNG保留模型原输出，
[轮廓比较](../../artifacts/joint_target_review/segmentation/comparison.json)单独保存人工参考分析；原网页统一展示。

| 已确认关键帧 | 最佳重合实例类别／置信度 | 可见掩码IoU | 可见像素召回 |
| --- | --- | --- | --- |
| 0秒 | `top_strip`／0.984 | 0.858 | 87.8% |
| 40秒 | `top_strip`／0.958 | 0.486 | 52.4% |
| 180秒 | `top_strip`／0.865 | 0.635 | 70.6% |
| 222秒 | `top_strip`／0.799 | 0.552 | 61.8% |

7帧均输出1个`top_strip`实例，目视可见木条位置被重新找到；25、27.84、80秒尚无人工真值，不计算IoU。
40秒夹爪后方整段未覆盖；180秒线缆后端未覆盖；222秒夹爪后方两块可见区域均未覆盖，这些区域的像素召回均为0。
因此高置信度不保证完整轮廓；当前模型可作为下一步目标定位候选来源，尚不能直接自动生成完整目标标签。
类别输出只作记录，未把夹持中的`top_strip`当作物理可夹证明。指标按人工真值选择最佳重合实例、忽略类别计算形状，
不证明自主身份关联；这是单条已查看录像的开发诊断，不是独立测试、全场景FP／F1或实机成功率。

**新分割掩码的24～30秒关联诊断。**源录像第600～750帧，共151帧；第625、696帧复用此前预测，
仅补149帧冻结模型推理，原尺寸、阈值、best及后处理保持一致，加载后149帧处理约5.33秒，0次优化。
GPU新增产物位于上述`target_video_diagnostic/association_clip/`；本机保存于原复核目录`association/`，不另建复核入口。

初始目标是助手在24秒从唯一预测中选定的`strip_001`，尚待用户复核，不回填成已确认真值。
先尝试仅补偿平移的几何关联：1帧初始化、95帧匹配、55帧丢失；27.84秒候选中心距26.41 px、长轴变化22.02°、
重合仅0.0756、综合分0.2069低于0.35，正确候选被规则拒绝。原始`association.json`、`association_frames.jsonl`及录像保留。
随后增加质心／长轴的旋转对齐来比较形状，所有数值门限保持不变，复用同一批预测在本机重算，未重复推理；
这属于根据失败分析改进的开发结果，不能当独立测试。变换后的旧掩码只用于评分，从不作为当前帧输出。

| 旋转对齐后的结果 | 核对记录 |
| --- | --- |
| 初始化／匹配／丢失 | 1／144／6帧；24～29.76秒持续匹配候选，29.80秒首次丢失。计数不是身份准确率。 |
| 丢失原因 | 29.80秒（第745帧）固定阈值下没有分割实例；之后保持丢失，不自动重新选目标。29.96秒也无分割实例。 |
| 预测数量 | 149帧各1个实例，2帧0个；没有多候选竞争场景。 |
| 目视抽查 | 24、27.60、27.80、27.84、27.88、28.20、28.60、29、29.40、29.76秒，新绿色轮廓随木条移动；未见旧光流那种停在背景的漂移。29.80、29.96秒不画绿色目标。 |
| 能力边界 | 尚无逐帧人工身份真值，用户尚未复核；遮挡侧轮廓仍可能不完整，单根片段不证明多木条身份稳定，未接入控制器。 |

仅几何关联基线的[报告](../../artifacts/joint_target_review/association/rigid_association.json)、
[逐帧关联](../../artifacts/joint_target_review/association/rigid_association_frames.jsonl)、
[固定条件](../../artifacts/joint_target_review/association/rigid_association_plan.json)、
[回放](../../artifacts/joint_target_review/association/rigid_association.mp4)保留原样；页面最前面改为下面的重新确认结果，原光流失败案例仍折叠保留。

**短时漏检后的受约束重新确认。**完全复用上述151帧预测，在本机运行，没有新增推理或训练。
实现保留最后接受状态作为几何参考，只对短时空检测启用恢复，最多等待5帧；要求连续两张不同来源图像满足更严格的匹配条件，
在此之前不输出目标掩码。歧义、不匹配、缺帧或超时仍要求显式重新选择，具体条件统一见手册和机器计划。

| 原录像时间 | 新状态与证据 |
| --- | --- |
| 29.80秒 | `unobserved`：0个分割实例，不画绿色目标。 |
| 29.84秒 | `confirming`：候选满足条件，第一张来源图，仅蓝色候选。 |
| 29.88秒 | `reconfirmed`：第二张不同来源图通过；原目标中心差0.12 px、长轴差0.47°、面积比0.992、形状重合0.960、综合分0.866，恢复`strip_001`，仅显示该帧预测轮廓。 |
| 29.92秒 | `matched`：继续关联。 |
| 29.96秒 | 再次`unobserved`，不画目标。 |
| 30秒 | `confirming`：仅第一张确认图，片段结束，无第二张证据，不记为第二次恢复成功。 |

合计1帧初始化、145帧常规匹配、2帧未观测、2帧待确认、1帧重新确认；未触发持续`lost`，但并非每帧都有目标输出。
逐帧核对29.76～30秒回放，暂未观测／待确认时无绿色目标，29.88秒绿色轮廓恢复在可见木条上，没有回填缺失帧。
本批单根录像没有竞争目标；恢复结果尚待用户复核，不能作为多目标身份正确率或训练标签。
当前[报告](../../artifacts/joint_target_review/association/reconfirmation.json)、
[逐帧状态](../../artifacts/joint_target_review/association/reconfirmation_frames.jsonl)、
[固定条件](../../artifacts/joint_target_review/association/reconfirmation_plan.json)、
[回放](../../artifacts/joint_target_review/association/reconfirmation.mp4)仍由原网页统一展示，不另建复核目录。

`tests/bamboo_sorting/test_joint_target.py`共14项离线测试通过，覆盖可见片段与遮挡间隙、原图保持、
帧／对象不匹配和目标不可见拒绝、草稿预览、实例ID选择、旧策略拒绝新契约、复核页生成及显式更新，
并覆盖合成纹理平移、目标消失后不得自动恢复、预测PNG实例ID映射、按真值比较而非按类别选目标、未标注帧不编造评价。
另有`test_joint_target_association.py`共15项通过，覆盖实例编号更换、类别改变、部分可见掩码不补全、
近似候选歧义、目标消失、位置突变、缺帧及旋转匹配，旧的持续失效模式仍不自动恢复；
新增覆盖两张来源图确认、重复来源不计数、超时失效、干扰／歧义拒绝、再漏检清空确认次数、缺帧和时间倒退拒绝。
合成测试通过不能替代上述真实录像结果；这些仅验证数据处理逻辑，不证明模型服从目标。本阶段未训练或连接硬件。

## 分割模型与独立测试

### 数据、模型与环境位置

| 项目 | 已核对记录 |
| --- | --- |
| 数据 | 400×25×8 mm；007～046为train/validation，047～051为test。划分及格式只维护在[项目手册](README.md#静态分割数据训练与独立评价)。 |
| 基座 | `facebook/mask2former-swin-tiny-coco-instance`；revision `22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6`。 |
| 首轮训练 | 640×480不拉伸／裁剪、无增强；30轮、batch=2、1800次更新；AdamW骨干5e-6、其余5e-5、weight decay 0.05、梯度裁剪1.0、seed=42。 |
| best选择 | 每5轮验证，按两类平均F1，同分比较验证损失；best第15轮，结束后重载验证；总训练291.16秒。 |
| GPU实验根目录 | SSH别名 `gpu`；`/home/rentao/program/lerobot-aubo-strip-segmentation-20261005`。 |
| 模型及日志（相对GPU根目录） | `outputs/short_strip_mask2former_baseline/best/`、`final/`（同级）；日志 `logs/train.log`；基座 `models/mask2former-swin-tiny-coco-instance/`。 |
| 复用Python | `/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv/bin/python`；SciPy放在实验根目录 `deps/`。 |
| 依赖 | Transformers 4.57.1、SciPy 1.15.3；GPU当时torch 2.10.0、torchvision 0.25.0、NumPy 2.2.6、Pillow 12.3.0。 |
| 独立测试执行位置 | GPU根目录 `independent_test/`；只读冻结best，20图、0次优化、加载后评价约1.12秒，不含模型加载。 |

本机训练报告为 `artifacts/short_strip_training/result/`，包括 `complete.json`、`best_validation.json`、
`baseline_validation.json`、`latest_validation.json`、`metrics.jsonl`、`loading_info.json`、`plan.json`、`data_audit.json`、40张 `validation_masks/`。
独立测试报告为 `artifacts/short_strip_training/independent_test/result/`，包括 `test_evaluation.json`、
`complete.json`、`plan.json`、`loading_info.json`、`data_audit.json`、20张 `test_masks/`；日志为上级 `evaluate.log`。
两处各自保存固定的 `prediction_review.html`、`contour_review.json`；权重仍在GPU，本机未取回。

首次真实启动在优化前因权重检查停止：Transformers的 `mismatched_keys` 为名称列表，原检查误按元组且漏列类别权重。
已修正，仅允许 `class_predictor.weight`、`class_predictor.bias`、`criterion.empty_weight` 随80类转2类重建；
其余正常加载，损失类别权重为 `[1,1,0.1]`。合成回归覆盖后真实训练完成；独立测试加载无任何差异。
`artifacts/short_strip_training/transfer.tar.gz` 是修正前首次部署快照，不能覆盖当前正确源文件。
GPU保留 `logs/startup_loading_failure.log`、`train_strip_segmentation.before_loading_fix.py` 供回溯。

### 验证集轮廓复核

[复核页面](../../artifacts/short_strip_training/result/prediction_review.html)及
[统计](../../artifacts/short_strip_training/result/contour_review.json)：10组／40图／60实例，10张空图无误检。
原图与采集／人工review副本一致，真值与人工可见掩码逐像素一致，40张预测PNG的ID与报告 `segments` 对应；
已复算IoU≥0.5指标并查看全部对照图。验证集参与选模型，不是独立成绩。

| 类别 | TP/FP/FN，IoU≥0.5 | 平均／最低IoU | TP/FP/FN，IoU≥0.75 |
| --- | --- | --- | --- |
| top_strip | 30/0/0 | 0.8747／0.7548 | 30/0/0 |
| covered_strip | 30/0/0 | 0.8508／0.6843 | 29/1/1 |

IoU≥0.9时两类都只匹配6/30；F1=1.00不表示轮廓完美。具体证据：

- `pile_031_frame_003 / strip_002`：IoU=0.6843、漏865像素，右可见段只覆盖51.7%；两段仍为同ID。邻条 `strip_001` 混入其262像素，存在局部归属混淆。
- `pile_021_frame_003 / strip_003`：IoU=0.7548、多721像素，长边与端部外扩；置信度0.9789不代表边缘准确。
- `pile_011_frame_002 / strip_001`：交叉与长边漏分，IoU=0.7723；`pile_044_frame_002 / strip_001` 也有端部漏分。
- `pile_041_frame_003 / strip_001`：54个被遮像素误归下层，其中19个距遮挡边缘超过2像素。

主要可见分段仍归同一预测ID，但非空图有小碎片。未见整根穿过上层的明显补全；
遮挡区域8722像素中误占518像素，去掉边缘≤2像素后仍误占60像素，不能声称完全无越界。

### 首批独立测试：047～051

[复核页面](../../artifacts/short_strip_training/independent_test/result/prediction_review.html)、
[逐图报告](../../artifacts/short_strip_training/independent_test/result/test_evaluation.json)、
[轮廓统计](../../artifacts/short_strip_training/independent_test/result/contour_review.json)。
五组在模型固定后新拍，操作者确认从上方加入、旧条未移动或明显倾斜，15个新增轮廓先人工确认再显示预测。
本批未参与训练、模型选择或阈值调整；包含20图、5张空图、30个可见实例，组内帧相关，仅有5个独立摆放。
固定第15轮best及[评价口径](README.md#独立评价与预测复核)，IoU≥0.5与≥0.75结果相同：

| 类别 | TP/FP/FN | precision（查准率） | recall（召回率） | F1 |
| --- | --- | ---: | ---: | ---: |
| top_strip | 15/1/0 | 0.9375 | 1.0000 | 0.9677 |
| covered_strip | 14/1/1 | 0.9333 | 0.9333 | 0.9333 |

两类平均F1=0.9505，空图误检0/5；非空图14/15数量与类别匹配，**不能换算为抓取成功率**。
按几何建立对应，30个真值均有IoU≥0.75的预测，29个类别正确、1个错误，另有1个额外预测。
几何平均IoU=0.8599（top 0.8622、covered 0.8576），包含错误类别配对、排除额外预测；不能单独据此判断类别可靠。
IoU≥0.9时top为2/14/13，covered为1/14/14，边缘仍有误差。

| 样本 | 具体失败依据 |
| --- | --- |
| `pile_051_frame_002 / strip_001` | 被压条误标上层：预测ID=2、置信度0.645844、IoU=0.8339；两段仍同ID。真正上层也正确检出，因此该帧输出两个top，有误选下层风险。 |
| 同帧额外实例 | ID=1、covered、置信度0.958676，但仅88像素、30块碎片；54像素在下层真值内、34在背景，与下层IoU仅0.0221。不能抵消整条误分类。 |
| `pile_051_frame_003 / strip_001` | 类别恢复正确，IoU最低0.7963；漏277、多204像素，6像素真值小块完全漏掉。 |
| `pile_049_frame_002 / strip_002` | 上层最低IoU=0.7965；漏368、多297像素，主要长边偏移与端部误差。 |
| `pile_050_frame_003 / strip_003` | IoU=0.7976；沿长边漏分，漏408、多193像素。 |
| `pile_048_frame_003 / strip_003` | IoU=0.8036；轮廓外扩，多473、漏122像素。 |

已查看全部20张阶段对照图；主要可见分段保持同ID，面积≥80像素的真值分段最低覆盖83.0%。
80只是统计界限，未过滤预测；遮挡区域4093像素中误占177像素，排除边缘≤2像素后仍7像素。
结论：固定机位与普通堆叠的新摆放已有较好初步识别效果，但明确存在被压条误判上层、碎片和边缘误差，尚不能直接作为可靠抓取目标。
若根据这些失败开发模型或调阈值，本批保留为首轮基线，后续泛化须使用新测试组；不修改原结果后再声称独立成功。

## 相机与环境记录

2026-10-03仅测试新全局相机 `04434000_P120800_SN0002`：预热3秒，MJPG 640×480测15秒得30.01 FPS，
1920×1080测8秒得30.00 FPS，读取错误均0；未开腕部相机或机械臂，不证明双路并发或最终视野。
[原始报告与样图目录](../../artifacts/new_global_camera_20261003_nkaq9n8h/)保留；切换方法见[相机配置](README.md#相机配置)。

2026-10-05 GPU仅按明确授权删除 `/home/rentao/.cache/pip` 和 `/tmp/aubo-joint-prep-20260922-v1/smoke`，
释放1977417728字节（约1.84 GiB）；训练结束时余量约11.69 GiB，是当时快照。
原始数据、模型、录像、uv缓存、trash、复用虚拟环境均保留；上述记录不是新的清理授权。

## 文档整理与历史原文

整理前检查点为 **`129513a`**，包含当时全部代码、测试、配置、48份人工标注JSON和文档；相关121项软件测试通过。
本次集中维护为[项目手册](README.md)＋本证据索引；[AGENT.md](../../AGENT.md#文档维护规则)规定后续更新位置。
四份已合并文档经用户明确确认后移除，原始产物与上游框架资料未改动；本轮整理不新增说明文件。
旧 `JOINT_CAPTURE.md`、`JOINT_TRAINING.md`、`JOINT_TRIAL.md`、
`docs/AUBO_I10_TOP_LAYER_VLM_RL_ROUTE_2026-09-26.md` 的全文与逐批记录可从该提交查阅。
旧路线中的只标上层、Mask R-CNN、150组预算、固定第三张goal_image及“分割未训练”不再作为当前决定。

```bash
git show 129513a:examples/phone_to_auboi10/JOINT_CAPTURE.md
git show 129513a:docs/AUBO_I10_TOP_LAYER_VLM_RL_ROUTE_2026-09-26.md
```

旧 `TopLayerTruthV1` / `TopLayerPredictionV1` 双图层序格式及 `evaluate_layer_order.py` 的操作规范也从上列旧路线回溯；
评价器保留，未用于当前单图分割。更早的代码、已退役说明及证据见下节。

## 历史原文与证据

清理前代码和文档基线为 `69435fe`。下列已替代文档从日常目录移除，原文完整保留于该提交：

- `AGENT.md`、`CLAUDE.md`、本目录旧 README：历史 ACT 操作与工具背景。
- `BAMBOO_GEOMETRY_ASSISTED_ACT_PLAN.md`：已退役几何预览对应的旧 ACT 增强计划。
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

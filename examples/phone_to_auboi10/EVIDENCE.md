# 证据与历史索引

更新：2026-10-16。操作方法与后续计划统一见[项目手册](README.md)；本文只维护结果摘要、原始证据路径和历史索引。
部署记录是当时快照，不代表远端当前状态。

2026-10-02文档与旧功能的删除范围、保留理由及验证见[清理记录](../../docs/AUBO_CLEANUP_2026-10-02.md)。

末端实为气动二指夹爪；本页链接的历史记录、实验产物和代码字段中的“吸盘/吸附/suction”均指夹爪开合指令。

## 目录

- [三根堆叠软件准备](#三根堆叠软件准备)
- [动作模型与部署](#当前双摆放模型)
- [现场抓放记录](#最新三轮现场记录)
- [目标条件模型A/B现场复核](#目标条件模型最新ab现场结果与b失败复核)（含[重新摆放与斜条J6欠转](#重新摆放后的指定ab抓取)）
- [40 cm单根试抓](#40-cm单根试抓)
- [指定目标输入离线原型](#指定目标输入离线原型)
- [分割训练、轮廓复核与独立测试](#分割模型与独立测试)
- [相机与环境记录](#相机与环境记录)
- [文档整理与历史原文](#文档整理与历史原文)
- [更早历史与证据](#历史原文与证据)

## 三根堆叠软件准备

2026-10-16按更新后的主线完成本地软件准备；操作顺序、字段和能力边界统一见[项目手册](README.md#三根堆叠目标示教方案)。本轮没有三根堆叠实采、冻结模型推理、真实训练、远端部署或机械臂动作，也未改写原始实验、录像、模型或已接受标注。

- **采集与数据**：新增堆内序号、可选上层条数和五项独立人工结果。假设备验证同堆六次尝试保存（交替失败与成功）、重录不占保存序号、失败／未知保留；实际LeRobot/PyAV合成视频往返覆盖采后检查、接近帧提取及训练／运行输入一致。抓错目标即使原菜单为`single_success`也不进入成功目标示教候选；按整堆检查训练／验证泄漏，包含被排除的失败步骤。
- **分离与运行接口**：新多实例可见种子方法与旧双根方法并存；合成测试覆盖交叉、遮挡缺口、实例编号变换、小位移、合并实例、单根消失和未知前景。离线标注与实时准备输出逐像素一致；单次运行计划覆盖1／2／3根剩余条，通信层拒绝请求新模式却返回旧模式的结果。这些不证明真实堆叠像素身份或物理抓放。
- **腕部工具**：清单取帧和时间对齐已实现。首次24条草稿经用户复核发现右夹指误识别（`sample-0010`的−6.7°／−107像素无效），成功B被夹指颜色错误排除；旧数字不作诊断结论。现增加固定张开夹指遮罩、候选主轴跨度大于ROI高度一半、适用于较宽竹条的形状门槛，橙红色显示排除区。同帧重测重点叠加图中拟合轴落在竹条上；六张原问题帧及各来源首末抽样图已由助手查看，不替代用户验收或逐帧定量真值。
- **腕部覆盖与草稿**：从78条已接受首帧目标掩码重新计算起始倾角，抽取全部6条−30°～−15°示教、6条中等倾角及全部2条±5°内直条；每条闭合前候选窗口等间隔取样。共84个专家画面＋失败B 4帧＋成功B 3帧，合计91帧／167条时间窗记录，均为程序判有效、未人工接受。旧同帧成功率筛选为0／3、失败4／4，新两组为3／3和4／4；小样本且前置时间筛选仍存在，不能宣称不存在选择偏差。按不同画面统计有效率，超过20个百分点差距提示风险。[固定复核入口](../../artifacts/joint_target_live_preparation/ab_outcome_review/review.html#wrist-angle)链接新叠加图与[修订检查记录](../../artifacts/joint_target_live_preparation/ab_outcome_review/wrist_angle/measurement_revision_check.json)，包含旧新比较、分层来源、可视检查范围。旧报告保留、旧页面标为失效，回归入口拒绝旧颜色单筛选报告；未做真实回归。
- **腕部完整序列与误差准备**：后续按“先分布、再误差、暂不回归”扩展：图像取样不再依赖未来动作窗口，6条斜条示教取完整闭合前记录，19轮具备录像和侧录的实机按配置逐轮保留，包含中止及启动报错轮。0.5秒间隔共643帧，专家155帧中44帧、实机488帧中138帧通过程序条件；无效帧保留，独立单位是6条示教／19轮实机，其中15轮有程序可测角度。未确认首次可见和真实下探边界。程序草稿范围专家−22.08°～15.97°、实机−12.91°～23.23°，这些极值包含早期视角／潜在邻条身份问题，不用于共同状态或覆盖不足结论。助手查看了6条示教与最新5轮各自最大绝对角度的叠加图及分布图；没有逐帧身份或真值验收。
- **人工参考与边界**：[分布复核入口](../../artifacts/joint_target_live_preparation/ab_outcome_review/wrist_angle/distribution_review/review.html)链接20帧两点标注页（15个来源、4帧程序无效），先看原图，不显示程序拟合结果。人工标注尚未导入，真实误差中位数／最大值均未知。新独立工具支持完整样本导入、无向轴误差计算、不可测记录保留和误差尺度对照，不执行回归。验证见[本次JUnit](../../artifacts/joint_stack_preparation/wrist_distribution_tests.xml)：16项通过，含无预测／无闭合／无未来动作仍取帧、独立来源计数、无效帧保留、合成2°人工差值及完整导出要求。PNG／PDF使用系统现有Matplotlib独立绘制，未改项目依赖。浏览器清单仍为空，两个标注页面的实际点选验收未完成；未提交、试采、推理或训练。
- **跳变敏感性检查**：[逐帧与统计对照](../../artifacts/joint_target_live_preparation/ab_outcome_review/wrist_angle/distribution_review/jump_sensitivity.html)及同目录`jump_sensitivity.json`、`jump_sensitivity.py`保存复现依据。按手册的排查门槛得到17对／27个端点（专家8、实机19），助手查看全部27张叠加图；紫线大多沿可见木条，若干B序列前后在不同可见木条区域间切换，不能仅凭腕部图确认目标身份或认定测错。暂时排除后专家36帧、范围−1.69°～15.97°、帧中位数5.62°；实机119帧、范围−6.88°～23.10°、帧中位数12.30°。原始44／138帧和所有无效帧、来源清单不改；6条示教／15轮实机仍有保留角度，数值重叠仍存在，不代表共同状态证据。JSON另列15°／20°门槛对照、逐来源统计和每5°区间独立来源名单。人工两点结果尚未导入；未作回归或零点重设。本次仅增加离线分析产物和文档，未改采集、标注或控制代码；运行脚本的帧数断言、报告重算与本地链接检查通过，不重跑无关软件测试。
- **验证**：首次软件准备203项相关测试通过，另2项真实数据集后端／假设备往返测试通过，合计205项；Python语法和`git diff --check`通过。前203项记录见[JUnit结果](../../artifacts/joint_stack_preparation/software_tests.xml)，范围及两个单独往返测试见[验证摘要](../../artifacts/joint_stack_preparation/verification.json)。后两项曾因默认Hugging Face数据缓存锁目录在沙箱内只读而失败，获权限后重跑通过；没有改共享环境配置。清单生成器与6次尝试补充后，重跑5个相关测试文件，84项通过（本次不重跑2项后端往返）；页面渲染及自定义时间窗边界补充后再跑腕部8项通过，不与84项重复相加。[补充JUnit](../../artifacts/joint_stack_preparation/wrist_followup_tests.xml)及原验证摘要中的`wrist_followup`记录该次范围。夹指误识别修正后，腕部11项测试通过，覆盖夹指与竹条同色、较宽竹条、短碎片、倾角层完整性、重复时间窗去重、分组有效率警告和旧报告拒绝；[本次JUnit](../../artifacts/joint_stack_preparation/wrist_gripper_mask_tests.xml)。本次未改采集或分离运行代码，不重跑其测试。
- **页面边界**：当前会话的浏览器控制清单为空，未完成真实浏览器的点选、导出和导入交互验收。Python导出／导入与像素一致性测试通过，不替代浏览器验收。合成预览：[三根种子选择](../../artifacts/joint_stack_preparation/synthetic_review/review.html)、[腕部角度](../../artifacts/joint_stack_preparation/synthetic_wrist_review/review.html)；预览不含真实实验复核结论。

修改前文件副本保存在`/tmp/aubo-stack-software-before-6uskjdkr/`，包含Claude本轮已经修改的文档，可用于按文件回退本轮软件改动；`/tmp`不是长期备份。本次清单生成器补充前副本在`/tmp/aubo-wrist-followup-before-to4cqaah/`，包含相应未提交文件和原复核入口。夹指遮罩修正前副本位于`/tmp/aubo-wrist-mask-before-h179tmpv/`。完整分布流程修改前副本在`/tmp/aubo-wrist-distribution-before-noybhoiw/`。新增文件及本轮差异以工作区为准，不用`git reset`覆盖Claude的未提交成果。改动尚未提交或推送；`artifacts/`仍由Git忽略。

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
- `onsite_second_grasp_review_20260928/`、`second_contact_candidate_20260928/`：旧第二根失败诊断；后者的高度补偿补丁当时被拒绝，未作为当前运行实现。后来规划的一般堆叠下探高度修正已于2026-10-15取消，从未实现，见[分阶段路线](README.md#分阶段路线)。
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
采前准备未运行模型、训练或浏览器交互验收，未修改首批原数据和标注；后续首对实际采集及检查结果见下节，操作统一见上述方案。

目标示教填写已按用户要求简化为`scene_id`（如`t-001`）与`target_id`（如`s-001`）；摆放恢复及目标对应由操作者现场确认，两个文字描述字段留空，旧数据描述保留。
`tests/bamboo_sorting/test_aubo_joint_capture.py`的27项检查通过，包括假设备两条同场景目标示教的简化输入、保存层元数据校验及普通采集兼容；未连接真实设备。

### 首对A／B示教采后检查

`datasets/joint_target_pair_pilot`已完成2/2条录制和视频整理，无重录或未完成条目，均为`train`、人工结果`single_success`。
固定[复核页](../../artifacts/joint_target_pair_pilot/review.html)与[录制检查](../../artifacts/joint_target_pair_pilot/capture_check.json)保留本批离线结果；原始数据、录像和元数据未改写。

| 人工条号（episode_index） | 原始场景／目标 | 帧数／时长 | 闭合／释放帧 | 接近／搬运帧 |
| --- | --- | --- | --- | --- |
| 1（0） | `t-001 / s-001` | 712／28.48秒 | 358／642 | 359／353 |
| 2（1） | `t-002 / s-002` | 804／32.16秒 | 368／669 | 369／435 |

- `audit_joint_dataset.py --require-gripper-quality`通过：1516帧七维状态／已接受动作、夹爪开合侧录及源时间记录一致；每条仅一次闭合→释放。
- 全局／腕部视频各1516帧完整解码，640×480、25 Hz，逐帧视频时间与Parquet对应；合计60.64秒。两条起始六关节最大差约0.000153°，夹爪均为张开。
- 每条抽查9张全局图及9张腕部图，共36张；起始画面均有两根分离竹条。第一条抓右侧抓取台上靠画面左侧的竹条，第二条抓靠画面右侧的竹条；抽查夹持、搬运及结束画面与单根移至左侧收集区并松开的人工结果一致。关键帧抽查不是全程逐帧视觉验收，更不是策略成功率。
- 用户明确确认两条恢复同一摆放，`t-002`只是编号递增；检查报告`pair_group`记录共同场景组`t-001`，原始编号保留。后续划分训练／验证时必须同组，不能只按原始`scene_id`拆分。
- 第二条释放后frame 740–742附近有一次采样间隔增大：机器人状态约248 ms、全局图约258 ms、腕部图约251 ms；没有源时间倒退、重复或缺失编码帧。原因未确定，保留记录、不重定时；完整视频不代表采样始终严格40 ms。
- [标注准备检查](../../artifacts/joint_target_pair_pilot/target_annotations/preparation_check.json)：728张接近阶段原图已按各条视频偏移提取，解码时间全部匹配；788帧搬运按接口规则保留原图。当前728张掩码均已由用户复核接受，`target_source_ready=true`、`target_training_ready=false`，验收结果见下文。

采后审计与原图准备阶段未加载模型、启动训练、远端推理或连接硬件；随后用户要求“进行下一步”，执行了以下限定本批728帧的冻结推理与草稿准备。

- [推理执行记录](../../artifacts/joint_target_pair_pilot/target_annotations/inference_request.json)：核对远端`gpu`现有Python、CUDA与第15轮`best`模型，在新的本批目录上传728张RGB、输入清单和独立预测脚本，完成一次推理并回传。原640×480、batch size 2、置信度与掩码阈值均0.5，0次优化，耗时25.72秒；模型加载无缺失、多余或不匹配参数。
- [预测统计](../../artifacts/joint_target_pair_pilot/target_annotations/prediction_check.json)：第一条1／2／3实例帧数为137／219／3，第二条为263／105／1。**全部728帧至少有一个预测实例同时覆盖左右两根的区域**，部分帧又将单根拆成多个实例；实例计数不等于正确分离的物体数，当前模型不能据此声称具有可靠实例分离能力。原预测保留在`predictions/`，未重跑或覆盖。
- 默认导入没有自动选目标，也未使用`--single-strip-scenes`。随后针对本批固定相机、左右分离摆放，用[区域选择记录](../../artifacts/joint_target_pair_pilot/target_annotations/target_region_selection.json)明确0条取左侧、1条取右侧：模型前景与目标矩形区域取交集，不填矩形、不补遮挡、不按实例ID跟踪。所有帧分隔带均无预测前景；区域外的误分被排除。此方法不适用于任意移动、交叉或重叠目标。
- [草稿生成检查](../../artifacts/joint_target_pair_pilot/target_annotations/target_region_check.json)：728张640×480非空二值掩码，均为原预测前景的子集；原预测、视频及动作不变，生成时全部`reviewed=false`，之后的用户确认记录见下文。生成脚本保存在同目录`select_target_regions.py`，拒绝覆盖已有掩码。
- [图像抽查记录](../../artifacts/joint_target_pair_pilot/target_annotations/visual_check.json)：抽查30个不同目标草稿帧，左右目标选择符合本批抓取记录；0条343／354／358帧及1条345／346帧有明显可见表面漏分或碎裂，抽查时设为待复核，随后用户接受现有误差。这是抽查发现，不是全部问题清单；原抽查报告保留。
- [浏览器检查](../../artifacts/joint_target_pair_pilot/target_annotations/browser_check.json)：独立临时Chrome配置打开file入口，确认图像加载、两条闭合观察帧切换、5个问题定位及已有异常阻止区间确认；导出名已改为随批次生成。未写入测试复核决定。

用户随后明确反馈“我已明确复核，识别情况正常，误差可以接受”。依此将本批728帧及2条episode的`reviewed`设为true；5个已知问题转为`accepted_annotation_limitations`，保留原问题、确认原话和逐帧复核历史，原预测与掩码未修改。原话及范围见`annotations.json`的`human_review`与[复核决定](../../artifacts/joint_target_pair_pilot/target_annotations/review_decisions.json)。这是接受现有标注的工程取舍，不是已修复误差或逐像素真值。

[完整数据接口验收](../../artifacts/joint_target_pair_pilot/target_annotations/dataset_check.json)通过：实际`LeRobotDataset`以PyAV读取2条1516帧，728个接近帧RGB与标注来源PNG完全一致，788帧搬运使用原图；七维状态、绝对关节动作、50步动作序列及每条49个尾部填充窗口保持原值。训练数据视图与`TargetInputSession`的输入图像和因果阶段切换逐帧一致，释放后不重新描边。

本批`target_source_ready=true`；本批验收时尚无独立验证集，`target_training_ready=false`，未训练目标动作模型、未验证物理目标服从。本次复核登记与接口验收未加载模型或连接设备，未更新权重、提交或推送。下一步确定成对训练规模及新物理摆放的独立验证数据，不要求重录本批或继续修补已接受误差。

### 成对训练数据采后检查

`datasets/joint_target_pair_train_batch01`已完成5组10条：`t-003`～`t-007`各有`s-001`和`s-002`，全部为`train`，人工结果均为`single_success`。原始数据未改写，检查保存在`artifacts/joint_target_pair_train_batch01`的[原始审计](../../artifacts/joint_target_pair_train_batch01/raw_audit.json)、[采后报告](../../artifacts/joint_target_pair_train_batch01/capture_check.json)和[固定复核页](../../artifacts/joint_target_pair_train_batch01/review.html)。

- 6711帧、268.44秒；七维状态／已接受动作、侧录与正式数据一致，每条仅一次闭合→释放。双相机各6711帧完整解码，640×480、25 Hz，全部视频时间与Parquet匹配。
- 日志含11次尝试：`t-007 / s-001`的一次434帧废弃尝试保留于日志，正式数据只含重录后的667帧；没有未完成正式条目。
- 每条抽查9张全局及9张腕部图，共180张。5组起始画面均有两根分离竹条，组内摆放视觉上恢复、存在小幅位置差异；各组分别抓走不同竹条并移至左侧收集区，关键帧未发现需重录问题。起始六关节组内最大差0.000153°，夹爪均张开。这是关键帧抽查，不是全程逐帧视觉真值或策略成功率。
- 观察到`t-003`～`t-006`的`s-001`抓画面左侧、`s-002`抓右侧；`t-007`的`s-001`抓画面右侧、`s-002`抓左侧。目标编号不固定表示左右，后续掩码须分别对应本条实际目标。
- 源采样间隔中位数约40 ms，最大78.27 ms，双相机最大源时间差40.01 ms；无时间倒退、重复或超过80 ms的间隔。80 ms仅为本次诊断统计阈值，完整编码不表示源采样严格等间隔。
- 接近／夹取3331帧、搬运3380帧。采后检查阶段未加载模型、远端推理、训练或连接硬件；用户随后授权“进行下一步”，本批目标草稿准备如下。累计6组训练（12条）均已完成用户标注复核与数据接口验收；完整首轮预算与余量以项目手册为准。

本批标注证据保存在`artifacts/joint_target_pair_train_batch01/target_annotations`：

- `inference_request.json`记录3331张接近RGB及一次完成的冻结推理。按用户指定，远端复用首对目录`artifacts/joint_target_pair_pilot/target_annotations`及其`images`／`predictions`，本批所有输入与预测文件使用`train_batch01_`前缀，拒绝覆盖。独立预测脚本副本只适配文件命名，不改变模型参数。首次启动遗漏原项目`deps`路径，在模型初始化时失败、没有预测输出；复用既有`deps/scipy`后完成，未安装软件或改变共享环境。
- 第15轮最佳模型，原640×480，batch size 2、阈值0.5，3331帧推理耗时117.36秒，0次优化；`predictions/train_batch01_loading_info.json`确认权重加载无缺失、多余、不匹配参数。原预测保留；`prediction_check.json`显示2816帧有实例横跨两根的选择区域，不能把实例编号当目标身份。
- `target_region_selection.json`按每条原始画面记录两根中心线和实际目标。模型前景与目标中心线20像素范围、较另一根更近的区域取交集，不填区域或补遮挡。`target_region_check.json`保留初始3315张非空草稿、16帧空目标区域的完整统计；此方法仅限本批接近期间静态分离摆放，不是运行时跟踪。
- 16帧空区域均在episode 9（`t-007 / s-002`）。助手查看每张原图及补绘结果，按可见区域绘制多边形，并用记录的颜色条件排除穿过多边形的暗色线缆／夹爪；最终几何与条件见`manual_missing_mask_drafts.json`。它们是助手草稿，不是模型输出或用户确认，边界仍需复核，原预测不变。
- `visual_check.json`保留80个模型草稿关键帧的抽查及全部16帧补绘检查范围；准备时页面标出22个已发现或疑似漏分位置，其中7个由图像抽查发现、15个由目标面积低于初始35%的诊断提示；其后用户复核接受现有轮廓，原抽查报告保留。抽查不等于完整逐帧验收，问题列表不保证穷尽。
- 用户明确反馈“复核完毕，没有问题”。据此登记本批3331帧及10条episode为`reviewed=true`，包括16帧助手补绘；22个既有问题或疑似异常转为`accepted_annotation_limitations`，保留来源、确认原话与逐帧复核历史，原预测和掩码未修改。这是接受当前标注，不表示误差已修复或逐像素真值。决定见`review_decisions.json`及`annotations.json`的`human_review`。
- `dataset_check.json`完整接口验收通过：实际LeRobotDataset／PyAV读取全部10条6711帧，3331帧接近RGB与标注来源一致，3380帧搬运使用原图；状态、绝对关节动作、50步动作序列及每条49个尾部填充窗口保持原值，训练／运行输入与阶段逐帧一致。`target_source_ready=true`；尚缺独立验证来源，`target_training_ready=false`。本次登记与验收未加载模型、启动训练或连接机器人。
- 准备阶段的`browser_check.json`记录独立临时Chrome的file入口检查：10条的闭合观察帧原图／轮廓均加载，22个问题定位项可见，已记录异常阻止直接区间确认，导出名为`joint_target_pair_train_batch01_review_decisions.json`。未写入测试复核决定；3331组原图、掩码和轮廓文件路径均已检查。

浏览器打开本批目标轮廓与原始录制复核页：

```bash
google-chrome --new-window "file:///home/rentao/program/lerobot-aubo/artifacts/joint_target_pair_train_batch01/review.html"
```

`datasets/joint_target_pair_train_batch02`已完成第二批5组10条，实际场景组为`t-008`～`t-012`，全部`train`、人工结果均为`single_success`。本批证据：[原始审计](../../artifacts/joint_target_pair_train_batch02/raw_audit.json)、[采后报告](../../artifacts/joint_target_pair_train_batch02/capture_check.json)、[固定复核页](../../artifacts/joint_target_pair_train_batch02/review.html)。

- 6472帧、258.88秒；七维状态／已接受动作及侧录一致，每条一次闭合→释放。双相机各6472帧完整解码、640×480，全部视频时间与Parquet匹配。接近／夹取3192帧，搬运3280帧。
- 第6条（episode 5）原始为`t-009 / s-002`，用户明确确认“是，实际属于 t-010，编号填错了”。报告的`pair_groups`及`effective_scene_group_id`按`t-010`登记，原始session、侧录和数据元信息不改；后续目标标注与训练来源整理必须沿用此纠正关系。
- 11次尝试中，第10条有一次127帧`abandoned_rerecorded`；正式保存的是重录后的674帧，无未完成正式条目。
- 共180张关键帧加12张采样间隔异常前后图像已由助手查看。5组内起始摆放大致恢复，有小幅自然位置差异；每组分别抓走不同竹条并移至左侧收集区，抽查未见需重录问题。组内起始六关节最大差0.000229°，夹爪均张开。`t-008`的`s-001`抓右、`s-002`抓左；`t-009`及`t-011`相反；`t-010`分别为画面左上／右下，`t-012`分别为画面上／下。编号不固定代表左右，后续掩码需按各条实际目标对应。此结论限于抽查，不是全程逐帧视觉真值或策略成功率。
- 源采样间隔中位数约40 ms、双相机最大源时间差35.85 ms。episode 3的frame 552（释放前）三路全局／腕部／状态间隔约253.55／245.08／219.65 ms；episode 9的frame 244（接近阶段）约188.01／183.74／153.71 ms。无源时间倒退或重复、无缺失编码帧，原因未确定，原记录保留、不重定时。80 ms仅为诊断阈值。两处间隔前后关节状态最大变化分别0.3753°／0.1628°，未超过各自条内最大单步变化；[补充检查](../../artifacts/joint_target_pair_train_batch02/timing_gap_check.json)和页面保留证据，不能据此证明严格等间隔采样。
- 采后检查阶段未加载模型、远端推理、训练或连接硬件。随后用户要求“可以先准备本批目标轮廓，然后我在录制”，完成下述草稿准备。当时本批尚待用户复核；现已按下述明确接受结论完成接口验收。第三批使用`t-013`～`t-017`，当前全量状态见本节末尾。

第二批标注证据集中在`artifacts/joint_target_pair_train_batch02/target_annotations`：

- `inference_request.json`记录本次授权及完成状态：3192张接近RGB按实际视频偏移提取、全部PTS匹配。远端沿用首对目录`artifacts/joint_target_pair_pilot/target_annotations`及其`images`／`predictions`，使用`train_batch02_`前缀，拒绝覆盖已有结果。沿用既有虚拟环境及项目`deps`，未安装软件或修改共享环境。
- 第15轮最佳冻结模型、640×480原图、batch size 2、阈值0.5；仅运行一次推理，耗时112.96秒、0次优化。`predictions/train_batch02_loading_info.json`确认权重无缺失／多余／不匹配参数；`inference.log`及3192张原预测保留。`prediction_check.json`显示3053帧存在预测实例横跨两根选择区域，不能按实例编号认定目标身份。
- `target_region_selection.json`记录每条两根竹条中心线及实际目标；用距目标中心线20像素内且较另一根更近的区域筛选模型前景，仅用于本批静态分离摆放。`target_region_check.json`保留初始3190张非空目标草稿、2帧空区域及逐帧统计，不补被遮挡区域、不作为运行时跟踪。
- 两帧空目标区域均在episode 1（`t-008 / s-002`）的234、266帧。助手查看各自原图后补绘可见多边形，并以记录的颜色条件排除暗色线缆／夹爪；几何、条件、来源和待复核状态见`manual_missing_mask_drafts.json`。两帧补绘结果已由助手查看，来源仍是助手补绘而非模型输出；后续整批用户接受范围明确包含这两帧。
- `visual_check.json`记录80个关键帧、4个额外诊断帧及两帧补绘原图／结果的抽查（集合有重叠），最终3192张掩码均存在。26个已知或疑似问题包括可见表面碎裂、遮挡后方漏段、夹爪／背景混入及两帧补绘待复核，草稿阶段写入各帧`review_issue`并在网页提供定位，清单不保证穷尽。用户现已明确整批复核，反馈“接受现有轮廓及这些误差，用于本轮训练候选，保留问题记录”；3192帧据此登记`reviewed=true`，26项转存`accepted_annotation_limitations`，问题原文及`review_history`保留，未修掩码，不宣称逐像素真值。
- `preparation_check.json`确认3192张640×480二值非空掩码及轮廓齐全、缺失0、已复核3192、已接受误差26帧。原`browser_check.json`保留草稿阶段（当时已复核0）的记录：独立临时Chrome的file入口验证：10条闭合观察帧原图／轮廓均加载、26个问题定位项可用，已记录异常阻止直接区间确认，导出名为`joint_target_pair_train_batch02_review_decisions.json`；9576个标注资源路径及210个HTML引用已检查，无缺失，未写入测试复核决定。
- 第6条标注侧录保留原始`scene_id=t-009`以匹配数据契约，并用`effective_scene_group_id=t-010`及`scene_group_correction`保存用户确认的实际配对关系；页面说明同步显示纠正关系。原数据、动作、预测和前批标注不改，未训练目标策略或连接硬件。

- `review_decisions.json`和`annotations.json.human_review`记录本次两步用户反馈：整批范围＋接受已知误差，覆盖全部10条、3192帧及两帧补绘；原侧录、准备报告和页面分别保存在`*_before_user_acceptance.json`及`review_before_user_acceptance.html`。未重复要求确认首对、第一批或已经明确的现场摆放。
- `dataset_check.json`完整接口验收通过：实际LeRobotDataset／PyAV读取10条6472帧，3192帧接近图像匹配标注原图、3280帧搬运保持原图，state/action、50步动作块、每条49个尾部填充窗口及训练／运行阶段逐帧一致。`target_source_ready=true`；验证原始录制已存在，但验证目标来源尚未验收，全量`target_training_ready=false`。首次检查在数据加载时因沙箱缓存目录只读停止，获缓存访问权限后完成；两次均未加载策略、远端推理或连接设备。
- `browser_acceptance_check.json`记录接受后的独立临时Chrome检查：3192已复核状态、26个已接受误差定位项及10条闭合观察帧原图／轮廓加载通过，未写入浏览器测试确认。历史`browser_check.json`及预测／抽查报告保留。

浏览器直接打开第二批复核页：

```bash
google-chrome --new-window "file:///home/rentao/program/lerobot-aubo/artifacts/joint_target_pair_train_batch02/review.html"
```

`datasets/joint_target_pair_train_batch03`第三批采后检查已完成，编号`t-013`～`t-017`，10条正式示教均为`train`、人工结果均记录为`single_success`。证据：[原始审计](../../artifacts/joint_target_pair_train_batch03/raw_audit.json)、[采后报告](../../artifacts/joint_target_pair_train_batch03/capture_check.json)、[固定复核页](../../artifacts/joint_target_pair_train_batch03/review.html)。

- 6672帧、266.88秒；七维状态／已接受动作及逐条侧录一致，每条一次闭合→释放。双相机各6672帧完整解码、640×480，全部视频时间与Parquet匹配；接近／夹取3215帧、搬运3457帧；采后检查时轮廓未准备，现已完成下述草稿。
- 12次尝试中，episode 5（`t-015 / s-002`）有479帧、111帧两次`abandoned_rerecorded`；正式保存的是最终707帧版本，废弃尝试未混入正式数据，日志保留，无未完成正式条目。
- 每条9张全局加9张腕部图，共180张关键帧，及5组起始对照已查看。抽查可见每条夹持单根移至左侧收集区，释放／结束画面与人工结果相符，未见明显抓放失败；这是抽查，不是全程逐帧视觉验收或策略成功率。
- **`t-013`右侧竹条在episode 0与1起始画面中位置、角度明显改变，不能算严格的相同摆放仅切换目标A/B对照。**报告登记`same_scene_pair_candidate=false`，两条作为额外示教保留；当时提出补足配对预算，当前全量30组预算已齐，不要求重录或改动原始编号与数据。放大的起始对照已嵌入复核页。其余4组为大致恢复摆放的配对候选，其中`t-015`左侧条仍有较小角度差异；手工恢复并非像素级一致，后续应注意两根端点位置。
- 实际目标：`t-013`、`t-014`的`s-001`抓右、`s-002`抓左；`t-015`先左后右；`t-016`先右上后左下；`t-017`先左上后右下。各条掩码须对应实际目标，不能固定按左右分配编号。组内起始六关节最大差0.000381°，夹爪均张开。
- 源采样间隔中位数约40 ms，三路最大75.14 ms，双相机最大源时间差34.81 ms，无时间倒退、重复或超过80 ms的源采样间隔；80 ms仅为本次诊断统计阈值。完整编码不代表严格等间隔采样。
- 本次只做离线采后检查，原始数据未改，未加载模型、远端推理、训练或连接硬件。当时第二批草稿尚待复核，未以第三批录制完成代替用户确认；第二批后来按本节明确反馈接受。此阶段累计32条，当前全量状态见下文。

第三批目标草稿集中在`artifacts/joint_target_pair_train_batch03/target_annotations`：

- 用户明确授权“可以，进行授权进行下一步”，范围为本批3215张接近图像的一次冻结推理、回传与待复核草稿准备。`inference_request.json`记录授权和完成状态；`remote_preflight.json`核对第15轮`best`、640×480契约、原虚拟环境／deps、CUDA可用且无本批同名前缀文件，当时GPU无计算进程。未安装软件或改变共享环境。
- `upload_check.json`记录3217个新文件（3215张图＋清单＋脚本）上传、0覆盖。远端沿用首对共享目录，使用`train_batch03_`前缀；仅运行一次，耗时113.57秒、0次优化，3215份结果全部回传。原预测、`inference.log`、空missing／unexpected／mismatched／error加载记录均保留；`frozen_model_check.json`确认模型文件大小及mtime未改变，未重复计算权重哈希。
- 双条预测不能直接用实例编号识别目标：`prediction_check.json`记录3024帧中有实例横跨两根候选区域。`target_region_selection.json`按本批10张起始原图及既有实际抓取映射确定两根中心线；颜色支持的主轴拟合仅细化初始几何。用距目标中心线18 px内且较另一根更近的区域筛选当前模型前景，不把整个候选区域填成前景；仅适用于本批静态分离摆放，不是运行时追踪。
- 初筛3200张非空草稿、15帧目标前景缺失，原统计保存在`target_region_check.json`。逐一查看这15帧原图后，以候选多边形与该帧颜色条件制作助手草稿，不沿用相邻帧掩码；来源、坐标和条件见`manual_missing_mask_drafts.json`。episode 9初稿有同色腕部盒体混入，复查后增加盒体边界排除，旧掩码字节保存在`previous_masks`；修正后仍有碎裂和同色像素混入风险，草稿阶段全部待复核；现已按下述用户结论接受，原风险记录不删除。
- `visual_check.json`记录80个关键帧、5个额外低面积诊断帧，以及全部15帧补绘原图和结果的助手检查，集合有重叠。页面提供54个已知或疑似问题定位，涵盖可见表面碎裂、遮挡后可见段漏分、夹爪／背景边界和补绘待确认；其中低于初始面积35%的提示仅为诊断，问题清单不保证穷尽。该抽查报告保留生成时3215帧`reviewed=false`的历史状态；当前接受状态见下文。
- 当前`preparation_check.json`确认3215张640×480二值非空掩码及内侧2 px轮廓齐全，缺失0、已复核3215、已接受误差54帧。`browser_check.json`检查10条闭合观察帧原图／轮廓加载、54项问题定位及异常阻止区间确认，导出名为`joint_target_pair_train_batch03_review_decisions.json`；未提交浏览器测试确认。原`browser_preparation_check.json`保留无掩码准备阶段的记录。
- 用户随后明确“明确可以全部接受。”，范围为本批3215张现有轮廓及54项问题（含15张助手补绘）。`review_decisions.json`和`annotations.json.human_review`记录原话、范围及修订号，54项问题转存`accepted_annotation_limitations`并保留逐帧历史，掩码大小和mtime未变；接受前注释／报告／页面另存原文件备份。该结论不扩展到后来生成的8批草稿。
- `dataset_check.json`与`dataset_verify.log`记录本批实际LeRobotDataset/PyAV完整验收：10条6672帧、3215接近／3457搬运，原始7D state/action和50步动作块／每条49个尾部填充窗口不变；来源PNG与实际加载RGB、训练／运行时图像和因果阶段逐帧一致。`target_source_ready=true`，全量仍`target_training_ready=false`。`review_acceptance_check.json`记录54项误差接受且掩码未改；`browser_acceptance_check.json`记录当前页面状态，原草稿`browser_check.json`保留。

第三批原始录制复核页：

```bash
google-chrome --new-window "file:///home/rentao/program/lerobot-aubo/artifacts/joint_target_pair_train_batch03/review.html"
```

第四批`joint_target_pair_train_batch04`在`t-021 / s-001`中断。`session.json`记录请求10条、正式保存6条、`capture_complete=false`且`dataset_finalized=true`：后者只表示已保存数据完成整理，不表示整批录完。证据：[原始审计](../../artifacts/joint_target_pair_train_batch04/raw_audit.json)、[采后报告](../../artifacts/joint_target_pair_train_batch04/capture_check.json)、[待复核页面](../../artifacts/joint_target_pair_train_batch04/review.html)。

- `t-018`～`t-020`正式6条共4092帧，双相机各4092帧完整解码、640×480，全部视频时间与Parquet匹配；七维状态／已接受动作及夹爪审计通过。接近／夹取1998帧、搬运2094帧，无超过80 ms的源采样间隔。人工侧录均为`single_success`，中断检查时未完成抓放关键帧视觉复核；随后已抽查108张关键帧，未见明显抓放失败，结论限于抽查。
- 第7次尝试为`t-021 / s-001`，记录234帧后以`incomplete`保留，原因是`no fresh C0 camera frame within 0.200s: grasp_rgb`；未进入正式Parquet或合并视频。随后按用户明确要求，删除该尝试两路各234张、共468张PNG（193597287字节）及两个`episode-000006`缓存目录，删除后未留下PNG缓存；正式数据／视频／元数据的大小及修改时间未变，再次核对正式数据仅episode 0～5、共4092帧。中断侧录与会话日志保留，清理结果写入`capture_check.json`的`interruption.residual_image_cleanup`。已生成108张关键帧及起始对照，随后完成关键帧及起始对照抽查：`t-019`右条起始角度／位置明显改变，不计严格同场景配对，保留两条额外示教；`t-018`、`t-020`为配对候选，后者左上条仍有小幅角度差异。
- 终端截图的首要问题为Sonix腕部相机读帧失败，继而触发200 ms新帧检查停止。只读主机检查时两路`by-id`节点仍存在，但无法读取内核日志，不能确定USB线、供电、驱动或设备层面的根因；本次未连接相机或机器人验证恢复。
- 另一个`NoneType.is_set`来自退出阶段：硬件读操作超过2秒退出等待后，清理代码清空`self.stop_event`，旧读线程返回后再次访问它。`camera_opencv.py`改为保留线程自己的停止事件，并在读操作返回／失败后检查停止状态，避免清理后的帧发布及该路径的设备重开。未放宽200 ms阈值或更改动作限制。新增两项模拟测试修复前复现、修复后通过（`2 passed, 40 deselected`）；完整OpenCV测试为14通过、28失败，图片解码用例受本地Git LFS占位文件影响，不能视为全套通过。
- 剩余4条由用户在新目录`joint_target_pair_train_batch04_remaining`手动补录，检查结果如下；原目录会话的中断状态保留。采后检查无模型加载、训练或硬件连接。

第四批补录`joint_target_pair_train_batch04_remaining`采后检查完成：两组`t-021`、`t-022`共4条，`capture_complete=true`、`dataset_finalized=true`。证据：[原始审计](../../artifacts/joint_target_pair_train_batch04_remaining/raw_audit.json)、[采后报告](../../artifacts/joint_target_pair_train_batch04_remaining/capture_check.json)、[复核页面](../../artifacts/joint_target_pair_train_batch04_remaining/review.html)。

- 正式帧数依次602、752、663、831，合计2848帧、113.92秒；双相机均完整解码，640×480且全部视频时间与Parquet匹配，七维状态／已接受动作及夹爪审计通过。接近／夹取1410帧、搬运1438帧，尚未准备目标轮廓。
- 共7次尝试：episode 0、2、3各有一次319、471、118帧的`abandoned_rerecorded`；正式数据仅包含最终版本，旧尝试未混入，两处第四批目录均无PNG缓存。旧中断的234帧图像此前按用户要求删除，中断日志仍保留。
- 已查看72张关键帧及两组起始对照，抽查显示单根被搬至收集区，未见明显抓放失败；结果限于抽查，不是全程逐帧视觉验收。`t-021`的`s-001`抓下侧、`s-002`抓上侧；`t-022`先录`s-002`抓左条，再录`s-001`抓右条，两组起始摆放大致恢复，均记为配对候选。目标编号须按实际对象准备掩码。
- 最大源采样间隔75.14 ms、最大双相机源时间差34.57 ms，无超过80 ms的间隔。正式补录数据未出现此前导致停止的长断帧，但不能由一次完成就断言硬件根因消除。
- 第四批两个目录合计10条、6940帧，其中接近／夹取3408帧。累计正式录制42条；第四批计4组成对来源候选及`t-019`的2条额外示教。此阶段第二批目标草稿尚待明确复核，第三／四批目标轮廓未准备，未训练；当时第五批`t-023`～`t-027`的10条命令只完成计划检查。后续录制已完成，当前标注状态统一见本节末尾及[项目手册](README.md)。

后续训练录制再次中断：`joint_target_pair_train_remaining`会话／元数据记录`t-023`～`t-028`共12条、8268帧已整理；中断排查时仅核对保存清单，完整采后检查现已完成，见下段。随后`joint_target_pair_train_t029_t033`仅保存`t-029 / s-001`一条632帧，`t-029 / s-002`记录538帧后因`no fresh C0 camera frame within 0.200s: grasp_rgb`中断。

- 本次对632帧执行动作／夹爪审计及双相机完整解码，均通过；正式Parquet仅episode 0，视频时间全部匹配，未做抓放视觉验收。证据：[中断检查](../../artifacts/joint_target_pair_train_t029_t033/interruption_check.json)、[动作审计](../../artifacts/joint_target_pair_train_t029_t033/raw_audit.json)。
- 538帧未完成尝试未混入正式Parquet或视频，现有两路各538张PNG缓存及故障日志保留；最后记录的夹爪动作为闭合100，不据此推定物理结果。截图未出现此前`NoneType.is_set`退出异常；腕部相机无新帧问题仍未排除。内核日志不可读，具体USB／驱动根因未知，未放宽新鲜度阈值。
- 已准备用户手动双相机预检（`diag_preflight_cams.py --camera-set original-global --seconds 60`，不连接机械臂）及从`t-029 / s-002`续录9条的方案，两者仅打印计划检查通过，尚未运行硬件。原始数据未改，未删除本次缓存。续录命令见[项目手册](README.md)。静态相机预检通过不能证明机械臂运动时的线缆连接稳定。

t-023～t-033的最后三处训练录制目录已完成采后检查，原录制目录不合并、不重编号。正式数据合计22条、14915帧；来源与结果如下：

| 数据目录 | 正式条数／帧数 | 接近夹取／搬运帧数 | 报告 |
| --- | --- | --- | --- |
| `joint_target_pair_train_remaining` | 12／8268 | 3939／4329 | [采后报告](../../artifacts/joint_target_pair_train_remaining/capture_check.json) |
| `joint_target_pair_train_t029_t033` | 1／632 | 320／312 | [采后报告](../../artifacts/joint_target_pair_train_t029_t033/capture_check.json) |
| `joint_target_pair_train_t029_s002_t033` | 9／6015 | 2918／3097 | [采后报告](../../artifacts/joint_target_pair_train_t029_s002_t033/capture_check.json) |

- 本轮三处动作／夹爪审计通过；双相机各14915帧完整解码、640×480且全部视频时间与Parquet一致。正式条目人工结果均为`single_success`，每条一次闭合→释放。396张关键帧抽查可见单根搬至收集区，未见明显抓放失败；不是全程逐帧验收或策略成功率。
- 旧`t-028 / s-002`225帧、最后目录`t-032 / s-001`426帧均为手动废弃重录；先前`t-029 / s-001`0帧中断和`t-029 / s-002`538帧中断均未混入正式数据。538帧的两路共1076张PNG缓存仍保留在旧目录，正式加载只使用已保存episode及Parquet／视频，不将缓存算入训练。中断会话保留`capture_complete=false`作为历史证据，后续补录完成不改写原会话。
- `t-029`实际两条为旧目录episode 0的`s-001`与最终目录episode 0的`s-002`，已在两边报告登记跨目录引用、核对分别抓左／右条；起始六关节最大差0.000155°，但右条起始角度有可见偏差。**`t-030`左条从斜放变成近竖直，不计同场景配对，保留为两条额外示教。**`t-028`右条及`t-032`左条有可见角度差异，保留近似候选与偏差说明，不能作为严格相同观察实验。其余新组起始摆放大致恢复。
- 实际目标：`t-023`、`t-026`、`t-029`、`t-030`、`t-032`为`s-001`左、`s-002`右；`t-024`、`t-025`、`t-027`、`t-028`、`t-033`为`s-001`右、`s-002`左；`t-031`先抓右下、后抓左上。轮廓必须对应实际目标，不按固定左右套用。
- 8条有超过80 ms的源采样间隔：t-024两条、t-025/s-002、t-027两条、t-028/s-002、t-031/s-002、t-032/s-002。最大250.49 ms，双相机最大源时间差35.52 ms。三路在附近均有间隔变长，不应仅归因于腕部相机；另查看48张异常附近图像未见明显滑落或抓放失败。原始时间保留、不重定时；新帧等待检查并非对相邻保存帧间隔的硬上限，完整视频也不代表严格等间隔采样。[异常抽查报告](../../artifacts/joint_target_pair_train_t029_s002_t033/timing_gap_check.json)。
- 汇总现有9处正式数据共64条、43226帧；扣除`t-013`、`t-019`、`t-030`三组，按29组近似配对候选＋6条额外示教记账。还需1组训练配对及6组独立验证。[训练录制清单](../../artifacts/joint_target_pair_train_t029_s002_t033/training_capture_inventory.json)；[复核入口](../../artifacts/joint_target_pair_train_t029_s002_t033/review.html)包含t-029跨目录对照、其他两目录链接及间隔异常附近图像。近似候选并非严格相同观察，也不代表目标标注验收。
- 本次新增7177帧接近／夹取目标轮廓待准备，未加载模型、未训练、未连接硬件，未删除／修改原始示教。首对和第一批的标注验收保持原状态，第二批草稿不因录制完成自动确认。当前余量及下一步统一见[项目手册](README.md)。

首轮最后14条录制采后检查已完成：

| 数据目录 | 正式条数／帧数 | 接近夹取／搬运帧数 | 报告 |
| --- | --- | --- | --- |
| `joint_target_pair_train_t034` | 2／1367 | 685／682 | [采后报告](../../artifacts/joint_target_pair_train_t034/capture_check.json) |
| `joint_target_pair_validation` | 7／4747 | 2386／2361 | [采后报告](../../artifacts/joint_target_pair_validation/capture_check.json) |
| `joint_target_pair_validation_remaining` | 5／3487 | 1729／1758 | [采后报告](../../artifacts/joint_target_pair_validation_remaining/capture_check.json) |

- 三处动作／夹爪审计通过，正式结果均记录为`single_success`，每条一次闭合→释放。双相机各9601帧完整解码，640×480，全部视频时间与Parquet一致；252张关键帧可见单根夹取、搬至收集区，未见明显抓放失败。属于关键帧抽查，不是全程逐帧验收或模型成功率。
- `t-034`及6组验证均按起始摆放大致恢复的近似配对候选登记；`t-101`左条、`t-102`端点及`t-106`右条存在小幅恢复差异。`t-104`跨目录引用为原验证目录episode 6（s-001）和续录目录episode 0（s-002），起始六关节最大差0.0000773°；两条画面恢复了两根摆放，不是严格相同观察。
- 实际目标：`t-034`、`t-101`、`t-103`、`t-105`为s-001左／s-002右；`t-102`、`t-106`为s-001右／s-002左；`t-104`为s-001右下／s-002左上。后续标注按实际目标匹配，不能把身份固定为左右。
- `t-102 / s-002`在frame 105附近约142 ms，`t-103 / s-001`在frame 52～53附近最大243.52 ms；三路源时间均有相邻间隔变长。另查14张附近图像，两处均在尚未夹取的接近早期，未见明显目标滑落或抓放失败；保留原时间、不重定时，根因未确立。本轮双相机最大源时间差35.41 ms。
- 原验证目录第8次尝试`t-104 / s-002`因腕部相机无新帧中断，仅299帧；未进入正式Parquet／视频，两路共598张PNG缓存仍保留。原会话`capture_complete=false`、`dataset_finalized=true`作为历史证据，补录完成不改写旧状态。此次没有删除或修改原始数据，也未连接硬件、加载模型或训练。
- 12处正式数据仍为78条52827帧：训练66条44593帧＝30组近似配对候选＋6条额外示教；验证6组12条8234帧。录制预算已齐；后续全部25851帧目标轮廓接受、12个来源完整接口及全量preflight现已通过。`target_data_preflight_ready=true`；后续已获明确执行授权，训练完成及基础离线验证证据见下文及`target_training_execution.json`；已接受误差不作精确像素真值声明。
- [统一录制清单](../../artifacts/joint_target_pair_train_t029_s002_t033/training_capture_inventory.json)已原地更新；[本轮复核入口](../../artifacts/joint_target_pair_validation_remaining/review.html)链接另两目录，并显示`t-104`跨目录起始对照。后续工作以[项目手册](README.md)为准，整组验证数据不得混入训练。

**全量标注本地准备与训练候选清单**

统一清单目录仍为`artifacts/joint_target_pair_train_t029_s002_t033/`。`annotation_preparation_check.json`记录新增9个来源、56条、18600张接近RGB的本地提取，全部按Parquet的episode视频偏移读取，PTS匹配；未重做双路全程解码。本地准备阶段各来源`preparation_check.json`为掩码0、复核0，网页仅展示原图；batch03后续草稿状态见上文。原`browser_preparation_check.json`记录逐条闭合观察帧加载及缺掩码禁止确认；这不是新轮廓或视觉验收。

| 待标注来源（共同前缀`joint_target_pair_`） | 条数 | 已提取接近RGB |
| --- | --- | --- |
| train_batch03 | 10 | 3215 |
| train_batch04 | 6 | 1998 |
| train_batch04_remaining | 4 | 1410 |
| train_remaining | 12 | 3939 |
| train_t029_s002_t033 | 9 | 2918 |
| train_t029_t033 | 1 | 320 |
| train_t034 | 2 | 685 |
| validation | 7 | 2386 |
| validation_remaining | 5 | 1729 |

- `target_annotation_worklist.json`逐条汇总78条正式数据的目标方向、原始／逻辑场景编号和证据来源；未重新推断已经确认的现场摆放。33个训练名义组中30组近似配对＋6条额外示教，验证6组；纠正后的训练／验证逻辑组无交集，跨目录`t-029`和`t-104`保持整组。
- `target_training_sources.json`包含10个训练来源和2个验证来源，现已全部通过完整数据接口验收及目标条件preflight：训练66条44593帧／21736接近帧，验证12条8234帧／4115接近帧。正式预检输出与数据就绪汇总见`artifacts/joint_target_pair_preflight/data_preflight.json`和`readiness_check.json`。
- `target_training_candidate_stats.json`保留最初仅从训练66条44593帧计算的候选统计；正式preflight已独立重算且一致，验证帧未参与统计。`target_training_plan.json`保留初始plan历史；当前30000步参数、授权状态及启动流程以`target_training_run_candidate.json`和[手册](README.md#同场景ab成对示教方案)为准。
- 本地准备时9批`inference_request.json`／`remote_transfer_plan.json`均未授权、未执行；batch03后来单独完成，剩余8批随后按一次性授权全部完成，执行结果如下。冻结模型／环境／参数及首对共享远端目录保持一致，所有批次文件名前缀独立。
- `annotation_preparation_input_stats.json`记录本轮开始时12处原始目录的文件路径、大小和纳秒mtime；结束比对一致（不是内容哈希证明）。旧1076张／598张中断PNG仍在原处，正式数据清单不含中断尝试。原始session、Parquet、视频、采后审计和历史推理均未改，原复核页在各新增标注目录留有备份；该本地准备阶段未提交、推送、训练、远端推理或连接设备。后续获授权的batch03推理记在上文，剩余8批结果见下文。

剩余8批一次性冻结推理及目标草稿现已完成，统一证据仍在上述清单目录：

- `remaining_inference_execution.json`保存用户“一次性把后面数据都推理完”的授权、8批15385张执行及回传记录，各批仅运行一次，0次优化。`remaining_inference_preflight.json`核对第15轮冻结模型、640×480、目的路径无冲突、GPU与空间；推理前后权重文件大小／mtime一致。没有覆盖首对或前批文件、安装依赖、训练动作模型或连接设备。
- 各批`prediction_check.json`验证完整同帧身份、实例PNG尺寸／类别／面积和空模型加载错误；`target_region_selection.json`记录46条实际目标方向与本条中心线。选择始终是当前预测前景的子集，训练／验证同参数，未依据验证结果调整模型。三条初始颜色区域相连时按当前RGB分别确定两根中心线。
- 15343张来自模型前景筛选，42张缺失帧以当前RGB颜色支持和可见目标区域生成助手补绘，不传播相邻帧掩码。助手查看365个关键／低面积模型草稿帧、全部46张起始映射及42张补绘的原图和结果（集合有重叠）；部分同色盒体／背景污染作局部排除并保存`previous_masks`，不宣称逐像素正确。各批`visual_check.json`保留199个已知或疑似问题，含自动面积诊断；`t-030`的episode 2后段目标偏移可能使固定初始区域裁边，已单独定位。

| 来源后缀 | 已接受接近轮廓 | 助手补绘 | 已接受误差帧 |
| --- | --- | --- | --- |
| train_batch04 | 1998 | 4 | 41 |
| train_batch04_remaining | 1410 | 5 | 13 |
| train_remaining | 3939 | 8 | 36 |
| train_t029_s002_t033 | 2918 | 4 | 25 |
| train_t029_t033 | 320 | 0 | 2 |
| train_t034 | 685 | 0 | 1 |
| validation | 2386 | 0 | 11 |
| validation_remaining | 1729 | 21 | 70 |

- `remaining_annotation_handoff_check.json`与`remaining_browser_checks.json`保留草稿交付时的15385张齐备、reviewed=0历史。用户随后反馈“确认，没有问题。进行下一步”，本次据此登记8批现有全部轮廓及199项已列误差的接受，包含42帧助手补绘，不新增逐像素正确声明。`remaining_review_acceptance.json`及逐批`review_decisions.json`、`review_acceptance_check.json`记录范围和原掩码未变；接受前JSON／页面留有备份，问题转入`accepted_annotation_limitations`，原`visual_check.json`和修订历史不删除。
- `target_training_run_candidate.json`、`target_training_remote_sources.json`和`target_training_transfer_plan.json`保留当前配置、原始来源身份和已部署清单。用户先要求准备完成后再授权，资源核对后明确“可以，我授权可以开始训练”；`target_training_execution.json`登记本次授权、远端目录、启动PID和状态。启动前GPU无计算进程，磁盘可用约68.1 GB；26444个文件／2561471731字节已传输，远端数量和大小一致，无文件删除。既有Python／CUDA与缓存复用，scipy沿用原deps，缓存写入新项目，未安装依赖或修改共享环境。预算和运行路径统一见[手册](README.md#同场景ab成对示教方案)。
- 远端preflight与smoke退出码均为0，日志／状态快照已回传`artifacts/joint_target_pair_training_run/`；smoke报告在`artifacts/joint_target_pair_model_smoke/`，loss=1.854486、梯度范数20.370108，零优化步，严格重载后预测逐元素一致，输出形状[1,50,7]。smoke使用合成batch=1；正式训练另外验证真实batch=8。
- 启动快照`artifacts/joint_target_pair_training_run/startup_check.json`记录2026-10-11T07:00:33Z时训练PID 320785仍在运行，已执行125步，已记录loss／梯度全部有限，进程显存3772 MiB，磁盘可用约64.1 GB。后台runner PID 320782保留最终退出码；此快照只证明当时启动状态，后续完成结果见下段。授权前`artifacts/joint_target_pair_preflight/readiness_check.json`保留原状，其未授权字段不代表后续执行现状。
- `artifacts/joint_target_pair_training_run/completion_check.json`登记最终完成：30000步连续且loss／梯度全部有限，runner退出码0；优化更新11337.18秒，全程2026-10-11T06:59:01Z至10:18:39Z（约3小时20分钟）。远端`outputs/joint_target_pair_smolvla/final/`已生成约1.20 GB模型权重，最终严格重载后预测逐元素一致，形状[8,50,7]。GPU计算进程已退出；本地仅回传报告、训练日志和预测数组，未下载权重。
- 完成报告与已有预测已回传`artifacts/joint_target_pair_training_run/results/`，重新计算基础指标与报告一致：独立验证12条8234帧，validation loss=0.043185，J1～J6 MAE=[0.2904,0.1260,0.1399,0.1221,0.0088,0.3659]度。夹爪准确率8191/8234（99.4778%）低于保持当前状态基线8210/8234（99.7085%）；专家命令切换帧闭合0/12、释放1/12命中，保持帧有20次错误切换预测。此定义按夹爪值50阈值评价保存专家观察下的第一步动作，不代表事件窗口召回或闭环结果；后续事件窗口／固定种子对照见下段；这些基础指标本身不能宣称目标服从或物理成功。
- 离线诊断统一在`artifacts/joint_target_pair_diagnostics/`：`event_windows.json`确认24次事件的下一帧均已由专家动作改变输入状态，不能据下一帧预测命中推断自主切换。`summary/report.json`记录96个保存观察、648次冻结推理、种子42/43/44：命令帧闭合／释放第一步分别命中1/36、0/36；相应完整50步动作块均包含切换，首次位置中位数第8／4步。只改变输入夹爪状态后，相同图像上的第一步均改为该状态（各36/36），支持当前状态依赖，但不是闭环模拟或固定步数补偿依据。
- 同帧目标互换覆盖12条验证示教各第0／100帧，共24个观察；替代掩码仅取该帧冻结分割前景与另一条既有选择区域的交集，24张原／替代／原图面板已由助手检查，未登记为用户接受标注。相同种子重复最大差0；目标互换的全动作块关节平均绝对变化0.317795°，不同种子同输入差异0.315496°，12/24观察的互换变化超过种子差异。证明数值输入敏感性，不证明方向正确或物理目标服从；`summary/review.html`集中展示记录和输入图。
- 最终模型身份SHA-256为`93fd4a6835a98e4d7a0b04bc6aafd51b7d07508e57dccd19e698472968ca6e0a`，用于新影子入口的加载身份检查。`transport_check/check.json`确认新目标契约握手、无损双RGB、50×7动作块和相同种子重复一致，两次往返约0.266／0.262秒，模型部分约0.100／0.099秒；仅用保存观察，未连接设备。推理服务脚本部署于远端独立`artifacts/joint_target_shadow_service/`，没有修改权重或训练来源。
- `run_joint_target_shadow.py`、`joint_target_shadow_review.html`准备静止只读采集、人工A/B可见区域复核与保存观察预测；`InferencePipe`仅增加显式目标协议，旧默认保持。针对诊断、目标契约、影子入口及既有trial/smooth共165项测试通过；无界面Chrome用合成图验证双目标多边形、确认勾选及绑定snapshot ID的导出。`preparation_check.json`明确`shadow_software_prepared=true`、`onsite_capture_executed=false`、`full_grasp_ready=false`；动态掩码／目标身份、实际闭合驱动的阶段切换和动作块执行时序仍需运行接入。没有自动采集、连接机械臂、执行动作、重训或放宽保护。
- 现场准备反馈：首次影子采集在上电检查处退出，用户确认当时未上电；上电后重试到达DO读取，报`(False, False)`，用户确认夹爪肉眼张开。该输出组合不能作为明确松开命令状态，当时未保存有效快照。按用户要求新增`release_joint_gripper.py`，复用既有驱动独立松开并核对读回，不连接相机、不发送机械臂运动；执行说明见[现场影子对照](README.md#当前模型的现场影子对照)。新入口8项、既有trial 99项、夹爪驱动27项假设备测试均通过，默认计划命令及`git diff --check`通过。用户随后手动执行，截图显示DO2=OFF、DO3=ON读回通过（2次读回尝试）；这是控制器输出证据，不是位置传感器反馈。
- 首次有效现场影子快照在`artifacts/joint_target_shadow_retry_20261011_060338/`，snapshot ID为`1435452ce75f4aa8b66c56f9f814420a`：双640×480 RGB、夹爪命令0、两相机时间差约9.3 ms，快照与复核页面匹配。用户接受的`pile-json--/target_shadow_1435452ce75f4aa8b66c56f9f814420a_selection.json`指定A=右条、B=左条；绑定、非空、不重叠及叠图检查通过。`artifacts/joint_target_shadow_retry_20261011_060338_predictions/result.json`保留种子42/43/44的六次50×7预测，`analysis.json`核对模型身份、形状及所有值有限。全50步／6关节平均绝对A/B差异分别为0.319934°、0.331395°、0.295080°，平均0.315470°；同目标不同种子平均差异0.208441°。本次保存观察预测未连接设备、未发送运动或夹爪命令，`execution_eligible=false`；一个场景的数值响应不证明运动方向、轨迹安全或物理目标服从。原`preparation_check.json`仍是现场采集前的历史快照。
- 用户选择停止离线轨迹检查、直接准备监督下真机实验后，新增`run_joint_smooth.py --model target-pair`：加载本轮模型及固定分割best，每个接近输入使用新分割和种子身份关联，模型输出仍通过既有平滑限速控制；仅在夹爪闭合命令实际成功后记录`TargetInputSession.accept_command`，按相机来源时间切换搬运输入并丢弃闭合前的在途预测。默认只显示计划，操作者现场执行一次周期，所有旧模型入口保持。新实时掩码明确`human_reviewed=false`，共享描边像素实现与训练一致；未自动执行机器人。
- 准备证据集中在`artifacts/joint_target_live_preparation/`。首轮保存图像检查`saved_input_check.json`显示A/B几何匹配均可返回，但叠图发现B实例混入右边A的437像素；不能据返回成功认定目标正确。加入另一根人工圈选区域的排除检查后，`guarded_input_check.json`确认A三次通过、往返0.3307／0.3215／0.3341秒、侵入左条0像素，空画面拒绝；B按437像素冲突拒绝。仅使用同一保存图像重复调用，不证明实际动态关联或实时相机时延。首轮目标明确A=右条；B须另行解决自动分割混入问题。
- 对应206项测试通过，覆盖新掩码与训练描边一致、失效不复用旧掩码、跨目标混入拒绝、单次抓放假设备循环、接受闭合后的因果阶段切换以及推理报错／超时停伺服；默认计划与`git diff --check`通过。服务代码部署到GPU独立`artifacts/joint_target_live_service/`，原始候选归档`service.tar.gz`及补充`competitor_guard.tar.gz`保留；修订前运行模块另存`joint_target_runtime_before_competitor_guard.py`，模型与环境未修改。`preparation_check.json`只登记监督试验启动准备通过，未登记物理成功；操作统一见[单次真机试验](README.md#目标条件模型单次真机试验)。
- 首轮A真机记录`artifacts/joint_target_live_A_20261011_063259_359810755/`：接受1次预测、发送4帧伺服指令，无夹爪切换，完成周期0；等待下一次推理时重选旧动作块，观察年龄539.3 ms超过500 ms而退出。第一条预测往返337.6 ms、远端处理155.6 ms，主线程接收时观察年龄378.5 ms；并非单次推理耗时539 ms。`final_state.json`显示`is_steady=true`、`servo_enabled=false`、夹爪命令0；双相机各21帧，录像收尾完整。用户随后授权放宽限制，实际仅增加目标模型显式`--latency-tolerant`：预测年龄500→750 ms、看门狗600→1000 ms、通信等待2→5秒，相机及机械、目标身份限制不变。`timing_review.json`保留诊断，`artifacts/joint_target_live_preparation/latency_tolerant_plan.json`保留新计划；67项针对性测试通过，包括每次340 ms串行预测的单周期、延迟首条预测及挂起后停伺服。原运行日志不改写；之后的现场复测及重叠停止调整见下一条。
- 放宽时间后的A真机记录`artifacts/joint_target_live_A_20261011_063838_533598984/`：接受34次预测、发送304帧，因新掩码与另一根人工圈选区域重叠85像素而停止；夹爪未切换、周期0。最终`is_steady=true`、`servo_enabled=false`，双相机各330帧、录像收尾完整。失败请求的原始掩码未保存，不能据85像素断定为噪声。用户明确要求取消该检查后，GPU生成端与本地接收端均去掉重叠拒绝，只记录`other_reviewed_target_overlap_px`，完整新掩码不裁剪；目标关联失效／歧义及机械、时间保护保持。176项针对性测试通过；GPU保存图像复查此前B的437像素冲突，现在成功返回有限50×7动作块，往返0.3354秒，证明取消已贯通，不证明分割已修好或真机成功。证据、部署脚本及本地修改前归档集中于`artifacts/joint_target_live_preparation/overlap_review/`，检查为`deployment_check.json`；GPU旧模块保存在独立服务根目录`joint_target_runtime_before_overlap_stop_removal.py`，部署时核对远端旧文件与本地备份一致。未连接机器人或发送动作，取消后的现场复测由操作者执行。
- 重叠停止取消后的两次现场记录：`artifacts/joint_target_live_A_20261011_065049_184074657/`未带时间容限参数，发送3帧后因536.1 ms超过500 ms退出；`artifacts/joint_target_live_A_20261011_065620_479632672/`已带参数，发送8帧后在重选旧动作块时因752.7 ms超过750 ms退出。后者首条预测往返371.3 ms、远端处理162.5 ms、接收观察年龄428.6 ms，后续等待使旧观察继续变老；不能把752.7 ms记作模型耗时。两次夹爪均未切换，最终静止、伺服关闭。按用户再次授权，仅将`target-pair --latency-tolerant`的年龄上限750→1500 ms、无新预测看门狗1000→1500 ms；通信等待仍5秒，其他模型、默认模式、相机与机械限制不变。修改前代码和文档归档`artifacts/joint_target_live_preparation/before_extended_prediction_budget.tar.gz`；计划与记录为同目录`extended_prediction_budget_plan.json`和`extended_prediction_budget_check.json`。60项smooth测试通过，包含每次550 ms串行预测时越过旧750 ms门限继续完成假设备周期、超过新期限停伺服、年龄边界及旧模型不变；未执行现场复测，无GPU部署或权重变更。
- 1.5秒容限后的A现场记录`artifacts/joint_target_live_A_20261011_070501_791659098/`：接受34次预测、发送301帧，关联第35帧失败。候选1位置差6.53像素、方向差0.81°，但面积为上一帧12.38%／种子12.5%，触发25%／20%面积下限；候选2位置差45.38像素、方向差75°，仍不符合目标。未夹紧，最终静止、伺服关闭。按用户授权，仅在`FreshTargetInputs`的实时关联实例把相邻帧面积比改为0.05～20、种子面积比下限改为0.05，其余关联条件及离线默认值保持；日志新增实际`association_parameters`。84项测试通过，GPU部署核对源码一致，远端合成掩码2000→100→2000和2000→250→2000像素均通过且返回当前像素；保存A图像完整推理返回50×7有限动作，往返0.3279秒。按原报错候选数值核算，候选1通过新门限、候选2仍拒绝；失败帧完整掩码未留存，因此不是原始失败输入精确复现。证据和修改前归档位于`artifacts/joint_target_live_preparation/partial_visibility/`，GPU备份为独立服务根目录`joint_target_runtime_before_partial_visibility.py`。没有连接机器人或执行现场复测，不将面积容限放宽记为分割修复或抓取成功。
- 面积放宽后的现场记录`artifacts/joint_target_live_A_20261011_071053_760394083/`：34次预测、303帧，唯一候选位置差29.49像素、方向差60.73°、面积比1.455、重合度0.64585、分数0.45320，因45°方向门限停止；无夹爪切换，最终静止且伺服关闭。用户明确要求只留清单第7／8／9项后，实时入口使用`enforce_geometry_limits=false`，删除中心距离、方向和两组面积上下限的六项硬性拦截；评分算法、重合度≥0.05、分数≥0.35、歧义分差0.15保持。位置40像素／方向45°仅保留为评分尺度，面积比只记录；离线默认继续执行原几何条件。101项测试通过，GPU远端验证远距离／90°转向／6倍面积实例在实时入口通过、离线仍拒绝；保存A图像推理返回有限50×7动作、往返0.3275秒，并记录硬性几何门限关闭。原报错候选数值通过剩余条件，但原始失败掩码未保存，不是失败输入精确复现。备份、部署与检查位于`artifacts/joint_target_live_preparation/association_score_only/`；GPU旧两个模块备份至服务根目录`before_score_only/`，部署前逐文件比对旧版本、部署后确认与本地一致。未连接机器人或执行现场复测，其他运动和时效条件保持。
- 仅保留第7／8／9项后的现场记录`artifacts/joint_target_live_A_20261011_071843_572750392/`：35次预测、314帧；候选1重合度0.046447低于0.05、分数0.356564通过，候选2重合度0.500667通过、分数0.3004低于0.35，故无合格候选退出。无夹爪切换，最终静止、伺服关闭。用户明确要求取消剩余三项后，实时入口关闭`enforce_match_thresholds`，有候选时直接选择综合分数最高者，分数并列按原顺序；既有评分公式保持，默认离线筛选仍保留。成功预测记录`association_match_thresholds_enforced=false`和选择模式，参数记录移除不再生效的三项门限。101项测试通过；GPU合成输入确认低重合度／低分／歧义三类实时通过、离线拒绝，保存A图像推理返回有限50×7动作、往返0.3270秒。原报错数字按新规则选择候选1，因完整失败掩码未留存，不能视为原输入精确复现。备份和部署验证在`artifacts/joint_target_live_preparation/best_candidate_selection/`，GPU备份目录为独立服务根目录`before_best_candidate/`。完全无候选、错误数据及通信／机械控制异常仍停止；未连接机器人或执行现场复测，选取正确性和抓取结果待现场判断。
- 新8批`dataset_check.json`完整检查46条31456帧（15385接近／16071搬运）：实际LeRobotDataset/PyAV读入的RGB与标注原图一致，训练／运行图像及闭合因果阶段一致，原始7D state/action和50步尾部padding不变。加上此前4批，全部78条52827帧接口通过。逐批报告中`independent_validation_available=false`是验收函数的单来源字段，不能据此否定当前全量独立验证；后者由正式preflight确认。
- `remaining_browser_acceptance_checks.json`和各批`browser_acceptance_check.json`核对8个当前页面的接受状态、图像加载、46条闭合观察帧、199项已接受误差定位及8批导航，未提交浏览器测试决定。
- 全量`readiness_check.json`确认训练／验证原始编号和逻辑组均无交集，25851张已接受轮廓完整，统计仅来自训练44593帧，全部3822个尾部padding窗口保持原规则；再次比对1783个原始文件路径／大小／mtime一致，15385张本次接受掩码与推理交付清单的大小／mtime一致。数据就绪不代表模型已经训练或物理成功。
- `remaining_work_integrity_check.json`核对12处原始目录1783个文件清单、大小和mtime未变（非内容哈希证明）；全部中断PNG保留并排除于候选传输。训练66条44593帧、验证12条8234帧以及逻辑组隔离不变。未提交、推送或执行机器人动作。

### 目标条件模型最新A/B现场结果与B失败复核

本节标题中的“最新”指首组实机场景（A=右、B=左），后续重新摆放的身份及结果见[重新摆放后的指定A/B抓取](#重新摆放后的指定ab抓取)。

**现场结果：A=右条成功；B=左条抓取点不对、没有夹住。** A来自用户交接中的“机械臂成功抓取了A并完成放置”；B先反馈“这次抓B时没有成功”，本轮进一步明确“没有夹住，且抓取点不对”。下表程序结果另行核对，不能将正常退出或DO读回当作物理成功。原始报告保持不改。

| 项目 | A | B |
| --- | --- | --- |
| 原始目录 | `artifacts/joint_target_live_A_20261011_072403_006168414/` | `artifacts/joint_target_live_B_20261011_072712_551780591/` |
| 程序结束 | `model_suction_cycle_completed`，`failure=null` | 同左 |
| 发送帧／接受预测／命令周期 | 1534／185／1 | 1797／220／1 |
| 双相机各编码帧数 | 1553 | 1817 |
| 接受闭合／释放，相对录像起点 | 17.831／61.884秒 | 20.693／72.441秒 |
| 原始物理结果字段 | `physical_grasp_success=null` | 同左 |
| 最终状态 | 夹爪命令0、`is_steady=true`、`servo_enabled=false` | 同左 |

两轮模型身份和主要控制参数相同，均使用最高分候选选择、几何及匹配拒绝门限关闭、1.5秒时间容限。原始录像报告均完整且无错误；本轮完整解码四段录像，均为640×480，帧数与上表一致。起始图看起来摆放相近，但没有确认人工恢复程度，不记为严格相同观察的A/B对照。

固定[录像与掩码复核页](../../artifacts/joint_target_live_preparation/ab_outcome_review/review.html)、
[逐预测统计](../../artifacts/joint_target_live_preparation/ab_outcome_review/review.json)集中在`artifacts/joint_target_live_preparation/ab_outcome_review/`。
分析读取A的49次、B的59次已接受接近预测，并分别保留1次闭合前在途丢弃预测。按`trace.sensor_times[0]`对齐全局相机sidecar的`source_timestamp`，最大对齐误差分别32.92／25.45 ms；关键B预测35恰有同来源录像帧。选中掩码来自日志中的原PNG，叠图RGB来自有损录像；图中绿色细线用于检查，真实策略渲染为紫色描边。所有接近预测均生成逐帧拼图，本轮逐图查看B全部60张及A关键时段；这不是逐帧人工身份真值。

| B时刻／预测index（从0开始） | 保存证据与判断 |
| --- | --- |
| 起始／0 | 选中掩码主要覆盖左条B，但侵入右条A人工种子区域312像素；不是本轮没有复现的旧437像素记录。 |
| 4.303秒／12 | 一个选中实例同时覆盖两根主要可见区域，右条种子重叠1581像素；后续曾恢复以B为主，混入仍间歇存在。 |
| 12.223秒／34 | 仍以B为主，与A／B固定种子区域分别重叠251／1612像素。 |
| **12.580秒／35** | 唯一候选转为以右条A为主、附带少量B；与A／B种子重叠1660／244像素，中心相对上一帧由(433.0,345.6)移到(507.1,339.8)。候选综合分0.07854、日志预测中心距离69.45 px、方向差71.33°；按现行最高分规则选中。12.910秒收到预测，12.913秒开始实际使用该动作块。 |
| 12.900秒起／36～57 | 多次选中右条A或右条为主的混合区域，后续分数可超过0.9；高分只说明与已漂移参考匹配。双相机接近片段显示左条被推移，夹指与竹条的相对位置偏离正常包夹。 |
| 20.075秒／58 | 最后用于闭合的预测再次包含两条主要区域。接受闭合发生于20.693秒，采用动作块第5行（从0开始）的闭合边界，选择时观察年龄343.50 ms；闭合前最后TCP为(0.544825,-0.348209,0.088530) m。预测59是闭合前在途结果，被丢弃，没有用于后续执行。 |
| 闭合后与40～72秒 | 录像可见夹爪空着离开、两根仍留在抓取区域，与用户“没有夹住、抓取点不对”一致；没有支持“成功夹住后搬运滑落”或“抓走A”的证据。 |

**A对照也存在输入错误。** 15.590秒预测43仍是右条局部掩码；15.906秒预测44只剩一个以左条为主的混合实例，分数0.01388、侵入B种子2093像素，随后至闭合前继续输出混合区域。该预测16.243秒开始用于指令，约1.589秒后闭合；B的上述跳变预测开始用于指令后约7.780秒才闭合。这是时序差异，不证明“延迟更长”就是单独原因。A录像可见右条随后被搬走放置，与用户成功反馈一致；不能由此反推最后几次掩码正确。

**机制与结论。** `TargetAssociation.step`每次选中候选后就更新下一次使用的掩码、中心、方向和速度参考；实时入口不再拒绝低分候选。当B受遮挡而唯一候选以A为主时，选取结果仍带固定字符串`target_id=B`，参考却已转到A，后续高分可能强化错误身份。当前只跟踪一个所选对象，另一根人工区域仅统计重叠；`mask_id`也是逐帧实例编号，不是跨帧物体身份。已确认的是**送入模型的目标像素发生漂移，而且最终抓取点错误、空夹**；尚不能仅凭两轮录像断言分割是唯一原因，或排除策略自身定位／姿态／闭合预测误差。原候选排名数值完整，但未保存其他候选的完整掩码，不能凭日志声称当时一直有干净B候选可直接改选。

固定种子区域在物体移动后不能作动态真值，不宜直接硬裁；不能把这一分析变成静默恢复停止门限、沿用旧掩码或加入层高补偿。首次失败复核只读取保存证据，未修改控制／关联代码、未推理或训练、未连接硬件。原始两轮192个文件的大小及mtime在该次派生产物生成前后一致（非内容哈希证明）；文档修改前副本保存在固定复核目录，已有未提交成果保留。随后经用户“可以，进行下一步”授权完成下列离线修正和对照。

#### 双目标前景分离与保存观察动作对照

实现为`src/lerobot/bamboo_sorting/joint_target_separation.py`的`PairedStripSeparation`，先完成以下离线实验，随后按用户要求通过显式开关接入一次B真机试验准备；`TargetAssociation`及默认最高分路径保持。原型用人工种子初始化两条有限长度中心线，把全部当前分割前景按至中心线的归一化距离分配给A／B；分别用Huber稳健直线拟合更新参考，中心只沿新法向调整，减少局部遮挡造成的轴向中心漂移。参考几何可保留，但输出严格来自本帧前景，空归属输出空掩码；不存在旧像素传播、模型漏检区域补画或恢复实时拒绝门限。此几何假设尚未验证交叉、快速位移、三根以上或一般堆叠。

全部产物放在原复核目录下`identity_separation/`，统一从[原复核页](../../artifacts/joint_target_live_preparation/ab_outcome_review/review.html#identity-separation)进入；没有另建review入口。`manifest.jsonl`记录108张图的源时戳、状态、JPEG／MP4来源与候选信息，`segmentation/`保存冻结分割模型重放产生的**全部**当前实例，`baseline/`与`separated/`保存同一实例结果上的旧关联和新分离掩码；`separation.json`记录独立身份参考。使用现有GPU模型和Python，SSH流式处理，没有部署新服务或修改远端模型、代码、环境。108张为A 49张、B 59张已接受接近观察；不是108个独立场景。

助手逐图查看108张双目标叠图，未见整根A／B身份互换，局部碎片、边缘误差及夹爪区域混入仍存在。B预测30／35重放后分别只有166／242像素归属于B，预测40为518像素；新方案没有为了轮廓完整而填补缺失。种子区域重叠为零不是独立正确率，尤其不能用旧种子评价已被推移的竹条。JPEG重放改变了部分分割结果：例如B预测30的旧关联重放已经选中A为主，原日志该帧仍以B为主，因此必须保留“原日志掩码”和“同次重放旧关联”两种对照，不能称为精确复现。

冻结目标SmolVLA对照为7张观察×4种描边×种子42／43／44，共84次有限50×7输出；每组RGB、腕部图和关节状态完全相同，只改变全局图描边。四种输入为原日志选中掩码、同次重放旧关联、新双目标分离、助手按当前RGB绘制的可见区域诊断草稿。草稿均`human_reviewed=false`，不进入训练或实机；B预测50／55只画明确可见的下部，略去夹指遮挡附近小段。原始响应见`policy_comparison/responses.jsonl`，草稿及输入图在同目录，汇总见[comparison_summary.json](../../artifacts/joint_target_live_preparation/ab_outcome_review/identity_separation/comparison_summary.json)。

下表为各输入产生的动作与**诊断参考描边产生的模型动作**之间的平均绝对关节差：每张50步×6关节，3个同种子对照取平均。单位是角度，不是TCP毫米误差或成功率。

| 观察／相机时间 | 同次重放旧关联→参考 | 新分离→参考 |
| --- | ---: | ---: |
| B起始／预测0 | 0.2307° | 0.0471° |
| B 10.797秒／30 | 1.5382° | 0.4994° |
| B 12.580秒／35 | 0.7891° | 0.1637° |
| B 14.207秒／40 | 0.4405° | 0.0899° |
| B 17.459秒／50 | 0.4397° | 0.1158° |
| B 19.087秒／55 | 0.1568° | 0.1228° |
| A 16.226秒／45 | 0.0526° | 0.0835° |

6张B观察的18组同种子结果均更接近参考，前10步的均值也一致改善；A的单张对照则未改善。B预测35的新分离相对旧关联，50步／3种子平均J1变化−1.725°、J6变化−1.167°；没有经标定的离线运动学／图像映射，不能把这些数值解释成正确的末端纠偏方向。闭合预测也会改变：B预测50、种子42，旧关联最早闭合行从索引3变为新分离的11，参考也是11；预测55参考为19，新分离仍为11。这些是动作块内索引，不是机械臂实际闭合时刻，也不是固定延迟建议。

结论：当前身份混淆能够实质改变动作预测，双目标分离是有依据的离线候选；**不能宣称B抓取点已修复**，因为参考动作本身没有物理正确性真值，重放不模拟纠偏后新的相机与机器人状态，严重漏分仍在，且本批用于开发而非独立验证。后续优先项统一见[项目手册](README.md#目标条件模型单次真机试验)。

离线阶段验证：6项新合成测试加15项既有关联测试共21项通过，覆盖混合实例分离、实例ID变化、单目标缺失不冒认邻条、局部遮挡、小幅运动、空帧及异常输入。`evaluate_separation.py`从冻结掩码重新运行108帧，324张派生掩码逐像素一致；原型输出是当前前景的不重叠划分，没有增加像素。84次模型推理均已完成、无优化步骤，结束后确认GPU计算进程退出。该阶段记录为`identity_separation/verification.json`，当时尚未修改实时入口。

实机准备：用户随后要求再试抓B，`--paired-target-separation`已接入`run_joint_smooth.py → prepare_live_setup → InferencePipe → FreshTargetInputs`，GPU独立目录为原实时服务下的`paired_foreground/`。客户端校验返回模式和分离状态，防止请求新方法却静默使用旧关联。默认入口和原服务保持；新路径保存原实例图、A／B掩码和几何参考。217项相关测试通过，覆盖新种子契约、混合前景分离、所选目标为空时停止、搬运输入、旧服务响应拒绝及原控制调用链。备份、计划、部署记录和复现脚本集中在`artifacts/joint_target_live_preparation/paired_foreground_trial/`。部署时源码中的失效文档软链接造成复制中断；核对未完成副本与原文件一致后保留链接完成复制，没有修改原服务。

同一独立GPU服务以真实冻结模型重放上次B的59张保存观察，再模拟1次搬运阶段协议请求，60次均返回。所有A／B掩码与本地算法逐像素一致，彼此不重叠且不增加当前分割前景之外的像素；B预测30／35／40仍仅166／242／518像素，漏分没有被填补。接近阶段往返延迟中位数0.296秒、P95 0.323秒、最大0.335秒，最大返回记录27773字节。记录见[verification.json](../../artifacts/joint_target_live_preparation/paired_foreground_trial/verification.json)及同目录`saved_B_responses.jsonl`。推理结束后GPU计算进程为空；此验证未连接机器人或相机，模拟搬运请求不代表真实夹爪闭合。随后实机结果见下节，现场命令统一见[项目手册](README.md#目标条件模型单次真机试验)。

#### B夹住后搬运停滞

最新目录`artifacts/joint_target_live_B_paired_20261011_081236_676123145/`，用户反馈“抓住了，但一直没有执行后续的搬运动作”。记录为夹持成功的现场反馈及搬运未完成，不能计作整次抓放成功；原始`physical_grasp_success=null`保持不改。实际发送2976帧，接受369次预测；录像第14.948秒闭合，后续329次预测均已切换到`transport`，使用闭合之后的全局帧、夹爪状态100及`raw_rgb_after_accepted_close`无描边输入。一条闭合前在途预测被丢弃，阶段切换与持续推理都没有中断。120秒因`time_limit`退出，`failure=null`，释放次数0；最终伺服关闭、机械臂静止、夹爪命令仍100。

所有搬运预测实际使用的目标都等于该动作块第0行，`action_selection=null`。这是该次试验采用、目前默认保留的“张开时按观察年龄取步，夹紧后只取首步”执行方式。329张观察中，第0行相对观察状态的六关节平均绝对变化中位数0.0808°，第49行为2.7005°；不能把更大关节变化直接解释为正确抬升。闭合后约105秒，TCP高度范围0.08318～0.09032米，末值约0.08536米，未形成搬运抬升；录像关键帧也显示竹条仍处于夹指间、靠近原支撑位置。所有329个完整动作块均没有释放行，故不存在执行器漏掉已预测释放的问题。

只读对照发现前一目录`joint_target_live_B_paired_20261011_081018_072323519`同配置在闭合后约37.7秒时已经抬升至TCP高度0.1443米，随后由操作者中止；其录像编码收尾报BrokenPipe，不作为完整录像。此前成功A也用搬运第0行，但首行关节变化更大并完成释放。因此证据支持“反复首步重规划与早期预测接近原位共同造成推进缓慢”的候选解释，尚不能断言单一代码原因或模型没有搬运能力。

保存状态上的取步对照：按每次命令时刻的真实观察年龄选择完整七维行，并停在最早预测释放边界，本轮B索引7～21、中位数12；相对当时真实关节的平均绝对目标差中位数从0.0830°变为0.1654°，仍然很小。该对照没有模拟改变动作后的新状态／图像，也没有执行运动，**不能证明时间对齐单独就能解决停滞**。下一候选是验证搬运阶段动作块的时间推进与模型早期预测，保留现有新鲜度、速度、工作区、到达条件及释放边界；不直接跳到第49行、延长超时或脚本强制抬升。

固定复核入口仍为[review.html](../../artifacts/joint_target_live_preparation/ab_outcome_review/review.html#transport-stall)，结构化结果为[transport_stall/review.json](../../artifacts/joint_target_live_preparation/ab_outcome_review/transport_stall/review.json)，同目录含双相机关键帧、三轮逐命令候选取步记录、复现脚本和修改前文档副本。该诊断阶段仅分析已有日志／录像并更新复核材料，没有改运行代码、启动GPU推理、连接硬件、训练或覆盖原始结果。

后续实现：按用户“进行下一步”，增加本地`--transport-age-aligned`显式试验开关，仅适用于`target-pair`。`select_transport_action`按最早观察时间到当前时刻的年龄、25 Hz动作频率选择完整七维行，并将索引封顶在最早预测释放行。只有已闭合且响应明确为`transport`才启用；每个控制周期可继续取步，到达释放行后保持目标并沿用原到达／低速条件才发IO。在途新预测仍可在IO之前重规划，IO之后仍丢弃旧预测并清空动作块。默认行为、模型、GPU服务、120秒上限及运动限制保持。

实现验证：199项相关测试通过，包括新开关计划与默认参数对照、六关节和夹爪取自同一行、不能越过首次释放行执行后续异常目标、观察过期拒绝、假设备完整抓放，以及搬运新预测挂起后停止伺服并保持夹紧。新函数在既有三轮记录的4626条搬运命令上与先前独立诊断逐条一致（本次B 2606条、中止B 932条、旧A 1088条），无释放边界跳过。机器记录、测试输出、候选计划、复现脚本及修改前文件备份位于[age_aligned_trial](../../artifacts/joint_target_live_preparation/ab_outcome_review/transport_stall/age_aligned_trial/verification.json)。上述检查仅验证软件选择规则和假设备调用链；保存状态重放不是改变动作后的闭环仿真。随后现场结果见下段。本轮未连接硬件、启动GPU推理、训练或改动原始实验记录。候选现场命令只维护在[项目手册](README.md#目标条件模型单次真机试验)中。

后续现场：两个新增目录`joint_target_live_B_transport_20261011_082742_854334863`与`joint_target_live_B_transport_20261011_082933_760394285`均启用双目标分离和搬运时间对齐，分别完成720／768帧控制、83／88次预测及1次夹紧→释放命令周期，无程序故障。最新一轮用户确认“成功抓到了并完成了搬运”，录像抽查支持左B搬到收集区；闭合／释放在录像16.197／31.180秒，工具最高0.1800米。前一轮仅确认命令周期完成，20秒抽查仍见两条在原区域，不计作第二次物理成功。原始完成报告中的物理结果字段保持不改。

用户进一步反馈“小幅换位置后夹取就偏，位置差不多”。两轮起始及最后接近预测的抽查描边均位于左B，未见整根换成A；这不代替全程掩码审查。闭合前最近控制记录的TCP参考点间距约0.23毫米，但腕部关节角有变化；参考点不是已标定的夹指接触点，不能仅据此断言没有跟随位置变化。两轮闭合前最大关节目标误差约0.122°／0.126°，均满足既有0.15°条件，但该条件不证明竹条对中。当前候选问题是从小幅目标移动到夹指对中的修正不足，感知漏分、接触推移和动作预测仍需分开检查；不能由一次成功宣称位置泛化或修正的独立因果效果。相同控制和模型下，重新确认当前目标、单变量小位移并重复，才能区分旧种子偏差和策略适应性。记录及关键帧见[onsite_outcomes.json](../../artifacts/joint_target_live_preparation/ab_outcome_review/transport_stall/age_aligned_trial/onsite_outcomes.json)和同目录`onsite_position_comparison.jpg`。本次只读核对并更新文档，没有改算法、训练或启动硬件。

#### 重新摆放后的指定A/B抓取

快照`artifacts/joint_target_reposition_20261011_083944_072360764`与人工标注`pile-json--/target_shadow_ae44bab73e02455197e55c1fc994a4bd_selection.json`匹配；本组A为左边较直竹条，B为右边较斜竹条，不能沿用上组左右身份。用户确认A成功、B抓歪失败，并明确两次之间“两根都大致恢复了原摆放”，登记`approximate_pair_candidate`，不作为严格同一观察或独立成功率估计。

| 本轮目标及目录后缀 | 预测／控制帧 | 闭合／释放时间（录像秒） | 现场结果 |
| --- | --- | --- | --- |
| A：`joint_target_reposition_A_20261011_084330_924664106` | 96／825 | 17.316／33.508 | 用户确认成功 |
| B：`joint_target_reposition_B_20261011_084525_275963007` | 119／1016 | 18.151／41.115 | 用户确认抓歪未成功 |
| B后续：`joint_target_reposition_B_20261011_085705_324199882` | 93／802 | 17.431／32.573 | 用户确认抓到B、摆放稍靠左；录像支持搬至收集区并松爪 |

最初A／B两轮均使用新种子、双目标分离与搬运时间对齐，`failure=null`、录像完整，并完成一次夹紧→释放命令周期；原始`physical_grasp_success=null`保持。失败B接近49次、搬运70次预测，搬运输入均为闭合后当前无描边图；未出现上一轮搬运停滞故障。对失败B全部49张接近预测掩码逐张检查，未见整根身份切到A；末段手指遮挡附近仍有漏分，最后一次可见掩码1538像素，不作为精确像素真值。

闭合前腕部画面中，B相对夹指区域仍明显倾斜，闭合后没有被稳定带离支撑，后续夹爪空手离开。首次接触时刻、是否提前推移，以及横向位置和角度误差各自的贡献尚未定量确定，不能仅凭投影角度直接计算J6补偿。两轮闭合前模型目标与实际关节的最大误差分别0.118°／0.124°，满足原0.15°到达条件；这仅证明关节到达，不能证明抓取姿态正确。最终闭合采用预测A46／B48的第4／5行，接受闭合时对应全局帧年龄约0.644／0.736秒；延迟可能影响接触阶段，但两轮都存在该延迟，尚无其独立致因证据。

后续B成功轮与失败轮的`plan.json`逐项一致，同模型、种子和控制配置；93次预测、802帧控制、一次命令周期，`failure=null`、录像完整，原始物理成功字段保持不改。用户反馈原文为“这次成功抓到了B但是摆放位置稍微靠左了一点”。抽查录像18／21／25／29／32.8秒支持B被夹持、带离支撑、搬到收集区并松爪。两轮起始／末段各3张目标描边均在右B，最新轮尚非全程掩码审查。

初始当前预测掩码的中心从(502.87,332.98)变为(488.04,332.58)像素，即画面左移约14.83像素；主轴与图像竖直方向夹角绝对值由28.61°降至24.65°，也略微转直。**位置和角度同时变化，不能将成功单独归因于左移。** 上述为预测掩码的图像几何估计，不是人工量取的实物位移／角度，不能直接换算毫米。闭合前最近控制记录的工具参考点变化为(-16.56,-17.35,+1.60)毫米，J6实际角变化-7.30°；这些是机器人姿态变化，不是竹条位移或应施加的补偿量。最新最大关节到达误差0.127°，失败轮0.124°；两者都到达各自模型目标，成功轮腕部画面中竹条更靠近夹指中间，仍不能据此量化接触误差。证据支持模型／执行输出有随观察变化，不支持“始终抓同一点”；相近位置与朝向下的稳健性仍待验证。

期间另有`joint_target_reposition_B_20261011_085411_363725530`及`joint_target_reposition_B_20261011_085538_380100803`，均为`operator_interrupt`、零个完整命令周期、录像未完整收尾，现场抓取结果未确认；两轮都已发出闭合命令，085411起始几何与失败B相同，085538已改变摆放。保留为结果未知，不能忽略后将挑出的成功／失败两轮当作总体成功率。

**斜条闭合时J6系统性欠转（只读对照）。** 用73条示教（78条中排除起始主轴与竖直夹角≥60°的5条）拟合“起始目标掩码角度与中心→示教闭合J6”的线性经验关系（下称专家趋势），RMSE 3.78°；改用J1−J6拟合为3.67°，结论一致。残差定义为实机首个闭合命令时实测J6减去同几何的专家趋势：

| 实机轮次 | 起始图像倾角 | J6残差 | 结果 |
| --- | ---: | ---: | --- |
| 旧场景A／B共5轮已闭合（`072403`、`081018`、`081236`、`082742`、`082933`） | −15°～+16° | −6.0°～+3.3° | 混合 |
| `joint_target_reposition_A_..._084330` | −10.8° | −5.3° | 成功 |
| 旧`joint_target_live_B_..._072712` | −22.9° | −9.7° | 失败（另有身份漂移） |
| `joint_target_reposition_B_..._084525` | −28.6° | −13.1° | 抓歪失败 |
| `joint_target_reposition_B_..._085411` | −28.6° | −15.9° | 中止，结果未知 |
| `joint_target_reposition_B_..._085538` | −22.7° | −15.9° | 中止，结果未知 |
| `joint_target_reposition_B_..._085705` | −24.7° | −14.4° | 成功 |

倾角绝对值约15°以内时残差在±6°内；倾角约23°～29°时一致偏负10°～16°，同一失败摆放的两轮可复现。示教中有几乎相同起始几何的`t-034 / s-002`（中心(501,337)、−28.1°），闭合J6为−120.3°，失败B模型为−133.6°；因此B不是完全未见场景，但−30°～−15°区间仅6／73条示教，覆盖偏稀。既有离线诊断（专家观察的教师强制输入，12条验证示教第0／100帧、3种子）第49行J6与示教差的均值绝对值不超过4.7°，未见同类偏差，说明偏差在闭环执行中出现；成因是闭环累积偏移、实时分离掩码与训练掩码差异还是其他因素，尚未区分。

欠转同时存在于B失败轮和成功轮，不是成败的唯一分界；更合理的解释是它压缩了横向对中的容差，成功轮竹条更居中。图像角度是斜视投影、不是标定的实物角度，J6也不等于世界偏航；专家趋势是经验关系，不是正确抓取真值，不能据此设置固定J6偏置（残差随倾角变化）。结果为[j6_residual_review.json](../../artifacts/joint_target_live_preparation/ab_outcome_review/reposition_review/j6_residual_review.json)，同目录`j6_residual_review.py`可复现（只读、拒绝覆盖）；本次未加载模型、未连接硬件、未改原始记录，文档修改前副本为同目录`*.before_j6_residual`。

（历史计划，2026-10-15取消：A／B实验结束，遗留J6问题改用[腕部角度离线诊断](README.md#斜条j6欠转的腕部角度诊断)）当时的优先项是固定模型和控制参数，以中心固定的角度序列和角度固定的横向位移序列分离两者影响，逐次记录J6残差、腕部画面中竹条相对夹指的偏移／夹角、闭合前推移及物理结果；方案见[项目手册](README.md#目标条件模型单次真机试验)。不由单次成败直接恢复关联门限、固定旋转腕部、强制延迟闭合或启动重训。固定入口仍为[review.html](../../artifacts/joint_target_live_preparation/ab_outcome_review/review.html#reposition-b-shift)，最初A／B记录与全部失败B掩码见[review.json](../../artifacts/joint_target_live_preparation/ab_outcome_review/reposition_review/review.json)，新增对照、来源帧时间和可复现脚本集中于[B_left_shift_review.json](../../artifacts/joint_target_live_preparation/ab_outcome_review/reposition_review/B_left_shift_review.json)及同目录`compare_B_shift.py`。该对照仅分析已有日志与录像、更新文档，未改算法或启动硬件／GPU推理／训练。

### 旧录像目标输入诊断

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

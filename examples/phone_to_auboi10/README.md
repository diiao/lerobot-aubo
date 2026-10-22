# AUBO i10 项目手册

更新：2026-10-22。本文集中维护当前状态、操作方法和后续计划；实验指标与原始路径只维护在
[证据索引](EVIDENCE.md)。所有命令从仓库根目录 `/home/rentao/program/lerobot-aubo` 执行。

## 目录

- [当前状态与系统边界](#当前状态与系统边界)
- [三根堆叠目标示教方案](#三根堆叠目标示教方案)：数据与训练已完成，当前为重复真机实验
- [代码导航与维护](#代码导航与维护)：按任务找入口，区分共享模块与历史工具
- [数据集目录](#数据集目录)
- [人工标注导出目录](#人工标注导出目录)
- [实验产物目录](#实验产物目录)
- [相机配置](#相机配置)
- 静态分割：[采集与人工复核](#静态分割采集与人工复核)、[数据、训练与独立评价](#静态分割数据训练与独立评价)
- 动作模型：[手机遥操采集](#动作模型手机遥操采集)、[训练与离线评价](#动作模型训练与离线评价)、[现场运行](#动作模型现场运行)
- [目标条件接口约定](#目标条件接口约定)：标注格式、训练／运行共用规则
- [斜条J6欠转的腕部角度诊断](#斜条j6欠转的腕部角度诊断)：只待现有人工标注收束，不阻塞堆叠试采
- [后续研究与待办](#后续研究与待办)：含[推进顺序](#当前位置与下一步)与[扩展到6～8根](#扩展到68根)
- [历史流程入口](#历史流程入口)：已结束的A／B示教、单根试采与跟踪诊断

先按当前任务进入对应章节。完成记录、历史参数和逐批结果统一查 [EVIDENCE.md](EVIDENCE.md)，不把旧批次的命令当作新实验授权。

## 当前状态与系统边界

当前主线是使用已训练的`target-stack`进行三根连续抓放重复实验。三根训练数据、标注和两组模型训练均已完成，trial04已有操作者确认的三根成功及真实空目标恢复记录；当前先保持模型和参数继续试验，不启动新训练或切换基线。提交与问题状态统一见[进展索引](EVIDENCE.md#提交与实验进展总览)。

系统包含感知与动作两部分：实例分割识别每根竹条的可见像素；动作模型根据指定目标轮廓、双路图像和机器人状态预测运动。

| 流程 | 输入 → 输出 | 当前状态 |
| --- | --- | --- |
| 静态 Mask2Former 分割 | 固定全局 RGB → 可见实例掩码、`top_strip` / `covered_strip` | 首轮训练及047～051独立测试完成；漏检、合并实例和被压条误判上层仍有遗留，见[分割证据](EVIDENCE.md#分割模型与独立测试)。 |
| 七维 SmolVLA 动作 | 全局＋腕部 RGB、七维状态、任务文字 → 七维绝对关节动作 | 三根数据上baseline与target两组训练及离线对照完成；当前真机使用目标描边组`target-stack`，不是旧`mixed-both-orders`或`target-pair`，见[模型身份](EVIDENCE.md#三根动作模型身份与训练结论)。 |
| 三根顺序抓放 | 已确认A/B/C种子 → 当前分割／运动参考 → 模型抓放 → 归位换目标 | trial04完成A→B→C，已有三根物理成功反馈；空目标停稳重观察在现场成功恢复一次。自动物理成败判断、稳定成功率与公平基线真机对照仍待验证，见[实机记录](EVIDENCE.md#第四轮三根成功与现场空目标恢复)。 |

静态研究与近期目标动作试采使用 **400×25×8 mm** 短条；旧160条动作训练示教使用60 cm长条。
当前目标服从实验沿用40 cm、原机位和双640×480，不同时切换长度与相机；60 cm及混合长度留待后续明确验证。
不能把短条分割成绩外推为短条抓取或任意无序堆叠成功率。特殊结构、VLM（视觉语言模型）、RL（强化学习）暂缓。

动作契约为 `[J1,J2,J3,J4,J5,J6,gripper_pos]`，关节单位为度，进入机器人 SDK 时转换为弧度；J5由模型预测。
`CameraSetV2` 的 `global_rgb` 为固定全局相机，`grasp_rgb` 为腕部相机，均为640×480。
末端为 Airtac HFKL20 气动平行二指夹爪：0松开、100夹紧；这只是命令，不是持物传感器。
旧字段 `suction_*` 与历史“吸盘/吸附”均指夹爪开合；DO2夹紧、DO3松开。
固定任务文字为 `Pick one strip and place it in the collection area.`。

## 三根堆叠目标示教方案

**状态（2026-10-19）**：两组模型均完成30000步、最终保存、严格重载及同一验证集验收；当前使用目标描边组`target-stack`开展重复真机实验，trial04已有一轮三根成功。离线专家观察上的动作块首步夹爪事件命中较低，不能据此说模型在闭环中完全不会切换；同样，一轮实机成功也不推翻离线指标或证明目标组优于基线。模型位置和评价口径见[训练结论](EVIDENCE.md#三根动作模型身份与训练结论)。

训练p-001～p-020共20堆60条40673帧（含pilot），验证p-101～p-103共3堆9条6023帧；原始中断与续录记录保留。数据采后检查通过不等于持续稳定25 Hz，数量、分组及完整采后证据见[正式采集记录](EVIDENCE.md#三根堆叠正式采集)。

**标注与接口验收**：全部69条示教的23107个接近帧已获用户对话确认，12批共46696帧通过真实LeRobot／PyAV接口验收，无排除episode。剩余10批的18013个接近帧包含179帧RGB辅助修订，确认后掩码与首帧种子未改；原问题提示、旧掩码及修订来源均保留。确认来源是用户对话，不声称逐帧点击或独立逐像素真值，人工审核用时未知。[原审核入口](../../artifacts/capture/stack/joint_stack_train_20261017/review.html#remaining-stack-seed-review)保留各批确认JSON与验收结果。标注阶段复用项目现有目录，没有重复上传或推理；本次获授权训练另建独立远端项目。

**训练与当前使用**：[数据准备检查](../../artifacts/capture/stack/joint_stack_train_20261017/training_data_readiness_check.json)核对两组来源、划分和训练集归一化一致；[完成验收](../../artifacts/capture/stack/joint_stack_train_20261017/training_completion_check.json)确认baseline和target均完成训练及严格重载。baseline输入原始双RGB，target在接近阶段添加指定目标轮廓；后者已接入连续抓放入口。当前使用最终权重做重复现场实验，运动跟随、预览与恢复是运行程序改进，没有重新训练动作模型。历史准备状态不再作为当前未训练的依据。
原“下探高度修正”方案已取消（见[分阶段路线](#分阶段路线)），堆叠层高由模型从示教中学习，本方案不加任何高度、姿态或轨迹补偿。

**场景规格**

- 3根400×25×8 mm竹条，在右侧堆放区支撑台上摆放，两两交角约不超过30°；每根待抓上层条由操作者摆放时保证下指空间。
- 当前布局（2026-10-16用户确认）为普通情况“一根压一根＋一根独立”：两根交叉，第3根单独放在旁边、不接触交叉两根。第1步交叉上层条与独立条二选一（k=2）；先抓上层条时第2步剩余两根均不被压（k=2），先抓独立条时第2、3步k=1。纯一根压一根每步k=1，不能提供服从证据，且描边总落在最上层条，易诱导“抓最上”捷径，故不单独采用。两根压一根、一根压两根等在本布局稳定后再加入；每步可选上层条数按实际登记。
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
- 结果除原`human_outcome`外保存`target_outcome`：`target_matched`、`held`、`single_strip`、`delivered`、`neighbor_disturbed`，分别为抓对、夹住、单根、送达和邻条扰动，取`true/false/null`。2026-10-17起按用户要求不再逐项询问：每条结束只按一次结果菜单，`1`表示操作者现场确认“抓对目标、单根夹起、送达松开”，前四项自动记为真；其他结果前四项记`null`，失败类别只保存在`human_outcome`，抓错目标选`6`；邻条扰动不再记录（`null`）。`plan.json`中`target_outcome_entry=single_operator_confirmation`标明来源；汇总的未知结果只计`6`及缺项的成功记录。首批`joint_stack_pilot`为逐项回答，不改写。前四项明确为真、原结果为`single_success`且恰好一个完整闭合／释放周期，才成为成功目标示教候选；未知或失败保留在原记录中，不自动改标成功。
- 训练集和验证集按整堆划分，同一堆各步不得拆到两边；泄漏检查包含被排除的失败步骤所在堆，避免只检查成功样本而漏掉同堆关联。

**目标轮廓标注（离线）**

- 现有按左右矩形区域筛选前景的办法只适用于两根分离，交叉堆叠不适用。复核页新增首帧种子模式，显示全部分割候选；可点选候选或画多段可见区域多边形，为目标和其余可见竹条分别指定身份。点选候选与多边形两种方式对每根互斥；合并实例不能直接同时分给两根。各根种子不得重叠：交叉处像素只属于上层条，被压条的多边形在上层条边缘断开、画成两段；页面确认时不查重叠，导入时拒绝。
- 首批支持先生成助手辅助自动草稿再人工复核：使用当前实例像素、经图像核对的区域提示，漏分的已放置条可用当前RGB颜色辅助提取。草稿保存在`draft_visible_seeds`，不写成人工已确认种子；复核页预载、按身份着色并支持修改，已有浏览器编辑不被自动覆盖。它是本批辅助标注，不是经过验证的通用身份识别；确认前不能导入为训练标签。导出目录见[人工标注导出目录](#人工标注导出目录)。
- 后续接近帧用冻结Mask2Former的当前帧分割，加多根身份分配生成草稿。现有`PairedStripSeparation`限定两根，按像素到中心线的距离分配归属；交叉处两根中心线都很近，距离无法判断上下遮挡，因此不能只把数量从2改成N。推广时以当前帧分割实例和人工可见区域种子为主、中心线为辅；输出只含当前帧前景，不补遮挡，不沿用旧掩码。“只用当前帧前景”不保证像素分给了正确竹条，交叉处的归属须在复核中重点检查。
- 交叉处一个实例覆盖两根时，须拆分或登记为已知误差；草稿经用户复核接受后才进入训练，沿用`reviewed`与`accepted_annotation_limitations`约定。
- 实时入口`run_joint_smooth.py`支持多根种子。每次重新摆放后采集新快照并确认身份；同一轮`--target-sequence A B C`跨三抓复用该轮种子并逐步移除已完成目标，不要求每抓一根重新圈选。每次推理使用当前图像分割，显式运动跟随模式更新参考位置；自动选择下一根上层目标尚未实现。

默认固定种子模式下，`MultiStripSeparation`以已确认可见种子作为固定身份参考，不把未审核轮廓写回参考，避免合并实例造成身份漂移和遮挡造成参考缩小。当前实例像素须在对应种子8像素范围内；这限制相对首帧的位移，不是逐帧累计移动范围，适用于本批夹紧前竹条基本静止的场景，明显位移需重新标种子。合并实例按可见区域距离拆分，中心线为次要依据；等距及超范围像素不分配，拆分处保留复核提示。主支持判定仍为至少占种子重叠80%、覆盖当前实例50%，并要求整个实例处于主种子的固定小抖动邻域（最多8像素；搜索半径更小时随之收紧），避免放宽搜索范围时把其他身份的小片吞掉。默认搜索半径仍为8像素；修复后的5094帧默认重放与原记录一致，16／24像素诊断不再触发本次发现的身份转移，详见[正式批次证据](EVIDENCE.md#三根堆叠正式采集)。这些是草稿启发式，不是识别精度或物理安全阈值。相关49项软件测试通过，并重放本批2004帧、查看42帧对照；不等于独立场景或实机验收。离线与显式多根模式实时接口共用此类；以上是当时的软件验收范围，后续现场结果见[trial04](EVIDENCE.md#第四轮三根成功与现场空目标恢复)。显式`--track-target-motion`使用移动后的参考位置，适用范围和失败处理见[运行说明](#目标条件模型单次真机试验)；旧`PairedStripSeparation`及其行为不变。

冻结分割漏检诊断已完成两次授权推理，原始候选已保留，后续后处理比较可在本地进行。当前证据显示部分目标形状存在于极低分、含多根竹条的候选中；降低阈值或改变插值方式尚不能替代完整人工审核。维持原参数及已确认标注，剩余10批已完成整段用户确认与接口验收；详细结果与限制统一见[正式批次证据](EVIDENCE.md#三根堆叠正式采集)。若后续要微调分割模型，须补充所选训练图中全部可见竹条的实例及类别，不能直接把仅含目标条的动作训练掩码当作完整分割真值；独立整堆验证与训练授权另行确定。

**当前问题与补录边界（2026-10-17）**：身份判定的半径耦合已修复，冻结分割漏检尚未解决。漏检影响离线轮廓草稿，不是原始RGB／关节示教录制故障，因此不把根治漏检作为补录前置条件；新批次仍须采后检查、完整人工标注审核和接口验收。采集侧另有两项未解决问题：腕部相机曾多次空帧超时，原因未确认；LeRobot的`clear_episode_buffer`不清理视频字段临时帧，中断后可能残留PNG，清理缺陷尚未修复。历史中断及一次收尾伺服关闭未及时确认的记录见[正式采集证据](EVIDENCE.md#三根堆叠正式采集)。不将已通过的采后检查当作当前相机或伺服状态保证；若中断，先核对现场状态与实际保存进度，再用新批次目录续录，不覆盖已有数据。

**规模与训练**

- 首批`p-001`、`p-002`试采计入训练；当前已录完20堆训练、3堆验证，验证堆按独立摆放分批穿插采集。用户已确定剩余10批全部逐帧完整审核，正式试做批审核用时未知，不据此估算人工总工时。
- 摆放与顺序：每次现场前列简单摆放清单（交叉位置、上层条倾斜方向、抓取顺序），验证堆和最终测试堆事先指定。本布局有三种抓取顺序：上层→下层→独立、上层→独立→下层（k为2／2／1）、独立→上层→下层（k为2／1／1），各堆轮换；最终训练A／B／C为7／6／7堆，验证各1堆（见[正式采集记录](EVIDENCE.md#三根堆叠正式采集)）。清单的左右与远近按操作者现场视角书写。首批两堆的交叉上层条同为“左下→右上”倾斜、支撑台右边缘均有竖直条，后续避免重复这一模式。`--split`每次运行固定、已有输出目录被拒绝，训练／验证因此分批分目录；失败占满名额时允许堆不完整，不为凑满补录。目标ID按本堆抓取顺序编号（2026-10-17用户确定，与首批实采一致）：最先抓的竹条为`s-001`，其后依次为`s-002`、`s-003`；失败后重试同一根沿用原ID，`pick_step`照常递增。各根角色（交叉上层／下层／独立）只写在摆放清单和俯视照上，不由ID推断。每堆开始前拍一张俯视照，种子页按同一身份命名。竹条上不贴标签或记号。首批身份见[本批记录](EVIDENCE.md#三根堆叠首批试采)。
- 全部目标标注及接口验收已完成；此前[已授权配置](../../artifacts/capture/stack/joint_stack_train_20261017/training_run_authorized.json)确定两组均从同一`SmolVLA base`开始，仅使用本次堆叠数据，每组30000步、batch 8、seed 42，共用训练集归一化。无描边基线与目标条件模型使用相同60条训练／9条验证示教，顺序运行；这样可控制初始化和数据差异，比较接近阶段描边的作用。旧成对检查点和数据保留，本次候选不混入。该方案及新目录、传输、模型检查和实际训练已获用户明确授权；两组模型检查、30000步、严格重载及保存观测验证均已完成，原顺序运行器正常结束。离线误差、夹紧／释放事件及逐堆结果用于诊断，目标服从与抓取效果仍需事先指定测试堆上的实机验证。
- 远端新建独立项目`/home/rentao/program/lerobot-aubo-joint-stack`，复用原Python环境及模型缓存；[授权传输清单](../../artifacts/capture/stack/joint_stack_train_20261017/training_transfer_authorized.json)只含训练所需原始视频／数据、采集证据、已确认标注及隔离代码，不重复上传分割RGB、旧预测或模型权重。用户已明确允许新建目录，原成对项目和结果保留；训练器拒绝已有输出目录，只保存最终检查点，不支持隐式断点续训。具体资源、估时和验收边界见[证据](EVIDENCE.md#三根堆叠正式采集)。

两组完成验收和离线对照后，原训练跟进已结束；逐关节、逐堆、分阶段和夹爪事件结论见[训练结论](EVIDENCE.md#三根动作模型身份与训练结论)。完成报告与预测保存在既有`training_results/`。当前下一步是按用户选择继续使用`target-stack`重复新摆放实验，先收集整轮与逐根结果，再判断是否需要额外诊断或对照；不因离线首步事件指标直接增加训练步数、改变阈值或加入脚本抓放。

**补录范围与原计划（2026-10-17，已完成）**：新增训练p-017～p-020共12条，验证p-103共3条，达到训练20堆60条、验证3堆9条。以下保留录制前建议；实际角色以原图及`layout_record.json`为准，p-020已确认改为C顺序。

| 堆编号 | 划分 | 建议抓取顺序 | 三步可选数k（按计划摆放时） |
|---|---|---|---|
| p-017 | 训练 | B：交叉上层→独立→交叉下层 | 2／2／1 |
| p-018 | 训练 | C：独立→交叉上层→交叉下层 | 2／1／1 |
| p-019 | 训练 | A：交叉上层→交叉下层→独立 | 2／2／1 |
| p-020 | 训练 | B：交叉上层→独立→交叉下层 | 2／2／1 |
| p-103 | 验证 | A：交叉上层→交叉下层→独立 | 2／2／1 |

原计划对应训练A／B／C为7／7／6堆；p-020实际C，因此最终为7／6／7堆，验证各1堆。左右位置按操作者视角描述，各堆改变交叉位置和朝向，独立条与交叉对留出可辨间隔；保持现有相机、起点和放置区。每堆按实际抓取顺序编号`s-001`～`s-003`，每条的`pick_step`与k依据实际现场填写；跨批续录同一堆保留身份及已保存尝试序号。新堆开录前拍照，并将实际角色／顺序补充到既有`layout_record.json`，不改旧堆记录。

**补录当前进度**：`joint_stack_train_20261017_p017_p020`保存7条4744帧，续录`joint_stack_train_20261017_from_p019_step2`保存5条3502帧，`joint_stack_validation_20261017_p103`保存3条2055帧；三批均完成采后检查及本地标注准备，无残留临时PNG。助手已核对五堆三步起始全局图：p-017 B、p-018 C、p-019 A、p-020 C、p-103 A；原摆放记录已补齐并保留修改前副本。p-020仍保留用户确认来源和原计划B，未改示教元数据或ID。采集已结束，下面两个补录输出目录均已存在，命令仅供历史回溯，不能直接重复执行。续录成功不证明腕部相机间歇故障根因已解决。

**评价**：每次指定目标，分别记录抓对、夹住、恰好一根、送达及邻条扰动，以及当时可选上层条数k。目标服从只统计k≥2的步骤；1／k仅是“均匀随机选择”的参考水平，主要对照是同样步骤下无描边基线的实际选择分布。k=1的步骤只用于抓放能力统计。中止轮记为未知，不剔除。

**软件入口与使用顺序**

以下为首批总计6次保存尝试的采集命令，**本批已完成，原输出目录不可重复使用；新批次须更换批次名**。省略`--record`只预览，不创建目录、不连接设备；它约对应无失败时的2堆，失败也计入总数，不保证清完2堆。每堆`pick_step`上限6与整批`--num-episodes`是不同限制。现场另行授权后由操作者添加`--record`，每条仍按`r`归位、`→`开始／结束。

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/record_joint.py \
  --stack-demonstrations --camera-set original-global --split train --num-episodes 6 \
  --dataset-root datasets/stack/train/joint_stack_pilot --evidence-root artifacts/capture/stack/joint_stack_pilot
```

正式批次按日期命名、每批换新目录，同日多批在日期后加堆号范围（如`_p010_p016`）：训练如`--split train --num-episodes 21 --dataset-root datasets/stack/train/joint_stack_train_<日期> --evidence-root artifacts/capture/stack/joint_stack_train_<日期>`，验证如`--split validation --num-episodes 3 --dataset-root datasets/stack/validation/joint_stack_validation_<日期> --evidence-root artifacts/capture/stack/joint_stack_validation_<日期>`，其余参数相同；各批分别走下方采后流程。

下列为已完成补录的历史命令，原输出目录均已存在，不再执行。`--num-episodes`计保存尝试数；中断续录的实际目录及进度见上文。

```bash
# 训练：p-017～p-020；首次输入p-017、s-001、pick_step=1、k=2。
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/record_joint.py \
  --stack-demonstrations --camera-set original-global --split train --num-episodes 12 \
  --dataset-root datasets/stack/train/joint_stack_train_20261017_p017_p020 \
  --evidence-root artifacts/capture/stack/joint_stack_train_20261017_p017_p020 --record

# 验证：p-103；独立新摆放，按A顺序录3条。
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/record_joint.py \
  --stack-demonstrations --camera-set original-global --split validation --num-episodes 3 \
  --dataset-root datasets/stack/validation/joint_stack_validation_20261017_p103 \
  --evidence-root artifacts/capture/stack/joint_stack_validation_20261017_p103 --record
```

采后始终显式传入同批`--dataset-root`和`--evidence-root`，依次调用`prepare_joint_target_review.py`：

1. `--stage capture-check`：新批次生成`raw_audit.json`、`capture_check.json`和固定`review.html`；完整解码双路视频并核对帧时间，保存少量关键帧；已内含`audit_joint_dataset.py`的只读审计，采集结束时提示的单独审计可省略。已有检查或复核产物时拒绝覆盖。
2. `--stage prepare`：提取接近阶段当前RGB，准备原有推理清单；不运行分割模型。获得另行授权后才对该清单运行冻结分割。
3. `--stage import-predictions --predictions /本批预测报告.json`：保存本批当前实例图和首帧候选；堆叠不得使用`--single-strip-scenes`。
4. 打开本批`review.html`，在每条首帧为全部可见竹条点选／画种子并明确确认，导出JSON；`--stage import-seeds --decisions /导出的记录.json`导入。
5. `--stage separate`生成全部接近帧草稿，离线与实时调用同一分离器。不覆盖已有掩码；发现误差用页面逐帧修改，或“登记可接受的已知误差”，再`--stage import-review --decisions /导出的记录.json`。修改后必须查看重新生成的轮廓，明确确认范围后再次导入；浏览页面不会自动复核。
6. `--stage verify`只对通过原始审计的成功目标示教做真实LeRobot/PyAV接口验收，报告排除的失败／未知episode；检查原始状态、动作、50步动作块尾部填充及训练／运行阶段一致。数据就绪仍不等于训练或物理成功。

实时软件的`--multi-target-separation`与旧`--paired-target-separation`互斥，支持1／2／3根剩余条的圈选。早期`--model target-pair`的软件准备绑定旧成对检查点及其`artifacts/joint_target_live_service/multi_foreground/`路径，不能代替本次模型。2026-10-18新增`--model target-stack`，绑定已完成训练的堆叠目标模型和独立服务，自动启用多目标分离；现行部署验收、现场快照和自动归位连续抓放流程统一见[真机入口](#目标条件模型单次真机试验)。

合成页面、检查结果及回滚位置统一见[软件准备证据](EVIDENCE.md#三根堆叠软件准备)。[腕部诊断工具](#斜条j6欠转的腕部角度诊断)独立于堆叠采集流程。

## 代码导航与维护

命令入口在 `examples/phone_to_auboi10/`，任务实现集中在 `src/lerobot/bamboo_sorting/`；机器人驱动在 `src/lerobot/robots/aubo_i10/`。先按下表定位，不按文件名猜测当前方案。

任务包按职责分为 `capture/`（采集、相机、时间戳、夹爪证据、采后检查）、`perception/`（分割、目标身份关联、前景分离、运动跟随）、`policy/`（目标图像、训练数据、动作模型适配）、`runtime/`（设备连接、推理通信、执行调度、预览录像）和 `diagnostics/`（腕部角度等离线诊断）。公共契约与路径常量保留在根级 `aubo_joint_contract.py`、`joint_gripper_quality.py`、`joint_stack_metadata.py`、`mask_codec.py`、`model_profiles.py`、`paths.py`、`start_poses.py`；命令入口只保留参数、调用和退出状态。

### 当前任务入口

| 要做的事 | 入口 | 主要实现／操作说明 |
| --- | --- | --- |
| 三根堆叠目标示教 | [record_joint.py](record_joint.py) 的 `--stack-demonstrations` | `capture/aubo_joint_capture.py`、`joint_stack_metadata.py`；[堆叠流程](#三根堆叠目标示教方案) |
| 采后检查、种子和逐帧轮廓复核 | [prepare_joint_target_review.py](prepare_joint_target_review.py)、[joint_target_review.html](joint_target_review.html) | `capture/target_review.py`、`perception/target_review.py`、`policy/target_verification.py`；`capture-check → prepare → import-predictions → import-seeds → separate → import-review → verify`；分割推理另行授权 |
| 审计原始关节数据 | [audit_joint_dataset.py](audit_joint_dataset.py) | `capture/aubo_joint_audit.py`、`joint_gripper_quality.py`；成功候选筛选不代替物理结果确认 |
| 多实例分离与目标输入 | [joint_target_separation.py](../../src/lerobot/bamboo_sorting/perception/joint_target_separation.py)、[joint_target_policy.py](../../src/lerobot/bamboo_sorting/policy/joint_target_policy.py) | `MultiStripSeparation`由离线与实时共用；[接口约定](#目标条件接口约定) |
| 动作训练／离线评价 | [train_joint_smolvla.py](train_joint_smolvla.py)、[diagnose_joint_target_policy.py](diagnose_joint_target_policy.py) | `policy/joint_training.py`、`policy/smolvla_joint_adapter.py`；[训练阶段](#阶段与默认值)，已有诊断不重复运行 |
| 目标圈选与现场运行 | [run_joint_target_shadow.py](run_joint_target_shadow.py)、[joint_target_shadow_review.html](joint_target_shadow_review.html)、[run_joint_smooth.py](run_joint_smooth.py) | `runtime/live_setup.py`、`runtime/joint_target_runtime.py`、`runtime/joint_smooth.py`；[运行说明](#目标条件模型单次真机试验)，`target-stack`已完成trial04一轮三根成功反馈及现场空目标恢复，见[实机复核](EVIDENCE.md#第四轮三根成功与现场空目标恢复) |
| 静态分割采集／人工修边 | [capture_placement_sequence.py](capture_placement_sequence.py)、[review_placement_sequence.py](review_placement_sequence.py) | `perception/placement_sequence.py`、`perception/placement_review.py`／`.html`；[采集与复核](#静态分割采集与人工复核) |
| 分割数据／训练／预测评价 | [export_strip_segmentation.py](export_strip_segmentation.py)、[train_strip_segmentation.py](train_strip_segmentation.py)、[predict_strip_images.py](predict_strip_images.py)、[evaluate_strip_segmentation.py](evaluate_strip_segmentation.py)、[review_strip_predictions.py](review_strip_predictions.py) | `perception/strip_segmentation_data.py`、`perception/strip_segmentation_training.py`；[分割流程](#静态分割数据训练与独立评价) |
| 腕部20帧人工结果导入 | [review_joint_wrist_distribution.py](review_joint_wrist_distribution.py) 的 `--stage errors` | `diagnostics/joint_wrist_distribution.py`、`diagnostics/joint_wrist_angle.py`；[人工复核流程](#斜条j6欠转的腕部角度诊断)；`diagnose_joint_wrist_angle.py`保留已完成测量的复现入口 |

### 共享依赖与现场辅助

这些文件仍被当前代码调用，旧名称不表示可删除。`examples/`中的脚本也并非全都适合直接运行。

| 文件／模块 | 当前用途 |
| --- | --- |
| `record_joint.py` + `capture/fresh_frames.py`、`capture/homing.py` | 当前关节采集的相机新鲜度检查与正常归位；旧`record.py`和`record_c0_batch.py`不再是当前采集依赖，`record_c0_batch.py`只保留兼容导入 |
| `capture/c0_batch_capture.py`、`capture/c0_gripper_*` | 关节采集的帧时间、夹爪命令与保存证据；不是当前8维末端动作方案 |
| `capture/joint_camera_config.py → capture/c0_smoke_capture.py` | 当前相机配置与冻结映射；配置文件仍在 `configs/aubo_i10/` |
| `run_joint_trial.py`、`runtime/inference.py`、`runtime/station.py` | 命令入口保留[归位入口](#2-归位)；`InferencePipe`和`ReadOnlyStation`在运行模块中供各入口复用 |
| `run_joint_live.py`、`runtime/task_prompt.py` | 保留停稳后单步观察的兼容入口；任务文字读取在运行模块中供持续伺服入口复用 |
| `joint_inference_stdio.py` | 远端模型数值推理协议；由运行链调用，不是本地机械臂控制入口 |
| `start_poses.py`、`aubo_start_poses.py`、`move_to_start.py`、`release_joint_gripper.py`、`teleoperate.py` | 起点常量在任务包中统一维护，`aubo_start_poses.py`只重新导出；移动、单独松爪和手机遥操涉及现场动作，须按相应授权执行 |
| `diag_preflight_cams.py`、`diag_cam_latency.py`、`read_pose.py` | 相机体检／延迟、机器人位姿读取；读位姿也需要设备连接 |

### 历史与专项工具

| 文件 | 状态与适用范围 |
| --- | --- |
| `review_joint_target.py` | 保存录像的关键帧与已有诊断结果展示；不能生成已接受训练标签 |
| `associate_joint_target.py` | 保存分割结果的离线关联诊断；底层 `perception/joint_target_association.py` 仍被实时路径使用，不能一起移除 |

旧ACT／C0聚合审计、双图层序评价、光流诊断和一次性现场探测入口已退役；删除范围、目录保留理由及源码回溯见[EVIDENCE](EVIDENCE.md#目录与退役代码清理)。历史记录中的旧命令不再是当前可用入口。

软件验证按实际改动选择 `tests/bamboo_sorting/` 及相关驱动用例，使用项目 `.venv/bin/python`。纯文档整理只检查差异、链接与回溯位置，不重跑训练、推理或无关测试。分割测试的额外依赖路径以本机实际存在的环境为准，不把历史 `/tmp` 路径当长期配置。

`datasets/`、`examples/phone_to_auboi10/datasets/`、`artifacts/`中的数据和人工结果保持原位；`pile-json--/`按下方导出分类保存，原入口保留兼容链接。源码与测试按完整调用链维护；测试已按用户确认的AUBO范围裁剪，无关框架功能的实现仍保留，但不再拥有原完整测试覆盖。目录名含 `test`、`images` 或 `c0` 不能作为删除依据。文档只维护本手册和[EVIDENCE](EVIDENCE.md)，不新增逐轮计划或总结。维护约定见[AGENT.md](../../AGENT.md#文档维护规则)。

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

## 数据集目录

本机数据集按用途存放在仓库根目录的`datasets/`，每个批次保留原名称和内部结构：

| 分类目录 | 用途 | 当前批次数 |
| --- | --- | --- |
| `datasets/target_conditioned/train/` | 已纳入训练来源的目标条件A／B示教 | 10 |
| `datasets/target_conditioned/validation/` | 目标条件独立验证示教 | 2 |
| `datasets/target_conditioned/pilot/` | 早期单目标试采`joint_target_pilot` | 1 |
| `datasets/single_strip/train/`、`validation/` | 历史单根关节示教 | 8／1 |
| `datasets/two_strip/train/`、`validation/` | 双根、两种上下摆放的关节示教 | 3／3 |
| `datasets/stack/train/` | 三根堆叠p-001～p-020，20堆60条40673帧；9批采后检查、完整标注接受及真实接口verify全部通过，训练候选已准备 | 9 |
| `datasets/stack/validation/` | 独立验证p-101～p-103，3堆9条6023帧；3批采后检查、完整标注接受及真实接口verify全部通过 | 3 |
| `datasets/segmentation/` | `short_strip_segmentation`（内含train／validation）与`short_strip_segmentation_test` | 2 |

`joint_target_pair_pilot`已纳入正式训练来源，因此归入`target_conditioned/train/`，不单凭名称中的pilot判断用途。
这些目录仅分类存储，不改变来源清单中的episode筛选、失败排除或训练／验证划分；同类批次没有合并。
新录制按用途显式设置`--dataset-root`，`--evidence-root`仍在`artifacts/`下；三根堆叠使用`datasets/stack/train/`和`datasets/stack/validation/`，本轮采集已完成（训练20堆、验证3堆），不预建空批次。

2026-10-16已移动30个数据集，保留旧`datasets/<批次名>`为相对符号链接，以兼容既有训练清单、脚本默认值和历史报告。
`datasets/.hidden`使支持该约定的文件管理器在普通视图隐藏这些旧入口；终端及不支持`.hidden`的编辑器仍会列出链接。
链接不占一份数据空间；备份／同步时应保留符号链接，避免跟随链接重复复制。
原始视频、Parquet、元数据和标注内容未改；`artifacts/`的分类见下节，远端路径未变。

移动映射与逐文件核对结果见`artifacts/maintenance/dataset_organization/manifest.json`，核对了9474个文件的相对路径、文件身份、大小和修改时间。
回滚预览：`python3 artifacts/maintenance/dataset_organization/rollback.py`；加`--apply`恢复原目录布局和整理前的本文，保留整理前已有文档修改。
若本文或兼容链接已被后续修改，回滚会要求先人工核对，不覆盖新成果。

## 人工标注导出目录

人工导出统一存入仓库根目录`pile-json--/`，按用途／批次分类；JSON内部内容不因移动而改写：

| 目录 | 内容 |
| --- | --- |
| `pile-json--/segmentation/` | 48份静态堆放`pile_XXX_polygons.json` |
| `pile-json--/target_selection/` | 2份历史现场`target_shadow_*_selection.json` |
| `pile-json--/stack/joint_stack_pilot/` | 本次首帧自动草稿、已确认首帧`joint_stack_pilot_review_decisions.json`，4帧补标来源草稿`joint_stack_pilot_sequence_repairs_draft.json`及全序列确认`joint_stack_pilot_full_sequence_review_decisions.json` |
| `pile-json--/stack/joint_stack_train_20261017_p010_p016/` | 正式试做批次的首帧草稿／确认、68帧RGB修订来源草稿及3090帧全序列确认JSON |

2026-10-16按用户要求移动原50份文件，根目录同名入口保留相对符号链接；`.hidden`在支持它的文件管理器中隐藏旧入口，不重复占用数据空间。迁移核对文件身份、大小、修改时间和JSON可读性；清单与回滚脚本在`artifacts/maintenance/annotation_export_organization/`。回滚预览：`.venv/bin/python artifacts/maintenance/annotation_export_organization/rollback.py`，加`--apply`恢复这50份文件的旧布局，不删除新批次导出。

当前自动草稿为`joint_stack_pilot_auto_seed_drafts.json`，不是人工确认记录。复核页导出时将文件保存到本批子目录；普通网页不能强制浏览器下载到指定绝对路径，不修改浏览器默认下载设置。若浏览器仍下载到其他位置，导入前将导出文件归入本批目录，已有同名人工结果须先核对，不静默覆盖。

## 实验产物目录

2026-10-16将本机`artifacts/`原有95个目录按用途分类，每批内部的录像、截图、日志和复核材料保留在一起：

| 分类目录 | 内容 | 原有目录数 |
| --- | --- | --- |
| `artifacts/capture/target_conditioned/` | 目标条件示教的采集记录、抽帧、掩码和复核页 | 13 |
| `artifacts/capture/single_strip/`、`two_strip/` | 单根／双根采集证据，包括历史试采 | 11／6 |
| `artifacts/inference/target_conditioned/` | 指定A／B目标、重新摆放后的实机录像与控制记录 | 19 |
| `artifacts/inference/single_strip/`、`two_strip/` | 单根短条／双根实机试验 | 5／9 |
| `artifacts/inference/snapshots/`、`homing/` | 静止圈选与影子预测／归位记录 | 4／7 |
| `artifacts/training/` | 动作训练准备、预检、训练结果及历史验收 | 8 |
| `artifacts/segmentation/` | 静态摆放采集与复核、分割训练和独立评价 | 3 |
| `artifacts/diagnostics/` | 目标跟踪、腕部角度、相机、堆叠软件准备及元数据修正 | 6 |
| `artifacts/maintenance/` | 目录整理、清理记录、备份与恢复材料 | 4＋本次整理记录 |

旧`artifacts/<原名称>`保留相对符号链接，旧清单与历史文档仍可访问；`.hidden`仅控制支持它的文件管理器中的显示。
复核网页中受移动影响的跨目录链接已改为本机绝对文件地址；当前分析脚本改为查找项目根目录，避免依赖固定层数。
仅这些路径引用及复核入口的路径兼容逻辑有修改，原始录像、图片、动作记录与标注未重写。
已存在的历史失效链接单独记录，不将它们视为此次搬迁产生，也不凭猜测修复。

新采集用对应`capture/`分类作为`--evidence-root`，新实机试验用对应`inference/`分类作为`--output`。
脚本未显式指定输出路径时仍沿用其原默认值；新批次须按当前操作命令显式指定分类路径。
远端目录未移动；同步时保留符号链接，不跟随链接重复复制。本机绝对网页链接跨机器使用前需要按目标机器路径重新生成。

迁移清单、路径修改前后副本、检查结果和回滚脚本集中在`artifacts/maintenance/artifact_organization/`。
回滚预览：`python3 artifacts/maintenance/artifact_organization/rollback.py`；加`--apply`恢复本次移动及路径修改。
若两轮整理都要撤销，先回滚artifacts，再按上一节回滚datasets；这样文档也按修改顺序恢复，保留两轮之前已有修改。

历史录制的`source_dataset_root`是原始身份，不能随移动改写。单独运行`audit_joint_dataset.py`检查迁移后的老批次时，需用
`--source-dataset-root /home/rentao/program/lerobot-aubo/datasets/<原批次名>`声明原路径；既有训练清单已包含该字段。
目标复核工具接受指向同一目录的新旧路径，并在接口验收时沿用标注记录的原始身份；不同来源仍拒绝。

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

原图集中于 `artifacts/segmentation/placement_sequences/pile_XXX/`；人工下载标注在 `pile-json--/segmentation/`；
[人工复核入口](../../artifacts/segmentation/placement_sequences/review/index.html) 是唯一人工标注页面入口，每组只维护 `review/pile_XXX/` 一份结果。
当前正式训练／验证为007～046，独立测试047～051已经完成，不能重复拍到这些目录。
每次只安排后续5组；安排前检查原始目录、统一review及下载JSON的组号，并核对脚本参数，不提前假定后续编号可用。

1. 每组清空后独立重摆，拍空场景及依次加入1、2、3根，共4张图；保持普通逐层结构，自然变化位置、方向、交叉点。
2. 逐根从上方加入；旧条不能移动或明显倾斜；相机、承托面保持不动。手退出、场景静止再拍，整根及端部余量留在原图内。
3. 每五组集中检查照片，由操作者确认上述现场条件。照片不能完全证明旧条未移动；条件不成立时不能直接传播遮挡真值。
4. 同一摆放的阶段图、衍生图、关联对照保持同一 split（数据集合）。测试组在采集前指定，不按结果重新划分。

以下为**参数模板**，须先填经核对的新组号和集合；省略 `--record` 只看计划、不创建目录：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/capture_placement_sequence.py \
  --camera-set original-global --output "artifacts/segmentation/placement_sequences/<新组号>" \
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
  --source "artifacts/segmentation/placement_sequences/<组号>" \
  --polygons "pile-json--/segmentation/<组号>_polygons.json"
```

省略 `--output` 更新该组统一复核目录及索引：先在临时位置生成，失败保留原结果，成功后替换同组派生结果。
原照片、下载JSON不变，不建多个最终review目录。确认“从上方加入且旧条静止”后，工具按后放轮廓扣除先放条的被遮部分；
可见区域断为两段仍是同一根实例。`training_ready=false`、`graspable=null` 保留原始含义：复核并不证明已训练或物理可夹。

## 静态分割：数据、训练与独立评价

### 数据与模型输入

| 集合 | 组号与选择清单 | 组数／图数／可见实例 | 现有导出目录 |
| --- | --- | --- | --- |
| train | 007～046中除下列validation；[选择清单](../../configs/aubo_i10/short_strip_segmentation.json) | 30／120／180 | `datasets/segmentation/short_strip_segmentation/` |
| validation | 011、016、021、026、031、036、041、044、045、046 | 10／40／60 | 同上 |
| test | 047～051；[选择清单](../../configs/aubo_i10/short_strip_segmentation_test.json) | 5／20／30 | `datasets/segmentation/short_strip_segmentation_test/` |

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
  --stage preflight --dataset datasets/segmentation/short_strip_segmentation
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

- [验证集预测复核](../../artifacts/segmentation/short_strip_training/result/prediction_review.html)：40图／10组。
- [独立测试预测复核](../../artifacts/segmentation/short_strip_training/independent_test/result/prediction_review.html)：20图／5组。

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
  --camera-set original-global --dataset-root datasets/single_strip/train/joint_train_run02 \
  --evidence-root artifacts/capture/single_strip/joint_train_run02 --num-episodes 1 --split train
```

正式采集由操作者经授权后添加 `--record`。填写场景及摆放编号，`r`归位，右箭头开始／结束，左箭头重录，Esc停止。
手机完成接近、夹紧、提起、搬运、释放，结束前须释放并填写实际结果；保留失败与中断记录。
录制和控制目标25 Hz；填表时暂停快捷键监听。归位使用 `NORMAL_START_DEG`；退出先停伺服再整理视频。
Esc不是硬件急停，不能中断等待完成的归位。既有 `single_success` 标签不自动表示双根或整堆成功。

```bash
.venv/bin/python examples/phone_to_auboi10/audit_joint_dataset.py \
  --dataset-root datasets/single_strip/train/joint_train_run02 \
  --evidence-root artifacts/capture/single_strip/joint_train_run02 --require-gripper-quality
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

当前本机清单：`artifacts/training/aubo_joint_both_orders_20260928/sources.json`。
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
  --data-manifest artifacts/training/aubo_joint_both_orders_20260928/sources.json
```

本机数据预检（输出须全新）：

```bash
.venv/bin/python examples/phone_to_auboi10/train_joint_smolvla.py \
  --stage preflight \
  --data-manifest artifacts/training/aubo_joint_both_orders_20260928/sources.json \
  --output artifacts/training/joint_preflight_run02
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
  --data-manifest artifacts/training/aubo_joint_both_orders_20260928/sources_remote.json \
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
目标条件入口中，`--model target-stack`绑定本次三根堆叠模型，`target-pair`保留旧成对模型；参数与命令只维护在[目标条件模型单次真机试验](#目标条件模型单次真机试验)。
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
  --output "artifacts/inference/two_strip/joint_both_top90_$(date +%Y%m%d_%H%M%S_%N)"
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

`run_joint_smooth.py --model target-stack`绑定本次30000步目标模型，自动使用多目标可见种子分离。`--target-id`只抓一个指定目标；`--target-sequence A B C`依次抓放三根，两次之间自动归位，最后一根结束后停止。`target-pair`保留旧成对模型，不能用它代替新模型试验。新模型位于`/home/rentao/program/lerobot-aubo-joint-stack/outputs/joint_stack_target_smolvla/final`，独立服务为同项目`artifacts/joint_stack_live_service/joint_inference_stdio.py`；身份与无硬件验收见[部署记录](EVIDENCE.md#三根堆叠正式采集)。旧A／B历史见[原证据](EVIDENCE.md#目标条件模型最新ab现场结果与b失败复核)。

此前`stack_target_20261018_trial03/three_pick_run`已由操作者运行：三次指令周期完成，但操作者确认B未抓到，不能记为三根物理成功。保存记录与成功下层示教对照表明，B闭爪高度偏高且腕部姿态存在差异；第118次非空小掩码异常不是唯一解释。详见[第三轮B抓空对照](EVIDENCE.md#第三轮b抓空与成功下层示教对照)。trial01～trial04的快照及输出均保留为历史证据，不可覆盖；下方通用命令必须使用新摆放、新目录及对应标注。

最新`stack_target_20261018_trial04/three_pick_run`已执行，2026-10-19操作者反馈“一轮抓取，三根竹条都抓取了”。日志确认A→B→C各一个抓放周期、两次归位，关键帧显示三根依次送至左侧、右侧抓取区清空；本轮启用了实时预览与运动跟随。A第33次请求出现空目标，停稳后第34次新观察恢复，随后完成整轮。B本次闭爪附近反馈高度比上次低约5.63 mm，未添加脚本高度补偿；新摆放与跟踪同时变化，不能据单轮归因或宣称稳定成功率。详见[第四轮实机复核](EVIDENCE.md#第四轮三根成功与现场空目标恢复)。当前快照和运行目录均为已完成实验，不可直接复用为新试验输出；标准标注`selection.json`保留，重复导出`selection (1).json`已按用户授权核对一致后删除。

**实时预览与扰动后的目标跟随（显式试验模式）**：用户已选择自动跟随移动目标、由模型持续修正。在本节运行命令上增加`--target-preview --track-target-motion`；后者只支持`target-stack`，前者也可单独用于目标模型。启动后终端显示`http://127.0.0.1:<端口>/`，在本机浏览器打开；地址同时写入新运行目录`preview.json`。左图显示该次预测的全局图像、实际输入模型的紫色轮廓及仅供观看的黄色外框，右图为同次请求的腕部图像；目标编号、阶段、预测编号、目标像素数和图像年龄同步显示。随新预测更新，不把旧框贴到较新的相机图像上；不是30 FPS无延迟检测。浏览器断开或图像变旧会显示提示，搬运和切换目标时清除旧框。预览是独立进程，队列满则丢预览帧，不阻塞伺服，不打开第二路相机，也不从网页发送机械臂命令。

默认模式仍使用最初人工种子附近8像素的当前分割。`--track-target-motion`则用连续RGB纹理运动估计平移／旋转，移动身份参考后，仍只把本帧分割前景作为模型轮廓；不把分割草稿反复当作新身份种子，也不改原始人工标注。A搬运期间继续更新B/C的运动参考，切换模型会话后保留这些参考，已夹取目标从跟踪集合中移除。光流（相邻图像特征点的移动）需通过前后匹配和刚性拟合检查：单次观测中心位移≤35 px、旋转≤15°、尺度0.97～1.03、至少6个一致点及足够轴向跨度。特征不足时保留最近可靠参考，只有当前分割覆盖该参考至少50%才继续；正常运动匹配也要求至少25%当前前景支持。参考发生可见区域重叠时不接受该运动更新。这些是实验性身份关联条件，不是已验证的物理运动范围。失败报错“运动超出限制”合并了位移、旋转、尺度和轴向跨度四项，失败时也不记录各项数值；trial04中静止的C有225／226次跟随失败，现场触发项无法读回，见[复现](EVIDENCE.md#第四轮三根成功与现场空目标恢复)。

现有空目标有界恢复仍保留。新跟随模式中，非空但支持不足的轮廓／无法确认的运动参考会按错误停止整轮，不自动改抓邻条，也不放宽阈值强行完成；它不是完整遮挡后重新识别同一根竹条的方案。该模式通过合成运动、保存录像、模拟三根流程及远端保存图像检查；trial04已有B参考随小幅运动更新并完成三根抓取的现场记录，仍未验证大位移、完全遮挡后身份恢复或稳定成功率。移除`--track-target-motion`即可回到固定种子关联，预览可以保留。新现场必须使用对应的新快照／人工种子和新运行目录；不能直接重用已移走A/C的trial03摆放。结果见[受控预测与跟随验证](EVIDENCE.md#受控轮廓预测与实时目标跟随)。

**三根连续抓放的现场流程（2026-10-18授权）**：摆两根交叉、一根独立，保持原机位、起点和放置区。当前现场身份A为交叉上层条、B为下层条、C为独立条。使用`--target-sequence A B C`后，操作者一次启动：A夹紧—释放完成→停止伺服→自动归位→切换B→B完成后同样归位→切换C→C完成后停止。每次都重新读取当前图像并分割，原人工标注只作身份种子；剩余目标按A/B/C→B/C→C递减。两次之间不再要求重新拍照、圈选或输入命令。若其他竹条被带动或抓放异常，由现场操作者停止并重建场景；程序的夹爪命令计数不能自动证明物体已移走。trial04已有操作者确认三根均抓取成功及对应关键帧，自动物理成功判断仍未实现。

每次归位复用现有`home_to_start`，只在释放命令成功且伺服关闭、机器人静止后执行；沿用60°/s、80°/s²及50%速度比例并在结束后恢复原比例，不受策略5°/s限速替代。每根使用独立推理会话，归位后会等待下一次模型加载，清空上一根动作块和阶段；权重、任务文字及模型动作不变。除下述有界空目标重新观察外，异常停止后续目标；不在最后一根后归位。`--max-seconds 600`在顺序模式中表示每根最多600秒（完成即结束），归位和模型加载另计。

**空目标暂停与恢复**：`target-stack`接近阶段若收到与当前请求／身份／快照一致的`target_foreground_missing`，先关闭伺服、等待静止，保持实际夹爪命令；清空旧动作块、待夹紧目标和轨迹速度。随后使用停止之后的双路新图像与当前关节状态请求预测，有效且通过原有校验后从实测位置、零轨迹速度重新接续。每根累计最多3次新观测重试（成功一次不重置预算），第4次空目标停止整轮，不跳过当前目标。3次是监督试验的有限恢复预算，并非已验证的漏检持续时间；每次仍受原响应等待限制，每根总时限不重置。已经接受夹紧后不进入此恢复路径，通信／模型／状态／停伺服错误仍停止。

远端仅对这种明确的多目标空分配保留会话；失败帧仍消耗自己的序号，不调用动作模型，不返回旧动作或旧掩码。错误证据保存在新运行目录`target_failures/<请求编号>/`：双路原始RGB无损PNG、候选实例PNG，以及`failure.json`中的请求编号、目标／阶段、观测关节、来源时间戳、候选列表、各目标掩码与分配报告。旧双目标服务不因本地修改自动获得此协议。trial04已在A的一次空目标后实测停稳并用新观察接续；这一例验证恢复流程，不代表分割漏检已修复或连续多帧丢失均可恢复；[软件与保存图像验证](EVIDENCE.md#三根堆叠正式采集)。

**下一轮操作模板**：保持当前模型、参数和机位，由操作者完成新摆放；先按[归位条件](#2-归位)执行归位并确认成功，再采集静止快照。以下只展示流程，不代表助手获得硬件操作权限。

先选择尚不存在的新实验根目录，并在同一终端保留变量。日期／试验号按实际填写，不能复用已完成trial04：

```bash
TRIAL_ROOT='artifacts/live/stack/stack_target_<日期>_<新试验号>'
```

尖括号内容是占位符，须替换成实际名称后执行。`TRIAL_ROOT`只是本终端定位实验产物的变量，不改变系统环境配置。核对根目录不存在后，由操作者采集快照（读取相机和状态，不发送运动／夹爪命令）：

```bash
.venv/bin/python examples/phone_to_auboi10/run_joint_target_shadow.py \
  --stage capture --onsite-confirmed \
  --output "$TRIAL_ROOT/scene_snapshot"
```

根据新图确认A上层／B下层／C独立条，可先生成可检查草稿；下层条在遮挡处拆成可见段，不补遮挡。审核导出到该快照目录的`selection.json`后，核对`human_reviewed=true`、snapshot ID、身份齐全、轮廓非空且互不重叠。草稿不能自动标成人工已审核。若浏览器导出为`selection (1).json`等名称，先检查实际文件和内容再规范文件名，删除重复导出须另有用户授权。

核对计划（不连接设备或加载模型）：

```bash
.venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
  --model target-stack \
  --target-snapshot "$TRIAL_ROOT/scene_snapshot" \
  --target-selections "$TRIAL_ROOT/scene_snapshot/selection.json" \
  --target-sequence A B C --target-preview --track-target-motion \
  --transport-age-aligned --latency-tolerant --max-seconds 600
```

计划须指向当前目标检查点、`multi_foreground`、三次抓放、两次归位、实时预览和运动跟随。保持对应摆放与起点，操作者在同一命令后添加`--execute --output "$TRIAL_ROOT/three_pick_run"`；运行输出必须尚不存在。任务文字为`Pick one strip and place it in the collection area.`。浏览器地址由终端打印；按终端提示确认后开始，异常时`Ctrl+C`停止，紧急情况使用现场急停。

抓完后记录本轮新摆放、逐根抓对／夹住／单根／送达、邻条扰动、是否中断及原因，再检查`complete.json`、`trace.jsonl`、`commands.jsonl`和双路录像。软件周期完成与操作者物理结果分别保留；不改变原始日志中的`physical_grasp_success`。每轮使用一个固定复核目录，摘要更新到[EVIDENCE](EVIDENCE.md#提交与实验进展总览)。

**中断后续抓**：只有操作者确认A已移走、剩余B/C与原参考及机位仍相符时，才可评估改用`--target-sequence B C`和新输出目录；若摆放变化或身份不确定，重新拍照标注。trial02的旧续跑命令仅可从Git历史回溯，不再作为当前默认操作。

仍可用`--target-id A`运行原单次模式，默认不自动归位。单纯加`--cycles 3`不能替代明确目标顺序；顺序模式的周期数由列表长度决定。每轮记录各目标完成的指令周期、归位反馈和当前目标；物理抓对、夹住、单根送达由现场反馈记录。

新入口沿用下述目标模型控制限制；`--transport-age-aligned`复用已有搬运取步方式，未加入夹爪触发补丁或轨迹补偿。首轮因观测年龄512.6 ms超过500 ms停止后，用户授权放宽截图中的时效限制和总运行时间；后续采用上述显式宽容选项。保存图像的通信耗时不能代替现场相机帧龄与控制时序验收。

| 选项 | 行为与边界 |
| --- | --- |
| 不加分离选项 | 按最高分选择当前分割候选；实时位置／方向／面积／重合度／分数拒绝门限已取消，评分不证明身份正确，完全无候选时停止 |
| `--paired-target-separation` | 已有双目标前景分离路径，限两根；保留历史部署 |
| `--multi-target-separation` | 与双目标选项互斥，使用全部可见种子和当前实例；新`target-stack`自动启用，支持剩余3／2／1根；旧`target-pair`同名选项不改变其旧检查点绑定 |
| `--transport-age-aligned` | 搬运按观察年龄取完整动作行，不跨过首次释放边界，到位后才松爪；省略时保留搬运首步行为 |
| `--latency-tolerant` | 两种目标模型的预测年龄与看门狗均为1.5秒，响应等待5秒。`target-stack`另放宽相机帧龄至400 ms、双相机时差300 ms、控制循环及伺服发送间隔700 ms、发送时状态年龄250 ms；旧`target-pair`对应四项仍为250／100／350／100 ms |
| `--max-seconds 600` | 将单次运行上限从120秒延长到600秒；顺序模式为每根600秒，完成后按目标列表继续，最后一根完成停止 |
| `--target-sequence A B C` | 仅用于`target-stack`，与`--target-id`互斥；使用已审核的目标身份顺序，释放后自动归位并开启下一根独立会话，剩余候选按3→2→1缩减 |

预测年龄／看门狗／响应等待沿用旧目标模型宽容配置，相机两项参考旧`mixed-both-orders`；700 ms控制间隔和250 ms状态年龄是堆叠宽容配置，trial04已在该配置下完成；这不证明容限边缘或其他时序条件均已验证。放宽意味着允许更旧的输入及更长的控制停顿，不会提高模型推理速度。名义控制频率仍25 Hz、伺服指令时长仍80 ms，控制器自身的超时行为不由这些Python阈值保证。待夹爪切换目标期限仍1.5秒，关节、速度、工作区和停止收尾逻辑不变。省略宽容选项及总时长选项即回到原默认值；单根模式不变。

历史trial01的`pick1_snapshot/selection.json`及两次运行目录均已存在，仅用于回溯。当前每轮身份和顺序必须匹配本轮新快照与人工导出，使用上面的新目录模板，不复用这些旧输出。

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

已实现 [joint_target.py](../../src/lerobot/bamboo_sorting/policy/joint_target.py)：
显式选择实例 → 当前帧可见掩码 → `prepare_target_images` → `target_global_rgb`＋`grasp_rgb`。
实例PNG的像素ID只用于查找选中实例，不能当类别或跨帧身份。
`target_id`固定表示一次抓放的对象，`frame_id`约束轮廓对应的当前全局帧；这两个字段由调用者提供，接口检查一致性本身不证明跟踪正确。
掩码内侧2像素用RGB紫色`[255,0,255]`描边，被夹爪遮断的可见区域分别描边，不补全不可见部分。
接近阶段目标无当前可见掩码时拒绝生成输入，实时分离路径停止本轮；尚未实现自动重识别。搬运阶段不要求掩码。
人工草稿仅在`allow_draft=True`时允许预览，并返回`preview_only=true`；正常入口要求`reviewed=true`。

关键帧复核原型仍保留`AuboI10JointTargetOutline`；实际训练／数值推理使用独立契约
`AuboI10JointTargetApproach`，图像键为`target_global_rgb`、`grasp_rgb`。
[joint_target_policy.py](../../src/lerobot/bamboo_sorting/policy/joint_target_policy.py)提供共享阶段规则、
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

从[分布入口](../../artifacts/diagnostics/joint_target_live_preparation/ab_outcome_review/wrist_angle/distribution_review/review.html)查看已有结果，在[20帧人工标注页](../../artifacts/diagnostics/joint_target_live_preparation/ab_outcome_review/wrist_angle/distribution_review/manual_axes.html)先看原图，逐帧填写可标注性、遮挡和阶段；可标注时沿竹条长轴点两个相距至少80像素的点，不可标注时写原因，完整导出全部20帧。助手不能代替操作者填写人工结果。

拿到真实导出后，先确认下面目录尚无`measurement_error.json`，再替换实际路径导入；输出使用独占创建，已有人工结果不得覆盖：

```bash
.venv/bin/python examples/phone_to_auboi10/review_joint_wrist_distribution.py \
  --stage errors \
  --output artifacts/diagnostics/joint_target_live_preparation/ab_outcome_review/wrist_angle/distribution_review \
  --annotations "/实际导出的文件路径.json"
```

报告可配对帧的绝对角度差中位数、最大值、人工不可标注数及程序漏测情况。这是程序与一名人工参考的差异，包含双方误差，不是校准真值；样本最大差异不是全数据误差上界。无向轴按180°周期计算差异。

现有643帧来自6条专家示教／19轮实机，程序可测44／138帧，其中15轮实机有可测角度；无效帧保留在时间轴，不能当作没有该状态。独立单位是示教条数／实机轮数，各轮控制配置不同。完整统计、夹指修正及历史验证范围见[腕部证据](EVIDENCE.md#腕部离线诊断与人工复核)，不从帧数推算实验样本量。

[跳变敏感性检查](../../artifacts/diagnostics/joint_target_live_preparation/ab_outcome_review/wrist_angle/distribution_review/jump_sensitivity.html)已完成：同来源相邻程序可测帧间隔≤1.1秒、无向轴角差≥10°，允许跨无效帧；共17对、27个端点。暂时排除两端只是统计敏感性对照，未标为确定错误，原测量不改。数值重叠仍存在，不证明共同状态；按跳变排除可能引入选择偏差。同目录`jump_sensitivity.py`拒绝覆盖已有结果，不为证明运行过而重跑。

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

**主问题已确认（2026-10-22用户同意）**：在限定的竹条堆叠场景中，描边条件化的VLA能否按指定目标完成抓取，并适应位置、角度和支撑高度变化？以“抓哪根／模型服从描边”为主线，以不同支撑高度下的下探与腕角表现作为能力和失败分析，自动清堆作为后续系统目标。目标服从只统计当时至少两根可选上层条的步骤（k≥2）；抓对、夹住、单根、送达及整堆完成情况分别记录。研究方向的确认不代表上述能力已经得到验证。

层高分析保持仅RGB、无深度输入和脚本高度／姿态补偿的动作路线。按清堆顺序抓下层条时上层条已被移走，故不宜表述为“被压条抓得更低”。现有层高证据一正一反：trial03 B闭爪仅比A低0.822 mm并抓空，trial04 B低4.875 mm并成功；21条下层示教估计闭爪Z中位数90.875 mm，见[第三轮对照](EVIDENCE.md#第三轮b抓空与成功下层示教对照)和[第四轮复核](EVIDENCE.md#第四轮三根成功与现场空目标恢复)。分割类别`top_strip`／`covered_strip`不进入动作模型输入（策略只见描边全局图、腕部图和七维状态），层判断须由策略从像素推断；被压条描边断成两段等形状线索是否构成捷径，尚待实验区分。

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
| 2. 三根堆叠（**当前：重复实机实验**） | 三根指定顺序抓放 | 数据／标注和两组训练已完成；当前target-stack连续抓放、空目标恢复、运动参考更新 | trial04已有一轮三根成功；继续逐轮记录物理结果，稳定性和相对无描边基线优势待检验 |
| 3. 自动成败反馈与清堆（后续） | 自动识别抓空、多抓、滑落和清空，再决定续抓／重试 | 磁性开关／DI结合图像、失败后的重新抓取、清空判定；当前只实现指定顺序及中间归位 | 自动物理结果判断和重抓尚未实现；不能把当前空前景重观察当成抓空重试 |
| 4. 长短与语言分拣 | 语言指定放置区 | 长／短×放置区A／B四种示教组合，同一目标只换目的地指令 | 放置行为随指令改变；此处A／B是放置区，不是目标身份 |

2026-10-15起取消原“下探高度修正”阶段：按层高`n×8 mm`在下探段做FK→竖直修正→IK的方案属于脚本混合，不符合纯具身智能方向；堆叠层高由模型从示教中学习，原方案可从Git历史回溯。
A／B成对实验按用户决定不再继续，原[位置／角度分离对照](#目标条件模型单次真机试验)取消；目标服从改在堆叠实验中逐次统计，但只有“当时至少两根可选上层条”的步骤才计入服从证据，并须与同数据无描边基线对照；只有一个可选目标的步骤只验证抓放能力。
VLM选目标、RL、深度相机、架空及不规则支撑暂缓；不为当前问题先换整套架构，前三阶段稳定后再判断是否需要。

### 当前位置与下一步

当前保持`target-stack`最终权重及trial04运行选项，按[下一轮流程](#目标条件模型单次真机试验)开展多轮新摆放实验。先记录逐根物理结果、整轮是否清空、邻条扰动、前景丢失与恢复；保留失败和中断，不只统计成功轮。当前只有一轮三根成功反馈，不预设成功率或将开发试验称为独立泛化测试。

暂不新训练、不切换baseline、不重开已结束的普通单根或A／B位置／角度补录。后续若开展两模型真机比较，先确定可比摆放、目标选择和运行预算；baseline没有显式指定目标，目标服从仍只统计k≥2的步骤。已有成对模型、训练数据及失败证据继续保留。

**推进顺序（2026-10-21确定，2026-10-22确认主问题与阶段安排）**：

| 顺序 | 事项 | 要点 |
| --- | --- | --- |
| P0（已完成） | 确定[主问题](#研究问题与总体架构)与措辞 | 描边指定目标为主线，位置／角度／支撑高度适应为能力分析，自动清堆为后续目标 |
| P1（当前） | 固定配置重复实验 | 冻结代码，沿用`target-stack`及trial04选项；先准备≥5堆摆放清单与结果记录口径，再开展现场实验，逐根记录角色（交叉上层／下层／独立）、闭爪Z、J6、物理结果与k |
| P2 | 同数据无描边基线真机对照 | P1之后安排，服从只统计k≥2步骤 |
| P3 | 第3层与扩展方案 | 依P1结果决定，见[扩展到6～8根](#扩展到68根) |
| 并行 | 腕部20帧人工标注导入 | 支撑“失败能说清原因” |
| 暂缓 | 感知升级离线比较、跟随失败分项记录 | P1期间不改感知链或运行代码，见[待办表](#各阶段待办与约束) |

新增离线指标（动作块夹爪切换时机、按层闭爪位姿误差）仍暂缓；P1所需闭爪Z已由现场日志记录，无需新脚本。

### 扩展到6～8根

目标为每堆6～8根，同层根数／交叉增多与层数增加两方面都要覆盖；排在P1之后，尚未开始。2026-10-21讨论结论：

- **台面**：相机、机器人起点、左侧放置区不动；只在右侧支撑台旁拼接垫块加宽，垫块须与原支撑台等高、外观一致，台面高度变化会整体偏移层高基准。操作者试摆确认原支撑台可逐根叠放6根，但叠放后旁边放不下独立条；下指空间由现场确认。
- **训练覆盖**：69条示教首帧目标全部位于原支撑台（全局图约x 362～589、y 235～429，主轴约60°～146°），见[覆盖统计](EVIDENCE.md#三根堆叠正式采集)。加宽区和超出该范围的位置对现有模型属于未见位置，模型输出绝对关节角，越远越依赖外推，需新示教补充。原台面内的结构可先用现有模型零样本试几堆，按感知错／选错／抓歪／碰动邻条分类失败；新布局摆好后先只拍全局图，与覆盖图叠加核对。
- **布局**：纯逐根叠放每步k=1，且下探高度与抓取步数完全耦合，模型可能只学“剩几根下探多少”。因此采用“原台面主堆＋加宽区独立条或矮小堆”，使多数步骤k≥2，并让相同剩余根数对应不同高度。摆放清单须平衡各层示教数量；第3层及以上在单堆中占比低，需专门安排。超出现有示教高度的层（如逐根叠6层，顶层比现有示教高约30 mm以上）现有模型不能零样本覆盖。
- **示教**：仍为每抓一根存一条，一堆产出6～8条，并自然覆盖剩余8→1根各状态；同堆示教彼此相关，独立样本按堆计，划分仍按整堆。示教偶尔带动下层条，这类示教保留不删；建议夹紧后先竖直抬起数厘米再平移，减少拖动。邻条扰动不增加现场询问，改为离线比较相邻两步首帧种子，估计剩余条位移。
- **标注**：沿用助手生成草稿、用户审核确认，开始记录审核用时。首帧种子传播（如SAM 2）能否提高草稿质量，在阶段切换时评估，不在P1中途替换。
- **连续作业**：整堆成功率约为单根成功率的N次方（单根0.9时，5根约0.59、8根约0.43），本阶段需要自动成败判断与重抓（第3阶段）。邻条被带动超出8像素邻域时身份参考会丢失，跟踪升级在此阶段才成为必要。
- **评价**：可安排数量泛化，例如用≤4根的数据训练、在5根以上测试；测试堆事先指定，不参与开发。

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
| 感知识别 | 保留047～051首轮测试原结果；分析051第二阶段误分类，补充普通结构的独立开发数据。若据此调数据、阈值或模型，泛化成绩需另批未参与开发的测试。升级候选（2026-10-21）：SAM 2.1首帧种子视频传播（同时替代当前帧分割、身份分配和跟踪）、SEA-RAFT稠密光流接原RANSAC、CoTracker3遮挡感知点跟踪；先在已人工确认的逐帧身份掩码上离线比较逐帧IoU、交叉处身份互换、跟踪丢失和延迟，在阶段切换时再决定是否替换。替换理由须来自数据，不以“方法较旧”为据；SAM 2对外观相近的细长物体易混淆身份，需重点检验。 |
| 目标条件化 | 两组训练和离线对照已完成，当前重复target-stack真机试验；尚不能宣称目标组优于baseline。A／B实验结束，旧跟踪诊断冻结。 |
| 可下指区段 | 当前实验由操作者在摆放时保证下指空间，不开发自动区段判断，不把尺寸测量或该模块作为推进门槛；实验结论限定于这一场景条件。 |
| 物理结果反馈 | 磁性开关及控制柜DI（数字输入）接入待现场确认，作为后续自动成败判断的准备，是贡献4及最终演示的前提；当前重复试验先人工判定，不等待自动成败系统。开关未闭合到底不独立证明夹住一根，需结合图像。 |
| 连续逐层抓放 | 指定A→B→C及两次归位已实现，trial04成功；继续记录抓对、单根、送达、邻条扰动与耗时。允许小幅扰动但如实记录后果；当前每根最多3次空前景重观察，不是3次物理抓空重试。自动清空检测、物理重抓与相应停止规则另行设计。 |
| 后续对照与纠错 | 保留同数据无描边VLA对照；脚本抓取仅作后续独立比较，不进入主线。保存完整运行及接管边界，再选择纠错片段训练；数据达到条件后安排数量泛化及消融。 |

`top_strip`不直接等于物理可夹；现有下层误判继续作为感知问题处理，与遮挡后的局部漏分分开判断。
旧双根数据已有两种上下摆放的抓放经验，见[已有模型与数据](EVIDENCE.md#当前双摆放模型)。复用优先从完整的第一次抓放片段开始，补当前帧目标轮廓；原始数据保留，以派生片段和来源映射接入。
旧示教含连续两次抓放，且没有目标元数据；当前目标接口首次闭合后本轮不再描边，不能把整条旧示教直接当作单目标样本。复用需适配片段边界、动作块与目标侧录，保持原训练／验证分组；它不能替代双可取目标A／B对照。
逐根摆放的差分标签仍须复核阴影、曝光与旧条被碰动的影响，历史完整形状不替代当前可见轮廓。
保留现有Mask2Former基线，不因斜条矩形框会重叠就断言其他实例分割方法不可用。局部深度仅在交叉处确有判断困难时再评估；论文贡献和泛化结论须由上述实验建立。

## 历史流程入口

这些记录用于理解已有数据和设计来源，不是当前实验任务。逐批结果只在EVIDENCE维护；整理前完整操作说明可从当前历史的`8bb390f6`读取，回溯命令见[文档历史](EVIDENCE.md#文档整理与历史原文)。

### 同场景A／B成对示教方案

该阶段已结束：78条示教完成目标轮廓接受和数据接口验收，训练66条、验证12条，目标条件模型完成30000步训练。近似配对和额外示教的区分、原始／逻辑场景编号、已接受误差及中断尝试均保留，不能把它们重新解释为严格同观察对照或新的堆叠数据。

| 要回溯的内容 | 唯一证据入口 |
| --- | --- |
| 首对示教与现场复核 | [首对A／B采后检查](EVIDENCE.md#首对ab示教采后检查) |
| 各批录制、轮廓接受、训练与离线评价 | [成对训练数据记录](EVIDENCE.md#成对训练数据采后检查) |
| 统一录制清单与训练来源 | [录制清单](../../artifacts/capture/target_conditioned/joint_target_pair_train_t029_s002_t033/training_capture_inventory.json)、[来源清单](../../artifacts/capture/target_conditioned/joint_target_pair_train_t029_s002_t033/target_training_sources.json) |
| 目标条件模型现场A／B及失败复核 | [现场证据](EVIDENCE.md#目标条件模型最新ab现场结果与b失败复核) |
| 搬运时序修正／新摆放斜条问题 | [搬运停滞](EVIDENCE.md#b夹住后搬运停滞)、[重新摆放复核](EVIDENCE.md#重新摆放后的指定ab抓取) |

### 指定目标输入：离线原型

首批5条单根试采和旧录像诊断已经完成／冻结。当前目标图像契约见[接口约定](#目标条件接口约定)，当前采集标注见[堆叠流程](#三根堆叠目标示教方案)。
旧关键帧、光流、冻结分割及重新确认对照见[离线原型证据](EVIDENCE.md#指定目标输入离线原型)和[旧录像诊断](EVIDENCE.md#旧录像目标输入诊断)。单根成功不证明目标服从；旧光流基线不能生成训练标签，全程搬运轮廓跟踪不作为当前前置条件。

# AUBO i10 项目手册

更新：2026-11-07。本文集中维护当前状态、操作方法和后续计划；实验指标与原始路径只维护在
[证据索引](EVIDENCE.md)。所有命令从仓库根目录 `/home/rentao/program/lerobot-aubo` 执行。

## 目录

- [当前状态与系统边界](#当前状态与系统边界)
- [三根堆叠目标示教方案](#三根堆叠目标示教方案)：数据与训练已完成，准备进入抓取场景验证
- [代码导航与维护](#代码导航与维护)：按任务找入口，区分共享模块与历史工具
- [数据集目录](#数据集目录)
- [人工标注导出目录](#人工标注导出目录)
- [实验产物目录](#实验产物目录)
- [相机配置](#相机配置)
- 静态分割：[采集与人工复核](#静态分割采集与人工复核)、[数据、训练与独立评价](#静态分割数据训练与独立评价)
- 动作模型：[手机遥操采集](#动作模型手机遥操采集)、[训练与离线评价](#动作模型训练与离线评价)、[现场运行](#动作模型现场运行)
- [上方起点的局部纠正示教](#上方起点的局部纠正示教)：采集步骤、批次计划与[任务接手](#局部示教任务接手)
- [目标条件接口约定](#目标条件接口约定)：标注格式、训练／运行共用规则
- [斜条J6欠转的腕部角度诊断](#斜条j6欠转的腕部角度诊断)：只待现有人工标注收束，不阻塞堆叠试采
- [后续研究与待办](#后续研究与待办)：含[推进顺序](#当前位置与下一步)、[自主选目标与逐根清空](#自主选目标与逐根清空)及[扩展到6根及以下](#扩展到6根及以下)
- [历史流程入口](#历史流程入口)：已结束的A／B示教、单根试采与跟踪诊断

先按当前任务进入对应章节。完成记录、历史参数和逐批结果统一查 [EVIDENCE.md](EVIDENCE.md)，不把旧批次的命令当作新实验授权。

## 当前状态与系统边界

当前主线保留已训练的`target-stack`及三根连续抓放方案。三根训练数据、标注和两组动作模型训练均已完成，trial04已有操作者确认的三根成功及真实空目标恢复记录。[六根及以下的静态分割扩展](#扩展到6根及以下)已完成052～060补采复核、首轮混合微调及061～064首次独立测试；061～064获准加入训练后，第二轮混合微调及52张固定验证图对照也已完成。2026-10-24用户确认**静态视觉识别表现达到本阶段预期，抓取场景中的实际效果仍需实验验证**，当前进入[下一阶段抓取验证准备](#当前位置与下一步)。短段漏检和局部实例拆分保留为已知问题，详见[结果证据](EVIDENCE.md#多根静态分割数据链路检查)；暂不继续补采065～069或追加视觉训练。061～064首次测试记录保留，此后属于已见训练样本。工作区最多容纳六根，八根扩展暂缓。旧模型与运行参数保留；新分割模型已接入`target-stack`，真实GPU加载和三次动作返回已确认，检查脚本的本地保存错误与后续现场验证边界见[当前位置与下一步](#当前位置与下一步)。提交与问题状态统一见[进展索引](EVIDENCE.md#提交与实验进展总览)。

**当前安排（2026-11-07）**：已先将原图70000步训练／部署及现场准备相关改动提交为`c403e24d`，未推送。最新原始双RGB模型的`best`与`final`经实际字节核对完全一致，按用户条件授权仅删除重复的`final/model.safetensors`，保留部署正在引用的`best_step_070000`、配置、处理器及全部报告。现已完成[目标轮廓100000步训练准备](#目标轮廓100000步训练准备)，从原始SmolVLA基座重新训练，复用已接受的2px轮廓、整堆划分及原选模规则；CPU预检通过，用户随后已明确“我同意开始训练”；本轮已于2026-11-07 00:36:56（北京时间）在远端启动，启动核对89次真实优化、loss／梯度有限，实际采样及选模配置符合提案。旧模型与实验状态见[原始双RGB重新训练](#原始双rgb重新训练)及证据索引。

**局部纠正模型与实机历史（2026-11-05～06）**：局部纠正示教已完成24堆92条54115帧，最后`014`～`024`的40条22777帧已完整收尾；采集检查及双路关键帧复查通过。用户取消另录验证集，保留原三根3堆，旧四根抽5堆、新局部示教抽4堆，按整堆形成训练65堆237条152192帧、验证12堆44条27819帧。来源、抽取编号及标注入口见[局部示教](#上方起点的局部纠正示教)；本批22904帧目标轮廓及所列误差已获用户接受，92条真实数据接口和严格混合预检均通过。用户明确要求重新训练：从原始SmolVLA预训练权重开始，使用本轮训练集统计，不加载旧AUBO任务权重或处理器。原数据、旧标注及缓存复用，未重复传输原数据。本轮已完成60000步优化和6轮固定监控验证，60000步成为最佳，最佳／最终模型已保存并通过重载。原进程于11月5日04:34在最佳全量复评期间被系统内存不足终止；已按用户授权清理旧全量报告并以分块独立进程于10:39完成27819帧复评，0新增优化步；本轮训练及评估收尾已完成，见[完成核对](../../artifacts/training/stack_local_alignment/reevaluation_completion_check.json)。本轮60000步已按用户授权部署为显式`--target-stack-model local-alignment`，198项CPU检查、新处理器往返检查及3次保存图像GPU服务验收通过。首轮C→A→B已由操作者执行，用户确认三根抓取成功；已保留运行与人工反馈，单轮结果不等于稳定成功率或精对准验收。第二轮A→B→C已实跑，用户反馈“完全抓偏”；程序在B阶段触发碰撞保护，且未确认关闭伺服。第三轮四根实验在A阶段人工中断，用户反馈抓取失败，B/C/D未执行；结束保存状态停稳、伺服关闭、夹爪命令100。第四轮C→D→A→B完成四个指令周期，但用户仍反馈抓取失败、画面向左偏；不据程序周期判定物理成功。当前项目已本地提交`9cb6fe66`（未推送），用户授权从本轮60000步继续30000步，已于21:36正常完成90000步及最佳／最终全量评估；本轮最佳为70000步，固定数据／选模／保存规则不变。原60000步best保留，其重复final权重已按用户授权删除，配置及历史报告保留。用户已授权部署70000步最佳，显式`--target-stack-model local-alignment-70000`已通过保存图像GPU服务验收；两次C前景丢失停止后，只对该入口将分割实例置信度0.5→0.2。最新一轮已完成四个指令周期，用户确认C/D/B成功、A抓歪失败，A后有人工取走／移动A，不能记为自主四根清空。A复核定位到接触前姿态偏差、竹条受扰后未有效纠偏仍闭爪；已获授权完成18次固定权重离线对照；4条纠正试录按用户要求暂停，未采集或训练。不同摆放试验`changed_layout/`已完成，用户确认首根C失败、后续D/A/B成功，且C后有人工取走／移动C；[本轮结果](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/changed_layout/review/outcome_summary.json)记录489份预测、4109条指令、四个指令周期、无程序异常，不记为自主清空。上一轮`changed_layout_repeat/four_pick_run/`实际已在A阶段人工中断，完成C/D两个指令周期，物理成功未确认；原始完成记录保留。2026-11-06用户要求再做四根实验，已准备[本轮手动命令](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/four_pick_20261106_trial01/operator_commands.json)，归位和新拍照已完成，新快照`12906476e8ad4498a2db36eb3428ec31`用户已确认四根轮廓和C→D→A→B，正式选择及本轮四根计划已保存；该轮随后在A阶段触发碰撞保护，退出未确认伺服关闭，用户反馈抓取失败并已重摆。已准备[重摆轮次命令](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/four_pick_20261106_rearranged/operator_commands.json)，操作者已完成归位和拍照，新快照`3ebb464eecef43c396b73cd46e4fad36`用户已确认轮廓和C→D→A→B，正式选择与四根计划已保存，CPU检查通过，等待操作者手动执行；70000步、0.2分割门槛和控制／时序保持，见[局部纠正](#上方起点的局部纠正示教)和[70000步实机复核](#70000步交叉上层A抓偏复核)。默认`three-four`和60000步`local-alignment`保持。默认入口仍为原三根＋四根60000步、2px描边`three-four`模型；不恢复退役候选，π0.5为独立路线。

系统包含感知与动作两部分：实例分割识别每根竹条的可见像素；动作模型根据指定目标轮廓、双路图像和机器人状态预测运动。

| 流程 | 输入 → 输出 | 当前状态 |
| --- | --- | --- |
| 静态 Mask2Former 分割 | 固定全局 RGB → 可见实例掩码、`top_strip` / `covered_strip` | 静态复核达到本阶段预期；第二轮第14轮best已接入`target-stack`，真实GPU加载及三目标动作返回已确认；本地检查脚本保存失败，轮廓质量及抓取仍待现场观察，见[接入状态](#当前位置与下一步)。 |
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

- `scene_id`为堆编号，同一堆各条共用：训练用`p-001`起，验证用`p-101`起，不与历史`t-*`混用。`target_id`为该堆内竹条编号，三根为`s-001`～`s-003`，四／六根相应延伸；不同竹条不共用ID，若失败后再试同一根仍沿用其ID。
- `--stack-demonstrations`自动启用目标示教；每条询问`pick_step`（本堆已保存抓取尝试序号，正整数）和`available_target_count`（当时可选且有下指空间的上层条数1～6）。失败也占一次已保存尝试，未保存的重录不占；同一会话内同堆序号递增，允许不完整堆，不从序号推算剩余根数。尝试序号不再受根数上限约束，支持六根失败后超过六次的人工重试；每次录制仍受`--num-episodes`控制。跨目录续录沿用本堆身份与累计序号，由采集清单和采后核对衔接。自动重试属于后续阶段。文字描述字段仍留空，旧数据无需补填新字段。
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

两组完成验收和离线对照后，原训练跟进已结束；逐关节、逐堆、分阶段和夹爪事件结论见[训练结论](EVIDENCE.md#三根动作模型身份与训练结论)。完成报告与预测保存在既有`training_results/`。2026-10-25用户暂停P1，转入P2同数据有／无描边对照准备，随后扩充四根／六根动作示教；具体顺序和混合训练范围见[当前位置与下一步](#当前位置与下一步)。

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

实时软件的`--multi-target-separation`与旧`--paired-target-separation`互斥，支持1／2／3／4根剩余条的圈选。早期`--model target-pair`的软件准备绑定旧成对检查点及其`artifacts/joint_target_live_service/multi_foreground/`路径，不能代替本次模型。2026-10-18新增`--model target-stack`，绑定已完成训练的堆叠目标模型和独立服务，自动启用多目标分离；现行部署验收、现场快照和自动归位连续抓放流程统一见[真机入口](#目标条件模型单次真机试验)。

合成页面、检查结果及回滚位置统一见[软件准备证据](EVIDENCE.md#三根堆叠软件准备)。[腕部诊断工具](#斜条j6欠转的腕部角度诊断)独立于堆叠采集流程。

## 代码导航与维护

命令入口在 `examples/phone_to_auboi10/`，任务实现集中在 `src/lerobot/bamboo_sorting/`；机器人驱动在 `src/lerobot/robots/aubo_i10/`。先按下表定位，不按文件名猜测当前方案。

任务包按职责分为 `capture/`（采集、相机、时间戳、夹爪证据、采后检查）、`perception/`（分割、目标身份关联、前景分离、运动跟随）、`policy/`（目标图像、训练数据、动作模型适配）、`runtime/`（设备连接、推理通信、执行调度、预览录像）和 `diagnostics/`（腕部角度等离线诊断）。公共契约与路径常量保留在根级 `aubo_joint_contract.py`、`joint_gripper_quality.py`、`joint_stack_metadata.py`、`mask_codec.py`、`model_profiles.py`、`paths.py`、`start_poses.py`。命令入口保留参数解析、计划生成、调用和退出状态；采集连接、持续执行、目标切换与收尾在任务模块中，`run_joint_trial.py`仍保留其单步／归位命令流程。

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
| `record_joint.py` + `capture/fresh_frames.py`、`capture/homing.py` | 当前关节采集的相机新鲜度检查与正常归位；旧`record.py`不再是当前采集依赖，`record_c0_batch.py`已移除 |
| `capture/c0_batch_capture.py`、`capture/c0_gripper_*` | 关节采集的帧时间、夹爪命令与保存证据；不是当前8维末端动作方案 |
| `capture/joint_camera_config.py → capture/c0_smoke_capture.py` | 当前相机配置与冻结映射；配置文件仍在 `configs/aubo_i10/` |
| `run_joint_trial.py`、`runtime/inference.py`、`runtime/station.py` | 命令入口保留[归位入口](#2-归位)；`InferencePipe`和`ReadOnlyStation`在运行模块中供各入口复用 |
| `run_joint_live.py`、`runtime/task_prompt.py` | 保留停稳后单步观察的兼容入口；任务文字读取在运行模块中供持续伺服入口复用 |
| `joint_inference_stdio.py` | 远端模型数值推理协议；由运行链调用，不是本地机械臂控制入口 |
| `start_poses.py`、`move_to_start.py`、`release_joint_gripper.py`、`teleoperate.py` | 起点常量在任务包中统一维护；移动、单独松爪和手机遥操涉及现场动作，须按相应授权执行 |
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
| AUBO采集、相机、夹爪证据、采后检查 | `tests/bamboo_sorting/capture/`；`c0_*`仍覆盖当前共享时间戳、夹爪和相机契约 |
| 目标关联、前景分离、静态分割、摆放复核 | `tests/bamboo_sorting/perception/` |
| 目标图像、训练数据与SmolVLA适配 | `tests/bamboo_sorting/policy/`；`tests/bamboo_sorting/test_joint_target_review_preparation.py`覆盖复核准备流程 |
| 伺服执行、目标影子、录像与归位入口 | `tests/bamboo_sorting/runtime/` |
| 腕部角度与目标策略诊断 | `tests/bamboo_sorting/diagnostics/` |
| 共享假设备与合成数据 | `tests/bamboo_sorting/capture_fakes.py`、`tests/bamboo_sorting/perception_fakes.py`、`tests/mocks/`、`tests/fixtures/` |
| AUBO驱动与手机动作处理 | `tests/robots/test_aubo_*.py`、`tests/processor/test_aubo_*.py` |
| SmolVLA损失、动作补齐与处理器兼容性 | `tests/policies/smolvla/`、`tests/processor/test_smolvla_processor.py`；`test_relative_xyz.py`含旧表示兼容检查，不代表当前采用相对TCP动作 |
| 通用录制、数据集、视频与处理器 | `tests/scripts/`、`tests/datasets/`、`tests/processor/`、`tests/test_control_robot.py` |
| 公共配置、优化器、工具与测试辅助 | `tests/configs/`、`tests/optim/`、`tests/utils/`、`tests/fixtures/`、`tests/mocks/`；`tests/motors/test_motors_bus.py`保留通用总线与假设备支持 |
| 相机或模型集成 | `tests/cameras/test_opencv.py`、`tests/training/`、`tests/test_cli_peft.py`等需先检查具体用例的设备／下载／训练条件，不是默认现场检查命令 |

AUBO离线回归统一用 `make test-aubo-offline`（等价于本手册记录的pytest范围）：只使用假设备与合成数据，禁止下载、GPU、真实模型推理和现场动作；命令显式设置`CUDA_VISIBLE_DEVICES=`，即使主机有GPU，直接指定`cuda`的处理器用例也会跳过。测试缓存写入临时目录并在结束时删除。`make test-aubo-collect`只收集AUBO用例，不执行。GPU／下载／真实模型集成检查单独列出为 `make test-aubo-gpu-integration`，需要相应授权和环境，不属于默认回归。上游CI的`pytest tests`是框架整体检查，不是AUBO验收，也不应据此启动训练、推理或现场操作。

日常小改动按上表选择受影响的测试；涉及多个模块的重构及交付前再跑完整离线回归。例如只修改持续运行流程时，可复用同一命令的隔离与缓存设置：

```bash
make test-aubo-offline AUBO_TEST_PATHS=tests/bamboo_sorting/runtime
```

若同时修改夹爪驱动、模型输入或采集逻辑，应把对应测试路径一并纳入。参数化测试会用多组输入执行同一个测试函数，因此执行项数不等于测试代码份数；代表性组合精简的依据与结果见[测试裁剪记录](EVIDENCE.md#按aubo用途裁剪测试)。

本地共有82个`test_*.py`文件，其中AUBO测试29个；不是所有保留测试都适合直接执行。删除范围及回溯见[测试裁剪记录](EVIDENCE.md#按aubo用途裁剪测试)。

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
| `pile-json--/segmentation/` | 57份静态堆放`pile_XXX_polygons.json`，含已导入的052～060 |
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

**2026-10-29授权清理后的可用状态**：历史三根及目标条件示教的50,706张标注RGB抽帧、采后检查帧、显示用描边及指定静态预览已清理，原始视频／Parquet、当前掩码、实例标签、人工JSON和四根p-021的1299张待标注RGB保留。历史页面可能暂缺图片；再次`verify`或渲染前，按现有`annotations.json`的`image`、`video`和`video_time_s`仅恢复抽帧，可调用`capture.target_review.extract_images`，不能重跑整套`prepare`覆盖已有标注。旧整理的`before/after`快照及传输包也已清理，该次整理不再具备原样回滚材料；具体范围与逐批恢复映射统一见[清理记录](../../artifacts/maintenance/storage_cleanup_20261029.json)。

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
每批最多安排后续5组，052～060补采、061～064首次独立测试及其获准加入训练后的混合微调均已完成；当前补采暂缓，逐组状态见[本批清单](#扩展到6根及以下)。恢复采集前检查原始目录、统一review及下载JSON的组号，并核对脚本参数，不提前假定后续编号可用。

1. 每组清空后独立重摆，可拍空场景后逐根加入（0→N），也可摆好整堆后逐根移除至空场景（N→0），均为N＋1张图；007～051的N为3，后续N按[本批清单](#扩展到6根及以下)确定。保持普通逐层结构，自然变化位置、方向、交叉点。
2. 加入时逐根从上方放下；移除时每次只拿当前未被压住的一根，不能先抽下层。其余竹条不能移动或明显倾斜；相机、承托面保持不动。手退出、场景静止再拍，整根及端部余量留在原图内。
3. 每批集中检查照片，由操作者确认上述现场条件。照片不能完全证明旧条未移动；条件不成立时不能直接传播遮挡真值。
4. 同一摆放的阶段图、衍生图、关联对照保持同一 split（数据集合）。测试组在采集前指定，不按结果重新划分。

以下为**参数模板**，须先填经核对的新组号、集合和本组根数；省略 `--record` 只看计划、不创建目录：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/capture_placement_sequence.py \
  --camera-set original-global --output "artifacts/segmentation/placement_sequences/<新组号>" \
  --placement-id "<新组号>" --split "<train或validation或test>" \
  --strips "<本组根数>" --capture-order "<add或remove>" \
  --roi 280 160 640 480 --min-chroma-change 8
```

拍摄由操作者经授权后添加 `--record`：回车保存各阶段，`q` / Ctrl+C提前结束并保留照片。
`--capture-order add`（默认）从空场景逐根加入；`remove`先拍整堆，再每次移除一根，最后必须拍空场景。
移除模式立即按实际拍摄顺序保存`capture_*.png`及`sequence.json.capture_frames`；完整拍完才倒序生成标注用的`frame_*.png`和差分候选，每帧`source`指回原拍摄记录。中止时只保留已拍原图，不把尚有竹条的末图误当空场景。`--input-images`也按所选模式接收实际拍摄顺序。
入口只打开全局相机，不连接腕部相机或机械臂。ROI（候选提取区域）及色度阈值仅用于差分草稿，模型使用完整640×480原始RGB。
当前模板依据054原图中的端部位置，将候选区域向上扩大40像素；054原图完整，旧候选框截掉的端部在人工标注时按原图修正。新摆放若超出当前ROI，先按新全局图确定候选区域；不能直接沿用旧区域而遗漏新增竹条。机位与模型输入分辨率保持不变。
默认RGB差分阈值25、候选最小面积80；同色交叉和阴影可能导致草稿断裂，不能将差分候选直接当真值。
曝光／白平衡尝试锁定并记录返回值，退出仅恢复成功写入的控制项；`camera_control_lock_unverified` 不代表锁定成功，需检查亮度漂移。
原目录保留 `plan.json`、`sequence.json`、`frame_*.png`、候选掩码与标签，已有目录拒绝覆盖。

### 人工修边与导入

每一步只画本步新增木条的多边形轮廓；移除采集按照片倒序显示，`strip_001`对应最后拿走的一根，ID不是拿走顺序。页面拖点修边，Shift＋点击加点、右键删点；
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
移除模式中，该确认针对倒序照片的等价加入顺序，须由操作者确认实际“先拿上层、其余不动”，并在`placement_confirmation_source`说明移除采集。工具不会因选择移除模式自动确认这些条件。
可见区域断为两段仍是同一根实例。`training_ready=false`、`graspable=null` 保留原始含义：复核并不证明已训练或物理可夹。

## 静态分割：数据、训练与独立评价

### 数据与模型输入

| 集合 | 组号与选择清单 | 组数／图数／可见实例 | 现有导出目录 |
| --- | --- | --- | --- |
| train | 007～046中除下列validation；[选择清单](../../configs/aubo_i10/short_strip_segmentation.json) | 30／120／180 | `datasets/segmentation/short_strip_segmentation/` |
| validation | 011、016、021、026、031、036、041、044、045、046 | 10／40／60 | 同上 |
| test | 047～051；[选择清单](../../configs/aubo_i10/short_strip_segmentation_test.json) | 5／20／30 | `datasets/segmentation/short_strip_segmentation_test/` |

上表为保留的首轮数据；六根及以下扩展另有本地[混合选择清单](../../configs/aubo_i10/short_strip_segmentation_up_to_six.json)和`datasets/segmentation/short_strip_segmentation_up_to_six/`，已导出并通过内置审计，数据组成及微调准备见[当前扩展方案](#扩展到6根及以下)。

每组包含1张空场景；三集合没有组级交叉。已有导出保留，无需重复导出，不为提高指标重划分。
`images/` 是无描边RGB；`instances/` 是16位实例PNG，0为背景、正整数为本帧实例ID；
`dataset.json` 的 `segments` 单独解释类别：0=`top_strip`，1=`covered_strip`。类别0不等于PNG背景0。
只标可见像素，完全不可见实例不构造前景；空图是全零掩码＋空实例列表，不等同于缺少标注。
轮廓、放入顺序与组号是监督和对应信息，不能泄漏到模型输入。

`export_strip_segmentation.py --selection <清单> --output <新目录>` 导出已复核数据；
仅测试清单需显式 `--test-only`，默认仍要求train＋validation。导出会审计确认、尺寸、像素ID、面积及组级划分，拒绝覆盖。

### 训练及环境

`train_strip_segmentation.py` 默认 `--stage preflight`，只审计数据；
默认读取数据目录内`dataset.json`，可用`--manifest-name`指定同一目录中的其他清单，共享图像和掩码而保留旧清单。
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
人工采集等待两路新帧的门槛为0.5秒（原0.2秒），实际值记入`plan.json.fresh_camera_frame_timeout_s`。等待期间不发送下一条动作、不保存重复帧；持续无新帧仍报错停止。相机帧龄检查和策略运行保护保持原值；延长等待可能造成短暂遥操停顿，不表示相机故障已修复。

```bash
.venv/bin/python examples/phone_to_auboi10/audit_joint_dataset.py \
  --dataset-root datasets/single_strip/train/joint_train_run02 \
  --evidence-root artifacts/capture/single_strip/joint_train_run02 --require-gripper-quality
```

审计通过表示契约、时戳、存储及开合标签一致，还需看双路视频确认实际抓放。
增加条目用 `--num-episodes`；验证／测试分别用 `--split validation` / `test`，按独立摆放划分。
数据目录与对应证据一起保留，不重写源路径；相机快照在证据 `plan.json` 及数据 `meta/camera_configuration.json`。
现有动作训练不自动读取静态分割数据或新相机数据；多来源组合使用下节的显式清单。

### 上方起点的局部纠正示教

目标是补充“尚未对准→主动修正位置／角度→对准后夹紧→提起→搬运→释放”的完整局部示教。
这属于局部起点的行为克隆（用人工正确动作训练策略）；只有从策略实际到达的状态接管并聚合专家动作，才直接对应DAgger的数据分布纠正。本入口不自动运行策略或接管。
共同起点是堆上方便于观察、尚有对准工作的位置，不是每根目标已经精确对准的位置；只录后者会漏掉希望模型学会的纠正过程。

`--demo-scope full`为原默认流程。新模式使用`--stack-demonstrations --demo-scope local_alignment`，仍使用原双640×480相机、25 Hz、7维绝对关节动作与原安全处理器，不修改`NORMAL_START_DEG`。
起点由操作者现场遥操决定，不执行固定上方航点；具体安全高度、位置和角度扰动范围需在现场确定。

局部模式的堆编号可直接输入`001`或`1`：按`--split`自动展开为`la-train-001`、`la-validation-001`或`la-test-001`，并在终端显示完整编号。同堆各条继续输入同一数字；完整文字编号仍原样接受。原`full`模式的数字堆编号仍展开为`p-*`，已有数据不改名。

**选哪根竹条**：由操作者从当前未被压住、且有下指空间的候选中指定，程序不会自动随机抽取。多个候选都可抓时，在不同堆之间轮换或随机选择，避免总选最左／最近的一根；只有一个可抓候选时就选它。四根分开时通常可选四根；一根压三根且只有上层可抓时先选上层；两组交叉时可在两根可抓上层条中选择。每次抓走后按实际剩余状态重新判断，`available_target_count`记录当时可抓候选数，不是剩余根数。

选定后，一条示教始终抓同一个`target_id`，同一物理竹条的ID随它保留，不能按第几次抓取重新编号。采后审核该目标在接近阶段的轮廓；训练通过全局图中的目标描边告知模型抓哪根，`target_id`文字本身不作为模型输入。目标身份及是否可抓依赖操作者确认，采集程序不会自动从图像核验。

1. 输入堆编号、实际目标ID、累计保存尝试序号`pick_step`、当时可下指的上层条数`available_target_count`。
2. 按`r`完成原归位，摆好竹条后按第一次`→`，进入**不录制的准备遥操**。此阶段同样会控制机械臂和夹爪，但不向数据集、视频或帧侧录写入样本。
3. 遥操到堆上方的预定起点，夹爪保持张开，松开手机移动按钮，再按第二次`→`。程序停伺服、重置遥操积分基准，从新的真实观察开始正式录制；夹爪命令未张开则继续准备。准备中`←`重新准备，`Esc`取消；`r`仅在前一归位等待阶段可用。
4. 录制精确对准到搬运放下的过程，完成释放后按`→`保存并如实填写结果。正式录制中`←`舍弃本次并重新准备，`Esc`不保存未完成示教。失败尝试另留结果，不作为成功监督；不把策略失败动作当专家标签。

下一根仍先完成原归位，再人工移到堆上方开始新条目；本模式没有自动保存并返回上方起点的功能。同堆不恢复中间摆放，失败后重试同一根沿用目标ID；已保存失败也占一次`pick_step`，未保存的重录不占。若重录前已改变堆内状态，重新核对目标身份和可抓候选数。

`demo_scope`标记整段／局部示教。按用户要求，起点分组填写和`--start-pose-group`选项已移除，新录制不再写入`start_pose_group`；既有数据中的该字段保留兼容，不改写原始记录。局部示教的`initial_state`自动取正式数据首帧`[J1…J6,gripper_pos]`，审计会与Parquet核对；准备阶段状态不冒充录制起点。


**当前针对A抓偏的小批纠正准备（2026-11-06）**：旧24堆92条保持，不将失败策略录像作为专家标签。已扫描92条按初始目标的图像位置／方向／长度检索相近成功样本，最近训练场景为`la-train-021`和`022`，几何相近不等于同一物理布局或失败A的正确动作。用户单独授权后完成6输入×3seed＝18次70000步GPU预测（另2次服务预热），其中4份专家输入是两条训练示教的闭爪前25／5帧，2份失败输入为本轮A的315／325次保存观察。4个熟悉专家输入的前20行六关节MAE为0.076～0.183°；末段失败输入三seed的第10行J6约−128.85／−128.56／−128.81°，接近输入−129.01°，且均预测后续闭爪。此小样本支持下一步检查失败附近状态的专家纠正监督，尚不能证明数据缺口是唯一原因，也不能把相近专家的关节角直接移到失败场景；没有失败状态专家真值，初始朝向与其他关节／深度贡献仍待分离。完整[输入与结果](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/four_pick_relaxed_run/review/alignment_probe/result.json)保留；保存live输入来自JPEG快照，原实际传输为`rgb_zlib`，该离线结果不是原输入精确重放。旧前景门槛CPU对照的“jpeg95匹配现场传输”描述已更正：那是JPEG变体测试，不能视为无损现场输入的精确复现。本轮GPU进程已关闭，0优化步、无硬件访问、权重与控制不变。

已按现有`record_joint.py --demo-scope local_alignment`准备[4条试录计划](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/four_pick_relaxed_run/review/alignment_probe/corrective_pilot_plan.json)，并通过无`--record`入口检查，未创建新数据／采集目录、未连接设备。拟用`datasets/stack/train/joint_stack_local_crossing_upper_recovery_pilot`与对应`artifacts/capture/stack/`目录：只抓交叉上层A，四次重摆分别用`la-train-025`～`028`，每个新场景目标A、首次保存尝试`pick_step=1`，可下指数按现场填写（仅A压B时通常1）。四条分别检查基准对准、偏位修正、偏角修正、组合修正；状态类别只写在计划中，不增加起点分组填写。每次按原流程先归位、进入不录制准备，移至尚未接触竹条且留有可见偏差的安全起点，**开始正式录制后才完成纠正→有效包夹→闭合→搬运释放**。不自动重放失败关节位姿、不故意复现压条／卡条；人工近似复现不称为DAgger接管。4是保存尝试数，失败如实保留且不作为成功监督。先复核这4条的数据和实际纠正行为，再决定扩充及独立布局验证；当前训练清单、划分、选模和保存规则保持，训练须另行授权，四条试录尚未开始，现按用户“先不录制”要求暂停。下一轮改用不同摆放，已在原实验根目录内准备[新摆放操作者命令](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/changed_layout/operator_commands.json)：手动归位后采集新快照，再按实际数量和可抓顺序审核。当前按仍为四根准备，身份／顺序等待新图，不复用旧圈选；模型、0.2门槛及控制参数保持。操作者完成归位、拍照和C→D→A→B确认后，已执行`changed_layout/four_pick_run/`：四个指令周期完成、无程序异常，用户报告C失败、D/A/B成功，C后有人工取走／移动C。详见[结果记录](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/changed_layout/review/outcome_summary.json)，本次未重新诊断C的具体失败机制。上一轮`changed_layout_repeat/four_pick_run/complete.json`现已记录A阶段人工中断，544份预测、4513条指令，仅C/D完成两个指令周期，物理成功未确认；结束时保存状态停稳、伺服关闭、夹爪命令100，录像报告BrokenPipe且不完整。2026-11-06用户要求再做四根，已准备[本轮命令](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/four_pick_20261106_trial01/operator_commands.json)，继续70000步、0.2门槛和原控制／时序，操作者已完成归位与拍照，新快照`12906476e8ad4498a2db36eb3428ec31`用户已确认轮廓与C→D→A→B，正式选择及[四根计划](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/four_pick_20261106_trial01/four_pick_plan.json)已保存，CPU检查较上一实际运行仅快照ID变化（运行时任务文本不参与比较）。该轮实际运行318份预测、2752条指令，C/D完成两个指令周期后在A阶段触发碰撞保护，退出报告伺服仍开启，未保存最终状态；用户反馈抓取失败并已重新摆放，逐根物理结果未确认。详见[本轮结果](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/four_pick_20261106_trial01/review/outcome_summary.json)。下一轮[重摆命令](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/four_pick_20261106_rearranged/operator_commands.json)已准备，操作者已完成归位和拍照，归位通过、拍照时碰撞状态检查通过且夹爪打开；新快照`3ebb464eecef43c396b73cd46e4fad36`四根可见草稿非空且不重叠，A为右侧斜向上层、B为近竖直下层（两段可见表面）、C/D为左侧独立斜条，用户已确认“轮廓正确，按 C→D→A→B”，`human_reviewed=true`正式选择与[四根计划](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/four_pick_20261106_rearranged/four_pick_plan.json)已保存；CPU对照上一实际运行仅快照ID变化（不比较运行时补入的任务文本），运行输出尚不存在，等待操作者手动执行；旧快照与选择不复用，70000步、0.2及控制参数保持。助手未连接硬件或启动新推理，原运行记录保持，示教补录和训练仍未启动。新布局的表现用于继续定位泛化范围，不视为旧交叉配置已修复。

局部示教已全部录完：24堆92条54115帧，覆盖下表七类摆放。原始数据均仍位于`datasets/stack/train/`，采集时的`split=train`保持不变；训练／验证的实际用途由[新清单](../../artifacts/training/stack_local_alignment/sources.json)中的`selection_scenes`指定，不搬移、复制或重写原数据。最后40条来源为`joint_stack_local_remaining_train_014_024`，所有来源与检查入口见[证据](EVIDENCE.md#局部起点纠正示教的软件准备)。

| 摆放类型 | 已采堆号 | 总堆／条 | 本轮局部验证堆号 |
| --- | --- | --- | --- |
| 三根：交叉两根＋独立一根 | `021`～`024` | 4／12 | `024` |
| 四根分开 `four_separate` | `001、014` | 2／8 | — |
| 一对交叉＋两根独立 `crossing_pair_two_separate` | `015`～`018` | 4／16 | — |
| 两组独立交叉 `two_crossing_pairs` | `003、011`～`013` | 4／16 | — |
| 一根压三根 `one_over_three` | `002、008`～`010` | 4／16 | `010` |
| 一根压两根＋一根独立 `one_over_two_one_separate` | `019、020` | 2／8 | `020` |
| 两根压两根 `two_over_two` | `004`～`007` | 4／16 | `007` |
| 合计 | `001`～`024` | 24／92 | 4堆15条8626帧 |

用户明确不再另录验证集，原新增7堆27条计划取消，不执行该旧命令。验证按初始堆及其全部后续剩余状态整体抽取，绝不随机拆帧或拆同堆条目：

| 数据来源 | 训练：堆／条／帧 | 验证：堆／条／帧 | 验证身份 |
| --- | --- | --- | --- |
| 原三根整段 | 20／60／40673 | 3／9／6023 | 保留`p-101、p-102、p-103` |
| 原四根整段 | 25／100／66030 | 5／20／13170 | `p-027、p-032、p-038、p-043、p-050` |
| 新局部纠正 | 20／77／45489 | 4／15／8626 | `la-train-007、010、020、024` |
| 合计 | 65／237／152192 | 12／44／27819 | 整堆无交叉 |

该选择使合并验证覆盖七类摆放，少量类别的验证堆数仍有限。本轮从原始预训练权重重新学习AUBO任务，12堆验证均不进入本轮任务训练及归一化统计。旧四根5堆曾被旧60000步AUBO模型训练见过，这一限制影响与旧模型的对照，不能把旧模型在这5堆的表现称为未见数据泛化。[划分记录](../../artifacts/training/stack_local_alignment/split_plan.json)保存每堆及跨目录来源，不改原ID。

`001`跨两个来源组成四条，原前三条来源的`capture_complete=false`保留；`003`原目录含`two_over_two`，用户确认实际是两组独立交叉，按实际分类计数。`011`～`013`第二条的可抓数以`recording_review/metadata_corrections.json`从1修正为2，标注与训练统一通过`reviewed_stack_metadata`读取，不改原session／侧录／审计。`022、024`先移除独立条后只剩交叉组，第二条可抓数1与关键帧一致，不追加修正。

本批22904个接近帧均已有目标轮廓，首次接受闭爪后继续使用原图搬运。用户授权的冻结分割推理已完成，优化步数为0；旧数据、标注和模型原地复用，新来源先盘点再只上传缺失原视频／Parquet／侧录。RGB在远端按实际视频偏移提取；本机浏览使用JPEG派生图，训练仍读取原视频。用户已明确接受整批现有轮廓、409帧RGB补绘及所列误差，逐帧接受记录保留问题说明，不宣称逐像素真值。当前状态统一见[训练准备入口](../../artifacts/training/stack_local_alignment/review_index.html)和[准备完成记录](../../artifacts/training/stack_local_alignment/pretraining_readiness.json)；全部真实数据接口及混合预检已通过；本轮真实训练及授权复评均已完成，部署及保存图像GPU服务验收已完成，当前场景已审核，真实抓取由操作者手动启动。

#### 局部示教任务接手

接手快照为2026-11-04：工作目录`/home/rentao/program/lerobot-aubo`，分支`research/depth-vla-phase-a`，HEAD为`cb4c0802`（局部示教修改前的本地检查点，未推送）。先读根目录[AGENT.md](../../AGENT.md)、本节完整计划和[EVIDENCE中的局部示教证据](EVIDENCE.md#局部起点纠正示教的软件准备)，再核对当前`git status --short --branch`与真实产物；新文件及操作者反馈优先于此快照。

- **保护工作区**：局部准备／录制、数字堆编号、采集等待与混合训练改动尚未提交。`src/lerobot/bamboo_sorting/capture/joint_local_preparation.py`、`src/lerobot/bamboo_sorting/joint_demo_metadata.py`、`tests/bamboo_sorting/capture/test_joint_local_preparation.py`仍为未跟踪文件，其余改动见Git差异。保留现有修改，不重置工作区；数据和实验产物被Git忽略，代码提交不是数据备份。必要时沿`record_joint.py`→`capture/joint_local_preparation.py`／`aubo_joint_capture.py`→`lerobot_record.py`阅读实际录制逻辑。
- **当前数据**：九个局部来源已完成24堆92条；最后40条在`joint_stack_local_remaining_train_014_024`。不要再次执行录制命令、删除该目录或创建后缀目录；早先清理只涉及当时0条正式保存的中断残留，不能用于当前完整数据。原新增验证录制已取消。
- **划分与标注**：使用`artifacts/training/stack_local_alignment/sources.json`及`split_plan.json`；整堆选择允许同一原目录被两集合以互不重叠场景引用，原采集split不改写。先读`review_index.html`、`pretraining_readiness.json`、`all_source_interface_check.json`及`combined_preflight_check.json`，再按需查看各来源`target_annotations`。轮廓已获用户接受，92条接口和混合预检均完成；复用完整推理结果、已传原数据及本轮远端预检，不从头重跑。
- **原始证据**：`001`跨来源及原中断标记、`003`实际分类、`011`～`013`三条人工可抓数修正均保留。菜单／开合周期／存储检查不能代替物理成功证明；未保存重录不计正式监督，未知物理字段保持未知。
- **授权边界**：本批冻结离线标注推理、目标轮廓接受及严格CPU预检已完成；用户已授权GPU空闲时启动本轮训练，已于2026-11-04 19:41后台启动并完成训练和复评；2026-11-05部署与真机实验已获授权，后续已按用户明确指令停止Ollama，并通过保存图像GPU服务验收；本轮场景和顺序已审核，助手尚未操作硬件。按用户更正采用[重新训练方案](#局部示教混合训练)，从原始SmolVLA预训练权重及新训练统计开始，不加载旧60000步AUBO权重或处理器。已授权配置和完整远端参数在`training_launch_proposal.json`，重新训练配置检查见`retraining_preparation_check.json`。实际启动、日志和退出状态见`training_launch.json`／`training_process.json`，硬件仍由用户手动操作。

## 动作模型：训练与离线评价

入口为 `train_joint_smolvla.py`，使用 [七维关节契约](#当前状态与系统边界)。
历史 `mixed-both-orders` 已完成微调和离线验收；当前主线为2px目标条件`three-four`。新局部示教使用独立候选目录，保留当前权重；实际模型身份见 [证据索引](EVIDENCE.md#三根动作模型身份与训练结论)。

### 阶段与默认值

| `--stage` | 行为 |
| --- | --- |
| `plan`（默认） | 只显示参数，不加载模型、不写结果 |
| `preflight` | 审计来源、筛选成功示范、核对划分、计算训练统计及文件哈希 |
| `smoke` | 本地真实模型的合成数据前向、反向和严格保存重载；零优化器更新 |
| `train` | 数据预检后训练，保存最终模型，重载检查并做离线评价 |
| `evaluate` | 加载已有检查点及其处理器，在相同来源上评价 |

默认 `device=cpu`、`steps=10000`、`batch-size=8`、`seed=42`；
模型一次预测50步，每步7维；各历史模型的实际步数以对应产物为准，不把入口默认值当作当前60000步模型的训练记录。
最终检查点始终保存，`--checkpoint-steps`可指定额外中间步数，没有隐式恢复优化器的接口；从旧权重继续时新建优化器。训练输出必须是全新目录。
本轮另用`--validation-every`启用周期选模，只保留严格验收的最佳及最终模型；此模式不允许同时指定`--checkpoint-steps`保留其他权重。旧的显式中间检查点流程保持不变。
`--learning-rate`可显式覆盖峰值学习率。旧`--preclose-loss-weight`与`--fit-observations`属于历史实验，当前入口不提供；不要复制旧实验参数启动本轮。

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

### 局部示教混合训练

原25个训练来源、3个验证来源的清单保留在`artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/target_training_sources_local.json`。本轮清单为`artifacts/training/stack_local_alignment/sources.json`，远端映射为同目录`sources_remote.json`，引用旧来源及九个新局部来源，不复制数据本体。

可选字段`selection_scenes`是该原目录中参与当前集合的完整场景ID列表，预检拒绝空／重复／不存在的场景、同场景跨集合或同原目录选择重叠；跨目录续录仍按共同场景ID检查泄漏。原始split保留，预检另记`effective_split`、`capture_splits`及选择依据。只有显式按整堆选择时允许从原train来源划作验证；省略字段的历史来源继续要求采集split一致。同一轮预检缓存原来源审计与指纹，避免因共享目录反复计算。旧项省略`demo_scope`即为`full`，新增局部项显式为`local_alignment`；预检核对实际侧录，不能只改清单冒充局部示教。

本轮抽样与归一化选项：

- `--local-demo-fraction 0.5`：指定整段／局部各占50%。默认`--sampling-mode weighted_random`保留历史的组内均匀、有放回加权抽样；它不保证每轮覆盖全部起始帧，也不保证每个batch各半。未传比例时沿用所有帧随机打乱。
- `--sampling-mode full_coverage_local_oversample --local-demo-fraction 0.5`：每轮先放入全部原训练样本索引各一次，再从局部纠偏索引池均匀有放回补采，整体打乱，使两组每轮使用次数各占50%。比例不得低于局部样本原有占比，避免以“过采样”名义丢弃原样本；整数数量向上取整，实际组占比写入`training_sampling.json`。私有seed生成器每轮推进，不在每轮重置。该选项仅改变起始帧索引，不增加或重新拼装轨迹。
- **动作块边界**：现有多来源目录通过`ConcatDataset`路由全局索引，各子数据集仍由原始`LeRobotDataset`在自身episode内读取未来50行动作。不能从打乱后的相邻样本拼成未来动作；不足50行时只重复本episode末行动作并保留`action_is_pad`。一个batch可以包含不同episode的独立样本，单个动作块不能跨episode；同一episode内可以自然跨越对准、夹紧、搬运阶段。
- `--normalization-source train_data`：本轮重新训练使用此模式，从已选65堆训练数据计算状态和动作统计，建立配套归一化／反归一化，验证数据不参与。`train_only_stats.json`和`data_preflight.json`保存训练统计，`normalization.json`记录实际使用值。抽样比例与归一化来源独立，50%局部抽样同样支持重新训练。既有`base_checkpoint`模式只供另行明确的旧AUBO检查点续训，保持该旧模型的处理器含义；本轮不选它。

`preflight`先检查来源和整堆划分并生成`training_sampling.json`，不加载模型。新清单需要重新审计，不能复用仅含旧25个来源的`--prepared-data`。本轮预检已含新整堆划分、全部目标接口和新训练统计，与模型初始化方式无关，重新训练可直接复用。验证集不做均衡抽样，`validation_by_demo_scope.json`分别报告整段和局部的离线动作指标，`validation_by_source.json`保留来源细分；这些均不能替代物理抓取评价。

本轮[严格混合预检](../../artifacts/training/stack_local_alignment/combined_preflight_check.json)已通过：训练／验证整堆无交叉，所选训练统计与独立检查一致，九个新来源92条54115帧的目标接口全部通过。[重新训练配置检查](../../artifacts/training/stack_local_alignment/retraining_preparation_check.json)在CPU确认原始基座和缓存存在、真实训练参数可解析、新处理器使用所选训练统计，动作极值归一化再还原的最大误差为0；未加载旧AUBO权重或处理器，没有加载策略模型、使用GPU或操作硬件。完整`data_preflight.json`保留在远端本轮`artifacts/training/stack_local_alignment/preflight/`，启动参数已引用该文件，无需再次传输原数据或重复计算内容指纹。先前针对旧处理器的比较报告保留为准备历史，不是本轮实际归一化配置。

[训练启动提案](../../artifacts/training/stack_local_alignment/training_launch_proposal.json)已填写原始`lerobot/smolvla_base`缓存、VLM缓存、隔离源码快照、数据清单及预检文件路径，并保存完整`train_argv`，已按用户“空闲时我们可以开始训练”授权于2026-11-04 19:41（北京时间）后台执行；[实际启动记录](../../artifacts/training/stack_local_alignment/training_launch.json)保留PID、启动前GPU占用和环境，`training_process.json`记录进程状态／最终退出码。本轮从原始SmolVLA预训练权重重新训练AUBO任务，不随机重建视觉语言骨干，不加载旧AUBO任务模型、旧处理器或优化器；使用本轮训练集归一化。本轮采用60000步预算，峰值学习率`1e-4`、batch 8、seed 42；按用户授权，改为每10000步验证并只保留最佳与最终模型。保持原模型结构、冻结配置、2px描边、50步动作块及七维绝对关节动作。训练预算及启动已获用户授权，输出为独立`outputs/joint_stack_local_alignment_retrained`；[启动检查](../../artifacts/training/stack_local_alignment/training_start_check.json)已确认真实优化器完成140步、loss／梯度有限，实际统计和均衡抽样正确；该报告为启动快照，后续进度以远端输出的`training.jsonl`和`artifacts/training/stack_local_alignment/training.log`为准。运行模型与旧实验保留，运动／相机时效检查不变。

**本轮检查点与选模规则**：`--validation-every 10000 --validation-points-per-phase 8 --validation-seeds 42 43 44`。原验证集、已接受轮廓和归一化统计不改。固定[验证采样计划](../../artifacts/training/stack_local_alignment/checkpoint_selection_plan.json)已在CPU准备，覆盖12堆44条、2540个观察；各条各阶段均匀选点，并包含闭爪／释放前19、10、1帧、事件帧及后1、10帧。周期验证在训练进程中离线调用模型，未启动真机策略执行。

1. **检查时机**：10000／20000／30000／40000／50000／60000步；最后一步即使不是整万也检查。三个固定种子使用相同观察顺序和批大小，取平均分并报告种子间标准差。验证恢复训练的随机数状态、训练模式及推理队列，不改变后续训练抽样或噪声。
2. **主指标越小越好**：起动、接近和闭爪前阶段，比较预测动作块第0～19行与专家真实动作的六关节平均绝对误差，单位为度，尾部补齐行不计。先在条内按阶段等权，再按堆内条目、各堆平均，最后让整段／局部各占50%。这样长视频、停留段及数量较多的整段示教不会掩盖局部对准问题。阶段按专家观察时刻划分，未来20行动作可能跨过闭爪事件。
3. **防退步条件**：主指标必须严格降低，且闭爪前六关节误差、最差堆闭爪前误差、闭爪前J6误差、搬运／释放关节误差、整段与局部各自的接近误差、闭爪与释放时序误差均不能明显恶化。上限为`max(历史已接受最佳值×1.20, 历史已接受最佳值＋绝对容差)`；关节容差0.05°，事件时序容差1帧（25 Hz下40 ms）。历史参考取历次已接受最佳的最小值，不能每次退步20%不断累积。事件时序比较完整50步块的首个事件，漏事件按该窗口的真实长度罚分，不能用“没预测事件”获得零时序误差。这些是离线筛选容限，不是实机抓取精度标准。
4. **仅供诊断**：验证loss及完整预测块的闭爪／释放事件错误数不直接排序或否决候选；完整块尾部可能被后续重规划替换。分数相同保留较早模型。各次指标、护栏阈值及接受／拒绝原因存于`validation/step_*/`和`validation_history.jsonl`。
5. **保留与删除**：仅改善且通过上述条件时，先写入本轮`checkpoints/best_step_*.incomplete`，在CPU严格重载模型及配套处理器，逐张量确认权重与处理器统计一致，再原子发布为完整最佳并更新`best_checkpoint.json`；之后才删除带本轮所有权记录、且由旧最佳指针指定的旧最佳权重包。不改善只存评估记录，不存候选权重；保存或重载失败保留旧最佳。历史模型、基座、数据和其他目录不在删除范围。

正常完成后保留一个`checkpoints/best_step_*`和`final/`；即使二者来自同一步，也保留两个完整模型包。最终模型沿用严格保存重载检查，并对最佳与最终分别完成全部44条27819帧的固定seed 42离线验证，结果在`full_validation/`；其差异单独报告，不用这次复评偷偷改写固定监控选出的最佳。当前实现不自动早停，`final`表示预算结束时的模型，不表示最好；离线最佳也不等于实机成功率最高；本轮部署已获授权，验收状态见下方现场准备。

**本轮收尾与内存隔离复评（2026-11-05）**：60000步固定监控主误差0.290547°、闭爪前误差0.176384°、闭爪／释放时序综合误差11.53075／4.89205步，验证loss 0.0422369，全部防退步条件通过。原进程在完成最终模型全量报告后，执行最佳模型全量复评时被系统OOM终止（04:34:39，退出码−9；匿名内存约56.4 GiB），原失败状态和日志保留，不能把原进程标记为正常退出。用户授权清除中断残留并复评；仅删除远端`full_validation/final.json`以重算，其旧分数和50个保护文件的大小／mtime记录在[清理证据](../../artifacts/training/stack_local_alignment/reevaluation_cleanup.json)，原始数据、已完成首行动作评估、周期监控、权重和优化器状态未删。新增复评脚本以每2048帧一个独立进程顺序执行，batch 8、seed 42，跨分块保存并恢复Python／NumPy／Torch CPU及CUDA随机数；只积累拥有独立存储的紧凑数值数组，避免持有原批次内存。78项CPU检查通过，包含跨分块与整段预测逐值一致、批次释放检查，见[软件检查](../../artifacts/training/stack_local_alignment/reevaluation_software_check.json)。最佳与最终的全部权重／配置／处理器逐文件一致，见[等价检查](../../artifacts/training/stack_local_alignment/model_equivalence.json)，因此只预测一次并为两者生成对应报告，不重复GPU计算。分块复评于09:58启动、10:39:35完成，耗时40分51秒；14个分块覆盖12堆44条全部27819帧，工作进程峰值RAM为3.654 GiB，全部分块结束后已生成`complete.json`、最佳／最终与比较报告，0新增优化步。全量seed 42复评主误差0.294291°、闭爪前误差0.175791°、闭爪／释放时序综合误差12.65561／8.48026步、验证loss 0.041622；这是完整帧集单seed的结果，不能与2540观察三seed固定监控的分数混用。旧全量final摘要中的所有原有指标均被逐值复现，最佳／最终报告一致，见[完成核对](../../artifacts/training/stack_local_alignment/reevaluation_completion_check.json)、[全量最佳](../../artifacts/training/stack_local_alignment/full_validation/best.json)、[全量最终](../../artifacts/training/stack_local_alignment/full_validation/final.json)及[比较](../../artifacts/training/stack_local_alignment/full_validation/comparison.json)。60000步最佳、最终与用户单独保留的30000步模型均在；原训练进程OOM失败事实保留，本轮完成来自随后成功补齐的评估收尾。不改原固定监控选出的最佳、不部署或操作硬件。

**用户另行指定保留的30000步模型**：2026-11-04 23:24（北京时间），按用户“30000步模型可以给我另存一份吗，不要被后面出现更好的被删除”要求，已在远端另存完整模型包至`/home/rentao/program/lerobot-aubo-joint-stack/outputs/joint_stack_local_alignment_retrained/retained_checkpoints/step_030000`。共14个原始文件、1676555283字节，含权重、配套处理器、归一化、数据说明及选模／严格重载记录；原件与副本逐文件字节比对一致，副本使用独立文件，不是软链接或硬链接。本目录在自动清理的`checkpoints/`之外；后续新最佳仍可替换其原`best_step_030000`，独立副本保持，只有用户明确要求删除时才清理。正常自动保留最佳／最终之外额外保留此份，不改变选模规则、不自动部署、不含优化器续训状态。细节见[独立保留记录](../../artifacts/training/stack_local_alignment/retained_step_030000.json)。

**60000步后继续30000步：已正常完成，最佳70000步（2026-11-05）**：用户在第四轮抓取失败、画面向左偏后要求继续训练，先完成本地提交`9cb6fe66`，未推送；提交时983项离线检查通过、11项跳过。随后新增显式`--resume-training-state`支持，原入口默认行为保持；续训准备代码目前为该提交之后的未提交改动。完整离线回归985通过／11跳过，50项针对性检查含假模型连续步数、优化器／学习率恢复、较差候选保留原最佳及旧模型不删；不代表真机改善。

唯一准备入口为[续训提案](../../artifacts/training/stack_local_alignment/continuation/training_launch_proposal.json)，[CPU准备检查](../../artifacts/training/stack_local_alignment/continuation/preparation_check.json)已通过。本轮从`best_step_060000`加载权重与原处理器，显式恢复同轮`training_state.pt`中的60000步优化器／调度器状态；`--steps 30000`表示额外更新，累计至90000。当前学习率已降至`2.5e-6`，保持原60000步余弦调度的末值，不自动重启预热或回升至`1e-4`；若后续另选升高学习率，应明确作为不同设置，不能称为原样恢复。旧状态未保存数据迭代位置及完整训练随机状态，数据读取从seed42重新开始，因此不是逐位一致的断点恢复。继续相同65堆237条训练、12堆44条验证、50%整段／局部抽样、batch8、2px、50×7绝对关节动作，复用原清单和预检，不新增失败策略动作作为示教。

固定监控在累计70000／80000／90000步执行，2540观察、seed42／43／44及全部主指标／护栏保持。新候选从原60000步主误差0.290547°和历史最小护栏参考开始比较，不能因进入新目录自动接受更差的首个候选；不改善则最佳继续指向受保护的原60000步。改善且通过严格重载后，仅删除**本次新输出目录**内被替代的旧最佳。续训自动清理不删除原60000步最佳或独立保留30000步；原final重复权重随后按用户单独授权删除，详见下段。新一轮正常收尾仍保存最佳和final，并分别做全量验证。全量评估保存的NumPy数组改为拥有独立存储，减少因持有批次底层Tensor而积累内存；输出指标不变；本次最佳及最终模型均已完成真实全量评估、进程退出码0，没有再次因收尾评估中断。未持续记录整段内存峰值，不将采样内存值当作峰值。

远端隔离准备目录为`artifacts/training/stack_local_alignment/continuation/`，独立输出为`outputs/joint_stack_local_alignment_continued`，本次产物已完成；只传准备源码约4.6MB，旧数据、权重及环境原地复用。CPU实际核对155份优化器状态、原处理器动作往返误差0、选模覆盖／种子及历史护栏一致，未重复全量哈希、加载策略权重或使用GPU计算。[当前指标摘要](../../artifacts/training/stack_local_alignment/continuation/reference_metrics.json)区分固定监控和全量验证；这些是示教动作一致性，不是现场抓取成功率，也不能保证续训消除图像左偏。用户已明确“然后我们可以开始训练”，[启动授权](../../artifacts/training/stack_local_alignment/continuation/training_authorization.json)已保存；2026-11-05 16:51:23（北京时间）启动监督进程及训练子进程，初始PID为3027303／3027306，启动核验已确认累计60148步／本次新增148步、loss与梯度有限、学习率`2.5e-6`、原最佳／优化器状态未变，见[启动检查](../../artifacts/training/stack_local_alignment/continuation/training_start_check.json)；该数值保留为启动快照。本轮于21:36:10（北京时间）正常结束、退出码0，30000次新增更新及三次固定监控完整，最佳为70000步、final为90000步，两个模型均保存且通过严格重载，见[完成核对](../../artifacts/training/stack_local_alignment/continuation/completion_check.json)。当前GPU无计算进程，未部署或操作硬件。

**续训评价结果与保存位置**：三seed固定监控主误差在70000／80000／90000步分别为0.281276°／0.287472°／0.286575°；70000步接受为新最佳，80000步主误差未改善且闭爪前J6触发护栏，90000步护栏通过但主误差未改善，不替换最佳。独立的27819帧、seed42全量复核中，70000步最佳主误差0.285033°、闭爪前误差0.169868°、loss0.0418343；90000步final对应0.289928°、0.179127°、0.0423795。final的全量loss来自同一完整评估流程的`validation.json`，其`full_validation/final.json`未单列loss；原报告不补写。全量结果不改写三seed固定监控选模，二者口径不混用；全部报告已同步到本地[结果目录](../../artifacts/training/stack_local_alignment/continuation/results/)。远端本轮只保留`checkpoints/best_step_070000`和`final`两个模型包，以及90000步`training_state.pt`。原60000步best和独立30000步模型仍在，重复原final权重仍已删除。训练及离线指标完成不证明现场左偏已解决；70000步随后已获授权部署并通过保存图像服务验收，尚无新增实机评价。

**70000步最佳与原`three-four`60000步：同输入离线对比已完成（2026-11-05）**：用户明确比较更早的`joint_stack_three_four_target_smolvla/final`，不是本轮局部重训60000步。原模型只有6023帧旧三根验证，本轮12堆验证中的五个旧四根堆曾被`three-four`训练见过；不能直接比较历史总分。本次按双方训练清单排除这五堆，保留p-101／102／103及la-train-007／010／020／024，共7堆24条，复用每阶段8点及事件邻帧规则选1385个观察；同一数据对象、相同索引／顺序、batch8、seed42、2px、50×7动作，各自使用配套处理器。两模型保存的状态、专家动作、padding逐值一致，共新增2770次观察预测、0优化步，权重大小／mtime未变，GPU进程已退出；未改部署或访问硬件。[配对结果](../../artifacts/training/stack_local_alignment/continuation/three_four_comparison/comparison.json)及[逐堆比较](../../artifacts/training/stack_local_alignment/continuation/three_four_comparison/scene_comparison.json)为固定证据入口。

本次共同观察的接触前主误差（旧→新）为0.398117°→0.277878°（−30.2%），闭爪前误差0.230123°→0.160260°（−30.4%）、闭爪前J6误差0.652668°→0.363772°（−44.3%）、最差堆闭爪前误差0.411406°→0.210485°（−48.8%）、搬运误差0.308578°→0.274418°（−11.1%）。局部纠正接触前误差0.578944°→0.325683°（−43.7%），四个局部堆均改善；原三根整段误差0.217289°→0.230073°（+5.9%），三个原三根堆均退步。闭爪事件综合时序误差9.91785→11.34517帧（+14.4%），释放为6.74864→5.66047帧（−16.1%）；事件指标含窗口内漏事件惩罚，不是实机控制延迟。总体主指标按整段／局部各50%平均，新模型不是所有指标都更好。各自归一化统计不同，不用归一化loss的直接差值推断物理改善。此为单seed配对诊断，不能作显著性或成功率结论；四个局部起点场景不能替代完整四根现场评价；数据、统计和权重谱系都不同，不能将差异归因于多训练一万步。保留原`three-four`作对照，70000步作为局部纠正候选，尚无修复实际左偏的物理证据；未改变原固定监控选模。

**按用户授权去除重复60000步权重**：已重新逐文件确认原`checkpoints/best_step_060000`与`final`的权重、配置和处理器一致；部署和续训均引用best。只删除远端`outputs/joint_stack_local_alignment_retrained/final/model.safetensors`，约1.12GiB，保留该目录全部其他配置、侧录、评价及历史日志，原final不再是可单独加载的模型包。保留best、`training_state.pt`和独立30000步模型，保护文件大小／mtime／inode未变；需要恢复原final时，从best复制同名权重即可。[删除记录](../../artifacts/training/stack_local_alignment/continuation/duplicate_model_cleanup.json)保存字面路径、删除前后空间及授权原话；原最佳／最终等价报告和复评结果作为删除前历史证据保留，不重写原始评价。

### 原始双RGB重新训练

本轮按用户确认，从远端已经缓存的原始SmolVLA权重重新学习AUBO任务，不加载已有60000／70000步AUBO权重、旧处理器或优化器。原始全局RGB、原始腕部RGB、七维状态和固定任务文字进入模型，输出50×7绝对关节／夹爪动作；模型不接收轮廓、目标身份或抓取顺序。现有训练入口与原目标模型继续保留，硬件控制及当前默认模型不变。

- **数据与采样**：沿用原整堆划分，训练65堆237条152192帧，验证12堆44条27819帧。完整示教160条106703帧、局部纠正77条45489帧；训练抽样的概率质量各占50%，不是删掉一半数据，也不保证每个batch恰好各半。验证不重采样。已有标注及原始数据不修改，刚才失败的模型动作不加入专家示教。
- **输入准备**：新[来源清单](../../artifacts/training/stack_raw_rgb/sources_remote.json)仅移除标注路径，来源／顺序／整堆选择不变。`--stage preflight --audited-target-data … --audited-target-manifest …`显式将原已审计结果转换成原图视图，沿用来源证据和所选训练统计，移除`target_frames`等会启用描边的字段；这不是一次新的全内容哈希审计。原图视图有独立契约和预检文件，普通无轮廓训练拒绝残留目标输入。所有45个来源抽查90个真实样本，原图双路及50×7动作接口通过；重新计算的训练统计一致，CPU处理器动作极值归一化往返误差为0，见[数据接口](../../artifacts/training/stack_raw_rgb/raw_dataset_interface_check.json)和[处理器检查](../../artifacts/training/stack_raw_rgb/processor_check.json)。
- **训练配置**：按用户最终授权70000步、batch 8、seed42、峰值学习率`1e-4`，保留既有骨干冻结／动作专家及状态投影配置；25Hz动作数据、50步动作块及10步生成保持。归一化仅由本轮所选成功训练示教计算，建立原始双RGB的新处理器。原数据和模型缓存复用，未重新上传原数据或复制权重。
- **验证与保存**：每10000步对固定观察使用42／43／44三种种子评估。原图监控直接读取Parquet和场景元数据，2540个固定观察、44条、12堆与原轮廓训练完全一致，见[监控核对](../../artifacts/training/stack_raw_rgb/validation_monitor_check.json)。沿用前20个未来动作行的按scope／场景／episode／阶段宏平均接近关节误差及原防退步项，保留严格重载后的最佳和最终模型，仅替换本轮拥有的旧best。模型自由选择另一根有效目标可能增加示教一致性误差，离线最优不等于实机最优；实际抓取、邻条扰动和自主清空必须另行现场验证。
- **准备与授权**：隔离源码及结果入口为`artifacts/training/stack_raw_rgb/`，本轮输出为远端`outputs/joint_stack_raw_rgb_smolvla`。[准备就绪记录](../../artifacts/training/stack_raw_rgb/pretraining_readiness.json)保留CPU预检完成时的未授权快照；用户随后明确“我们训练70000步，选模规则跟之前一样不变，我们可以开始训练”，已登记[本轮授权](../../artifacts/training/stack_raw_rgb/training_authorization.json)。[启动提案](../../artifacts/training/stack_raw_rgb/training_launch_proposal.json)及[预算核对](../../artifacts/training/stack_raw_rgb/authorized_budget_check.json)确认软件计划只从60000改为70000步，选模指标／防退步／验证观察与种子全部保持。训练已于2026-11-06 11:56:58（北京时间）后台启动，[启动检查](../../artifacts/training/stack_raw_rgb/training_start_check.json)验证PID／进程身份和实际180个优化步、有限loss与梯度；[启动记录](../../artifacts/training/stack_raw_rgb/training_launch.json)保存命令和环境，[进程记录](../../artifacts/training/stack_raw_rgb/training_process.json)由远端监督器在结束时写退出状态。本地JSON是拉取时快照，实时进度查远端`training.log`及输出目录`training.jsonl`。训练准备时1002项CPU离线回归通过、11项跳过；该检查保留为启动前历史，当前训练和部署状态见本节后续。原目标60000步与本轮预算不同，旧70000步续训模型的学习率历程也不同，不将后续差异单独归因于去轮廓。

**70000步部署与原图四根实验准备（2026-11-06）**：优化及全量评估均已完成，`checkpoints/best_step_070000`为固定监控最佳，三seed固定指标见[70000步对照](../../artifacts/training/stack_raw_rgb/comparison_with_target_70000/comparison.json)。最佳严格CPU重载以及最终模型重载预测逐值一致检查通过；全量评估于20:02:11（北京时间）正常结束、退出码0。最佳／最终模型各27819帧、seed42全量主误差均为0.290338°，验证loss均为0.0440937；全量诊断不替代固定三seed选模，原始报告见[结果目录](../../artifacts/training/stack_raw_rgb/results/completion_check.json)。用户已授权部署，新增`--model target-stack --target-stack-model raw-rgb-70000 --raw-rgb`，使用原图训练契约和配套处理器，直接输入`global_rgb`／`grasp_rgb`，不走旧模型临时去轮廓的适配。远端服务为`artifacts/joint_stack_raw_rgb_live_service/joint_inference_stdio.py`，复用本轮训练源码快照及已有权重，不修改原服务。旧`local-alignment-70000 --raw-rgb`对照入口保留，新旧入口、原图像素、四周期／三次归位及归位失败停止共381项CPU运行层检查通过；[参数核对](../../artifacts/training/stack_raw_rgb/deployment/software_settings_check.json)确认25Hz、速度、时序和保护门槛与上一轮原图实验一致。一次性后台验收等现有评估退出后，于20:02:41完成3次保存图像GPU预测，50×7动作均有限，同seed动作逐值一致；往返约0.247～0.252秒，原始契约、无目标及无分割接口检查通过，见[GPU验收](../../artifacts/training/stack_raw_rgb/deployment/gpu_worker_check/complete.json)和[部署记录](../../artifacts/training/stack_raw_rgb/deployment/deployment_receipt.json)。验收进程已退出；此为保存输入服务验收，现场相机新鲜度与物理抓取仍待实验验证。

本轮目录为`artifacts/live/stack/stack_raw_rgb_70000_first_trial/`，[操作者命令](../../artifacts/live/stack/stack_raw_rgb_70000_first_trial/operator_commands.json)已准备归位、可选快照、无设备计划检查和四根执行；助手未启动硬件。操作者已完成归位，`home/complete.json`显示通过且无异常；GPU验收已通过，无设备计划与原四根计划一致。单独`scene_snapshot`可省略：它只作场景留证，不需圈目标、生成selection或给顺序，也不作为运行模型输入；正式实验本身记录双路视频。归位后未再移动机械臂时，摆好四根、手退出并待场景静止，即可由操作者执行`four_pick_run`。正式抓取使用每轮实时双RGB，由模型选目标，程序按四个开合指令周期及三次轮间归位组织；周期数不证明物理成功。出现漏抓／意外扰动需由现场操作者停止并反馈，不以自动计数替代人工结果。

### 目标轮廓100000步训练准备

用户要求参考Git历史配置，恢复目标轮廓并重新训练100000步，选模规则保持；原先等待最终授权，现已获用户明确“我同意开始训练”并正式启动。已先提交当前项目`c403e24d`，未推送；历史依据为`9cb6fe66`的局部示教混合训练。目标输入模块、SmolVLA适配器及`SELECTION_RULE`与该历史提交逐项核对一致，详见[历史配置核对](../../artifacts/training/stack_target_outline_100k/git_history_configuration_check.json)。

- **初始化与数据**：使用远端已缓存的原始`lerobot/smolvla_base`，从0步学习AUBO任务；不从原图70000步或旧轮廓任务权重继续。复用原[轮廓来源清单与已审计预检](../../artifacts/training/stack_local_alignment/sources_remote.json)，训练65堆237条152192帧、验证12堆44条27819帧；归一化沿用这些未重复训练示教的专属统计，验证不参与。按用户随后确认启用`--sampling-mode full_coverage_local_oversample --local-demo-fraction 0.5`：152192个原始训练起始帧每轮各一次，再额外抽取61214个纠偏索引，共213406个独立样本，整段／局部各106703次。batch 8每完整轮26676个更新步（末batch为6），100000步完成3个全覆盖轮并进入第4轮。原数据、已接受标注与缓存原地复用；没有按过采样后的次数重算归一化。
- **目标轮廓的含义**：同帧目标可见掩码生成向内2像素紫色边界，只在接近阶段写入全局RGB（`target_global_rgb`）；腕部图保持原图，首次实际接受闭爪后的搬运阶段恢复原始双RGB。目标身份用于绑定轮廓，A／B／C及顺序不作为额外数值输入。输出仍为50×7绝对关节／夹爪动作，数据25Hz、生成10步、骨干冻结及动作专家设置沿用历史。
- **训练预算**：总计100000次新优化，batch 8、训练seed 42、峰值学习率`1e-4`、PyAV；沿用1000步预热＋余弦衰减的方法，总调度长度为100000步，终值`2.5e-6`。预算延长会改变同一步数的学习率；与旧60000步轮廓、70000步原图或低学习率续训模型比较时，不能把差异只归因于目标轮廓。
- **选模与保存**：完全沿用[局部示教混合训练的主指标与防退步规则](#局部示教混合训练)，每10000步检查，共10轮；固定2540个观察、每阶段8点、seed42／43／44，主指标严格改善且所有原护栏通过才替换best，loss仅诊断。不自动早停，仅保留严格重载后的best及final，并保留所有评估报告；新训练的自动删除只作用于新输出拥有的被替代best。全量复核维持44条27819帧、seed42，不改写固定监控选出的best。[规则核对](../../artifacts/training/stack_target_outline_100k/selection_rule_preservation_check.json)已通过。
- **准备验收**：45个来源180个真实样本检查通过，覆盖各来源接近首／末及搬运首／末；轮廓内边界、搬运原图以及腕图／状态／动作／补齐标记不变均核对通过。86409个接近帧引用的已接受轮廓文件存在；复用原审计，未重复全量内容哈希或逐帧重新审核。训练专属统计与原图视图一致，新目标处理器动作极值往返最大误差0。目视抽查完整／局部训练的两张目标输入图，不当作全视频质量或实机验收。见[输入接口](../../artifacts/training/stack_target_outline_100k/target_dataset_interface_check.json)、[处理器](../../artifacts/training/stack_target_outline_100k/processor_check.json)及[准备就绪](../../artifacts/training/stack_target_outline_100k/pretraining_readiness.json)。原准备的197项CPU检查保留为历史；采样更新后58项相关CPU回归通过，含原始读取器的非连续episode／多来源动作块边界检查及训练／选模／续训编排。生产数据采样边界核对见[采样与episode检查](../../artifacts/training/stack_target_outline_100k/sampling_boundary_check.json)。以上准备验收仅使用CPU；随后正式训练的启动状态见下段。

唯一[启动提案](../../artifacts/training/stack_target_outline_100k/training_launch_proposal.json)已填写完整`train_argv`。用户最终授权已保存至[本轮授权](../../artifacts/training/stack_target_outline_100k/training_authorization.json)，源码提交为`76736c46`，未推送；训练前实际比对远端采样模块、训练入口和监督脚本内容一致，GPU无计算任务、磁盘约49.7GB，输出尚不存在。于2026-11-07 00:36:56（北京时间）启动单一后台训练，训练PID `4065549`、监督PID `4065515`；独立准备目录为远端`artifacts/training/stack_target_outline_100k/`，输出为远端`outputs/joint_stack_target_outline_100k_smolvla`。新增训练使用GPU，硬件执行与部署未授权；不自动重启或额外启动第二轮。

[启动核对](../../artifacts/training/stack_target_outline_100k/training_start_check.json)已确认进程起始身份、89次真实优化及全部已记录loss／梯度／学习率有限；运行中的配置与审阅计划逐项一致，采样确为每轮213406项、原训练统计保持、2540观察及三seed选模保持。89步是启动检查时快照，首次验证尚未到10000步；不能据启动loss判断精对准或物理成功。本地[启动记录](../../artifacts/training/stack_target_outline_100k/training_launch.json)及授权／提案已拉取，实时进度读取远端`artifacts/training/stack_target_outline_100k/training.log`或输出目录`training.jsonl`。监督器结束时写`training_process.json`及退出码；保留先前准备就绪／授权门检查中的未授权字段作为当时快照，不代表当前仍未授权。

### 通用数据预检示例

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

### π0.5 三根与四根训练准备

**当前状态（2026-11-02，训练已中止，按用户决定不再重启）**：本轮FP32 expert训练最后记录27481／30000步（91.6%），原进程以CUDA错误退出。GPU机器已由用户重启；基础CUDA探针通过，本次启动未查到新的Xid／AER错误，但尚不能据此证明长时稳定或真机抓取通过。用户决定只保留本轮21000步best，并授权删除独立副本及可再生成的训练缓存。清理已完成，best权重、处理器、tokenizer、指针，以及全部原始日志／评价／预测／源码审计均保留；详见[退役与清理记录](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/pi05/fp32_retraining/training_retirement_cleanup.json)和[EVIDENCE](EVIDENCE.md#π05训练准备)。

用户随后决定暂缓π0.5排查，当前成果已保存为本地提交`81938577`，新实验切回[原60000步SmolVLA](#三根四根模型试用准备)；以下π0.5模型、失败事实及诊断结论保留。

**本轮保留模型**：远端根目录为`/home/rentao/program/lerobot-aubo-pi05`，检查点为`outputs/pi05_joint_three_four_fp32_expert/checkpoints/best_step_021000`，约8.63GB。其保存前1000步平均训练loss为0.00812535，监测验证loss为0.01595757，阶段平衡接触前TCP主误差为7.3188mm。监测仅含三根场景644个专家观察，不是6023帧完整复核，也不是抓取成功率。最后27000步候选主误差虽为6.6393mm，但闭爪时机9.8849步超过best的20%退步约束，未替换或保存权重；27481步也没有独立检查点。没有生成`final/`、`training_state.pt`、`reload_check.json`、`full_validation/comparison.json`或训练`result.json`。当前状态见[status.json](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/pi05/fp32_retraining/status.json)，原训练退出事实保留在[process.json](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/pi05/fp32_retraining/process.json)。本轮best已按用户要求作为显式实验选项`pi05-fp32-21000`部署，严格重载及原644观察复核通过；6023帧完整复核尚未完成，现场默认仍为`three-four` SmolVLA。本轮真机结果见下段。

**21000步best当前部署与首轮准备（2026-11-02）**：显式使用`--model target-stack --target-stack-model pi05-fp32-21000 --target-outline-width-px 2`，直接读取上述best原件，不新增权重副本。推理入口为远端`artifacts/pi05_fp32_21000_live_service/joint_inference_stdio.py`，复用本轮隔离源码`artifacts/pi05_fp32_retraining/src`及项目自己的`.venv/bin/python`；按保存的处理器、训练集q01/q99和精度约定恢复FP32动作专家，以BF16计算。实际SSH握手已核对模型SHA及精度契约；重载后同seed42／batch16的644观察完整动作块与训练原记录逐值相同，保存图像的分割／描边／搬运协议也通过。具体证据见[EVIDENCE](EVIDENCE.md#π05训练准备)，这些检查不证明现场抓取成功。

首轮新目录为`artifacts/live/stack/pi05_fp32_21000_live/`；[操作者命令](../../artifacts/live/stack/pi05_fp32_21000_live/operator_commands.json)准备三根场景仅抓A上层一个闭合／释放周期，不自动归位。沿用上轮2倍限速（10°/s、40°/s²、0.10m/s）、2px、默认异步重规划、预览／运动跟随、搬运按观察年龄取行和原宽容时序。[软件计划](../../artifacts/live/stack/pi05_fp32_21000_live/software_settings_plan.json)确认相对于`pi05-fp32-128`仅模型身份、推理入口与源码路径变化；使用历史已审核种子验证软件配置，snapshot明确为`pending_new_scene`，不能据此执行。操作者已完成新归位和快照`146dd0cdce734a539db656d73723bae7`；按此前明确委托由助手对新图圈选并保存[selection.json](../../artifacts/live/stack/pi05_fp32_21000_live/scene_snapshot/selection.json)：A右侧近竖直交叉上层、B斜向下层两段可见区域、C左侧独立条，记录委托来源，不声称用户逐多边形审核。[圈选检查](../../artifacts/live/stack/pi05_fp32_21000_live/selection_check.json)、[真实单目标计划](../../artifacts/live/stack/pi05_fp32_21000_live/single_pick_plan.json)及[新图GPU协议检查](../../artifacts/live/stack/pi05_fp32_21000_live/current_scene_protocol/result.json)通过；抽查模型输入紫色描边对应A。操作者随后已实际执行`single_pick_run/`并人工中断，结果见下段；该准备计划及命令转为历史记录，本轮快照和输出目录不再复用执行。现场动作由操作者手动运行，原默认模型和历史模型选项保留；回退时在新输出目录选择`--target-stack-model three-four`。部署及保存图像检查由助手完成，未训练或直接访问硬件；[准备记录](../../artifacts/live/stack/pi05_fp32_21000_live/experiment_plan.json)和[重载检查](../../artifacts/live/stack/pi05_fp32_21000_live/reload_check.json)保留本次范围。

**21000步首轮真机结果：目标选择观察正确、对准与连续性仍未通过**：用户反馈“找准了目标，知道要抓谁但是没有对准”，并确认“没有夹起，只碰推或空抓”、接近及闭爪后两个阶段均不连贯。实际114次已处理预测、1068条伺服指令、约42.82秒；21.66秒接受一次闭爪，未释放、未完成周期，随后人工中断，仅尝试A。指定单根抓取记失败；目标选择正确是本轮操作者观察，不扩大为普遍目标服从或物理成功。原结果`physical_grasp_success=null`与`failure=null`保持。

[本轮CPU复核](../../artifacts/live/stack/pi05_fp32_21000_live/review/findings.json)、[闭爪图](../../artifacts/live/stack/pi05_fp32_21000_live/review/close_alignment_detail.png)和[关节／连续性曲线](../../artifacts/live/stack/pi05_fp32_21000_live/review/trajectory_and_joints.png)显示：抽查11次接近描边仍位于A；闭爪前约29ms的六关节反馈距模型闭爪目标最大0.1254°，腕图仍偏离理想夹持对准。1068条目标均与记录的选中模型行完全一致，raw／decoded六关节差0；接近新块切换J1／J6目标最大变化4.0756°／3.7560°，13次待闭爪重规划中7次重新改为张开，控制指令速度反复换向。95%指令记录间隔不超过45.35ms，最大81.24ms；证据支持优先排查模型目标变化与动作块衔接，对准问题另查末段姿态／视觉输入／主动纠偏，尚未确定唯一原因。平滑运动本身不证明对准恢复。双路各1084帧完整解码，原收尾BrokenPipe与`complete=false`保持。末段布局已变化；结束记录停稳、伺服关闭、夹爪命令100，实际卡条／持物仍须现场确认后再恢复。保留21000步best、不重启训练；本次复核无新GPU预测、硬件操作或控制参数修改。

**连续性排查与行保持候选（真机未通过，已退出试用）**：用户要求继续排查并尝试解决后，已对本轮114份保存观察冻结图像／状态／原掩码，完成249次零更新GPU预测。同一输入seed42重复输出逐值一致，改变种子会改变关节及闭爪时机；单独固定seed42没有改善控制回放的速度变化与换向，未作为修复启用。发送时序中唯一81.24ms间隔位于闭爪IO切换周期，其他最大54.25ms；推理响应与40ms伺服周期分开运行，不修改IO读回或时效／停止检查。具体诊断、取样与限制见[assessment.json](../../artifacts/live/stack/pi05_fp32_21000_live/continuity/assessment.json)和[EVIDENCE](EVIDENCE.md#π05训练准备)。

本地新增显式`--hold-action-until-replan`，仅允许`--model target-stack --target-stack-model pi05-fp32-21000 --transport-age-aligned`：新预测到达时仍按观察年龄选择一条完整7维模型动作行，随后保持至下一份新预测；继续异步观察和到达前重规划，不等待到位才观察，不固定取第0行，不平均关节或夹爪。它排除`--observe-after-arrival`／`--execution-paced`组合；原模型动作、精度、速度限制和新鲜度／停伺服检查保持，默认不开启。省略此参数即恢复原来的块内按年龄逐行推进。相关全量离线回归936通过／11跳过，覆盖保持期间过期拒绝、异步替换、完整行／IO一致及原运行路径。

在同一保存观察与原取行／阶段时间表的CPU回放中，仅启用行保持后，指令速度变化绝对值累计量接近下降34.6%、闭爪后下降58.2%；J6指令速度换向（0.5°/s死区）分别56→18、70→29。[对照图](../../artifacts/live/stack/pi05_fp32_21000_live/continuity/overview.png)展示趋势；这不是物理平顺性或对准通过。动作进度也会变化，回放名义TCP相对原命令位置差95百分位约21.06mm（不是目标误差），43条夹爪候选与原记录时间表不同，尚未重现其新闭环IO时机。随后在`artifacts/live/stack/pi05_fp32_21000_hold_replan/`准备同best／同2倍限速、仅A一次的真机对照；[软件计划](../../artifacts/live/stack/pi05_fp32_21000_hold_replan/software_settings_plan.json)保留当时待定snapshot，[真实计划](../../artifacts/live/stack/pi05_fp32_21000_hold_replan/single_pick_plan.json)绑定新快照`551f97aaf97a468b92c1b8a2acb724b7`并确认行保持开启。操作者完成新归位／拍照后，助手按既有明确委托重新圈选三根身份，[选择检查](../../artifacts/live/stack/pi05_fp32_21000_hold_replan/selection_check.json)及[新图GPU协议检查](../../artifacts/live/stack/pi05_fp32_21000_hold_replan/current_scene_protocol/result.json)通过；具体准备证据见EVIDENCE。

**行保持真机结果：运动稍稳、对准更差、未夹起**：操作者已实际执行并人工中断，反馈“运动稳定了点，但是抓不准，比第一次还差”“没有夹起，只碰推或空抓”。实际110次已处理预测、1031条指令、约41.34秒；22.96秒接受一次闭爪，未释放、未完成周期，仅尝试A，B／C未尝试。候选未通过，已退出后续真机试用；[操作者命令](../../artifacts/live/stack/pi05_fp32_21000_hold_replan/operator_commands.json)和[实验状态](../../artifacts/live/stack/pi05_fp32_21000_hold_replan/experiment_plan.json)已转为历史记录，旧场景及输出目录不可复跑。代码实验开关保留且默认关闭，后续π0.5对准诊断省略`--hold-action-until-replan`使用原选择方式；现场默认`three-four`、21000步best及不重训决定保持。

[本轮CPU复核](../../artifacts/live/stack/pi05_fp32_21000_hold_replan/review/findings.json)和[闭爪图](../../artifacts/live/stack/pi05_fp32_21000_hold_replan/review/close_alignment_detail.png)显示，闭爪前A已被碰偏／转动；抽查12次接近描边可见部分仍在A，遮挡与交叉下的逐帧分割正确性仍未证明。闭爪前约29.54ms的关节反馈距所选闭爪目标最大0.1377°，目标与完整模型行一致，但现场仍抓偏，说明执行到目标不等于目标对准竹条。保持策略改变了动作时间进度；两次摆放及推理噪声不同，[跨轮对照](../../artifacts/live/stack/pi05_fp32_21000_hold_replan/review/comparison.json)不能将抓偏唯一归因于该开关或某一关节。双路各1046帧完整解码，原中断收尾BrokenPipe及`complete=false`保持，不能据其认定抓偏原因。结束记录停稳、伺服关闭、夹爪命令100，恢复前现场核对卡条／持物；下一重点是接触前的位置／姿态以及碰偏后的主动纠正，不能只按运动变化指标选方案。回退包仍位于原试验`continuity/before_runtime_patch.tar`，本次复核未新增GPU预测或操作硬件。

**进一步对准排查与零训练修复尝试**：用户要求继续排查后，已核对观察时间／动作取行及两轮接近轨迹，并从成功三根上层示教检索相近摆放的p-006、p-012、p-019，保存[腕图并排对照](../../artifacts/live/stack/pi05_fp32_21000_hold_replan/review/alignment_diagnosis/approach_comparison_grasp_rgb.jpg)和[诊断入口](../../artifacts/live/stack/pi05_fp32_21000_hold_replan/review/alignment_diagnosis/findings.json)。成功样本在闭爪前可见竹条已位于夹口中间；两轮失败在此前数秒仍偏斜，单纯跟到模型目标或换动作行不能保证纠偏。两轮起点的局部相机参照一致，未发现新的跨轮机位漂移，但不据此排除训练／现场完整成像差异。另用9条验证示教的81个固定观察、三种相同噪声初值，对比原10次与20次去噪，0训练更新；20次在三个种子下的近闭爪位置和姿态误差均退步且更慢，未采用。[成对结果](../../artifacts/live/stack/pi05_fp32_21000_hold_replan/review/alignment_diagnosis/inference_step_probe_result.json)及指标定义见EVIDENCE。当前保留原10次去噪、21000步best和原选行动作路径；未再修改运行代码或安排真机复跑。抓偏尚未解决，也尚未分离模型拟合、视角、随机性与接触时序的独立影响。下一项所需证据是当前机位／限速下同摆放的人工正确接近与纠偏对照，采集及训练仍须分别授权，不用固定关节或几何补偿代替模型修复。

**规则修复历史**：最初一轮因旧best规则误用0步模型的零次额外夹爪切换作独立约束，在2419步按授权停止并清除权重，小型证据保留于`aborted_run_evidence.tar.gz`。修复通过98项CPU测试后从官方基座重新训练，原PID为`1617217`，不接续旧2419步。随后在6066步将`release_event_errors_per_window`改为诊断项，18234步同样处理`close_event_errors_per_window`，两次均未重建训练；本地99项CPU测试及独立CPU替换／状态保持／回滚验证通过。闭爪／释放时机与运动约束仍有效，实际19000／20000／21000步依次更新best，旧best按规则移除。18000步及后续未保存的候选权重不能从评价报告恢复。原始`plan.json`、两次源码回退包、[闭爪应用审计](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/pi05/fp32_retraining/close_rule_live_apply.json)及`source_manifest.json.runtime_close_rule_update`均保留；远端墙钟未校准，历史启动／停止时间只作日志原值，当前检查采用独立UTC时钟。

**原30000步训练与部署结果**：已完成最终保存和严格重载；`best`为29000步，`final`为30000步。两者均完成全部6023帧三根验证，六关节平均绝对误差分别为0.25917°／0.26580°，三四根SmolVLA为0.13523°。完整动作块中的短促闭爪更少，但精细关节拟合仍落后，指定目标服从和四根泛化尚未证明。29000步已作为显式实验选项部署；默认与2倍限速的两轮三根A→B→C真机试验均在A接近阶段由操作者中断，用户确认放宽限速后仍与首轮相同、现场效果不如SmolVLA，均未完成闭爪或任何目标。当前为轨迹异常待解决，尚未通过真机验证。口径见[能力与部署证据](EVIDENCE.md#π05训练准备)和[原训练完整验证对照](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/pi05/deployment/comparison.json)。

**29000步部署与本轮操作**：在现有`--model target-stack`上显式增加`--target-stack-model pi05-29000 --target-outline-width-px 2`。检查点是独立项目的`outputs/pi05_joint_three_four/checkpoints/best_step_029000`，SHA-256为`03468b3f0e18229fd8b1d66cf6e815c414575e04f371c8c6969d9976d06127d6`；服务位于同项目`artifacts/pi05_joint_live_service/`，使用自己的`.venv/bin/python`和隔离源码。直接读取已有权重，不新增权重副本，不改原训练源码或SmolVLA服务。严格加载保存的PI05处理器、训练集q01/q99及检查点自身BF16/FP32混合精度，不追加autocast。默认模型仍为`three-four`；显式选回`--target-stack-model three-four`可恢复原模型，另选新运行输出。

本轮目录为`artifacts/live/stack/pi05_29000_three_pick/`，精确命令保存在[operator_commands.json](../../artifacts/live/stack/pi05_29000_three_pick/operator_commands.json)，[准备记录](../../artifacts/live/stack/pi05_29000_three_pick/experiment_plan.json)现已绑定操作者新快照`521060c44c9d407fadb9925e798aa7c0`。本轮用户明确委托助手标注并保存：A为中间近竖直交叉上层，B为左下至右上的斜向下层、只标两段可见区域，C为最右侧独立条；[selection.json](../../artifacts/live/stack/pi05_29000_three_pick/scene_snapshot/selection.json)记录委托审核来源。[选择检查](../../artifacts/live/stack/pi05_29000_three_pick/selection_check.json)和[真实场景计划](../../artifacts/live/stack/pi05_29000_three_pick/three_pick_plan.json)通过，计划确认29000步、A→B→C及剩余身份3→2→1；这是执行前的准备记录。随后试验失败，末段保存图像显示摆放已变化，本轮标注不可直接用于复跑。先完成下述轨迹诊断，再按[现场三根流程](#目标条件模型单次真机试验)为新的现场状态重新拍照和标注。保留默认异步重规划、预览、运动跟随、搬运按观测年龄取行、原宽容时序和运动限制；两次目标间归位，C结束后停止，不加几何补偿或脚本抓放。A未移走仍压住B或场景被带乱时须现场停止，指令周期完成不是物理成功。

[部署回放](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/pi05/deployment/protocol_replay/complete.json)中54份固定输入预测与离线结果逐值相同；保存场景的A/B/C分割、描边、搬运原RGB路径均通过。预热后含网络往返的接近响应约0.33–0.37秒、搬运约0.29–0.30秒，每目标重新加载约32秒。数字来自保存图像，不含现场相机采集和机器人时序；历史快照不能作为本轮场景。控制程序仍只在接受闭爪指令后切搬运阶段，模型预测闭爪本身不触发该切换。

**本轮起步轨迹异常**：[原始运行记录](../../artifacts/live/stack/pi05_29000_three_pick/three_pick_run/complete.json)包含81次预测、777条伺服指令、约31.10秒执行；结束原因为`operator_interrupt`，A/B/C均未完成，最终伺服关闭。逐条核对执行目标等于对应模型输出行，六关节没有额外解码变换；检查点、起点和相机特征顺序一致，首帧A有2354个前景像素。空目标发生在请求81，晚于用户报告的起步异常。先完成38次保存JPEG冻结预测，再补充27个正常示教起步观察×两模型×3seed共162次预测，优化步和硬件访问均为0。现已确认π0.5在正常示教输入上也有起步动作幅度、尤其抬升不足；本轮现场轨迹很早便缺少示教中的抬升，原始模型后续动作已要求下降。控制器能复现原指令，限速会影响后续轨迹，但不足以单独解释起步异常；固定输入改取第0行也未恢复抬升，不是已验证修复。J4须与J2、J3联合解释，不能单独当作腕部姿态错误。具体数值、几何核验与边界统一见[EVIDENCE](EVIDENCE.md#π05训练准备)、[诊断图](../../artifacts/live/stack/pi05_29000_three_pick/diagnosis/approach_dynamics/overview.png)和[结果](../../artifacts/live/stack/pi05_29000_three_pick/diagnosis/approach_dynamics/conclusion.json)。当前场景仍没有专家动作真值，视角、模型拟合与闭环误差的独立因果贡献尚未完全分离；不直接归因于30000步过拟合，也不以放宽限速、几何补偿或直接换模型作为修复。后续候选须先比较起步多行动作的空间误差和当前速度限制下的轨迹，再另行安排新场景真机验证。

**2倍限速对照已运行，仍未通过（2026-11-01）**：用户确认本轮效果仍与首轮相同、现场推理不如SmolVLA。目录为`artifacts/live/stack/pi05_29000_limits_double_three_pick/`；[实际运行结果](../../artifacts/live/stack/pi05_29000_limits_double_three_pick/three_pick_run/complete.json)为`operator_interrupt`，81次预测、767条指令，仅尝试A，未闭爪、无完成目标，B/C未独立尝试。结束保存状态为停稳、伺服关闭、夹爪0，不代表当前硬件状态。实际`plan.json`确认`pi05-29000`、2px、快照`327cad5ffa2348809f7563be8f6168dc`和`motion_limit_scale=2.0`。显式`--motion-limit-scale 2`仅用于`target-stack`，把关节参考速度／加速度／工具中心速度从5°/s、20°/s²、0.05 m/s改为10°/s、40°/s²、0.10 m/s；不加参数或设为1恢复默认。轨迹生成器与SDK使用同一倍率，归位参数和原停止检查不变。没有加入几何补偿或脚本抓放。本次摆放与首轮不同，不能当作严格同场景速度因果对照。新录像双路各785帧均完整解码；原侧车`complete=false`和BrokenPipeError仍作为中断收尾记录保留。`experiment_plan.json`、`three_pick_plan.json`和`operator_commands.json`是本轮执行前的准备记录，不能再据其“待执行”状态判断尚未实跑，也不可复用旧场景继续执行。

**全链路排查与精度修复准备**：数据标签、训练q01/q99、关节与相机特征顺序、部署处理器和执行取行未发现错位；主要发现是动作专家BF16权重直接由AdamW更新，小更新可能被舍入，以及第0行选模指标与未来执行行不匹配。修复代码及CPU检查已完成：新训练默认`--precision fp32-expert`，冻结视觉语言主干并保留其原精度，可训练动作专家／投影采用FP32，计算按需BF16；检查点新增`aubo_joint_precision.json`（权重存储与计算精度约定），加载前先恢复FP32，避免保存后的细小变化在重载时又被舍入。旧BF16检查点没有该文件，保持原推理路径。新评价保留首动作、第8–19行、完整块及起步抬升指标；正式重训的`best`采用下述分阶段TCP主分数与独立退步约束，七维动作平均误差只作诊断，指定目标服从仍需独立验证。

**第一阶段精度成对验证已完成（2026-11-01）**：用户清理远端磁盘后明确“我们可以继续任务”，本次恢复已准备的精度验证。两组分别从同一29000步检查点初始化，使用三组训练来源的24个固定起步／接近／闭爪观察，每组128步、batch 8、同seed42／恒定学习率`2.5e-6`及BF16计算，仅改变可训练权重及相应AdamW状态的存储精度；另144个三根验证观察不参加更新。真实前向／反向、参数变化抽样、FP32保存／严格重载和运行适配器输出一致性均已检查。FP32对小批训练观察拟合更好、验证起步平均抬升更接近专家，但验证总体关节／TCP误差未改善，因此仅通过数值更新与接口检查，未通过泛化或抓取验收。具体指标统一见[EVIDENCE](EVIDENCE.md#π05训练准备)和[诊断结论](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/pi05/precision_repair/assessment.json)。

仅在远端`/home/rentao/program/lerobot-aubo-pi05/artifacts/pi05_precision_repair/run/candidate`保存一份FP32诊断候选，不另存优化器、不替换原部署。原训练项目、旧检查点与现场服务保持；验证进程已退出，GPU计算进程为空。相关CPU回归179项是准备阶段结果，本轮未重复运行。启动范围及命令见[实际执行记录](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/pi05/precision_repair/execution.json)，原[准备快照](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/pi05/precision_repair/preparation.json)只反映执行前状态，不能再据其判断未执行。此次不包含正式重训、补录、部署或现场动作。

**B组候选真机试验已完成，仍未抓取成功（2026-11-01）**：用户随后明确要求将B组上真机，新增`--target-stack-model pi05-fp32-128`。它从上述29000步权重追加128次更新而来，不是从基座训练128步，也未恢复原优化器历史；默认模型仍为`three-four`，原`pi05-29000`选项保留。候选直接读取既有远端权重及隔离源码，不复制完整权重；按保存的精度文件恢复FP32 expert并使用BF16计算，握手拒绝精度约定缺失或不匹配。此为用户明确授权的诊断试验，不因离线检查通过而视为泛化／抓取验收通过。

本轮固定目录为`artifacts/live/stack/pi05_fp32_128_live/`。[操作者命令](../../artifacts/live/stack/pi05_fp32_128_live/operator_commands.json)包含归位、新快照、审核后计划检查及执行；先在三根场景中只抓A上层一根，沿用上一轮2倍限速、2px、默认异步重规划、预览、运动跟随、搬运按年龄取行和原宽容时序，一次闭合／释放周期后停止，不自动归位。软件比较确认仅模型路径／权重身份和精度元数据变化；[软件计划](../../artifacts/live/stack/pi05_fp32_128_live/software_settings_plan.json)的snapshot为待定占位，不能当作当前场景已审核。244项相关CPU检查和三次保存图像GPU协议检查通过，详细结果见[EVIDENCE](EVIDENCE.md#π05训练准备)。

操作者完成归位和快照`8385f5094c734ec1bbcb9325e0a793b4`后，助手沿用此前明确委托按新图重新标注并保存[selection.json](../../artifacts/live/stack/pi05_fp32_128_live/scene_snapshot/selection.json)：A右侧近竖直交叉上层、B右侧斜向下层两段可见区域、C左侧独立条。[圈选预览](../../artifacts/live/stack/pi05_fp32_128_live/scene_snapshot/annotation_preview.png)和[选择检查](../../artifacts/live/stack/pi05_fp32_128_live/selection_check.json)保留角色及委托复核来源，不声称用户逐多边形审核。[真实单目标计划](../../artifacts/live/stack/pi05_fp32_128_live/single_pick_plan.json)及[新图GPU协议检查](../../artifacts/live/stack/pi05_fp32_128_live/current_scene_protocol/result.json)通过后，操作者已实际执行`single_pick_run`。程序完成一个闭合／释放指令周期，用户反馈“抬升有改善、比之前好，但没抓对目标”，并确认“没有夹起，只碰推或空抓”；另报告运动抽动及途中J6转动。本轮记为指定目标抓取失败，不能记为误抓C成功或根据周期完成记为物理成功。

[本轮CPU复核](../../artifacts/live/stack/pi05_fp32_128_live/review/findings.json)、[关键帧](../../artifacts/live/stack/pi05_fp32_128_live/review/keyframes.jpg)与[关节／轨迹曲线](../../artifacts/live/stack/pi05_fp32_128_live/review/trajectory_and_joints.png)支持早期抬升改善；抽查轮廓仍在A，夹爪却接近C附近，剩余问题包括指定目标服从与动作连续性。原始输出、模型和控制参数保持，该次复核未训练或进行新GPU推理；详细数值及推断限制统一见[EVIDENCE](EVIDENCE.md#π05训练准备)。末段竹条布局已改变，本轮快照、圈选及已存在的运行目录不可复用执行。用户随后授权按现有配置重训，现已在修复选模规则后重新启动，当前状态见本节开头；同图换目标等离线检查保留为诊断选项，相对动作及RTC仍须分别实现和验证。

**用户确认的后续路线（2026-11-01）**：采用精度修复、失败阶段数据改进、相对动作和训练期RTC（实时动作分块）方向，按阶段逐项验证。全程保持视觉编码器及VLM（视觉语言模型）主干冻结，仅训练动作专家及其投影；不将解冻主干或VLM LoRA纳入当前路线。第一阶段已执行，后续真实训练、补录、部署及现场动作仍按各自明确授权范围推进。

| 阶段 | 工作与比较方式 | 验收重点 |
| --- | --- | --- |
| 第一阶段：精度与评价 | 已完成上述同一起点的128步成对验证；数据、学习率、BF16计算及冻结范围一致，仅改变权重及相应优化器状态精度 | 数值更新、保存／严格重载与适配器检查通过；验证总体动作误差未改善，不据此替换默认模型；后续用户授权的显式真机诊断见上段 |
| 第二阶段：数据与相对动作 | 先审查起步／接近／对准／闭爪采样比例、示教与部署速度及主动纠偏／恢复覆盖；通过派生清单保留原始数据，再按剩余错误分别安排阶段采样或补录、绝对／相对动作对照，每次只改变一个主要因素 | 相对动作使用`a[t+k] - s[t]`，六关节共用同一次观测状态，夹爪仍为绝对`0/100`；仅用训练集重算相对统计，反变换加回同一次观测状态，最终接口仍为7D绝对关节目标；独立场景检验对准与目标服从 |
| 第三阶段：训练期RTC | 在基本动作拟合可靠后，按真实观测年龄和控制步长确定训练延迟范围，接通与执行队列一致的已承诺动作前缀及运行时调度 | 比较延迟下的轨迹衔接与闭环抓取；现有按观测年龄跳行不等于训练期动作前缀条件化，不能只添加一个配置参数就视为完成 |

第一阶段已完成，后两阶段的训练改进尚未实现或验证。用户当前已授权并启动全数据FP32重训；不将阶段采样、相对动作或训练期RTC算作本轮已具备的改进。不能把24个观察的小批拟合当作正式训练结果。当前没有独立四根验证集，不能用三根离线改善证明四根泛化。

独立 GPU 项目为 `/home/rentao/program/lerobot-aubo-pi05`，以提交 `28abc519` 为准备前代码检查点，使用其自己的 `.venv`。2026-10-31用户已授权30000步动作专家训练，并明确四根全部训练、不划分四根验证集。训练复用既有25个来源、50堆、180条119873帧；验证仍为三根的3堆、9条6023帧。不重新导出视频、不改原始标签。三根验证不能判断四根泛化，腕部机位差异仍是待验证因素。启动与观测见[π0.5训练证据](EVIDENCE.md#π05训练准备)。

使用冻结视觉语言主干、仅训练动作专家的 `train_expert_only` 路线，启用 `bfloat16` 和梯度检查点（反向时重算部分中间结果以节约显存）；本轮不混装 PEFT/SmolVLA 依赖。独立环境固定 Transformers OpenPI 补丁提交 `dcddb970176382c0fcf4521b0c0e6fc15894dfe0`、SciPy `1.14.1`，实际依赖清单保存在远端 `artifacts/pi05_preparation/environment_freeze.txt`；系统驱动和原环境保持。

基座是官方公开的 `lerobot/pi05_base`，固定 revision `22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0`。不需要登录下载受限的 Google 仓库：使用 [OpenPI 官方 tokenizer 源码](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py)明确匿名读取的 `big_vision/paligemma_tokenizer.model`，转换为本地 Gemma tokenizer，并校验其词元编号与 SentencePiece 一致。模型、tokenizer 均保存在新项目的 `models/`。

`pi05_joint_adapter.py`保留六关节绝对角度、夹爪`0/100`、英文任务、双RGB及接近2px描边／接受闭合后原图规则。`q01/q99`是训练数据的第1／99百分位，仅由已选成功训练帧计算，供状态与动作归一化及动作还原使用；验证帧不参与统计。严格加载拒绝缺失／不匹配权重，训练损失排除动作块末尾补齐帧和32维内部表示中的非AUBO坐标。π0.5图像归一化在补边前执行，保证黑边是模型期望的`-1`。

准备入口只提供 `preflight` 和 `smoke`，**两者均为0优化步，不启动正式训练或连接机械臂**。`preflight`生成统计、真实输入缩略图及小批样本；`smoke`加载本地权重，验证前向／反向、保存／严格重载及同seed预测一致性。已有输出不覆盖；如需重复检查，使用新的输出目录。在独立 GPU 项目根目录执行：

```bash
.venv/bin/python examples/phone_to_auboi10/prepare_joint_pi05.py \
  --stage preflight \
  --data-manifest /home/rentao/program/lerobot-aubo-joint-stack/artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/target_training_sources_remote.json \
  --prepared-data /home/rentao/program/lerobot-aubo-joint-stack/artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/combined_data_preflight.json \
  --tokenizer-path models/paligemma_tokenizer \
  --output artifacts/pi05_preparation/preflight

.venv/bin/python examples/phone_to_auboi10/prepare_joint_pi05.py \
  --stage smoke --preflight artifacts/pi05_preparation/preflight \
  --base-path models/pi05_base --output outputs/pi05_joint_smoke
```

持续训练入口为`train_joint_pi05.py`，默认`plan`只读检查。`probe`从完整数据加载器随机读取真实观察，进行前向／反向但不更新参数、不保存权重；`train`才执行AdamW更新。训练与探测共用动作补齐屏蔽损失，均保留七维绝对关节契约。仅本条进程启用Hugging Face离线模式，不修改系统环境或联系机械臂。

本轮确认从同一官方`pi05_base`重新开始，batch 16、30000步、seed 42、峰值学习率`2.5e-5`、1000步预热、余弦衰减至`2.5e-6`；约4.00次训练帧遍历。视觉编码器和VLM冻结，完整训练动作专家及其投影，不加LoRA。CLI的`plan`／`probe`／`train`默认batch为16，历史128步成对诊断的`validate`默认仍为8；首次训练已实测batch 16稳态GPU总占用约18GiB，修复重启后的实际状态以本节进度记录为准。正式训练及其计划必须提供`--evaluation-geometry`，几何仅用于离线评分，不参与机器人控制。

0步和每1000步评价固定的三根监测观察：每条示教32个均匀帧，另补各阶段样本、事件邻帧和相隔约8帧的动作块重叠观察，总数由实际覆盖决定。阶段依据专家状态／已接受动作划分为初始静止、起步1秒、接近、闭爪前1秒（含接受闭爪命令的观察）、搬运、释放前1秒及后续；不是依据模型预测划分。每次固定seed42并恢复训练RNG和模型模式，不改变训练抽样。

**best是离线候选，不是抓取成功判定**：取每个观察第8–19行的有效动作，先在同一示教／阶段内求平均TCP位置误差，再跨示教等权平均，最后对起步／接近／闭爪前各阶段等权平均，得到`episode_phase_macro_precontact_tcp_error_mm`主分数。只有该值下降，且接近关节误差、闭爪前姿态、最差示教位置P95、搬运位置、闭合／释放含漏检惩罚的时间误差、相邻动作块同一未来时刻的关节不一致均未超过0步基准与上一best中较严格参考的1.2倍，才替换best。闭合／释放事件错误预算为每窗口的漏检＋误触发＋额外切换，各分项继续记录；6066步起释放事件计数、18234步起闭爪事件计数分别取消硬否决，选模结果在`diagnostic_only_guards`中列明两项，时机约束仍参与比较。固定0步参考防止受约束项多次小幅退步累积突破预算。20%是相对退步筛查预算，不是毫米／角度合格线。缺失事件覆盖保留`null`，该项明确未验证，不能当成零错误；覆盖改变时拒绝跨条件排名。候选被哪些约束拒绝会写入`validation_history.jsonl`。

阶段指观察时刻，未来8–19行可能跨越闭爪等事件。TCP几何和姿态均为离线名义模型；动作块衔接指标是专家观察下同一未来时刻的计划差异，不是实测抖动。逐关节、夹爪、完整块、每阶段及最差示教指标分别保留，七维动作平均误差不再决定正式选模；这些量均不能直接证明目标服从、主动纠偏或四根泛化。

训练入口按用户要求默认关闭验证早停：候选未通过best替换条件时原计划仍训练至30000步；本轮因CUDA故障实际中止于27481步，现已决定不再重启。5000步后若连续3次TCP主分数和验证损失均比最佳记录高20%以上，只记录`regression_warning`；此为工程提示，不能诊断四根过拟合。只有显式`--early-stopping`才启用该规则的提前停止；运行故障仍可能中断。

保存协议原计划保留监测best和final两份完整权重，最终另存优化器状态；本轮未到收尾阶段，实际仅保留21000步best。FP32准备预算约28.23GB（包含4GiB余量），只读`plan`也计入最终优化器预算。更新最佳时先完整保存新权重、原子更新`best_checkpoint.json`，再仅移除本轮被替代的最佳目录。全部验证记录保留，不在每1000步额外累积完整权重。输出必须是新目录；保存先写`.incomplete`，完成后改名。保存的优化器状态尚不构成精确续训接口。

曾按用户要求在`outputs/pi05_joint_three_four_fp32_expert/retained_checkpoints/step_021000`另存独立副本，历史[保留审计](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/pi05/fp32_retraining/retained_checkpoint_021000.json)不代表它仍存在。2026-11-02按最新决定删除该副本及本轮隔离源码内27个可再生成的`__pycache__`目录；原best的tokenizer路径指向自身，不依赖被删副本。共回收8628281344字节，清理后根分区可用91313922048字节（91.31GB），详见[精确路径及保护核验](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/pi05/fp32_retraining/training_retirement_cleanup.json)。原始日志、全部评价／预测、隔离源码及回退审计保留，数据、基座、环境和其他模型未清理。

正常结束的设计流程是依次严格加载final和best：final核对保存前后同seed动作逐值相同，best重跑原监测观察并核对全部动作块与帧身份逐值相同。两者均对全部6023帧分别使用seed42／43／44评价，保存每seed完整分项及均值、标准差、范围；原输出根目录保留final的seed42结果，其他结果在`full_validation/`。`full_validation/comparison.json`仅在3个seed一致支持同一候选且通过相对退步约束时给出离线偏好，否则记录`inconclusive`；不改监测best指针、不自动部署、不把seed数量当作独立场景数。上述流程全部结束后才写训练`result.json`。本轮故障中止后不再重启训练，结束阶段的重载／完整复核尚未执行；现有best仍属于离线候选。

原30000步训练的后台记录仍在远端`artifacts/pi05_preparation/`，只代表历史训练。修复重启后的实际命令、监督进程／训练PID、源码身份和计划分别保存在当前快照的`launch.json`、`process.json`、`source_manifest.json`、`reviewed_plan.json`，运行日志为同目录`training.log`；失败一轮的旧记录与日志已归档到`aborted_run_evidence.tar.gz`，旧权重已按授权删除，小型源码回退包`before_evaluation_update.tar`保留。本次已经运行，以下为只读计划格式示例；实际输出已存在，禁止再次使用该目录启动或将已有输出当作隐式续训：

```bash
PYTHONPATH=artifacts/pi05_fp32_retraining/src \
  .venv/bin/python artifacts/pi05_fp32_retraining/examples/phone_to_auboi10/train_joint_pi05.py \
  --stage plan --batch-size 16 --steps 30000 --eval-interval 1000 \
  --evaluation-geometry artifacts/pi05_fp32_retraining/evaluation_geometry.json \
  --output outputs/pi05_joint_three_four_fp32_expert
```

原准备阶段的batch 8零更新容量报告位于`artifacts/pi05_preparation/training_probe/`，只代表历史容量。本轮batch 16已完成27481次真实AdamW更新，初期GPU总占用约18GiB，抽查权重／梯度／AdamW状态均为FP32；未生成收尾资源报告。无需重跑旧预检或额外容量实验。`training.jsonl`保留实际更新，`process.json`记录失败退出；不将91.6%进度记为训练完成。原2px及三根＋四根SmolVLA模型和环境保留；旧1px与baseline权重已按用户要求退役，见[1像素状态](#1像素描边正式训练)。

### 结果与评价

训练入口正常完成时输出数据预检、训练统计、训练日志、`final/`、重载检查及逐帧预测；本轮FP32训练未生成final及结束复核，实际保留范围以上文当前状态为准。
正常生成的`final/`保存权重、配置、输入/输出处理器和契约。
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
二者均不是当前160条模型。`baseline-stack`是P2无描边入口，绑定三根baseline检查点，
命令、参数限制与验收边界见[P2对照](#p2对照与多根动作数据扩充)。

`run_joint_trial.py` 的 shadow/single-step/short-loop 和 `run_joint_live.py`
保留作既有诊断，并向持续入口提供共享实现；它们的旧首步时限不等同于本页持续伺服限制。
其历史操作和逐次故障分析统一从 [证据索引](EVIDENCE.md) 回溯。

### 当前模型的现场影子对照

这是保留的辅助入口：`run_joint_target_shadow.py`支持`plan/capture/predict`，没有伺服或夹爪写入。`capture`连接设备获取静止快照，`predict`加载模型计算保存观察上的动作，分别需要相应授权；都不等于抓放验证。
既有目标互换对照已完成，快照和预测证据见[成对模型记录](EVIDENCE.md#成对训练数据采后检查)。当前不重跑旧场景；新现场圈选须绑定新的snapshot ID，不能使用旧照片执行动作。

### 目标条件模型单次真机试验

**同一70000步模型的四根原始RGB对照（2026-11-06，已结束，操作者确认0成功）**：本轮保持70000步权重和原处理器，使用`--raw-rgb --cycles 4`，不指定轮廓、对象或顺序，每轮清空动作、归位、重新观察。程序完成四个闭合／释放周期、三次轮间归位，接受515份预测、发送3826帧指令，无程序异常；用户反馈“这次完全失败，一根都不会抓”，据此记录实际抓取0成功，不记为自主清空。原始`complete.json.physical_grasp_success=null`保持，人工结果单列于[结果复核](../../artifacts/live/stack/stack_local_alignment_70000_raw_rgb_four_pick_20261106/review/outcome_summary.json)，原[命令](../../artifacts/live/stack/stack_local_alignment_70000_raw_rgb_four_pick_20261106/operator_commands.json)转为执行历史。保存结束状态为停稳、伺服关闭、夹爪命令0，不代表当前硬件读数。该轮属于目标条件模型临时移除输入条件，不能单独证明轮廓无效或无轮廓重新训练会失败；后续按[原始双RGB重新训练](#原始双rgb重新训练)准备。

推理连接启用仅作用于本次SSH进程的压缩，保留`rgb_zlib`无损RGB和原有worker协议。`trace.jsonl`新增`inference_request_completed`记录，保存编码、JSON序列化、发送、等待响应和校验的分段耗时；即使控制循环在等待期间停止，已完成请求仍会在收尾时留下记录。预测年龄仍从原图像时间戳计算，不由压缩或新日志延长。有限保存图像对照及现场验证边界见[传输修复证据](EVIDENCE.md#三根现场实验进程)。

`run_joint_smooth.py --model target-stack`在2px时默认使用60000步三四根SmolVLA，并自动使用多目标可见种子分离；早期三根30000步模型须显式`--target-stack-model original`。当前21000步π0.5须显式`--target-stack-model pi05-fp32-21000`；`pi05-29000`仍指旧模型。独立环境和操作入口见[π0.5部署](#π05-三根与四根训练准备)。`--target-id`只抓一个指定目标；`--target-sequence A B C`依次抓放三根，两次之间自动归位，最后一根结束后停止。`target-pair`保留旧成对模型。`target-stack`使用第二轮混合微调第14轮best分割模型，`target-pair`继续用旧baseline；各模型、服务身份与历史验收见[EVIDENCE](EVIDENCE.md#三根堆叠正式采集)。

<a id="三根四根模型试用准备"></a>

**70000步最佳模型已部署，最新四根试验A失败（2026-11-06）**：用户明确“部署这个70000步模型，我们上真机实验一下”。显式入口为`--model target-stack --target-stack-model local-alignment-70000 --target-outline-width-px 2`，绑定`outputs/joint_stack_local_alignment_continued/checkpoints/best_step_070000`和SHA `e162240f…42cd7968`，使用该模型保存的处理器及本轮续训隔离源码；默认`three-four`和`local-alignment`60000步不变。原通用worker保留；当前70000步入口改用同服务下的`foreground_score/joint_inference_stdio.py`封装器，仅优先加载一份更新的目标感知模块，再运行原通用worker。训练源码快照、模型和原60000步服务保持，未复制权重；[部署记录](../../artifacts/training/stack_local_alignment/continuation/deployment/deployment_receipt.json)保存唯一模型身份及回退入口。199项CPU检查通过；相对于前轮60000步四根软件计划只变化检查点／SHA／模型选项／源码路径，25 Hz、5°/s、20°/s²、0.05m/s、400ms相机年龄及1.5秒预测保护等全部保持。回退使用`local-alignment`或`three-four`，仍须新现场照片和新输出目录。

[保存图像GPU验收](../../artifacts/training/stack_local_alignment/continuation/deployment/gpu_worker_check/complete.json)复用历史明确审核图，仅验证服务：两次接近加一次搬运、3份有限50×7动作、同图同seed动作与掩码一致，请求耗时0.282～0.322秒；另有服务固定2次合成预热。推理进程随后关闭，0训练更新，助手未访问机械臂、相机或IO；不能视为实时链路或物理抓取通过。新目录`artifacts/live/stack/stack_local_alignment_70000_first_trial/`已准备，[操作者命令](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/operator_commands.json)已更新为本次新场景的计划检查和手动执行入口；归位与拍照已完成。本轮首次执行在C阶段因`TargetForegroundMissing`停止，40份预测、367条指令、0个完成抓放周期；保存结束状态停稳、伺服关闭、夹爪0，不是当前硬件观测，不能据此前景丢失认定唯一根因。用户明确要求“删除残留，复用目录重新开始”：已在保留[停止与清理摘要](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/interruption_cleanup.json)后，仅删除本实验`four_pick_run/`、`home/`、`scene_snapshot/`三处共61个文件、13053058字节。该次短录像、原图和逐帧日志已按授权删除，原complete／结束状态／快照／人工圈选等小型JSON内容收录于清理记录；模型、训练报告和其他实验未动，控制／感知阈值未改。随后操作者已重新归位通过并拍摄新快照`ae76a5e7578d4c03bb718ad12755e1e9`，明确确认“轮廓正确，按 C→D→A→B”。新选择与四根计划已原位保存，CPU计划检查通过；与上次实际运行计划相比仅快照ID变化，70000步模型、控制及时序参数保持，独立条C/D优先、再取上层A及下层B，目标间三次自动归位。该次重拍后运行仍在C阶段停止：41份已接收预测、379条指令、0个完成周期，4次空前景及其间3次停稳重观察均已保存于现有`four_pick_run/`，不覆盖或删除。按用户要求参考旧模型降低门槛，核对旧新分割设置均为实例置信度0.5、像素0.5、重叠面积0.8，实时关联拒绝门槛已关闭；保存失败图确认C在夹爪遮挡下仍有可见部分，但原分割不给C当前像素。CPU同图jpeg95回放比较0.5／0.4／0.3／0.2／0.1后，仅给`local-alignment-70000`把**实例置信度从0.5降到0.2**，像素0.5、重叠面积0.8、16px关联范围及3次重观察保持；旧模型仍0.5。部署后的CPU代码在4张失败图恢复C的1763／1235／1234／1189个当前像素，静止起点图不变，全黑图仍触发空目标；未载入动作模型、未启动GPU或硬件，不能据此认定实机已修复或建立误检率。295项CPU测试通过，当前像素为空仍停止，客户端会拒绝未使用所配置信度的旧worker。详细回放、部署与回退记录见[门槛调整](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/foreground_threshold_review/change_receipt.json)；低置信度可能增加夹爪附近局部误检。操作者确认竹条和相机未变后，已重新归位并按C→D→A→B完成`four_pick_relaxed_run/`；前景放宽后的本轮没有空目标停止，抓取结果及边界见下段。新旧运行记录均保留，旧[操作者命令](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/operator_commands.json)已转执行历史，当前场景已变化，不直接重跑。


<a id="70000步交叉上层A抓偏复核"></a>

**70000步交叉上层A抓偏复核（2026-11-06）**：[本轮记录](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/four_pick_relaxed_run/complete.json)有543份已接收预测、4568条指令、四个闭合／释放周期、三次目标间归位、无程序异常；物理结果以用户反馈为准：C、D、B成功，A失败，且A后曾人工取走／移动A，不记为自主清空。[复核结论](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/four_pick_relaxed_run/review/findings.json)与[A/B闭爪前腕部对照](../../artifacts/live/stack/stack_local_alignment_70000_first_trial/four_pick_relaxed_run/review/A_B_preclose_wrist_comparison.png)保留证据：81份A接近轮廓均已查看，主要沿A，无观察到整条身份切到B，但交叉局部边界仍有误差；A在闭爪前约3秒受夹指接近扰动，图像轮廓主轴由约134.6°变到148.0°，随后仍下探、未恢复有效包夹就闭爪。该角度为图像平面估计，不是机器人腕角或物理真值。闭爪前约29ms的状态与模型目标最大关节误差0.120°，成功C/D/B分别0.122／0.130／0.100°，只说明接近模型目标。程序闭爪检查“到达模型关节目标且低速”，没有独立视觉包夹确认，因此模型目标不合适时仍可闭爪。A闭爪所用观察年龄0.563秒，成功D为0.651秒；不能将时延单独认定为主因。当前最有依据的推断是交叉上层场景的接触前对准及受扰后纠偏不足；横向、角度、下探深度和训练覆盖的贡献尚未分离。下一项有区分力的证据是同配置A的人工正确抓取，与本轮接触前及受扰状态对照后再决定纠偏示教；此次只读分析、无GPU／硬件访问、无参数变更。原始`physical_grasp_success=null`保持。

**新局部纠正60000步模型：首轮成功、第二轮抓偏碰撞停止，准备四根（2026-11-05）**：使用显式`--model target-stack --target-stack-model local-alignment --target-outline-width-px 2`，绑定本轮`joint_stack_local_alignment_retrained/checkpoints/best_step_060000`、SHA `bd95ca61…ac8d22e`和其新训练集归一化处理器；`three-four`保留原路径／SHA和默认行为。独立远端服务为`/home/rentao/program/lerobot-aubo-joint-stack/artifacts/joint_stack_local_alignment_live_service`，源码复用本轮隔离快照，不复制权重、不覆盖旧服务。198项CPU运行器／目标分离／运动预览检查通过，实际保存处理器在CPU加载、动作极值还原最大误差0；25 Hz、原运动限制、400 ms相机时效、1.5秒预测保护及SSH压缩保持。[部署记录](../../artifacts/training/stack_local_alignment/deployment/deployment_receipt.json)与[保存图像GPU检查](../../artifacts/training/stack_local_alignment/deployment/gpu_worker_check/complete.json)记录3次接近／搬运请求、50×7有限动作、同图同seed动作最大差0及掩码一致，请求耗时0.261～0.337秒；另有服务固定2次合成预热。只使用保存RGB，未连接机械臂或相机，不代表实际相机时效或抓取成功率已验收。

本轮目录为`artifacts/live/stack/stack_local_alignment_60000_first_trial/`。操作者归位通过，新快照`d9217824a2d74746bd41ab9177996013`距START最大0.0001466°、夹爪0，双路原图未改。用户明确“轮廓正确，按C→A→B（先独立条）”：C为左侧独立，A为右侧近竖直交叉上层，B为斜向下层、仅标两段可见区域。[selection.json](../../artifacts/live/stack/stack_local_alignment_60000_first_trial/scene_snapshot/selection.json)记录确认原话；[场景计划](../../artifacts/live/stack/stack_local_alignment_60000_first_trial/three_pick_plan.json)核对三会话都使用新权重、C/A/B→A/B→B及两次目标间归位，最后不归位。Ollama在用户明确要求“帮我停掉这个本地模型的推理服务”后已停止，服务文件和模型保留；不再沿用此前恢复容器的临时状态。该轮`three_pick_run/`已由操作者执行，用户明确“这次三根抓取成功了”，[人工反馈](../../artifacts/live/stack/stack_local_alignment_60000_first_trial/operator_feedback.json)分别记录C/A/B抓取成功；单根限定、精确对准、邻条扰动、送达和人工介入未另行逐项确认，不扩展成独立证据。原始`complete.json`的`physical_grasp_success=null`保持，物理结果来自操作者反馈。[本轮收尾检查](../../artifacts/live/stack/stack_local_alignment_60000_first_trial/completion_check.json)核对525份预测、4460条指令、3个周期及两次目标间归位，全部已消费预测使用新模型SHA；双路各5259帧、录像报告完整无编码错误，未重复全段解码。结束保存状态停稳、伺服关闭、夹爪命令0，不是当前硬件观测。用户已重新摆放，该轮[命令](../../artifacts/live/stack/stack_local_alignment_60000_first_trial/operator_commands.json)转为执行历史，原快照／种子不可复跑。

**第二轮重摆：抓偏，B阶段碰撞停止**：`stack_local_alignment_60000_rearranged_trial/`已实跑A→B→C；[原始结果](../../artifacts/live/stack/stack_local_alignment_60000_rearranged_trial/three_pick_run/complete.json)记录272份预测、2333条指令，只完成A一个闭合／释放指令周期，在B阶段以`isCollisionOccurred=True`停止，C未执行；收尾同时报告`servo remains enabled`，未确认安全停机，也没有有效最终状态。用户明确反馈第二次“完全抓偏”，[人工反馈](../../artifacts/live/stack/stack_local_alignment_60000_rearranged_trial/operator_feedback.json)按整体评价保存，未推算逐根成败及成功数量，原物理字段null保持。与首轮相同模型及全部控制参数的[准备检查](../../artifacts/live/stack/stack_local_alignment_60000_rearranged_trial/preparation_check.json)保留为执行前历史，不能当作当前安全状态。新旧快照、录像、预测和动作日志保持；旧命令已标为历史，不复跑。

**第三轮四根：A阶段失败，已人工中断**：`stack_local_alignment_60000_four_pick_trial/`已实跑，[原始结果](../../artifacts/live/stack/stack_local_alignment_60000_four_pick_trial/four_pick_run/complete.json)为`operator_interrupt`、135份预测、1262条指令，A未完成闭合／释放周期，B/C/D未执行，`failure=null`。用户反馈“这次抓取失败”，[人工反馈](../../artifacts/live/stack/stack_local_alignment_60000_four_pick_trial/operator_feedback.json)记录整体失败，不推算成功数量或未执行目标的物理结果，原物理字段null保持。[收尾记录](../../artifacts/live/stack/stack_local_alignment_60000_four_pick_trial/completion_check.json)核对保存结束状态为停稳、伺服关闭、夹爪命令100，不是当前硬件状态。录像报告双路各1278帧，但`complete=false`且两路有`BrokenPipeError`，未做完整解码，不认定为抓取失败根因。原快照、确认轮廓、计划、日志和录像保留，用户已重摆，旧命令标为执行历史。

**第四轮四根重摆：四个指令周期完成，现场仍失败并向左偏**：`stack_local_alignment_60000_four_pick_rearranged_trial/`已执行C→D→A→B，[原始结果](../../artifacts/live/stack/stack_local_alignment_60000_four_pick_rearranged_trial/four_pick_run/complete.json)记录500份预测、4378条指令、四周期及`failure=null`。用户明确“还是抓取失败”，并反馈在助手所见视角总体向左偏；[人工反馈](../../artifacts/live/stack/stack_local_alignment_60000_four_pick_rearranged_trial/operator_feedback.json)按图像方向的定性观察保存，不解释为已标定的机器人坐标偏差，不推算逐根成败、成功数量或唯一根因，原物理字段null不改。[收尾检查](../../artifacts/live/stack/stack_local_alignment_60000_four_pick_rearranged_trial/completion_check.json)核对保存结束状态停稳、伺服关闭、夹爪0；不是当前硬件状态。双路各5582帧，录像报告完整、编码错误空，未做全段解码。旧命令转为历史，原始照片／圈选／日志／录像保留。用户要求先提交当前项目，再准备从60000步续训30000步；固定验证／选模／保存规则不变，训练启动另等明确授权，不加入固定横向补偿。

**原three-four状态：四根连续试验已结束，转入模型与闭环诊断**：最新`stack_three_four_four_pick_fresh_layout/four_pick_run/`已由操作者执行；用户反馈这几批均有部分抓到、部分失败，未逐根确认结果，原始`physical_grasp_success=null`保持。该轮及前四轮的录像、动作和选择全部保留，旧[操作者命令](../../artifacts/live/stack/stack_three_four_four_pick_fresh_layout/operator_commands.json)已标为执行历史。当前固定诊断入口为[review/findings.json](../../artifacts/live/stack/stack_three_four_four_pick_fresh_layout/review/findings.json)，包括[五轮指令审计](../../artifacts/live/stack/stack_three_four_four_pick_fresh_layout/review/batch_audit.json)、[当前模型配置](../../artifacts/live/stack/stack_three_four_four_pick_fresh_layout/review/model_audit.json)、[远端处理器统计核对](../../artifacts/live/stack/stack_three_four_four_pick_fresh_layout/review/normalization_check.json)及[最新闭爪前对照](../../artifacts/live/stack/stack_three_four_four_pick_fresh_layout/review/stack_three_four_four_pick_fresh_layout_preclose.jpg)。现有证据优先指向末段模型位姿／纠偏与训练部署分布问题，扰动后的目标跟随退化及闭爪预测稳定性并列检查，尚无唯一根因；具体结果见[EVIDENCE](EVIDENCE.md#三根四根感知链路复核)。本次只做CPU保存产物复核、远端配置／处理器只读检查及官方／论文资料查询，没有新增策略推理、训练、部署、硬件访问或运行代码修改。

**上一轮重摆四根：效果不错，B阶段因腕部相机中断**：`stack_three_four_four_pick_rearranged/four_pick_run/`已由操作者执行，[原始结果](../../artifacts/live/stack/stack_three_four_four_pick_rearranged/four_pick_run/complete.json)为`error`、275份预测、2564条指令，C→D→A完成三个闭爪／释放指令周期；切换B后腕部Sonix CAM1最新帧9550.5ms超过400ms而停止。C/D/A各625／818／1121条动作指令，B会话虽开始但未发送策略动作，不能记B物理失败。用户反馈“这次抓取效果不错”，作为整体定性反馈保存在[operator_feedback.json](../../artifacts/live/stack/stack_three_four_four_pick_rearranged/operator_feedback.json)，未逐根确认成功及数量；原`physical_grasp_success=null`保持。快照`11e3cdad587b4bbe963c8b88da5282af`、[标注叠图](../../artifacts/live/stack/stack_three_four_four_pick_rearranged/scene_snapshot/annotation_preview.png)、[选择检查](../../artifacts/live/stack/stack_three_four_four_pick_rearranged/selection_check.json)、原始录像与日志均保留。结束保存状态停稳、伺服关闭、夹爪0且距START最大0.0001467975°，不是当前硬件观测。操作者已重新摆放，旧实验／命令状态已转为历史，继续准备见上段。

**上一轮四根：程序完成四周期，现场仍有漏抓**：`stack_three_four_four_pick_retry/four_pick_run/`已由操作者完成重启运行，[原始结果](../../artifacts/live/stack/stack_three_four_four_pick_retry/four_pick_run/complete.json)为`model_suction_cycles_completed`、415份预测、3604条指令、C→D→A→B四个闭爪／释放指令周期及`failure=null`；用户确认任务未完全完成、有漏抓，具体目标及成功数量未知，见[现场反馈](../../artifacts/live/stack/stack_three_four_four_pick_retry/operator_feedback.json)。原`physical_grasp_success=null`、录像与日志保持；结束保存状态停稳、伺服关闭、夹爪0，不代表当前硬件状态。旧实验／命令状态已转为执行历史。此前相机超时的10份预测／88条指令及按用户授权清理19个文件的记录仍保留在[旧实验入口](../../artifacts/live/stack/stack_three_four_four_pick_retry/experiment_plan.json)，不是本次重启运行的结果；相机停顿根因仍未确定。

**上一轮：SmolVLA独立条优先三根（2026-11-02，C/A成功、B未夹起）**：本轮`artifacts/live/stack/stack_three_four_independent_first/three_pick_run/`已由操作者执行。按新快照`d17653cd974f4241a4e8ce2f40e30b62`及助手受明确委托保存的[selection.json](../../artifacts/live/stack/stack_three_four_independent_first/scene_snapshot/selection.json)，顺序为C左侧独立条→A右侧交叉上层→B原交叉下层，目标间归位两次。原60000步`three-four`、2px及25项关键模型／控制／时序设置与上一单A保持。[人工反馈](../../artifacts/live/stack/stack_three_four_independent_first/operator_feedback.json)确认C/A成功、B“没有夹起，只碰推或空抓”，B开始前未人工移动；不据此认定C/A精对准或稳定成功率。[CPU复核](../../artifacts/live/stack/stack_three_four_independent_first/review/three_pick_summary.json)核对324份预测、2801条指令、三个闭爪／释放指令周期，双路各3584帧完整解码；全部目标匹配对应模型动作行，模型SHA一致。B的46次接近响应均有前景、无合并提示或运动参考失败；[输入描边](../../artifacts/live/stack/stack_three_four_independent_first/review/B_input_masks.jpg)抽查落在当前B。[闭爪前对照](../../artifacts/live/stack/stack_three_four_independent_first/review/preclose_comparison.jpg)显示B斜穿夹口并靠近右夹指，[闭爪前后](../../artifacts/live/stack/stack_three_four_independent_first/review/B_before_after_close.jpg)仍留在台面，支持末端位置／角度对准不足。A抓起时B有轻微转动，但新轮廓仍跟随B，不能单独归因于旧标注、相机、层高或扰动；具体三维误差尚未分离，见[复核结论](../../artifacts/live/stack/stack_three_four_independent_first/review/findings.json)及[EVIDENCE](EVIDENCE.md#三根动作模型身份与训练结论)。原始`physical_grasp_success=null`保持；结束保存状态为停稳、伺服关闭、夹爪命令0，不代表当前硬件状态。[实验／命令状态](../../artifacts/live/stack/stack_three_four_independent_first/operator_commands.json)已转为执行历史，场景已变，不复用旧快照和输出。成功下层示教对照已由用户暂缓，下一轮四根安排见上段；本次复核没有新增GPU推理、训练、硬件操作或运行源码修改。

**切回SmolVLA首轮单A结果：扰动后抓取成功，对准仍未通过**：本轮`stack_three_four_return_from_pi05/single_pick_run/`已完成，用户反馈“A算是抓成功了，但是是抓的过程中扰动了竹条才抓住的，并不是对准才抓的”。据此记录本轮A成功并保留限定，不记为稳定精对准或已学会主动纠偏；被扰动竹条的具体身份及邻条微小变化未独立确认。[CPU复核](../../artifacts/live/stack/stack_three_four_return_from_pi05/review/single_target_summary.json)核对107份预测、932条指令、一个闭爪／释放周期，双路各952帧完整解码、录像完整且无编码错误；[关键帧](../../artifacts/live/stack/stack_three_four_return_from_pi05/review/single_target_keyframes.jpg)可见A被带到左侧放置区。原始`physical_grasp_success=null`保持，结束记录停稳、伺服关闭、夹爪命令0。场景因A移走而变化，[旧命令](../../artifacts/live/stack/stack_three_four_return_from_pi05/operator_commands.json)已转为执行历史，不复用旧快照或输出目录。π0.5排查提交`81938577`、模型及证据继续保留。

**固定机位四根试验结果（2026-10-31，偏左及人工移条反馈）**：操作者已运行本轮A→B→C→D，并报告“这次抓取在你的视角上偏左”“我有手动移走”；各目标物理成败和人工移除时刻未逐根确认，不计自主四根清空。原始`complete.json`的物理字段null保留。本轮[准备记录](../../artifacts/live/stack/stack_three_four_fixed_camera_four_pick_20261031/experiment_plan.json)、[已审核选择检查](../../artifacts/live/stack/stack_three_four_fixed_camera_four_pick_20261031/selection_check.json)和[场景计划](../../artifacts/live/stack/stack_three_four_fixed_camera_four_pick_20261031/four_pick_plan.json)保留；当前场景已改变，[历史现场命令](../../artifacts/live/stack/stack_three_four_fixed_camera_four_pick_20261031/operator_commands.json)不可复用原输出重跑。保存日志有400份已处理预测、3617条指令、4个闭合／释放指令周期、3次目标间归位，无记录异常；8次IO读回符合请求，最终停稳、伺服关闭、夹爪命令0。双路报告各4836帧、无编码错误且完整，CPU按来源时间抽取了闭爪前后样本，未声称全帧图检。[本轮诊断](../../artifacts/live/stack/stack_three_four_fixed_camera_four_pick_20261031/review/four_pick_wrist_diagnosis.json)中3617条目标均匹配完整预测动作行，闭爪前最近状态距模型关节目标最大0.117～0.138°；[独立条腕图对照](../../artifacts/live/stack/stack_three_four_fixed_camera_four_pick_20261031/review/independent_strip_wrist_comparison.jpg)显示C、D闭爪前竹条偏在夹口右侧，符合夹爪相对竹条偏左。与混合模型实际四根来源16批120条的精确PTS起始／闭爪前图对照，按本轮真实推理首帧的夹爪金属部位作参照：104条起始图在左侧约17～26px、16条在右侧约8～12px；同类摆放6堆24条见[完整并排图](../../artifacts/live/stack/stack_three_four_fixed_camera_four_pick_20261031/review/live_vs_same_layout_training_wrist.jpg)。本轮四个目标开始时夹爪主体仅0～1px差异；此前“相对加固基准稳定”不能替代与录制参照一致。数字是局部图像位置，闭爪前73条模板匹配不足仍保留不确定，不换算机械臂补偿或据此筛数据。框选是全局图的目标提示，腕图原样进入模型；末段位姿与当前腕部视角适配仍是重点，尚未确定相机变化的因果份额。助手此次仅CPU复核保存产物，未新增GPU、硬件操作、训练或运行控制修改。

**固定腕部相机后的三根试验结果（2026-10-31）**：用户要求摆放、准备并进行一次三根抓取，建立新目录`artifacts/live/stack/stack_three_four_fixed_camera_20261031/`，见[实验准备](../../artifacts/live/stack/stack_three_four_fixed_camera_20261031/experiment_plan.json)和[操作者命令](../../artifacts/live/stack/stack_three_four_fixed_camera_20261031/operator_commands.json)。本轮建议一对上下交叉＋一根独立，拟定A上层→B下层→C独立；实际身份须由新快照确认，目标间归位2次、C后停止。保留三根＋四根60000步混合模型、2px、默认异步重规划、预览／运动跟随、搬运年龄取行及原时序限制，未做图像／轨迹补偿或微调。当前代码的CPU参数组合与此前三根计划关键设置一致，[软件参数计划](../../artifacts/live/stack/stack_three_four_fixed_camera_20261031/software_settings_plan.json)明确使用待定snapshot占位，不是已审核的现场计划。操作者已完成归位和新`scene_snapshot/`（ID `4cf65484ce0e468b80e3d96f6fa14ae2`），快照六关节距起点最大0.00014625°、夹爪命令0；[腕图核对](../../artifacts/live/stack/stack_three_four_fixed_camera_20261031/review/camera_reference_check.json)中左右金属主体相对[加固后基准](../../artifacts/live/stack/wrist_camera_reference_20261030/after_fastening/grasp_rgb.jpg)最佳位移均(0,0)、相关分数约0.984／0.995。已按当前图预填[三根审核页](../../artifacts/live/stack/stack_three_four_fixed_camera_20261031/scene_snapshot/review.html)：A右侧左上到右下交叉上层、B另一根交叉下层两段可见区域、C左侧独立条；[草稿检查](../../artifacts/live/stack/stack_three_four_fixed_camera_20261031/scene_snapshot/annotation_draft_check.json)确认非空且互不重叠，仍为`human_reviewed=false`。操作者已保存本轮`selection.json`：`human_reviewed=true`、绑定当前snapshot，A/B/C轮廓与草稿像素差均0；[选择检查](../../artifacts/live/stack/stack_three_four_fixed_camera_20261031/selection_check.json)通过。[真实三根计划](../../artifacts/live/stack/stack_three_four_fixed_camera_20261031/three_pick_plan.json)已通过无`--execute`的实际命令入口生成，确认A→B→C、剩余身份3→2→1、两次目标间归位、C后停止，模型及参数与准备时一致。操作者随后完成本轮并报告“两根成功、最后一根失败”；按实际A/B/C顺序记A、B成功，C失败，非三根全成功。[本轮复核](../../artifacts/live/stack/stack_three_four_fixed_camera_20261031/review/three_pick_summary.json)核对303份已处理预测、2708条指令、3个指令周期和2次目标间归位；双路各3531帧完整解码，录像报告无编码错误，最终停稳、伺服关闭、夹爪命令0。[关键图](../../artifacts/live/stack/stack_three_four_fixed_camera_20261031/review/three_pick_keyframes.jpg)中A/B送向左侧收集区，C周期结束仍留在右侧源区，与用户反馈一致；未进行全段逐帧物理审核，不据一轮建立稳定成功率或相机改动因果。原始`complete.json`的物理字段null保留，失败原因尚未确定；下一轮四根安排见上段。命令仍由现场操作者执行，归位会按原逻辑先确保夹爪释放，须先确认未持物及路径安全。

**三根＋四根模型试用准备（2026-10-29）**：用户选择三根场景，首轮准备单目标一个抓放周期。上述`target-stack`命令使用`--target-stack-model three-four --target-outline-width-px 2`选择60000步混合模型。2026-10-31两续训候选退役后，2px的`target-stack`入口省略模型选项也默认该60000步模型；更早的三根2px须显式`--target-stack-model original`，`three-four`不能与1px组合。复用现有2px SSH worker，每次从所选`final`加载权重及其预处理／反归一化，不上传数据、不改远端源码或替换旧模型。新模型文件、2px契约、处理器状态及服务实际导入路径已只读核对，随后操作者已完成本轮模型加载与单目标实机运行。新目录为`artifacts/live/stack/stack_three_four_20261029_trial01/`，见[准备记录](../../artifacts/live/stack/stack_three_four_20261029_trial01/experiment_plan.json)与[操作者命令](../../artifacts/live/stack/stack_three_four_20261029_trial01/operator_commands.json)。准备保持默认异步重规划、预览／运动跟随、原搬运取行及宽容时序配置；仅`--target-id A`、不自动扩大到B/C，不启用`--observe-after-arrival`。本轮审核选择中的A为左侧独立条，身份由新快照与审核文件确定。操作者先归位并记录到新`home/`，再拍新`scene_snapshot/`、审核三根全部可见身份并导出`selection.json`，之后生成真实场景计划；`software_plan.json`仅使用合成掩码做接口检查，不是现场计划。2026-10-30操作者已完成归位、新快照及[圈选页](../../artifacts/live/stack/stack_three_four_20261029_trial01/scene_snapshot/review.html)审核导出；原导出保留，原样复制为运行用`selection.json`。三根身份绑定当前快照、掩码非空且互不重叠，C保留两段可见区域，见[选择检查](../../artifacts/live/stack/stack_three_four_20261029_trial01/selection_check.json)。[真实场景单目标计划](../../artifacts/live/stack/stack_three_four_20261029_trial01/single_pick_plan.json)已通过CPU核对：新混合2px模型、A一个周期、默认异步重规划、无自动归位；`A_pick_run/`尚不存在。**首轮A已完成，操作者确认成功**：保存记录为86次新模型预测、792帧指令，一次闭合→释放、无记录异常，最终停稳且伺服关闭；双路各816帧完整解码，关键图可见A移至左侧放置区。物理成功来源为用户反馈，原日志`physical_grasp_success=null`保留；邻条微小扰动未知，不据一次成功建立稳定成功率，见[单目标复核](../../artifacts/live/stack/stack_three_four_20261029_trial01/review/single_target_summary.json)。用户随后明确要求一轮三根抓取，已在新目录`artifacts/live/stack/stack_three_four_20261030_trial02/`准备同模型／参数的A→B→C、两次目标间归位及最后停止，见[三根准备记录](../../artifacts/live/stack/stack_three_four_20261030_trial02/experiment_plan.json)和[操作者命令](../../artifacts/live/stack/stack_three_four_20261030_trial02/operator_commands.json)。首轮快照中的独立条已移走，该旧快照不用于新三根执行。第二轮操作者已完成新归位与新快照，并明确指定“上层条为A”；本轮身份方案改为A右侧交叉上层→B交叉下层→C左侧独立条，不沿用首轮A独立条的角色。[第二轮圈选页](../../artifacts/live/stack/stack_three_four_20261030_trial02/scene_snapshot/review.html)已预填当前图草稿：A/C各一段、B仅两段可见区域，掩码互不重叠，原始RGB保持；操作者现已审核保存本轮选择，原导出保留并原样复制为`selection.json`；新[真实场景三根计划](../../artifacts/live/stack/stack_three_four_20261030_trial02/three_pick_plan.json)及[检查记录](../../artifacts/live/stack/stack_three_four_20261030_trial02/selection_check.json)已通过CPU验收：A上层→B下层→C独立条、三个指令周期、两次目标间归位，C后停止，每根最多600秒。模型、分割、运动与时序设置均与首轮成功试抓一致，随后操作者已完成本轮运行并明确反馈三次都抓歪、一次未抓对，中途由人工取走竹条；本轮记为同一场景3次失败尝试，不能记为自主三根清空。保存记录有3个指令周期、2次归位、322份已处理预测与2898帧指令，`failure=null`仅表示无记录运行异常，最终停稳且伺服关闭。[本轮复核](../../artifacts/live/stack/stack_three_four_20261030_trial02/review/three_pick_summary.json)及双路关键图显示：A/B/C在闭合前后偏斜、未稳定带离源区；约45.6～46.4秒和102.0～102.8秒分别有人取走A、B，末段C仍留在源区。人工移除发生在对应失败之后，不用于解释最初抓偏；后续剩余场景含干预，不作为自主连续抓取证据。抽查的闭合前同帧轮廓位于对应竹条，尚不证明逐帧分割完全正确；最近闭合前关节状态已接近模型目标，仍不能证明目标位姿对准真实竹条。末段对准／稳定夹持问题尚未定位根因，不从日志认定固定方向、单个关节或延迟为原因。用户随后要求同配置再试一次，已新建`artifacts/live/stack/stack_three_four_20261030_trial03/`，见[重试准备](../../artifacts/live/stack/stack_three_four_20261030_trial03/experiment_plan.json)与[操作者命令](../../artifacts/live/stack/stack_three_four_20261030_trial03/operator_commands.json)：仍为同一新混合2px模型、A上层→B下层→C独立条和原控制参数，实际身份待新快照确认。第三轮操作者已完成归位及新拍照，起点与夹爪打开检查通过；本轮布局与第二轮不同，交叉对在左、独立条在右，[第三轮圈选页](../../artifacts/live/stack/stack_three_four_20261030_trial03/scene_snapshot/review.html)按新图预填A左下到右上的交叉上层、B近竖直下层两段可见区域、C右侧独立条。操作者已审核保存第三轮选择，导出绑定本轮新snapshot，A/B/C掩码与当前图草稿一致、非空且不重叠；原导出保留并原样复制为`selection.json`。[第三轮真实场景计划](../../artifacts/live/stack/stack_three_four_20261030_trial03/three_pick_plan.json)及[选择检查](../../artifacts/live/stack/stack_three_four_20261030_trial03/selection_check.json)已通过CPU验收：同模型与全部准备参数、A上层→B下层→C独立条、两次目标间归位、C后停止。第三轮操作者现已运行并反馈“还是抓歪了，我手动停止了”；[第三轮复核](../../artifacts/live/stack/stack_three_four_20261030_trial03/review/interrupted_A_summary.json)确认仅A尝试、77份预测／697帧指令，约16.93秒接受一次闭爪命令、未释放，完整周期0，B/C与目标间归位均未开始，不能记为三次失败。画面可见交叉处对准仍偏、接触后竹条被推移，未稳定提起；中止前最近闭合状态距模型目标最大约0.12609°。结束记录为停稳、伺服关闭，但夹爪命令仍为闭合100，持物／卡条状态未知，现场确认后再决定恢复操作。录像收尾报告双路BrokenPipe、完整标记false，现有MP4分别可解码712／711帧，保留不完整状态；录像错误原因未确立，也不作为抓偏原因。新布局第一目标已复现失准，不能仅归为B/C切换或上一轮人工取条。原始证据保留，本次CPU复核无新增GPU预测、补偿或控制逻辑修改。用户随后选择做一次含交叉／上下层的四根试验，已建立`artifacts/live/stack/stack_three_four_20261030_trial04/`，见[四根准备](../../artifacts/live/stack/stack_three_four_20261030_trial04/experiment_plan.json)及[操作者命令](../../artifacts/live/stack/stack_three_four_20261030_trial04/operator_commands.json)。沿用同一新混合2px模型和控制参数，准备`--target-sequence A B C D`、三次目标间归位、D后停止，实际角色／可取顺序由新图确认。圈选模板仅新增4根选项、D身份与绿色显示；151项相关CPU检查和四目标含交叉种子的分离／4→3→2→1计划检查通过，准备阶段仅CPU验证。第四轮操作者已完成归位与新拍照，归位记录通过、快照夹爪命令为0，关节距起点最大约0.00015°；当前图为右侧交叉对加左侧两根独立条。已在[第四轮圈选页](../../artifacts/live/stack/stack_three_four_20261030_trial04/scene_snapshot/review.html)预填新图四目标草稿：A右侧反斜线交叉上层、B正斜线下层两段可见区域、C中间独立条、D最左独立条，操作者现已审核保存本轮四目标选择，导出`human_reviewed=true`并绑定当前snapshot，轮廓与已检查草稿一致、非空且互不重叠，B保留两段可见区域。原导出保留并原样复制为`selection.json`，原始RGB保持。新[四根真实场景计划](../../artifacts/live/stack/stack_three_four_20261030_trial04/four_pick_plan.json)与[选择检查](../../artifacts/live/stack/stack_three_four_20261030_trial04/selection_check.json)已通过CPU核对：A上层→B下层→C中间独立→D最左独立、四个指令周期、三次目标间归位、D后停止，每根最多600秒；新模型、2px、分割、运动和时序参数保持。操作者随后运行并反馈“这次抓歪了，一次性抓起了两根”；[第四轮复核](../../artifacts/live/stack/stack_three_four_20261030_trial04/review/two_strip_pick_summary.json)确认A闭爪后交叉A/B一同带离源区并向左搬运，本轮记为A指定单根抓取失败、误带邻条，不能记为成功抓了两根。只有A的121份已处理预测和1117帧指令，一次闭合／释放、一次目标间归位；B会话准备阶段被手动中止，B/C/D均无独立动作，不能算四次失败。结束记录停稳、伺服关闭、夹爪命令0，物理持物反馈仍无。末段出现A/B实例合并与A光流点不足，根因尚未确立；当前场景已改变，不继续复用原快照执行。双路各1200帧可解码，但录像报告收尾失败及完整标记false保留，未作为抓偏解释。顺序流程按闭合／释放指令周期切换，不自动判断夹持成败；未抓走或场景被扰乱时由操作者停止。

**旧模型可用性更新（2026-10-31）**：1px及`baseline-stack`远端权重已按用户授权删除，下文两组入口仅作历史说明，恢复权重后才能使用；详见[退役记录](#1像素描边正式训练)。

**1像素试用（2026-10-26，两轮A失败；第三轮在A接近阶段中断；第四轮到位后观察超时，后续沿用默认异步重规划）**：在本节`target-stack`命令上显式增加`--target-outline-width-px 1`，会一起选择新30000步`joint_stack_target_thin_outline_smolvla/final`、独立的`artifacts/joint_stack_thin_outline_live_service/`服务和1像素输入约定；预览与保存的目标证据使用同一宽度。省略该参数仍为原2像素模型。分割、跟踪、阶段切换、动作取行、运动及新鲜度限制沿用当前设置。前两轮各只试A一次，操作者均报告抓歪失败，第二轮明确仍向右偏；两轮B/C均未尝试，原始产物保留。用户随后明确授权第三轮一次启动A→B→C，使用新目录`artifacts/live/stack/stack_thin_outline_20261026_trial03/`，按本节三根流程使用`--target-sequence A B C`，两次归位后在C结束停止。顺序切换依据闭合／释放指令周期，不检测物理成功；A未移走仍压住B或场景被带乱时由操作者停止。第三轮已由操作者运行，在A接近时仍观察到偏斜并通过示教器停止／急停中断；尚未接受闭爪命令，B/C及目标间归位均未开始。控制器报告碰撞标志，程序收尾未确认伺服关闭且无final_state；操作者随后已现场确认停稳及伺服关闭或急停。三轮保存证据诊断已完成；用户随后授权下述到位后观察对照。固定输入的三模型离线对照仅完成CPU准备，尚未执行新GPU推理。已有部署和兼容性检查不重复。结果、计划及证据见[EVIDENCE](EVIDENCE.md#1像素同预算正式训练结果)。

**到位后观察对照（2026-10-26，已实机超时结束，后续不启用）**：显式增加`--observe-after-arrival`（仅`target-stack`），关闭运动中的新预测覆盖及同一动作块内随年龄换行；每份预测仍按原规则在接收时选定完整七维动作行，然后冻结该目标。实测六关节最大目标误差<0.15°且轨迹速度<0.3°/s后，才采集下一份推理观察；连续伺服与双路录像保持。闭合／释放继续由模型离散行、到位和IO确认驱动，接受闭合后才切原双RGB搬运。原观察年龄、watchdog、状态／相机新鲜度和运动限制不放宽，因此到位加下一次推理可能超时并停止；到位不等于夹住。默认不加选项即原连续重规划方式。第四轮使用新目录`artifacts/live/stack/stack_thin_outline_20261026_trial04/`，先只试A一个周期，无自动归位；操作者已完成归位、新快照、人工审核和实机执行：接受4份接近预测、发送100帧，未闭爪，在等待到位后下一份预测期间触发原1500 ms watchdog；最终日志确认停稳、伺服关闭。本轮尚未进入抓取，不能比较对准或抓取是否改善。用户决定后续不启用此选项，使用默认异步滚动重规划：本地持续25 Hz有界伺服，GPU后台一次一份在途推理，收到有效新动作块后更新目标、继续采集新观察，运动中不等待到位；闭合／释放阶段边界保护保持。回退只需省略`--observe-after-arrival`，无需修改代码或重部署服务。112项相关CPU假设备检查已通过；本轮实机模型由操作者启动，助手只复核保存结果。结果入口见[EVIDENCE](EVIDENCE.md#1像素同预算正式训练结果)。

**按执行进度取首行候选（默认关闭，尚不推荐实抓）**：`--execution-paced`仅用于`target-stack`，隐含`--observe-after-arrival`，不能与`--transport-age-aligned`或`--approach-age-aligned`组合。接近和搬运均取模型完整第0行，到位前冻结、不发新请求；下一次双路图像与状态时间戳必须晚于到位及本次IO接受时刻。模型关节值、夹爪值、限速、到位判据及原新鲜度／watchdog保持，默认异步行为不变。原`--observe-after-arrival`仍在接收时按年龄选行，与此候选不同。到位只表示原误差／轨迹速度判据满足，不保证完全静止或物理对准。候选可能停滞或超时；现有预测对照仍发现首行拟合误差，尚未证明抓取改善。用户随后明确要求先做真机实验，已在`artifacts/live/stack/stack_three_four_execution_paced_20261031/`准备单目标A对照：保留原60000步2px模型、预览／运动跟随及原限制，仅一个抓放周期、无自动归位；操作者现已归位并拍摄新快照`f840aac74ebe46a1a363c0bb21103888`，助手按既有明确委托审核保存圈选：A为近竖直交叉上层条，B为斜向下层两段可见区域，C为左侧独立条。起点最大误差约0.00015°，三身份掩码非空无重叠，单目标新场景计划通过。首轮`A_pick_run`接受6份预测／发送60帧后因全局相机重复缓存帧中断，尚未闭爪；最终停稳、伺服关闭、夹爪0。新模式现已修复为遇到尚未更新的缓存帧时留待下个控制周期，不送入模型、不阻塞伺服；双相机时间戳成对提交，原帧年龄、倒退检查、异常及超时保持，其他模式仍立即拒绝重复帧。完整CPU回归874通过／11跳过。缓存修复后重试`A_pick_retry_fresh_frames`接受17份预测／315帧后触发1500ms无新预测超时，仍未闭爪，最终停稳且伺服关闭。用户明确选择恢复原连续重规划：省略`--execution-paced`及`--observe-after-arrival`，恢复`--transport-age-aligned`，其余模型／控制／时序25项已与之前运行核对一致；只试A一个周期，不自动归位换目标。当前离起点最大5.473°，必须先归位到新`home_continuous_replanning/`，再使用`A_pick_continuous_replanning/`运行；若现场竹条或相机被移动则需重新拍照圈选。操作者随后完成归位及`A_pick_continuous_replanning`，明确反馈“没问题这次抓对了”：99份预测／913帧指令、一个闭合释放周期、无运行异常，最终停稳、伺服关闭、夹爪0；录像末帧显示A位于左侧放置区、B/C仍在源区。本轮记为A成功、B/C未尝试，继续保留连续重规划，不默认启用首行到位等待；单次成功不证明稳定性或抓偏根因。A已移走，本轮命令和快照转为历史，后续新场景需重新审核。[成功复核](../../artifacts/live/stack/stack_three_four_execution_paced_20261031/A_pick_continuous_replanning/review/single_target_summary.json)保存证据分层，原`complete.json.physical_grasp_success=null`不改。不得复用旧快照，省略新选项即可回到默认模式。软件验收及后续修复方向见[当前诊断](#三根堆叠目标示教方案)与[候选复核](../../artifacts/live/stack/stack_three_four_fixed_view_repeat_20261031/review/fit_execution_diagnosis/execution_paced_check/review.json)。

**连续重规划三根新试验准备**：单A成功后用户要求一次运行三根，已新建`artifacts/live/stack/stack_three_four_continuous_three_pick_20261031/`，准备A交叉上层→B下层→C独立条、两次目标间归位、C后停止，实际身份以新图为准。模型、速度及时间限制、预览／运动跟随沿用刚成功的连续配置，`execution_paced=false`、`observe_after_arrival=false`；CPU入口检查通过。操作者已完成归位及新拍照，快照`a21e29c396c1416caf74e9f964346da1`；助手按既有明确委托检查保存三身份：A左侧交叉近竖直上层、B斜向下层两段、C右侧独立条。起点误差约0.0000744°、夹爪0，掩码非空无重叠，3→2→1会话及20项成功运行参数检查通过。[本轮命令](../../artifacts/live/stack/stack_three_four_continuous_three_pick_20261031/operator_commands.json)和[新场景计划](../../artifacts/live/stack/stack_three_four_continuous_three_pick_20261031/three_pick_plan.json)准备后操作者已执行`three_pick_run`并报告A抓歪，36份预测／356条指令后发生`servoJoint failed: 1`，尚未接受闭爪，B/C未尝试。收尾检测到碰撞标志、未确认伺服关闭、无最终状态文件；用户随后明确表示已经停止，未具体说明是否按过急停。两轮接触前同源画面抽查显示成功时目标位于两夹指之间，本轮目标靠右夹指；全局轮廓仍落在A。恢复连续配置只撤销逐步等待，并未修复模型对准，单次成功不能当作修复证据。暂停无改动重复实抓，后续转向固定视角及限速一致的主动纠偏示教、按场景独立验证和隔离模型修复；当前未采集或训练。命令／快照转历史，[本轮复核](../../artifacts/live/stack/stack_three_four_continuous_three_pick_20261031/three_pick_run/review/findings.json)保留失败及限制。

**默认异步模式恢复核对（2026-10-26）**：用户要求恢复后，真实无设备入口省略`--observe-after-arrival`生成软件核对计划，确认`observe_after_arrival=false`、`mode=asynchronous_inference_continuous_servo`、原接近／搬运观察年龄取行及接近中继续推理策略；模型、独立服务、运动与新鲜度限制与第四轮一致。核对仅用第四轮审核种子验证配置，已明确标记`software_check_only/not_for_current_execution`，不能作为新现场计划。下一轮目录为`artifacts/live/stack/stack_thin_outline_20261026_trial05/`，当前仅配置准备完成、未归位／拍照／抓取，仍先只试A一次。保存日志的时序统计确认存在视觉反馈延迟，但未确立其为右偏原因；完整计时定义、结果和恢复检查见[EVIDENCE](EVIDENCE.md#1像素同预算正式训练结果)。

此前`stack_target_20261018_trial03/three_pick_run`已由操作者运行：三次指令周期完成，但操作者确认B未抓到，不能记为三根物理成功。保存记录与成功下层示教对照表明，B闭爪高度偏高且腕部姿态存在差异；第118次非空小掩码异常不是唯一解释。详见[第三轮B抓空对照](EVIDENCE.md#第三轮b抓空与成功下层示教对照)。trial01～trial04的快照及输出均保留为历史证据，不可覆盖；下方通用命令必须使用新摆放、新目录及对应标注。

最新`stack_target_20261018_trial04/three_pick_run`已执行，2026-10-19操作者反馈“一轮抓取，三根竹条都抓取了”。日志确认A→B→C各一个抓放周期、两次归位，关键帧显示三根依次送至左侧、右侧抓取区清空；本轮启用了实时预览与运动跟随。A第33次请求出现空目标，停稳后第34次新观察恢复，随后完成整轮。B本次闭爪附近反馈高度比上次低约5.63 mm，未添加脚本高度补偿；新摆放与跟踪同时变化，不能据单轮归因或宣称稳定成功率。详见[第四轮实机复核](EVIDENCE.md#第四轮三根成功与现场空目标恢复)。当前快照和运行目录均为已完成实验，不可直接复用为新试验输出；标准标注`selection.json`保留，重复导出`selection (1).json`已按用户授权核对一致后删除。

**实时预览与扰动后的目标跟随（显式试验模式）**：用户已选择自动跟随移动目标、由模型持续修正。在本节运行命令上增加`--target-preview --track-target-motion`；后者只支持`target-stack`，前者也可单独用于目标模型。启动后终端显示`http://127.0.0.1:<端口>/`，在本机浏览器打开；地址同时写入新运行目录`preview.json`。左图显示该次预测的全局图像、实际输入模型的紫色轮廓及仅供观看的黄色外框，右图为同次请求的腕部图像；目标编号、阶段、预测编号、目标像素数和图像年龄同步显示。随新预测更新，不把旧框贴到较新的相机图像上；不是30 FPS无延迟检测。浏览器断开或图像变旧会显示提示，搬运和切换目标时清除旧框。预览是独立进程，队列满则丢预览帧，不阻塞伺服，不打开第二路相机，也不从网页发送机械臂命令。

默认模式仍使用最初人工种子附近8像素的当前分割。`--track-target-motion`用连续RGB纹理运动估计平移／旋转，实时搜索范围已扩为移动参考附近16像素；仍只把本帧分割前景作为模型轮廓，不把分割草稿反复当作新身份种子，也不改原始人工标注。A搬运期间继续更新B/C的运动参考，切换模型会话后保留这些参考，已夹取目标从跟踪集合中移除。光流（相邻图像特征点的移动）仍检查前后匹配和刚性拟合：单次观测中心位移≤35 px、旋转≤15°、尺度0.97～1.03、至少6个一致点及足够轴向跨度。这些条件只决定是否接受参考更新；现场调用使用`required=False`，特征不足、拟合不可靠或参考重叠时保留最近参考，不因该次拟合失败直接停止。2026-10-24本地已改为连续保留成功匹配的特征点，图像／掩码／点坐标一同更新；点数不足时尝试一份曾成功匹配的配套备用参考，拟合歧义或越界不重试，备用恢复仍核对最近接受位置的步长。诊断记录`reference_frame_id`及备用成功时的`recovered`／`primary_failure`。该补丁已通过离线检查并用于一轮真机实验，A/B参考更新连续通过但A仍抓偏，见[P1复核](EVIDENCE.md#p1首堆中止与当前目标检查修正)；持续遮挡仍可使点数耗尽。运动边界和拟合不确定的失败消息已补充具体子项数值，便于区分跨度、尺度、角度、位移及内点比例；历史trial04缺少这些数值，不能补作当时证据。

**2026-10-24统一放宽视觉门槛**：用户在B自身遮挡导致停止后要求一次梳理相关限制，当前已取消正常跟踪25%／失败回退50%的前景占比硬拦截，改为`support_policy=diagnostic_only`。部分遮挡时使用本帧分配给当前目标的可见区域，不再因面积占比低而单独停止；`current_foreground_support`和`insufficient_support_targets`仅作诊断，后者沿用原25%／50%参考线。运行计划记录`target_motion_support_policy`和`target_motion_search_radius_px=16`。较小片段也可进入动作模型，误分割风险因此增加；本帧前景、身份分离和互不串条的约束继续保留。GPU服务已更新，150项离线检查、330帧保存结果回放及遮挡／目标缺失用例通过。新配置首轮已实机运行，A闭爪时未稳定夹住、发生位移，随后由操作者中断；本轮没有占比停止，47幅A掩码按8／16像素回放完全一致，不能将夹持失败归因于这次门槛调整。见[P1修改与证据](EVIDENCE.md#p1首堆中止与当前目标检查修正)。

当前目标没有分配到任何本帧前景时，仍执行下述停稳、清除旧动作并有限次重新观察，不使用旧轮廓补齐遮挡，也不自动改抓邻条。像素归属的歧义处理、同帧／身份／快照绑定、分割模型阈值、相机与状态新鲜度、运动限制及伺服所有权保持；完整审查见本轮`review/visual_support_relaxation/audit.json`。移除`--track-target-motion`即可回到固定种子关联，预览可以保留。新摆放必须使用对应的新快照／人工种子和新运行目录；实验仍不具备任意大位移或完全遮挡后的可靠重识别能力。

**三根连续抓放的现场流程（2026-10-18授权）**：摆两根交叉、一根独立，保持原机位、起点和放置区。当前现场身份A为交叉上层条、B为下层条、C为独立条。使用`--target-sequence A B C`后，操作者一次启动：A夹紧—释放完成→停止伺服→自动归位→切换B→B完成后同样归位→切换C→C完成后停止。每次都重新读取当前图像并分割，原人工标注只作身份种子；剩余目标按A/B/C→B/C→C递减。两次之间不再要求重新拍照、圈选或输入命令。若其他竹条被带动或抓放异常，由现场操作者停止并重建场景；程序的夹爪命令计数不能自动证明物体已移走。trial04已有操作者确认三根均抓取成功及对应关键帧，自动物理成功判断仍未实现。

每次归位复用现有`home_to_start`，只在释放命令成功且伺服关闭、机器人静止后执行；沿用60°/s、80°/s²及50%速度比例并在结束后恢复原比例，不受策略5°/s限速替代。每根使用独立推理会话，归位后会等待下一次模型加载，清空上一根动作块和阶段；权重、任务文字及模型动作不变。除下述有界空目标重新观察外，异常停止后续目标；不在最后一根后归位。`--max-seconds 600`在顺序模式中表示每根最多600秒（完成即结束），归位和模型加载另计。

**空目标暂停与恢复**：`target-stack`接近阶段若收到与当前请求／身份／快照一致的`target_foreground_missing`，先关闭伺服、等待静止，保持实际夹爪命令；清空旧动作块、待夹紧目标和轨迹速度。随后使用停止之后的双路新图像与当前关节状态请求预测，有效且通过原有校验后从实测位置、零轨迹速度重新接续。每根累计最多3次新观测重试（成功一次不重置预算），第4次空目标停止整轮，不跳过当前目标。3次是监督试验的有限恢复预算，并非已验证的漏检持续时间；每次仍受原响应等待限制，每根总时限不重置。已经接受夹紧后不进入此恢复路径，通信／模型／状态／停伺服错误仍停止。

远端仅对这种明确的多目标空分配保留会话；失败帧仍消耗自己的序号，不调用动作模型，不返回旧动作或旧掩码。错误证据保存在新运行目录`target_failures/<请求编号>/`：双路原始RGB无损PNG、候选实例PNG，以及`failure.json`中的请求编号、目标／阶段、观测关节、来源时间戳、候选列表、各目标掩码与分配报告。旧双目标服务不因本地修改自动获得此协议。trial04已在A的一次空目标后实测停稳并用新观察接续；这一例验证恢复流程，不代表分割漏检已修复或连续多帧丢失均可恢复；[软件与保存图像验证](EVIDENCE.md#三根堆叠正式采集)。

**下一轮操作模板**：保持当前模型、参数和机位，由操作者完成新摆放；先按[归位条件](#2-归位)执行归位并确认成功，再采集静止快照。以下只展示流程，不代表助手获得硬件操作权限。

2026-10-24修复后的`stack_target_p1_pile01_point_tracks_20261024`已执行并人工中断，操作者确认A闭爪前未对准、没有抓到，B/C未尝试；本轮目录、快照和输出只用于回溯，不再复用执行。结果见[P1证据](EVIDENCE.md#p1首堆中止与当前目标检查修正)。2026-10-25用户暂停P1，转入[P2准备](#当前位置与下一步)。以下保留target-stack现场流程模板；任何新实验都须另取未使用的目录并采集新快照。修复在本机运行，无需为此重新部署GPU服务。

```bash
TRIAL_ROOT='artifacts/live/stack/stack_target_<日期>_<新试验号>'
```

<a id="静止快照与种子导出"></a>

尖括号内容须换成后续确定的新目录名；`TRIAL_ROOT`只是本终端定位实验产物的变量，不改变系统环境配置。完成归位、夹爪打开且摆放静止后，由操作者采集快照（读取相机和状态，不发送运动／夹爪命令）：

```bash
.venv/bin/python examples/phone_to_auboi10/run_joint_target_shadow.py \
  --stage capture --onsite-confirmed \
  --output "$TRIAL_ROOT/scene_snapshot"
```

target运行需要根据新图确认A上层／B下层／C独立条。打开捕获命令打印的`scene_snapshot/review.html`，先把“可见竹条数”从默认2改成3，再按身份描出可见区域；下层条在遮挡处拆成可见段，不补遮挡。可先生成可检查草稿，草稿不能自动标成人工已审核。页面默认导出`target_shadow_<snapshot_id>_selection.json`，将本次实际审核的文件保存到本快照目录并命名为`selection.json`，核对`human_reviewed=true`、snapshot ID、身份齐全、轮廓非空且互不重叠。若浏览器导出为`selection (1).json`等名称，先检查实际文件和内容再规范文件名，删除重复导出须另有用户授权。P2的baseline也拍快照作复摆和身份评价依据，但不需要圈选，也不把这些评价信息传给模型。

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

本轮重点观察目标轮廓是否随扰动保持对应、是否仍长期停留在旧参考、下探是否碰条，以及实际夹住并送达情况；跟踪更新和物理抓取分开判定。抓完后记录新摆放、逐根抓对／夹住／单根／送达、邻条扰动、是否中断及原因，再检查`complete.json`、`trace.jsonl`、`commands.jsonl`和双路录像。软件周期完成与操作者物理结果分别保留；不改变原始日志中的`physical_grasp_success`。每轮使用一个固定复核目录，摘要更新到[EVIDENCE](EVIDENCE.md#提交与实验进展总览)。

**中断后续抓**：只有操作者确认A已移走、剩余B/C与原参考及机位仍相符时，才可评估改用`--target-sequence B C`和新输出目录；若摆放变化或身份不确定，重新拍照标注。trial02的旧续跑命令仅可从Git历史回溯，不再作为当前默认操作。

仍可用`--target-id A`运行原单次模式，默认不自动归位。单纯加`--cycles 3`不能替代明确目标顺序；顺序模式的周期数由列表长度决定。每轮记录各目标完成的指令周期、归位反馈和当前目标；物理抓对、夹住、单根送达由现场反馈记录。

新入口沿用下述目标模型控制限制；`--transport-age-aligned`复用已有搬运取步方式，未加入夹爪触发补丁或轨迹补偿。首轮因观测年龄512.6 ms超过500 ms停止后，用户授权放宽截图中的时效限制和总运行时间；后续采用上述显式宽容选项。保存图像的通信耗时不能代替现场相机帧龄与控制时序验收。

| 选项 | 行为与边界 |
| --- | --- |
| 不加分离选项 | 按最高分选择当前分割候选；实时位置／方向／面积／重合度／分数拒绝门限已取消，评分不证明身份正确，完全无候选时停止 |
| `--paired-target-separation` | 已有双目标前景分离路径，限两根；保留历史部署 |
| `--multi-target-separation` | 与双目标选项互斥，使用全部可见种子和当前实例；新`target-stack`自动启用，支持剩余4／3／2／1根；旧`target-pair`同名选项不改变其旧检查点绑定 |
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

2026-10-24用户进一步要求把“分割后自主选择下一根、顺序不限、逐根拿走直到清空”加入计划，具体任务与验收统一见[自主选目标与逐根清空](#自主选目标与逐根清空)。指定目标服从作为动作能力验证继续保留；自主清堆的目标由系统选择，两类实验分别评价。

层高分析保持仅RGB、无深度输入和脚本高度／姿态补偿的动作路线。按清堆顺序抓下层条时上层条已被移走，故不宜表述为“被压条抓得更低”。现有层高证据一正一反：trial03 B闭爪仅比A低0.822 mm并抓空，trial04 B低4.875 mm并成功；21条下层示教估计闭爪Z中位数90.875 mm，见[第三轮对照](EVIDENCE.md#第三轮b抓空与成功下层示教对照)和[第四轮复核](EVIDENCE.md#第四轮三根成功与现场空目标恢复)。分割类别`top_strip`／`covered_strip`不进入动作模型输入（策略只见描边全局图、腕部图和七维状态），层判断须由策略从像素推断；被压条描边断成两段等形状线索是否构成捷径，尚待实验区分。

```text
固定全局相机观察
  → ① 感知：Mask2Former可见实例＋top_strip／covered_strip
  → ② 选择：从当前未被压的上层／独立条候选中选一根（现为人工指定，自动选择待实现）
  → ③ 动作：当前全局图描出目标可见轮廓 → 目标条件SmolVLA输出七维关节动作，单次抓放
  → ④ 确认：抓对、恰好一根、送达（初期人工判定）
  → 重新观察，不沿用旧判断
```

分工理由：“选哪根”和“怎么抓”分开后，失败可定位到看错、选错或抓歪；目标只通过当前图像描边表达，七维动作契约、示教流程和训练入口可复用；每次抓放后重新观察，剩余场景仍是同类问题，邻条移动也不会沿用失效的初始顺序。

### 分阶段路线

| 阶段 | 目标 | 核心技术 | 通过依据 |
| --- | --- | --- | --- |
| 1. 指定目标抓取（**已阶段性结束**） | 两根分离摆放时，指定哪根抓哪根 | 描边式目标条件化、双目标前景分离、按观察年龄取步 | 已有约2组近似配对显示换描边后实际抓取对象随之改变，但无无描边基线，只算初步迹象；斜条J6欠转遗留，改由离线诊断处理 |
| 2. 三根堆叠（**P1暂停，进入P2准备**） | 三根指定顺序抓放及有／无描边对照 | 数据／标注和两组训练已完成；静态视觉已阶段性复核，已有target-stack连续抓放、空目标恢复、运动参考更新 | trial04已有一轮三根成功；P1末段抓偏未解决，暂停不等于通过；两模型的可比真机结果仍待取得 |
| 3. 自主选目标与逐根清空（新增明确目标，待实现） | 分割后自行决定下一根，允许上层条或独立条先抓，逐根移走直到源区清空 | 自动目标选择接描边式VLA；先验证三根自动选择与人工结果确认，再补自动结果／清空判断和失败重抓，见[实施阶段](#自主选目标与逐根清空) | 按实际单根移除、送达、整堆清空和人工介入评价；自由顺序不算错误，现有指令周期与空前景恢复不等于自动清堆 |
| 4. 长短与语言分拣 | 语言指定放置区 | 长／短×放置区A／B四种示教组合，同一目标只换目的地指令 | 放置行为随指令改变；此处A／B是放置区，不是目标身份 |

2026-10-15起取消原“下探高度修正”阶段：按层高`n×8 mm`在下探段做FK→竖直修正→IK的方案属于脚本混合，不符合纯具身智能方向；堆叠层高由模型从示教中学习，原方案可从Git历史回溯。
A／B成对实验按用户决定不再继续，原[位置／角度分离对照](#目标条件模型单次真机试验)取消；目标服从改在堆叠实验中逐次统计，但只有“当时至少两根可选上层条”的步骤才计入服从证据，并须与同数据无描边基线对照；只有一个可选目标的步骤只验证抓放能力。
新增VLM选目标、RL、深度相机、架空及不规则支撑暂缓；自动目标选择先采用下述轻量任务层方案，不为当前问题更换整套架构。

### 自主选目标与逐根清空

**任务定义（2026-10-24用户新增，计划阶段）**：在现有机位、支撑台和六根及以下的研究范围内，系统从右侧堆放源区的当前分割结果自行选择下一根，由目标条件VLA完成单根抓取并送到左侧放置区；每次抓放后重新观察剩余场景，直到确认源区清空。上层条与独立条不规定优先级，不预先固定整堆A→B→C顺序；放置区中的已送达竹条不再进入待抓候选。

例如A压着B、C独立时，初始候选为A和C：A→B→C、A→C→B或C→A→B均可，只要每步确实单根移走。B仍被A压住时不因“顺序不限”而直接抽取，移走A后依据新观察再评估B。现有`top_strip`类别包含未被压的交叉上层和独立条，可用于形成候选；类别、分割置信度或可见面积都不单独证明物理可夹。初期沿用操作者保证下指空间的限定摆放条件，不据此宣称任意散堆可抓。

**推荐实现**：增加一个轻量任务层选择器，读取当前源区实例与类别，筛出未被压候选，再按经离线复核的可见区域质量、实例分离和已验证工作区等信息排序，每次仅选一根。此排序是待检验的工程规则，不是已训练的抓取成功概率；不需要先训练一个完整顺序规划模型。选中实例的可见轮廓仍作为紫色描边送给SmolVLA，关节、夹爪与抓放轨迹继续由动作模型产生，保留纯模型动作路线。是否需要学习式选择器，取决于简单选择规则暴露的具体不足。

**当前代码缺口**：`resolve_targets()`仍按人工`--target-sequence`预建目标会话，`prepare_live_setup()`读取绑定快照的人工审核种子；运行在开合指令周期完成后切换下一目标，`physical_grasp_success`仍为`null`。需新增由当前分割建立目标／种子并动态切换的接口，明确自动来源，不把自动掩码标成人工已审核；保留既有人工指定入口用于对照。还需独立确认物体是否真正拿走、是否单根送达及源区是否清空，不能只按完成周期删除目标。

| 子阶段 | 具体工作 | 验收与边界 |
| --- | --- | --- |
| 自动选择离线准备 | 复用已有分割结果，在源区列出候选、选中实例与理由，检查错类、合并、短段和放置区排除；准备自动目标输入接口 | 用人工复核检查选择合理性与目标轮廓身份；此阶段不驱动机械臂，也不把分割分数当作抓取成功率 |
| 三根自主选择、人工确认结果 | 保留为后续独立事项；当前先P2、再扩充四根／六根动作示教。接入动态目标选择后，每次抓放由操作者确认结果，再从新观察选下一根 | 不作为四根／六根人工示教采集的前置条件；仍需人工结果确认，因此只验收自主选择与循环流程 |
| 自动结果反馈与清空闭环 | 用抓放前后新图像及可用夹爪反馈判断实际移除／送达；无遮挡新观察核对源区剩余物体；异常时重观察，按明确次数预算重试或停止 | 检查抓空、多抓、滑落、漏检误判清空和人工介入；单次无检测结果、空目标错误或IO周期完成均不能直接判清空 |
| 扩展四根／六根及更多层 | 四根／六根人工示教按[当前数据路线](#p2对照与多根动作数据扩充)推进；自主清堆验收仍先检查三根，数量增加与更多层数分开验证 | 按整堆记录单根移除、送达、清空、尝试次数和介入情况；新数量／层数的能力须另行验证 |

P1的六组新增训练及无训练推理改进均未选出新运行候选；[固定观察拟合历史结果](EVIDENCE.md#p1新增动作模型横向对比)中闭合拟合通过、关节拟合未通过，诊断权重不部署。2026-10-25用户暂停P1，改为先P2、再四根／六根采集与混合训练。自主选择仍保留为后续能力，不因阶段顺序调整而视为已实现；其离线准备可复用既有结果。

**评价区分**：自主清堆主要看逐根移除、送达、整堆清空、误判清空、尝试次数及人工介入；不要求按指定类别或身份排序。每次系统选中目标与实际抓走对象是否一致仍记录，用于定位感知／选择／动作错误。它不能替代P2中“同场景由外部指定不同目标”的描边服从对照；两类结果不混算。固定A→B→C的历史P1记录及五堆对照保留，不改写为自主选择成绩。

### 当前位置与下一步

**最新决定：恢复四根→六根人工遥操动作示教准备，与另一会话的1像素、30000步正式训练及既定离线评价并行。** 本任务先做本地录制准备、CPU检查和逐堆指导；新数据不加入该训练，混合训练、模型部署及自主抓取另行安排。旧“暂停新采集”决定已被本次授权替代。P2首轮8次现场尝试和对准差异结论保留；第8次C拍照前右移的条件见[首轮结果](EVIDENCE.md#p2首轮配对结果)，不据此添加固定位置、腕角或高度补偿。P1旧候选搜索和五堆验证仍暂停、未通过。

**P2已有证据的离线诊断及固定观察对照已完成**：核对8次接近阶段468份预测、4028条闭爪前命令，未发现下发目标与记录的模型动作行混用。target额外处理确实使反馈更慢、通常选取更后的动作行，但08成功时的观察反而更旧，不能单独归因于延迟；03／08的C摆放位置、角度与夹口相对关系不同，仍未建立固定偏移或右移因果。首轮固定专家持位对照新增baseline 27次预测、复用target 27份旧结果，另2次热身；预定总体指标没有跨场景／seed一致优势，按角色事后拆分，上层条3份观察×3种子均为baseline误差更小，独立条则为target均值更小。后续用户授权的训练内上层检查也已完成：新增9次baseline预测＋2次热身，8/9个种子配对baseline更小，但一个训练场景的跨种子均值方向反转，不能仅归因于新场景泛化。指标、全部分组和限制统一见[P2对准差异离线诊断](EVIDENCE.md#p2对准差异离线诊断)，图像与结果沿用[原复核页](../../artifacts/live/stack/stack_p2_paired_comparison_20261025/review/target_alignment/review.html#upper-training-fit)。P1旧训练搜索与固定补偿继续暂停；新采集已按下方最新授权恢复。

**1像素正式训练与离线评价已完成**：旧7组短续训候选及两份插值检查点按用户要求清理，既有结果保留于[EVIDENCE](EVIDENCE.md#短续训候选退役与1像素正式训练准备)。从原预训练基座进行的同预算30000步1像素对照未显示整体改善，具体结果见[正式训练结果](EVIDENCE.md#1像素同预算正式训练结果)；参数与执行状态统一见[正式训练](#1像素描边正式训练)。独立目标输入结构继续暂停，现有运行模型与默认2像素输入保持。

**P1旧候选已退役**：历史结果见[EVIDENCE](EVIDENCE.md#p1新增动作模型横向对比)，权重及专属代码按最新授权删除；现场证据保留。相对关节表达、几何补偿及分割训练不随本轮启动。

2026-10-24用户确认静态视觉识别表现达到本阶段预期，下一步进入抓取场景验证准备。动作侧保持`target-stack`最终权重及trial04选项；视觉侧已把第二轮混合微调第14轮best接入`target-stack`。操作者随后完成真实GPU调用：A／B／C三次均通过模型身份检查并返回50×7动作块；一次性检查脚本在本地保存时出错，未留下目标轮廓图，具体原因见[证据记录](EVIDENCE.md#多根静态分割数据链路检查)。这证明模型加载与保存观察的推理调用已发生，不代表轮廓质量或物理抓取验证完成。

P1五堆验证暂停，不再作为启动P2或四根／六根人工示教的前置条件。后续实验继续记录逐根物理结果、邻条扰动、前景丢失与恢复，保留失败和中断。当前只有旧配置下的一轮三根成功反馈，不预设新摆放的抓取成功率；已知短段漏检和拆分保留，视觉侧暂不追加训练。

已完成的续训候选均不替换运行模型；持位与闭合候选各1000步及固定评价也未通过筛选，结论见本轮结果。P2使用原来同数据、各30000步的baseline与target检查点；`uniform`不是无描边baseline，不能与其混用。已有成对模型、训练数据及失败证据继续保留，不重开旧普通单根或A／B位置／角度补录。

**P1暂停时保留的接触失败诊断（2026-10-24）**：连续特征点与备用参考修复已接入本地、通过离线检查，并按用户决定完成一轮真机实验。A/B参考更新连续通过，但A仍抓偏失败；用户进一步确认闭爪前夹口已经没有对准竹条。已核对当前轮廓、模型动作行与实际关节，未发现旧图卡住或动作行混用；关节到达模型目标不等于夹口对准。本轮支持把主排查点转向末段抓取位姿与视觉反馈，尚不能确定横向位置、腕角及高度各自的贡献。保持现有跟踪修复，不原样重跑、不优先补纠偏行为／示教，也不添加固定位置、腕角或高度补偿；后续训练对照见下文。具体指标、画面和限制见[P1复核](EVIDENCE.md#p1首堆中止与当前目标检查修正)。

**P1历史修复路线（2026-10-24用户确认，当前暂停）**：继续以纯模型为主，由模型产生抓取动作。中心线对准、几何视觉伺服及混合纠偏方案仅留作讨论记录，**暂不考虑、不推进实现，也不作为当前修复的前置条件**。现有异步预测、耗时分解、无损编码对比及腕部测量结果保留；增加动作块长度不能代替提高新观察频率。继续排查模型输入、动作选步、执行时序与反馈链路的一致性，优先寻找可验证的软件问题；完成诊断后已按用户授权进行下述训练对照，不添加外部位置／姿态／高度补偿。结果与技术来源统一见上述[P1复核](EVIDENCE.md#p1首堆中止与当前目标检查修正)。

已完成保存配置／输入契约核对、4轮A段执行日志对比、6组现场观察×3个种子的18次离线复算，以及用户同意继续后的完整动作块评价：3个验证场景、9条成功示教、27组观察，共81次50步预测。现有6023帧验证仅保存首行动作的缺口已在这些选定观察上补查；图像、未来动作与真实远端归一化参数核对通过。专家已停稳等待闭爪时，模型闭爪姿态仍有偏差，常用选步位置附近未见误差突然增大。另以9组末段观察×3个种子完成27次内部生成步数10→20的对照，收益不稳定且增加计算耗时，**不改运行默认值**。指标与限制统一见[P1复核](EVIDENCE.md#p1首堆中止与当前目标检查修正)。

现有60条训练示教都包含闭爪前稳定保持段，不能把问题直接归为缺少这类示范。用户授权后，已完成3个训练场景、9组末段输入×3种子的27次预测，并复用验证侧27次结果，见[P1训练内对照](EVIDENCE.md#p1首堆中止与当前目标检查修正)。三个场景对和三个种子的首次闭爪关节误差均值都为训练侧更小，但训练内仍有明显残差，固定第10行的差距也小于首次闭爪指标；当前支持“训练内生成精度与泛化问题并存”，不能归为仅泛化不足。尚未确立具体软件缺陷或有效修复。用户同意的普通续训与末段监督加权对照已完成，两组各1000步、完整验证及共108次末段动作块预测均完成。两组有小幅局部改善，加权未稳定优于普通续训，闭合时机及部分示教残差仍在；两组均不提升为运行模型。参数、逐项结果与限制统一见[P1训练对照](EVIDENCE.md#p1首堆中止与当前目标检查修正)及评价入口。用户同意的持位／闭合拆分已完成首轮：复用162个动作块，并完成243次输入扰动预测及3次一致性核对。专家持位时模型仍有初始偏差与后续动作变化；首次闭合包含短脉冲，首行闭合对输入夹爪状态高度敏感。定义、条件限制和结果见上述P1证据及研究入口。两个候选现已分别实现并通过本地离线验证，监督定义、评价与预算见[EVIDENCE历史结果](EVIDENCE.md#p1新增动作模型横向对比)；本轮真实训练与配套离线评价已完成：A未改善持位，B改善持续闭合时机但增加短脉冲与提前闭合，两者均未通过筛选，详见结果入口。随后已完成一组关键阶段均衡采样，也未通过筛选，见采样结果。当前运行检查点保持，全部已授权实验已结束，物理抓偏尚未验证修复。

**推进顺序（2026-10-25按用户最新决定调整；P1暂停，不要求通过后才启动P2）**：

| 顺序 | 事项 | 要点 |
| --- | --- | --- |
| P0（已完成） | 确定[主问题](#研究问题与总体架构)与措辞 | 描边指定目标为主线，位置／角度／支撑高度适应为能力分析，自动清堆为后续目标 |
| P1（暂停，未通过） | 纯模型末段对准、持位与闭合可靠性 | 六组新增训练及无训练推理改进均未选出新运行候选；保留失败、代码、模型与报告，暂停新诊断和五堆验证，扩充数据后的新模型再单独评价末段抓取 |
| P2（首轮已收官，8/8次尝试） | 同数据有／无描边模型真机对照 | 4组有限配对全部结束，失败与人工中断均保留；服从只统计k≥2步骤。逐次结果及限制见[首轮汇总](EVIDENCE.md#p2首轮配对结果)，流程见[首轮配对](#p2首轮配对现场流程) |
| 已完成诊断，另会话继续 | 有／无描边末段差异与1像素正式对照 | P2已有输入、模型输出和时序诊断保留；1像素正式训练及既定离线评价由另一会话负责，采集任务不修改冻结源码、来源、归一化、参数或输出 |
| 当前采集 | 四根动作示教 → 六根动作示教 | 接口准备和CPU验证后，先四根小批，检查通过再扩量及推进六根；一组一反馈，数量预算未确认，混合训练不在本次范围 |
| P3 | 三／四／六根混合训练与新场景真机评价 | 合并合格且契约一致的训练示教；验证与最终测试独立保留，分别报告三／四／六根效果。更多层数另行验证 |
| 后续独立事项 | [自主选目标与逐根清空](#自主选目标与逐根清空) | 自动选择及结果反馈尚未实现，保留研究目标，不作为四根／六根人工示教的前置条件 |
| 并行 | 腕部20帧人工标注导入 | 支撑“失败能说清原因” |
| 暂缓 | 感知升级离线比较、跟随失败分项记录 | 感知架构升级暂缓；当前已处理明确的跟踪缺陷，见[待办表](#各阶段待办与约束) |

完整动作块、夹爪切换时机及闭爪关节目标误差已用于本次离线诊断；按层的末端位姿误差指标仍暂缓。P1所需闭爪Z已由现场日志记录。

#### P2对照与多根动作数据扩充

**P2准备**：两组原模型均使用同一60条训练／9条验证、SmolVLA基座、30000步和seed42；完成记录见[模型身份](EVIDENCE.md#三根动作模型身份与训练结论)。原无描边baseline已通过`run_joint_smooth.py --model baseline-stack`接入（2026-10-25完成本地实现与离线验证）；它使用原始双RGB，不接受任何目标提示或目标元数据。target组保留现有描边流程。

**已退役的无描边入口**：2026-10-31已删除远端权重，恢复权重后才能使用。历史`baseline-stack`绑定`/home/rentao/program/lerobot-aubo-joint-stack/outputs/joint_stack_baseline_smolvla/final`，SHA-256为`9c4a474c0702bddbe8c44c68b4306bbc3febad967474a6473275af5778151b14`；复用现有`artifacts/joint_stack_live_service/joint_inference_stdio.py`，以普通契约和完整50×7动作块启动，不加载Mask2Former。默认一次指令周期、120秒；接近阶段自动按观察年龄取行，搬运阶段仅在显式`--transport-age-aligned`时按年龄取行，否则保持闭爪后首步。`--target-snapshot`、`--target-selections`、`--target-id`、`--target-sequence`、`--target-preview`、`--track-target-motion`、`--paired-target-separation`、`--multi-target-separation`及`--cycles`大于1会在连接设备或模型前报错。查看计划：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
  --model baseline-stack --transport-age-aligned
```

现场执行仍在同一命令加`--execute --output <新目录>`，由操作者现场授权。一次周期结束只表示指令周期完成；`target_id`保持空，`physical_grasp_success`仍为`null`，评价目标另记在实验记录中。权重身份、执行参数对照与离线验收见[证据](EVIDENCE.md#三根动作模型身份与训练结论)。

**有限GPU入口检查（2026-10-25已完成并通过）**：用户授权后，`artifacts/capture/stack/joint_stack_train_20261017/p2_baseline_entry_check/check_baseline_entry.py`完成p-101／102／103各一个接近和一个搬运观察，共6份原始双RGB预测（seed42），随后`verify`通过。服务代码含2次合成warmup，冻结预算为8次动作调用、0训练步、300秒和100MB；本轮在预算内完成，结束检查时无GPU计算进程。真实模型身份、普通七维契约、无target输入／无分割及全部有限50×7动作块已核对，见[执行证据](EVIDENCE.md#三根动作模型身份与训练结论)和[验收报告](../../artifacts/capture/stack/joint_stack_train_20261017/p2_baseline_entry_check/verify.json)。目录已保存真实结果，不重跑`prepare`／`predict`覆盖。该检查只写盘，不能据此判断物理抓取成功；随后P2现场进展见下文。

实验按新摆放配对、交替两模型顺序，并保存每次重摆快照；相同布局下更换指定目标，记录baseline实际选择分布。目标服从仅统计k≥2的步骤，抓对、夹住、单根、送达分别统计，失败／中断／不可判断均保留。baseline没有显式指定目标，不能假定其必须按固定A→B→C顺序抓取。若两组抓取成功都很低，报告结论受限，不宣称描边无效，也不把P2改成无限P1修复。

##### P2首轮配对现场流程

**状态（2026-10-25）**：首轮已结束8/8次尝试、4/4组配对，其中6次软件正常完成、2次人工中断。第8次target/C由操作者确认仅独立C送达并松开、无邻条扰动，记录已完成；全部结果及按模型计数见[首轮汇总](EVIDENCE.md#p2首轮配对结果)。有描边路线已有一次按指定C完成抓放的证据，但末段对准仍不稳定；无描边也有抓偏，不能把全部失败归结为统一右偏。前两次A/C有限离线检查保留在[查看页](../../artifacts/live/stack/stack_p2_paired_comparison_20261025/review/target_alignment/review.html)。布局相近且样本少，不据此建立模型整体优劣。用户随后补充08的C在拍照前向画面右侧稍移，快照对应调整后摆放；成功结果保留，同时记录位置敏感性线索，不能据此确定右移的因果作用。**此后对准差异的离线诊断已完成；四根／六根人工采集现按[最新采集安排](#四根与六根动作数据扩充)恢复，混合训练仍不在本次范围。** P1旧候选训练和五堆验证仍暂停、未通过。

首轮按**2个计划布局 × 2个评价目标 × 2个模型，共4组配对／8次单次抓放**执行，当前已收官。排程与模型／参数准备快照见[experiment_plan.json](../../artifacts/live/stack/stack_p2_paired_comparison_20261025/experiment_plan.json)，各次人工反馈见[trial_records.json](../../artifacts/live/stack/stack_p2_paired_comparison_20261025/trial_records.json)，派生计数见[summary.json](../../artifacts/live/stack/stack_p2_paired_comparison_20261025/summary.json)；准备报告保留当时状态。可在项目根目录运行下方命令重算本地汇总，只读原记录并更新派生summary，不调用设备或GPU。以下现场流程保留用于回溯，已有8个尝试目录不再执行；没有新增P2补试安排。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python artifacts/live/stack/stack_p2_paired_comparison_20261025/summarize_results.py
```

两个布局均为三根40 cm竹条：“A交叉上层、B交叉下层、C独立”，A／C都须有下指空间，预期可选数k=2，实际值逐次填写。布局02改变位置／朝向等空间安排，但保持该结构。每次尝试前恢复该布局的三根初始状态、同一起点和打开的夹爪，再独立拍新快照；同一布局以它的第一张初始全局图为复摆参照。人工重摆并非完全相同输入，位置／角度差异如实记录。不要让第二个模型接着第一个模型抓完后的剩余场景运行，也不要在竹条上添加身份标记。

| 配对 | 布局 | 预设评价目标 | 第一次（目录后缀） | 复摆后的第二次（目录后缀） |
| --- | --- | --- | --- | --- |
| pair_01 | layout_01 | A：交叉上层 | baseline：`pair_01/01_baseline` | target：`pair_01/02_target` |
| pair_02 | 同一layout_01 | C：独立 | target：`pair_02/03_target` | baseline：`pair_02/04_baseline` |
| pair_03 | 新layout_02 | A：交叉上层 | target：`pair_03/05_target` | baseline：`pair_03/06_baseline` |
| pair_04 | 同一layout_02 | C：独立 | baseline：`pair_04/07_baseline` | target：`pair_04/08_target` |

评价目标在运行前确定。baseline只记录它实际选择哪根，表内A／C仅用于事后评价，不进入baseline命令、任务文字或图像。两模型都使用`--cycles 1 --transport-age-aligned --latency-tolerant --max-seconds 120`；target另保留既有预览和运动跟随。控制参数的本地计划核对见[preparation_check.json](../../artifacts/live/stack/stack_p2_paired_comparison_20261025/preparation_check.json)。每次120秒指策略运行时限，模型加载、归位、摆放与标注另计。两条部署路线的分割／跟踪及推理耗时本就不同，不人为拖慢baseline，不将整条路线的差异单独归因于描边像素。

1. **选择本次目录并准备场景（历史模板）。** 下方变量保留第8次的历史取值，相关home、scene_snapshot和run目录均已有记录。未来如另行安排试验，先改用新的实验／尝试目录，再按[归位流程](#2-归位)准备；需要保存归位日志可加`--output "$TRIAL_ROOT/home"`。变量只作用于当前终端，旧运行和快照目录不得覆盖。

   ```bash
   P2_ROOT='artifacts/live/stack/stack_p2_paired_comparison_20261025'
   TRIAL_ROOT="$P2_ROOT/pair_04/08_target"
   P2_TARGET_ID='C'
   ```

2. **拍照并核对当前场景。** 使用[共用快照命令](#静止快照与种子导出)，输出到`$TRIAL_ROOT/scene_snapshot`。baseline拍完后只核对三根位置与A／B／C角色；target还须完成该页的三根可见区域审核，并将本次文件保存为`$TRIAL_ROOT/scene_snapshot/selection.json`。重摆后始终重新拍照；target不能复用上一条目的快照或圈选。

3. **只查看对应模型的计划。** 以下命令没有`--execute`，不连接设备或GPU；target必须等本次新快照和人工审核文件存在后才能核对。

   baseline：

   ```bash
   PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
     --model baseline-stack --cycles 1 \
     --transport-age-aligned --latency-tolerant --max-seconds 120
   ```

   target：

   ```bash
   PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
     --model target-stack --cycles 1 \
     --target-snapshot "$TRIAL_ROOT/scene_snapshot" \
     --target-selections "$TRIAL_ROOT/scene_snapshot/selection.json" \
     --target-id "$P2_TARGET_ID" --target-preview --track-target-motion \
     --transport-age-aligned --latency-tolerant --max-seconds 120
   ```

4. **现场启动一次。** 在该模型的同一命令后添加`--execute --output "$TRIAL_ROOT/run"`，核对终端计划与当前场景后按运行器提示启动。单次完成即停止，不自动归位；Ctrl+C停止及夹爪保持规则沿用[现场流程](#目标条件模型单次真机试验)。每次完成或中断后先保存人工反馈，再按表重摆并进入下一条目。首轮不自动补试；如因相机／通信／控制故障暂停，记录原因，修复后补试须用新编号，不能抹掉失败。

5. **分别记录软件与实物结果。** 在结果表按`trial_id`填写：实际可选数k、复摆差异、实际选中／夹起的条、目标一致、夹住、是否单根、是否送达、邻条扰动、中断及原因，并给出操作者或录像时间依据。`true`表示确认是，`false`表示确认否，`null`表示未知；未运行保持`not_started`。`complete.json`只作软件证据，不改写其`physical_grasp_success`，也不据周期完成填物理成功。

汇总按模型给出启动／未启动、真／假／未知计数；只在实际k≥2时比较目标一致。baseline称“与预设评价目标一致”，target可报告指定目标服从。共同的单根送达成功要求夹住、单根、送达都为真，指定目标送达另要求目标一致；baseline选了另一根但单根送达，仍保留其通用抓放成功。这里仅有**2个独立布局**，不得把8次运行当成8个独立场景或宣称稳定成功率。两组都抓得差时保留有限结论，不无限追加P1修复；本轮策略录像留作评价，四根／六根训练数据继续通过人工遥操采集。

##### 四根与六根动作数据扩充

**当前四根训练采集已完成；本轮准备原三根＋现有四根**：用户追加授权完成该范围的训练前准备，并明确新四根验证另行录制；六根不加入本轮候选清单。冻结GPU分割已获授权，包括与Ollama共享剩余资源；实际动作训练须最后单独明确授权，部署与自主抓取不在本次范围。四根30堆已完成采后检查，不以模型自主成功、自动选目标或自动成败判断为前置条件。现场相机、机械臂、归位和IO均由操作者手动执行，助手不连接设备。原1像素正式训练及后续评价的隔离目录、输出、冻结源码、数据和归一化保持；本轮另存候选清单和源码快照，复用已有数据与模型缓存。

**数据依据**：原动作训练来源为20堆／60条／40673帧，验证为3堆／9条／6023帧，新增四根数据尚未加入训练；相邻录像帧高度相关，帧数不代表独立摆放数。已有训练内姿态残差和新摆放退化同时存在，数据不足是合理假设，尚未证明是唯一瓶颈。

**研究参考与本项目适用边界**：[Data Scaling Laws in Imitation Learning for Robotic Manipulation（正文第3～4、7节）](https://arxiv.org/html/2410.18647v4)研究人工示教如何影响机器人在未见环境／物体上的泛化，即面对训练中没见过的条件仍完成任务的能力。作者分别改变环境数量、同类物体种类和每种条件下的示教次数，使用Diffusion Policy（一种动作生成模型），在倒水、摆放鼠标等任务中发现增加环境／物体多样性有益，而固定条件下增加重复示教的收益会逐渐减小；研究只覆盖该学习算法及四个任务，不能直接外推到所有机器人任务。

本项目借鉴的是采集预算分配思路：保持机位、竹条长度和支撑台不变，在五类上下结构、位置、朝向、交叉点及可行抓取顺序上增加变化，而非重复几乎相同的摆放。这是针对本项目的工程推论；论文主要研究跨环境／跨物体泛化，没有直接验证SmolVLA多竹条指定抓放，也未证明四根示教必然改善三根准确率。四→三→二→一的剩余状态来自本项目的采集流程本身。30个训练堆是用户确定的预算，五类各6堆及分批安排是工程建议，均非论文给出的阈值；论文中的数据规模和约90%任务成功率不作为本项目预期。新增数据审核及未来另行授权训练后，仍须用独立三根场景检验收益，当前1像素正式训练保持原来源。

**四根30个训练堆已完成，验证另录（2026-10-29）**：用户确定的30堆全部属于train，p-021～p-050现已完成；四根验证及六根采集数量尚未确定。五类各6堆是原均衡执行计划，实采p-041、p-044为一根压两根＋独立，按真实结构保留，不自动补录超出30堆预算。详细编号、来源和实际结构见[机器清单](../../artifacts/capture/stack/joint_stack_four_six_20261025/collection_manifest.json)。用户确认的节奏为同类型集中录完、整批检查后再切换；同类型内改变位置、朝向、间距和可行抓取顺序。原“每类10～20个新训练堆”不再作为本轮预算。训练／验证／测试在摆放前按整堆指定，同堆所有剩余状态不能跨集合；最终测试保留新布局，原验证集不称新测试。沿用40 cm竹条、原机位／支撑台／左侧放置区、`original-global`双640×480 RGB和七维绝对关节／夹爪契约，本轮保持一至两层。保留自然的末段调整、稳定夹持、搬运和释放，不为凑帧数延长等待。

**四根结构覆盖（2026-10-26用户补充）**：四根不是固定的“一对交叉＋两根独立”。用户明确包含以下五类；“两独立”已澄清为“两对交叉，共四根”。原计划五类合计30个训练堆已录完，实际分类及数量见下方总进度；后续仍按同类型整批指导及检查。

| 初始结构 | 初始可取数的摆放预期 | 主要覆盖 |
| --- | --- | --- |
| 四根独立、互不压住 | 4 | 多目标选择、不同位置／朝向，以及四→三→二→一的剩余状态 |
| 一对交叉＋两根独立 | 3 | 上层与独立条之间轮换目标，移除上层后的新可取状态 |
| 两对交叉 | 2 | 两个交叉上层之间选择，交叉对逐步解除 |
| 一根压在三根彼此分开的下层条上 | 1 | 一对多遮挡、上层移除后出现多个可取目标 |
| 两根互不压住的上层条压在两根彼此分开的下层条上 | 2 | 二上二下的关系；具体哪根压住哪根按实际摆放记录 |

表中数量以对应上层／独立条都能下指为前提，不能替代现场`available_target_count`。上层条数、局部下指空间和物理抓取结果分别记录；二上二下中的“独立”表示同层两根彼此分开，不表示与另一层没有接触。每堆在机器清单中分开保存计划和实际的身份／上下关系，抓取后按真实剩余状态继续；不能从`pick_step`或原计划推算关系。只有一个可取目标的步骤也应保留，它提供抓放与解除遮挡示教，只是不单独作为多目标服从证据。各类之间改变位置、朝向、交叉点及可行的抓取顺序，不必穷举所有组合；验证用独立重新摆放的堆，不把同堆剩余状态划到另一集合。

**对三根及以下的覆盖与限制**：每堆若依次完成四次单根移除，会在四、三、二、一根源区状态各录一条；本轮按操作者结果菜单及双路关键帧核对，30堆已保存120条，其中起始源区三根、两根、单根各30条，合计90条三根及以下示教；未保存重录和中断历史另外保留，不重复计入。同堆剩余状态相关，仍只有30个独立初始摆放，不能把90条当90个独立新布局。一交叉＋两独立先抓独立条可留下原三根“一交叉＋一独立”，先抓上层可留下三根独立；两对交叉先抓一根上层也可留下“一交叉＋一独立”，因此应轮换可行顺序以覆盖不同剩余结构。一压三先移除上层后提供三根独立，二上二下移除上层后的关系取决于实际接触。左侧已送达竹条和邻条扰动也会改变后续观察，不能视为与全新三根摆放完全相同。新增覆盖有望帮助三根及以下动作学习，但不保证修复已有末段对准、闭合或释放问题；原三根数据保持，未来另行授权混合训练后，须用独立三根场景检验收益，当前1像素正式训练不加入新数据。

- 所有同一初始堆及其逐根移除状态按整堆分组，再划分训练／验证／最终测试，不能拆帧跨集合。原9条验证继续用于开发诊断；因已反复查看，不称为新的盲测，最终真机评价另用未参与开发的新摆放。
- “合并所有数据”指合并审核通过、双RGB／七维绝对关节／夹爪与目标标注契约一致的训练示教。旧不同长度或任务、旧TCP动作数据不直接混入；静态分割照片不替代动作示教。自动抓偏录像保留为评价证据，不直接作为正确动作监督；人工接管段须另标来源和结果后再评估是否纳入。
- 录制提示与共享校验现支持`available_target_count=1～6`和任意正整数`pick_step`；保存、审计、标注清单沿用同一校验，原三根记录可读。一次结果菜单保留；未知字段为`null`，失败及中断保留，不从IO或夹爪周期推断物理成功。原始双RGB不烘焙紫色描边，掩码后续人工审核，以便派生不同描边宽度。
- 四根／六根采集复核完成后，混合清单、归一化、各数量采样比例、训练预算及真机评价再另行授权，不修改当前1像素正式训练。若以后研究描边收益，两组须使用相同扩充数据和预算，并分别报告三／四／六根效果。

**首堆试录入口回溯（已完成，以下为原执行前方案）**：[机器清单](../../artifacts/capture/stack/joint_stack_four_six_20261025/collection_manifest.json)保存计划摆放、固定集合和实际反馈，计划不冒充实采。首堆`p-021 / train`仍建议采用五类中的“一对交叉＋两根独立”试录，不代表后续全部采用该结构，实际摆放尚待确认。身份为`s-001=交叉上层`、`s-002=靠操作者的独立条`、`s-003=交叉下层`、`s-004=远侧独立条`；正常计划上层→近侧独立→下层→远侧独立。现场实际可取数逐条填写，不预填成功或默认剩余状态。第一条只录一次保存尝试，目标`s-001`、`pick_step=1`，若上层与两根独立条均可下指，`available_target_count=3`。同堆身份不变，之后不恢复初始摆放；失败重试沿用目标ID并递增已保存序号。先保留一张初始俯视照用于身份核对，不在竹条上贴标记。

```bash
# 由操作者执行；先检查相机输出预览，确认两路角色、非黑帧与报告通过。
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/diag_preflight_cams.py \
  --camera-set original-global --seconds 15 \
  --output artifacts/capture/stack/joint_stack_four_six_20261025/camera_preflight_p021

# 首条试录已完成，以下仅供回溯，不再执行该已占用目录。
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/record_joint.py \
  --camera-set original-global --stack-demonstrations --split train --num-episodes 1 \
  --dataset-root datasets/stack/train/joint_stack_four_train_p021_step01_20261025 \
  --evidence-root artifacts/capture/stack/joint_stack_four_six_20261025/p021_step01 --record
```

程序内填`p-021`、`s-001`、`1`和现场实际可取数；`r`归位，`→`开始，抓一根送到原左侧放置区、松开后`→`结束，按原一次结果菜单记录。首条结束即退出并整理视频；保留实际剩余摆放，先反馈结果并采后检查，再给下一条命令。若相机中断，先核对已保存／未保存状态，使用新目录续录，不能重跑覆盖旧路径。

**p-021续录回溯（已完成，不重跑）**：首条保存646帧后，以下3次保存已完成，续录共1970帧。两目录合计4条2616帧，双路视频完整解码、时间对应、状态／动作及每条一次夹紧释放检查通过；菜单均登记成功。`--num-episodes 1`只用于首条链路检查，不表示一堆只录一根。四条身份／序号和同堆剩余状态已核对，当前进入下方整批录制。

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/record_joint.py \
  --camera-set original-global --stack-demonstrations --split train --num-episodes 3 \
  --dataset-root datasets/stack/train/joint_stack_four_train_p021_from_step02_20261026 \
  --evidence-root artifacts/capture/stack/joint_stack_four_six_20261025/p021_from_step02 --record
```

同堆编号始终为`p-021`。若每次单根成功且剩余条均可下指，正常顺序为`s-002`靠操作者的独立条（step 2，k=3）→`s-003`原交叉下层（step 3，k=2）→`s-004`另一根独立条（step 4，k=1）；k仍按现场实际填写。每条先`r`归位再`→`开始，送达松开后`→`结束并填写原结果菜单；不重摆剩余场景。若失败，保存实际结果，重试同根时保留ID并递增`pick_step`，不机械套用正常顺序；3次保存尝试不保证清堆，未完成部分反馈后另开续录目录。

**当前状态：四根独立p-022～p-027已完成采后检查（2026-10-26）**。本类型共6堆24条15617帧，全部为train。p-025～p-027的最后一批已完整收尾；p-026第1条有一次作废重录，原记录保留，未混入保存数据。本类型的旧录制目录均已占用，不再重跑。详细结果见[本批检查](../../artifacts/capture/stack/joint_stack_four_six_20261025/independent_p025_p027_restart_20261026/capture_check.json)和[整类图检及顺序记录](../../artifacts/capture/stack/joint_stack_four_six_20261025/independent_p025_p027_restart_20261026/recording_review/batch_review.json)。

| 四根独立训练堆 | 已保存条数 | 帧数 | 确认后的实际初始位置抓取顺序 |
| --- | --- | --- | --- |
| p-022 | 4 | 2717 | 2→4→1→3 |
| p-023 | 4 | 2753 | 4→2→1→3 |
| p-024 | 4 | 2611 | 1→3→2→4（后两根现场调整） |
| p-025 | 4 | 2460 | 3→1→4→2（后两根现场调整） |
| p-026 | 4 | 2498 | 2→4→3→1 |
| p-027 | 4 | 2578 | 4→1→3→2 |

位置以操作者视角的初始左到右1～4为准，同堆不重新编号；表中实际顺序对应本堆记录的`s-001→s-002→s-003→s-004`。p-024、p-025在邻条被碰动后调换最后两根，已获操作者明确确认；原计划与实际角色在[机器清单](../../artifacts/capture/stack/joint_stack_four_six_20261025/collection_manifest.json)分开保留，不修改原始目标ID、录像、动作或结果菜单。后续画身份种子时按每条实际抓取的竹条核对，不能照搬旧计划位置。

**检查结论与范围**：全部保存批次通过双640×480 RGB完整解码、视频时间／Parquet对应、七维状态动作／接受命令及每条一次夹紧释放检查；跨目录序号均为1～4，划分始终为train，无重复保存序号或残留临时PNG。助手已查看本类型24条的双路起始／闭合／结束图，源区各堆均按4→3→2→1→0继续并清空。图中可见p-022第1、2条、p-024第2条、p-025第2条及p-026第1条邻条旋转／移位，后续保留变化后的场景；这些观察单独登记，原始`neighbor_disturbed=null`保留。不因扰动自动删除或判失败，当前未发现需因存储或抽样场景连续性问题重录的条目；菜单均为操作者一次确认的单根成功，抽样图检及IO周期不替代全段物理结果或逐像素标注审核。

**当前总进度与后续（2026-10-29）**：四根新训练采集已完成30／30堆，p-021～p-050共120条79200帧，全部为train。16个来源目录逐条对账，无重复保存条目；每堆保存序号1～4、整堆划分一致，双路存储与关键帧检查通过。原三根数据保持独立，本轮数据尚未进入任何训练。详见[全量采集检查](../../artifacts/capture/stack/joint_stack_four_six_20261025/collection_check.json)。

| 实际初始结构 | 训练堆数 | 保存条数 | 帧数 |
| --- | --- | --- | --- |
| 四根独立 | 6 | 24 | 15617 |
| 一交叉＋两独立 | 6 | 24 | 15717 |
| 两对交叉 | 6 | 24 | 16658 |
| 一根压三根 | 4 | 16 | 10784 |
| 一根压两根＋独立 | 2 | 8 | 5202 |
| 两上两下 | 6 | 24 | 15222 |
| 合计 | 30 | 120 | 79200 |

应用操作者已确认的6处独立更正后，现场可取数k≥2共75条；另两处已写回的更正不重复应用。120条结果菜单均登记单根成功，原始邻条扰动未知字段保持null，IO及关键帧检查不替代全段物理结果或逐像素标注。24次未保存重录、5次未保存中断记录保留，不计入120条；旧p-030的734张未保存临时PNG已于2026-10-29按用户授权清理，保存视频、数据表和侧录未改，见[清理记录](../../artifacts/maintenance/storage_cleanup_20261029.json)。新验证另录；当前目标掩码审核、60000步动作训练及既有验证评估已完成，状态与局限见下方。

**全量RGB与冻结分割候选已齐备（2026-10-29）**：[输入清单](../../artifacts/capture/stack/joint_stack_four_six_20261025/annotation_input_manifest.json)覆盖16个来源、120条的40398个接近帧；其此前“39099张待提取”是CPU索引阶段快照。现已复用p-021的1299张RGB并提取其余图片，使用当前第14轮best冻结分割生成全部实例候选，0优化步。原三根12个已确认标注来源原地复用；远端缺少的四根原始数据与必要侧录一次增量传输308个文件、3700582319字节，未重传原三根、展开RGB或权重。远端直接从这些原视频生成分割输入，后续训练继续引用同一份数据。

**远端提取RGB存储状态（2026-10-31）**：按用户指定清理，`gpu`上这16批40398张标注用提取RGB已删除，空`images/`目录保留；本地RGB、原视频、Parquet、目标掩码、实例候选及标注JSON保持。训练继续从原视频解码并读取掩码；远端复核页面看图或再次做依赖提取RGB的检查前，须按保留`annotations.json`中的`image`、`video`、`video_time_s`恢复指定RGB，勿对已有标注目录重跑`prepare`。空间与核验结果见[清理证据](EVIDENCE.md#目录与退役代码清理)。

**首帧种子已确认并导出**：用户反馈“没有问题，可以导出”，按本版完整首帧上下文记录120条、480个可见身份区域确认；16份JSON已按来源导出到原`pile-json--/stack/`对应目录并经原接口导入。[下载确认JSON压缩包](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/visible_seed_completion/confirmed_visible_seed_exports.zip)与[导出／导入检查](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/confirmed_visible_seed_exports.json)保留用户原话、对话来源和首帧绑定，多边形未改，审核用时及人工修改数量未知；不声称浏览器下载或逐帧点击，该次首帧确认的范围不含整段，后续整段确认另记如下。

**整段轮廓已确认，全部新来源接口验收通过**：用户反馈“复核完毕，效果可以”，据当前完整序列上下文登记120条、40398帧确认，含46帧RGB辅助修订；确认时掩码及480个首帧身份区域不变。16份[全序列确认JSON](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/full_sequence_review_exports.zip)绑定当前帧ID与修订号；原21972条问题提示（含助手另标的9处疑似局部漏标）保存在导出与审核历史，不代填已知误差接受、逐帧点击、审核用时或像素真值。经原LeRobotDataset／PyAV按episode并行检查全部79200帧，接近图像与标注源、训练／运行时图像及阶段、7维状态／动作、50步动作块及尾部填充均一致，见[全部来源接口验收](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/all_source_interface_check.json)。[原复核入口](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/sequence_review/index.html)及[统一准备入口](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/review_index.html)已同步；原始视频、数据表及身份／顺序未改。

`approach_manifest`与`prepare_target_source`共用`reviewed_stack_metadata`读取各来源`recording_review/metadata_corrections.json`：6处独立更正按`dataset_root＋episode_index`应用，校验场景／目标／保存序号和原值，并在派生标注与导出中保留原值与确认来源；已写回的2处不重复覆盖。更正来源身份在本地与远端路径变化后保持一致，远端已核对实际6处原值1／有效值2。`verify`汇总使用有效元数据，原session、侧录及旧采后报告保留。54项针对性CPU检查通过（目标准备／接口34项、训练清单20项），原[CPU索引检查](../../artifacts/capture/stack/joint_stack_four_six_20261025/annotation_preparation_check.json)保留阶段快照，当前状态见上方就绪入口。

**三根＋四根60000步训练完成，既有验证关节误差改善**：从原SmolVLA基座、2px目标描边、60000步、batch 8、seed 42、学习率1e-4完成训练，未续训或替换部署模型。实际使用25个训练来源、50堆180条119873帧；验证仍为原三根3堆9条6023帧，整堆无交叉，训练归一化与保存统计一致。训练循环22611秒（约6小时17分），含初始化、保存、严格重载及完整验证共23080秒（约6小时25分），于2026-10-29 17:02（America/Los_Angeles）正常退出；60000条损失／梯度／学习率记录均有限，首100／末100步平均损失1.41086／0.01849，final重载预测逐值一致、形状8×50×7。模型保存在远端`/home/rentao/program/lerobot-aubo-joint-stack/outputs/joint_stack_three_four_target_smolvla/final`，见[完成记录](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/training_launch.json)及[CPU评估](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/evaluation/result_evaluation.json)。

本次新旧模型的6023份保存首行动作已按来源、episode、frame、状态和专家动作逐值对齐，并与原Parquet核对；全部验证指标复算一致。下表为六关节平均绝对误差，单位度、越低越好，不是闭环抓取误差。

| 验证范围 | 原三根2px／30000步 | 三根＋四根2px／60000步 | 误差下降 |
| --- | --- | --- | --- |
| 全部6023帧 | 0.18239 | 0.13523 | 25.9% |
| 接近2998帧 | 0.17196 | 0.11615 | 32.5% |
| 搬运3025帧 | 0.19273 | 0.15414 | 20.0% |
| 闭合前后5帧，共99帧 | 0.20318 | 0.12254 | 39.7% |

三堆分别均改善，J6的全验证MAE从0.41917°降到0.27034°；[训练曲线与阶段对照图](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/evaluation/training_and_validation.png)保留实际数据。夹爪总体准确率从99.585%变为99.485%，均低于仅复制当前开闭状态的99.701%；闭合／释放切换当帧命中为0/9、0/9（旧2px为0/9、1/9），持位误切换8→13/6005。各episode保存首行动作序列的9次闭合阈值越过均在专家事件后一帧，释放8次后一帧、1次前一帧；这些是不同专家观察下首行动作的拼接，不能等同真实切换时间或断言未来动作块不会闭合。

**完整动作块与同帧目标响应复核完成**：用户授权“可以，进行复核”后，用固定权重新增371次离线GPU预测，0优化步，无硬件或部署。沿用63份既有观察×3推理seed，新模型189份完整块与旧2px的189份缓存配对；另对6堆四根独立条训练场景和3堆三根验证场景做30种同首帧目标输入、两模型共180份预测，加2份控制。旧1px素材未直接输入2px模型：从原视频与掩码重建对应输入，原1px像素逐值一致，状态／专家50步动作／padding一致后才生成实际2px输入。旧缓存重放及同输入重复的最大差均为0，见[复核页](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/evaluation/action_chunk_review/review.html)与[机器结果](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/evaluation/action_chunk_review/comparison.json)。

闭合前主指标沿用`before10`观察未来第8～19行“每行六关节最大绝对误差”的均值，不与上表首行动作六关节MAE混用。原三根验证3堆9份观察×3seed：0.71638°→0.38272°（下降46.6%），25/27配对、8/9观察均值改善，p-103上层观察0.49232°→0.68363°退步；角色均值上层／下层／独立条分别0.66470→0.47193、0.92073→0.31507、0.56371→0.36115°。所选训练内9份观察主指标只下降约4.0%（0.40671→0.39054°），不能包装为全训练集或四根独立验证。

| 完整块复核项 | 原三根2px | 三根＋四根2px |
| --- | --- | --- |
| 验证before10连续3行闭合的时机绝对误差（秒） | 0.34963 | 0.25185 |
| 验证before10首个闭合段仅1～2行／27份预测 | 10 | 17 |
| 三个闭合前时刻合计短脉冲／81份验证预测 | 17 | 28 |
| 搬运中点完整块每行最大关节误差均值（度） | 1.57161 | 1.70339 |
| 释放前10帧持续释放时机绝对误差（秒） | 0.20148 | 0.17926 |
| 释放当帧持续释放时机绝对误差（秒） | 0.26963 | 0.32444 |

临近闭合`before10`两模型27份预测均能在未来块内连续闭合，但新模型短脉冲与闭合后重新打开行数增多（44→53）；`before40`有2/27份新预测在50行内没有闭合，均为p-102 episode 2的seed42／43，旧模型为0。它们是固定专家观察下的预测，不能直接当真实闭爪延迟或抓取成功率。[逐条夹爪图](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/evaluation/action_chunk_review/gripper_before10.png)和[汇总图](../../artifacts/capture/stack/joint_stack_four_six_20261025/training_preparation/evaluation/action_chunk_review/review_summary.png)保留原始现象。

同一首帧、状态和腕部RGB不变，仅换已确认且当前可选的目标轮廓，新旧模型均在117/117组成对比较中产生非零输出差异（数值阈值1e-5°）。新模型四根训练场景第8～19行换目标差异均值0.12570°，同目标换seed差异0.18372°；原三根验证分别0.13078／0.19417°。可确认模型对目标提示敏感，不能将响应幅度或对原目标轨迹的偏好当换目标正确率；没有同状态替代目标专家轨迹，且只测首帧，未证明末段目标跟随或四根泛化。

**当前判断与下一步**：关节末段拟合改善，但夹爪稳定性与部分搬运／释放完整块指标退步，当前保留离线候选，不直接替换运行模型。短闭爪脉冲及p-103上层退步样本仍待独立定位，四根独立验证仍待补录。新模型先完成一轮操作者确认成功的独立条试验，随后上层A→下层B→独立C的同场景三次尝试均被操作者确认失败，A/B由人工取走、C留在源区；程序周期完成不计为物理成功，见[真机复核](#三根四根模型试用准备)。同配置第三轮在另一新布局的首个上层A又抓偏，操作者及时中止，B/C未尝试。当前优先诊断闭合前的横向／姿态对准和稳定夹持，保留成功与失败两类现场证据，不据单次成功宣称已修复，也不把后续人工干预或目标切换当最初失败的唯一解释。数据量、步数及归一化同时变化，不能把改善单独归因于四根数据或认定抓偏解决。原三根、1px模型及当前部署保持。

**现场失准链路CPU复核（2026-10-30）**：已检查四轮6次抓取尝试的5504条保存指令，未发现关节／夹爪动作行拼接错误；闭爪前接近模型目标仍不代表对准竹条。新发现第二轮A/B/C及第三轮A实际闭爪均选中模型块内1～2行短闭爪段，取行器会将其作为正式闭爪；第四轮误带两根则选中持续闭合段，需与前四次失准分别诊断。当前同时检查感知上游、末段模型位姿和闭爪触发，不能将人工框偏小或短闭爪脉冲单独认定为全部失败原因；CPU对照未证明任何修复。证据、对照图与限制见[现场失准链路复核](EVIDENCE.md#三根四根现场失准链路复核)。

**感知链路专项复核（2026-10-30）**：286份已处理接近输入的分割／身份关联CPU重放均与现场保存轮廓一致，未见帧身份串线；第四轮末段却存在光流更新连续失败、沿用旧运动参考、实例合并拆分及前景支持不足仍继续预测的降级路径。最终A参考在源图时已约4.36秒未成功更新，参考覆盖比例约38.8%（不是分割准确率），模型提示在夹爪附近明显不规则。其他失败并非都有同类异常；短闭爪也可能是上游输入变化后的反应，不能据此排除感知。代码实际将支持不足作为诊断，原启动提示“支持不足时停止”不准确。

**当前修复与待验收项**：先以本地提交`9a70aae5`保存本轮代码、文档和已导出标注；未推送，Git忽略的录像／模型等不属于该提交备份。之后仅修正启动说明，并让只读预览明确显示所选目标的跟踪失败／沿用旧参考、前景支持不足、合并实例拆分和参考覆盖比例；这些提示不代表人工像素验收，模型输入、关联阈值、闭爪取行及停止行为保持原样。34项针对性CPU检查通过，尚无真实抓取改善证据。

**感知取帧与人工确认**：操作者指出原请求43中竹条已翘起，CPU回看也见闭合指令之前的明显转动；不能用43单独判断感知是否先导致抓偏。43／53及原局部草稿保留为翘起后感知退化的补充。审核前移到[请求30](../../artifacts/live/stack/stack_three_four_20261030_trial04/review/perception_mask_repair/request_30/review.html)（较早接近）和[请求34](../../artifacts/live/stack/stack_three_four_20261030_trial04/review/perception_mask_repair/request_34/review.html)（遮挡及实例拆分阶段），双路录像源时间戳均精确匹配；它们早于所观察到的明显转动，确切首次接触仍未知。操作者随后反馈“我看过没有错误”，已按对话确认原样保存两帧的`selection.json`，详见[确认记录](../../artifacts/live/stack/stack_three_four_20261030_trial04/review/perception_mask_repair/review_confirmation.json)。两帧与现场原掩码的像素差均为0，故原轮廓／修正轮廓推理对照已关闭，无需再次圈选或导出；确认不扩展到其他帧或物理成功，原草稿仍保留未审核历史。

**当前诊断重点：末段位姿、纠偏覆盖与执行时序**：全局描边提示目标身份，双RGB动作模型仍需自行预测正确关节位姿；接近模型关节目标不等于夹口对准。五轮新增诊断已核对完整动作行和归一化统计，优先检查“观察到偏位／偏角后，是否预测出能纠正它的动作”，并把初始抓偏与其后扰动、跟随失败、漏抓后的目标切换分开。`n_action_steps=1`不是当前自定义入口固定执行首行的保证：适配器直接调用`predict_action_chunk`返回完整块，运行器按观察年龄选完整七维行。当前视觉冻结／仅训练动作专家也是[SmolVLA原论文](https://arxiv.org/html/2506.01844v1)的训练路线，本身不能定为配置错误。详细证据与限制统一见[固定诊断入口](../../artifacts/live/stack/stack_three_four_four_pick_fresh_layout/review/findings.json)。

优先方案是固定当前机位，建立按完整场景划分的四根独立评价，再审查并补充与当前限速匹配的“偏位／偏角→主动纠正→对准后闭爪”及邻条扰动后的剩余条恢复示教；失败策略动作不能直接当正确监督。先只改变数据覆盖，保留原模型、训练一个隔离候选并按相同场景／分阶段指标验证；采集、训练和部署尚未开始，各自需要后续明确指令。[NXP实际部署经验](https://huggingface.co/blog/nxp/bringing-robotics-ai-to-embedded-platforms)强调固定相机、位置／旋转多样性及恢复示教，[DAgger原论文](https://arxiv.org/abs/1011.0686)解释策略动作会改变后续观察分布。相对关节动作、增强目标条件、RTC及闭爪稳定性只作为后续单因素候选：需要匹配训练／统计／反变换及离线对照，不把旧绝对动作权重直接切成相对动作，不把RTC当错误空间目标的修复。

**腕部相机新基准（2026-10-30）**：用户重新调整相机，指定以当前画面为准，并确认机械臂断电但仍在起始位置；由操作者运行独立双相机体检取得[加固前腕图](../../artifacts/live/stack/wrist_camera_reference_20261030/camera_only/grasp_rgb.jpg)，没有新的关节读数。最初[CPU对照图](../../artifacts/live/stack/wrist_camera_reference_20261030/wrist_reference_comparison.png)及[机器结果](../../artifacts/live/stack/wrist_camera_reference_20261030/reference_check.json)显示，加固前两侧夹指金属细节相对历史trial04共同向右约32px；与抽查的10月25日四根训练首帧共同细节约差横向1～2px、纵向3～4px，与10月28日样本仍约差横向30px。这些数值只对应加固前图，不代表所有训练数据统一或抓偏已解决。

用户随后加固相机，再次采图；**当时参考改为[加固后腕图](../../artifacts/live/stack/wrist_camera_reference_20261030/after_fastening/grasp_rgb.jpg)**，此前图像及报告保留为历史。[加固前后对照](../../artifacts/live/stack/wrist_camera_reference_20261030/fastening_comparison.png)与[检查结果](../../artifacts/live/stack/wrist_camera_reference_20261030/fastening_check.json)确认小幅成像变化：左侧约向左12px、向下5px，右侧约向左9px、垂直基本不变；四个金属局部模板相关分数均大于0.98，两次配置记录一致。加固后的训练视角比较见下段，不沿用加固前的1～2px结论。当时保持该机位继续排查视觉到抓取位置的映射，之后再次微调与两轮结果见下段；助手当轮仅复核保存图像，没有新增GPU预测或修改模型、输入及控制规则。独立相机入口与操作方式统一见[相机配置](#相机配置)，断电时不使用需要读取机器人状态的影子快照入口。

**加固后腕图与实际训练数据对照**：按混合模型实际清单25个训练来源，对180条episode各解码一张精确PTS的起始腕图；[代表样本对照](../../artifacts/live/stack/wrist_camera_reference_20261030/training_comparison/representative_comparison.png)、[25来源总览](../../artifacts/live/stack/wrist_camera_reference_20261030/training_comparison/all_source_starts.png)和[完整数值](../../artifacts/live/stack/wrist_camera_reference_20261030/training_comparison/result.json)以加固后图为零点。原三根60条中51条满足描述性的接近规则，代表样本夹指横向约差2～5px；四根120条中16条在当前图右侧约9～11px，67条在左侧约19～26px；其余三根9条、四根37条匹配不足或属其他情形，保留不确定。此为夹指金属局部图像位置，不是竹条抓偏距离或训练数据质量分级；只覆盖每条首帧，未检查整段机位稳定。它确认训练腕图存在不同成像参照，不能单独认定为失准根因，不据此删数据、自动筛选训练或添加补偿。

**当前固定机位与两轮三根失败（2026-10-31）**：用户再次微调腕部相机并明确固定不再移动，当前参照为[新腕图](../../artifacts/live/stack/wrist_camera_readjustment_20261031/capture/grasp_rgb.jpg)，[对比记录](../../artifacts/live/stack/wrist_camera_readjustment_20261031/comparison.json)显示较上次加固后左右金属主体分别位移(−4,+1)／(−5,−5)px，不等于标定或抓偏修复。原60000步模型、2px和既有控制参数下，两次三根实验均在A阶段人工中断：首轮[反馈](../../artifacts/live/stack/stack_three_four_readjusted_camera_three_pick_20261031/operator_feedback.json)为夹带两根；重试明确为接触前位置／角度不对，B/C均未尝试。重试[复核](../../artifacts/live/stack/stack_three_four_readjusted_camera_three_pick_retry_20261031/review/findings.json)确认发送目标匹配完整模型动作行，重预测存在但没有完成有效精对准；后段目标移动、跟踪失败和轮廓混杂不能直接认定为最初原因。[相近方向示教对照](../../artifacts/live/stack/stack_three_four_readjusted_camera_three_pick_retry_20261031/review/p010_demonstration_vs_retry.jpg)表明训练中存在相近目标方向，不能简单归为从未见过该角度，也不能将两场景关节差当补偿。当前场景和两轮快照不可复用执行，保持机位，继续区分视觉到动作映射与执行动态；未重训、修改控制或新增GPU预测，细节见[EVIDENCE](EVIDENCE.md#三根四根感知链路复核)。

**固定机位再次重复：A/B均失败（2026-10-31）**：操作者确认“都没抓到”，C未尝试；A仅完成软件开闭爪周期，随后B阶段中断。[本轮复核](../../artifacts/live/stack/stack_three_four_fixed_view_repeat_20261031/review/findings.json)核对274份预测与2411条目标，均对应原模型完整动作行；两次闭爪前最后状态距模型关节目标最大约0.147°／0.107°，不等于正确抓取位姿，也不排除接近过程影响。[A输入](../../artifacts/live/stack/stack_three_four_fixed_view_repeat_20261031/review/A_input.jpg)／[B输入](../../artifacts/live/stack/stack_three_four_fixed_view_repeat_20261031/review/B_input.jpg)抽查轮廓仍对应目标，但最终接近没有形成可靠夹持对准。最终采用的闭合段均持续至块尾，起始腕图局部参照仅差1～2px，不能简单归因于单帧闭合或相机再次移动。暂停无改动重复实抓，优先区分已学到的精对准、现场输入差异与执行动态；当时尚未执行新的固定权重对照；后续49份对照结果见下段，仍未证明根因或修复。当前快照不可复用。

**精对准拟合与执行时序的CPU复核**：[结果](../../artifacts/live/stack/stack_three_four_fixed_view_repeat_20261031/review/fit_execution_diagnosis/findings.json)／[对照图](../../artifacts/live/stack/stack_three_four_fixed_view_repeat_20261031/review/fit_execution_diagnosis/fit_and_execution.png)复用原模型完整动作块、成功示教和现场日志。闭爪前0.4秒的9个训练输入（三场景、每帧三seed），模型关节MAE约0.147°，保持当前关节仅0.00073°；对应9个验证输入为0.140°／0.00084°。这些输入已由人工对准并停住，低绝对误差不能证明从偏位纠正能力；较早1.6秒验证窗口模型0.231°优于保持0.412°，说明存在部分接近行为。p010的50×7人工动作与原Parquet逐值一致，模型仍出现末段关节偏移。现场A/B执行时腕图平均年龄0.613／0.580秒，最后两秒输入状态与当前状态最大关节差均值约0.34／0.33°；最终到位不能排除过程中时序影响。已确认评估覆盖与精对准拟合缺口，尚未分离视角／模型／执行的因果贡献；不据此改速度、补偿或固定等待。仅分析已保存预测，无新增GPU、训练和真机执行。

**主动运动状态固定权重复核与全量时序检查（2026-10-31）**：用户明确授权的49份原模型预测＋2次预热已完成，[结果](../../artifacts/live/stack/stack_three_four_fixed_view_repeat_20261031/review/fit_execution_diagnosis/active_correction_probe/findings.json)／[腕角对照图](../../artifacts/live/stack/stack_three_four_fixed_view_repeat_20261031/review/fit_execution_diagnosis/active_correction_probe/active_correction_J6.png)显示熟悉示教输入中仍有转角幅度及保持误差：相对输入状态的1秒后J6目标，p010接近阶段人工约0°、模型三seed均值+0.752°；p103转腕状态人工+1.285°、模型+3.775°。这些运动状态含“实际关节追赶已接受目标”，不等同于人工持续增加转腕指令或受扰恢复。两个稳定输入×3seed完整块与历史预测逐值一致；JPEG配对平均变化较小但个别动作／闭爪时机变化，未证实为主因。实际训练清单180条、25来源按真实机器人状态时间戳检查，约0.2秒窗口仍有40.7%超过当前5°/秒限速，速度95分位11.94°/秒；示教帧至同帧命令写入平均约25ms，现场执行所用腕图约580～610ms，二者不能当同一种闭环条件。确认拟合与时序两项缺口，未分离各自抓偏因果贡献；保持速度限制，不据此加角度补偿或固定等待，未训练／连接机械臂／改远端代码。

**软件修复尝试与后续方案**：新增默认关闭的`--execution-paced`，以模型首行和到位后新观察替代“用观察年龄代表动作进度”；118项运行器检查及完整CPU回归868通过／11跳过。已保存的30份原图／现场重建预测对照显示，它可缩小部分目标跳变，但p010接近／保持首行的最大单关节标签误差仍达1.376°／1.055°，且到位加下次推理仍可能超过原1.5秒预算。因此该实现只保留为实验选项，尚未验收为抓偏修复、未上机。下一步优先用固定腕部视角、与执行器相同限速采集小批“偏位→人工纠正→正确夹持”的示教，按完整场景划分训练／验证，先验收状态／动作／图像和时间戳，再经单独授权训练隔离候选；不能只改时间戳把旧高速数据当慢速数据，也不能只用已经对准的停住帧证明纠偏能力。评估须分开接近、纠偏、保持、闭爪和搬运，最后以新场景指定单根实际抓取验收。[复核与步骤](../../artifacts/live/stack/stack_three_four_fixed_view_repeat_20261031/review/fit_execution_diagnosis/execution_paced_check/review.json)／[CPU结果](../../artifacts/live/stack/stack_three_four_fixed_view_repeat_20261031/review/fit_execution_diagnosis/execution_paced_check/results.json)。补充：当前目标模式推理传输本已使用`rgb_zlib`；此前JPEG95对照不能当作部署额外JPEG95压缩的证据，相机MJPG／录像编码是另一层问题。

**腕部视角适配与两组续训（已结束，候选已退役）**：只用现有示教完成37份固定权重视图检查，并从原60000步权重独立训练普通续训和腕部增强各10000步；[离线比较](../../artifacts/training/stack_wrist_view_adaptation/comparison.json)、[分阶段检查](../../artifacts/training/stack_wrist_view_adaptation/phase_comparison.json)与[498份冻结对比](../../artifacts/training/stack_wrist_view_adaptation/frozen_comparison/comparison.json)保留历史证据。总体误差的小幅改善主要来自搬运，闭爪前精对准没有稳定改善；四根没有独立验证。普通组现场A送达、B抓偏、C/D未尝试；增强组351份预测／3237帧指令／四个命令周期完成，但操作者确认只有A抓取成功。两组均未解决实际抓偏，不能视为有效修复。用户要求退役后，已删除两个候选权重、专属训练／增强代码及测试、隔离源码／预热权重／重跑脚本／打包缓存；只保留训练指标、原始现场数据与审核记录，详见[EVIDENCE](EVIDENCE.md#三根四根感知链路复核)。

**当前入口：原三根＋四根60000步模型**：`run_joint_smooth.py --model target-stack --target-outline-width-px 2`现默认选择`three-four`，也可显式加`--target-stack-model three-four`；绑定`joint_stack_three_four_target_smolvla/final`、SHA `80bde548…489a0`。参数名`original`仍表示更早的三根模型，不是本次恢复对象；1px与其他模型入口保持。两个续训选项已移除，不能继续调用已删除权重；当前场景已变化，不能复用旧快照执行。151项CPU假设备检查通过，四目标软件检查确认全部会话均使用原60000步权重，未新启动硬件、GPU推理或训练。[退役与清理记录](../../artifacts/live/stack/stack_wrist_model_comparison_20261031/candidate_retirement.json)列明删除路径和原模型核验；两轮录像、动作、轮廓及人工反馈完整保留，后续围绕接触前对准继续诊断。

用户另行授权的固定输入稳定性检查已完成，[实际记录](../../artifacts/live/stack/stack_three_four_20261030_trial04/review/pretilt_prediction_stability/execution_result.json)保存13份预测及重复控制。它确认同输入换随机种子会改变部分末段关节／闭爪输出，但未证明造成定位抓偏；保留为旁证，不据此改变当前控制规则。无训练、硬件或部署。详细结果与限制见[感知链路证据](EVIDENCE.md#三根四根感知链路复核)。

**一交叉＋两独立整类检查已完成**：本次末批从p-031第2条续录的7条4665帧完整收尾，双路视频完整解码、时间／Parquet对应、七维状态动作／接受命令及每条一次夹紧释放检查通过；p-031第2条有一次476帧作废重录，原session保留且不占保存序号。合并此前来源，p-028～p-032共20条13101帧；加上p-021，本类型共6堆24条15717帧。

| 一交叉＋两独立训练堆 | 保存条数 | 帧数 |
| --- | --- | --- |
| p-021 | 4 | 2616 |
| p-028 | 4 | 2502 |
| p-029 | 4 | 2524 |
| p-030 | 4 | 2723 |
| p-031 | 4 | 2769 |
| p-032 | 4 | 2583 |

助手已查看p-028～p-032全部20条的双路起始／闭合／结束图，初始结构均为一交叉对加两独立条，各堆源区按4→3→2→1→0连续减少，p-030、p-031跨目录的剩余场景一致，末图源区已清空；p-021复用已有图检。此次抽样未见明确新增的严重邻条扰动，原始`neighbor_disturbed=null`不回填。菜单均登记单根成功，但抽样图检及IO周期不替代完整物理结果或逐像素标注审核。当前未发现需因存储或抽样场景连续性问题重录的条目。旧p-030未保存第3次尝试的双路各367张、共734张临时PNG在该批检查时保留，后于2026-10-29授权清理，未混入保存视频。详见[本次CPU检查](../../artifacts/capture/stack/joint_stack_four_six_20261025/crossing_pair_two_separate_from_p031_step02_20261028/capture_check.json)、[整类图检](../../artifacts/capture/stack/joint_stack_four_six_20261025/crossing_pair_two_separate_from_p031_step02_20261028/recording_review/batch_review.json)与[跨目录存储核对](../../artifacts/capture/stack/joint_stack_four_six_20261025/crossing_pair_two_separate_from_p031_step02_20261028/recording_review/family_storage_check.json)。旧录制目录均已占用，不重跑。

**两对交叉p-033～p-038整类检查已完成（2026-10-28）**：6堆24条16658帧完整保存并正常收尾，全部为train；双路视频完整解码、PTS／Parquet、七维状态动作／接受命令和每条一次夹紧释放检查通过。全局视频正常分为3个文件，合计帧数与腕图一致，不是缺片。p-035第1条两次258／443帧、p-037第2条两次434／110帧、p-038第4条一次441帧作废重录，原session保留，均不占保存序号；本目录无残留临时PNG。

| 两对交叉训练堆 | 保存条数 | 帧数 |
| --- | --- | --- |
| p-033 | 4 | 2793 |
| p-034 | 4 | 2811 |
| p-035 | 4 | 2732 |
| p-036 | 4 | 2679 |
| p-037 | 4 | 2829 |
| p-038 | 4 | 2814 |

助手查看全部24条双路起始／闭合／结束图，各堆初始为两对交叉，剩余场景按4→3→2→1→0连续减少并清空；此次抽样未见明确新增的严重邻条扰动，原始未知字段保持null。菜单均登记单根成功；图检范围是关键帧，不能替代全段物理结果或目标掩码审核。当前未发现因存储或抽样连续性问题需重录的条目。详见[CPU检查](../../artifacts/capture/stack/joint_stack_four_six_20261025/two_crossing_pairs_p033_p038_20261028/capture_check.json)和[整批图检](../../artifacts/capture/stack/joint_stack_four_six_20261025/two_crossing_pairs_p033_p038_20261028/recording_review/batch_review.json)，旧目录不重跑。

**可取数6处更正已由操作者确认**：p-033～p-038每堆第2条原填1，漏算第一条移除后已解除遮挡的独立条，实际均为2。原session、episode侧录和原始CPU检查继续保留填写值1；[更正记录](../../artifacts/capture/stack/joint_stack_four_six_20261025/two_crossing_pairs_p033_p038_20261028/recording_review/metadata_corrections.json)保存原值、确认值、帧依据及操作者来源，机器清单逐条链接。后续可取数统计、标注导出及评价须按数据目录＋episode索引应用更正；本类k≥2条目由原统计8条更正为14条，不能直接沿用原`capture_check.stack_summary`。不改录像、动作、物理结果或ID，不因此重录；完整标注及训练来源准备仍未完成。

**p-039～p-044整批检查及实际结构更正已完成（2026-10-28）**：6堆24条15986帧完整保存并正常收尾，全部为train。双路视频完整解码、PTS／Parquet、七维状态动作／接受命令和每条一次夹紧释放检查通过；p-041第1条579帧、第3条410帧两次未保存重录保留，均不占保存序号，本目录无残留临时PNG。

| 训练堆 | 保存条数 | 帧数 | 核对后的实际初始结构 |
| --- | --- | --- | --- |
| p-039 | 4 | 2832 | 一根压三根 |
| p-040 | 4 | 2604 | 一根压三根 |
| p-041 | 4 | 2626 | 一根压两根＋右侧独立条（按图像视角） |
| p-042 | 4 | 2600 | 一根压三根 |
| p-043 | 4 | 2748 | 一根压三根 |
| p-044 | 4 | 2576 | 一根压两根＋左侧独立条（按图像视角） |

助手查看全部24条双路起始／闭合／结束图，各堆沿用剩余场景，源区按4→3→2→1→0减少并清空；此次抽样未见明确新增的严重邻条扰动，原始`neighbor_disturbed=null`保留。菜单均登记单根成功，抽样图检不替代全段物理结果或完整掩码审核。当前没有因存储或抽样连续性问题需要重录的条目，详见[CPU检查](../../artifacts/capture/stack/joint_stack_four_six_20261025/one_over_three_p039_p044_20261028/capture_check.json)与[整批图检](../../artifacts/capture/stack/joint_stack_four_six_20261025/one_over_three_p039_p044_20261028/recording_review/batch_review.json)。

**操作者确认并授权更正**：p-041、p-044各有一根未被上层压住，实际为“一根压两根＋独立”，首条可取数原填1、实际为2。已备份原记录，将两条保存episode侧录及session中的对应值改为2，同步检查报告的目标元数据、k≥2统计和复核页；其他保存／未保存记录、ID、结果、录像及Parquet状态／动作不变。原值、确认来源、备份及修改位置见[更正记录](../../artifacts/capture/stack/joint_stack_four_six_20261025/one_over_three_p039_p044_20261028/recording_review/metadata_corrections.json)。本批k≥2条目为14条；实际一根压三根为4堆16条10784帧，另外2堆8条5202帧按实际结构保留，不计作一根压三根。两处已保存元数据更正后核对侧录与session一致、共享序号合法；视频与状态动作的完整检查复用本批已有结果。

**p-045已完成并通过采后检查（2026-10-29）**：续录目录`two_over_two_from_p045_step03_20261029`已保存第3条638帧、第4条707帧，共2条1345帧，正常finalize，`capture_complete=false`。加上原目录前两条，本堆合计4条2578帧；保存序号1～4、目标标签s-001～s-004及整堆train划分一致。此次[CPU检查](../../artifacts/capture/stack/joint_stack_four_six_20261025/two_over_two_from_p045_step03_20261029/capture_check.json)通过双视频完整解码、PTS／Parquet、七维状态动作／接受命令与每条一次夹紧释放，旧目录检查复用，原数据不改。

助手合并查看本堆4条双路起始／闭合／结束图，第2条结束与第3条开始的剩余场景一致，源区按4→3→2→1→0减少，末图已清空；菜单均登记单根成功，原始邻条扰动未知字段保留。抽样范围及双路拼图见[整堆图检](../../artifacts/capture/stack/joint_stack_four_six_20261025/two_over_two_from_p045_step03_20261029/recording_review/batch_review.json)，不替代完整物理结果或掩码审核。截图停在下一条堆编号输入的`KeyboardInterrupt`，并有腕部相机后台OpenCV读帧／解码警告，具体根因未确认；没有新一条元数据、帧观察器、未保存尝试或p-046记录，本次无临时PNG待删，不重录p-045。此前452张临时帧清理和启动计划空壳清理证据仍保留在原目录。

**两上两下p-045～p-050整类收尾通过（2026-10-29）**：最后来源`two_over_two_from_p048_step04_20261029`保存p-048末条及p-049、p-050各4条，共9条5775帧，`capture_complete=true`、`dataset_finalized=true`。[CPU检查](../../artifacts/capture/stack/joint_stack_four_six_20261025/two_over_two_from_p048_step04_20261029/capture_check.json)通过双路全帧解码、PTS／Parquet（视频帧时间与数据表）、七维状态动作／接受命令及每条一次夹紧释放。旧来源检查复用，中断来源的原`capture_complete=false`不改写。

| 两上两下训练堆 | 保存条数 | 帧数 |
| --- | --- | --- |
| p-045 | 4 | 2578 |
| p-046 | 4 | 2354 |
| p-047 | 4 | 2588 |
| p-048 | 4 | 2611 |
| p-049 | 4 | 2644 |
| p-050 | 4 | 2447 |

助手查看p-048跨目录完整4步及p-049、p-050全部条目的双路起始／闭合／结束图，原剩余场景连续、各堆按4→3→2→1→0减少并清空；此次抽样未见明确新增的严重邻条扰动。p-049、p-050第1条各有一次479帧未保存重录，保留在session且不占保存序号，最新目录无残留临时PNG。具体范围与双路拼图见[末批图检](../../artifacts/capture/stack/joint_stack_four_six_20261025/two_over_two_from_p048_step04_20261029/recording_review/batch_review.json)，菜单、IO与抽样图检分别保留，原始未知字段不回填。当前未发现因存储或抽样连续性问题需要重录的条目；完整目标掩码仍待审核，非训练就绪。

**后续标注与采集**：四根30个训练堆的录制预算已完成，既有录制目录均已占用，不重跑旧录制命令。四根验证和六根数量待另行确定；验证使用新摆放整堆，不能从本轮训练堆中拆条。采后`capture-check`只首次生成，已有报告直接读取；后续`prepare`用于提取接近RGB和`inference_manifest.json`，不会启动GPU或自动接受掩码。全量标注输入准备、实际GPU草稿任务、人工审核及最终`verify`分阶段推进；导出和目标选择统计须应用上方确认的更正记录。混合训练、归一化及部署不在本次范围。

**新分割模型接入（2026-10-24）**：`target-stack`的`segmentation_checkpoint`已指向`/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/outputs/short_strip_mask2former_failure_finetuning/best`（第二轮混合微调第14轮best）。动作模型仍为`/home/rentao/program/lerobot-aubo-joint-stack/outputs/joint_stack_target_smolvla/final`，任务文字、七维状态、50×7动作块、时效与运动限制沿用trial04选项；`target-pair`和历史入口继续使用旧`short_strip_mask2former_baseline/best`。

保存图像的三次真实GPU调用已返回动作；检查未通过的原因是助手对一次性脚本的修改错误地向Python列表调用`.tolist()`，与机械臂是否上电无关。用户明确要求检查后删除该脚本和本次输出目录，已撤下复跑命令；检查结论保留在EVIDENCE。后续使用P1的新摆放、新快照及实时预览观察目标轮廓，不把本次缺失的轮廓图记为已经复核。

回退只把`live_setup.py`中`target-stack`分支的`segmentation_checkpoint`改回`SEGMENTATION_CHECKPOINT`（或删除该覆盖），动作配置不变；新权重、旧baseline和既有记录均保留。

#### 旧短续训与推理候选已退役

用户于2026-10-25明确删除7组1000步候选、两份插值权重及相关目录／脚本，专属训练入口与测试同步移除。旧目的、参数及结果已记在[EVIDENCE](EVIDENCE.md#短续训候选退役与1像素正式训练准备)，此处不重复指标，不再提供旧实验命令；原现场记录与运行模型保留。

#### 1像素描边正式训练

**2026-10-31可用性更新：用户已授权删除远端1px正式模型和无描边baseline对照模型的`final/model.safetensors`，合计释放2.23GiB。** `--target-outline-width-px 1`与`baseline-stack`旧入口需先恢复各自权重才能运行；下文训练结果和历史接口不表示模型当前可用。日志、配置、评价和预测保留，原2px、三根＋四根模型及当前π0.5训练不受影响。具体路径和核验见[固定清理记录](../../artifacts/maintenance/storage_cleanup_20261029.json)。

**当前状态：30000步训练、严格保存重载和离线评价均完成；原2像素默认模型保留，新1像素前两轮单独试A均报告抓歪失败，第三轮已授权按真机入口尝试三根连续流程，状态与证据见对应章节。** 汇总脚本的路径类型错误已仅用CPU修复，未重训或重复GPU预测。进一步CPU诊断显示不同场景改善与退化并存，当前重点为p-103等场景下的J6条件响应；具体指标与限制统一见[EVIDENCE](EVIDENCE.md#1像素同预算正式训练结果)及[诊断图](../../artifacts/training/stack_thin_outline_formal/evaluation/diagnosis.png)。[只读状态](../../artifacts/training/stack_thin_outline_formal/remote_status.json)与[比较报告](../../artifacts/training/stack_thin_outline_formal/evaluation/comparison.json)保存实际结果。从与原baseline／target相同的SmolVLA预训练基座开始，未接续1000步候选。独立目标输入结构暂停；单独试用选项与现场流程见[真机入口](#目标条件模型单次真机试验)。

**补充训练集评价已完成**：仅对比2像素与1像素两个final，完整原训练集固定权重推理及CPU复核均通过。1像素改善训练内关节拟合，既有验证集仍无一致优势，训练内短闭合脉冲也未改善；不替换运行模型。范围、实际耗时与指标统一见[EVIDENCE同节](EVIDENCE.md#1像素同预算正式训练结果)，原始汇总见[训练集比较](../../artifacts/training/stack_thin_outline_formal/training_set_evaluation/comparison.json)，关键动作块复核见[CPU诊断](../../artifacts/training/stack_thin_outline_formal/training_set_evaluation/diagnosis.json)。本轮未改变数据划分或追加训练。

| 项目 | 正式对照配置 |
| --- | --- |
| 输入 | 接近阶段全局图为1像素不透明紫色内边界，腕图原样；接受闭合后的搬运仍为原双RGB |
| 数据 | 原20堆60条40673帧训练、3堆9条6023帧验证，来源及归一化保持 |
| 训练 | 30000步、batch 8、seed 42、峰值LR 1e-4；1000步warmup，原余弦衰减及0.025最低比例 |
| 冻结与损失 | 原视觉语言主干冻结、动作专家及state投影训练；原七维完整动作损失／补齐屏蔽，无旧候选加权、辅助项或重采样 |
| 检查点 | 第5000、10000、20000步曾保存中途权重，第30000步保存final；三份中途权重已于2026-10-29清理，final权重于2026-10-31按用户授权删除；配置及已保存评价保留，不提供隐式续训 |
| 主比较 | 30000步final对照原30000步2像素target与无描边baseline；中途指标只观察走势，不事后挑最优替代主比较 |
| 离线评价 | final完整6023帧；63份原数据观察（36份接近＋27份搬运／释放）×3推理seed；中途检查点同口径观察走势，报告角色、姿态、持位、闭合与释放 |
| 资源 | 实际训练循环约3小时9分钟；含保存、评价及CPU汇总修复，约3小时23分钟完成，低于原6小时总时限；产物预留8 GiB |
| 本地准备 | `artifacts/training/stack_thin_outline_formal/` |
| 远端隔离代码／计划 | `/home/rentao/program/lerobot-aubo-joint-stack/artifacts/stack_thin_outline_formal/` |
| 新训练输出 | `/home/rentao/program/lerobot-aubo-joint-stack/outputs/joint_stack_target_thin_outline_smolvla/` |

[机器计划](../../artifacts/training/stack_thin_outline_formal/plan.json)登记完整基座路径、参考身份、参数、数据及评价；[CPU准备检查](../../artifacts/training/stack_thin_outline_formal/cpu_preflight.json)与[隔离传输记录](../../artifacts/training/stack_thin_outline_formal/staging.json)是启动前的准备事实。本次[明确授权](../../artifacts/training/stack_thin_outline_formal/authorization.json)后的任务已收官，已有输出不重复启动。状态用本地`.venv/bin/python artifacts/training/stack_thin_outline_formal/check_training.py`只读查看。相机、机械臂及部署不在本次范围。离线指标不等于物理抓取成功；验证仍只有3个已用于诊断的场景。

#### P1五堆摆放与结果记录

**2026-10-25状态：随P1暂停。** 下文保留此前摆放清单及执行事实；未完成的第2～5堆不计失败，旧流程不再作为P2前置要求。

2026-10-22确定三根固定配置实验清单，随后先进行了静态分割扩展；2026-10-24用户确认视觉阶段表现符合预期后开始执行。第1堆已保留6次运行，依次为非当前C支持门槛停止、腕部相机超时、当前B支持门槛停止、统一放宽视觉门槛后的两次A夹持失败，以及跟踪修复后仍闭爪前未对准的一次A失败；后三次均人工中断，见[结果与修改边界](EVIDENCE.md#p1首堆中止与当前目标检查修正)。最新摆放中的竹条已移动，重试须绑定新快照和人工审核标注；B/C本轮未尝试，不计为两根抓取失败。用户报告下探碰条并确认人工遥操能稳定抓起；已有相近成功示教对照及运动参考本地修复；用户选择先做跟踪修复后的真机实验，该轮现已失败；当前先诊断末段对准与反馈，纠偏行为和示教仍暂不优先。现场失败后已完成两组各1000步的离线续训对照，均未部署；未新采动作数据或增加高度／姿态补偿。第2～5堆未开展。本地运行代码以`70d6b4ef`为起点，动作权重和运行选项沿用[trial04流程](#目标条件模型单次真机试验)；新分割模型路径与首堆选项已记录在本轮计划。修正前后不合并为同配置成功率，也不与旧trial04混算。

共同条件沿用[三根场景规格](#三根堆叠目标示教方案)。左右、远近均按操作者现场视角描述；每堆重新摆放，仍限定在原支撑台可用区内。每堆开始前移走上一堆留在左侧的竹条，使工作区只保留本堆三根；同一堆运行中已送到左侧的竹条保留原位。A始终为交叉上层条、B为交叉下层条、C为独立条，顺序统一为A→B→C；C与交叉对不接触，操作者确认下指空间。

| 堆次 | 独立条C位置 | 交叉对摆法 |
| --- | --- | --- |
| 第1堆 | 交叉对右侧 | 交叉点靠两条中段；A相对B略向左倾 |
| 第2堆 | 交叉对左侧 | 交叉点仍靠中段；A相对B略向右倾 |
| 第3堆 | 交叉对右侧 | 交叉点沿条长向操作者方向小幅移动；A相对B略向右倾 |
| 第4堆 | 交叉对左侧 | 交叉点沿条长向远离操作者方向小幅移动；A相对B略向左倾 |
| 第5堆 | 交叉对右侧 | 交叉点回到中段，交叉对在原支撑台内小幅横移；A相对B略向左倾 |

“略倾／小幅”不要求毫米网格，仍受既有交角和支撑范围约束；每堆用新快照核对实际位置与身份，若空间不足则记录实际调整，不把表格当作已验收摆放。归位、快照、人工审核、计划检查和执行按[现场流程](#目标条件模型单次真机试验)由操作者逐堆完成，使用各自尚不存在的新输出目录。

结果统一记录在各轮固定`review/`中，结论汇入[EVIDENCE](EVIDENCE.md#三根现场实验进程)：

- **整堆**：清单堆次、运行目录、本轮快照、实际摆放调整、是否清空、中止及原因。五堆的失败和中止均保留，不另挑成功摆放替换；本批用于固定配置重复观察，不能单独建立描边优势或独立泛化结论。
- **逐根**：目标ID与角色、实际可选数k、抓对／夹住／恰好一根／送达、邻条扰动。正常逐根移走且未发生扰动时预计k为2／2／1，记录仍以当时场景为准；已知失败记失败，未执行或无法判断项记未知。按每项报告成功／失败／未知数量，目标服从只统计k≥2步骤。
- **日志量**：闭爪Z（mm）、J6（度）、前景丢失与恢复、耗时及软件完成状态。闭爪位姿沿用[trial04采样口径](EVIDENCE.md#第四轮三根成功与现场空目标恢复)，标注采样与IO时刻的偏差；Z为机器人坐标高度，不是竹条表面间隙。物理结果单独保留人工来源，不改写原始`physical_grasp_success`。

本清单随P1暂停保留，不再要求完成后才安排P2。恢复时须记录模型、参数及代码变化边界，修改前后的结果分开统计。

### 扩展到6根及以下

2026-10-23用户确认工作区最多容纳六根，八根暂缓。2026-10-24静态复核达到本阶段预期；2026-10-25用户暂停P1，改为先P2对照，再四根／六根动作示教及混合训练，见[当前路线](#p2对照与多根动作数据扩充)。已有静态数据主要覆盖一至两层，不是动作示教，更多层数另行验证。

**分割扩展顺序（2026-10-22用户同意，23日收窄数量范围）**：先复核4／6根静态新摆放的标注，再用现有冻结Mask2Former检查漏检、实例合并、可见分段归属和上下层误分类；该批用于发现问题，属于开发数据。再依据失败补充数量、遮挡结构和位置覆盖，对所选训练图标注全部可见竹条的实例与类别，用新旧训练数据共同微调并保留原模型；不只增加旧数据的训练轮数。划分按整堆进行，用未参与开发的新测试摆放比较新旧模型，同时检查原有1～3根场景是否退化。先完成静态感知评估，再进入多根抓放；新增位置或支撑高度下若动作能力不足，另补动作示教。分割训练只需图像与完整实例标注，现有仅标指定目标的动作掩码不能直接充当完整分割真值。两轮混合微调、首次独立测试及最新固定验证复核均已完成；2026-10-24用户确认本阶段表现符合预期，暂缓继续补采／训练，准备在抓取场景检验实际效果。

**静态数据工具准备**：`capture_placement_sequence.py --strips N`、逐根多边形复核、可见区域生成和数据导出均按实际数量处理，无三根上限。4／6／8根已通过合成图像的数据链路检查，含草稿拒绝导出、遮挡分段同ID、16位实例PNG及训练标签读取；[检查证据](EVIDENCE.md#多根静态分割数据链路检查)不代表真实模型精度或浏览器交互验收。复核总览仅用三种颜色循环显示，颜色重复不表示ID合并，身份以`strip_XXX`及逐根页面为准。动作示教的目标数／尝试序号已另行适配，当前规则见[人工采集章节](#四根与六根动作数据扩充)，不重开065～069静态补拍。

**首批四组开发清单（人工复核、导入及诊断完成）**：`pile_052～055`已由用户采集并完成原图初查，其中054为误操作后重拍，共24张原图、20个人工确认的新增条轮廓，均已更新到[统一复核入口](../../artifacts/segmentation/placement_sequences/review/index.html)。用户分别确认了四组的“先拿上层、其余静止”现场条件，以及下载JSON中的20个轮廓；四份人工JSON已原样导入，20个非空阶段的可见掩码和遮挡关系检查通过。[结果记录](EVIDENCE.md#多根静态分割数据链路检查)保留采集、草稿与人工确认状态区别。原拟八根方案未执行，不计入本批；八根继续暂缓，`pile_056`已改用于下方的四根补采组。本批按用户建议先摆整堆、逐根移除拍摄，操作及候选区域见[采集规则](#采集规则与编号)。各组预先设`--split train`，用于发现问题及后续候选补训，不能事后改标为独立测试。

| 组号 | 根数N | 已采结构 | 图数（含空图） |
| --- | ---: | --- | ---: |
| pile_052 | 4 | 一对交叉＋两根独立，先观察数量增加时的分离 | 5 |
| pile_053 | 4 | 两对交叉，改变交叉位置与方向 | 5 |
| pile_054 | 6 | 两对交叉＋两根独立，增加同帧实例数 | 7 |
| pile_055 | 6 | 三对交叉，增加遮挡发生的位置 | 7 |

该批以一至两层为主，把总根数增加与新增层高分开；各交叉对及独立条之间留出可辨间隔。当前按用户确认的六根工作区上限安排，不要求为八根加宽台面。相对旧示教的新位置错误单独记录，不能全归因为根数增加。

本批沿用[逐根采集与人工复核](#静态分割采集与人工复核)完成确认和导入，054端部按完整原图保留；被遮成多段的同一根保留同ID。后续新组若旧条被碰动或倾斜，不能继续使用原轮廓传播生成真值。记录实际审核用时，不用20个轮廓数推定全部标注工时；本批未获得审核用时记录。

**冻结模型开发诊断已完成（2026-10-23）**。用户手动执行了[预测脚本](../../artifacts/segmentation/multi_strip_development/run_frozen_prediction.sh)，第15轮冻结模型的24图预测已回传，优化步数为0。结果见[逐图对照页](../../artifacts/segmentation/multi_strip_development/prediction_review.html)、[分析JSON](../../artifacts/segmentation/multi_strip_development/analysis.json)和[证据记录](EVIDENCE.md#多根静态分割数据链路检查)。模型可以输出六个实例，没有三根硬上限，但新摆放中存在实例合并、分段身份错配和上层误判被压；部分两根／三根阶段也有错误，不能只归因于数量增加。四组是开发数据，阶段图相互相关，不能据此报告独立泛化能力或抓取成功率。原始预测、图像和人工标注保留，无需重跑预测脚本。

**当前小批：按失败补数据（056～060人工复核及导入完成）**。本批为三组训练摆放、两组独立验证摆放；验证集用于后续选择训练轮次和配置，最终测试仍须另采未参与开发的摆放。每组清空后重新摆放，改变位置、方向和交叉点，不能把同堆不同阶段拆到不同集合。

| 组号 | 集合 | 根数 | 结构及重点操作 | 状态 |
| --- | --- | ---: | --- | --- |
| pile_056 | train | 4 | 四根互不遮挡、近似平行但留间隙，变化前后位置；逐根移除，覆盖多个同时为上层的目标 | 现场条件、人工轮廓已确认；导入及掩码检查通过 |
| pile_057 | train | 6 | 两对交叉＋两根独立，改变054的位置与交叉点；先移除两根交叉上层，使剩余四根均不被压 | 现场条件、人工轮廓已确认；导入及掩码检查通过 |
| pile_058 | train | 6 | 三对交叉，改变055的交叉点与方向；先移除三根上层，再逐根移除下层 | 现场条件、人工轮廓已确认；导入及掩码检查通过 |
| pile_059 | validation | 4 | 两对交叉，独立重摆位置与方向；每次只移除当前上层 | 完整重拍组已确认；人工轮廓导入及掩码检查通过 |
| pile_060 | validation | 6 | 两对交叉＋两根独立，独立重摆；先移除两根交叉上层，再逐根移除其余 | 现场条件、人工轮廓已确认；导入及掩码检查通过 |

已采共31张图（含5张空图），26个新增条轮廓均已由用户确认，五份下载JSON已原样导入[统一复核入口](../../artifacts/segmentation/placement_sequences/review/index.html)，无需重复复核或采集。进度与导入检查见[机器清单](../../artifacts/segmentation/multi_strip_development/next_capture_plan.json)，具体结果见[证据记录](EVIDENCE.md#多根静态分割数据链路检查)。

**新旧数据混合准备（已用于首轮微调）**：[新增选择清单](../../configs/aubo_i10/short_strip_segmentation_up_to_six.json)保留007～046全部原划分，加入052～058为train、059～060为validation。独立本地导出在`datasets/segmentation/short_strip_segmentation_up_to_six/`，不覆盖旧导出；不包含047～051测试组。两组数据组成如下，图数包括空图，实例数包含同条跨阶段重复出现：

| 集合 | 旧数据：堆／图 | 新增数据：堆／图 | 合计：堆／图／可见实例 |
| --- | ---: | ---: | ---: |
| train | 30／120 | 7／43（052～058） | 37／163／294 |
| validation | 10／40 | 2／12（059～060） | 12／52／91 |

六根整堆训练摆放只有054、055、057、058四堆，验证只有060一堆；对应六根阶段仅4张训练图、1张验证图，阶段图相关，不能以总图数推定覆盖充分。当前适合作为首轮小规模微调试验，仍须根据错误补充独立摆放，并另采未参与开发的最终测试组。旧047～051可用于历史场景回归比较，不能代替新增布局的最终测试。

**首轮微调配置（用户已手动执行并完成）**：从保留的第15轮旧best权重初始化，新建优化器；用完整640×480无描边RGB、可见实例与原两类标签。沿用均匀按图打乱采样，不额外重复抽样，新图占训练图43／163；混入旧数据用于观察并减轻旧场景退化。脚本设置15轮、batch=2、每轮验证、seed=42，非骨干学习率`1e-5`、骨干`1e-6`，均为首轮的五分之一，实际1230次参数更新；这只是首轮试验，不保证覆盖充分或不发生退化。训练入口在优化前执行第0轮验证，按混合验证集两类平均F1（同时考虑漏检与误检的指标）选best，结束后重载验证。本轮采用所选best而非最后一轮；旧验证40图、新验证12图及各当前根数的结果已分别统计，掩码分析见[验证对照页](../../artifacts/segmentation/multi_strip_development/finetuning_result/prediction_review.html)，数值与剩余问题见[证据记录](EVIDENCE.md#多根静态分割数据链路检查)。最终测试不用于选择轮次或阈值。

**远端复用约束（2026-10-23用户明确）**：项目、环境和数据目录全部复用，不重复上传已有数据；用户进一步确认仅允许在原项目`outputs/`内新建本次训练结果子目录。继续使用[原分割实验根目录、模型和Python](EVIDENCE.md#数据模型与环境位置)。已只读核实旧数据目录`datasets/short_strip_segmentation/`内的160张RGB及160张实例图，以及`multi_strip_development_052_055/input/images/`的24张RGB；旧数据清单与本机一致。

增量仅为056～060的31张RGB和052～060的55张实例图，共约19.5 MB，另加清单与小范围源码修改。052～055原图通过远端硬链接复用：在既有`images/`中增加文件名引用，指向原图，不重新传输图像内容；原图继续只读。旧`dataset.json`保留，混合清单保存为同目录`dataset_up_to_six.json`。新结果限定为`outputs/short_strip_mask2former_up_to_six/`及其内部模型／验证产物，日志为既有`logs/finetune_up_to_six.log`；旧best、final、预测和训练记录保留。

远端仍采用早期模块布局，准备脚本只给原位置的三个训练相关文件加入`--manifest-name`支持，不迁移项目结构。应用前核对远端源码仍与已检查内容一致，保存每个文件的`.before_named_manifest`备份；若出现其他改动就停止，不覆盖。回退时可用这三份备份恢复对应源码，原默认清单一直保留；新增图像／掩码不影响旧清单，无需删除数据。

以下两条为本轮已执行的命令，保留用于追溯，当前不要重复运行。用户先运行**仅增量准备**命令；它不创建远端目录、不加载模型、不训练，准备结果已回传为`artifacts/segmentation/multi_strip_development/remote_preparation_result.json`：

```bash
bash artifacts/segmentation/multi_strip_development/prepare_remote_finetuning.sh
```

准备成功后，用户已**单独启动真实微调**并完成：

```bash
bash artifacts/segmentation/multi_strip_development/run_mixed_finetuning.sh
```

训练脚本检查准备结果，只向获准的新结果子目录写入；JSON、日志及52张验证掩码已回传到固定`finetuning_result/`，权重保留在GPU。结果已存在时停止，避免重复训练或覆盖。助手随后只读核实完成记录，并基于保存的预测做离线分析；未重跑052～055预测、追加训练或切换真机模型。

**061～064首批独立测试比较已完成；2026-10-24获准用于下一轮训练。** 用户已手动执行两模型预测，助手完成保存结果的离线分析及24图对照检查，见[三栏对照页](../../artifacts/segmentation/multi_strip_development/independent_test/prediction_review.html)。旧baseline第15轮best与本轮第5轮best、置信度／掩码阈值均在测试前确定，未用本批选轮次或调阈值。以下为首次测试时的采集记录：四组全部为`test`，每组清空后独立重摆，改变位置、方向、交叉点；同堆所有阶段归同一集合。原始记录不改写，新训练用途见下方授权与清单。

| 组号 | 根数 | 测试结构与移除顺序 | 图数（含空图） | 当前状态 |
| --- | ---: | --- | ---: | --- |
| pile_061 | 4 | 四根独立、不接触、不交叉，留清晰间隙；逐根移除 | 5 | 本组现场条件已确认；4/4人工轮廓已原样导入 |
| pile_062 | 4 | 两对交叉，各对之间留间隙；先逐根拿走两根上层，再拿两根下层 | 5 | 本组现场条件已确认；4/4人工轮廓已原样导入 |
| pile_063 | 6 | 两对交叉＋两根独立；先拿两根交叉上层，再逐根拿走其余四根 | 7 | 本组现场条件已确认；6/6人工轮廓已原样导入 |
| pile_064 | 6 | 三对交叉，各对之间留间隙；先逐根拿走三根上层，再拿三根下层 | 7 | 本组现场条件已确认；6/6人工轮廓已原样导入 |

合计24张图、4个空场景、20个新增轮廓；只有4个独立摆放，其中六根两堆，不能据此估计稳定泛化准确率。新模型改善已延续到这批未参与训练和选轮次的新摆放，但两张六根图仍有类别或身份错误，五根图还出现过整段漏掉却通过IoU门槛；目前继续静态数据改进。旧047～051保留为历史回归。061～064首次独立成绩及原始`test`记录原样保留；用户随后明确允许这四堆在新训练清单中作为`train`，此后它们的预测只能检查训练拟合和已知错误，不能评价泛化能力。开发期间使用固定验证集；候选模型稳定后再留一批新摆放做最终独立验收，不每轮重采测试。

四份人工JSON保留在`pile-json--/segmentation/`，已原样导入[统一复核入口](../../artifacts/segmentation/placement_sequences/review/index.html)。正式测试导出位于`datasets/segmentation/short_strip_segmentation_up_to_six_test/`，选择清单与导入证据在`artifacts/segmentation/multi_strip_development/independent_test/`；通过内置审计，与混合训练／验证组无交叉。064首张亮度变化已记录，使用原RGB，不修改人工答案。

**远端继续复用原项目、环境和目录**：只上传本批24张新RGB及不含人工掩码的预测清单至既有`datasets/short_strip_segmentation/`，不重传旧数据，不改变旧训练／验证清单。两模型结果写到已有`multi_strip_development_052_055/predictions/`，分别使用`test_061_064_old_`与`test_061_064_new_`文件名前缀；不创建远端目录，052～055原预测保留。目录名沿用历史名称，本批数据性质由独立清单及报告的`independent_test=true`标明。

以下为用户已执行完成的命令，保留追溯，**不要重复运行**。每模型24图、优化步数0；输入为完整640×480原始RGB，置信度／掩码／重叠阈值固定为0.5／0.5／0.8：

```bash
cd /home/rentao/program/lerobot-aubo
bash artifacts/segmentation/multi_strip_development/run_independent_test.sh
```

原始预测已回传到`multi_strip_development/independent_test/result/`；[离线分析](../../artifacts/segmentation/multi_strip_development/independent_test/analysis.json)与对照页保留逐图IoU、类别、可见分段覆盖及图检备注。统计与具体失败只在[证据记录](EVIDENCE.md#多根静态分割数据链路检查)维护，机器进度仍在`multi_strip_development/request.json.independent_test`。

**当前：现有数据的第二轮混合微调及固定验证分析已完成（2026-10-24）**。用户明确要求先解决视觉问题，再推进P1／P2，并确认“将061～064用于下一轮训练，保留历史记录”。新增四／六根数据改为11堆训练（052～058、061～064）和2堆验证（059～060），分别67／12张、176／31个跨帧可见实例。007～046原训练／验证划分保留，047～051不加入训练；合计训练41堆／187张／356实例，验证12堆／52张／91实例。原验证中的10堆三根场景用于检查旧能力是否退化，新增验证中的六根仍只有060一堆，不能据此宣称广泛稳定。

新清单为`datasets/segmentation/short_strip_segmentation_up_to_six/dataset_failure_finetuning.json`，复用原数据目录，仅补入061～064已复核导出的24张RGB及24张实例图。原`dataset.json`、首轮配置、独立测试导出、原图／人工JSON及首次预测均保留；新清单明确记录原始`test`来源、用户授权和本轮有效`train`用途，不修改源序列或放宽普通导出器的划分检查。一次内置数据审计已通过，准备阶段记录见`audit_failure_finetuning.json`及`multi_strip_development/failure_finetuning_preparation.json`；随后操作者已完成增量准备和训练，最新结果见下方固定入口。

本轮从首轮第5轮best继续微调，重建优化器；沿用15轮、batch=2、非骨干／骨干学习率`1e-5`／`1e-6`、seed=42、完整640×480输入，每轮验证，实际1410次参数更新。保留初始化模型的第0轮验证作为比较，按固定验证集选出第14轮best；不增加额外采样、增强或后处理调参。原项目、Python和数据目录继续复用；24张RGB已在首次测试时上传，本轮仅增传24张实例标签和新清单。新权重只写入原项目`outputs/short_strip_mask2former_failure_finetuning/`，旧两轮模型和结果保留。以下两条命令已由操作者执行完成，仅供追溯，**不要重复运行**。

操作者先运行增量准备（只传标签／清单并检查数据，不加载模型或训练；缺少原目录／RGB或遇到已有目标文件则停止）：

```bash
cd /home/rentao/program/lerobot-aubo
bash artifacts/segmentation/multi_strip_development/run_failure_finetuning.sh prepare
```

终端明确提示“增量准备完成”后，操作者单独启动训练：

```bash
bash artifacts/segmentation/multi_strip_development/run_failure_finetuning.sh train
```

数据准备结果已回传到`failure_finetuning_remote_preparation.json`；训练报告与52张验证预测保存在固定`failure_finetuning_result/`，权重留在GPU。助手仅用保存的预测完成[固定验证三栏对照](../../artifacts/segmentation/multi_strip_development/failure_finetuning_result/prediction_review.html)，具体指标、短段漏检及旧场景实例拆分见[证据记录](EVIDENCE.md#多根静态分割数据链路检查)。用户已确认静态视觉表现达到本阶段预期，下一步按[抓取验证准备](#当前位置与下一步)检验实际效果；残余问题保留，不追加训练或补采作为本阶段结束的前置条件。脚本拒绝覆盖已有结果；需续作时先检查保存状态，不直接重跑或删除目录。

后续若恢复视觉开发，流程为：现有数据混合微调→固定验证集选轮次并检查具体错误→仅在仍缺某类结构时补对应训练摆放。若需宣称独立泛化成绩，另安排未参与开发的摆放；当前阶段认可仅针对已保存的固定验证结果，不据此要求立即补拍。061～064若再预测须标为已见训练样本，不用其分数冒充独立改进。

**候选补采清单（暂缓，尚未采集）**：下面065～069保留为按失败补数据的候选，当前不要求采集。五组原始目录、review目录及人工JSON均不存在；此前已通过不带`--record`的计划模式，无参数变化无需重复检查。原拟33张图、5张空图、28个新增轮廓；是否仍需这些组及其用途，依现有数据微调后的具体缺口确定，再在采集前冻结划分。

| 组号 | 集合／根数 | 摆放重点与移除顺序 |
| --- | --- | --- |
| pile_065 | train／4 | 两对较小夹角的X形交叉，各对分开；一对交点偏离中点，形成长短不同的可见段。先逐根拿两根上层，再拿两根下层 |
| pile_066 | train／6 | 三对交叉，改变位置与方向，包含小夹角及偏心交点；先逐根拿三根上层，再拿三根下层 |
| pile_067 | train／6 | 两对交叉＋两根相邻、近似平行但留清晰间隙的独立条；至少一根被压条露出长短不同的两段。先拿两根交叉上层，再逐根拿其余四根 |
| pile_068 | validation／6 | 三对交叉，独立重新选择位置、方向和交点，不照搬065、066；先拿三根上层，再拿三根下层 |
| pile_069 | validation／6 | 两对交叉＋两根相邻独立条，独立重新摆放，不照搬067；先拿两根交叉上层，再逐根拿其余四根 |

**恢复补采时的065模板（当前不执行）**：四根分为两对X形交叉，夹角大约20～30°即可，不必测量；两对之间留间隙，不增加额外交叉。一对交点略偏向端部，使下层两段长短不同，但短段仍清楚可见。保持原机位和支撑面，四根及两端完整入镜，重新摆放位置和方向。采用四根是为了先补清晰的交叉身份样本，随后再放入六根背景中。

```bash
cd /home/rentao/program/lerobot-aubo
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/capture_placement_sequence.py \
  --camera-set original-global \
  --output artifacts/segmentation/placement_sequences/pile_065 \
  --placement-id pile_065 --split train --strips 4 --capture-order remove \
  --roi 280 160 640 480 --min-chroma-change 8 --record
```

上述命令仅在决定恢复补采后由操作者执行：只打开全局相机，先拍整堆；逐根移除两根上层，再逐根移除两根下层，完成4→0共5张，其余不动，手退出后再拍。采后检查与本组现场确认完成，再生成草稿并推进下一组。当前静态视觉阶段已获用户认可，下一步是抓取验证准备，此补采模板不执行。

2026-10-21讨论的扩展约定按当前六根上限调整如下；台面加宽仅保留为未来方向：

- **台面**：相机、机器人起点、左侧放置区不动；当前使用能容纳六根的现有工作区，下指空间由现场确认。未来若在右侧支撑台旁拼接垫块加宽，须与原支撑台等高、外观一致，记录变化后再补覆盖；台面高度变化会整体偏移层高基准。
- **训练覆盖**：69条示教首帧目标全部位于原支撑台（全局图约x 362～589、y 235～429，主轴约60°～146°），见[覆盖统计](EVIDENCE.md#三根堆叠正式采集)。加宽区和超出该范围的位置对现有模型属于未见位置，模型输出绝对关节角，越远越依赖外推，需新示教补充。原台面内的结构可先用现有模型零样本试几堆，按感知错／选错／抓歪／碰动邻条分类失败；新布局摆好后先只拍全局图，与覆盖图叠加核对。
- **布局**：纯逐根叠放每步k=1，且下探高度与抓取步数完全耦合，模型可能只学“剩几根下探多少”。在现有工作区内安排交叉对、独立条或矮小堆，使多数步骤k≥2，并让相同剩余根数对应不同高度。摆放清单须平衡各层示教数量；第3层及以上在单堆中占比低，需专门安排。超出现有示教高度的层（如逐根叠6层，顶层比现有示教高约30 mm以上）现有模型不能零样本覆盖。
- **示教**：仍为每抓一根存一条，一堆最多产出6条，并自然覆盖剩余6→1根各状态；同堆示教彼此相关，独立样本按堆计，划分仍按整堆。示教偶尔带动下层条，这类示教保留不删；建议夹紧后先竖直抬起数厘米再平移，减少拖动。邻条扰动不增加现场询问，改为离线比较相邻两步首帧种子，估计剩余条位移。
- **标注**：沿用助手生成草稿、用户审核确认，开始记录审核用时。首帧种子传播（如SAM 2）能否提高草稿质量，在阶段切换时评估，不在P1中途替换。
- **连续作业**：在各步成功概率相同且相互独立的简化假设下，整堆成功率为单根成功率的N次方（单根0.9时，5根约0.59、6根约0.53），实际结果仍按整堆测量。本阶段需要自动成败判断与重抓（第3阶段）。邻条被带动超出8像素邻域时身份参考会丢失，跟踪升级在此阶段才成为必要。
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
| 感知识别 | 静态复核已达到本阶段预期，下一步准备第14轮分割候选的抓取验证；047～051和061～064首次测试原结果保留，剩余短段／身份问题按实际抓取影响跟踪，详见[扩展记录](#扩展到6根及以下)。其他升级候选仍暂缓（2026-10-21）：SAM 2.1首帧种子视频传播（同时替代当前帧分割、身份分配和跟踪）、SEA-RAFT稠密光流接原RANSAC、CoTracker3遮挡感知点跟踪；若恢复比较，先在已人工确认的逐帧身份掩码上检验逐帧IoU、交叉处身份互换、跟踪丢失和延迟，再决定是否替换。替换理由须来自数据，不以“方法较旧”为据；SAM 2对外观相近的细长物体易混淆身份，需重点检验。 |
| 目标条件化 | 两组训练和离线对照已完成，当前重复target-stack真机试验；尚不能宣称目标组优于baseline。A／B实验结束，旧跟踪诊断冻结。 |
| 可下指区段 | 当前实验由操作者在摆放时保证下指空间，不开发自动区段判断，不把尺寸测量或该模块作为推进门槛；实验结论限定于这一场景条件。 |
| 物理结果反馈 | 磁性开关及控制柜DI（数字输入）接入待现场确认，作为后续自动成败判断的准备，是贡献4及最终演示的前提；当前重复试验先人工判定，不等待自动成败系统。开关未闭合到底不独立证明夹住一根，需结合图像。 |
| 连续逐层抓放 | 指定A→B→C及两次归位已实现，trial04成功；新增[自主选目标与逐根清空](#自主选目标与逐根清空)为后续目标，具体范围、阶段和指标只维护在该节。当前每根最多3次空前景重观察，不是3次物理抓空重试；自动选择、结果判断、清空检测及物理重抓尚未实现。 |
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

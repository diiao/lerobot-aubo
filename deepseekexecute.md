# DeepSeek执行任务：接入已复核的新视觉模型

编写日期：2026-10-24。用户明确要求创建本文件，作为交给DeepSeek的一次性执行说明。长期操作与实验结论仍只维护在README和EVIDENCE，不另外建立同内容的计划或报告文档。

执行状态：本地接入已交付，用户随后完成三次真实GPU动作返回；一次性检查脚本因助手修改引入的列表类型错误在本地保存失败。用户已要求检查后删除该脚本与本次输出，本文件保留为历史任务说明，不据此重建或复跑检查。当前状态见[README](examples/phone_to_auboi10/README.md#当前位置与下一步)，原因见[EVIDENCE](examples/phone_to_auboi10/EVIDENCE.md#多根静态分割数据链路检查)。

## 1. 任务目标与交付边界

请在`/home/rentao/program/lerobot-aubo`中完成**新分割模型接入`target-stack`的本地软件准备**，使后续三根抓取实验能够使用第二轮混合微调的第14轮best。完成最小代码修改、相关离线检查，并准备操作者手动进行真实GPU加载／保存图像推理验证的命令。

本次更换的是视觉分割模型Mask2Former：输入全局RGB，输出每根竹条的可见区域及上层／被压类别。动作模型SmolVLA继续使用原`target-stack`最终权重；它根据目标描边全局图、原始腕部图、七维状态和固定任务文字生成动作。

本次不执行P1五堆抓取实验。相机采集、机械臂／夹爪／归位和真实GPU加载／推理分别由用户手动执行；准备工作不要因尚未执行这些步骤而停在口头计划上。直接完成已明确的本地修改与无设备验证，再给出具体手动命令和待验证项目。

约束：

- 保留原动作权重、任务文字、目标身份关联、运动跟随、时效和运动限制，不加入高度／姿态／轨迹补偿。
- 不补采065～069，不继续训练，不开展无描边基线对照，不更换整套感知架构。
- 保留旧模型、原始数据、人工JSON、历史预测和运行目录。复用现有GPU项目、模型目录与Python环境，不为接入重传权重、复制整套项目或安装依赖。
- 修改前说明文件作用、原因与影响范围。优先最小实现，不增加通用配置框架、重复哈希校验或无关测试。
- 本次Git交付范围只到报告修改差异；不自动提交或推送新改动。此前`7f1af80b`的提交／推送已结束。

## 2. 先读当前状态

交接时分支为`research/depth-vla-phase-a`，HEAD为`7f1af80b`（`feat(aubo): complete six-strip static segmentation review`），该提交已推送远端。本文件创建前Git工作区干净；执行时重新检查，保留届时已有改动。

```bash
cd /home/rentao/program/lerobot-aubo
git status --short --branch
git log -1 --oneline
```

按以下顺序阅读：

1. [AGENT.md](AGENT.md)：维护约定与现场操作边界。
2. [README：当前位置与下一步](examples/phone_to_auboi10/README.md#当前位置与下一步)、[现场流程](examples/phone_to_auboi10/README.md#目标条件模型单次真机试验)、[目标条件接口](examples/phone_to_auboi10/README.md#目标条件接口约定)。
3. [EVIDENCE：静态分割结果与阶段认可](examples/phone_to_auboi10/EVIDENCE.md#多根静态分割数据链路检查)。
4. 下列真实代码及回传记录；以当前文件为准，不从旧交接推测运行行为。

用户已认可“静态视觉识别表现达到本阶段预期，抓取场景中的实际效果仍需实验验证”。固定验证非空图整图达标率为97.5%（IoU≥0.5）、92.5%（IoU≥0.75）；这不是抓取成功率。已知011两根图局部拆分、060六根图短段漏检继续保留，不以修复全部问题作为本任务前置条件。

061～064已获用户授权加入本轮训练，首次独立测试记录保留；以后不能将它们用作独立泛化成绩。

## 3. 模型身份与证据

以下绝对路径均指GPU主机（现有SSH别名`gpu`）上的文件，不是本机权重目录。

| 用途 | 路径／身份 |
| --- | --- |
| 本次新视觉候选 | `/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/outputs/short_strip_mask2former_failure_finetuning/best` |
| 旧视觉模型，保留回退 | `/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/outputs/short_strip_mask2former_baseline/best` |
| 保持不变的动作模型 | `/home/rentao/program/lerobot-aubo-joint-stack/outputs/joint_stack_target_smolvla/final` |
| 现有堆叠推理服务 | `/home/rentao/program/lerobot-aubo-joint-stack/artifacts/joint_stack_live_service/joint_inference_stdio.py` |
| 现有远端Python | `/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv/bin/python` |

新视觉模型来自15轮、1410步训练，`best_epoch=14`，结束后已重载验证。不要误选首轮混合微调第5轮best、第二轮final或终端最后一行的第15轮。

本地证据位于既有目录[训练回传结果](artifacts/segmentation/multi_strip_development/failure_finetuning_result/)：

- [complete.json](artifacts/segmentation/multi_strip_development/failure_finetuning_result/complete.json)：最佳轮次、训练完成和重载状态。
- [plan.json](artifacts/segmentation/multi_strip_development/failure_finetuning_result/plan.json)：真实输出目录、类别和训练参数。
- [loading_info.json](artifacts/segmentation/multi_strip_development/failure_finetuning_result/loading_info.json)：缺失、多余、不匹配参数与加载错误均为空。
- [analysis.json](artifacts/segmentation/multi_strip_development/failure_finetuning_result/analysis.json)及[逐图对照](artifacts/segmentation/multi_strip_development/failure_finetuning_result/prediction_review.html)：静态表现与剩余错误。

这些本地报告不是权重副本。真实加载验证前，再核对远端候选目录中的权重、模型配置、处理器配置及`strip_contract.json`，并与同一训练结果目录的完成记录对应；无需再扫描或重算所有旧数据／模型哈希。远端检查命令与模型启动命令分开交付用户。

## 4. 已核实的调用链与最小修改位置

```text
run_joint_smooth.py --model target-stack
  → resolve_targets()
  → prepare_live_setup(..., model='target-stack')
  → options['segmentation_checkpoint']
  → InferencePipe
  → 远端参数 --target-segmentation-checkpoint
  → FrozenStripSegmenter
  → FreshTargetInputs：当前分割 + 已审核种子身份
  → 全局RGB目标描边 + 原始腕部RGB
  → 原target-stack动作模型
```

| 文件 | 已有作用／本任务关注点 |
| --- | --- |
| [model_profiles.py](src/lerobot/bamboo_sorting/model_profiles.py) | 纯常量，无模型／设备加载。现有`SEGMENTATION_CHECKPOINT`指向旧baseline；`TARGET_STACK_CHECKPOINT`和`TARGET_STACK_MODEL_SHA`属于动作模型，不能替换成视觉模型身份。 |
| [live_setup.py](src/lerobot/bamboo_sorting/runtime/live_setup.py) | `prepare_live_setup()`先构造旧成对模型选项，再在`model == 'target-stack'`分支替换动作模型和服务配置；目前该分支尚未替换分割路径。 |
| [run_joint_smooth.py](examples/phone_to_auboi10/run_joint_smooth.py) | 为单目标或A→B→C逐次生成选项；`build_plan()`已输出`segmentation_checkpoint`。不带`--execute`时只打印计划。 |
| [inference.py](src/lerobot/bamboo_sorting/runtime/inference.py) | 已把分割路径传到远端，并检查远端`ready.segmentation_checkpoint`与请求一致；不要删除该一致性检查。 |
| [joint_inference_stdio.py](examples/phone_to_auboi10/joint_inference_stdio.py) | 已接受`--target-segmentation-checkpoint`，加载分割器并在ready消息中回报实际路径。远端部署脚本可能是历史副本，不能仅凭本地文件断言远端已更新。 |
| [joint_target_runtime.py](src/lerobot/bamboo_sorting/runtime/joint_target_runtime.py) | `FrozenStripSegmenter`加载真实GPU模型，检查类别映射；`FreshTargetInputs`负责当前帧前景、身份和描边。注意：构造分割器就会加载CUDA并预热，不属于纯计划检查。 |

推荐实现：

1. 在`model_profiles.py`新增描述性常量，例如`TARGET_STACK_SEGMENTATION_CHECKPOINT`，绑定本次新候选；保留旧`SEGMENTATION_CHECKPOINT`及旧权重。
2. 在`live_setup.py`的`target-stack`分支中设置对应的`segmentation_checkpoint`。让单目标和A→B→C每个会话都走同一选择逻辑。
3. `target-pair`及其他历史入口继续使用原有配置；不要全局搜索替换所有`short_strip_mask2former_baseline`引用，尤其不要改写历史记录或标注准备提示。
4. 现有推理协议已经支持这个参数，优先复用。没有具体缺口时，不新增命令行开关，不改加载器、分离器、跟踪器或控制器。

预期主要生产代码修改只有上述两个配置／选择文件。若当前代码已变化，依据实际调用链调整，并解释超出此范围的必要性。

## 5. 输入输出与行为保持

- 分割输入为完整640×480原始RGB，不裁剪、不缩放、不预先描边。类别保持`0: top_strip`、`1: covered_strip`；分数／掩码／重叠阈值沿用0.5／0.5／0.8。
- 实例图中的像素值是实例ID，类别从`segments[].class_id`读取。模型输出的实例ID不等于跨帧目标A／B／C，仍由已有种子与身份分离链处理。
- 接近阶段，使用本帧目标可见区域给全局图描紫色轮廓；腕部图保持原图。搬运阶段沿用现有原始双RGB输入，不新增分割调用。
- 状态／动作为七维，动作块为50×7；动作权重、动作模型身份、归一化、固定文字、三次独立会话及两次归位安排均沿用既有实现。
- 空目标停止与有界重观察、运动跟随及预览保持现状，不借本任务放宽阈值或填补模型未看到的像素。

## 6. 按顺序完成验证

### A. 本地代码与无设备检查：直接完成

先用已有测试检查真实的配置选择、计划和参数传递，优先补充现有用例，不为每个常量另建测试文件。

重点是：

1. `target-stack`使用新分割路径，`target-pair`仍用旧路径，动作模型与服务身份不变。
2. 单目标及A→B→C的每个会话均传入正确路径，计划与实际构造的推理选项一致。
3. 无`--execute`的计划模式不创建推理进程、不加载模型、不连接设备。
4. 既有远端ready路径不一致时仍报错；当前帧描边、搬运原图等行为不因路径选择改变。

特别注意已有`test_joint_smooth.py::test_multi_target_plan_uses_explicit_service_for_remaining_pile`：它精确断言`target-stack`与`target-pair`计划的差异字段。新行为应增加`segmentation_checkpoint`这一项，并分别验证两种路径，不能为了让测试通过而去掉差异断言或无关断言。

相关测试命令：

```bash
make test-aubo-offline AUBO_TEST_PATHS='tests/bamboo_sorting/runtime/test_joint_smooth.py tests/bamboo_sorting/runtime/test_joint_target_shadow.py tests/bamboo_sorting/runtime/test_joint_target_runtime.py'
```

这是CPU、假设备／合成输入的离线通道。按实际修改缩小或补充测试范围；完成后不重复跑全量测试、旧训练或旧静态预测。遇到缺失依赖如实记录跳过／受阻项目，不顺手更改环境。

修改后可用已保存的trial04快照核对参数计划。下面命令只读取旧种子、打印计划，不连接设备或GPU：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
  --model target-stack \
  --target-snapshot artifacts/live/stack/stack_target_20261018_trial04/scene_snapshot \
  --target-selections artifacts/live/stack/stack_target_20261018_trial04/scene_snapshot/selection.json \
  --target-sequence A B C --target-preview --track-target-motion \
  --transport-age-aligned --latency-tolerant --max-seconds 600
```

`PYTHONPATH=src`仅用于这次进程读取项目源码，不修改系统环境。计划中应显示新分割路径、原动作模型、`multi_foreground`、A→B→C及原运行选项。这个旧快照只用于离线检查，不能在该命令后加`--execute`用于新现场抓取；不要覆盖trial04的旧计划或输出。

### B. 真实GPU加载与保存图像链路验证：准备命令，由用户手动执行

本地测试通过只证明代码选择与接口逻辑。还需准备一次范围有限的真实验证，检查现有远端环境能否加载新权重，以及新掩码能否通过目标输入链。

- 优先复用现有远端服务和绝对权重路径，不移动权重到动作项目。只有确证远端脚本／模块缺少所需能力时，才列出必要同步文件、现有文件保护和回退办法，交用户执行；不要整体部署当前仓库覆盖冻结服务。
- 使用已有保存的双相机图像、状态和已审核种子即可，不需要重新拍摄。trial04的`scene_snapshot/`中已有`snapshot.json`、`selection.json`、`global_rgb.png`和`grasp_rgb.png`。
- 验证必须调用`prepare_live_setup(..., model='target-stack', multi_separation=True)`产生的选项及现有`InferencePipe`，确认真实使用的是新视觉模型与原`target-stack`动作模型。准备脚本不得构造或连接`ReadOnlyStation`、控制器或夹爪。
- 不要直接把`run_joint_target_shadow.py --stage predict`当作本任务验收：当前该入口绑定旧`target-pair`，直接使用人工轮廓；成功运行它不能证明新分割与`target-stack`链路已接通。
- 操作者执行后，保存实际模型路径、加载／ready记录、目标前景与描边图、返回动作块形状及耗时。查看目标身份和轮廓是否合理；动作只记录，绝不发送给机器人。
- 少量保存观察足以做接入检查，不重跑全部52张静态验证。若保存图上出现已知分割／关联错误，保留原始结果，区分“模型已加载／链路已接通”和“此图目标输入失败”，不靠放宽现有规则强行通过。

如需要一次性验证脚本或结果，放在现有`artifacts/segmentation/multi_strip_development/`下面，采用描述性名称、先检查是否已存在；不覆盖旧产物，不另外创建远端项目或Python环境。交付命令时明确哪条只是检查文件，哪条会启动真实GPU模型；代码和命令准备就绪后等待用户手动执行，不自行启动推理。

### C. 留给下一步的现场实验

本任务结束后，用户另行启动P1：新摆放、新快照、新审核种子及新输出目录，按README的五堆计划执行。固定新视觉模型、原动作模型和运行选项，记录抓对／单根／送达／整堆清空、邻条扰动及前景恢复。

这部分只交付操作入口，不在本任务中执行。已有静态F1、离线动作预测和指令周期均不能写成新模型的物理抓取成功率。

## 7. 回退与文档更新

回退仅将`target-stack`的分割选择恢复为保留的旧baseline路径，动作配置保持不变；不删除新权重、不覆盖旧记录、不使用`git reset --hard`清除其他改动。重新生成无执行计划确认路径；若已有实验数据，明确新旧模型使用边界，分开统计。

完成本地改动后，原位更新：

- [README](examples/phone_to_auboi10/README.md)：当前选择的新视觉路径、实际接入状态、手动GPU验证入口与回退方式。
- [EVIDENCE](examples/phone_to_auboi10/EVIDENCE.md)：本次实际执行的检查、结果路径、失败／未执行项目。GPU验证尚未执行时明确写“本地软件接入完成，真实加载／推理待操作者验证”，不能提前写“部署验证完成”。

不复制训练历史，不重做本阶段指标；本文件作为任务交接说明保留，不再扩展为第三份长期进度台账。

## 8. 向用户交付的内容

完成后用中文简洁报告：

1. 修改了哪些文件，为什么这样改，旧入口如何保留。
2. 新视觉路径、保持不变的动作模型，以及计划／推理选项核对结果。
3. 测试实际通过／跳过／失败数量；不要把只收集测试或模拟验证说成真实GPU通过。
4. 可直接运行的操作者手动验证命令、输出位置、预期结果；如已收到执行反馈，报告真实结果。
5. 仍未验证的内容、简单回退方式、进入P1前剩余的具体事项。

任务完成的最低交付是：本地接入代码与相关验证完成、原入口保留、手动真实验证命令已具体准备好、README／EVIDENCE状态一致。真实GPU验证和现场抓取是否完成，分别按实际证据标注，不能相互替代。

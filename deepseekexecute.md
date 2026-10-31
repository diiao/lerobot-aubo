# DeepSeek执行任务：跨轮复核1px／2px的J6动作差异

更新：2026-10-27。工作目录：`/home/rentao/program/lerobot-aubo`。

本文件替换已完成的三模型对照任务。上一批27份动作块保留在 [fixed_observation_comparison](artifacts/live/stack/stack_thin_outline_20261026_trial03/review/fixed_observation_comparison/)，不重跑baseline或掩码修边。本批只补两个保存输入，用一次小对照回答J6差异能否跨轮出现。

## 1. 目标、范围与执行方式

**问题：在另外两轮的接近末段观察中，1px模型是否仍比2px模型给出更高的J6目标，而且这种差异是否出现在现场实际使用过的动作行范围内？** J6是第六关节角，数值更高不等于夹口对准更好。

用户把本文件交给DeepSeek并要求执行后，连续完成输入准备、已有结果复用、12次必要的新样本推理、分析和EVIDENCE小范围更新，不逐帧／逐seed重复确认。当前编写会话只更新本文，未启动推理。

范围固定为两种现有模型、三轮各一帧、seed42／43／44。seed是推理随机种子，不是独立场景。三轮都使用过1px策略，本批是在这些历史观察上离线比较两种模型，不模拟2px闭环运行。

不训练、不部署、不连接相机／机器人／IO，不改运行模型、补偿、新鲜度或系统参数。沿用现有环境与协议；一个小脚本、必要图片和JSON即可，不增加通用框架、全量测试或文献搜索。

## 2. 先读现有成果，避免重复劳动

读取 [AGENT.md](AGENT.md)，执行一次 `git status --short --branch`，保留其他任务改动，不提交／推送。按需读以下材料的相关部分，不通读全部历史：

- [README当前运行说明](examples/phone_to_auboi10/README.md#目标条件模型单次真机试验)和[EVIDENCE的1像素章节](examples/phone_to_auboi10/EVIDENCE.md#1像素同预算正式训练结果)。
- 上一批[输入](artifacts/live/stack/stack_thin_outline_20261026_trial03/review/fixed_observation_comparison/inputs.json)、[预测](artifacts/live/stack/stack_thin_outline_20261026_trial03/review/fixed_observation_comparison/predictions.jsonl)、[分析](artifacts/live/stack/stack_thin_outline_20261026_trial03/review/fixed_observation_comparison/analysis.json)及[脚本](artifacts/live/stack/stack_thin_outline_20261026_trial03/review/fixed_observation_comparison/run_fixed_observation_comparison.py)。
- [trial01复核](artifacts/live/stack/stack_thin_outline_20261026_trial01/review/extracted_evidence.json)、[trial02复核](artifacts/live/stack/stack_thin_outline_20261026_trial02/review/run_summary.json)以及各自原始trace／commands。
- [模型配置常量](src/lerobot/bamboo_sorting/model_profiles.py)、[InferencePipe](src/lerobot/bamboo_sorting/runtime/inference.py)和[目标图像renderer](src/lerobot/bamboo_sorting/policy/joint_target.py)。

上一批只读复核已确认：trial03 #45实际使用第10～18行；按这些不同动作行等权、再跨三个seed取均值，1px／2px的J6相对观察状态分别约+0.521／−0.134°，两模型差约+0.655°。这比仅看第49行更接近运行所用区间，但仍是新离线预测的比较。

## 3. 固定本批三份观察

编写时已按保存文件和原始日志确定以下输入。trial01／02选择规则是：**同请求双JPEG齐全、接受的A／approach响应带掩码、闭合前确有普通接近取行的最后一份保存观察**。规则不看新模型输出；并非因为请求编号相近就认为场景相同。

| case_id | 来源目录（均在 `artifacts/live/stack/` 下） | 请求 | 历史普通接近选行（从0计数） | 本批处理 |
| --- | --- | ---: | --- | --- |
| `trial01_A_request_0045` | `stack_thin_outline_20261026_trial01/A_pick_run/` | 45 | 10～18 | 两模型各3个seed，新增6份 |
| `trial02_A_request_0040` | `stack_thin_outline_20261026_trial02/A_pick_run/` | 40 | 9～17 | 两模型各3个seed，新增6份 |
| `trial03_A_request_0045` | `stack_thin_outline_20261026_trial03/three_pick_run/` | 45 | 10～18 | 从上一批原值复用两模型共6份 |

trial01 #45的最旧输入时戳距接受闭合约2.313秒；trial02 #40约3.353秒，两帧不是严格相同的闭爪前时刻。trial02更晚的#45虽有保存图，但没有符合上述条件的普通接近取行，不用它代替#40。trial03没有接受闭爪，不编造其闭爪前秒数。

输入提取与检查一次完成：

1. trial01读取 `0045_global_rgb.jpg`／`0045_grasp_rgb.jpg`，trial02读取 `0040_global_rgb.jpg`／`0040_grasp_rgb.jpg`。从各自 `trace.jsonl` 对应接受请求读取 `observed_state`、`sensor_times`、`prediction.target_input`和`prediction.target_evidence.mask_png`，不从其他帧拼装状态／掩码。
2. 同一JPEG只解码一次，冻结为PNG供两模型共用。目视检查原双图和在线A轮廓，确认目标身份、掩码可用和接近阶段；不重新分割、不修边、不补遮挡区。若某帧确有错配／缺项，只在推理前按同一规则选择本轮邻近更早的保存帧，记录理由，每轮仍只取一帧。
3. trial03直接引用上一批冻结输入、在线掩码和6份预测，按case／模型／seed及来源核对一次即可。不要重复校验全部27份，也不为验证复用启动worker。
4. 从各轮 `commands.jsonl` 按 `prediction_index`关联对应请求，提取 `action_selection.selected_index`。trial01／02仅取接受闭合之前的记录，排除 `suction_target_frozen=true` 或 `closure_boundary=true` 的记录；trial03同样排除冻结／闭合边界记录。保存关联命令与动作行来源。
5. 主分析使用实际选行的**去重集合，每行等权**，不以最小／最大行补齐未用过的行，也不把某行重复下发次数当独立样本。以上集合由原1px运行产生，仅作为两模型共同的离线比较窗口，不声称2px实际运行也会选择这些行。

选帧及输入在新推理前写入 `inputs.json`。查看新输出后不换帧、不增加样本追求同向结论。保存JPEG无法恢复现场原始通信RGB，不把重算预测与旧现场响应当作逐值等价。

## 4. 模型配置与调用预算

| 模型记录名 | 配置来源 | 输入 |
| --- | --- | --- |
| `target_1px` | `TARGET_STACK_THIN_*` 的final／service／pythonpath | 同一在线A掩码，1像素紫色内描边，原腕图 |
| `target_2px` | `TARGET_STACK_*` 中原2像素final／service／pythonpath | 同一在线A掩码，2像素紫色内描边，原腕图 |

复用 `outline_target_images` 和 `InferencePipe`：`target_conditioned=True`，宽度分别1／2，图像键为 `target_global_rgb`／`grasp_rgb`，`return_action_chunk=True`。显式传对应frame ID、A、approach及seed；不传 `live_target_seed`／`segmentation_checkpoint`。

两模型共用冻结双RGB、七维状态、`JOINT_TASK`、`rgb_zlib`和10步生成，保持 `fast_matmul=False`／`bf16=False`／`cuda_graph=False`；各自使用检查点保存的归一化参数。权重和训练约定的描边宽度同时不同，因此本批不能证明描边宽度的单因素因果。

| 来源 | 目标动作块 | 复用 | 新样本预测 |
| --- | ---: | ---: | ---: |
| trial01 #45 | 6 | 0 | 6 |
| trial02 #40 | 6 | 0 | 6 |
| trial03 #45 | 6 | 6 | 0 |
| 合计 | 18 | 6 | 12 |

**本批最多12次新样本预测＋两个worker各2次热身＝16次模型调用。** 每个模型仅启动一次，完成两个新输入的6次预测后关闭，再启动另一个模型。已有完全匹配的新结果也可复用并减少调用；不重跑trial03、不启动baseline、不额外试推理或自动加seed。

执行前简短报告冻结输入、复用数和调用数，随后直接完成本批。沿用已有GPU资源检查；资源不足则保留准备并报告，不干预其他进程。身份／形状／有限值错误、输入错配或worker失败时停止并保留部分结果，不自动重试扩量。保留现有超时与本任务进程清理，不改远端服务或环境。

## 5. 最小实现与产物

固定输出目录：

```text
artifacts/live/stack/stack_thin_outline_20261026_trial03/review/cross_trial_model_comparison/
```

该目录承载跨轮派生分析，原始三轮日志、上一批输入／预测／报告保持原样。目录若已有结果，先识别完成项，只补缺项，不另开版本目录或覆盖预测。

使用项目 `.venv/bin/python`，实现一个小脚本 `run_cross_trial_model_comparison.py`，提供 `prepare`／`predict`／`analyze` 三个阶段，复用已有图像处理、推理和夹爪统计函数。默认不加载模型；不复制成另一套大型实验框架。实现后实际运行这三个阶段，交付真实可用的命令，不把尚未实现的命令写成已完成。

- `prepare`：冻结两份新增输入、引用六份复用预测、提取三帧实际取行集合；保存 `inputs.json` 和一张原图／在线轮廓检查图。CPU检查配对、尺寸、非空、有限状态、来源和预算即可。
- `predict`：只调用12次缺项，逐条保存原始及解码后的完整50×7动作、case／模型／seed、身份和来源。复用的六份原值汇入 `predictions.jsonl`，保留 `reused`、源文件和源记录标识，不记为新推理。
- `analyze`：生成 `analysis.json`、一张J6相对观察状态的轨迹图（标出历史取行范围，细线为各seed，粗线为均值）及 `complete.json`；后者记录实际复用、新样本、热身、错误和完成状态。六关节完整数值保留JSON，无需堆大量图。

不重跑全量回归或部署验收。只有新解析／配对逻辑确有疑点时做针对性CPU检查；沿用worker已有50×7、有限值和夹爪解码校验，不重复计算权重哈希。

## 6. 主分析与收尾判断

**主分析看各帧历史实际取行集合，完整50行及第0／10／49行只作补充。** 同case、同seed、同行配对，统一计算 `1px−2px`；同时报告各模型预测关节角减该帧观察关节角。

| 分析项 | 交付要求 |
| --- | --- |
| J6主结果 | 每帧／每seed的窗口有符号均值、平均绝对差、最大绝对差；各模型相对观察状态的J6均值；再给跨seed汇总，明确统计口径 |
| 方向重复性 | 三个seed的窗口均值是否同号，窗口内是否变号；分别列出两份新帧，不能只报告有利帧 |
| J6采样参照 | 在**同一取行集合、同一个J6分量**上计算同模型不同seed的两两差；不要用六关节总体均值代替J6波动，也不把更大／更小当显著性检验 |
| 其他关节 | J1～J5同窗口的均值／最大差和相对状态动作，防止只挑J6；完整50行差异保留供检查 |
| 夹爪 | 完整块的首次闭合、首次连续至少3行闭合、短脉冲／重新打开；说明闭合行是否落在历史取行集合，窗口外闭合不写成现场已执行 |

预先固定的结论规则：若两份新帧各自三个seed的窗口平均 `1px−2px` J6差均为正，可描述“该差异在这三个保存观察中跨轮同向重复”；另行报告窗口内变号和采样波动，不能升级为普遍规律。若任一新帧不满足，就如实报告适用范围，不换帧／加seed补到满足。

“1px较高”与“1px相对当前状态为正、2px为负”是两个不同结论，后者必须分别核对。即使跨轮复现，也不能据此宣布2px更准、J6导致右偏或应加固定角度补偿。三轮的布局与阶段不同，不是稳定成功率证据；没有可靠FK（由关节角计算末端位姿）就正常交付关节差异，不新建标定系统。

**完成本批后收尾。** 若主窗口差异复现，只提出一个能比较实际对准表现的后续验证建议；若没有复现，降低这条J6线索优先级，说明现有证据不足。不要默认再追加一批离线帧，也不在本任务内切模型、训练或执行真机。

## 7. 两处文案修正与最终交付

在EVIDENCE既有“三模型保存观察对照”段落局部修正：

- “J6在三模型中都上升”改为符合区间的描述：trial03 #40／#45在历史取行窗口内，2px和baseline的平均J6低于观察状态，1px高于观察状态；不要把模型间差值为正与相对当前状态上升混为一谈。
- 1px−2px表格若标为跨seed均值，J6范围应为 **0.377～0.712°**；原0.36～0.73°是分别按seed计算后取范围，统一口径。原始JSON不改。

上一批色相上限和掩码措辞已修复，不再重做。在EVIDENCE相邻位置补一小段本批跨轮结论并链接产物；README无操作变化就不改，不新增计划／总结Markdown。

最终给用户一张表，列出三帧来源、历史取行、两模型J6相对状态、模型间差、seed参照和是否跨轮重复；另报实际12次新样本／4次热身或更少的复用后数量。明确未做真机验证，给出本批结论与唯一下一步即可。

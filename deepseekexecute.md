# DeepSeek执行任务：补齐P2无描边模型运行入口

编写日期：2026-10-25。工作目录：`/home/rentao/program/lerobot-aubo`。

这是用户明确要求交给DeepSeek的本次执行说明，替换此前已结束的“新视觉模型接入”任务；旧文可从Git提交`76a80b61`回溯。不要重建、重跑旧视觉检查脚本。**2026-10-25本任务状态：本地入口、离线检查和6份保存观察的真实GPU预测／验收均已完成。** 下文作为实施说明保留，不重跑已完成的prepare／predict；实际结果及后续P2真机进展统一见[证据记录](examples/phone_to_auboi10/EVIDENCE.md#三根动作模型身份与训练结论)。长期操作和实验结论见README／EVIDENCE。

## 1. 当前决定、任务目标与范围

用户已决定暂停P1末段抓偏诊断，先进入P2，再采集四根／六根动作示教，扩充数据后训练并做真机评价。P1尚未通过；本任务不恢复P1诊断，不开展新训练，也不修改四根／六根采集接口。

**本次请实际完成本地软件实现和离线验证**：给现有`run_joint_smooth.py`增加明确的`--model baseline-stack`入口，使用已经训练好的三根无描边模型。它与原`target-stack`使用相同的七维动作执行逻辑，使后续可以进行公平的有／无目标描边对照。不能只增加一个模型名称或只输出计划而不实现执行路径。

两个模型的区别：

- `baseline-stack`：整个抓放过程始终输入原始全局RGB、原始腕部RGB、七维状态及原固定任务文字，没有指定目标提示。
- `target-stack`：接近阶段全局图带指定目标描边，接受闭爪后的搬运阶段用原图；现有目标身份、分割、跟踪及恢复逻辑保留。
- 两组都由SmolVLA预测`50×7`动作块，七维为`[J1…J6绝对关节角, gripper_pos]`。共享现有按观察年龄取行、伺服平滑及夹爪事件执行机制。

本次首个可比单位为**一次抓取—搬运—释放指令周期**。baseline不指定A／B／C，不实现自动清堆或强制三根顺序；target侧使用已有单目标入口进行配对。一次指令周期结束不代表物理抓取成功。

直接完成：本地代码、必要测试、无设备计划、有限GPU验证的输入和手动命令准备、README／EVIDENCE同步。只读检查远端文件可用于核实身份；相机、机械臂、IO、归位、真实GPU加载／推理、远端文件写入和部署由用户在具体范围明确后手动执行，不在本任务中自动启动。不要提交或推送。

## 2. 阅读顺序与工作区保护

编写时分支`research/depth-vla-phase-a`，HEAD为`76a80b61`（`feat(aubo): preserve P1 hold-close and fixed-observation diagnostics`）。此前代码已按用户要求提交；当前README和EVIDENCE有必须保留的未提交结果与新路线修改，本文件也将成为未提交改动。

```bash
cd /home/rentao/program/lerobot-aubo
git status --short --branch
git log -1 --oneline
git diff --stat
```

依次阅读：

1. [AGENT.md](AGENT.md)。用户明确指定本文件，因此可以维护这份执行说明；不要再创建平行的计划Markdown。
2. [当前阶段](examples/phone_to_auboi10/README.md#当前位置与下一步)、[P2与数据路线](examples/phone_to_auboi10/README.md#p2对照与多根动作数据扩充)、[模型身份](examples/phone_to_auboi10/EVIDENCE.md#三根动作模型身份与训练结论)。
3. 下文列出的当前代码及真实模型配置；行为以当前代码为准，不复用旧文中已失效的“先完成P1”要求。

使用项目`.venv/bin/python`。修改前简要解释文件作用、修改原因与影响；保留已有Makefile改动、原模型、数据、报告、其他人的修改。不要reset／clean、删除原成果、改环境或安装依赖。最小修改即可，不重构整个模型注册／控制系统。

## 3. 模型身份与已核实事实

两组原模型用同一60条训练／9条验证示教，从同一SmolVLA基座训练，各30000步，batch 8、seed42。`uniform`是P1有描边续训候选，**不是P2无描边baseline**。

以下绝对模型路径在SSH别名`gpu`主机上：

| 项目 | 路径／值 |
| --- | --- |
| 远端项目根 | `/home/rentao/program/lerobot-aubo-joint-stack` |
| 本次无描边模型 | `/home/rentao/program/lerobot-aubo-joint-stack/outputs/joint_stack_baseline_smolvla/final` |
| 保留的有描边模型 | `/home/rentao/program/lerobot-aubo-joint-stack/outputs/joint_stack_target_smolvla/final` |
| 已有堆叠服务脚本 | `/home/rentao/program/lerobot-aubo-joint-stack/artifacts/joint_stack_live_service/joint_inference_stdio.py` |
| 已有远端Python | `/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv/bin/python` |
| 固定任务文字 | `Pick one strip and place it in the collection area.` |

本地证据：

- [完成验收](artifacts/capture/stack/joint_stack_train_20261017/training_completion_check.json)：`both_models_accepted=true`、`offline_comparison_completed=true`；这是既有训练验收，不表示baseline现已接入运行。
- [baseline训练计划](artifacts/capture/stack/joint_stack_train_20261017/training_results/baseline/plan.json)、[配置](artifacts/capture/stack/joint_stack_train_20261017/training_results/baseline/final/config.json)、[契约](artifacts/capture/stack/joint_stack_train_20261017/training_results/baseline/final/aubo_joint_contract.json)。
- baseline的契约版本确实是`AuboI10JointLegacyTeleopV2`，使用`joint_contract_record()`与`SmolVLAJointOfflineAdapter`。版本名称含Legacy不意味着应加载旧单根模型。
- target契约是`AuboI10JointTargetApproach`，使用`SmolVLAJointTargetOfflineAdapter`。不能将两组契约混同或绕过严格加载。

**待补的模型身份（2026-10-25已只读补齐）**：本次已按下面命令对指定文件计算并登记SHA-256为`9c4a474c0702bddbe8c44c68b4306bbc3febad967474a6473275af5778151b14`，同时只读核对契约、配置与服务脚本能力；未复制target的SHA、未填假值、未关闭身份核验。现有target SHA及模型不修改。详细记录见[证据](examples/phone_to_auboi10/EVIDENCE.md#三根动作模型身份与训练结论)。以下命令只读远端文件，不加载GPU或写入远端：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 gpu \
  sha256sum /home/rentao/program/lerobot-aubo-joint-stack/outputs/joint_stack_baseline_smolvla/final/model.safetensors
```

同时核对对应小配置、契约、processor文件和服务脚本能力。只读访问受权限限制时，先完成独立的软件准备并集中报告缺失信息；不能宣称真实入口已通过。不重复整目录哈希、不下载大权重、不扫描无关文件；保留服务原有启动身份检查。

## 4. 当前调用链与必须处理的差异

| 文件 | 当前作用／接入点 |
| --- | --- |
| [model_profiles.py](src/lerobot/bamboo_sorting/model_profiles.py) | 纯常量，无设备／模型加载；当前只注册target-stack身份，未注册baseline-stack。注意路径不在`runtime/`下。 |
| [run_joint_smooth.py](examples/phone_to_auboi10/run_joint_smooth.py) | `parse_args()`、`resolve_targets()`、`resolve_config()`、`build_plan()`；当前目标模型身份与部分执行配置耦合。 |
| [live_setup.py](src/lerobot/bamboo_sorting/runtime/live_setup.py) | 读取人工审核种子并构造目标模型选项。baseline不需要种子，不应调用这一目标准备流程。 |
| [continuous.py](src/lerobot/bamboo_sorting/runtime/continuous.py) | `infer()`、`request_observation()`、`accept_prediction()`、`command_goal()`等实际执行逻辑；修改必须同时覆盖运行路径，不能只改计划。 |
| [inference.py](src/lerobot/bamboo_sorting/runtime/inference.py) | `InferencePipe`已支持普通七维契约、显式检查点和完整动作块；优先复用。 |
| [joint_inference_stdio.py](examples/phone_to_auboi10/joint_inference_stdio.py) | 通用GPU服务已有非target适配器及`--return-action-chunk`；本地能力不能代替远端副本核实。 |
| [joint_smooth.py](src/lerobot/bamboo_sorting/runtime/joint_smooth.py) | 共用50×7检查、按观察年龄选取完整七维行、闭爪／释放边界和伺服平滑；本任务原则上复用，不另造控制器。 |
| [joint_target_policy.py](src/lerobot/bamboo_sorting/policy/joint_target_policy.py) | `is_transport()`依据全局图时间与首次接受闭爪时间划分阶段；`TargetInputSession`专属于有目标身份输入。 |

当前代码中已确认的四个陷阱：

1. `mixed=True`的非目标默认分支是2次周期、360秒；target单次是1次、120秒。只加choices会错误继承旧模型设置。
2. `continuous.infer()`给target显式使用`rgb_zlib`，普通分支默认`jpeg95`。若不处理，对照同时改变图像压缩质量。
3. `accept_prediction()`目前依据返回的`target_input.phase`启用搬运按年龄取行。baseline没有此字段，单独放开CLI开关仍会在搬运阶段使用第0行。
4. 闭爪接受时间、在途预测废弃、观察时间和阶段有关联；不能为了复用target阶段机制给baseline伪造目标ID或向服务发送target元数据。

## 5. 推荐的最小实现

### A. 模型选项与CLI

在`model_profiles.py`增加描述清楚的`BASELINE_STACK_*`常量：真实checkpoint、已核实SHA及实际采用的服务路径／必要Python路径。可以复用已有堆叠根目录，不引入通用注册框架。

优先只读确认现有堆叠worker支持普通契约与完整动作块；支持时复用同一脚本，以不同checkpoint和契约参数启动，不修改target服务。若远端副本确实缺能力，准备必要文件清单和独立服务目录`.../artifacts/joint_stack_baseline_live_service/`的手动同步命令，保护已有文件，禁止覆盖target服务或整库部署。

`baseline-stack`应满足：

- 无`--execute`只打印计划，保持原默认模型不变；baseline不需要快照、selection或目标ID。
- `target_conditioned=False`、`live_target_seed=None`、`segmentation_checkpoint=None`、`return_action_chunk=True`。使用原始`global_rgb`／`grasp_rgb`，不构造分割器、不描边、不调用目标输入会话。
- 明确按一次周期运行：默认`--cycles 1`、120秒；本次不扩展多周期baseline，超过1的cycles明确报错，不能静默采用旧两次周期逻辑。
- 接近阶段自动启用既有观察年龄取行。允许`--transport-age-aligned`和`--latency-tolerant`，两组在相同开关下必须得到相同执行设置；默认不自动开启latency-tolerant。
- 拒绝baseline上的目标专用参数：`--target-snapshot`、`--target-selections`、`--target-id`、`--target-sequence`、`--target-preview`、`--track-target-motion`、`--paired-target-separation`、`--multi-target-separation`。这些参数要在模型／设备连接前报错，不能被静默忽略。
- 保持`target_mode`的含义为“需要目标条件输入”。可用一个局部的堆叠执行分支覆盖baseline与target，不要把baseline伪装成target，也不要扩大改写历史模式。

### B. 对齐执行与图像输入

本次已用旧trial04快照只读生成target单目标计划，核实如下默认配置。baseline在相同选项下应逐项匹配；精确值仍从共享代码派生，不复制一套易漂移的控制常量。

| 配置 | 默认值 | 两组显式加`--latency-tolerant`后 |
| --- | ---: | ---: |
| 指令周期／默认时限 | 1／120秒 | 相同，显式max-seconds另计 |
| 控制频率／servo_command_time | 25 Hz／0.08秒 | 不变 |
| 预测年龄上限／看门狗 | 0.5／0.6秒 | 1.5／1.5秒 |
| 相机年龄／相机时间差 | 250／100毫秒 | 400／300毫秒 |
| 控制间隔／状态年龄 | 0.35／0.1秒 | 0.7／0.25秒 |
| 模型响应等待 | 2秒 | 5秒；显式inference-timeout-s仍按既有规则覆盖 |
| 状态过旧时刷新次数 | 1 | 1 |
| 图像编码 | `rgb_zlib` | 不变 |

速度、加速度、TCP／关节范围、跟踪限制、IO到位条件沿用现有代码。该表描述当前入口，不代表允许为了通过实验放宽门槛。

在`infer()`只把新baseline-stack纳入与target相同的无损编码；其他旧无描边模型保持原行为。两组共用动作块验证和选行，六关节与夹爪必须来自同一行，保留首次闭爪／释放边界，不能固定第10行，也不增加平均或滤波方法。

baseline不加载Mask2Former，实际推理耗时可以比target短。记录真实耗时和观察年龄即可，不人为等待制造相同延迟；P2属于两条运行链的比较，不能把所有差别都单独归因于描边像素。

### C. baseline搬运阶段：本地记录，不注入目标提示

给baseline单次会话保存首次**成功接受闭爪命令**的单调时钟时间。该字段在SDK／既有IO调用成功后更新，不能由预测值、计划行或`gripper_pos==100`的猜测提前更新。

对每次发出的观察，使用与target相同的`is_transport(global_rgb_timestamp, accepted_close_time)`语义判定阶段，并把阶段与该次请求索引绑定。本地字段足够；不要创建虚假的`TargetInputSession('baseline')`，不要给baseline的请求塞入`target_input`、`target_seed`、掩码或目标文字。

收到结果时，baseline的搬运取行依据该请求绑定的阶段，不能读取不存在的`result.target_input`，也不能使用收包时的当前阶段覆盖旧观察的阶段。保留共用IO切换后的在途结果丢弃、旧动作块清空、新观察和首次释放保护。关闭transport-age-aligned时继续保持共用的“闭爪后第0行”行为；打开时必须在实际执行路径中按年龄取行。

优先只给baseline增加必要本地状态，target继续复用现有已验证的会话逻辑；若抽取共用逻辑确有必要，必须证明原target行为不变。新请求阶段要能在本地日志中追溯，不宣称它是物理夹持反馈。

### D. 计划与结果可核查

baseline计划至少明确模型名、检查点、SHA、服务路径、原始双RGB、无目标条件、图像编码、周期、取行方式以及上述执行参数，不能只留一个模型别名。可以在两组计划增加必要的共同说明字段，保留旧字段含义。

实际`complete.json`保留`model=baseline-stack`、单次指令周期计数、失败／中止及录像状态；`target_id`保持空，`physical_grasp_success`仍为`null`。实验者预先指定的“评价目标”另记在实验记录中，不传入baseline模型，不能按期望身份自动把实际抓走对象记为成功。

## 6. 离线验收：实现后直接完成

优先扩展已有测试，不为常量单独建文件：

- [test_joint_smooth.py](tests/bamboo_sorting/runtime/test_joint_smooth.py)：已有计划无设备访问、target两模式差异、完整假设备执行、在途丢弃与释放边界测试。
- [test_joint_target_shadow.py](tests/bamboo_sorting/runtime/test_joint_target_shadow.py)：已有`InferencePipe`握手／请求协议模拟，可复用其fake process。
- [test_joint_target_runtime.py](tests/bamboo_sorting/runtime/test_joint_target_runtime.py)：仅当触及目标输入逻辑时纳入；不要为baseline改分割与身份关联。

必要的验收行为：

1. baseline默认及带transport／latency开关的计划正确；与target单目标的执行参数相同，差异只在模型／输入／目标专有机制。无execute时禁止构造InferencePipe、Station、相机或真实模型。
2. baseline不要求目标文件，目标专用参数和非单周期请求在连接前被拒绝。原默认及旧各模型计划行为保留。
3. fake process检查真实构造的SSH参数和JSON请求：baseline路径／SHA／普通契约、返回完整块、原始图像键、`rgb_zlib`；没有target-conditioned／分割／target元数据；错误身份／契约／不完整块仍拒绝。
4. 给实际执行模拟提供“无target_input”的baseline结果，走完接近、成功闭爪、搬运、释放和停止；搬运开启时选中按观察年龄对应的行，而不是总选第0行。
5. 预测闭爪但未执行成功、闭爪前拍摄但晚返回的结果、IO切换后的旧在途结果、释放边界均有对应行为验证；成功走完周期仍不能填写物理成功。优先在已有执行测试中增加少量参数场景，不复制整套大fixture。
6. 保留target-pair、target-stack单次／序列、跟踪与恢复的原断言。已有`test_multi_target_plan_uses_explicit_service_for_remaining_pile`精确比较模式差异，不允许通过删除断言掩盖执行配置漂移。

先运行影响范围测试：

```bash
make test-aubo-offline AUBO_TEST_PATHS='tests/bamboo_sorting/runtime/test_joint_smooth.py tests/bamboo_sorting/runtime/test_joint_target_shadow.py'
git diff --check
```

若确实修改其他模块，增加相应已有测试。最终共享执行链有修改时，运行一次`make test-aubo-offline`做交付回归；已经作为最终验证运行并通过后不要无变化重复跑。此通道只用CPU、假设备和合成输入，不加载真实模型，不接触GPU／硬件。记录真实通过、跳过、失败情况。

实现后还应执行以下**纯计划**命令，核对实际CLI输出：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
  --model baseline-stack --transport-age-aligned --max-seconds 120

PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
  --model target-stack \
  --target-snapshot artifacts/live/stack/stack_target_20261018_trial04/scene_snapshot \
  --target-selections artifacts/live/stack/stack_target_20261018_trial04/scene_snapshot/selection.json \
  --target-id A --transport-age-aligned --max-seconds 120
```

`PYTHONPATH=src`仅对该进程生效。第二条旧快照只用于检查计划，不能加execute拿去新现场运行。第一条baseline入口现已实现并通过离线计划核对，不代表真实GPU或真机已验证。

## 7. 真实GPU验证：准备好，交用户手动执行

本地验收后，准备一个范围有限的“保存观察→baseline真实服务→保存动作块”检查。可以把脚本、冻结输入清单与报告放在既有三根实验目录下的新子目录：

`artifacts/capture/stack/joint_stack_train_20261017/p2_baseline_entry_check/`

该目录为计划输出，不得覆盖已有内容。脚本默认只打印计划；CPU准备与真实预测用明确的stage区分。调用实际入口产生的同一套`InferencePipe`选项，禁止在检查脚本里重建另一套检查点／参数选择。

输入及预算建议：

- 从既有p-101／p-102／p-103各选一条已确认示教，每条一个接近、一个搬运观察，共6个真实保存输入；seed固定42。冻结到JSON后不挑有利样本或扩大扫描。这些是已使用的开发验证数据，不是新盲测。
- 可读取旧关键观察清单定位源数据、episode和frame，但必须从原始双RGB录像及对应状态读取baseline输入。旧P1的`evaluation_manifest.json`和`hold_close_results/prepared_evaluation_inputs.json`中接近图是`target_global_rgb`，已经描边；**不能改键名后喂给baseline**。
- 来源清单位于`artifacts/live/stack/stack_target_p1_pile01_point_tracks_20261024/review/close_precision_training/`。只复用其中定位信息；不要启动旧训练／评价总运行器，不把未来专家动作发给推理服务。
- 6次实样本预测，现有worker启动另有2次合成warmup，合计8次动作模型调用、0训练步。建议整个GPU检查硬超时5分钟，保存输入／日志／输出不超过100 MB，不包括既有权重；不并行启动target模型。
- baseline不加载分割模型；保存实际ready契约、模型身份、请求图像来源、50×7原始与解码动作、有限值／夹爪编码核对及实际耗时。保存数组时兼容JSON list和NumPy类型，不重复旧检查脚本向list调用`.tolist()`的错误。
- 全部预测只落盘，脚本不构造或连接Station、相机、机械臂、夹爪或归位接口；正常／错误／超时均关闭本脚本自己的推理子进程，不停止其他GPU任务。

交付用户可直接运行的准备、必要同步、真实预测命令，注明输入／输出与上述预算。只有确实需要同步文件时才提供精确清单；优先复用已有worker和环境，不为本任务新建项目或Python环境。真实预测及远端写入未获具体执行安排时，完成脚本与命令后停在这一步，不擅自运行，也不把整个本地实现留成待办。

**2026-10-25状态**：脚本`artifacts/capture/stack/joint_stack_train_20261017/p2_baseline_entry_check/check_baseline_entry.py`与6份冻结原始双RGB输入已准备，`prepare`已在本地CPU执行；`predict`未执行，未启动GPU推理或远端写入。CPU假进程复核后已修正脚本读取不存在的`pipe.segmentation_checkpoint`、整体超时未覆盖加载与预测、`verify`未拒绝失败／未完成／超预算结果等问题；请求配置改从入口`options`记录、服务实际配置从`pipe.ready`记录，硬截止覆盖本地准备、worker加载、预测、关闭和complete写入，并强制使用脚本常量300秒／100MB／6预测／2 warmup／8调用，篡改`prepared_inputs.json`预算或扩大`--max-seconds`会被拒绝；`verify`重读输入哈希、拒绝非对象记录，最新状态始终写`verify.json`，旧报告归档。新增`tests/bamboo_sorting/runtime/test_p2_baseline_entry_check.py`36项假进程测试通过，未重复全量回归。

## 8. 现场P2与本次交付区分

本任务不直接执行P2真机实验。真实GPU入口验收通过后，再由用户按新摆放、新快照和新输出目录进行单次配对实验；不能复用旧场景照片驱动新场景。

baseline由模型自行选择对象，评价目标在运行前另行记录；target通过描边指定目标。两组配对摆放、交替运行顺序、使用相同控制选项，报告实际选择、夹住、恰好一根、送达、扰动及中断。目标服从只统计当时可选数k≥2的情况。baseline更快或target遭遇分割恢复等差别需记录，不能把两条链的全部性能差异都说成动作模型的目标理解差异。

若两组物理成功都低，保留失败与不确定结论，不自动恢复P1长期诊断。四根／六根采集及混合训练仍按README的后续路线推进，均不在本任务中执行。

## 9. 文档、回退与最终验收

完成后原位更新[README](examples/phone_to_auboi10/README.md#p2对照与多根动作数据扩充)的实际入口／命令和[EVIDENCE](examples/phone_to_auboi10/EVIDENCE.md#已解决与仍待验证的问题)的真实检查结果。清楚区分：本地实现完成、离线通过、远端服务是否已准备、真实GPU是否执行、真机是否执行。本文件只保留执行说明，不追加第三份长期台账。

回退以停用新`baseline-stack`入口、继续显式选择既有`target-stack`为主；共用代码若出现回归，只撤销本任务对应修改，不reset整个工作区。原模型、旧服务、原始结果均保留，不删除新增实验产物来隐藏失败。

向用户交付：

1. 修改文件、实现理由；特别说明如何处理无损图像编码、baseline搬运阶段与旧模式兼容。
2. baseline真实模型路径／SHA来源、服务路径；两组计划配置的相同项与有意差异。
3. 实际测试数量、失败或跳过项目；完整假设备周期及物理成功字段的结果。
4. 操作者手动执行的有限GPU检查命令、输出、调用预算；尚未执行部分如实标注。
5. 新运行入口的使用说明、回退方式和现场P2前剩余事项。

**最低软件完成标准**：入口真实贯通到执行链，正确模型／契约可核查，原始RGB无目标泄漏，接近与搬运取行按共同语义工作，单次结束与异常收尾通过离线验证，旧target行为保留，真实GPU验证入口具体可运行。代码通过不等于模型已部署，更不等于物理抓取成功。

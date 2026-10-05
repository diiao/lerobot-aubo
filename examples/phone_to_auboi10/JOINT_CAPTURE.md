# 当前采集：双 RGB＋七维关节示教

入口为 `record_joint.py`。模型的数据契约见 [README](README.md#当前契约)。
采集复用手机遥操：手机控制平移和 J6，内部逆运动学生成六关节目标，夹爪开合由操作者切换。
记录的 action 是控制器接受的目标，不能用实测关节角替代；TCP 不进入模型状态或动作。

## 相机配置与体检

本节统一维护相机更换的配置、测试证据、回退与后续操作，其他项目入口仅链接到这里。

2026-10-05 决定暂缓新相机，采集和体检默认恢复 `--camera-set original-global`：
旧 GENERAL 全局相机＋Sonix 腕部相机，沿用 `CameraSetV2.json` 双640×480和原机位。
不改变旧配置文件、历史数据或模型。

新相机功能保留为显式可选项 `--camera-set wide-global`：新 `2MP USB Camera` 全局相机＋原 Sonix 腕部相机，
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

### 新相机恢复使用时的复核（暂缓）

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
  --camera-set original-global \
  --dataset-root datasets/joint_train_run02 \
  --evidence-root artifacts/joint_train_run02 \
  --num-episodes 1 --split train
```

省略 `--record` 只打印计划，不连接设备、不创建目录。数据与证据目录必须互相独立且均不存在。
正式采集由操作者在确认工作区和归位路径后，在同一命令末尾加 `--record`。

## 逐根摆放序列与候选标签

2026-10-05 新增 `capture_placement_sequence.py`，默认选择旧全局相机640×480。
它只打开 `global_rgb`，不打开腕部相机、不连接机械臂、不发送IO。
相机、支架和旧模型机位保持不动。每个独立摆放使用独立新目录和 `placement-id`，
同一序列所有帧及衍生数据属于同一 split，不能拆到训练和测试两边。

先查看计划，不连接设备、不写文件：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/capture_placement_sequence.py \
  --camera-set original-global \
  --output artifacts/placement_sequences/pile_001 \
  --placement-id pile_001 --split train --strips 5
```

现场操作者在同一命令末尾加 `--record` 开始拍摄：

1. 清空料堆区域，手退出画面，按回车保存空场景 `frame_000.png`。
2. 从上方放一根本批选定长度的竹条，不碰动已有竹条；手退出、场景静止后按回车。
3. 重复到3～5根；默认5根，可用 `--strips 3` 指定3根。`q` 或 Ctrl+C 提前结束并保留已有照片。
4. 查看每步的 `*_review.png`，红线表示本次差分候选边界；检查阴影、断裂、邻条位移和交叉处漏标。
5. 第一批拍10个独立摆放。打乱照片、不展示摆放过程，再检查旧640×480图能否判断上下关系。

程序预热后尝试固定当前曝光和白平衡，并保存设置返回值及读回。
可通过 `--exposure <驱动原生数值>`、`--white-balance <色温K>` 显式指定；不同驱动的曝光单位不同。
这些设置只在本次拍摄中应用，退出时只恢复成功写入过的控制项，不修改系统或相机配置文件。
若自动模式读回为 -1（接口不支持），跳过该组写入；不会把 -1 当作白平衡色温写回。
没有任何成功写入时，记录 `restore_status=not_required_no_successful_writes`，不再误报恢复失败。
若出现 `camera_control_lock_unverified`，表示驱动未完整确认锁定；照片仍保留，
不能假定差分变化全来自竹条，先检查亮度漂移。恢复未获确认也会打印提示、写入记录。

输出集中在指定目录：

| 文件 | 内容与用途 |
| --- | --- |
| `plan.json`、`sequence.json` | 相机配置、物理摆放分组、split、阈值、曝光读回和每帧索引；提前退出保留原因。 |
| `frame_000.png`、`frame_001.png`… | 空场景及逐步加条的原始RGB照片，以PNG无损保存。 |
| `frame_*_candidate.png` | 差分候选二值掩码，255为变化区域；不是已确认实例轮廓。 |
| `frame_*_unfiltered.png` | 启用颜色筛选时额外保留的、只限定ROI但未做颜色筛选的差分掩码。 |
| `frame_*_review.png` | 当前原图上的红色候选边界，仅用于复核。 |
| `frame_*_labels.json`、`*_visible.png` | 在旧竹条不移动等假设下估计的实例、遮挡后可见区域、上下关系和上层候选。 |

所有标签都是 `annotation_status=needs_review`、`training_ready=false`；`graspable=null` 表示可夹性未知。
历史完整轮廓只保存为 `footprint_candidate`，不声称恢复了当前被遮挡形状。
同色重叠、阴影或旧竹条位移会造成误标。多边形人工复核见下文；SAM修边和训练数据导出尚未接入。
不自动删掉可疑帧；复核后再决定修正或重拍。已有目录会拒绝覆盖。

差分默认按RGB任一通道变化≥25提取区域，保留面积≥80像素的连通区域。
`--threshold`、`--min-area` 可调整；先用少量训练/开发组定参数，不用保留测试组调参。
空候选、多块变化、大面积变化会在记录中分别提示；这些提示不能发现所有错误。

### 真实首组的离线修正

`pile_001` 首组的完整原图可用，但整图差分混入右上角人体、桌面阴影与旧竹条边缘变化。
操作者确认旧竹条没有移动，不能把这些边缘变化直接解释为物体位移。
曝光/白平衡设置调用全部失败，尚不能声称锁定。随后操作者提供的
[`v4l2-ctl --list-ctrls-menus` 输出](../../artifacts/placement_sequences/pile_001_roi_chroma/camera_controls_user_report.txt)
仅列出 brightness、contrast、hue，当前驱动未暴露曝光或白平衡控制。
继续使用旧相机，保持拍摄光照一致；不继续尝试写不支持的项，不把固定的曝光读数当作已锁定证据。

现增加两个可选参数（不传时保留原始差分方式）：

- `--roi X0 Y0 X1 Y1`：只在原图指定矩形内分析，右边界、下边界不包含；原图仍全幅保存。
  这组旧机位可先用 `280 200 640 480`，复核图用蓝色框显示。以后换摆放范围需要重新确认，不能裁掉竹条。
- `--min-chroma-change 8`：在RGB差分之外，要求Lab颜色空间的a/b颜色分量有足够变化，
  用于抑制主要只有亮度变化的阴影。8是首组开发参数，不是经过独立验证的最优值；0关闭筛选。

颜色筛选也会漏掉同色交叉处；多块变化、候选触及ROI边界等情况也可能导致轮廓不完整。
这些情况下 `layer_order_status=unresolved`、`top_ids_candidate=null`，并传递到序列后续帧。
不能把没检测到重叠当作两根都在上层；`relations_candidate` 仍只是已观测变化区域的重叠假设。

本组离线处理命令如下，输出目录必须尚不存在（这次已运行，重跑请换目录名）：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/capture_placement_sequence.py \
  --output artifacts/placement_sequences/pile_001_roi_chroma \
  --placement-id pile_001 --split train \
  --roi 280 200 640 480 --min-chroma-change 8 \
  --input-images artifacts/placement_sequences/pile_001/frame_000.png \
                 artifacts/placement_sequences/pile_001/frame_001.png \
                 artifacts/placement_sequences/pile_001/frame_002.png \
                 artifacts/placement_sequences/pile_001/frame_003.png
```

对照：[原始与修正候选图](../../artifacts/placement_sequences/pile_001_roi_chroma/comparison.png)、
[离线复核记录](../../artifacts/placement_sequences/pile_001_roi_chroma/offline_review.json)。
人体干扰被ROI排除、阴影明显减少；第一根下端及同色交叉仍有漏标，不作为训练真值。
本轮12项相关测试通过，助手未打开真实相机；下一步拍一组不同摆放，先确认处理效果再决定扩拍。

已有照片也可以离线处理，顺序必须明确列为“空场景、加第1根、加第2根……”：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/capture_placement_sequence.py \
  --output artifacts/placement_sequences/pile_001_relabel \
  --placement-id pile_001 --split train \
  --input-images artifacts/placement_sequences/pile_001/frame_000.png \
                 artifacts/placement_sequences/pile_001/frame_001.png \
                 artifacts/placement_sequences/pile_001/frame_002.png
```

离线模式不加载设备驱动、不连接相机；重新生成到新目录，保留原数据。
软件测试使用合成交叉竹条和假相机；旧相机锁定设置、真实标签质量仍待现场验证。
2026-10-05：周期控制、相机选择、摆放序列及示教采集四组相关测试共89项通过，
另完成三根合成图的离线命令行验证；不作为真实分割质量或抓取成功证据。

### 多边形修边与人工复核

2026-10-05 按用户要求统一复核目录：所有组当前的复核结果集中在
`artifacts/placement_sequences/review/pile_XXX/`，每组只维护这一份，草稿确认后在同一目录更新。
入口为 [review/index.html](../../artifacts/placement_sequences/review/index.html)，待复核组排在前面。
已有13个最终复核组（002～004、007～016）及5个待复核组（017～021）已直接迁入，未重新生成一份副本。
旧采集目录、下载JSON和更早的历史草稿保留；原 `short_strip_review_index.html` 仅跳转至统一入口。

```bash
google-chrome /home/rentao/program/lerobot-aubo/artifacts/placement_sequences/review/index.html
```

`review_placement_sequence.py` 默认保存到上述统一位置，省略 `--output` 即可。
导入同一组的新JSON时，先在临时位置完成生成，再替换本组派生结果；失败保留原结果，成功后清除临时文件。
只允许更新相同源序列和组号的复核目录，不改原始采集，不把草稿自动确认为真值。
导入完成后自动更新统一索引。已通过15项修边工具测试，包含同目录更新、原图保留和更新失败恢复。
本次迁移清单及修改前工具／文档／索引副本位于 `/tmp/aubo-unified-review-b5y8kjf2/`，可按清单反向移动回退。
以下保留各组处理历史，最终结果链接已改为统一位置。

`pile_002` 再次出现同色交叉处差分断裂。2026-10-05 操作者确认：第2根从上方放在第1根上，
第3根再放在前两根上，已有竹条未移动或明显倾斜。因此可按新增轮廓和放入顺序传播遮挡标签，
但不能用该确认代替像素边界复核。第3根右端超出原始640×480画面，扩大ROI也不能找回。

新增 `review_placement_sequence.py`，只读取本地照片和多边形JSON，无新增依赖。
多边形用一圈顶点描述竹条轮廓，每一步只标本步新增的竹条；同色交叉处无需依赖差分。
本组的助手草稿和结果位于：

- [草稿顶点与摆放确认](../../artifacts/placement_sequences/pile_002_polygon_draft.json)。
- [差分与多边形对照图](../../artifacts/placement_sequences/pile_002_polygon_review/comparison.png)：左侧红色为差分，右侧绿色为草稿。
- [离线复核页](../../artifacts/placement_sequences/pile_002_polygon_review/review.html)：内嵌原照片，无需网络或服务器。

在仓库根目录打开复核页：

```bash
google-chrome artifacts/placement_sequences/pile_002_polygon_review/review.html
```

1. 每页检查本步新增的一根，拖动圆点修边；Shift＋点击添加顶点，右键顶点删除。
2. 新生成的页面无需填写复核人。每根检查后点“确认当前轮廓”；修改顶点或截断标记会取消该根确认。
   下载前检查页面上的已确认数量。旧页面和记录保持原样，导入兼容带有或不带 `reviewed_by` 的文件。
3. 第3根保留“竹条被画面截断”，只标画面内部分，不推测画面外端点。
4. 点击“下载修正 JSON”，得到 `pile_002_polygons.json`。浏览器下载不会修改原始文件或已有掩码。
5. 用下载文件更新该组统一复核目录。以下以017为例，替换成实际组号和下载路径；无需另建最终目录：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/review_placement_sequence.py \
  --source artifacts/placement_sequences/pile_017 \
  --polygons pile-json--/pile_017_polygons.json
```

输出包含原图副本、`polygons.json`、`review.json`、各步多边形掩码、可见区域掩码、层序候选、对照图和复核页。
原始采集目录不修改，复核目录按同组更新；所有衍生数据保留同一 `placement_id` 和 split。
如摆放顺序或旧条静止未确认，工具不会向后传播旧条可见区域，也不会给出上层候选。
摆放确认成立时，后放轮廓遮掉先放轮廓的投影重叠部分；这个规则依赖投影不变，旧条移动或倾斜的序列需另行处理。

原始草稿保留为 `polygon_draft`。2026-10-05 操作者下载的 `pile-json--/pile_002_polygons.json`
保存了第3根的顶点修正，但因修改复核人字段仅留下第3根的确认标记；随后在对话中明确回复
“是的，我都确认过”，确认前两根保持原轮廓、三根均已复核。
据此新建[确认记录与多边形](../../artifacts/placement_sequences/pile_002_polygons_confirmed.json)，
记录对话来源、原下载路径和原确认标记，未覆盖下载文件、原图或草稿，未改动任何导出的顶点。
最终结果位于 [review/pile_002](../../artifacts/placement_sequences/review/pile_002/review.json)，
三步均为 `polygons_reviewed`，可见区域和层序候选已重新生成。
前两帧的 `full_frame_mask_eligible=true`，表示通过此工具的轮廓确认与非截断条件；
有任一截断物体的整帧 `full_frame_mask_eligible=false`，本组第3帧不能直接进入完整轮廓训练集。
所有结果仍保留 `training_ready=false`、`graspable=null`：边界与层序复核不等于已经判定二指夹爪有下指空间。
本组已完成轮廓复核流程；后续用新编号采集独立摆放，让整根竹条落在原图内，先检查新组再扩大采集。

验证：修边与序列生成相关21项测试通过，覆盖同色交叉、遮挡传播、分组一致性、截断和原始文件保留。
另用独立临时配置的无界面Chrome检查页面渲染，并在测试副本验证拖点、增删顶点、翻页、确认与JSON导出。
这证明工具逻辑，不代表真实图像的标签精度或抓取效果。

### pile_003 采集检查

2026-10-05 检查了本组4张原图和采集记录：空场景及3次新增均已保存，`stop_reason=completed`。
前两根完整入镜，第3根右端仍超出原图右边界；这是实际视野截断，不能靠扩大ROI恢复。
差分连通块依次为1、2、2，第2、3步同色交叉处漏标，颜色筛选尚不能代替轮廓复核。
相机曝光/白平衡依旧未获锁定确认，未写入不支持的控制项。

操作者已对本组单独确认逐根从上方放入、旧条未动且未明显倾斜。
据此生成[带摆放确认的草稿](../../artifacts/placement_sequences/pile_003_polygons_with_order.json)及
[离线复核页](../../artifacts/placement_sequences/pile_003_polygon_review_with_order/review.html)：

```bash
google-chrome /home/rentao/program/lerobot-aubo/artifacts/placement_sequences/pile_003_polygon_review_with_order/review.html
```

三步上层候选依次为 `strip_001`、`strip_002`、`strip_003`，来自摆放顺序与轮廓重叠，
不是单帧感知模型的预测。上述目录保留最初的未确认草稿；第3根保留截断标记。
2026-10-05 操作者已完成复核，下载文件 `pile-json--/pile_003_polygons.json` 三根均为 `reviewed=true`。
已用该文件生成[最终复核记录](../../artifacts/placement_sequences/review/pile_003/review.json)和
[实例可见区域叠加图](../../artifacts/placement_sequences/review/pile_003/frame_003_instances.png)。
三步均为 `polygons_reviewed`；前两帧 `full_frame_mask_eligible=true`，第3帧因截断为 false。
已核对导入多边形与下载文件一致，4张原图副本与源文件一致；未修改顶点或原始采集。
所有帧仍为 `training_ready=false`，可夹性尚未标注，尚未接入训练。
原始采集和最初草稿均保留。后续采集前应查看相机画面确认竹条两端留有余量，
仅凭桌面位置估计范围不能避免截断；这两组尚不能证明自动标签质量达标。

### pile_004 采集检查与免填复核人

2026-10-05 本组空场景及3次新增均完整保存；逐帧查看确认三根竹条均完整入镜，没有截断。
差分连通块仍为1、2、2，第2、3步同色交叉处漏标，因此继续使用多边形修边。
操作者对本组确认从上方逐根放入、旧条未动且未明显倾斜。
[草稿文件](../../artifacts/placement_sequences/pile_004_polygon_draft.json)记录了确认来源，
[复核页](../../artifacts/placement_sequences/pile_004_polygon_review/review.html)可直接打开：

```bash
google-chrome /home/rentao/program/lerobot-aubo/artifacts/placement_sequences/pile_004_polygon_review/review.html
```

按用户要求，新页面已去掉复核人输入框和姓名必填检查；逐根确认、修改当前轮廓后取消该根确认仍保留。
导入接受没有 `reviewed_by` 的JSON，也保留旧文件中的该字段；不会自动把未确认轮廓标记为已确认。
最初草稿保留为 `truncated=false`、`reviewed=false`。2026-10-05 操作者已下载
`pile-json--/pile_004_polygons.json`，三根均为 `reviewed=true`、`truncated=false`，未修改草稿顶点。
已生成[最终复核记录](../../artifacts/placement_sequences/review/pile_004/review.json)和
[实例可见区域叠加图](../../artifacts/placement_sequences/review/pile_004/frame_003_instances.png)。
三步均为 `polygons_reviewed`、`full_frame_mask_eligible=true`，是目前已复核组中首个最终三根帧也无截断的序列。
已核对导入多边形与下载文件一致、4张原图副本与源文件一致；原图、下载文件和历史复核结果保持原样。
仍未接入训练或可夹性判断，`training_ready=false`；一组完整复核结果不代表自动标注或感知模型已经达标。
本次11项修边工具测试通过，覆盖无姓名、空姓名与旧版有姓名文件导入；
另在Chrome测试副本验证了无姓名逐根确认、编辑后取消确认和JSON导出。

### pile_005 采集检查与每五组集中检查

2026-10-05 已查看本组4张640×480原图及3张差分复核图，记录为 `finished=true`、
`stop_reason=completed`、`split=train`。前两根完整入镜；第3根右端贴到原图右边界，
完整端头无法确认，暂按疑似截断处理，不作为完整轮廓已通过的阶段。
差分连通块为1、2、2，第2、3步在同色交叉处漏标，第3步另有 `candidate_touches_roi_boundary`。
这是原图边界处的问题，扩大ROI不能恢复画面外内容。后续摆放需给竹条两端留明显余量。
本次只检查采集质量，未生成新多边形或将其标记为人工已复核；本组仍为 `needs_review`、
`training_ready=false`，保留全部原图，前两阶段可继续进入轮廓复核。

用户说明此后拍摄场景保持不变，并改为每拍5组集中检查一次、每次提供后续5组现场拍摄命令。
本组[空场景](../../artifacts/placement_sequences/pile_005/frame_000.png)中未见先前单张场景检查里的
后侧新增深色表面，后续以本组实际布局为参照，不把先前快照当成当前背景。
固定的是相机、承托布局和拍摄条件；每组仍独立重新摆放竹条，并拍自己的空场景。
集中检查原图完整性、入镜范围、差分缺失和组内场景变化，不自动把采集检查等同于轮廓或摆放条件已确认。
需要补充的现场条件按批次集中确认，异常按组号记录，不反复询问已经确认过的旧组。
拍摄仍由用户手动启动，助手不因批次安排自动打开设备。

此前提供的 `pile_006～010` 长条批次计划已被下述短条重启安排替代，不再沿用旧命令中的集合划分。
`pile_006` 已完成4帧采集，记录为 `split=train`、`stop_reason=completed`；本次仅核对记录，尚未逐图复核。

**2026-10-05 改用短条重新开始静态采集。**用户明确选择改用更短的竹条，宽25 mm、厚8 mm及其他条件不变，
当时实际长度待提供；2026-10-05 用户后续明确确认本批为40 cm，导出登记为400×25×8 mm。
旧 `pile_001～006`、已有多边形、动作示教和模型全部保留，
不自动删除、覆盖或并入本轮短条训练。
短条第一批为 `pile_007～011`：007～010为 `train`，011为 `validation`；已完成拍摄，检查见下节。
每组3根同长度短条，只拍普通“一根压一根”，组内不混用60 cm长条；011独立重新摆放。
拍摄程序没有长度参数，批次长度先在本节登记，之后导出时沿用该批次范围，不由编号或图像猜测长度。
保持已约定的相机与承托布局，每组拍自己的空场景；完成这5组后集中检查，再给下一批5组命令。
若看到明显截断或旧条移动，可提前报告异常组。长度改变不自动证明旧60 cm动作模型能抓短条，
静态采集不启动动作示教、训练或实机执行。

### pile_007～011 短条首批检查

2026-10-05 已逐张查看5组共20张原图及15张差分复核图，并核对采集记录和文件引用：

- 五组均保存空场景及3次新增，图像为640×480，`finished=true`、`stop_reason=completed`；引用文件无缺失。
- 15张非空阶段图的竹条均完整入镜，未见截断或触及ROI边界；未见明显相机／承托布局变化或手部遮挡。
  从本次采集质量检查未发现需立即重拍的组，不代表像素标签已验收。
- 每组差分连通块依次为1、2、2；第2、3次新增在同色交叉处漏标，仍需多边形修边和人工确认。
  曝光／白平衡仍为 `lock_verified=false`，不重新尝试驱动不支持的锁定项。
- 007～010为 `train`，011为 `validation`：共4组训练、1组验证，不能将15张非空图当作15组独立样本。
- 用户在本次对话集中回复“五组均满足”，明确确认007～011都从上方逐根放入、旧条未移动或明显倾斜。
  该确认只适用于这五组，不自动套用未来批次，也不等于多边形轮廓已确认。

首批检查时尚无多边形；现已随第二批生成草稿，见下节。原始采集及候选标签保持原样，
不把交叉漏标的差分掩码直接用于训练。短条实际长度仍待用户提供。

### pile_012～016 检查与前10组轮廓复核入口

2026-10-05 已逐张查看012～016的20张原图及15张差分复核图，核对采集记录与文件引用。
五组均为640×480，保存空场景及3次新增，`finished=true`、`stop_reason=completed`，无缺失引用。
15张非空阶段图的竹条均完整入镜，无截断或ROI边界告警；未见明显相机／承托布局变化或手部遮挡，
本次检查未发现需要立即重拍的问题。差分连通块仍全部为1、2、2，同色交叉处漏标，不能直接充当完整实例标签。
相机锁定仍未获确认，不重复尝试不支持的控制项。

用户在本次对话回复“五组均满足”，单独确认012～016均从上方逐根放入、旧条未移动或明显倾斜。
该确认已连同007～011之前的批次确认分别写入对应草稿的 `placement_confirmation_source`，不推及未来组。
第二批检查时短条共10个已拍摄组：007～010及012～015为8组 `train`，011和016为2组 `validation`；
共40张原图、30张非空阶段图，当时尚无人工确认完成的短条轮廓组；后续导入结果见下一节。

已根据原图逐根编写助手视觉多边形草稿，生成到全新 `pile_007_polygon_review`～`pile_016_polygon_review`，
原始草稿另存各自 `pile_XXX_polygon_draft.json`。本次未接入自动拟合或SAM，没有修改原图与采集标签。
统一入口：[轮廓复核索引](../../artifacts/placement_sequences/review/index.html)。

```bash
google-chrome /home/rentao/program/lerobot-aubo/artifacts/placement_sequences/review/index.html
```

最初索引链接这10个现有模板生成的复核页；现已在同一入口更新待复核和已完成分组，历史草稿页保留。
每组逐根检查、修边、确认后下载对应JSON，无需填写复核人。索引不会随下载自动刷新。
初始草稿均为 `reviewed=false`、`truncated=false`，
输出均为 `polygon_draft`、`full_frame_mask_eligible=false`、`training_ready=false`、`graspable=null`。
层序候选由已确认的摆放条件和草稿重叠推导，不是感知模型预测，也不表示边界已确认。
下载后的JSON离线导入并更新统一的 `review/pile_XXX`，不再另建最终目录，也不能跳过人工轮廓确认。

本次已查看10组最终阶段实例叠加图，核对40张原图副本与源文件一致、30个草稿阶段的未确认状态及分组保持一致；
沿用现有生成工具与模板，未改工具代码，未为本次批量生成重复跑历史回归。
后续017～021批次已完成，见下一节。继续保持每5组集中检查的安排，不把已拍摄数量当作已完成训练标签数量。

### pile_017～021 检查与007～016人工复核导入

2026-10-05 用户误触造成017提前结束后，明确要求删除重拍；仅删除当次 `pile_017`，并确认目录已不存在。
当前017是重拍结果，记录为 `finished=true`、`stop_reason=completed`，保存空场景及3次新增，已无原误触缺少新增物体的情况。
本次逐张查看017～021的20张原图及15张差分复核图；五组图像均为640×480、记录完整、引用文件无缺失。
15张非空阶段图竹条均完整入镜，无截断或ROI边界告警，未见明显相机／承托布局变化或手部遮挡，
未发现需立即重拍的问题。每组差分连通块仍为1、2、2，同色交叉处仍漏标；相机锁定未获确认，未重试控制写入。
用户对“重拍后的017及018～021是否从上方逐根放入、旧条未移动或明显倾斜”回复“五组均满足”，
已将本批专属确认来源写入对应草稿，未套用旧批次确认。

同时发现 `pile-json--/pile_007_polygons.json`～`pile_016_polygons.json` 的30根轮廓均为
`reviewed=true`、`truncated=false`。与初稿相比，012修改1根、014修改2根、015修改2根、016修改3根，
其余组顶点未变。已原样导入，最终结果现集中在 `review/pile_007`～`review/pile_016`，
没有改写下载文件或用户顶点。10组的30张非空阶段图均为 `polygons_reviewed`、`full_frame_mask_eligible=true`，
但仍为 `training_ready=false`、`graspable=null`；完成轮廓复核不代表物理可夹或已接入训练。

017～021已按原图编写15根助手视觉草稿，保存为各组 `pile_XXX_polygon_draft.json`，
生成的复核页现集中在 `review/pile_017`～`review/pile_021`，供后续同组更新。
全部 `reviewed=false`、`truncated=false`，不自动确认轮廓。
统一复核索引已更新：待复核017～021置前，已完成组展示确认状态，旧组无需重复确认。
修改索引前保留副本，原始采集、旧草稿页、下载JSON及最终结果均保留。
本次核对新生成15个目录的60张原图副本与源文件一致、导入多边形与下载或草稿文件完全一致，
并检查30个已复核阶段与15个草稿阶段的状态；查看有顶点修改组及新五组的最终阶段实例叠加图。
未修改生成工具或模板，未重复运行历史回归；未启动设备、训练或模型推理。

短条当前共15个拍摄组（12组train、3组validation），60张原图、45张非空阶段图。
其中10组已完成轮廓复核（8组train、2组validation），另5组待复核；实际短条长度仍待用户提供。
下一批 `pile_022～026` 已核对未占用：022～025为 `train`，026为 `validation`。
继续同长度短条的普通逐层叠放，各组独立重新摆放并拍空场景；拍完026后集中检查。

### 短条 pile_022～026 检查（2026-10-05）

五组均正常完成空场景＋逐根新增3根，共20张640×480原图；022～025为 `train`，026为 `validation`。
逐张查看原图及15张差分复核图，木条两端均完整入镜，未发现需要重拍的截断或缺帧。
同色交叉处的差分仍会断裂，不能直接当成完整轮廓；每组新增阶段的连通分量依次为1、2、2。
用户集中确认本批“五组均满足”从上方逐根放入、旧条未移动或明显倾斜，已记录到本批标注中。

人工观察照片后生成15根轮廓草稿，集中保存在 `artifacts/placement_sequences/review/pile_022～026/`，
入口仍为 [统一复核索引](../../artifacts/placement_sequences/review/index.html)。
每组只维护这一份结果，后续下载JSON导入时直接更新本组，不新增最终目录或另存根目录草稿。
目前均为 `reviewed=false`、`training_ready=false`、`graspable=null`；摆放确认不代替逐根轮廓确认。
已核对20张原图副本与源文件一致、15个阶段的草稿状态和上层候选、5张最终阶段实例叠加图及46个索引链接。
本次未发现 `pile-json--/` 中017～026的新下载复核JSON；017～026共10组仍待轮廓复核。

短条当前共20组（16组train、4组validation），80张原图、60张非空阶段图；
007～016共10组已完成轮廓复核，017～026共10组待复核。短条实际长度仍待提供。
下一批 `pile_027～031` 已核对未占用：027～030为 `train`，031为 `validation`。
继续同长度短条的普通逐层叠放，各组独立重新摆放并拍空场景；拍完031后集中检查。
本次只新增派生复核结果并更新本说明，原始采集未修改；说明与索引修改前副本在
`/tmp/aubo-review-022-026-2ww0f53x/`。未修改代码，未启动设备、训练或模型推理。

### 短条 pile_027～031 检查（2026-10-05）

五组均完成空场景＋逐根新增3根，共20张640×480原图；027～030为 `train`，031为 `validation`。
查看20张原图及15张差分复核图，未发现缺帧、出画或需要重拍的问题；027最右端仍有余量。
029、031中两根条夹角较小，复核时关注交叉处边界；后续仍按普通逐层叠放采集，不特意构造特殊情况。
五组各阶段差分连通分量均为1、2、2，同色交叉处漏分仍需人工轮廓修正。
用户集中确认本批“五组均满足”从上方逐根放入、旧条未移动或明显倾斜，已写入本批标注。

15根人工轮廓草稿已加入 `artifacts/placement_sequences/review/pile_027～031/`，
入口仍为 [统一复核索引](../../artifacts/placement_sequences/review/index.html)，每组仅维护一份结果。
均保持 `reviewed=false`、`training_ready=false`、`graspable=null`，需用户逐根确认轮廓。
核对20张原图副本与源文件一致、15个阶段的草稿状态及上层候选，并查看5张最终实例叠加图；56个索引链接有效。
`pile-json--/` 未发现017～031的新复核JSON。短条累计25组（20组train、5组validation），
100张原图、75张非空阶段图；007～016共10组已复核，017～031共15组待复核，实际短条长度仍待提供。

下一批 `pile_032～036` 已核对未占用：032～035为 `train`，036为 `validation`。
保持场景和短条长度，各组独立重摆、先拍空场景，拍完036后集中检查。
本次只新增派生复核结果并更新本说明；修改前说明与索引副本在 `/tmp/aubo-review-027-031-meruq3p1/`。
原始采集及代码未修改，未启动设备、训练或模型推理。

### 复核导入至 pile_031 与短条 pile_032～036 检查（2026-10-05）

用户报告已复核至031；本次实际收到 `pile-json--/pile_017～031_polygons.json` 共15份新增复核文件，
45根均为已确认、未截断，其中13组的26根修改了顶点。核对编号、尺寸、摆放确认来源及多边形有效性后，
按下载JSON原样更新 `review/pile_017～031/`，不另建最终目录；旧组007～016保持原结果。
检查15组最终阶段叠加图，导入轮廓与下载JSON一致，新增45张非空阶段图通过当前轮廓资格检查。

新拍032～036各有空场景＋3张逐根新增图，共20张640×480原图；032～035为train，036为validation。
查看全部原图及15张差分复核图，未发现出画、缺帧或需重拍问题；差分连通分量均为1、2、2，
同色交叉处漏分仍以人工轮廓处理。用户确认本批“五组均满足”从上方逐根放入、旧条未移动或明显倾斜。
15根人工草稿已加入 `review/pile_032～036/`，保持未复核；统一入口为
[review/index.html](../../artifacts/placement_sequences/review/index.html)。

短条当前共30组（24组train、6组validation），120张原图、90张非空阶段图。
007～031共25组完成轮廓复核（20组train、5组validation），75张非空阶段图通过当前完整轮廓资格检查；
032～036共5组待复核（4组train、1组validation）。此计数不包含旧长条及开发组，编号031不代表31个已复核短条组。
所有组仍为 `training_ready=false`、`graspable=null`，短条实际长度待提供。
本次核对20个更新／新增目录的80张原图副本、60个阶段标签、15份导入JSON及66个索引链接。
修改前017～031结果、说明和索引保存在 `/tmp/aubo-review-through036-fvpvj91i/`；原始采集和代码未修改。
未启动设备、训练或模型推理。

下一批037～041已核对未占用：037～040为train，041为validation。
保持当前场景、同长度短条及普通逐层叠放，每组独立重摆并先拍空场景，拍完041后集中检查。

### 短条 pile_037～041 检查（2026-10-05）

五组正常完成空场景＋逐根新增3根，20张640×480原图齐全；037～040为train，041为validation。
查看全部原图、15张差分复核图，未发现木条出画或需重拍问题。037～040差分连通分量为1、2、2，
041为1、2、1；041第三阶段虽连通，交叉处边界仍不准确，不能省略人工轮廓复核。
用户集中确认本批“五组均满足”从上方逐根放入、旧条未移动或明显倾斜，已记入标注。
15根人工草稿已加入 `review/pile_037～041/`，仅维护这一份结果；
[统一复核入口](../../artifacts/placement_sequences/review/index.html)已更新。
核对20张原图副本、15个阶段草稿标签及上层候选，查看5张最终实例叠加图；76个索引链接有效。

未发现032～041的新下载复核JSON。短条累计35组（28组train、7组validation），
140张原图、105张非空阶段图；007～031共25组已复核，032～041共10组待复核。
草稿保持未确认；所有结果仍为 `training_ready=false`、`graspable=null`，短条实际长度仍待提供。
本次仅新增派生结果并更新本说明；修改前说明及索引在 `/tmp/aubo-review-037-041-g0xd1ayk/`。
原始采集和代码未修改，未启动设备、训练或模型推理。

下一批042～046已核对未占用。为补齐首轮30组train＋10组validation，
042～043安排为train，044～046安排为validation；在采集前指定，不改变旧组划分。
继续普通逐层叠放，保持场景及短条长度，每组独立重摆并先拍空场景。
拍完046后集中检查并完成剩余轮廓复核，再进行导出／训练准备检查；数量达标不等于可直接训练或抓取已验证。

### 短条 pile_042～046 检查与首轮拍摄完成（2026-10-05）

五组均完成空场景＋逐根新增3根，共20张640×480原图；042～043为train，044～046为validation。
查看全部原图及15张差分复核图，木条完整入镜，未发现缺帧或需要重拍问题。
各组差分连通分量均为1、2、2，同色交叉处仍漏分，使用人工轮廓草稿补充。
用户集中确认本批“五组均满足”从上方逐根放入、旧条未移动或明显倾斜，已记入本批标注。
15根草稿已加入 `review/pile_042～046/`，保持未复核；
[统一复核入口](../../artifacts/placement_sequences/review/index.html)继续每组仅维护一份结果。
核对20张原图副本、15个阶段的草稿状态与上层候选，查看5张最终阶段叠加图；86个索引链接有效。

首轮短条007～046共40组已拍齐：30组train＋10组validation，160张原图、120张非空阶段图。
其中007～031共25组已完成轮廓复核（20组train、5组validation）；032～046共15组待复核
（10组train、5组validation），本次未找到这些组的新下载JSON。
所有结果仍为 `training_ready=false`、`graspable=null`，实际短条长度仍待提供。
数量达标不代表已完成训练数据导出、模型训练或抓取验证。

建议先完成032～046逐根复核，下载JSON放入 `pile-json--/` 后集中导入，再检查导出及训练准备条件。
本次不继续预分配新的五组，普通场景首轮结果检查后再决定补采范围；特殊结构仍不在当前采集范围。
本次仅新增派生结果并更新本说明；修改前说明及索引副本在 `/tmp/aubo-review-042-046-kpt1jewg/`。
原始采集和代码未修改，未启动设备、训练或模型推理。

### 首轮40组轮廓复核完成（2026-10-05，导出前记录）

032～046共15份下载JSON已全部导入原有 `review/pile_XXX/`，045在本轮由用户补齐下载。
45根轮廓均已确认、未截断，其中14组的26根相对草稿调整了顶点；按下载JSON原样导入，
检查15张最终阶段实例叠加图，未额外创建最终目录。修改前15组结果、索引及说明保存在
`/tmp/aubo-final-review-import-yg9rexgr/`，原始采集及下载JSON保持原样。

短条007～046首轮40组现已全部完成轮廓复核，无待复核短条组：

| 集合 | 独立摆放组 | 空场景 | 非空阶段图 |
| --- | ---: | ---: | ---: |
| train | 30 | 30 | 90 |
| validation | 10 | 10 | 30 |
| 合计 | 40 | 40 | 120 |

完整性检查通过：160张RGB可解码且尺寸为640×480；120张非空阶段图满足当前完整轮廓资格条件；
240份逐帧可见实例掩码均非空、尺寸正确、二值、位于对应整根投影内，且同帧可见实例无像素重叠。
编号、原始序列与结果的split一致，无重复placement_id跨集合；普通场景各阶段上层候选符合新增顺序。
本次更新的60张RGB副本与原图一致，15份导入标注与下载JSON一致，86个索引链接有效。
上述检查证明文件和标签一致性，不独立证明所有物理摆放互异、泛化效果或三维可夹性。

下一步为静态感知数据导出与训练入口准备：将原始RGB、可见实例掩码及上层／被压类别转换成训练格式，
显式保留空场景、placement_id和既有30/10划分；不使用复核描边图作为模型输入。
当前源码入口仍仅实现采集和复核，尚无本流程的训练格式导出器或分割训练入口；未选定权重或安装新依赖。
全部结果继续保持 `training_ready=false`、`graspable=null`；未启动训练、推理或硬件。
短条实际长度尚未提供，不能从图像或组号推定；长度登记与后续执行适配仍待完成。
本阶段先进入导出准备，不继续扩大普通照片数量；特殊结构仍待普通场景可行性验证后另行安排。

### 分割数据导出与训练准备（2026-10-05，当前状态）

用户确认007～046短条长度为40 cm，本批尺寸为 **400×25×8 mm**。
已完成离线导出和训练入口准备。用户先要求清理空间，随后明确授权删除指定pip缓存和旧smoke目录，
并继续部署、启动训练。目前GPU独立目录中的首轮Mask2Former分割训练已完成，机械臂动作模型未改动。
历史复核JSON中的长度null保持原始记录，导出选择清单使用本次用户确认的尺寸，不回写旧采集证据。

**文件与格式。**选择清单为 `configs/aubo_i10/short_strip_segmentation.json`，显式列出40组及原有30/10划分。
导出目录为 `datasets/short_strip_segmentation/`（约97 MiB），包含 `dataset.json`、`audit.json`、
`images/` 原始RGB副本和 `instances/` 16位实例PNG。PNG的0表示背景，正整数仅表示该帧实例身份；
类别在JSON的 `segments` 中单独记录：0=`top_strip`，1=`covered_strip`。
这两个0属于不同字段，背景不作为第三种前景类别。空场景显式使用全零实例图和空 `segments`。
下层被遮住部分从可见掩码中扣除，断成两段的同一根仍是同一个实例；完全不可见实例不生成前景目标。
原图无复核描边、无顺序提示，组号和帧号仅用于数据对应，不作为模型输入。此格式为本项目便携训练格式，并非COCO JSON。

| 集合 | 组 | RGB图（含空场景） | 空场景 | 可见实例 | 上层实例 | 被压实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 30 | 120 | 30 | 180 | 90 | 90 |
| validation | 10 | 40 | 10 | 60 | 30 | 30 |

本次无排除帧。导出前核对源图、确认标记、可见掩码、上下关系和集合；导出后检查图像／实例图尺寸、
像素ID、面积、边界框和组级划分。原始review中的 `training_ready=false` 和 `graspable=null` 未修改；
导出资格由单独的 `audit.json` 表达，不意味着物理可夹或已开始训练。

```bash
# 已执行；导出目录必须是新目录，禁止隐式覆盖。
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/export_strip_segmentation.py \
  --selection configs/aubo_i10/short_strip_segmentation.json \
  --output datasets/short_strip_segmentation

# 已通过；默认只核对数据，不加载模型、不训练。
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/train_strip_segmentation.py \
  --stage preflight --dataset datasets/short_strip_segmentation
```

**训练入口。**`train_strip_segmentation.py` 已提供显式 `--stage train`，基座候选为
[`facebook/mask2former-swin-tiny-coco-instance`](https://huggingface.co/facebook/mask2former-swin-tiny-coco-instance)。
依照 [Transformers 4.57.1 Mask2Former接口](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/mask2former)，
训练时加载本地模型目录，分类头改为上述两类，同时重建与类别数量相关的 `criterion.empty_weight`；
其余权重要求正常载入，不自动联网下载。
RGB保持640×480，采用预训练处理器的归一化，首版不做裁剪、拉伸或数据增强。
掩码和类别直接作为每帧实例集合；全空批次只计算无目标分类损失，不构造假的背景实例。
默认30轮、batch=2、AdamW，骨干学习率5e-6、其余5e-5，梯度裁剪1.0，seed=42；本批对应1800次更新。
这些是本次实际启动的首轮基线参数，尚未调优。

每5轮验证，记录两类的precision/recall/F1、空场景误检率及损失；匹配阈值为可见掩码IoU≥0.5，
置信度阈值0.5。它们是固定阈值指标，不是COCO mAP；验证集用于选模型，不是独立测试集。
按两类平均F1选择best，同分时比较验证损失，保留训练前基线；仅保存 `best/` 与 `final/`。
结束后重载best再验证，写入逐帧预测掩码、`best_validation.json` 和 `complete.json`。
本入口不支持隐式续训，输出必须为新目录。真实预训练权重已加载并完成1800次优化；
首轮60步完成时GPU总显存约6204 MiB，含桌面原占用，日志计时约12秒（含初始验证）。

依赖补充清单为 `examples/phone_to_auboi10/strip_segmentation_requirements.txt`：
Transformers 4.57.1、SciPy 1.15.3。当前本机原 `.venv` 未改动；测试用补充依赖位于
`/tmp/aubo-segmentation-deps/`，通过单条命令的PYTHONPATH使用。
GPU机现有可复用环境为 `/home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv`，
只读核对得到torch 2.10.0、torchvision 0.25.0、Transformers 4.57.1、NumPy 2.2.6、Pillow 12.3.0，缺少SciPy。
已在新训练目录的 `deps/` 单独安装SciPy 1.15.3，通过启动脚本的PYTHONPATH使用，原环境未修改。

**验证证据。**新增 `tests/bamboo_sorting/test_strip_segmentation.py` 的10项测试通过，覆盖
原RGB保留、交叉遮挡、空场景、集合与确认错误、缺失／重叠掩码、截断排除、RGB通道、实例匹配，
以及随机小模型在纯合成数据上的空批次／混合批次前反向、2次优化更新、保存重载与评价入口。
首次真实启动在权重检查阶段停止，未执行优化：Transformers 4.57.1的 `mismatched_keys` 为名称列表，
旧检查误按元组读取，并漏列 `criterion.empty_weight`。已修正名称解析，只允许三个与类别数量相关的字段变化；
原合成测试新增80类预训练模型转2类的保存／加载／优化／重载覆盖，10项测试再次通过。
修正后真实160张图的远端preflight通过，已完成训练前40图验证、30轮真实训练与best重载验证。
训练前随机分类头的验证macro F1为0，不是训练后性能；合成测试和训练损失都不证明物理抓取成功。

**GPU空间及已授权清理。**别名 `gpu` 为RTX 4090 D（约48 GiB）。清理前根分区可用11475922944字节。
`~/.cache/pip` 约744 MiB；`~/.cache/uv` 约14 GiB，但archive内13.35 GiB文件有多个硬链接，
仅约0.08 GiB为单链接，因此不能把14 GiB都算成清缓存后可释放空间。
旧 `/tmp/aubo-joint-prep-20260922-v1` 约1.2 GiB，主要为 `smoke/` 测试产物；
`~/trash` 约19 GiB，主体是另一个羽毛球项目的 `20260928_matrix30`（约18 GiB），不是本次竹条训练产物。
用户本轮仅授权并已删除 `/home/rentao/.cache/pip` 和 `/tmp/aubo-joint-prep-20260922-v1/smoke`，
逐项验证路径不存在；清理后可用13453340672字节，实际释放1977417728字节（约1.84 GiB）。
原始动作数据、已验证SmolVLA模型、复用虚拟环境、uv缓存和trash均未删除。

**远端运行。**独立目录为 `/home/rentao/program/lerobot-aubo-strip-segmentation-20261005`。
首版部署包 `artifacts/short_strip_training/transfer.tar.gz` 已上传解包；上述加载检查修正随后单独部署，
因此首版tar包不代表最新入口，恢复时应使用当前源代码。远端保留旧入口 `train_strip_segmentation.before_loading_fix.py`
及失败日志 `logs/startup_loading_failure.log`，没有覆盖既有训练结果。
官方基座为 `facebook/mask2former-swin-tiny-coco-instance`，固定revision
`22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6`，权重与配置保存在 `models/mask2former-swin-tiny-coco-instance/`，
来源记录为同目录 `download_source.json`。已核对无missing/unexpected/error，仅三个预期类别字段重建。
后台tmux会话为 `aubo-strip-segmentation-20261005`，启动脚本为远端根目录 `launch_train.sh`；
本机脚本副本位于 `artifacts/short_strip_training/`。脚本的环境变量仅作用于该训练进程，不改变系统设置。
输出为 `outputs/short_strip_mask2former_baseline/`，日志为 `logs/train.log`；
完成须检查 `complete.json` 以及重载best后的 `best_validation.json`，不能只凭进程退出判断成功。

**首轮结果。**`complete.json` 确认30轮、1800步、用时291.16秒，best为第15轮且已重载评价。
best在40张验证图上，两类各TP=30、FP=0、FN=0，两类F1及macro F1均为1.0；10张空场景误检0张。
这是置信度0.5、可见掩码IoU≥0.5下的实例匹配结果，不代表像素轮廓完全准确，也不是独立测试或实机抓取成功率。
训练进程正常结束、GPU无计算任务；训练后根分区可用12557283328字节（约11.69 GiB）。
报告及40张预测实例PNG已取回本机 `artifacts/short_strip_training/result/`，模型best/final保存在远端输出目录。
后续先查看预测轮廓，再用新摆放场景做独立验证；尚未接入机械臂执行。

```bash
# 查看最新训练进度；不重复启动任务。
ssh gpu 'tail -n 20 /home/rentao/program/lerobot-aubo-strip-segmentation-20261005/logs/train.log'
```

### 验证集预测轮廓复核与独立测试安排（2026-10-05）

**结论：可以进入新摆放的独立感知测试，但轮廓并不完全准确，尚不能据此宣称可直接抓取。**
本次只读取本机保存结果，没有重新推理、训练、导出数据或操作硬件，也没有取回远端权重。
核对 `complete.json` 确认best来自第15轮且已重载；验证集仍为011、016、021、026、031、036、041、044、045、046，
共40张图、60个实例，与训练组无交叉。
40张导出RGB与原始采集、人工review副本逐字节一致；60个真值实例与人工review可见掩码逐像素一致，
类别与已确认的上层集合一致。40张预测PNG均为640×480、16位，像素ID与 `best_validation.json` 的 `segments`
逐一对应；重算IoU≥0.5时每帧TP/FP/FN与原报告一致。没有通过重复哈希或重新导出来完成核对。

**固定复核入口：**[prediction_review.html](../../artifacts/short_strip_training/result/prediction_review.html)。
支持按组／阶段查看原始RGB、人工真值、预测、轮廓误差；可切换全图／料堆放大、填充／描边，或跳到最低IoU样本。
图片内嵌，不依赖网络或本地服务器。编号和颜色按匹配实例对齐，预测原始ID另列于表格；这不是跨帧追踪。
类别从JSON读取，T=`top_strip`，C=`covered_strip`；青色为漏分，洋红为多分或归错实例。
原始人工复核入口 `artifacts/placement_sequences/review/index.html` 及其内容未改动。

```bash
google-chrome /home/rentao/program/lerobot-aubo/artifacts/short_strip_training/result/prediction_review.html
```

统计明细：[contour_review.json](../../artifacts/short_strip_training/result/contour_review.json)。
可复现入口为 `examples/phone_to_auboi10/review_strip_predictions.py`，使用已有NumPy/OpenCV与临时SciPy依赖，
不加载模型或设备。下面是已执行命令；输出存在时会拒绝覆盖，日常复核直接打开上述页面即可。

```bash
PYTHONPATH=/tmp/aubo-segmentation-deps:src .venv/bin/python \
  examples/phone_to_auboi10/review_strip_predictions.py
```

**更严格的轮廓统计。**IoU（交并比）是预测和真值的重叠面积除以合并面积；以下均沿用已保存的置信度≥0.5预测，
不更改后处理。匹配为同类别一对一，优先最大化达到门槛的匹配数，再比较IoU；不是COCO mAP。

| 类别 | 实例数 | 平均IoU | 中位IoU | 最低IoU | IoU≥0.75的TP/FP/FN | F1@0.75 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| top_strip | 30 | 0.8747 | 0.8835 | 0.7548 | 30/0/0 | 1.0000 |
| covered_strip | 30 | 0.8508 | 0.8717 | 0.6843 | 29/1/1 | 0.9667 |

60个实例平均IoU为0.8628；不按类别限制、只按轮廓建立对应后，60对的类别均一致。
IoU≥0.9时两类均只有6/30实例匹配成功（各TP=6、FP=24、FN=24、F1=0.2），
说明0.5门槛下F1=1.00不能解释成像素级轮廓完美。FP/FN在这里包括轮廓重合不够的已有实例，不全是额外物体或整根漏检。
10张空场景预测均全零，没有零碎误检；非空图仍存在小碎片。

**图像复核发现。**已查看10组全部40张阶段对照图，重点对照以下样本；数值相对于现有人工可见掩码，
不把多边形栅格化真值视为没有边缘不确定性的物理测量。

| 图像／实例 | 具体证据与判断 |
| --- | --- |
| `pile_031_frame_003 / strip_002` | 最低IoU=0.6843；漏865像素、多43像素。右侧可见段只保留51.7%，主要为沿长边漏分，另一段保留88.9%。两段仍属于预测ID=1，但不代表轮廓完整。 |
| `pile_031_frame_003 / strip_001` | IoU=0.7682；对应预测ID=2混入262个属于`strip_002`的真值像素，右侧出现额外局部片段。说明存在邻条区域归属混淆，不能笼统说“没有粘连”。两根仍输出为不同实例。 |
| `pile_021_frame_003 / strip_003` | 上层最低IoU=0.7548；多721像素、漏15像素，主要是长边外扩及端部外延，类别正确。高置信度0.9789不代表边缘精度高。 |
| `pile_011_frame_002 / strip_001` | IoU=0.7723；交叉附近及长边漏分，两段召回率72.8%和87.6%。最终三根图中的`strip_002`也有长边漏分，IoU=0.7790。 |
| `pile_044_frame_002 / strip_001` | 端部及长边有局部漏分；最终帧与其他组仍可见边缘偏移。没有发现整根消失，但不能据此说端部无误差。 |
| `pile_041_frame_003 / strip_001` | 被遮区域中有54个像素误归给下层，其中19个位于遮挡区域内部（离该区域边缘>2像素）；未见整段穿过上层的补全，但存在局部越界。 |

30个被压实例的主要可见分段仍能归到各自同一预测ID；所有面积≥80像素的真值连通块均有对应预测覆盖，
其中最低召回率51.7%。这里80只用于区分主要分段与栅格碎片，没有过滤或修改任何预测。
若把所有1～15像素小块也计入，则存在完全漏掉的小块，不能声称所有连通块完整保留。
未见把被遮部分整根补全的明显案例；30个被压实例的遮挡区域共8722像素，其中518像素误占，
排除遮挡区边缘≤2像素的部分后仍有60像素误占。此处遮挡区来自已确认整根投影减去可见真值，仅用于误差分析，
不是新增训练目标或不可见形状的模型输出。

**独立测试：本次只安排 `pile_047～051` 五组，全部 `--split test`。**
已核对原始组号最高为046，047～051在采集目录、统一review和下载JSON中均未占用。
五条命令已用省略 `--record` 的计划模式检查，参数有效且没有创建采集目录、打开相机或连接机械臂。
这是第一批独立测试，共5个独立摆放、20张图、预期30个可见实例；组内四帧相关，不能称为20个独立场景。
这么小的一批用于初步检查新摆放表现，不足以估计广泛无序堆叠的可靠成功率。

1. 保持400×25×8 mm、旧全局相机、支撑物、机位和通常光照。每组彻底清空并独立重摆，
   在普通“一根压一根”范围内自然变化位置、角度和交叉点；不照着031或021的失败图复刻测试，不引入特殊结构。
2. 每组拍空场景及逐根从上方加入1、2、3根。旧条不移动或明显倾斜，木条完整入镜并留出端部余量；
   本轮沿用ROI和差分参数，它们只影响候选标注，模型仍接收完整640×480原始RGB。
3. 拍完五组集中检查原图，确认摆放条件；人工标注沿用统一review目录。先确认真值再显示预测，
   避免按模型输出修改答案；下载JSON保留原始证据。本批不加入当前训练数据或更改30/10划分。
4. 固定当前第15轮 `best/`、处理器及置信度0.5、mask阈值0.5和现有后处理；报告IoU≥0.5／0.75的两类TP/FP/FN、
   F1、逐实例IoU、空图误检，并列出每组及最差轮廓，继续检查断段身份、邻条混入、交叉越界和端部漏分。
   不依据测试结果调整阈值或挑模型后仍把同一批称为独立测试；若用于开发，后续需另采未参与调整的测试组。
5. 模型权重仍在GPU机，本机只有报告与掩码。当前训练入口只有 `preflight/train`，尚无独立测试命令；
   新图复核完成后应增加只加载冻结best的离线评价入口，不能重跑 `--stage train` 来测试。
   模型位置及复用环境见上节；相机采集仍由用户明确授权或自行运行下列命令，本轮未执行。

在仓库根目录，操作者每次运行一条，完成051后集中检查；`--record` 会打开旧全局相机：

```bash
PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/capture_placement_sequence.py \
  --camera-set original-global --output artifacts/placement_sequences/pile_047 \
  --placement-id pile_047 --split test --strips 3 --roi 280 200 640 480 --min-chroma-change 8 --record

PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/capture_placement_sequence.py \
  --camera-set original-global --output artifacts/placement_sequences/pile_048 \
  --placement-id pile_048 --split test --strips 3 --roi 280 200 640 480 --min-chroma-change 8 --record

PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/capture_placement_sequence.py \
  --camera-set original-global --output artifacts/placement_sequences/pile_049 \
  --placement-id pile_049 --split test --strips 3 --roi 280 200 640 480 --min-chroma-change 8 --record

PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/capture_placement_sequence.py \
  --camera-set original-global --output artifacts/placement_sequences/pile_050 \
  --placement-id pile_050 --split test --strips 3 --roi 280 200 640 480 --min-chroma-change 8 --record

PYTHONPATH=src .venv/bin/python examples/phone_to_auboi10/capture_placement_sequence.py \
  --camera-set original-global --output artifacts/placement_sequences/pile_051 \
  --placement-id pile_051 --split test --strips 3 --roi 280 200 640 480 --min-chroma-change 8 --record
```

软件验证：40图来源／掩码映射检查及原0.5指标复算通过；临时独立Chrome中验证40图切换、10组选择、
全图／放大、填充／轮廓、前后翻页和最低分跳转。检查截图和修改前说明备份保存在
`/tmp/aubo-prediction-inspection/`，不是另一份review目录；持久入口只维护上面的 `prediction_review.html`。
本次仅新增离线复核脚本和两份派生结果，更新本说明；原始人工review、训练产物及工作区其他研究改动均保留，未提交或推送。

### 独立测试 pile_047～051 拍摄检查与轮廓草稿（2026-10-05）

五组均已完成空场景＋逐根新增3根，共20张640×480原始RGB，`finished=true`、`stop_reason=completed`，
全部保留采集时指定的 `split=test`。已查看20张原图及15张新增木条的放大图，未发现木条出画、缺帧或明显需重拍的问题。
本批仍使用400×25×8 mm短条、旧全局机位和普通逐层堆叠；不能仅凭照片证明全部摆放与历史场景完全不同。
差分连通块047依次为1、1、2，其余四组为1、2、2；同色交叉与阴影仍会影响候选，不能代替人工真值。
曝光／白平衡未获驱动锁定确认，控制项未成功写入，无恢复失败；此情况与前批一致。

用户在本轮对话明确回复“五组均满足”，确认047～051均从上方逐根加入，旧条没有移动或明显倾斜。
已将确认来源和已知尺寸写入这五组新标注；没有回写原始采集或旧标注。
在原有统一目录新增 `review/pile_047～051/`，15根草稿均保持 `reviewed=false`、`polygon_draft`，
`training_ready=false`、`graspable=null`。草稿以原图目视绘制调整，差分候选外接框仅作初始位置参考，
未加载或查看待测分割模型对本批的预测。按已确认摆放顺序生成的可见区域及上层候选仍需轮廓复核，不能先当测试答案。

**人工复核已完成并导入（2026-10-05，本节最新状态）。**已收到 `pile-json--/pile_047_polygons.json`～
`pile_051_polygons.json` 五份下载文件，15根全部 `reviewed=true`、未截断，尺寸及摆放确认记录一致。
五组共8根修改了顶点：047第2根，048第1、3根，049第3根，050全部3根，051第2根。
按下载JSON原样更新同一 `review/pile_047～051/`，没有额外修改顶点，未另建review版本。
入口仍为 [review/index.html](../../artifacts/placement_sequences/review/index.html)，五组均显示已确认3/3。

最终核对：20张原图副本与源文件一致，导入多边形与下载JSON一致；15个非空阶段均为 `polygons_reviewed`，
`full_frame_mask_eligible=true`，30份逐帧可见实例掩码均非空、位于相应整根投影内、同帧互不重叠。
已查看全部15张阶段实例叠加图，未发现明显的轮廓归错条或截断；遮挡区域按已确认放入顺序扣除。
上层候选分别为该阶段新增实例，数据保持5组 `test`、5张空场景＋15张非空图；未混入训练或验证集。
`training_ready=false`、`graspable=null` 保留，不把轮廓真值确认当作物理可夹性验证。
原始采集、下载JSON和历史训练产物未改动；更新前五组结果、说明和索引备份位于 `/tmp/aubo-test-review-import-1dy8bp97/`。

**已完成冻结best的独立评价。**用户随后授权“进行下一步”，本批结果及失败样本见下面的独立测试结果小节。
仍使用第15轮best与既定阈值，没有重新训练或修改原30/10划分。

```bash
google-chrome /home/rentao/program/lerobot-aubo/artifacts/placement_sequences/review/index.html
```

检查通过：20张review原图副本与源文件一致，15个阶段的草稿、test划分及上层候选符合当前记录，
30份逐帧可见实例掩码均非空、同帧互不重叠；已查看全部15张阶段实例叠加图，统一索引96个链接有效。
复用已有复核工具与页面模板，未修改代码或重复运行无关测试。
修改前说明、索引及临时观察图保存在 `/tmp/aubo-review-test-047-051-o3f_65ej/`，不是另一份review入口。
本次未启动相机、连接机械臂、运行模型推理、重新导出或训练；旧成果及工作区其他改动保持原样，未提交或推送。

### 首批独立测试结果：pile_047～051（2026-10-05，当前结果）

**结论：新普通摆放上的轮廓识别总体可用，但发现明确的被压条误判为上层，尚不能把 `top_strip` 输出直接视为可靠可抓目标。**
本批为模型固定之后新拍摄、人工真值先于预测确认的5组test，未参与训练、模型选择或阈值调整。
5组包含5张空场景、15张非空阶段图、30个可见实例；组内阶段图相关，不能当作20个独立摆放。
当前结论只适用于这批固定机位、40cm短条、普通逐层堆叠，既不是特殊结构泛化证明，也不是实机抓取成功率。

**结果位置。**本机只维护一个独立测试复核入口：
[prediction_review.html](../../artifacts/short_strip_training/independent_test/result/prediction_review.html)。
点击“类别／数量异常”可直接跳转 `pile_051_frame_002`；支持原图／真值／预测／误差、按组切换、放大和额外实例详情。
原验证集复核页和人工标注review均保留，不用测试结果覆盖验证结果。
逐图报告：[test_evaluation.json](../../artifacts/short_strip_training/independent_test/result/test_evaluation.json)；
严格轮廓统计：[contour_review.json](../../artifacts/short_strip_training/independent_test/result/contour_review.json)。
同目录含 `complete.json`、`plan.json`、`loading_info.json`、`data_audit.json`、20张 `test_masks/*.png`。

```bash
google-chrome /home/rentao/program/lerobot-aubo/artifacts/short_strip_training/independent_test/result/prediction_review.html
```

**固定评价口径。**第15轮best，640×480完整RGB，处理器不resize；batch=2，置信度0.5、mask阈值0.5，
沿用同一Transformers 4.57.1的实例后处理（默认 `overlap_mask_area_threshold=0.8`）。
IoU（交并比）匹配为同类别一对一，不采用COCO mAP；本次没有扫阈值、过滤碎片或按测试结果选择模型。

| 类别 | IoU≥0.5的TP/FP/FN | precision | recall | F1 | IoU≥0.75的TP/FP/FN、F1 |
| --- | --- | ---: | ---: | ---: | --- |
| top_strip | 15/1/0 | 0.9375 | 1.0000 | 0.9677 | 15/1/0，0.9677 |
| covered_strip | 14/1/1 | 0.9333 | 0.9333 | 0.9333 | 14/1/1，0.9333 |

两类平均F1为0.9505；5张空场景均无误检。按几何轮廓建立对应，30个真值实例均有IoU≥0.75的预测对应，
其中29个类别正确、1个类别错误，另有1个额外预测实例。
几何平均IoU为0.8599（top真值0.8622、covered真值0.8576）；这个均值**包含类别判错的轮廓配对，且不包含额外预测**，
必须同时看上面的类别指标。最低IoU分别为0.7965、0.7963。IoU≥0.9时，上层TP=2/FP=14/FN=13，
被压TP=1/FP=14/FN=14，仍存在明显边缘偏差。

| 组号 | 非空图数量与类别均匹配（IoU≥0.75） | 几何平均／最低IoU |
| --- | --- | --- |
| pile_047 | 3/3 | 0.9024／0.8853 |
| pile_048 | 3/3 | 0.8615／0.8036 |
| pile_049 | 3/3 | 0.8380／0.7965 |
| pile_050 | 3/3 | 0.8502／0.7976 |
| pile_051 | 2/3 | 0.8473／0.7963 |

**具体图像证据。**已查看全部20张阶段对照图及异常图：

- `pile_051_frame_002`：斜放的 `strip_001` 真值为 `covered_strip`，预测ID=2却是 `top_strip`，
  置信度0.645844、几何IoU=0.8339；两段可见轮廓仍归为同一实例，主要错误是上层类别判断。
  真正上层 `strip_002` 正确，预测ID=3、置信度0.984083、IoU=0.8976。因此这一帧输出了两个top，不能认为“找到真正上层”就没有误选风险。
  同帧额外预测ID=1标为 `covered_strip`、置信度0.958676，但只占88像素、分成30块（每块1～10像素）；
  与被压真值的IoU只有0.0221，其中54像素落在该真值内、34像素在背景。类别正确的小碎片不能抵消整条类别错误。
- `pile_051_frame_003 / strip_001`：类别恢复正确，但该帧几何IoU最低0.7963，漏277像素、多204像素；
  6像素真值小块完全漏掉。后一帧正确不能抵消上一帧错误，也不证明持续跟踪能力。
- `pile_049_frame_002 / strip_002`：上层最低IoU=0.7965，漏368、多297像素，主要是长边偏移及局部端部误差。
- `pile_050_frame_003 / strip_003`：上层IoU=0.7976，漏408、多193像素，沿长边有明显漏分。
- `pile_048_frame_003 / strip_003`：上层IoU=0.8036，多473、漏122像素，存在轮廓外扩。

本批被压实例的主要可见分段保持同一预测ID，面积≥80像素的真值分段最低覆盖率83.0%；
80仅为描述主要分段的统计界限，没有用于过滤预测。未见明显整根穿越上层的补全，
遮挡区4093像素中有177像素误归给下层，排除距遮挡边缘≤2像素的部分后仍有7像素误占。
邻条边缘仍有少量像素归属错误；不能用平均IoU掩盖上述类别错误、碎片和端部偏差。

**实现、执行与保留范围。**新增 `evaluate_strip_segmentation.py`，只加载本地冻结best，禁用梯度，
复用原 `StripDataset`、处理器适配和 `evaluate`，没有优化器或权重写入。加载记录的missing/unexpected/mismatched/error全部为空。
数据导出／审计增加显式 `test_only`，默认训练检查仍要求train＋validation；
`configs/aubo_i10/short_strip_segmentation_test.json` 仅选择047～051，首次生成到 `datasets/short_strip_segmentation_test/`。
没有重复导出旧160张图，未修改 `datasets/short_strip_segmentation/`。新20图导出、test独占、与原40组隔离检查通过。
复核脚本增加 `--split test`，支持类别错误及额外实例详情；默认validation入口兼容，旧验证页面未重写。

远端独立运行目录：`/home/rentao/program/lerobot-aubo-strip-segmentation-20261005/independent_test/`。
上传输入包保存在本机同名工作目录的 `transfer.tar.gz`；远端新目录解包并复制原实验的三个包初始化文件
（`lerobot/__init__.py`、`lerobot/__version__.py`、`bamboo_sorting/__init__.py`）以使用隔离源码，不覆盖原实验入口。
结果在远端 `independent_test/result/`，日志 `independent_test/evaluate.log`，均已取回本机。
本次复用已有Python和独立SciPy，无安装或环境配置修改。`complete.json` 确认20图完成、`optimizer_steps=0`、组交叉为空；
记录的约1.12秒仅为模型加载后的评价耗时，不含加载、传输、人工复核或机械臂执行。

以下为已执行命令，结果目录已存在，不应重复运行或覆盖：

```bash
# 在GPU独立实验的 independent_test/ 内执行；变量仅影响此进程。
PYTHONPATH=../deps:src HF_HUB_OFFLINE=1 \
  /home/rentao/program/lerobot-aubo-smolvla-c0-pilot-b3a9c8a/.venv/bin/python \
  examples/phone_to_auboi10/evaluate_strip_segmentation.py \
  --dataset datasets/short_strip_segmentation_test \
  --training-manifest ../datasets/short_strip_segmentation/dataset.json \
  --model-path ../outputs/short_strip_mask2former_baseline/best \
  --output result --device cuda
```

软件检查：原10项分割测试通过；新增合成测试验证test-only导出、误用训练入口被拒绝、组交叉拒绝、
冻结模型无优化／反向传播且文件不变、空／非空评价、保存与test复核页生成，修正测试夹具路径后通过。
页面增强后重跑该相关测试通过；20图文件对应及0.5指标复算通过，临时Chrome验证20图、5组切换、
全图／放大、轮廓／填充、最低分和类别／数量异常跳转、88像素额外预测说明。`git diff --check`通过。
修改前源码及说明备份在 `/tmp/aubo-independent-test-before-st77dse3/`，图像检查与浏览器截图在
`/tmp/aubo-independent-test-inspection-g8eh3_qm/`；持久结果只有上述固定入口。未启动相机或机械臂，未重新训练、提交或推送。

**后续建议。**先复核051第二阶段这一类别错误，继续在普通堆叠范围补充独立的开发数据、变化方向和交叉位置；
当前不进入特殊结构或直接自动逐层抓取。不要因一次测试失败临时提高置信度或过滤小块后宣称本批问题已解决。
本批原始结果保留为首轮独立基线；若依据这些失败选择数据、调参或改模型，新的泛化成绩应使用另批未参与开发的测试组。
补采、重新训练及硬件执行仍需各自明确授权，本轮没有预分配或启动下一批采集。

### 后续标注、训练准备与目标执行小实验

2026-10-05 交接补充：以下保留早期路线讨论及后续执行候选，数量安排不是当前采集指令；
当前40组、训练完成状态、轮廓复核与下一批五组独立测试均以上节为准。
研究文档中的旧 Mask R-CNN、只标上层和第三张 `goal_image` 设计不作为当前决定。

**早期基线（历史快照）。**当时的`pile_002～004` 共3个独立摆放组、9张非空阶段图，
其中7张通过当前完整轮廓检查；三组均为 `train`，尚无独立验证／测试组。
本次仅核对 `pile_004` 最终记录和工具实现，不重做这些组的人工复核。
`pile_001` 保留为开发资料，不计入最终已复核组。`full_frame_mask_eligible` 只表示通过当前
轮廓确认、摆放确认和非截断条件；所有结果仍为 `training_ready=false`、`graspable=null`。
多边形初稿由助手观察照片后编写，尚无自动轮廓拟合或 SAM 修边。

**当前阶段：先采普通逐层叠放。**2026-10-05 用户明确决定：先采“一根压一根”的普通情况，
证实可行后再另行采特殊情况，替代此前下一组优先拍多个独立上层的建议。
目前改用3根同长度短条，具体批次和长度登记见上节：第1根放在承托面，第2根从上方压住第1根，第3根从上方压住第2根，
形成明确的逐层关系、最终只有一个未被压住的上层。不额外要求第3根与第1根完全不相交。
在这种普通结构内变化整体位置、角度和交叉点，避免每组原样重复；不要求精确复现指定角度。
暂不安排多个独立上层、分叉堆叠、特意构造的近平行／端部轻微交叠或拥挤下指等特殊情况。
短条先积累约10个已复核开发组，再检查标注流程和普通场景下的识别、单次抓放表现；
静态标注完成不等于抓取可行，普通场景通过也不能外推为特殊堆叠或未知无序堆叠已验证。

`pile_005` 已完成拍摄，检查结果和下一批安排见上节。此前承托面积调整时，用户确认支撑高度未变；
实际布局以各组空场景为准，保留旧组，新组从当前布局的空场景重新拍起，不沿用旧布局的空图做差分。
组内承托面保持不变；从上方新增、旧条未移动或明显倾斜的条件按组记录，可在每五组检查时集中确认。

以下保留为普通阶段可行后的扩展候选，不是本批采集任务，也不预分配编号：

| 新组主题 | 摆放要点与目的 |
| --- | --- |
| 多个独立上层 | 先放底条，再放两根分别压底条、彼此不相交的上条；最终应有两个上层候选，避免模型只学“最后一根”。 |
| 一根压两根 | 两根底条互不压住，第三根同时跨过两根；与上一种结构形成对照。 |
| 链式交叉 | A压B、B压C，A与C不交叉；用来检查露出较多的中层仍不能选。 |
| 交换上下之一 | 选择一组便于复现的位置和角度，保存一种上下次序。 |
| 交换上下之二 | 尽量保持前组位置和角度，反转关键交叉的上下次序；两组作为关联对照，保持同一 split。 |
| 近平行或端部交叉 | 覆盖细窄遮挡和“下层大部分露出”的困难情况。 |
| 位置、方向和邻条间距变化 | 在完整入镜范围内改变分布，兼顾上层两侧宽松与拥挤；间距只作场景变化，不先判定可下指。 |

保持本批选定的同一种短条长度；先检查原图完整入镜、两端有余量，再拍空场景及逐根新增阶段。
范围变化时检查 ROI；`280 200 640 480` 和色度阈值8只是开发设置。
单根、双根及空场景也需保留，因为逐层移除会经过这些状态。
新照片仍按“检查原图与差分 → 确认本组摆放条件 → 写多边形草稿 → 用户逐根确认 → 更新统一review内本组结果”处理。

**数量与划分。**先用约10个开发组核对规则和耗时；累计到30组训练＋10组验证时，
可做首版训练评估。首轮完整预算建议为80组训练＋20组验证＋50组独立保留测试，
之后按失败类型补采，数量本身不保证效果。验证组在采集前指定 `--split validation`，
不能把后续所有组都默认设为训练；保留测试不用于调整参数或挑模型。
同一物理摆放的阶段图、移除序列、增强图及共享合成来源不能跨集合；同组多帧不是多个独立样本。

**感知职责与导出原则（首轮训练已完成）。**当前实现与检查见上节。感知模型负责“看见哪根、是否被压”，
VLA负责从图像和状态预测动作；静态照片不能替代动作示教。
最新候选是评估预训练 Mask2Former 的微调，输出每根的可见实例掩码及
`top_strip`（未被压住）／`covered_strip`（被其他条压住），保留已获得的下层监督信息。
首轮已使用Swin-tiny COCO实例分割预训练基座完成微调；SAM 3仅为后续修边助手或对照候选，尚未接入。

导出准备使用最终复核目录中的原始RGB副本、`visible_mask`、`top_ids_candidate`、
`placement_id` 和 `split`。只在轮廓与摆放条件成立时按候选集合派生两类标签，
不将 `footprint_in_image` 当成当前可见区域，不把完全不可见实例当作可见目标。
首版完整轮廓数据排除截断阶段，保留原始文件和排除原因；空场景显式表示无实例，
不能与缺失标签混为一谈。按独立摆放检查集合交叉，并核对图像尺寸、掩码尺寸和实例引用。
原始RGB是模型输入，轮廓与层序是监督答案；复核描边、放入编号和顺序提示不能泄漏给感知模型。
这些是导出要求，不修改现有 `training_ready`，也不把历史静态层序评价器直接用作新分割评价器。

**下指可用段（尺寸待定）。**分开记录上层状态、允许抓取中心落入的区段、实际抓取结果。
暂按“从上方接近，两指位于竹条两侧并横跨宽度夹紧”讨论；一根可以有多个可用区段。
位置标签建议为可下指／不可下指／不确定，原因包括邻条、支撑物或间隙不足。
未标位置保持未知；下层不能选，不等于其所有位置都应标成碰撞负例。
判断需覆盖张开宽度、手指下降和闭合经过的空间、下探深度、桌面／中央支撑块及夹爪本体。
橡胶垫张开和闭合净间距、手指下探部分宽厚、计划夹持深度仍待用户提供；当前不设置距离阈值，
不实现依赖这些尺寸的页面判定。斜视RGB不能证明三维无碰撞，也不能全图共用固定像素／毫米比例。
不确定处以后可考虑腕部 Mech-Eye NANO 的局部深度核验，当前未接入此流程。
这些标签用于目标筛选，不给VLA添加固定高度、强制姿态或脚本夹取轨迹。

**尽早验证目标条件化。**目标条件化指“场景相近但指定目标不同，模型实际取走的对象随之改变”。
普通阶段先检验识别正确上层与单次抓放是否可行，不等全部静态数据收齐才检查执行端。
按本次阶段范围，暂不采下面的双可取目标A/B示教；设计保留到普通阶段可行后再安排。
只有一个可选上层的成功结果不能证明模型具备多个可取对象间的目标服从性。
以下接口、示教、微调和实机验证均待实施：

1. 先选两根都在上层且经现场确认可取的场景，以人工正确目标隔离感知错误。
   同类摆放分别指定A、B并采集对应单次抓放示教，覆盖位置／角度交换；关联示教保持同一数据集合。
   现有160条示教没有自动获得该条件，不能把抓A的动作重标成目标B。
2. 优先在实时全局图副本中描出目标，保留腕部RGB和七维动作；原始图不覆盖。
   描边样式、执行中目标身份跟踪、遮挡与位移时的更新／失效处理，要先在离线录像上明确并验证，
   再接训练和运行接口；不能把段首轮廓固定画在后续已移动的目标位置上。第三张 `goal_image` 仍非当前决定。
3. 目标接口和微调具备后，在独立摆放上成对比较指定A／B，并保留旧模型自选目标的对照。
   逐次记录指定对象、实际取走对象、是否恰好一根、是否送达、邻条变化和人工接管。
   `--cycles 1` 只负责命令周期结束；离线动作变化、周期完成或DO读回都不能证明服从目标。
4. 若实际选择总跟随固定位置／角度，先处理条件化数据或接口，再扩大静态标注。
   即使整根目标服从性通过，仍需另验是否服从指定下指区段；目标描边本身不保证抓在可用段内。

## 现场操作

1. 填写场景编号和物理摆放参考。同一摆放保持同一编号，不能跨训练、验证或测试。
2. 按 `r` 归位，摆好竹条，再按右箭头开始录制。
3. 手机完成接近、夹紧、提起、搬运及释放；必须在录制结束前释放。
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

审计核对七维契约、控制器指令、时间戳、保存一致性及夹爪标签。
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

# AUBO i10 无序竹条分拣（离线阶段）

这里是独立于 `phone_to_auboi10` 的几何分拣入口。当前实现只处理保存的 NPZ 场景，不导入
Mech-Eye SDK、不连接 AUBO，也不发送夹爪或运动命令。

## 为什么单独建立接口

LeRobot 的 RGB `Camera` 接口一次主要返回一幅图像，而分拣规划需要同一时刻的 RGB、米制深度、
点云、内参、时间戳和坐标变换。为避免把不兼容的数据硬塞入图像接口，本模块使用
`SceneSensor.capture() -> SceneFrame`。将来 Mech-Eye 适配器实现这个接口；现阶段使用只读
`NpzSceneSensor`。

离线处理链为：

```text
SceneFrame
  -> 工作区/桌面过滤与质量报告
  -> 可见直线段提取（不补全遮挡部分）
  -> 交汇区域估计
  -> 硬约束拒绝
  -> 分项评分
  -> geometry_valid
  -> 外部 MoveIt/运动学与碰撞验证
  -> execution_validated
```

最后一步尚未接入。没有外部验证器时，候选永远不会标成 `execution_validated`，确定性状态机也
不会执行它。

全局三维规划由 Mech-Eye 场景承担；腕部 RGB 相机将来只做抓取前的有界小范围对准和抓后验证。
任何视觉修正后的位姿仍须重新经过同一运动学、碰撞和安全边界检查。真机执行继续使用现有
`pyaubo_sdk` 链，ROS 2/MoveIt 仅作为模型、TF、可达性、碰撞、RViz 和仿真辅助层。

## NPZ 数据约定

必需数组：

| 键 | 形状/类型 | 含义 |
|---|---|---|
| `points_xyz` | `(N, 3)` float | `frame_id` 下的米制点云 |
| `timestamp_s` | 标量 float | 采集时间戳 |
| `frame_id` | 标量字符串 | 点云坐标系 |

可选数组为 `color`、`depth_m`、`points_rgb`、`intrinsics=[width,height,fx,fy,cx,cy]` 和
`T_base_camera`（4×4 齐次变换）。当 `frame_id != "base"` 且没有 `T_base_camera` 时，结果会被
标记 `missing_base_calibration`，不得执行。

## 配置与离线运行

配置 JSON 顶层必须包含 `preprocess`、`segments`、`planner`。其中工作区、桌面高度、竹条长度、
竹条宽度、夹爪允许宽度和预抓距离都必须来自现场测量或明确标记的合成场景，仓库不提供猜测值。
各字段定义见：

- `PreprocessConfig`：工作区上下界、桌面高度、体素、半径离群点过滤和点云质量门；
- `SegmentExtractionConfig`：RANSAC 距离阈值、最少点数、可见长度和交汇阈值；
- `GraspPlannerConfig`：可夹持长度/宽度、完整度、净空、交汇排除半径和 Pre-Grasp 距离。

只读运行命令：

```bash
.venv/bin/python examples/aubo_bamboo_sorting/plan_replay.py scene.npz measured_config.json
```

输出 JSON 为每个候选保存 `score_components` 和 `rejection_reasons`。其中的 4×4 矩阵是任务坐标
框架，不是经过 AUBO TCP 标定、逆解和碰撞检查的可执行轨迹。

## 后续硬件阶段门

接入真实设备前仍需确认：Mech-Eye 型号/SDK/IP、点云单位和坐标系、相机外参、竹条尺寸、夹爪
行程、TCP、工具负载、气压以及是否有可用夹爪反馈。任何一个值都不能由代码猜测。

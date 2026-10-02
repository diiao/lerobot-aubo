# 当前现场运行：归位与持续抓放

当前入口为 `run_joint_smooth.py --model mixed-both-orders --approach-age-aligned`。
运行前核对模型部署、CameraSetV2 和 GPU 占用；全部实机命令由现场操作者手动执行。
当前代码不检测整堆清空，不支持独立周期数参数；混训模型在第二次吸附→释放完成后结束。

## 1. 查看计划

在仓库根目录运行，不连接相机、机械臂或远端模型：

```bash
.venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
  --model mixed-both-orders --approach-age-aligned
```

核对 `requested_suction_cycles=2`、
`pending_suction_policy=continue_inference_while_open` 和模型哈希。
当前模型身份、远端脚本和验收证据见 [证据索引](EVIDENCE.md)。

## 2. 归位

确认未持物、返回路径安全，由现场操作者执行：

```bash
.venv/bin/python examples/phone_to_auboi10/run_joint_trial.py --stage home --onsite-confirmed
```

已读到释放状态时直接归位；否则发送释放指令，核对读回后归位。
无需额外释放参数或第二次 Enter。默认不创建日志目录，显式 `--output <新目录>` 才记录。
该阶段不打开相机、不加载模型。释放或归位失败后不要继续抓取。

## 3. 双根抓放

上90°下45°示例：

```bash
.venv/bin/python examples/phone_to_auboi10/run_joint_smooth.py \
  --model mixed-both-orders --approach-age-aligned --execute \
  --output "artifacts/joint_both_top90_$(date +%Y%m%d_%H%M%S_%N)"
```

上45°下90°保持参数相同，将输出前缀改为 `joint_both_top45_`。
启动后输入完整任务文字 `Pick one strip and place it in the collection area.`，
待模型和设备就绪，按现场提示确认开始。本地入口通过 SSH 启动远端服务，无需另开服务进程。

起点要求吸盘释放，各关节距 `START_DEG` 不超过 1.5°。
若数字输出均低、状态未知，入口保留一次“未持物”确认再释放；
这与独立 home 的无二次确认流程不同。

**单根使用 mixed-both-orders 仍会要求两个周期。** 不要把改用旧单根模型当作同一模型的停止条件修复。
当前没有 `--cycles` 或 `--second-grasp-z-mm` 参数。取走上层与判断整堆结束是不同能力。

## 4. 当前动作与保护

动作块是模型一次预测的未来 50 步完整七维动作。吸盘指令为释放时，按观测年龄选取整行，
遇到首次预测闭合边界时不越过它；等待到位期间继续单个在途观测/推理，新结果可更新或取消闭合。
吸附后沿用首步预测；IO 切换后废弃旧动作块并丢弃切换前观测的在途结果。
不添加接触高度、关节偏置、脚本化下降或提起。

| 当前 mixed-both-orders 限制 | 值 |
| --- | --- |
| 本地命令目标频率 | 25 Hz |
| 关节参考速度 / 加速度 | 5°/s / 20°/s² |
| TCP（工具中心点）参考速度 | 0.05 m/s |
| 伺服跟踪误差 | 3° |
| 预测观测年龄 / 双相机时间差 | 500 ms / 100 ms |
| 采集时帧年龄 / 发送时状态年龄 | 250 ms / 100 ms |
| 无新鲜预测停止 / 控制间隔 | 600 ms / 350 ms |
| 待吸盘切换目标期限 / 总时限 | 1.5 s / 360 s |

关节限位来自控制器，当前与目标 TCP 检查工作区边界。
发送前状态过期时最多刷新一次并重新检查；仍过期则停止。
工作区和限速不是全机械臂碰撞检测，日志中的限速是参考命令约束。

## 5. 停止与证据

Ctrl+C、异常及超时进入停止伺服流程，不自动释放或归位。
模型正常释放达到两个周期后结束。若停止时持物，先由操作者安全处置，再准备下一轮。

每轮保存 `plan.json`、`trace.jsonl`、`commands.jsonl`、`complete.json`、
`final_state.json`、双路 `videos/*.mp4` 及帧时间侧录。
`requested_suction_cycles` / `completed_suction_cycles` 仅为指令计数；
现场另记首抓是否上层、是否只带走一根、放置及停止结果。

## 兼容入口

`legacy-single` 绑定旧单根模型，最多120秒、一个周期；
`mixed-double` 绑定旧单根＋一种双根模型，最多360秒、两个周期。
二者均不是当前160条模型。

`run_joint_trial.py` 的 shadow/single-step/short-loop 和 `run_joint_live.py`
保留作既有诊断，并向持续入口提供共享实现；它们的旧首步时限不等同于本页持续伺服限制。
其历史操作和逐次故障分析统一从 [证据索引](EVIDENCE.md) 回溯。

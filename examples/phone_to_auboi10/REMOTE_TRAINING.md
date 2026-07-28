# 远程训练说明

当前可执行流程统一维护在 [README.md](README.md) 的以下章节：

- “运行机器”
- “同步到 GPU 机”
- “GPU 训练”
- “纯 ACT 推理”

## 当前机器信息

| 机器 | 地址 | 用途 |
|---|---|---|
| 工作站 | `192.168.31.222` / Tailscale `100.93.231.40` | 相机、机械臂、录制、推理客户端 |
| GPU 机 `510` | Tailscale `100.88.143.45` | ACT 训练和推理服务端 |
| AUBO I10 | `192.168.31.200:30004` | 仅由工作站通过局域网控制 |

GPU 机 SSH：

```bash
ssh rentao@100.88.143.45
```

## 安全约束

- GPU 机原仓库包含未提交实验文件和旧模型，不要执行 `git reset --hard`。
- 不要使用 `rsync --delete` 覆盖 `/home/rentao/program/lerobot-aubo`。
- 新代码使用 `/home/rentao/program/lerobot-aubo-pure-act` 隔离工作树。
- 数据集只同步 `bamboo_newview_full`，不要把旧视角数据混入新训练集。
- 每次训练使用新的 `MODEL_PATH`，避免不同实验的 checkpoint 相互覆盖。
- 启动录制、训练和推理命令均由操作者在终端手动执行。

# AUBO i10 竹条抓放

基于 LeRobot 的竹条感知与抓放研究。日常操作从带目录的[项目手册](examples/phone_to_auboi10/README.md)开始。

| 工作 | 说明 |
| --- | --- |
| 操作与研究 | [项目手册](examples/phone_to_auboi10/README.md)：当前状态、静态分割、动作采集／训练／运行、后续计划 |
| 结果回溯 | [模型与实验的证据索引](examples/phone_to_auboi10/EVIDENCE.md) |
| 开发维护 | [项目约定](AGENT.md#文档维护规则)、[历史与框架资料导航](docs/README.md) |

项目说明集中维护在手册与证据索引，不按日期、批次或交接新增说明文件。

`examples/phone_to_auboi10/` 是项目入口，`src/lerobot/bamboo_sorting/` 是竹条任务模块，
`src/lerobot/robots/aubo_i10/` 是机械臂驱动。`tests/` 中的软件检查使用不同输入验证代码；
测试通过数量不是实机抓取次数，也不是抓取成功率。

`datasets/` 存放真实数据，`artifacts/` 存放实验产物。这些目录通常被 Git 忽略，代码提交不能代替数据备份。

本仓库保留所基于的 LeRobot 框架、其他硬件和模型实现，供共同基础组件和后续开发使用。
上游技术参考在 [docs/source](docs/source/index.mdx)，源项目为 [Hugging Face LeRobot](https://github.com/huggingface/lerobot)。
项目遵循 [Apache-2.0 许可证](LICENSE)，源文件版权声明保持原样。

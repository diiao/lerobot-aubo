# 文档导航

## 当前项目

- [当前工作流](../examples/phone_to_auboi10/README.md)：采集、训练、运行三份操作说明的统一入口。
- [证据索引](../examples/phone_to_auboi10/EVIDENCE.md)：模型身份、数据来源、现场反馈和历史回溯。
- [三根以上逐层抓取研究范围](AUBO_I10_TOP_LAYER_VLM_RL_ROUTE_2026-09-26.md)：自然叠放，逐根取走可分离的上层；尚未实现的新方向。
- [清理与目录检查记录](AUBO_CLEANUP_2026-10-02.md)：删除依据、保留理由、测试含义及回滚方法。

## 历史证据

- [ACT 对照结论](ACT_COMPARISON_CLOSURE_2026-09-19.md)。
- [run06 实验技术记录](robot_arm_technical_documentation.md)。
- [2026-09-27 数据清理记录](AUBO_I10_LOCAL_DATA_CLEANUP_2026-09-27.md)。

这些记录用于回溯，不提供当前运行指令。更多已退役原文的提交号和路径见证据索引。

## LeRobot 通用参考

[source](source/index.mdx) 保留本仓库版本对应的框架资料，例如 [SmolVLA](source/smolvla.mdx)、
[数据集](source/lerobot-dataset-v3.mdx)、[处理器](source/introduction_processors.mdx) 和 [相机](source/cameras.mdx)。
这些是通用组件说明；AUBO 的参数、相机角色和命令以当前项目操作说明为准。

仅维护上游文档网站时需要 `docs-requirements.txt` 中的构建依赖及 Node.js。
已有构建环境下可运行 `doc-builder build lerobot docs/source/ --build_dir /tmp/lerobot-docs`；
普通采集、训练和运行不需要构建这个网站。

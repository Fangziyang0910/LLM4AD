# TraceAAD 版本收口

2026-07-10

TraceAAD 的早期实现与已弃用的 prompt-guidance 原型已从平台中移除，包括方法包、测试、示例和对应 TSP Construct 实验 artifact。过程信息融合搜索实现成为唯一的 `TraceAAD`：包路径、公开类、profiler、测试、示例、正式 runner、实验目录和研究文档均统一为 `traceaad` / `TraceAAD`。

当前 TraceAAD 保留三层记忆、三回路、stepwise credit、多维 trajectory value、operator portfolio、islands、novelty gate 和对比反馈。历史运行日志保留其原始输出文本；其目录和 `run_config.json` 已迁移到当前 TraceAAD 命名。

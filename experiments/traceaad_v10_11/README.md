# TraceAAD V10.11

当前提示策略为 `generic_design_v1`：本批次只修订任务上下文中的 fitness 表达、算法标题、设计历史用途说明、Init／Refine／Tune／Pivot／Fuse 指令和 Idea 摘要要求；预算、算子比例、选择与历史检索机制保持不变。V10.10 保持冻结。

LLM 每次看到任务描述和目标函数 stub，返回一个不超过 200 words 的 Idea 以及一个完整候选 Python 程序。解析器在候选模块中寻找模板声明的目标函数；imports、helper、常量和其他支持代码会与固定模板合并，目标函数必须存在且保持声明的接口。解析器另外以模型 tokenizer 判定 Idea 不得超过 500 tokens。Idea 缺失、超长、目标函数缺失或模板重建失败都会进入一次修复。

搜索使用四个等概率算子、质量分布父代选择、Pivot 均匀混合和 Fuse donor 混合。任务上下文使用任务描述和目标函数 stub，不注入 evaluator 的 design_notes。搜索上下文包含当前候选程序、最近形成记录和 donor；固定任务模板只在解析后用于重建和执行。模块独立实现搜索、提示、解析、模板重建、持久化与修复。

默认形成记录只保留 Idea、算子和 fitness。使用 `--history-code` 时，每条最多八步的形成记录额外展示该步的完整候选程序，用于与默认的无历史代码上下文做对比；该开关写入运行配置并参与冻结批次身份。

## 运行监控

只监控 V10.11 批次：

```bash
experiments/traceaad_v10_11/start_monitor.sh 8765
```

打开 `http://127.0.0.1:8765/` 查看当前批次的排队、运行、完成、评价进度、best fitness、算子和最近错误。旧版本不再注册到此监控。

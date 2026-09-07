# TraceAAD V10.7

V10.7 以 V10.6 为基线，只取消独立的第二次实现 Idea 调用。每个候选由一次调用生成简短 Idea 和完整 Code；第一次调用的 Idea 直接写入节点和后续历史。

搜索树、8 根初始化、父代联合分布、R/P/F 请求比例 0.50/0.15/0.35、Fuse donor 与回退、公共任务信息、评价预算、上下文裁剪和恢复规则均沿用 V10.6。协议标识为 `idea_code_single_call`，V10.6 检查点不会作为 V10.7 恢复。

单路运行示例：

```bash
uv run python -m experiments.traceaad_v10_7.run \
  --task tsp_construct \
  --run-name smoke_v107_tsp \
  --budget 10 \
  --backend local
```

生成 15 路正式运行计划并检查可启动项：

```bash
uv run python -m experiments.traceaad_v10_7.launch --dry-run
```

每路仍写入 `run_config.json`、`tree_state.json`、`pending_candidate.json`、`llm_calls.jsonl`、`events.jsonl`、`evaluations.jsonl`、`tokenizer_calls.jsonl` 和 `logs/run_summary.json`。`llm_calls.jsonl` 每个候选只有生成调用；不再记录 `thought_alignment` 阶段。事件不再包含摘要状态、摘要 token、独立摘要调用及分拆耗时字段，`llm_seconds` 表示唯一生成调用耗时。

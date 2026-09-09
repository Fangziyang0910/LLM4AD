# TraceAAD V10.8

[机制设计](../../docs/methods/TraceAAD-V10.8-机制设计.md)。已实现可核验近期形成轨迹、同代码组机会计量与单次 Idea + Code；尚未运行正式机制比较，不代表优于 V10.7R。

从仓库根目录启动单路：

```bash
uv run python -m experiments.traceaad_v10_8.run --task tsp_construct --seed 0 --run-name v108_tsp_rep1
```

默认 8 个有效根记录、1000 次真实评价，Refine/Pivot/Fuse = 0.50/0.15/0.35。上下文 32768、输出预留 16384、安全余量 256，完整输入上限 16128 tokens；历史最多 8 条完整边、8192 tokens。模型与评价器由公共实验基础设施配置。

结果独立保存到 `experiments/traceaad_v10_8/results/<task>/<run>/`。用相同命令恢复，源码、机制、评价配置或后端不一致会拒绝恢复。评价提交状态不明时停止，不自动重评；传输失败保留 pending，恢复不会再次增加组级机会次数。

`tree_state.json` 保存全部有效记录与原始 Idea/代码；`llm_calls.jsonl` 保存完整请求、响应和成本；`evaluations.jsonl` 保存评价收据；`events.jsonl` 保存真实转移、提示视图哈希、历史容量删减、donor 尝试及组级选择概率。`logs/run_summary.json` 的 `best` 按同代码首次有效评分选取，`fitness_instability` 单独列出重复评价分数不一致。

画同口径 best-so-far 时使用事件 `best_so_far`，横轴使用 `evaluation_id`（无评价的解析失败不推进评价轴）。`fitness`、`parent_delta` 等保留本次真实测量，重复高分不抬高 `implementation_fitness` 和 `best_so_far`。

`ablations.build_history_ablation` 提供 `code_only`、`single_edge`、`multi_edge` 三臂，使用相同组级调度和最大输入预算，各自记录实际 tokens。该构造器用于独立实验脚本，生产 CLI 不提供混合历史模式。旧 Idea 链、关系对照及组级调度开关的进一步实验见设计文档；本次未启动这些实验。

本地验收（不调用生成服务）：

```bash
uv run pytest -q tests/method/test_traceaad_v108.py tests/method/test_traceaad_v105.py tests/method/test_traceaad_v106.py tests/method/test_traceaad_v107.py tests/method/test_traceaad_v107_sampling.py
```

# TraceAAD V10.8

[机制设计](../../docs/methods/TraceAAD-V10.8-机制设计.md)。已实现可核验近期形成轨迹、同代码组机会计量与单次 Idea + Code；正式运行状态见[启动记录](launch_20260909.md)，不代表优于 V10.7R。

从仓库根目录启动单路：

```bash
uv run python -m experiments.traceaad_v10_8.run --task tsp_construct --seed 0 --run-name v108_tsp_rep1
```

默认 8 个有效根记录、1000 次真实评价，Refine/Pivot/Fuse = 0.50/0.15/0.35。上下文 32768、输出预留 16384、安全余量 256，完整输入上限 16128 tokens；历史最多 8 条完整边、8192 tokens。模型与评价器由公共实验基础设施配置。

结果独立保存到 `experiments/traceaad_v10_8/results/<task>/<run>/`。用相同命令恢复，源码、机制、评价配置或后端不一致会拒绝恢复。评价提交状态不明时停止，不自动重评；传输失败保留 pending，恢复不会再次增加组级机会次数。

`tree_state.json` 保存全部有效记录与原始 Idea/代码；`llm_calls.jsonl` 保存完整请求、响应和成本；`evaluations.jsonl` 保存评价收据；`events.jsonl` 保存真实转移、提示视图哈希、历史容量删减、donor 尝试及组级选择概率。`logs/run_summary.json` 的 `best` 按同代码首次有效评分选取，`fitness_instability` 单独列出重复评价分数不一致。

画同口径 best-so-far 时使用事件 `best_so_far`，横轴使用 `evaluation_id`（无评价的解析失败不推进评价轴）。`fitness`、`parent_delta` 等保留本次真实测量；`frontier_delta` / `frontier_improved` 统一使用同代码首次有效评分相对请求前最佳实现的差值/比较，重复高分不抬高前沿。只保留这一套默认前沿字段，`best_so_far` 相邻差分才是累计前沿的实际增加。

`ablations.build_history_ablation` 提供 `code_only`、`single_edge`、`multi_edge` 三臂，使用相同组级调度和最大输入预算，各自记录实际 tokens。该构造器用于独立实验脚本，生产 CLI 不提供混合历史模式。`ablations.build_representation_pair` 补充固定锚点的代码转移 / 短 Idea-Result-Fitness 对照：冻结同一档案、底座、算子和 donor，将两臂可见历史约束为共同最近后缀，返回完整提示、哈希、实际 tokens 和来源。调用方再用相同模型参数重复生成与评价，无效输出保留在分母中；生产 CLI 不变。关系对照及组级调度开关的进一步实验见设计文档。

本地验收（不调用生成服务）：

```bash
uv run pytest -q tests/method/test_traceaad_v108.py tests/method/test_traceaad_v105.py tests/method/test_traceaad_v106.py tests/method/test_traceaad_v107.py tests/method/test_traceaad_v107_sampling.py
```


15 路启动与恢复（5 任务 × 3 重复）：

```bash
uv run python -m experiments.traceaad_v10_8.launch --batch 20260909_v108_formal --watch
```

分配为 server1/server3/server3b 各 4 路、本地 3 路；同任务重复轮换生成端点。搜索进程与评价器沿用本地执行，远端承接模型服务。批次 manifest 固定分配，恢复不更换后端，最多 5 次进程启动；未知评价收据状态保持 blocked。本地 GGUF 与远端 INT4 量化实现不同，报告保留 backend 分组，不将其视作严格相同模型实现。

共享监控支持 `?version=v10_8`；曲线与最佳节点使用同代码首次有效评分，原始散点仍展示实际测量。

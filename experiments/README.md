# experiments 目录索引

原始工件只留本地，Git 只跟踪各版本与基线实验包、plotting 与评估入口。凝练结果按 docs/experiments 的主实验 / 机制实验 / 其他实验分类，工件侧不再设归档层。

## 布局

- **全量自包含实验包**：全部方法（包括 TraceAAD 各版本 `traceaad_*` 以及外部对照基线 `eoh`、`reevo`、`mcts_ahd`、`pathwise`、`calm`、`shinka_evo`）均在 `experiments/<method>/` 根下以一等公民身份自包含，内含运行脚本与发射器（`run.py`、`launch.py`）以及各任务运行结果（`results/<task>/<run>`），全仓没有任何跨目录软链接。
- **共享底座**：`runners/` 仅保留所有方法共享的统一后端表、任务构建与 LLM 客户端（`runners/_common.py`）。

## 主实验

主表版本为 `traceaad_v9_16`、`traceaad_v9_17`。五对照 TSP/CVRP/OP/OBP 的正式数字读各方法目录下 `eval_best_20260825_rerun`（原批最终结果见[基线原批次结果](../docs/experiments/其他实验/基线原批次结果.md)）；VRPTW 对照用 20260822 原批，凝练结果见[主实验/结果](../docs/experiments/主实验/结果.md)。

## 机制实验批次

- `traceaad_v9_18`：V9.18-R0 机会评分实验工件；过程审计 `traceaad_v9_18/analyze.py`。
- `traceaad_v9_17`：V9.17 竞争质量门控实验工件；过程分析 `traceaad_v9_17/analyze.py`。
- `traceaad_v9_19`（原批 + `fixed_20260829` 修订批）、`traceaad_v9_20`：行为度量时代的机制版本。
- `traceaad_v9_21`：思想假设双重实现搜索，首批 `v9_21_core_20260830`，入口 `experiments/traceaad_v9_21/launch.py`。
- `traceaad_v10`：design opportunity 分配，批次 `v10_20260831_q38`（2026-09-01 主动停止、checkpoint 完整可恢复），入口 `experiments/traceaad_v10/launch.py`。
- `traceaad_v10_1`：V10.1 质量概率选父 + Refine/Pivot/Fuse 三算子扩展（机制见 [V10.1 完整机制设计](../docs/methods/TraceAAD-V10.1完整机制设计.md)），正式批次 `20260902_*`（5 任务 × 3 重复、1000 预算），入口 `experiments/traceaad_v10_1/launch.py`；过程工件为各 run 的 `events.jsonl` / `tree_state.json`。
- 历史版本 `traceaad_v9_7` / `traceaad_v9_14` / `traceaad_v9_15`：主表历史对照，数字见[历史版本](../docs/experiments/其他实验/历史版本.md)；V9.7 的过程分析读各 run 的 `artifacts/candidates.jsonl`（`traceaad_v9_7/analyze.py`）。

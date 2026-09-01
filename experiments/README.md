# experiments 目录索引

原始工件只留本地，Git 只跟踪 runners、analysis、plotting 与评估入口。凝练结果按 docs/experiments 的主实验 / 机制实验 / 其他实验分类，工件侧不再设归档层。

## 布局

工件按 `<task>/<method>/<run>` 平铺在规范根：五任务 × 五对照（`eoh`、`reevo`、`mcts_ahd`、`pathwise`、`calm`）+ TraceAAD 各版本（`traceaad_v9_7` … `traceaad_v9_21`、`traceaad_v10`）。批次用 run 名前缀区分，如基线 `20260824_rerun_*` 重跑批、`20260822_142500_*` VRPTW 原批。

实验入口在 `runners/`，每个实验一个独立包：`runners/<experiment>/run.py` 只服务一个实验，批次发射器为包内 `launch.py`（V9.18 双臂战役的发射器在 `runners/traceaad_v9_18/`）。共享的后端表、任务构建、LLM 客户端与统一采样口径在 `runners/_common.py`。

## 主实验

主表版本为 `traceaad_v9_16`、`traceaad_v9_17`。五对照 TSP/CVRP/OP/OBP 的正式数字读各方法目录下 `eval_best_20260825_rerun`（原批最终结果见[基线原批次结果](../docs/experiments/其他实验/基线原批次结果.md)）；VRPTW 对照用 20260822 原批。表格数字由 `analysis/recompute_rankings.py` 从各方法 `eval_best_*` 工件重算，凝练结果见[主实验/结果](../docs/experiments/主实验/结果.md)。

## 机制实验批次

- `traceaad_v9_18_q_atomic` / `traceaad_v9_18_q_opportunity`：V9.18-R0 机会评分 A 阶段两臂，共享 `q_atomic` 下 `v9_18_bootstrap_*` 八根根池；过程审计 `analysis/analyze_v918_process.py`。
- `traceaad_v9_17_fixed_cycle`：V9.17 自适应调度配对消融，`analysis/analyze_v917_scheduler_ablation.py` 以 `traceaad_v9_17` 的 20260823_adaptive 批为对照臂。
- `traceaad_v9_19`（原批 + `fixed_20260829` 修订批）、`traceaad_v9_20`：行为度量时代的机制版本。
- `traceaad_v9_21`：思想假设双重实现搜索，首批 `v9_21_core_20260830`，入口 `runners/traceaad_v9_21/launch.py`。
- `traceaad_v10`：design opportunity 分配，批次 `v10_20260831_q38`（2026-09-01 主动停止、checkpoint 完整可恢复），入口 `runners/traceaad_v10/launch.py`。
- 历史版本 `traceaad_v9_7` / `traceaad_v9_14` / `traceaad_v9_15`：主表历史对照，数字见[历史版本](../docs/experiments/其他实验/历史版本.md)；V9.7 的过程分析读各 run 的 `artifacts/candidates.jsonl`（`analysis/analyze_v97_allocation.py`）。
- `<task>/<method>/eval_best_budget500_20260830/`：主表方法集取前 500 次评价 best 的 held-out 重评工件，入口 `analysis/run_budget500_eval.py`，结果见[500预算对照](../docs/experiments/其他实验/500预算对照.md)。

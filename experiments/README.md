# experiments 目录索引

与 docs/experiments 的三分类对应。原始工件只留本地，Git 只跟踪 runners、analysis、plotting 与评估入口。

## 主实验（规范根）

布局 `<task>/<method>/<run>`：五任务 × `traceaad_v9_16`、`traceaad_v9_17`（主表版本，VRPTW 含 20260822_142500 原批次）+ 在跑机制版本。五对照 TSP/CVRP/OP/OBP 正式数字读 `其他实验/基线重跑-20260824/` 的 `eval_best_20260825_rerun`（原批归档见 [基线原批次结果](../docs/experiments/其他实验/基线原批次结果.md)）；VRPTW 对照仍在规范根。表格数字由 `analysis/recompute_rankings.py` 从各方法 `eval_best_*` 工件重算，凝练结果见 [主实验/结果](../docs/experiments/主实验/结果.md)。

## 机制实验

`traceaad_v9_18_q_atomic` / `traceaad_v9_18_q_opportunity`：V9.18-R0 机会评分 A 阶段，两臂共享 `q_atomic` 下的 `v9_18_bootstrap_*` 八根根池；过程审计用 `analysis/analyze_v918_process.py`。战役结束出结论后整体迁入 `机制实验/2026-08-25-V9.18-R0机会评分/`。V9.17 自适应调度消融读取规范根 `traceaad_v9_17`（20260823_adaptive 批）与 `其他实验/历史版本/*/traceaad_v9_17_fixed_cycle`，脚本为 `analysis/analyze_v917_scheduler_ablation.py`。

`traceaad_v9_21`：思想假设双重实现搜索；每个普通 batch 独立提出 `continue` 与 `branch` 两个 Idea，并各生成两份实现，在线不启用 BehaveSim。15 路首批为 `v9_21_core_20260830`，调度入口为 `runners/traceaad/launch_v921.py`。

`traceaad_v9_22`：在 V9.21 的双重实现协议上加入动态 scaffold mid-rank、working/scaffold 双基准信用、action-UCB 和批内冻结上下文。正式批次尚未启动，调度入口为 `runners/traceaad/launch_v922.py`。

## 其他实验

- `其他实验/历史版本/<task>/traceaad_v9_7|v9_14|v9_15|v9_17_fixed_cycle/`：[历史版本](../docs/experiments/其他实验/历史版本.md)表的证据工件，`recompute_rankings.py` 的 `artifact_dir` 从这里解析；V9.7 的 `analyze_v97_allocation.py` 读这里的 `artifacts/decisions.jsonl`（其余 v9_7 过程分析所需 checkpoints/logs 已在 2026-08-23 清理中删除）。
- `其他实验/基线重跑-20260824/<task>/<method>/`：五对照 20260824 重跑批次，见[基线重跑对照](../docs/experiments/其他实验/基线重跑对照.md)；`plotting/plot_search_curves.py` 的对照曲线从该批次读取。
- `<task>/<method>/eval_best_budget500_20260830/`：主表方法集取前 500 次评价 best 的 held-out 重评工件（基线重跑批同理），入口 `analysis/run_budget500_eval.py`，结果见[500预算对照](../docs/experiments/其他实验/500预算对照.md)。

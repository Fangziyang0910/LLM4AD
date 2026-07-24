# TSP-GLS 实验

单目标（平均 tour cost）；底层任务实现为 `tsp_gls_2O`，实验用 `TSPGLSEvaluation` 只取 `-mean_cost`。

权威数据配置见 `docs/实验配置.md`：训练 TSP200，测试 TSP100/200/500/1000。

## 本轮正式跑（20260720_140109，旧协议）

本批仍按旧配置：训练/测试均为 TSP100。新协议重跑前勿覆盖结果页。

| Method | Budget | Run dirs | GPU 源 |
|---|---:|---|---|
| MCTS-AHD | 1000 | `mcts_ahd/20260720_140109_tspgls_rep{1,2,3}` | rep1 zhong / rep2 server1 / rep3 Fang_lab |
| PathWise | 500 | `pathwise/20260720_140109_tspgls_rep{1,2,3}` | 同上 |
| TraceAAD | 1000 | `traceaad/version2/20260720_140109_tspgls_rep{1,2,3}` | 同上 |

- 训练（旧）：`n_instance=16`，`problem_size=100`，`seed=2024`，`timeout_seconds=60`
- 分数：`-mean_tour_cost`，越高越好
- 启动日志：`launch_20260720_140109.log`

## 已完成评估（旧协议）

三方法 held-out `seed=2025`（同规模 TSP100）与训练曲线见 `docs/results/tsp_gls/`。

```bash
uv run python experiments/tsp_gls/evaluate_best_on_test.py \
  experiments/tsp_gls/pathwise/20260720_140109_tspgls_rep{1,2,3} \
  --output-dir experiments/tsp_gls/pathwise/eval_best_20260720_140109
MPLCONFIGDIR=/tmp/matplotlib uv run python experiments/plotting/plot_tsp_gls_three_method_search.py
```

新的单次搜索直接使用各方法目录中的 `run_experiment.py`。实验覆盖与完成状态
统一维护在 `docs/实验覆盖.md`，不在任务目录重复维护版本状态。

# TSP-GLS 实验

单目标（平均 tour cost）；底层任务实现为 `tsp_gls_2O`，实验用 `TSPGLSEvaluation` 只取 `-mean_cost`。

## 本轮正式跑（20260720_140109）

| Method | Budget | Run dirs | GPU 源 |
|---|---:|---|---|
| MCTS-AHD | 1000 | `mcts_ahd/20260720_140109_tspgls_rep{1,2,3}` | rep1 zhong / rep2 server1 / rep3 Fang_lab |
| PathWise | 500 | `pathwise/20260720_140109_tspgls_rep{1,2,3}` | 同上 |
| TraceAAD | 1000 | `traceaad/version2/20260720_140109_tspgls_rep{1,2,3}` | 同上 |

- 训练：`n_instance=16`，`problem_size=100`，`seed=2024`，`timeout_seconds=60`
- 分数：`-mean_tour_cost`，越高越好
- TraceAAD：`SEARCH_SEED=2024`
- tmux：`tspgls_<method>_rep{1,2,3}`
- 启动日志：`launch_20260720_140109.log`
- 重启：`bash experiments/tsp_gls/launch_nine_tmux.sh`

查看进度：

```bash
tmux ls | rg '^tspgls_'
tail -20 experiments/tsp_gls/*/20260720_140109_tspgls_rep1/tmux_run.log
tail -20 experiments/tsp_gls/traceaad/version2/20260720_140109_tspgls_rep1/tmux_run.log
```

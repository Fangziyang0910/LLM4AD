# Knapsack Construct 实验

## 本轮正式跑（20260719_223427）

| Method | Budget | Run dirs | GPU 源 |
|---|---:|---|---|
| MCTS-AHD | 1000 | `mcts_ahd/20260719_223427_kp_rep{1,2,3}` | rep1 zhong / rep2 server1 / rep3 Fang_lab |
| PathWise | 500 | `pathwise/20260719_223427_kp_rep{1,2,3}` | 同上 |
| TraceAAD | 1000 | `traceaad/version2/20260719_223427_kp_rep{1,2,3}` | 同上 |

- 训练数据：`n_instance=32`，`n_items=50`，`knapsack_capacity=100`，**`seed=2024`**
- 分数语义：平均总价值，**越高越好**（启动前修正了评估符号：原先误返回负值）
- TraceAAD 搜索种子：`SEARCH_SEED=2024`
- tmux 会话：`kp_<method>_rep{1,2,3}`
- 启动日志：`launch_20260719_223427.log`
- 重启命令：`bash experiments/knapsack_construct/launch_nine_tmux.sh`

查看进度：

```bash
tmux ls | rg '^kp_'
tail -20 experiments/knapsack_construct/*/20260719_223427_kp_rep1/tmux_run.log
tail -20 experiments/knapsack_construct/traceaad/version2/20260719_223427_kp_rep1/tmux_run.log
```

测试评估（权威结果见 `docs/results/knapsack_construct/`）：

```bash
uv run python experiments/knapsack_construct/evaluate_best_on_test.py <run_dirs...> --output-dir <eval_dir>
MPLCONFIGDIR=/tmp/matplotlib uv run python experiments/plotting/plot_knapsack_construct_three_method_search.py
```

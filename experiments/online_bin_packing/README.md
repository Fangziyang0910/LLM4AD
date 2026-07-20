# Online Bin Packing 实验

## 本轮正式跑（20260719_150058）

| Method | Budget | Run dirs | GPU 源 |
|---|---:|---|---|
| MCTS-AHD | 1000 | `mcts_ahd/20260719_150058_obp_rep{1,2,3}` | rep1 zhong / rep2 server1 / rep3 Fang_lab |
| PathWise | 500 | `pathwise/20260719_150058_obp_rep{1,2,3}` | 同上 |
| TraceAAD | 1000 | `traceaad/version2/20260719_150058_obp_rep{1,2,3}` | 同上 |

- 训练数据：Weibull，`n_instances=5`，`n_items=5000`，`capacity=100`，**`seed=2024`**
- TraceAAD 搜索种子：`SEARCH_SEED=2024`
- tmux 会话：`obp_<method>_rep{1,2,3}`
- 启动日志：`launch_20260719_150058.log`
- 重启命令：`bash experiments/online_bin_packing/launch_nine_tmux.sh`

查看进度：

```bash
tmux ls | rg '^obp_'
tail -20 experiments/online_bin_packing/*/20260719_150058_obp_rep1/tmux_run.log
tail -20 experiments/online_bin_packing/traceaad/version2/20260719_150058_obp_rep1/tmux_run.log
```

测试评估（已完成权威结果见 `docs/results/online_bin_packing/`）：

```bash
uv run python experiments/online_bin_packing/evaluate_best_on_test.py <run_dirs...> --output-dir <eval_dir>
MPLCONFIGDIR=/tmp/matplotlib uv run python experiments/plotting/plot_online_bin_packing_three_method_search.py
```

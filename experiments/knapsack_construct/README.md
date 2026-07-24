# Knapsack Construct 实验

权威数据配置见 `docs/实验配置.md`：训练 KP100，测试 KP50/100/200。

## 历史正式跑（20260719_223427，旧协议）

本批仍按旧配置训练与测试：`n_items=50`、同规模 held-out。新协议重跑前勿覆盖结果页。

| Method | Budget | Run dirs | GPU 源 |
|---|---:|---|---|
| MCTS-AHD | 1000 | `mcts_ahd/20260719_223427_kp_rep{1,2,3}` | rep1 zhong / rep2 server1 / rep3 Fang_lab |
| PathWise | 500 | `pathwise/20260719_223427_kp_rep{1,2,3}` | 同上 |
| TraceAAD | 1000 | `traceaad/version2/20260719_223427_kp_rep{1,2,3}` | 同上 |

- 训练数据（旧）：`n_instance=32`，`n_items=50`，`knapsack_capacity=100`，**`seed=2024`**
- 分数语义：平均总价值，**越高越好**
- 启动日志：`launch_20260719_223427.log`

测试评估（新协议默认扫 KP50/100/200）：

```bash
uv run python experiments/knapsack_construct/evaluate_best_on_test.py <run_dirs...> --output-dir <eval_dir>
MPLCONFIGDIR=/tmp/matplotlib uv run python experiments/plotting/plot_knapsack_construct_three_method_search.py
```

新的单次搜索直接使用各方法目录中的 `run_experiment.py`。实验覆盖与完成状态
统一维护在 `docs/实验覆盖.md`，不在任务目录重复维护版本状态。

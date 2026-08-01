# Online Bin Packing 实验

权威数据配置见 `docs/experiments/配置.md`：训练使用四个固定实例
`1k/5k × C∈{100,500}`；测试使用不同的固定实例
`1k/5k/10k × C∈{100,500}`，其中只有 10k 为 OOD。

## 历史正式跑（旧单容量训练协议）

既有运行均只在 `5k_100` 上搜索，与当前多容量训练协议不一致。
结果仅作为历史证据保留；MCTS-AHD、PathWise 和后续指定的 TraceAAD
版本需要按当前 task 配置重新训练，不能复用旧 heuristic 作为新协议结果。

| Method | Budget | Run dirs | GPU 源 |
|---|---:|---|---|
| MCTS-AHD | 1000 | `mcts_ahd/20260719_150058_obp_rep{1,2,3}` | rep1 zhong / rep2 server1 / rep3 Fang_lab |
| PathWise | 500 | `pathwise/20260719_150058_obp_rep{1,2,3}` | 同上 |
| TraceAAD | 1000 | `traceaad/version2/20260719_150058_obp_rep{1,2,3}` | 同上 |

- 历史训练数据：Weibull，`n_instances=5`，`n_items=5000`，`capacity=100`，**`seed=2024`**
- 启动日志：`launch_20260719_150058.log`

测试评估（当前协议默认扫 1k/5k/10k × 100/500）：

```bash
uv run python experiments/online_bin_packing/evaluate_best_on_test.py <run_dirs...> --output-dir <eval_dir>
MPLCONFIGDIR=/tmp/matplotlib uv run python experiments/plotting/plot_online_bin_packing_three_method_search.py
```

TraceAAD 新搜索统一使用 `experiments.runners.traceaad.run`，三重复使用
`experiments.runners.traceaad.launch`，通过 `--task online_bin_packing` 与
`--version v4|v5` 选择实验。入口会为每个 run 自动保存真实配置，不再创建
批次专用脚本。实验覆盖与完成状态统一维护在 `docs/experiments/覆盖.md`。

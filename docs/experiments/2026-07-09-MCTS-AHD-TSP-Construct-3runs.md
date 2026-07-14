# MCTS-AHD TSP Construct 三次重复

使用 `qwen3.6-27b-awq` 在 `tsp_construct` 上复现 MCTS-AHD，预算 1000，4 个 evaluator，三次独立 tmux run。三次运行均正常完成；TSP50/100/200 测试结果已汇总到 [TSP Construct 结果](../results/tsp-construct-qwen36-27b.md)。

主要 artifact 位于 `experiments/tsp_construct/mcts_ahd/`，实验入口为 `experiments/tsp_construct/mcts_ahd/run_experiment.py`。

# PathWise TSP Construct 三次重复

使用 `qwen3.6-27b-awq` 在 `tsp_construct` 上运行 PathWise，预算 500，4 个 evaluator，三次独立 run。各 run 的 best heuristic 已在 TSP50/100/200 上评估，汇总见 [TSP Construct 结果](../results/tsp-construct-qwen36-27b.md)。

实验入口为 `experiments/tsp_construct/pathwise/run_experiment.py`，artifact 位于 `experiments/tsp_construct/pathwise/`。

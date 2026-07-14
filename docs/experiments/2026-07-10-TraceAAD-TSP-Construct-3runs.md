# TraceAAD TSP Construct 三次重复

使用当前 TraceAAD 实现和 `qwen3.6-27b-awq` 在 `tsp_construct` 上运行三次独立实验，预算 1000，采用 trajectory-UCB、islands、novelty gate 和六类搜索算子，并通过 `NO_PROXY` 直连 vLLM。

各 run 的 best heuristic 已完成 TSP50/100/200 测试，汇总见 [TSP Construct 结果](../results/tsp-construct-qwen36-27b.md)。

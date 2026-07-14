# CVRP-ACO 三方法搜索

在同一 CVRP-ACO 任务上并行运行 MCTS-AHD、PathWise 和 TraceAAD，模型为 `qwen3.6-27b-awq`，训练集为 10 个 CVRP50 实例，ACO 配置为 30 ants × 100 iterations；预算分别为 1000、500、1000。

三种方法均完成三次重复，test_50/test_100 结果已汇总到 [CVRP-ACO 结果](../results/cvrp-aco-qwen36-27b.md)。

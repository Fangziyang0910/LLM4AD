# TraceAAD TSP 首次完整 vLLM 实验

首次在 TSP Construct 上完成真实 vLLM 搜索：`qwen3.6-27b-awq`、预算 1000，最终有效样本 977 个、0 个错误，best score 为 `-6.371328`。

best 在 sample 318 后长期停滞，促成后续的算子、trajectory selection、novelty 和 credit 机制审计。

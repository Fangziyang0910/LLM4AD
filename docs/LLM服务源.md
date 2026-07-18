# LLM 服务源

客户端统一用 `OpenAIAPI`，在 `run_experiment.py` 顶部写 `base_url` / `model` / `api_key`。远端访问加 `NO_PROXY=<ip>,localhost,127.0.0.1,::1`。

| 服务 | base_url | model | api_key | 配置 | 单请求 | 3 并发总吞吐 |
|------|----------|-------|---------|------|--------|--------------|
| llama.cpp GPU1 | `http://183.36.243.124:9000/v1` | `Qwen3.6-27B-MTP-Q4_K_M` | `4bda78bfe7b538e6057c561a6692724b133758cb86964400482c56b56f01c7d2` | 并发 2；上下文 64K；MTP≤2 | ~146 tok/s | — |
| vLLM | `http://222.201.145.8:8080/v1` | `qwen3.6-27b-awq` | `EMPTY` | 主实验 endpoint | ~50 tok/s | ~128 tok/s（单路 ~45） |
| llama.cpp 本机 | `http://127.0.0.1:8001/v1` | `Qwen3.6-27B` | `EMPTY` | `llama qwen3.6-27b`；`-np 3`；总上下文 96K（每槽 32K）；MTP≤2 | ~88 tok/s（`-np 1`） | ~145 tok/s（单路 ~60–66） |

# LLM 服务源

客户端统一用 `OpenAIAPI`。实验入口从环境变量读 `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`；仓库根目录 `.env`（gitignore）可放密钥，由 `llm4ad.tools.env` 自动加载。Zhong 密钥字段为 `ZHONG_API_KEY`：当 `LLM_API_KEY` 未设且 `base_url` 指向 Zhong 时自动使用。远端访问加 `NO_PROXY=<ip>,localhost,127.0.0.1,::1`。

`OpenAIAPI` 默认 `enable_thinking=False`（经 `chat_template_kwargs` 关闭思考）；需要思考时显式传 `enable_thinking=True`。

| 服务         | base_url                          | model                  | api_key                     | 配置                              | 单请求     | 3 并发总吞吐                |
| ------------ | --------------------------------- | ---------------------- | --------------------------- | --------------------------------- | ---------- | --------------------------- |
| zhong-server | `http://183.36.243.124:9000/v1` | `Qwen3.6-27B-Q4_K_M` | `.env` 中 `ZHONG_API_KEY` | 并发 4；总上下文 96K（每槽 32K）  | ~134 tok/s | ~226 tok/s（单路 ~88–104） |
| server1      | `http://222.201.145.8:8080/v1`  | `qwen3.6-27b-awq`    | `EMPTY`                   | 并发 4；总上下文 96K（每槽 32K）  | ~50 tok/s  | ~128 tok/s（单路 ~45）      |
| Fang_lab     | `http://127.0.0.1:8001/v1`      | `Qwen3.6-27B`        | `EMPTY`                   | 并发 4；总上下文 96K（每槽 32K）  | ~88 tok/s  | ~145 tok/s（单路 ~60–66）  |

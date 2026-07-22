# LLM 服务源

客户端统一用 `OpenAIAPI`。实验入口从环境变量读 `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`；仓库根目录 `.env`（gitignore）可放密钥，由 `llm4ad.tools.env` 自动加载。Zhong 密钥字段为 `ZHONG_API_KEY`：当 `LLM_API_KEY` 未设且 `base_url` 指向 Zhong 时自动使用。远端访问加 `NO_PROXY=<ip>,localhost,127.0.0.1,::1`。

`OpenAIAPI` 默认 `enable_thinking=False`（经 `chat_template_kwargs` 关闭思考）；需要思考时显式传 `enable_thinking=True`。

### 调度口径（权威）

| 服务 | 允许并发实验请求 | 备注 |
| ---- | ---: | ---- |
| **zhong-server** | **10** | 2026-07-22 长请求探针确认；后续排期可同时挂最多 10 路 |
| server1 | 3 | 服务端宣称并发 4，实验侧保守用 3 |
| Fang_lab | 3 | 本机 `-np 3` |

空闲即用；跨源视为等价模型，按上表分别计数。Zhong 的 `LLM_MODEL` 必须用 listed id（见下表）。

| 服务         | base_url                          | model                  | api_key                     | 配置 | 单请求 | 并发总吞吐 |
| ------------ | --------------------------------- | ---------------------- | --------------------------- | ---- | ------ | ---------- |
| zhong-server | `http://183.36.243.124:9000/v1` | `Qwen3.6-27B-NVFP4`（listed id: `/home/fzy/models/Qwen3.6-27B-NVFP4`） | `.env` 中 `ZHONG_API_KEY` | 并发 **10** | ~79 tok/s | 3 路 ~200；**10 路 ~617** |
| server1      | `http://222.201.145.8:8080/v1`  | `qwen3.6-27b-awq`    | `EMPTY`                   | 并发 4；总上下文 96K（每槽 32K） | ~50 tok/s | 3 路 ~128（单路 ~45） |
| Fang_lab     | `http://127.0.0.1:8001/v1`      | `Qwen3.6-27B`        | `EMPTY`                   | 并发 3（`-np 3`）；总上下文 96K（每槽 32K） | ~88 tok/s | 3 路 ~145（单路 ~60–66） |

## Zhong 2026-07-22 长请求并发探针

任务：结合代码用中文介绍 5 种排序算法（冒泡 / 插入 / 归并 / 快速 / 堆），`max_tokens=8192`，`enable_thinking=False`，`temperature=1.0`。每档并发重复 **3** 轮。全部 `finish=stop`（未被长度截断），单次输出约 **3k–5k** completion tokens。

| 并发 | 成功 | 系统吞吐 mean±std | 单路墙钟 tok/s | 输出 tokens | batch 墙钟 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3×1/1 | **79.1±2.2** | 79.1±2.2 | 3724±503 | 47.0±5.3s |
| 3 | 3×3/3 | **199.9±8.0** | 78.9±0.5 | 3725±280 | 56.1±6.1s |
| 10 | 3×10/10 | **617.2±26.5** | 76.7±2.4 | 3964±129 | 64.3±1.9s |

结论：

- **10 路满并发稳定可用**（42/42 成功），单路速率几乎不掉（~79 → ~77）。
- 系统吞吐近似线性：10 路 ≈ 单路 ×7.8，约为 3 路的 **3.1×**。
- 旧模型名 `Qwen3.6-27B-Q4_K_M` 已 404；续跑/新跑必须用 listed id `/home/fzy/models/Qwen3.6-27B-NVFP4`。

原始结果：`/tmp/zhong_concurrency_long_probe.json`。

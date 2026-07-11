# TraceAAD TSP Construct vLLM

正式实验已用 tmux 启动。

- tmux session: `traceaad_tsp_construct_20260708_150716`
- 历史启动参数: `LLM4AD/experiments/tsp_construct/traceaad/20260708_150716/run_config.json`
- 控制台日志: `LLM4AD/experiments/tsp_construct/traceaad/20260708_150716/tmux_run.log`
- profiler 日志目录: `LLM4AD/experiments/tsp_construct/traceaad/20260708_150716/logs/`

主要设置：`qwen3.6-27b-awq`，远程 vLLM endpoint `http://222.201.145.8:8080/v1`，`train` split，`max_sample_nums=1000`，`n_init=4`，`actions_per_iteration=2`，`eval_workers=16`，`eval_backend=process`。

启动后健康检查：tmux session 和 Python 进程仍在运行；LLM 请求返回 `HTTP 200 OK`；已写出 `llm_calls.jsonl`、`method_events.jsonl`、`method_state.jsonl`、`samples/samples_1~200.json` 和 `samples_best.json`；暂无 `errors.jsonl`。

初始进展：已评估 5 个样本。前三个 init score 分别为 `-7.714284413378637`、`-7.340271469613269`、`-7.161891941301638`，当前 best score 是 `-7.161891941301638`。iteration 0 已完成，iteration 1 已开始。

## 2026-07-08 20:48 vLLM 请求异常排查

当前 run 没有表现为远端 vLLM 完全不可访问：`/v1/models` 可返回模型 `qwen3.6-27b-awq`，短 chat 探针可在约 1.5s 返回；512-token 项目内探针约 57.8s 返回。

异常主要集中在 code 生成阶段。`run_log.txt` 中 15-18 点有大量 `HTTP/1.1 200 OK`，19 点后 retry 增多；到 20 点这一小时约 4 次 200、14 次 retry、5 次新的 `APITimeoutError`，累计 `errors.jsonl` 中 13 条错误均为 `APITimeoutError: Request timed out.`。`llm_calls.jsonl` 中失败间隔稳定在约 6 分钟，说明这是 OpenAI SDK 在 120s timeout 上多次 retry 后才抛给方法层。

实验进程继承了本机代理环境：`HTTP_PROXY=http://127.0.0.1:7890`，`NO_PROXY` 未包含 `222.201.145.8`，错误栈经过 `httpcore._sync.http_proxy`。因此当前判断不是 endpoint 断开，而是远端 vLLM 在两个实验并发、长 prompt / 长生成和 `max_tokens=16384` 设置下吞吐/队列不稳定，且请求路径经过本机代理，放大了长请求超时风险。

## 2026-07-08 21:01 timeout=400 重启

按新的运行判断，将 LLM 客户端 timeout 从 `120s` 提高到 `400s`，保留 evaluator 的 `TASK_TIMEOUT_SECONDS=20` 不变。旧日志已备份到 `logs_timeout120_before_restart_20260708_205826/`，旧控制台日志备份到 `tmux_run_timeout120_before_restart_20260708_205826.log`。

新 tmux session 仍为 `traceaad_tsp_construct_20260708_150716`，启动时显式设置 `NO_PROXY=222.201.145.8,localhost,127.0.0.1,::1`。健康检查显示新 run 从 `_tot_sample_nums=0` 干净启动，profile 中 `timeout: 400`，并已产生多次 `HTTP/1.1 200 OK`；约 21:01 时已写出新的 `llm_calls.jsonl`、`method_events.jsonl`、`method_state.jsonl`、`samples/samples_1~200.json` 和 `samples_best.json`。

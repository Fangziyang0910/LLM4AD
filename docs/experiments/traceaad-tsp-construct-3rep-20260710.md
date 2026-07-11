# TraceAAD TSP Construct — 正式三次重复实验 (1000 samples)

日期：2026-07-10
状态：运行中（20:35 启动）

## 目标

TraceAAD 在 `tsp_construct` 上的正式三次重复实验，`max_sample_nums=1000`，用于与 mcts_ahd(1000) / pathwise(500) 横向对比。

## 三个 run

| rep | run_dir | tmux session | 启动时间 |
|---|---|---|---|
| rep1 | `LLM4AD/experiments/tsp_construct/traceaad/20260710_203531` | `traceaad_tsp_rep1` | 20:35:29 |
| rep2 | `LLM4AD/experiments/tsp_construct/traceaad/20260710_203541` | `traceaad_tsp_rep2` | 20:35:39 |
| rep3 | `LLM4AD/experiments/tsp_construct/traceaad/20260710_203551` | `traceaad_tsp_rep3` | 20:35:49 |

入口脚本：`LLM4AD/experiments/tsp_construct/traceaad/run_experiment.py`（默认 `MAX_SAMPLE_NUMS=1000`，`SEARCH_SEED=2024`）。

## 配置（与 mcts/pathwise 对齐以便对比）

- model：`qwen3.6-27b-awq`，vLLM endpoint `http://222.201.145.8:8080/v1`
- `max_sample_nums=1000`，`random_seed=2024`（三次同 seed，差异来自 LLM temperature=1.0 采样，与 mcts/pathwise 三重复惯例一致）
- `n_init=4, actions_per_iteration=2, n_islands=4, max_per_island=40, novelty_threshold=0.92`
- train：`problem_size=50, n_instance=16, seed=2024`
- `LLM timeout=600, max_tokens=16384, enable_thinking=False`
- **关键运行时**：`NO_PROXY=222.201.145.8,localhost,127.0.0.1,::1` 且清空 `http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` 直连（规避 [[traceaad-tsp-construct-vllm]] 记录的 7-08 代理超时）

## 并发策略

并行三个（用户选定）。启动时 vLLM 上 pathwise rep3 仍在运行（约 404/500），峰值 4 并发。7-08 在「2 并发 + 经代理 + timeout=120」下出现过 timeout 崩溃；本次 timeout=600 + NO_PROXY 直连。启动后约 5 分钟观察：三进程存活、连续 `HTTP/1.1 200 OK`、无 `errors.jsonl`、vLLM `/v1/models` 5–9ms 响应，稳定。

## 启动后健康快照 (20:40)

- 三进程存活
- rep1≈5 / rep2≈6 / rep3≈6 samples；rep3 best=`-6.82`（接近 7-08 同期水平）

## 监控命令

```bash
# 进度（迭代 / best）
tail -n 30 LLM4AD/experiments/tsp_construct/traceaad/<run_dir>/logs/run_log.txt
# 已评估样本数
python3 -c "import json;print(len(json.load(open('LLM4AD/experiments/tsp_construct/traceaad/<run_dir>/logs/samples/samples_1~200.json'))))"
# attach 看 LLM 流
tmux attach -t traceaad_tsp_rep1   # Ctrl-b d 退出
# 错误
ls LLM4AD/experiments/tsp_construct/traceaad/<run_dir>/logs/errors.jsonl
```

## 完成后

三 run 全部完成后，对每个 run 的 best 在 TSP50/100/200 (seed=2025, held-out) 上评估：

```bash
cd /home/fang/code/LLM4AD/LLM4AD
NO_PROXY=222.201.145.8,localhost,127.0.0.1,::1 uv run python experiments/tsp_construct/eval_best_on_test.py <run_dir>
```

结果定稿后写入 `docs/results/traceaad-qwen36-27b-tsp-construct.md`。

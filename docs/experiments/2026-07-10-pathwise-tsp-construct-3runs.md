# PathWise TSP Construct 3-run launch

日期：2026-07-10
分类：experiment

mcts-ahd 三重复于 2026-07-10 上午全部正常跑完（见 [2026-07-09-mcts-ahd-tsp-construct-3runs.md](2026-07-09-mcts-ahd-tsp-construct-3runs.md)，三路 best 分别 -6.245 / -6.291 / -6.091，均 `status=finished`，0 error）后，按计划启动 PathWise 在 `tsp_construct` 上的 3 个独立 run。入口：

```bash
cd /home/fang/code/LLM4AD/LLM4AD
NO_PROXY=222.201.145.8,localhost,127.0.0.1,::1 uv run python experiments/tsp_construct/pathwise/run_experiment.py
```

入口 `experiments/tsp_construct/pathwise/run_experiment.py` 是为本次新建的，镜像 mcts_ahd runner（复用同一 vLLM endpoint / 同一 tsp_construct train 任务 / `VLLMOpenAIAPI` 客户端 / `NO_PROXY` / 3 个 tmux 各一 run），PathWise 参数取 `llm4ad/method/pathwise/paras.yaml` 默认。冒烟测试（py_compile + `profiler=None` 构造 `PathWise`）已通过。

## 启动设置

- 任务：`tsp_construct`，split=`train`
- 方法：`pathwise`
- 模型：`qwen3.6-27b-awq` @ `http://222.201.145.8:8080/v1`（`VLLMOpenAIAPI`，与 mcts_ahd 同 endpoint）
- PathWise 参数：`max_sample_nums=500, pop_size=6, init_pop_size=None(→30), num_actions=2, num_rollouts=2, max_inner_steps=3, num_evaluators=4`，扰动概率 0.5→0.25
- LLM：timeout=600s, max_tokens=16384, temperature=1.0, enable_thinking=False
- 启动时间 2026-07-10 12:34:42，3 个 tmux 间隔 6s 启动以避开时间戳目录冲突

## 运行目录

| repeat | tmux session | run directory |
|---:|---|---|
| 1 | `pathwise_tsp_repeat_1_20260710_123442` | `experiments/tsp_construct/pathwise/20260710_123444` |
| 2 | `pathwise_tsp_repeat_2_20260710_123442` | `experiments/tsp_construct/pathwise/20260710_123450` |
| 3 | `pathwise_tsp_repeat_3_20260710_123442` | `experiments/tsp_construct/pathwise/20260710_123456` |

## 启动健康检查（12:36）

3 个 tmux session + 3 组 `uv`/python 进程存活；3 个 run_dir 均已生成 `run_config.json` / `tmux_run.log` / `logs/`。三路均已完成初始化进入搜索，出现真实 evaluate：

- `20260710_123444`：sample ~12，best `-6.926`
- `20260710_123450`：sample ~13，best `-6.454`
- `20260710_123456`：sample ~12，best `-6.824`

## 卡住判定与重启 runbook

- tmux 面板只显示一行 `run_dir=`（`run_experiment.py` 把 stdout/stderr 重定向到 `tmux_run.log`），进度看 `logs/`。
- 判定卡住：`logs/` 下 jsonl 与 `tmux_run.log` 的 mtime 超 ~15 分钟无更新，且 python 进程仍在（典型为 LLM 请求挂起）。
- 重启命令（**必须带 `NO_PROXY`**）：见上方入口；注意 `run_experiment.py` 用 `exist_ok=False` 时间戳目录，**不支持原地续跑**，重启等于开新 run（损失进度）。
- 注意：pathwise 单样本含多次串行 LLM 调用（policy + world model + policy/world critic），单样本比 mcts_ahd 慢；500 预算下总时长可能 8–16h。

## 监控

/loop phase-2 cron 每 30 分钟复查这 3 个 pathwise run；卡住则重启；三路跑完则读各自 `logs/run_summary.json` 汇总并停止监控。

## 测试集评估 (tsp50 / tsp100 / tsp200) — 2026-07-10

对每个 run 的 **best 启发式**（train best，score 最高者）在 held-out **eval 种子 (seed=2025)** 下、不同 `problem_size` 评估。`score = -平均巡回长度`（n_instance=16，越高/越接近 0 = 巡回越短 = 越好）。**不同 size 的 score 量级不同，不可跨 size 比较**，只能跨方法在同一 size 上比。

评估脚本：`LLM4AD/experiments/tsp_construct/eval_best_on_test.py <run_dir>`（自动取 best，并跑 train sanity——与 logged score 完全一致，验证取函数与评估方式正确）。注意：tsp_construct 平台 split 只有 train(seed=2024)/eval(seed=2025) 且都 problem_size=50；tsp100/tsp200 用 eval 种子自行构造更大实例（n_instance=16, timeout=120s）。

| run (run_dir) | best sample | train (seed2024, size50) | tsp50 (seed2025) | tsp100 (seed2025) | tsp200 (seed2025) |
|---|---:|---:|---:|---:|---:|
| rep1 (`20260710_123444`) | 450 | -6.296053 | -6.801010 | -9.132097 | -12.985336 |
| rep2 (`20260710_123450`) | 385 | -6.408664 | -6.635784 | -9.192803 | -12.789604 |
| rep3 (`20260710_123456`) | — (运行中) | -6.195* | 待评 | 待评 | 待评 |

\* rep3 仍在运行（404/500），train best 为暂取值、best sample 可能更新；rep3 完成后补齐该行并给出三路 mean±std。

初步观察（待 rep3 完善后确认）：
- tsp50 上两路都比 train 掉 ~0.2–0.5（轻度泛化 gap，正常）。
- 有意思：rep2 的 train best (-6.409) 差于 rep1 (-6.296)，但 tsp50 测试分 (-6.636) 反而优于 rep1 (-6.801)——train best ≠ test best，提示最终应以 test 分为准。
- 尺度放大到 tsp100/tsp200，score 量级随巡回长度自然增大（同 size 内才可比）。

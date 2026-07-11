# MCTS-AHD TSP Construct 3-run launch

日期：2026-07-09
分类：experiment

本次按 MCTS-AHD 原始论文主实验的重复运行设置，在 `tsp_construct` 上同时启动 3 个独立 run。入口脚本均为：

```bash
cd /home/fang/code/LLM4AD/LLM4AD
uv run python experiments/tsp_construct/mcts_ahd/run_experiment.py
```

## 启动设置

- 任务：`tsp_construct`
- split：`train`
- 方法：`mcts_ahd`
- 模型：`qwen3.6-27b-awq`
- endpoint：`http://222.201.145.8:8080/v1`
- `max_sample_nums=1000`
- `init_size=4`
- `pop_size=10`
- `selection_num=2`
- `num_samplers=4`
- `num_evaluators=4`
- `alpha=0.5`
- `lambda_0=0.1`
- `eval_executor=thread`
- LLM timeout：`600s`
- tmux 启动环境额外设置：`NO_PROXY=222.201.145.8,localhost,127.0.0.1,::1`

## 运行目录

| repeat | tmux session | run directory |
|---:|---|---|
| 1 | `mcts_ahd_tsp_repeat_1_20260709_205348` | `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_205353` |
| 2 | `mcts_ahd_tsp_repeat_2_retry_20260709_205426` | `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_205428` |
| 3 | `mcts_ahd_tsp_repeat_3_20260709_205348` | `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_205354` |

## 启动健康检查

启动后确认 3 个 tmux session 均存活，3 个目录均有独立 `run_config.json`、`tmux_run.log`、`logs/run_log.txt` 和 `logs/llm_calls.jsonl`。初始日志中均已出现 `HTTP/1.1 200 OK` 与真实 sample 输出：

- `20260709_205353`：已到 sample 5，当前 best score `-7.042707172128555`。
- `20260709_205354`：已到 sample 4，当前 best score `-6.567232442996065`。
- `20260709_205428`：已到 sample 3，当前 best score `-7.3845946210529885`。

备注：第一次批量启动时第 2 个 session 因脚本秒级时间戳目录冲突退出，随后以 `mcts_ahd_tsp_repeat_2_retry_20260709_205426` 补启，最终保持 3 个有效重复实验同时运行。

## 异常诊断

2026-07-09 晚间复查时，3 个 tmux session 已全部退出，`run_experiment.py` 无存活进程。复查 endpoint 时，`http://222.201.145.8:8080/v1/models` 和最小 `chat/completions` 请求均返回 200，因此当时并不是模型服务不可达。

3 个 run 都提前结束，不应作为有效的 1000-sample 重复实验结果：

| run directory | summary status | samples | LLM calls | best score |
|---|---:|---:|---:|---:|
| `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_205353` | `finished` | 20 | 42 | `-7.042707172128555` |
| `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_205354` | `finished` | 15 | 30 | `-6.3057556716094085` |
| `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_205428` | `finished` | 20 | 42 | `-6.719436068623354` |

备注：上述 3 个旧 run artifact 已按要求删除，目录 `20260709_205353`、`20260709_205354`、`20260709_205428` 不再保留。

根因定位到集成实现的 `s1` expansion：`llm4ad/method/mcts_ahd/mcts_ahd.py` 中 s1 重复检查误引用未定义变量 `indivs`，会在 s1 成功采样并评估后触发异常；同时 `run()` 的 `finally` 会把未处理异常误写成 `status=finished`，导致 summary 表面看起来正常。已修复为使用 s1 的 `path_set` 做重复检查，并让未处理异常写出 `status=error`。验证：`uv run pytest tests/method/test_mcts_ahd_mechanics.py -q` 通过，结果为 `20 passed`。

## 修复后重启

修复后重新启动 3 个独立 tmux run：

| repeat | tmux session | run directory |
|---:|---|---|
| 1 | `mcts_ahd_tsp_repeat_1_fix_20260709_211837` | `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_211852` |
| 2 | `mcts_ahd_tsp_repeat_2_fix_20260709_211837` | `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_211854` |
| 3 | `mcts_ahd_tsp_repeat_3_fix_20260709_211837` | `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_211856` |

健康检查快照：3 个 tmux session 与 3 个 Python run 进程均存活，均持续出现 `HTTP/1.1 200 OK`。三个 run 都已通过真实 `s1` expansion 分支：

- `20260709_211852`：sample 24，LLM calls 48；已在 sample 17、23 记录 `op=s1` expanded。
- `20260709_211854`：sample 19，LLM calls 38；已在 sample 17 记录 `op=s1` expanded。
- `20260709_211856`：sample 24，LLM calls 48；已在 sample 15、22 记录 `op=s1` expanded。

备注：上述修复后重启的 3 个 run artifact 已按要求停止并删除，目录 `20260709_211852`、`20260709_211854`、`20260709_211856` 不再保留。

## 新三次独立实验

按要求重新启动 3 个独立 tmux run：

| repeat | tmux session | run directory |
|---:|---|---|
| 1 | `mcts_ahd_tsp_repeat_1_20260709_213449` | `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_213505` |
| 2 | `mcts_ahd_tsp_repeat_2_20260709_213449` | `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_213507` |
| 3 | `mcts_ahd_tsp_repeat_3_20260709_213449` | `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_213510` |

启动健康检查：3 个 tmux session 与 3 个 Python run 进程均存活；endpoint `http://222.201.145.8:8080/v1/models` 返回 200，run 日志持续出现 `HTTP/1.1 200 OK`。快照：

- `20260709_213505`：sample 13，LLM calls 26，当前 best `-6.638237231415435`。
- `20260709_213507`：sample 7，LLM calls 14，当前 best `-6.820211505568105`。
- `20260709_213510`：sample 10，LLM calls 20，当前 best `-7.121947378699973`。

未发现 `Traceback`、`NameError`、`Timeout`、连接错误、4xx/5xx 或 `sample_error`。

2026-07-09 21:55 复查：3 个 tmux session 与 Python 子进程仍存活，run 日志仍持续出现 `HTTP/1.1 200 OK` 与 `expand` 事件。

- `20260709_213505`：`samples_1~200.json` 中 52 条样本，日志最新样本约 sample 52。
- `20260709_213507`：`samples_1~200.json` 中 47 条样本，日志最新样本约 sample 47。
- `20260709_213510`：`samples_1~200.json` 中 52 条样本，日志最新样本约 sample 52。

本次复查未发现 `Traceback`、`NameError`、`Timeout`、连接错误、4xx/5xx 或 `sample_error`。当前三路 MCTS-AHD 运行未卡死。

## 自动监控 (/loop) — 2026-07-09 22:28 起

已设置 /loop 每 30 分钟自动复查（cron `*/30 * * * *`，session-only，job id `675fe539`，7 天后自动过期，可用 CronDelete 提前取消）。复查逻辑：确认 3 个 tmux session 与 python 子进程存活 → 看 `logs/mcts_events.jsonl`、`tmux_run.log` 的 mtime 是否持续更新判定是否卡住 → curl endpoint 确认 vLLM 可达 → 若某路卡住按下方 runbook 重启 → 若三路均到 `samples=1000` 则启动 pathwise 三重复。

> 说明：用户原话是「每 45 分钟」，但 45 不能整除 60，cron 的 `*/45` 会给出 45min/15min 交替的不均匀间隔；最近可整除间隔 30 与 60 等距，取 **30 分钟**以更快发现卡住。需要改成 60 分钟可随时说。

22:28 快照（捕获时点，仍在推进）：三路均健康迭代，endpoint `http://222.201.145.8:8080/v1/models` 返回 200。

| run directory | samples | best score | tmux_run.log mtime |
|---|---:|---:|---|
| `20260709_213505` | 124/1000 | `-6.638237231415435` | 22:28:24 |
| `20260709_213507` | ~117/1000 | `-6.377014109814672` | 22:28:22 |
| `20260709_213510` | ~129/1000 | `-6.619615444190888` | 22:28:17 |

按 ~54 分钟跑了约 124 sample 估算，三路跑满 1000 约需 6–8 小时。

### 卡住判定与重启 runbook

- tmux 面板只显示一行 `run_dir=`（`run_experiment.py` 把 stdout/stderr 重定向到了 `tmux_run.log`），进度必须看 `logs/`。
- 判定卡住：`logs/mcts_events.jsonl` 与 `tmux_run.log` 的 mtime 超过 ~15 分钟无更新，且 python 进程仍在（典型为 LLM 请求挂起）；或 python 进程已退出但 sample 未到 1000。
- 重启命令（**必须带 `NO_PROXY`**，否则对 `222.201.145.8` 的请求可能走代理而挂起，这正是「请求卡住」的常见来源）：

```bash
cd /home/fang/code/LLM4AD/LLM4AD
NO_PROXY=222.201.145.8,localhost,127.0.0.1,::1 uv run python experiments/tsp_construct/mcts_ahd/run_experiment.py
```

- 注意：`run_experiment.py` 用 `exist_ok=False` 在导入时固定时间戳目录，**不支持原地续跑**；重启等于开一个新 run（会损失已跑的 sample 进度）。method 目录里有 `resume.py` 但未接入此入口；若需要断点续跑需另行改造。

### pathwise 入口（已就绪）

已新建 `LLM4AD/experiments/tsp_construct/pathwise/run_experiment.py`，镜像 mcts_ahd runner：复用同一 vLLM endpoint（`qwen3.6-27b-awq` @ `http://222.201.145.8:8080/v1`）、同一 `tsp_construct` train 任务、`VLLMOpenAIAPI` 客户端、`NO_PROXY`、3 个 tmux 各跑一个 run。PathWise 参数取 `llm4ad/method/pathwise/paras.yaml` 默认：`max_sample_nums=500, pop_size=6, init_pop_size=None(→30), num_actions=2, num_rollouts=2, max_inner_steps=3, num_evaluators=4`，扰动概率 0.5→0.25。冒烟测试通过（`py_compile` + 以 `profiler=None` 构造 `PathWise` 成功，evolve fn=`select_next_node`，未发请求/未建目录）。

mcts 三路全部到 `samples=1000` 后，启动 pathwise 三重复（每个 tmux 一个 run，命令一致）：

```bash
cd /home/fang/code/LLM4AD/LLM4AD
NO_PROXY=222.201.145.8,localhost,127.0.0.1,::1 uv run python experiments/tsp_construct/pathwise/run_experiment.py
```

建议 tmux session 命名：`pathwise_tsp_repeat_{1,2,3}_<时间戳>`。注意：pathwise 单样本需多次串行 LLM 调用（policy + world model + policy/world critic），单样本比 mcts_ahd 慢，500 预算下总时长可能更长。

## 完成状态与论文汇报口径 — 2026-07-10

2026-07-10 复查：最新三次独立 MCTS-AHD run 均已完成到 `samples=1000`，没有搜索中止或错误计数；原 mcts tmux session 已退出，后续 pathwise 三重复已启动。

| run directory | status | finished_at | samples | eval success / fail | best sample | train best score |
|---|---|---|---:|---:|---:|---:|
| `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_213505` | `finished` | `2026-07-10T11:43:39+08:00` | 1000 | 958 / 42 | 960 | `-6.245046936508112` |
| `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_213507` | `finished` | `2026-07-10T11:02:00+08:00` | 1000 | 974 / 26 | 816 | `-6.291167317523522` |
| `LLM4AD/experiments/tsp_construct/mcts_ahd/20260709_213510` | `finished` | `2026-07-10T11:30:25+08:00` | 1000 | 933 / 67 | 736 | `-6.091060868244764` |

三个 run 的 `run_summary.json` 均为 `error_count=0`、`search_aborted=false`；`samples_1~200.json` 到 `samples_801~1000.json` 各分片均为 200 条样本。

论文 `papers/MCTS-AHD/icml2025.tex` 对 step-by-step construction 的 TSP/KP 主表说明：每个 LLM-based AHD 方法独立运行三次，并汇报平均表现；表中同时列 test objective 和 optimality gap，主表不汇报方差或标准差。实验设置中也说明几乎所有任务使用 `T=1000`，每个 application scenario 做 3 个 independent runs 以降低统计偏差。附录的 p-value 表是额外显著性分析，最多 10 runs，才单独列 `avg` 和 `std`；TSPLib 附录是例外，先从三次设计 run 中选 best-performing constructive heuristic，再对不同起点运行三次取平均。

因此，若按 MCTS-AHD 主表口径汇报本次 `tsp_construct` 三重复，不应只报三次中的最好 run，也不需要在主表中报 std；应把每个 run 找到的 best heuristic 拿到同一批 test sets 上评估，然后对 3 个 run 的 test objective / gap 取平均。当前上表的 `train best score` 只能作为 run 完成后的搜索结果快照；若只对 train split 临时汇总，三次对应的正向平均 tour length 为 `6.209091707425`，sample std 为 `0.104786600453`，但这不是论文主表的正式 test-set 口径。

## Best heuristic eval/test score — 2026-07-10

按上面的论文主表口径，取三个 MCTS-AHD 独立 run 各自最终 `samples_best.json` 中的 best program，在同一个 `tsp_construct` eval split 上重新评估。评估命令：

```bash
cd /home/fang/code/LLM4AD/LLM4AD
uv run python experiments/tsp_construct/mcts_ahd/evaluate_best_on_eval.py
```

eval/test split 配置来自 `get_generated_task_kwargs("tsp_construct", "eval")`，并对三个 problem size 使用完整评估结果：

- `n_instance=16`
- `problem_size=50`
- `seed=2025`
- `problem_size in {50, 100, 200}`

`TSPEvaluation` 的 score 是负平均 tour length，因此 score 越高越好；下面同时记录正向 objective/tour length，objective 越低越好。

| source run | best sample | operator | train score recomputed | eval/test score | eval/test objective |
|---|---:|---|---:|---:|---:|
| `20260709_213505` | 960 | `e2` | `-6.245046936508112` | `-6.396046074894754` | `6.396046074894754` |
| `20260709_213507` | 816 | `s1` | `-6.291167317523522` | `-6.387044402353435` | `6.387044402353435` |
| `20260709_213510` | 736 | `s1` | `-6.091060868244764` | `-6.171379941595068` | `6.171379941595068` |

汇总结果：

- 平均 eval/test score：`-6.318156806281086`
- 平均 eval/test objective：`6.318156806281086`
- eval/test objective sample std：`0.1271921520080091`（辅助记录；论文主表通常不汇报 std）

sanity check：三个 best program 在 train split 上的重算分数与搜索 artifact 中记录的 best score 完全一致，说明评估脚本确实取到了对应 best heuristic。

结果 artifact：

- 评估脚本：`LLM4AD/experiments/tsp_construct/mcts_ahd/evaluate_best_on_eval.py`
- JSON 结果：`LLM4AD/experiments/tsp_construct/mcts_ahd/eval_best_qwen36_27b_20260710/results.json`
- best program 快照：`LLM4AD/experiments/tsp_construct/mcts_ahd/eval_best_qwen36_27b_20260710/*_program.py`

这个分数可作为当前 MCTS-AHD + `qwen3.6-27b-awq` 在平台 `tsp_construct` eval split 上的三重复平均结果。

更新：正式结果已整理到 `docs/results/mcts-ahd-qwen36-27b-tsp-construct.md`，其中包含 TSP50、TSP100、TSP200 三个测试规模。最终统计采用三个独立 run 的 best heuristic 在测试集上的完整运行结果，记录各次实际评估时间、score、objective 及三次平均表现。TSP200 三个 objective 的三次平均为 `12.234488353074918`，第三个 best heuristic 的完整评估耗时约 `121.5s`，已纳入统计。

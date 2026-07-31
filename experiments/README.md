# Experiments

实验入口按 task 和 method 组织，运行工件仍写入对应方法目录：

```text
experiments/<task>/<method>/<version>/<run_name>/
  run_config.json
  tmux_run.log
  logs/
```

`run_config.json` 是每个 run 自动生成的参数快照，不需要手写。三个重复必须拥有
独立目录，才能分别恢复、检查和测试。

这些运行目录只保存在实验机器本地，不进入 Git。Git 只跟踪本目录中的实验入口、
评估脚本和绘图脚本；完成实验后，将三次重复的聚合结果和必要曲线整理到
`docs/results/<task>/`。不要单独提交 `run_config.json`、`results.json`、日志、
checkpoint 或生成程序。

## TraceAAD

TraceAAD V4/V5 共用参数化入口，不再为每个 task 和版本复制
`run_experiment.py`。

单次运行：

```bash
uv run python -m experiments.traceaad.run \
  --task online_bin_packing \
  --version v4 \
  --backend local \
  --run-name 20260729_obp_v4_rep1 \
  --repeat 1
```

三重复 tmux 批量启动：

```bash
uv run python -m experiments.traceaad.launch \
  --task online_bin_packing \
  --version v4 \
  --backend local \
  --repeats 3
```

支持的 backend 为 `local`、`server1` 和 `zhong`。可以用 `--base-url`、
`--model`、`--no-proxy` 覆盖 profile；API key 只从环境或仓库 `.env`
读取，不作为命令行参数。

恢复单个运行：

```bash
uv run python -m experiments.traceaad.run \
  --task online_bin_packing \
  --version v4 \
  --backend local \
  --resume-from experiments/online_bin_packing/traceaad_v4/version4/<run_name>
```

其它方法目前仍使用各 task/method 目录中的原生入口。原始运行工件仅保留在本地，
不进入 Git。

## EoH

EoH 使用统一参数化入口。默认运行 20 代、评估预算 1000；OBP 使用种群 20，
其余三个 task 使用种群 10。算子保留原始
发布代码实际启用的 `e1/e2/m1/m2`，不启用 `m3`。

单次运行：

```bash
uv run python -m experiments.eoh.run \
  --task tsp_construct \
  --backend zhong \
  --repeat 1 \
  --seed 0
```

四任务各重复三次，并在 Zhong/server1 各分配六路：

```bash
uv run python -m experiments.eoh.launch
```

## ReEvo

ReEvo 默认使用公平比较预算 `max_sample_nums=1000`，其余 `pop_size=10`、`init_pop_size=30`、
`mutation_rate=0.5`、`temperature=1`（初始化 `+0.3`）。

单次运行：

```bash
uv run python -m experiments.reevo.run \
  --task tsp_construct \
  --backend zhong \
  --repeat 1 \
  --seed 0
```

## ShinkaEvolve

ShinkaEvolve 机制参数沿用 Circle Packing 主表；默认预算
`max_sample_nums=num_generations=1000`。novelty 关闭；
meta LLM 复用同一 Qwen 端点。

## PathWise

PathWise 默认预算 `max_sample_nums=1000`，`pop_size=6`、
4 evaluators。

## 公平预算统一调度（推荐）

PathWise + ReEvo + ShinkaEvolve 共 36 路（4 task × 3 rep × 3 method），
预算均为 1000，按空闲槽位自动补位：

```bash
uv run python -m experiments.fair1000.launch --watch
```

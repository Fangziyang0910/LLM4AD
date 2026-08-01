# Experiments

目录只保留两类内容：

1. **可复用入口**（`runners/`、`plotting/`、各 task 的评估脚本）
2. **原始实验结果**（`experiments/<task>/<method>/...`，本地工件，默认不入库）

```text
experiments/
  runners/                         # 跨 task 的统一 run / launch 入口
    traceaad/  eoh/  pathwise/  reevo/  shinka_evo/  fair1000/
  plotting/                        # 正式搜索曲线
  <task>/
    evaluate_best_on_test.py       # task 级测试评估
    mcts_ahd/run_experiment.py     # 尚未统一的方法入口（仅 mcts_ahd）
    <method>/<run_or_version>/...  # 本地运行工件与测试评估结果
```

运行工件布局：

```text
experiments/<task>/<method>/<version>/<run_name>/
  run_config.json
  tmux_run.log
  logs/
```

`run_config.json` 由入口自动生成。三次重复必须各有独立目录。运行工件只保存在实验机器本地；Git 只跟踪入口、评估与绘图脚本。权威结果整理到 `docs/results/<task>/`。

## TraceAAD

```bash
uv run python -m experiments.runners.traceaad.run \
  --task online_bin_packing \
  --version v4 \
  --backend local \
  --run-name 20260729_obp_v4_rep1 \
  --repeat 1

uv run python -m experiments.runners.traceaad.launch \
  --task online_bin_packing \
  --version v4 \
  --backend local \
  --repeats 3
```

backend 为 `local`、`server1`、`zhong`。可用 `--base-url` / `--model` / `--no-proxy` 覆盖；API key 只从环境或仓库 `.env` 读取。

## EoH / ReEvo / PathWise / ShinkaEvolve

```bash
uv run python -m experiments.runners.eoh.run --task tsp_construct --backend zhong --repeat 1 --seed 0
uv run python -m experiments.runners.reevo.run --task tsp_construct --backend zhong --repeat 1 --seed 0
uv run python -m experiments.runners.pathwise.run --task tsp_construct --backend zhong --repeat 1 --seed 0
uv run python -m experiments.runners.shinka_evo.run --task tsp_construct --backend zhong --repeat 1 --seed 0
```

公平预算统一调度（PathWise + ReEvo + ShinkaEvolve，4 task × 3 rep）：

```bash
uv run python -m experiments.runners.fair1000.launch --watch
```

## MCTS-AHD

暂无各 task 目录中的 `mcts_ahd/run_experiment.py`，尚未收入 `runners/`。

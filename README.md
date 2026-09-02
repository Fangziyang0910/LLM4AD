# LLM4AD — Active Algorithm Development Platform

本仓库是 [LLM4AD](https://github.com/Optima-CityU/llm4ad) 的分支，作为我们课题组的 **active platform**，用于统一运行和对比各类 LLM 驱动的自动启发式设计方法。当前主线方法为 **TraceAAD**（V9），对照方法包括 MCTS-AHD / PathWise / EoH / ReEvo 等，任务为组合优化（TSP / CVRP / OP / OBP）。

> 本仓库的协作规则见仓库根目录 `AGENTS.md`。相关论文位于上级目录 `../papers/`，原始参考代码位于 `../reference_code/`，两者默认只读。

## 目录结构

```
LLM4AD/
├── experiments/      # runners/ 可复用入口；<task>/<method>/ 本地结果
│   ├── runners/                   # TraceAAD / EoH / PathWise / ReEvo / ...
│   ├── plotting/                  # 正式搜索曲线
│   └── <task>/<method>/           # 评估脚本与本地运行工件
├── llm4ad/           # 平台核心
│   ├── method/       # mcts_ahd / pathwise / traceaad / eoh / reevo / meoh ...
│   ├── task/         # tsp / cvrp / orienteering / knapsack / bp / jssp ...
│   ├── tools/        # LLM 客户端、安全执行器等
│   └── base/         # Evaluation / Function 等基类
├── docs/             # 轻量科研记录（ideas / experiments / results / worklog）
├── tests/            # method / task 机制单测
└── pyproject.toml    # uv 管理依赖
```

## 运行实验

每个实验一个独立入口包，运行目录与工件布局见各实验 runner：

```bash
uv run python -m experiments.runners.traceaad_v9_16.run \
uv run python -m experiments.traceaad_v10_1.run \
  --task tsp_construct --backend local
```

批次发射器为各包内的 `launch.py`（自动建立独立 tmux 会话与 run 名称，如
`experiments.runners.traceaad_v9_16.launch`）。每次运行在对应
`experiments.traceaad_v10_1.launch`）。每次运行在对应
`experiments/<task>/<method>/<run_name>/` 下保存 `run_config.json`、
`tmux_run.log` 和 `logs/`，不需要手写配置或批次脚本。这些原始工件只保存在
实验机器本地，不进入 Git。其它方法入口见 `experiments/runners/`（EoH / ReEvo /
PathWise / ShinkaEvolve / CALM），统一预算为 1000 次搜索评估。
实验机器本地，不进入 Git。对照基线入口见 `experiments/`（EoH / ReEvo /
PathWise / ShinkaEvo / CALM），统一预算为 1000 次搜索评估。

完整参数见 `python -m experiments.runners.traceaad_v9_16.run --help`。
完整参数见 `python -m experiments.traceaad_v10_1.run --help`。

## 当前实验矩阵

定稿结果见 [`docs/experiments/主实验/结果.md`](docs/experiments/主实验/结果.md)，协议见 [`docs/experiments/主实验/配置.md`](docs/experiments/主实验/配置.md)。运行过程见当周 `docs/worklog/YYYY-Www.md`。

## 配置要点

- **LLM**：通用 OpenAI-compatible 客户端（`OpenAIAPI`）；TraceAAD 用
  `--backend` 或显式 URL/model 参数配置服务，API key 只从环境读取
- **task 数据**：`llm4ad/task/optimization/generated_data_config.py`（按 task 注册 train/eval 的 `problem_size`、`n_instance`、`seed` 等）
- **method 与 task 解耦**：method 从 evaluation 对象读取 `template_program` / `task_description` 构造所有 prompt，换 task 通常只需换 evaluation 实例
- **task 默认参数未必对齐论文**：新 task 上线前应对照 `../papers/` 核对设置，并明确记录有意采用的差异

## 测试

```bash
uv run pytest tests/ -q
```

## 致谢

基于 [Optima-CityU/LLM4AD](https://github.com/Optima-CityU/llm4ad)（BSD License）。

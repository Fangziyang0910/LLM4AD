# LLM4AD — Active Algorithm Development Platform

本仓库是 [LLM4AD](https://github.com/Optima-CityU/llm4ad) 的分支，作为我们课题组的 **active platform**，用于统一运行和对比各类 LLM 驱动的自动启发式设计方法。当前重点对比 **MCTS-AHD / PathWise / TraceAAD** 在组合优化（routing 类等）任务上的表现。

> 本仓库的协作规则见仓库根目录 `AGENTS.md`。相关论文位于上级目录 `../papers/`，原始参考代码位于 `../reference_code/`，两者默认只读。

## 目录结构

```
LLM4AD/
├── experiments/      # 各 task × method 的实验入口；运行工件仅保存在本地
│   ├── traceaad/                  # TraceAAD V4/V5 统一参数化入口
│   └── <task>/<method>/           # task 评估入口与本地运行目录
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

TraceAAD 使用统一参数化入口：

```bash
uv run python -m experiments.traceaad.run \
  --task tsp_construct --version v5 --backend local

uv run python -m experiments.traceaad.launch \
  --task tsp_construct --version v5 --backend local --repeats 3
```

`launch` 自动建立独立 tmux 与 run 名称。每次运行仍在对应
`experiments/<task>/traceaad_<version>/<version>/` 下保存 `run_config.json`、
`tmux_run.log` 和 `logs/`，不需要手写配置或批次脚本。这些原始工件只保存在
实验机器本地，不进入 Git。其它方法保留各自的原生入口。完整参数见
`python -m experiments.traceaad.run --help`，布局细则见
`experiments/README.md`。

## 当前实验矩阵

当前覆盖统一见 `docs/实验覆盖.md`。定稿结果见
`docs/results/<task>/结果汇总.md`，运行过程和研究记录见当周
`docs/worklog/YYYY-Www.md`。

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

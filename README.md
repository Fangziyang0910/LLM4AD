# LLM4AD — Active Algorithm Development Platform

本仓库是 [LLM4AD](https://github.com/Optima-CityU/llm4ad) 的分支，作为我们课题组的 **active platform**，用于统一运行和对比各类 LLM 驱动的自动启发式设计方法。当前重点对比 **MCTS-AHD / PathWise / TraceAAD** 在组合优化（routing 类等）任务上的表现。

> 本仓库的协作规则见仓库根目录 `AGENTS.md`。相关论文位于上级目录 `../papers/`，原始参考代码位于 `../reference_code/`，两者默认只读。

## 目录结构

```
LLM4AD/
├── experiments/      # 各 task × method 的实验入口与运行 artifact
│   └── <task>/<method>/
│       ├── run_experiment.py      # 入口脚本（模型与超参写死在顶部）
│       ├── evaluate_*on_test.py   # best 启发式在测试规模上的评估
│       └── <timestamp>/           # 每次 run 的 artifact（自动生成）
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

每个实验入口是 `experiments/<task>/<method>/run_experiment.py`，模型与超参写死在脚本顶部。

```bash
# 前台单跑（调试 / 冒烟）
uv run python experiments/<task>/<method>/run_experiment.py
```

长跑用 tmux 后台（必须用绝对路径 `/usr/local/bin/tmux`，并带 `NO_PROXY` 绕过代理访问 LLM endpoint）：

```bash
TS=$(date +%Y%m%d_%H%M%S)
/usr/local/bin/tmux new -d -s <method>_<task>_$TS \
  "cd $(pwd) && NO_PROXY=<endpoint>,localhost,127.0.0.1,::1 \
   uv run python experiments/<task>/<method>/run_experiment.py"
```

每次运行会在脚本目录下生成 `<timestamp>/`，含 `run_config.json`、`tmux_run.log`、`logs/`（`run_log.txt`、`llm_calls.jsonl` 等）。论文主实验通常跑 **3 个独立 repeat** 并行；多个 run 同时启动时彼此错开约 5 秒，避免秒级时间戳目录冲突。布局细则见 `experiments/README.md`。

## 当前实验矩阵

| task ＼ method | mcts_ahd | pathwise | traceaad |
|---|:---:|:---:|:---:|
| tsp_construct | ✅ | ✅ | ✅ |
| cvrp_aco | ✅ | ✅ | ✅ |

✅ 完成　🔄 进行中　— 未开始。定稿结果见 `docs/results/<task>/结果汇总.md`，运行过程和研究记录见当周 `docs/worklog/YYYY-Www.md`。

## 配置要点

- **LLM**：通用 OpenAI-compatible 客户端（`OpenAIAPI`）；实验入口从 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` 读取连接配置
- **task 数据**：`llm4ad/task/optimization/generated_data_config.py`（按 task 注册 train/eval 的 `problem_size`、`n_instance`、`seed` 等）
- **method 与 task 解耦**：method 从 evaluation 对象读取 `template_program` / `task_description` 构造所有 prompt，换 task 通常只需换 evaluation 实例
- **task 默认参数未必对齐论文**：新 task 上线前应对照 `../papers/` 核对设置，并明确记录有意采用的差异

## 测试

```bash
uv run pytest tests/ -q
```

## 致谢

基于 [Optima-CityU/LLM4AD](https://github.com/Optima-CityU/llm4ad)（BSD License）。

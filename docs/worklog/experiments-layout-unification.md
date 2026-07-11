# Experiments layout unification

2026-07-09

本次把 LLM4AD 平台实验目录规范调整为 `experiments/<task>/<method>/<timestamp>/`，用于先按 task 聚合，再在 task 内按 method 对比，单次 run 以时间戳目录保存结果和日志。

按后续修正，启动入口不是一个大 CLI，而是每个 task/method 一份最小脚本。当前 TSP Construct 有：

- `LLM4AD/experiments/tsp_construct/mcts_ahd/run_experiment.py`
- `LLM4AD/experiments/tsp_construct/traceaad/run_experiment.py`

这些脚本不使用 `argparse`，参数直接写在文件顶部常量里。运行时自动创建当前时间戳目录，写入 `run_config.json`、`tmux_run.log` 和 profiler `logs/`。

旧实验结果做目录迁移后，保留 `logs/`、备份日志和 samples；原来的分散 timestamp 内 `run_*.py` 启动脚本已替换为每个时间戳目录下的 `run_config.json`，作为历史启动参数快照。未来启动入口使用对应 task/method 目录下的 `run_experiment.py`。

## 2026-07-09 LLM 参数统一

当前两个 TSP Construct 实验脚本的 LLM 调用参数统一为：`LLM_TIMEOUT=600`、`LLM_TEMPERATURE=1.0`、`MAX_TOKENS=16384`。这样后续 `mcts_ahd` 与 `traceaad` 的 vLLM 请求超时预算一致，避免因长 prompt / 长生成导致客户端过早 timeout。

## 2026-07-09 清理 MCTS-AHD 历史结果

按要求删除了 `LLM4AD/experiments/tsp_construct/mcts_ahd/` 下的历史结果目录 `20260706_203308/` 和 `20260707_162652/`，仅保留该 method 的未来启动脚本 `run_experiment.py`。其他方法的实验结果未清理。

## 2026-07-09 恢复 optimization task 目录

按要求将 `LLM4AD/llm4ad/task/optimization/` 从 `main/other/cobench` 分类结构恢复为 `main` 分支原始平铺结构，并把 `cobench` 命名恢复为原来的 `co_bench`。新增 optimization task 代码和对应测试已从工作树移除；已生成的数据先备份到 `/home/fang/code/LLM4AD/task_data_backup_20260709_142302/`，其中 main 原有 task 的数据已迁回各自原始 task 目录。`.gitignore` 现在忽略 `llm4ad/task/optimization/*/data/`，避免本地数据进入提交。

## 2026-07-09 generated data 复现配置

对 18 个非 CO-Bench optimization task 做了固定配置复现验证：同一 task/split 的 evaluator 连续实例化两次，生成数据指纹一致；除 `admissible_set` 这类确定性任务外，`train` 与 `eval` 使用不同 seed 并生成不同指纹。统一入口为 `llm4ad/task/optimization/generated_data_config.py` 的 `get_generated_task_kwargs(task_name, split)`。

非 CO-Bench 本地固定 `data/` 目录已删除；当前仅保留 `llm4ad/task/optimization/co_bench/data/`。后续实验脚本应通过统一配置传入 `seed`、规模和 timeout，不再依赖本地保存的生成式数据文件。

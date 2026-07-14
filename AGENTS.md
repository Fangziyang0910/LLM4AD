# LLM4AD Repository Instructions

1. 当前目录 `LLM4AD/` 是 active algorithm development platform，后续优先在本仓库中统一运行和对比各类机制算法。
2. `../TraceAAD/` 是正在设计和实现的 TraceAAD 方法机制。
3. `../papers/` 中放相关方法的原始论文，机制判断以论文为最高优先级。
4. `../reference_code/` 中放相关方法的原始代码实现，例如 `../reference_code/MCTS-AHD-master/`。这些代码默认只读参考，不作为日常运行入口。
5. 上级目录中尚未归档到 `../reference_code/` 的相关工作目录，也按只读参考代码处理。
6. 科研协作重点是理清机制思路与实现细节、复现实验、改进方法和设计新的创新机制。
7. 当前重点是基于轨迹的搜索引导机制 TraceAAD，以及 MCTS-AHD / PathWise 等方法在组合优化任务上的统一比较。
8. 运行代码使用 uv 和本仓库自己的虚拟环境。
9. `docs/` 是本项目的轻量科研记录系统，不要求固定格式，也不要求按日期划分。
10. `docs/ideas/` 记录 idea、机制思考、论文启发、假设、问题和可能的改进方向。
11. `docs/experiments/` 记录实验设置、实验过程、artifact 路径、现象和阶段性结论，可以是不完整的、运行中的记录。
12. `docs/results/` 记录每个「方法 × 模型 × 任务」组合的最终/权威测试结果汇总。内容包含实验配置、各重复 run 的 artifact 路径、各测试规模的测试分、多次重复的 mean±std、评估脚本与命令，以及可选的搜索演化曲线图。只有所有重复和测试评估完成后，才更新该目录。
13. `docs/worklog/` 记录科研过程中的工作内容，例如集成 task、适配实现、跑实验、发现规律和形成判断。
14. 当出现新的机制理解、实验发现、异常现象、重要实现改动或设计判断时，主动轻量地记录到 `docs/` 的合适位置。

## 运行与实验工作流

15. `experiments/` 的布局是 `experiments/<task>/<method>/run_experiment.py`。每个 task×method 组合一个入口脚本，模型与超参写在脚本顶部；每次运行在脚本目录下生成 `<timestamp>/`，含 `run_config.json`、`tmux_run.log` 和 `logs/`。
16. 运行实验统一用 `uv run python experiments/<task>/<method>/run_experiment.py`，前台调试或使用 tmux 后台长跑。tmux 启动命令必须带 `NO_PROXY=<endpoint_ip>,localhost,127.0.0.1,::1` 绕过代理访问 vLLM endpoint。
17. 论文主实验通常对每个 task×method 跑 3 个独立 repeat 并行，启动时错开约 5 秒，避免 timestamp 冲突。先单跑一个做冒烟，确认 endpoint 返回 200、evaluate 产生有效分数、best 会随 sample 推进，再补启其余两个。
17a. 测试集评估必须最终完成，不能因单个 heuristic 的固定 timeout 到期就放弃并把 `n/a` 写入最终结果。评估脚本应允许关闭 timeout，并优先对独立的 run×测试规模增加并行 worker；只有全部测试结果拿到后，才更新 `docs/results/`。
18. `llm4ad/method/*` 从 evaluation 对象读取 `template_program` / `task_description` 构造 prompt，因此换 task 通常只需新建 runner 并替换 evaluation 实例，method 零改动。
19. task 默认参数未必对齐论文标准。新 task 上线前应对照 `../papers/` 核对设置，并用 default 启发式和贪心启发式确认评估有区分度。OP 已按 ReEvo/DeepACO 标准修正 budget/prize，`run_config.json` 中可能残留不反映真实运行值的 legacy 字段。
20. 结果定稿写入 `docs/results/<method>-<model>-<task>.md`；论文和原始代码参考路径分别是 `../papers/` 和 `../reference_code/`。

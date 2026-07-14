# MCTS-AHD Orienteering Construct 3-run launch

日期：2026-07-13
分类：experiment

> ⚠️ **更新（2026-07-13 12:54）**：早期用默认 `max_length_ratio=0.35` 启动的 3 个 run（`20260713_111009 / 111935 / 111940`）**已作废并删除**（原 ~26M artifact 已于 2026-07-13 清理）。该配置下 budget=17.5 使 OP 退化为无约束、best 卡在 prize 上界 27.43、无区分度。已修正 budget/prize 对齐 ReEvo/DeepACO 标准，重启 3 个有效 run（`20260713_125413 / 125707 / 125712`）。详见下方「问题发现与修正」。

## 任务选型：为什么是 Orienteering Problem

调研 `papers/` 中 ReEvo / EoH / CO-Bench 三篇代表性论文的实验任务，TSP/CVRP 之外高频出现的候选有：Bin Packing、Knapsack、Flow Shop、Orienteering、JSSP、Max Cut、QAP。

选定 **Orienteering Problem (OP)**：① ReEvo (NeurIPS 2024) 的 6 大 benchmark 之一，routing 类标准任务；② 与 TSP/CVRP construct 范式同构（逐个选下一节点），routing 三件套补齐；③ 迁移成本最低（method 与 task 解耦）。

## 集成状态

OP 在 active platform 中完整集成，MCTS-AHD 零改动即可适配。`MCTS_AHD` 在 `mcts_ahd.py:94-95` 从 evaluation 对象读取 `template_program`/`task_description` 构造所有 prompt，method 与 task 完全解耦。新建 `experiments/orienteering_construct/mcts_ahd/run_experiment.py` 相对 tsp 模板仅改 TASK/import/实例化 3 处。

## 问题发现与修正（关键）

### 现象
默认配置启动后，三个 run 的 best 从 sample 6 起完全卡死在 `27.429235509812024`，到 sample 132 仍无变化，且三个 run 完全一致。

### 根因（数据证实）
`best == sum(prizes) 的均值（27.429236）`，即启发式收完了几乎所有 prize。LLM4AD 官方 OP 实现 `max_length_ratio=0.35` → budget = 0.35×50 = **17.5**，远超单位正方形内 TSP50 最优 tour（≈5.5），约束永不触发 → OP 退化为无约束收集 → 丧失区分度。同时 prize 用 `Uniform[0.1,1.0]`，与 ReEvo/DeepACO 标准（OP50 budget=3、Kool2019 离散 prize）不一致，无法对比。

### 修正
改 `llm4ad/task/optimization/orienteering_construct/get_instance.py`：
- **budget**：按 problem_size 走 ReEvo/DeepACO 标准分档 `{50:3, 100:4, 200:5, 500:8, 1000:12}`，非标准 size 保留 `max_length_ratio×size` fallback。
- **prize**：从 `Uniform[0.1,1.0]` 改为 Kool2019 离散分布 `p_i=(1+⌊99·d_{0i}/max_j d_{0j}⌋)/100`，depot prize=0。

### 验证（修正后区分度恢复）
| 启发式 | score | 占上界(29.01) |
|---|---|---|
| default（选首个 feasible）| 3.71 | 12.8% |
| ratio（prize/距离贪心）| 14.07 | 48.5% |

区分度 10.36（修正前为 0）。budget=3 真正卡住路径，搜索恢复有效。

## 有效 run（v2，budget=3）

启动设置：task `orienteering_construct` (split train, problem_size=50, budget=3)，method `mcts_ahd`，model `qwen3.6-27b-awq`，`max_sample_nums=1000, init_size=4, pop_size=10, selection_num=2, num_samplers=4, num_evaluators=4, alpha=0.5, lambda_0=0.1, eval_executor=thread`，其余同 tsp 主实验。

| repeat | tmux session | run directory |
|---:|---|---|
| 1 | `mcts_ahd_op_v2_r1_20260713_125411` | `experiments/orienteering_construct/mcts_ahd/20260713_125413` |
| 2 | `mcts_ahd_op_v2_r2_20260713_125705` | `experiments/orienteering_construct/mcts_ahd/20260713_125707` |
| 3 | `mcts_ahd_op_v2_r3_20260713_125710` | `experiments/orienteering_construct/mcts_ahd/20260713_125712` |

健康检查（启动 ~5 min）：3 session 存活，0 errors，best 随 sample 推进上升（r1: 14.07→14.30，已突破 ratio 基线 14.07；r2/r3 在 init 阶段，init best 各不相同 13.14/14.09，随机性正常）。

## 后续

- 待 3 个 run 各自到 `samples=1000` 后，按 tsp/cvrp 流程做 best 启发式在测试规模上的评估，汇总写入 `docs/results/mcts-ahd-qwen36-27b-orienteering-construct.md`。
- OP 上铺 pathwise / traceaad 时复用同一（已修正）task 配置，各 method 目录新建 `run_experiment.py` 即可。
- ratio 启发式 score 14.07 可作为各方法的参考下界。

# TraceAAD 实现完成

2026-07-08

## 做了什么

按 [TraceAAD 机制设计](../ideas/traceaad-mechanism-design.md) 完整实现了 TraceAAD method，落地在 `LLM4AD/llm4ad/method/traceaad/`。这是"过程信息为一等公民"的融合搜索：三层记忆（Program/Trajectory/Pattern）+ 三回路（进化/蒸馏/反思）+ stepwise 泛化信用 + 因果叙事 context + 6 算子 portfolio + islands + 鲁棒对比反馈。

模块清单（14 个）：
- `schema.py` / `derivation_graph.py`：扩展数据结构（ProgramNode+runtime/complexity/robustness/mechanism_tag；Edge+operator/delta/outcome/gen_signal；Trajectory+island/value；ValueVec/EvalResult/Pattern）+ 单父 DAG（Program Memory）
- `trajectory_memory.py` / `pattern_memory.py`：Trajectory/Pattern Memory
- `similarity.py`：程序层/机制层/轨迹层多层相似度（token Jaccard，无外部 embedding 依赖）
- `credit.py`：stepwise path value + 泛化信号（跨轨迹 pattern + 步一致性近似）
- `value.py`：多维 ValueVec + trajectory-UCB 选择
- `islands.py` / `feedback.py`：islands migration + Elo 对比排名
- `operators/`：base + 6 算子（endpoint_refine / backtrack_branch / mechanism_crossover / distill_simplify / scale_transfer / novelty_jump）
- `portfolio.py`：bandit + 阶段感知（role-phase bonus + τ 衰减）
- `reflection.py`：蒸馏回路（机制泛化统计）+ 反思回路（best/worst → lesson/anti_pattern）
- `context.py`：因果叙事三段式 action prompt；`prompt.py`：initial/code
- `traceaad.py`：主类 + 主循环

## 验证

- `tests/method/test_traceaad_smoke.py`（mock LLM + FakeEvaluation）跑通：14 采样、14 轨迹、best_fitness 从 -10 → -9.3，portfolio 实际调度了 endpoint/crossover/simplify/scale_transfer（backtrack/novelty 在 mock 持续改进下不触发，符合 trigger 设计）。
- 平台顶层导入健康：`from llm4ad.method import TraceAAD` 可用。

## 关键决策

1. **以 `traceaad/` 作为唯一实现路径**：方法、测试、示例和实验入口使用同一命名，避免平行版本造成机制和结果归属混乱。
2. **泛化信用近似**：平台 task（TSP）只返回标量，无法做真·跨 instance 泛化；当前用「PatternMemory 跨轨迹机制泛化分 + 步内改进持续性 + endpoint robustness」近似，`EvalResult.fitness_vector` 字段已留，未来 task 提供 per-instance 时升级为真·跨 instance（无需改 method 主体）。
3. **不引入外部 embedding**：多层相似度用 token/mechanism/pattern 的 Jaccard，避免重依赖。

## 入口与下一步

- 真实实验入口：`example/methods/traceaad/run_traceaad_tsp.py`
  `uv run python example/methods/traceaad/run_traceaad_tsp.py`
- 下一步：先用小 `--max-sample-nums`（如 100）验证真实 LLM 链路，再放大并做多 seed 对照。

## 风险/待观察

- 真实 landscape 下各算子触发频率、islands/对比反馈的实际收益需实验验证。
- code 相似度用 token Jaccard 较粗，若多样性/novelty 行为不理想，可升级为 AST 或 embedding。

# TraceAAD 首次完整真实实验结果（TSP Construct）

2026-07-09

## 实验

- **Method**: TraceAAD（[机制设计](../ideas/traceaad-mechanism-design.md)实现），含用户运行中改进（novelty 改用 best_stagnation 触发 + 机制信用贯通 is_anti_pattern/mechanism_improve_rate + Operator.select_trajectory）。
- **Task**: TSP Construct（train split，process backend / 16 workers）。
- **LLM**: vLLM endpoint，qwen3.6-27b-awq，timeout=600s。
- **目录**: `LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/`
- **起止**: 2026-07-08 23:52 → 2026-07-09 11:51，历时 ~12 小时。

## 结果（run_summary.json）

| 指标 | 值 |
| --- | --- |
| status | **finished**（正常结束，非 aborted） |
| best_score | **-6.371328**（sample 318 / node 308 找到） |
| num_samples | 977（942 valid + 35 eval_failed） |
| n_trajectories | 1310 |
| n_edges | 620 |
| error_count | **0** |
| search_aborted | false |
| llm_call_count | 1322 |

**结论**：
1. **稳定性验证**：本次 run 跑完 977 samples、0 error，验证三层记忆、三回路、6 算子 portfolio、泛化信用和因果叙事 context 能在真实 LLM 链路中完整工作。
2. **机制验证（从日志）**：6 算子全部被调度（endpoint_refine / backtrack_branch / mechanism_crossover / distill_simplify / scale_transfer / novelty_jump）；三回路均触发（distill 产出 pattern、reflect 产出 lesson/anti_pattern、islands migrate）；novelty gate 也实际拒绝相似候选。

## 观察与改进方向

- **best 长平台期**：best 在 sample 318（-6.371）后停滞 338 轮至结束。-6.371 是当前 LLM(qwen3.6-27b)+算子组合下较硬的局部最优。
- **novelty 探索效率下降（已知缺陷）**：极长平台期里 novelty_jump 高频触发且 `_pick_family` 反复选 `adaptive_exponent`（因 PatternMemory 标它有 improve 记录），但作为全新起点重生成时质量很差（-7~-27）。差解被正常归档不损 best，但浪费样本。
  - 改进方向：novelty 在 stagnation 极高时轮换机制族 / 对长期无增益的 novelty 降 portfolio 权重 / _pick_family 加入"该族近期 fresh_start 成功率"惩罚。
- **速率波动**：1.5–2.0 samples/min，受 vLLM 负载与算子 context 长度影响。

## 产物路径

- 最终摘要：`logs/run_summary.json`
- best 程序：`logs/samples/samples_best.json`
- 事件流：`logs/method_events.jsonl`、`logs/method_state.jsonl`
- LLM 调用：`logs/llm_calls.jsonl`
- 旧 run 备份（对比用）：`logs_run1_crash_at_212/`

## 下一步

- 在固定 token budget 和多个 search seed 下重复 TraceAAD，比较配置消融。
- 换更强 LLM 看 best 能否突破 -6.371。
- 改进 novelty 的长平台期策略（见上）。

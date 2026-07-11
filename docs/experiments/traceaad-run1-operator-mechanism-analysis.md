# TraceAAD 首轮实验：算子与机制效力分析

2026-07-08

实验：`LLM4AD/experiments/tsp_construct/traceaad/20260708_203505/`（TSP construct，max_sample_nums=1000，运行至 ~sample 193/iter 102 时的中间数据分析）。

本分析基于 `logs/method_events.jsonl` + `logs/method_state.jsonl`，挖出每个算子与每个机制的真实收益，诊断当前机制在真实 landscape 上的缺陷，给出数据驱动的优化方案。结论将反哺 [TraceAAD 机制设计](../ideas/traceaad-mechanism-design.md)。

## 1. 算子收益矩阵（child_accepted）

| operator | n | improve | regress | plateau | imp% | meanΔ | best_child |
| --- | --- | --- | --- | --- | --- | --- | --- |
| novelty_jump | 75 | 0 | 0 | 0 | — | +0.000 | **-6.454** |
| distill_simplify | 45 | 19 | 18 | 8 | 42% | -0.674 | -6.987 |
| mechanism_crossover | 26 | 13 | 12 | 1 | 50% | -3.609 | -6.693 |
| endpoint_refine | 25 | 10 | 13 | 2 | 40% | -0.937 | -6.736 |
| scale_transfer | 17 | 4 | 10 | 3 | 24% | -4.964 | -6.603 |
| backtrack_branch | 2 | 0 | 2 | 0 | 0% | -0.293 | -6.987 |

## 2. best score 演进

```
sample   1: -6.987  by init
sample  67: -6.789  by novelty_jump
sample  89: -6.603  by scale_transfer
sample 105: -6.454  by novelty_jump
final:    -6.454
```

**关键事实：3 次 best 刷新全部来自探索/泛化类算子（novelty×2 + scale_transfer×1），深化/重组/简化/回溯类一次都没刷新。**

## 3. 机制效力（mechanism_tag）

```
adaptive_exponent   n=13  imp=77%  avg=-7.48   ← 最有效，却用得最少
randomization       n= 9  imp=89%  avg=-8.23   ← 第二有效，用得少
other               n=23  imp=39%  avg=-8.37
sparsified_candidate n=10 imp=30%  avg=-9.86
local_density       n=26  imp=19%  avg=-9.82
edge_contrast       n=18  imp=39%  avg=-11.34
row_normalize       n=66  imp= 0%  avg=-11.31  ← 完全无效，却用得最多（最大浪费）
generalize          n=17  imp=24%  avg=-13.11
hybrid_distance     n= 6  imp= 0%
nn_rank             n= 2  imp= 0%
```

## 4. 诊断（5 个真实缺陷）

1. **机制信用完全没贯通（最致命）。** distill 算出了 adaptive_exponent 77% improve，但该信号只进 context 的 patterns 块，没进 value/portfolio/trigger。搜索把 66 次预算砸在 0% improve 的 row_normalize 上，几乎不碰 77% improve 的 adaptive_exponent。
2. **novelty 的 credit 失真。** 新起点 delta=0 → credit=0，但贡献 2/3 的 best 刷新。真实价值高却 delta=0，只能靠 explore role-bonus 被选 → 占 43%，75 个里 73 个无效但偶尔命中 best。价值信号与选择信号脱钩。
3. **exploit 类算子净退步（meanΔ 全负）。** base 主要是 endpoint（46 次），但多为 novelty 起源的低质起点（row_normalize 类），深化自然退步。selection 没偏好高效机制 trajectory。
4. **backtrack 空转（1 次）。** 依赖"被选中 trajectory 最后一步 regress"，但 selection 偏好高潜力（改进中）trajectory → 永不满足。
5. **无机制级早停。** row_normalize 连续 66 次失败仍被生成；anti_pattern 存在但没接回 selection/operator。

## 5. 优化方案（数据驱动，反哺 fusion design）

| 缺陷 | 优化 |
| --- | --- |
| 机制信用失效 | 让 PatternMemory.improve-rate 贯通三处：① `V_generalization` 用 endpoint mechanism 的 improve rate 加权；② anti_pattern 早停（机制连续 N 次失败 → 标记 → novelty/crossover 避开 + 该族 trajectory 降权）；③ endpoint/simplify base 选择偏好高效机制 trajectory |
| novelty credit 失真 | ① trigger 改"best 连续 N 轮停滞"（取代永真的 unique_ratio）；② 目标族从"最罕见"改"高 improve 且探索不足"；③ portfolio 的 novelty gain 用产出 trajectory 的 endpoint quality 而非 delta=0 |
| backtrack 空转 | 独立选题：主动挑"endpoint 退步但内部前缀高 value"的 trajectory backtrack |
| exploit 净退步 | 机制信用贯通后 selection 自动偏好高效机制 trajectory；simplify trigger 收紧 |
| 参数 | role-bonus：explore 早期 +0.5→+0.2；novelty 衰减更快 |

核心：**把"机制效力"从被动统计量升级为驱动 value/selection/operator 的主动信号**——这是 fusion 设计"泛化信用 + 知识层"的初衷，当前实现没接通。

## 6. 仍有效的部分（保留）

- novelty 贡献了 best（探索有价值，只是效率低 + 信用失真）。
- crossover 50% improve（重组本身有效，regress 大需门控）。
- novelty gate 挡了 11 个重复、eval_failed 仅 12（基础设施健康）。
- distill_simplify 42% improve（simplify 方向对，只是 base 选错）。

## 7. 下一步

当前轮继续跑完并保留为历史参考；随后按 §5 实现优化后的 TraceAAD，比较 best 曲线与机制分配是否更均衡（adaptive_exponent 占比上升、row_normalize 早停）。

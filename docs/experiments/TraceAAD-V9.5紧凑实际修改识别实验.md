# TraceAAD V9.5 紧凑实际修改识别实验

> 状态：已完成。432/432 次生成和评价全部收口，无缺失、重复或解析失败。这是固定 anchor 的单步 generation-interface probe，不是完整搜索消融。

## 问题

在已经提供 `Source + Idea + Result + Fitness transition` 简洁历史的前提下，
增加确定性构造的 Compact Actual Change 是否能进一步改善 Qwen3.6-27B 的
下一步算法修改，还是只增加理解负担？

## 固定设计

- 完整复用第一轮的 72 个真实 anchor，包括 task、quality stratum、current code、fitness 与当时历史事件。
- 每个 anchor 在 A/B 下各 3 次，共 216 个 pair、432 次生成。anchor 是独立单位，3 次采样是 anchor 内重复。
- A 为 `Concise History`；B 为完全相同的 prompt 加 `Compact Actual Changes` block。
- Task、Current Fitness、Current Code、Concise History、instruction、output contract、temperature 和 pair sampling seed 全部一致。
- 每个 task × quality stratum 内的 A→B / B→A 顺序精确各 9 个。

## Compact Actual Change

每个历史事件的表达从已记录的 parent→child unified diff 确定性产生：

```text
Event i: Diff size: +A/-R lines.
Removed examples: `<code>` | `<code>`.
Added examples: `<code>` | `<code>`.
```

- 不用 LLM 摘要，不生成因果解释。
- 只取非空、非注释的改动代码行；删除和新增各最多两个代表。
- 多于两行时取首尾，对于给定 diff 输出唯一。
- 每个事件最多 520 字符；不放入 raw diff。
- 532 条历史事件中 528 条有 actual diff，4 条显式写为未记录可执行修改。

## 执行与评价

为避免第一轮 24 路本地 evaluator 放大 TSP timeout，本轮分为两个阶段：

1. 生成阶段使用 zhong / server3 / server3b 的空闲模型容量并行生成，不同时启动 evaluator。
2. 432 次生成全部落盘后，在本地用一个顺序 evaluator 进程按冻结 schedule 统一评价，任务内使用原生 worker 配置。

主指标与第一轮一致：valid rate、conditional Δq、parent improvement rate、
line-change ratio、absolute LOC change、prompt / response tokens 和 failure kind。先在
`anchor × condition` 内聚合三次采样，再计算 B−A 配对差。

## 结果

下表的 \(\Delta q=q_{child}-q_{parent}\) 只在 valid candidate 上计算；
数值是先在 anchor 内聚合三次采样后的配对统计。区间是对 anchor
重采样 10,000 次的 95% bootstrap interval。

| Task | Valid A → B | Conditional \(\Delta q\) A → B | B−A（95% interval） | Improve A → B | Change ratio A → B |
| --- | ---: | ---: | ---: | ---: | ---: |
| CVRP | 98.1% → 98.1% | -0.264 → -0.026 | **+0.237 [0.088, 0.409]** | 13.9% → 33.3% | 0.587 → 0.447 |
| OBP | 98.1% → 100.0% | -503.421 → -720.125 | -216.704 [-545.158, 50.371] | 0.0% → 1.9% | 0.740 → 0.695 |
| OP | 96.3% → 90.7% | -1.254 → -0.988 | +0.266 [-0.227, 0.777] | 0.0% → 3.7% | 0.693 → 0.583 |
| TSP | 72.2% → 75.9% | -0.970 → -0.850 | +0.120 [-1.285, 1.143] | 40.6% → 30.2% | 0.698 → 0.692 |

A 是 `Concise History`，B 是 `Concise History + Compact Actual Change`。
CVRP 是唯一同时出现明确 conditional \(\Delta q\) 和 improvement-rate 增益的任务；
18 个 anchor 中 13 个的 \(\Delta q\) 方向更好，且高、低质量层的配对区间均不跨 0。
OP 与 TSP 的总体 \(\Delta q\) 向好但不确定，并且存在质量层间反向；
OBP 的总体方向为负，其中中质量层有明显负信号。

跨任务不汇总原始 \(\Delta q\) 数值，因为四个 evaluator 的尺度不同。
在 70 个两条件都有 conditional \(\Delta q\) 的 anchor 中，B 方向更好
39 个，A 更好 29 个，2 个持平。这是方向性描述，不是跨任务总效应。

## 生成行为与成本

- B 在 CVRP、OBP 和 OP 上进一步降低 line-change ratio，配对区间不跨 0；TSP 基本不变。在 70 个可配对 anchor 中，54 个在 B 下的修改比例更小，16 个更大。
- 平均 prompt 增量为 CVRP +725、OBP +422、OP +541、TSP +627 tokens；各任务平均总输入均远低于 32,768-token 总上下文上限。
- 两个条件各有 19 个 invalid。A 为 13 timeout + 6 runtime error，B 为 11 timeout + 8 runtime error；没有请求失败或输出解析失败。
- 本轮生成与 evaluator 已分离，valid-rate 不再混入 24 路本地评价的 CPU 竞争。

## 当前判断

Compact Actual Change 不只增加 token：它稳定改变生成分布，并在 CVRP
上提供了明确的额外单步价值。同时，它的主要行为影响是让修改进一步收缩，
这种归纳偏置在 OP/TSP 的不同质量层上并不稳定，在 OBP 上还可能有害。
因此它值得进入下一轮小型 generation-intent probe，但尚不能作为四任务通用的
V9.5 默认替换，更不能由此推断完整搜索或 held-out 性能改善。

## 复现入口

- 准备、生成与评价：`experiments/runners/traceaad/compact_change_probe.py`
- 分析：`experiments/analysis/analyze_traceaad_generation_probe.py`
- 测试：`tests/test_traceaad_compact_change_probe.py`
- 本地工件：`experiments/generation_probe/20260812_v95_compact_actual_change_probe/`

原始 prompt、response、代码和 evaluator 结果只留本地，不进入 Git。

# TraceAAD V9.5 阶段性运行机制分析

> **冻结说明（2026-08-12）：** 本文保留为 19:10 运行中截点的过程诊断。完整 1000-budget
> 与 held-out 终局复盘见 [TraceAAD-V9.5终局复盘.md](TraceAAD-V9.5终局复盘.md)。终局文已重算
> Allocation / 曲线 / 结构等指标；本文数字不再作为最终结论。
>
> 冻结对象：正式批次 `20260811_171029` 的运行中快照。快照时间为 2026-08-11
> 19:10（Asia/Shanghai），统一 evaluator 截点分别为 TSP 193、CVRP 154、OP 238、
> OBP 478。每个任务均含 3 个独立 repeat。
>
> 本文只分析 V9.5 搜索过程，不是完成实验报告。所有 run 当时仍在运行，尚无 1000-budget
> 最终搜索结论和 held-out 结果。本文中的分组关系是观察性诊断，不是组件因果消融。

## 1. 核心判断

V9.5 的统一 allocation rule

\[
S(a)=q(a)+\frac{s}{\sqrt{n(a)+1}}
\]

因固定尺度 `s` 与各任务 competitive quality gap 的相对关系不同，进入了三种 operational
regime：

1. **Active optimistic continuation（TSP / CVRP）**：optimism 在 56.9%–94.3% 的选择中
   改变纯 `q` argmax，大量新形成的 state 获得一次继续发展的机会。
2. **Greedy mature-anchor refinement（OP）**：`s` 小于多数轮次偏离纯 `q` 所需的临界尺度，
   changed rate 只有 0%–7.7%，预算主要反复投给成熟高质量 anchor。
3. **Pure-`q` + tie-driven diversification（OBP）**：三路 `s=0`，optimism 完全关闭；离散
   objective 产生的大量同分 state 通过 tie-break 带来 state/clade 分散。

当前观察到的是：bootstrap-derived `s` 在不同 objective landscape 上诱导出了明显不同的
allocation intensity。它可能是统一公式对任务尺度的合理适应，也可能暴露 scale estimator 在
特定 landscape 上的局限；两种解释目前都与数据相容，必须等待完整实验与消融后再判断。OP 的
低 changed rate 和 OBP 的 `s=0` 是运行事实，不能直接定性为设计缺陷。

在 optimism 活跃的 TSP/CVRP 上，V9.5 当前更准确的运行形态是 **quality-guided local frontier
expansion**。non-best 选择的平均 `(q_max - q_selected) / s` 只有约 0.11–0.15；它没有跨越很大的
质量差，而是在当前优质 frontier 附近优先验证更新鲜的 executable states。changed rate 本身不是
优化目标，最终仍应由相同 evaluator budget 下的 best-at-budget 与 held-out quality 判断机制价值。

### 1.1 阶段性研究决策

这批统计用于解释 V9.5 **实际如何运行**，不用于在正式批次尚未结束时反向修改机制。当前决策是
保持 V9.5 冻结，让 12 路正式运行完成；不修改代码、prompt、EvidenceBuilder、allocation 公式或
`s` estimator。

| 层 | 当前判断 | 本阶段决策 |
| --- | --- | --- |
| State / Forest | 运行结构正常 | 不改 |
| Evidence | actual diff、Idea 与 context budget 管道正常；composition 随 anchor 状态变化 | 不加固定 formation/direct quota |
| Generation | repeat 间 proposal productivity 差异明显，LLM 随机性仍是重要变量 | 保持 `Idea + Full Code`，不加 planner、critic、operator 或 reflection |
| Allocation | `Quality + Opportunity` 的简洁机制已产生可解释的多种运行形态 | 不以 changed rate 异常为由修改公式 |
| `s` estimator | 简洁地适应 objective scale；zero-heavy objective 是待观察边界 | 完整结果与消融前不判定缺陷 |
| Regression preservation | 部分退步 state 位于后续成功 lineage | 保留 state，不引入 descendant credit |
| Cross-clade mechanism | clade coverage 不是算法状态探索的充分度量 | 暂不增加跨 clade 控制器 |

Evidence 层的核心假设仍是：当前 executable code 不是下一步算法改进所需的全部决策信息，真实的
formation 与 exact-state direct attempts 可能提供增量信息。现有快照没有给出必须修改这一设计的
证据。TSP/CVRP formation-heavy、OP/OBP direct-heavy 可以由 anchor 新鲜度自然解释；强制固定
`2+6` 或 `4+4` quota 反而会把一种运行状态误写成规则。

Generation 层同样不应因单个弱 repeat 改动。TSP optimism-induced improvement 在三个 repeats
中为 15.2%、17.3%、33.1%，CVRP 为 38.8%–53.4%，说明相似 allocation regime 下仍存在显著的
proposal realization 差异。OP r1 可能来自早期 lineage 与一系列低质量 proposals 的共同作用，
不能单独归因于 allocation、evidence composition 或 `s`。

## 2. 统计口径

### 2.1 公平性与截点

版本间性能比较只按 evaluator budget。本文中的 selection、proposal、duplicate、cache 和
prompt evidence 只用于 V9.5 内部机制诊断，不用于版本间公平性判断。

每个任务先读取三个 repeats 的 `candidate_count`，取最慢 repeat 作为统一截点；所有 candidate、
selection、edge 和 evidence snapshot 均截断到该 evaluator order。选择行为使用当时日志保存的
完整 `(q,n,S)` snapshot，不用最终 forest 反推历史决策。

### 2.2 Pure-q 与 critical s

Pure-`q` 对照使用与 V9.5 相同的确定性次级 tie-break：先最大 `q`，再选 `n` 更小、创建更早、
`state_id` 更小的 state。

设本轮 pure-`q` 最优 state 为 `b`。对满足

\[
\frac{1}{\sqrt{n(a)+1}}-\frac{1}{\sqrt{n(b)+1}}>0
\]

的其他 state `a`，定义使其刚好追平 `b` 所需的尺度：

\[
s_{crit}(a,b)=
\frac{q(b)-q(a)}
{\frac{1}{\sqrt{n(a)+1}}-\frac{1}{\sqrt{n(b)+1}}}.
\]

本轮临界尺度为所有合格 `a` 的最小值。若不存在访问数更少的竞争 state，则该轮
`s_crit` 不可定义，且 optimism 不可能改变选择。日志中 `actual s > s_crit` 的比例与实际
`optimism_changed_argmax_rate` 逐 run 完全一致，验证了该诊断量。`s_crit` 只用于解释某轮选择
为何改变，不作为在线调节 `s` 或追求目标 changed rate 的新控制器。

### 2.3 Evidence 与 lineage value

Evidence composition 统计实际进入 prompt 的 attempt exposure，而不是 forest 中可用事实总量。
exact dedup 只指 EvidenceBuilder 按 evaluator-input/raw-code key 折叠的完全重复，不代表语义重复。

`future lineage value` 定义为：某次 optimism-induced selection 产生的新 child state，在当前
截点前是否有严格 global-best breakthrough 出现在其后续 subtree。另行统计 child 当下就突破、
后代延迟突破，以及立即退步 child 的后代突破。多个祖先可能共享同一个后续突破，因此该量是
谱系关联，不是每个祖先的独立因果贡献。

## 3. Allocation 的真实作用

### 3.1 Scale 与 critical s

| Run | actual `s` | `s_crit` Q25 / median / Q75 | median `s/s_crit` | `s>s_crit` | changed rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSP r1 | 2.487 | .172 / .353 / .780 | 7.04 | 94.3% | 94.3% |
| TSP r2 | .720 | .142 / .213 / .453 | 3.37 | 81.9% | 81.9% |
| TSP r3 | 1.384 | .124 / .267 / .957 | 5.17 | 79.5% | 79.5% |
| CVRP r1 | .750 | .128 / .257 / .596 | 2.92 | 71.7% | 71.7% |
| CVRP r2 | .730 | .213 / .405 / .681 | 1.80 | 56.9% | 56.9% |
| CVRP r3 | .859 | .177 / .322 / .561 | 2.67 | 74.5% | 74.5% |
| OP r1 | .088 | .169 / .196 / .289 | .45 | 0.0% | 0.0% |
| OP r2 | .194 | .221 / .289 / .495 | .67 | 7.7% | 7.7% |
| OP r3 | .035 | .111 / .122 / .270 | .29 | .5% | .5% |
| OBP r1 | 0 | .307 / .412 / 4.000 | 0 | 0.0% | 0.0% |
| OBP r2 | 0 | .854 / 3.750 / 5.121 | 0 | 0.0% | 0.0% |
| OBP r3 | 0 | 2.957 / 4.141 / 14.088 | 0 | 0.0% | 0.0% |

当前 `s = median |delta q_bootstrap|` 的原始含义是 task-level 一步变化尺度：让 exploration bonus
与该任务中一次普通 modification 所造成的 fitness 变化处于同一量级。它无需 task-specific
coefficient、min-max normalization、动态 percentile 或 learned uncertainty，是一个简洁的
objective-scale adaptive inductive bias。

OBP 的 `s=0` 直接来自 bootstrap delta 的中位数：三个 repeats 的 8 个 delta 中分别有 6、5、6
个零值。这说明 zero-heavy / discrete objective 是该 estimator 值得观察的边界情况，但尚不证明
校准错误。OBP 的大量 plateau 使 pure-`q` 下的 `q` tie 与 `n` tie-break 本身也能扩展许多同质量
state；额外抬高 `s` 也可能把预算导向明显更差的状态。只有稳定消融显示 `s=0` 显著弱于 `s>0`
时，才能判定 bootstrap median 在这类 landscape 上校准不足。

### 3.2 Freshness、clade 与 depth

| Run | n=0 / n=1 / n>=2 | 使用 clade | Top-1 clade share | 不同 anchor | selected depth mean / max | best depth | init -> best clade |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TSP r1 | 60.2 / 29.0 / 10.8% | 3 | 97.7% | 106 | 13.9 / 38 | 14 | 1 -> 1 |
| TSP r2 | 52.5 / 32.2 / 15.3% | 1 | 100% | 93 | 16.4 / 41 | 28 | 6 -> 6 |
| TSP r3 | 68.8 / 24.4 / 6.8% | 5 | 94.9% | 122 | 20.9 / 46 | 38 | 4 -> 4 |
| CVRP r1 | 80.4 / 15.9 / 3.6% | 2 | 99.3% | 112 | 28.4 / 62 | 51 | 4 -> 4 |
| CVRP r2 | 75.2 / 19.7 / 5.1% | 1 | 100% | 104 | 34.7 / 64 | 57 | 2 -> 2 |
| CVRP r3 | 72.3 / 20.4 / 7.3% | 1 | 100% | 99 | 27.2 / 57 | 53 | 1 -> 1 |
| OP r1 | 9.0 / 9.0 / 82.0% | 1 | 100% | 20 | 4.9 / 12 | 1 | 3 -> 3 |
| OP r2 | 19.9 / 16.7 / 63.3% | 1 | 100% | 44 | 11.8 / 25 | 19 | 4 -> 4 |
| OP r3 | 8.6 / 8.1 / 83.3% | 1 | 100% | 20 | 10.7 / 14 | 14 | 6 -> 6 |
| OBP r1 | 7.8 / 5.0 / 87.2% | 7 | 94.8% | 42 | 5.2 / 11 | 10 | 1 -> 2 |
| OBP r2 | 51.3 / 25.3 / 23.4% | 8 | 45.5% | 242 | 13.7 / 36 | 16 | 1 -> 5 |
| OBP r3 | 22.1 / 21.2 / 56.7% | 6 | 95.0% | 108 | 18.4 / 30 | 19 | 1 -> 1 |

TSP 最强的 r3 触及 5 个 clade，但 94.9% 预算仍位于同一个 clade，且 init-best clade 与
current-best clade 相同。它支持的是 **deep within-clade continuation**：较早找到一条可发展
lineage，然后不断创建新 executable state，并让新 state 获得继续发展的机会。它不支持把
V9.5 描述成 cross-clade exploration。

OP r1 是 greedy mature-anchor regime 的极端：222 次正式选择只覆盖 20 个 anchor，82% 落在
`n>=2`，global best 仍是 depth 1 的初始化程序。OP r3 同样接近 pure-`q`，却从 13.064 提高到
14.300。这说明 OP 同时存在明显的 **early-route sensitivity**：当前 `q` 与后续 development
potential 并不等价，greedy 搜索可能成功，也可能永久锁进错误早期路线。

OBP 的跨 clade 行为来自 pure-`q` 下的同分 tie-break，不能作为 optimistic allocation 的正证据。

这里的核心搜索对象是 **algorithm modification states**，不是 root strategies。单个 clade 内的
`P_0 -> P_1 -> ... -> P_k` 可以累积许多实质不同的可执行算法；因此预算高度集中在一个 clade
不等于探索不足。TSP/CVRP 的深 lineage 与持续 best 更新反而说明，有价值的算法可能通过同一
来时路上的多次 modification 逐步形成，而不要求频繁跨 clade restart。

## 4. Allocation 与 Evidence 的动力学耦合

| Run | direct / formation 均值 | formation missing | 有 formation 但被 direct 挤出 | exact dedup | actual diff | Idea missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSP r1 | .53 / 6.77 | 0% | 0% | 0% | 99.7% | 0% |
| TSP r2 | .66 / 6.77 | 0% | 0% | 0% | 99.8% | 0% |
| TSP r3 | .60 / 6.65 | 2.8% | 2.3% | 0% | 99.9% | 0% |
| CVRP r1 | .24 / 7.36 | .7% | 0% | 0% | 99.9% | 0% |
| CVRP r2 | .31 / 7.23 | 2.2% | 0% | 0% | 99.8% | 0% |
| CVRP r3 | .40 / 7.15 | 0% | 0% | 0% | 99.5% | 0% |
| OP r1 | 5.21 / 2.29 | 44.6% | 44.6% | 1.3% | 97.7% | 0% |
| OP r2 | 3.36 / 4.31 | 20.8% | 20.8% | .7% | 98.2% | 0% |
| OP r3 | 5.47 / 2.09 | 54.5% | 53.7% | 0% | 100% | 0% |
| OBP r1 | 6.45 / .96 | 72.9% | 72.6% | 4.4% | 95.2% | 0% |
| OBP r2 | 1.67 / 5.32 | 11.5% | 10.5% | 2.4% | 98.8% | 0% |
| OBP r3 | 3.06 / 4.26 | 16.9% | 15.8% | .2% | 98.3% | 0% |

12 路都没有因为 token context limit 删除 evidence item；formation 消失来自固定 8-item budget
下的 direct-first selection，而不是 prompt 超长。Correction evidence 管道本身正常：Idea 均存在，
valid correction 的 actual diff 可用率为 95.2%–100%，exact duplicate 很少。

实际运行形成了两种隐式 generation mode：

```text
optimism active -> fresh anchor -> direct history 少 -> formation-heavy
greedy revisit  -> mature anchor -> direct history 增长 -> direct-heavy
```

即：

\[
Allocation \rightarrow n\text{ 分布} \rightarrow Evidence\ composition
\rightarrow Generation\ behavior.
\]

这不破坏 Allocation 与 Evidence 的概念职责分工，但说明二者在运行时存在动力学耦合。当前数据
只能证明 allocation regime 系统性改变 prompt composition，不能证明 formation-heavy 更好，
也不能证明 formation missing 导致失败。OP r3 和 OBP r1 都是 direct-heavy 但仍产生了较强结果。

## 5. Outcome-conditioned allocation

以下只分析 TSP/CVRP。`sample` 是 selection 后完成的 candidate response 数；`valid` 是实际调用
evaluator 并创建有效 child state 的数量；improvement 与 `delta q` 只在 valid children 上计算；
global-best rate 以全部 selection samples 为分母。

### 5.1 三个 repeats 合并

| Task / choice | sample | valid | parent improvement | `delta q` mean / median | global-best breakthroughs | breakthrough / sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSP greedy | 78 | 62 | 4.8% | -1.427 / -.536 | 3 | 3.85% |
| TSP optimism-induced | 451 | 393 | 21.4% | -.705 / -.005 | 8 | 1.77% |
| TSP optimism, n=0 | 295 | 262 | **28.2%** | -.645 / .000 | 6 | 2.03% |
| TSP optimism, n>=1 | 156 | 131 | 7.6% | -.825 / -.060 | 2 | 1.28% |
| CVRP greedy | 133 | 127 | 25.2% | -.836 / -.102 | 32 | **24.06%** |
| CVRP optimism-induced | 279 | 265 | **45.7%** | -.114 / -.006 | 29 | 10.39% |
| CVRP optimism, n=0 | 240 | 229 | **48.0%** | -.072 / .000 | 24 | 10.00% |
| CVRP optimism, n>=1 | 39 | 36 | 30.6% | -.382 / -.061 | 5 | 12.82% |

在两个任务中，optimism-induced choice 的 parent improvement rate 都高于 greedy choice，且
平均退步幅度更小。fresh `n=0` state 的即时 parent improvement rate 高于已经重访过的
optimism anchor：TSP 为
28.2% 对 7.6%，CVRP 为 48.0% 对 30.6%。这与“给新形成的 executable state 一次 continuation
opportunity”一致，是当前最直接的正向过程关联信号。

但 optimism-induced choice 的 global-best rate 不高于 greedy。原因是两组起点不同：greedy
从当前最高 `q` 出发，只要改善就更容易刷新 global best；optimism 从略低但更新鲜的 frontier
出发，可以产生局部改善而仍未越过全局最好。这进一步说明 V9.5 主要做 local frontier expansion，
不能用即时 global-best rate 单独评价其价值。

### 5.2 Optimism-induced choice 的 repeat 差异

| Run | valid children | parent improvement | `delta q` mean / median | global-best breakthroughs |
| --- | ---: | ---: | ---: | ---: |
| TSP r1 | 145 | 15.2% | -.859 / -.005 | 2 |
| TSP r2 | 127 | 17.3% | -.842 / -.005 | 2 |
| TSP r3 | 121 | **33.1%** | **-.378 / -.005** | 4 |
| CVRP r1 | 94 | 46.8% | -.077 / -.004 | 8 |
| CVRP r2 | 73 | **53.4%** | **-.061 / .001** | 12 |
| CVRP r3 | 98 | 38.8% | -.190 / -.052 | 9 |

TSP 最强 r3 的 optimism-induced generation 同时具有最高 improvement rate、最小平均退步和
最多 global breakthroughs。CVRP 弱 r3 并非 optimism 不活跃，而是 optimism-induced children
的即时产出低于 r1/r2。这把问题从 `where to continue` 推向了 `LLM modification 是否 productive`；
仅让 optimism 生效不能保证最终质量。

这些分组仍不是随机消融。Greedy 与 optimism anchor 的当前 `q`、`n`、depth 和历史上下文不同，
不能把组间差异解释为 optimism 的因果效应。

## 6. Immediate gain 与 future lineage value

### 6.1 Optimism child 的延迟突破

| Run | optimism child states | immediate breakthroughs | 有延迟突破后代 | delayed rate |
| --- | ---: | ---: | ---: | ---: |
| TSP r1 | 145 | 2 | 13 | 9.0% |
| TSP r2 | 127 | 2 | 22 | 17.3% |
| TSP r3 | 121 | 4 | 28 | **23.1%** |
| CVRP r1 | 94 | 8 | 33 | 35.1% |
| CVRP r2 | 73 | 12 | 30 | **41.1%** |
| CVRP r3 | 98 | 9 | 34 | 34.7% |

TSP r3 的强结果不仅对应更好的即时 improvement，也对应最高的 delayed descendant breakthrough
rate。CVRP r3 的 delayed rate 与 r1 接近，显示其形成事实中仍存在后续突破关联；当前差距更多
表现为即时修改质量和突破幅度较弱，而不是 optimism 没有形成可发展的 lineage。

### 6.2 Fresh 与 revisited optimism child

| Task / selected anchor | child states | immediate breakthrough | 有延迟突破后代 | immediate-regress children | 退步 child 后续突破 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSP n=0 | 262 | 6 (2.3%) | 41 (15.6%) | 130 | 7 (5.4%) |
| TSP n>=1 | 131 | 2 (1.5%) | 22 (16.8%) | 81 | 9 (11.1%) |
| CVRP n=0 | 229 | 24 (10.5%) | 79 (34.5%) | 114 | 23 (20.2%) |
| CVRP n>=1 | 36 | 5 (13.9%) | 18 (50.0%) | 23 | 7 (30.4%) |

立即退步并不等于没有后续 lineage value：TSP/CVRP 中分别有 16 和 30 个 optimism-induced
regressed children 的后代后来刷新 global best。这个结果支持 V9.5 保留 regression states，
也说明不能只按一步 improvement 判定一个中间状态没有发展价值。

但该比例不能直接变成 online credit：同一条 breakthrough lineage 上的多个祖先会共享同一个
后续突破，深 lineage 因而产生相关、重叠的正标签。它证明这些状态位于成功来时路上，不证明
每个退步中间状态都是因果必要条件，也不证明回到该状态重新生成仍有同等价值。

## 7. 分任务机制定性

### 7.1 TSP：deep within-clade continuation 的正信号

TSP 同时满足：active optimism、较高 fresh-state 比例、formation-heavy prompt、深 lineage，且
最强 r3 的 optimism-induced immediate/future value 均最高。当前数据支持 V9.5 在优质 clade
附近持续累积 modification；不支持把收益解释为广泛 root-clade switching。

### 7.2 OP：near-greedy regime 与 high route sensitivity

OP r1/r3 的 `s/s_crit` 中位数仅 .45/.29，optimism 基本无法偏离 pure-`q`。r1 停在初始化 best，
r3 却在相同 near-greedy regime 下显著改善。这首先说明 OP 存在明显 early-route sensitivity；
弱 r1 也可能只是某条早期 lineage 与连续低质量 LLM proposals 共同形成的不利 realization。
当前快照不能区分这是自然的任务适应、generation 随机性，还是 `s` 偏小造成的路线锁定。

### 7.3 CVRP：allocation active，不保证 modification productive

三路 optimism、fresh state 和 formation 均正常。弱 r3 的即时 improvement 与平均 delta 较弱，
但 delayed lineage rate 并未消失。当前问题更接近 generation productivity / improvement magnitude
差异，而不是 allocation 未运行或 formation 丢失。

### 7.4 OBP：当前批次不能检验 optimism

三路 `s=0`，所有选择都等价于 pure-`q`；多 state/clade 来源于离散同分和 tie-break。因此 OBP
当前结果只能评价 V9.5 的 pure-`q`、Evidence 和 Generation 联合行为，不能评价额外 optimism
的增量收益。该运行形态可能已经适合 plateau-heavy landscape，不能按 changed rate 为零定性为
allocation failure。

## 8. 原始证据与复核入口

所有统计均来自本地正式批次目录：

```text
experiments/<task>/traceaad_v9_5/
  v9_5_20260811_171029_<task>_rep{1,2,3}/
```

逐轮 `(q,n,S)` 与 evidence snapshot 来自 `artifacts/decisions.jsonl`；candidate budget、评价结果和
parent relation 来自 `artifacts/candidates.jsonl`；严格 global-best breakthrough 与 child edge
来自 `artifacts/edges.jsonl`；state parent/depth/root 关系来自 `checkpoints/latest.json`。所有
checkpoint 只用于稳定的结构字段，历史选择值均读取当时的 decision snapshot。

实现口径可由 [selection.py](../../llm4ad/method/traceaad_v9_5/selection.py)和
[evidence.py](../../llm4ad/method/traceaad_v9_5/evidence.py)复核。本文没有改写原始工件，也没有
使用运行结束后的未来事件补充当前截点内的 outcome；future-lineage 查询同样只保留
`sample_order <= cutoff` 的 states 与 edges。

## 9. 证据边界、冻结决策与后续检验

本快照支持：

1. `critical s` 准确解释 allocation 是否会偏离 pure-`q`；
2. TSP/CVRP 的 active regime 主要执行 local optimistic frontier expansion；
3. fresh optimism choices 在 TSP/CVRP 具有较高一步 productivity；
4. allocation regime 系统性改变 evidence composition；
5. 立即退步 state 有时位于后续成功 lineage，支持保留 regression states；
6. clade concentration 与 algorithm-state exploration 不是同一个概念。

本快照不支持：

1. formation-heavy 比 direct-heavy 更好；
2. optimism-induced choice 因果上优于 pure-`q`；
3. future-breakthrough ancestor 应获得数值 credit；
4. OBP 的当前改善来自 optimism；
5. OP 的 near-greedy 或 OBP 的 `s=0` 已经证明 scale estimator 存在缺陷；
6. 某个目标 changed rate 应被写入在线控制器；
7. 未完成搜索的过程指标能够替代 1000-budget 与 held-out 结果。

因此本阶段不根据该快照修改 V9.5。正式运行完成后，先在相同 1000 evaluator budget 上重算
本文指标，并按以下问题顺序判断：

1. **联合系统是否有效**：比较 V9.5、V9.4/V9.3 与强基线的完整 search best 和 held-out quality；
2. **History 是否有增量价值**：优先比较 Current Code Only 与 Current Code + Evidence；
3. **简单 optimism 是否优于 greedy-`q`**：比较 `q` 与 `q + s/sqrt(n+1)`，再解释任务差异。

`s/s_crit` 和 changed-rate replay 继续只作为行为敏感性分析，不升级为机制。只有完整结果提示稳定的
task-specific failure 后，才考虑追加固定 `s` / scale estimator 消融。所有报告继续分开呈现搜索
过程、最终 search best 与 held-out 结果。

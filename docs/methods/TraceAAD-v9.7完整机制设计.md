# TraceAAD V9.7 完整机制设计

> 版本：V9.7（`llm4ad/method/traceaad_v9_7/`），protocol id
> `traceaad-v9.7-route-refine-explore`。状态：已实现并通过集成测试与冒烟，
> 正式批次 `20260813_184519` 运行中。
> V9.7 在 V9.6 基础上只改两处：**预算分配**从锚点级单层改为路线→锚点两级，
> **生成指令**从单一自由指令改为 Refine / Explore 两种固定概率意图。
> 历史上下文（挑法 + 写法）、初始化、输出契约、去重与 cache、预算单位
> （1000 次真实评价）、checkpoint 结构均与 V9.6 相同，见
> [TraceAAD-v9.6完整机制设计.md](TraceAAD-v9.6完整机制设计.md)。
> 研究背景：分配见 [RQ-003](../research/RQ-003-轨迹上下文与搜索评分.md)，
> 历史见 [RQ-009](../research/RQ-009-锚点历史上下文.md)。

## 0. 设计立场

每消耗一份评价预算，系统依次回答三个问题：

1. **从哪里继续**（分配）：哪条路线、哪个锚点；
2. **怎么走到这里**（历史）：这个算法经过哪些想法和结果形成，附近试过什么；
3. **这次怎么改**（意图）：沿当前设计继续，还是改变当前设计。

分工：历史只提供事实，不携带方向；方向由意图显式给出；分配负责预算落点。

V9.7 只针对 V9.6 暴露的两个问题：**预算是否长期集中于单一路线**（分配层）、
**历史是否让生成过度保守**（生成层）。不引入其他机制；跨路线知识重组
（把其他路线的算法作为参考给模型）会同时改变意图与上下文两层，不属于本版，
留作后续单独研究。

## 1. 预算分配：路线 → 锚点两级

**路线**定义为同一初始根衍生的全部锚点状态（事实拓扑，不声称语义类别）。

每次分配分两步，公式与 V9.6 锚点层完全相同，只是先在路线层用一次：

1. **选路线**：`score(r) = max_{a∈r} q(a) + s / sqrt(N(r) + 1)`，其中
   `N(r)` 为该路线全部锚点已发起的生成次数之和，`s` 沿用 bootstrap |Δ|
   中位数。取 argmax；同分取 `N(r)` 更小者，再按根创建顺序。
2. **选锚点**：在选中路线内 `score(a) = q(a) + s / sqrt(n(a) + 1)`，
   tie-break 与 V9.6 相同。

两级的职责用同一句式陈述：**锚点级 optimism 防止单个状态被过度或不足访问，
路线级 optimism 防止预算长期集中在某一初始分支。** 该公式不估计路线潜力——
"当前强 ≠ 继续有潜力"是这一层的研究动机，但本版只做投入均衡这一种极简
处理，不声称建模了 trajectory potential，也不引入趋势项、路线信用或概率
控制器（V5 / V9.1 / V9.2 / V9.4 均未证明其价值）。

审计新增 `route_selected` decision：全部路线的 `best_q / N / optimism /
score` 与选中路线；`anchor_selected` 保持 V9.6 字段并增加 `route_id`。

## 2. 生成意图：Refine / Explore

意图在锚点选定后确定，与搜索状态无关——不依据 plateau、成熟度或信用切换
（单步 plateau 已被证明是不可靠的切换信号）。按固定概率
**Refine 0.7 / Explore 0.3** 抽取，由 `hash(generation_seed, iteration)`
确定性映射，可恢复、不在 checkpoint 中保存 RNG 状态。比例是第一版固定
mixture，不赋予理论意义，其作用只是保证搜索中既持续利用来时路、也固定保留
结构性探索机会。初始化阶段的 bootstrap 生成固定为 Refine。

两种意图共享同一历史块与输出契约，只更换 `[Instruction]` 段，唯一受控差异
是修改尺度：

- **Refine**（沿当前设计继续）：

  > Continue improving the current algorithm within its existing design.
  > Make one focused modification based on the current algorithm and its
  > improvement history.

- **Explore**（改变当前设计）：

  > Seek a materially different way to improve the current algorithm. Do
  > not merely tune parameters or make a small local modification. You may
  > replace or substantially restructure an important part of the current
  > design.

历史在两种意图下内容不变、读法不同：Refine 把它当作继续发展的依据，
Explore 把它当作已经走过的路。方向由指令承担，历史保持纯事实。

审计：每条 attempt / edge 记录 `intent`（替代 V9.6 的单一生成算子标签），
使"意图 × 结局"与"意图 × 编辑幅度"可直接统计。

## 3. 历史上下文：与 V9.6 一字不动

挑法（formation 为主、direct 改进/退步各封顶 2、合计至多 8 条、按时间排列）
与写法（Idea / Change / Result / Fitness）不变——第一、二层效果已被
V9.5→V9.6 证据链确认。Context 回退同 V9.6（从最早事件逐条丢弃）。

## 4. 闭环

```text
路线层分配（预算是否已过度集中于此）
  -> 锚点层分配（从哪个状态出发）
  -> 构造历史（怎么走到这里、附近试过什么）
  -> 抽取意图（沿当前设计继续 / 改变当前设计）
  -> 一次 Idea + Code 生成 -> 一次真实评价
  -> 更新 q、n(a)、N(r) -> 重新分配
```

## 5. 验证

- 运行入口 `experiments.runners.traceaad.run --version v9_7`，运行配置与
  V9.6 一致（8 根、32768 context、8192 输出 tokens、1000 次评价）；
  历史与来源模块与 V9.6 逐字节相同，由集成测试强制。
- 净效应由正式三重复 + held-out 判断，遵循"大而一致"判读纪律（三重复同向
  且幅度明显超过同方法批次间抖动）。过程报告按意图分列有效率、编辑幅度、
  improve rate 与新 best 产生率，按路线分列预算占比与末次刷新位置；这组
  数据直接回答本版的两个问题（预算是否太集中、历史是否让生成太保守），并为
  "什么时候应该 Explore"积累证据——状态依赖的意图选择在有该证据前不设计。
- 冒烟已确认：两种意图的 prompt 正确渲染、模型调用与评价方向正确、
  路线分配按预期运行。

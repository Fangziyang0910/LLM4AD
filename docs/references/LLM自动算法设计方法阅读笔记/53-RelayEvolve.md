# RelayEvolve

- 论文：*Relay, Don't Route: Adaptive Population Handoff for Cost-Efficient LLM-Driven Evolution*；本地来源：`../../../../papers/Relay_Dont_Route_Adaptive_Population_Handoff/paper.pdf`；设计对象：强弱 LLM 共同驱动的程序进化过程。

## 1. 核心问题与方法

论文先分析 cheap/strong model 的搜索轨迹，发现进步前置、早期质量只有噪声预测性。RelayEvolve 因而让廉价模型分块探索多条路线，以质量—多样性 relay bank 的边际增益作为 Grow/Deepen bandit 奖励和停机信号；随后精选 population 整体交给强模型共同精炼。

## 2. 论文宣称的机制贡献（逐项）

- 模型调用是由 population state 耦合的，调度单位应是路线/群体而非独立 query。
- Relay Gain 同时控制路线扩展、交接时机和可交接种群质量。
- cheap-first、strong-later 的 population handoff 改善固定成本下表现。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|进步前置且 cheap model 捕获大部分早期增益|Figures 1–2|部分支持|四任务轨迹支持描述；早期—最终 Spearman 随时点/任务变化且可为负。|
|完整 handoff 在固定成本有效|Table 1、Figure 5|间接支持|12 个任务×预算设置中 11 个平均分最高，但为完整 scheduler+curation+cascade。|
|Relay Gain 的 allocation/stopping 有益|Figure 4(a)|直接支持|random allocation 和 no stopping 均退化。|
|质量—多样性 curation 有益|Figure 4(b)|直接支持|与 quality-only、diversity-only、random seeds 比较。|
|方向应为 cheap→strong|Figure 4(c–d)|部分支持|cascade 方向和 strong reserve 敏感性支持所测模型对；不是普遍模型大小定律。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

早期廉价探索的价值是形成多个可继续的状态，而非把每次调用独立判成简单/困难。交接一整个互补种群可保留路线多样性，让强模型在共同 archive 中重组和深挖。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：以路线边际贡献而非单节点即时增益调度预算。最小验证：同模型也可先比较 call-level、trajectory-level、population-level 三种分配。

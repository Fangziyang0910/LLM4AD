# DGA²D

- 论文：*DGA²D: Directed Graph-Guided Automated Algorithm Design*；本地来源：`../../../../papers/DGA2D_Directed_Graph_Guided_Automated_Algorithm_Design/paper.pdf`；设计对象：由功能算子有向 walk 组成的完整算法 pipeline。

## 1. 核心问题与方法

DGA²D 把算法空间表示为带多份代码实现的 directed operator graph。policy 采样有界 walk 形成 pipeline；候选的标准化终端收益被分配给相邻 implementation transition，随后聚合到实现、算子和边，用于删改低信用元素并训练 pipeline policy。

## 2. 论文宣称的机制贡献（逐项）

- 图拓扑允许组件复用、重排与带环的完整 pipeline 表达。
- first-order path credit 区分同一实现处在不同前驱上下文中的价值。
- 实现池和图连接可按局部信用双层演化。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|完整方法跨 12 个 CO benchmark 有竞争力|Table 1、Figure 3、Appendix Table 8|间接支持|联合图表示、RL policy、credit 与 LLM 编辑的结果。|
|directed graph 表示有益|§4.3、Table 2|直接支持|同搜索设置逐步放松/替换结构表示，DGA²D 最好。|
|first-order credit 优于无上下文信用|§4.3、Table 3|直接支持|四任务上 first-order 一致优于 zero-order。|
|更长历史一定更好|Table 3|反向或混合证据|second-order 与 full-path 更差，支持“适度上下文”而非越长越好。|
|first-order 的有限样本逻辑|Appendix D.2、Theorem 2|部分支持|在 first-order sufficiency 等假设下给出偏差—方差分析；真实任务未验证该假设普遍成立。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

zero-order 混合了不同前驱造成偏差，full-path 又把样本切得太碎造成高方差；first-order 是局部语境与统计复用之间的折中。这与“完整历史全部塞进 prompt”不同，历史粒度本身就是估计问题。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：比较 node、edge、短 path、full lineage 四种信用粒度。最小验证：固定候选和预算，报告各信用 cell 的样本数、方差及 held-out 选择质量。

## 6. 证据边界

Table 3 消融只覆盖四个代表任务；first-order 不是理论上无条件最优。完整系统同时学习 policy 并修改图与实现池，终端信用仍可能错误归因到 pipeline 内无关操作。

## 7. 论文内定位

Figure 2；§3；Algorithms 1–2；Table 1；§4.3 Tables 2–3；Appendix D.1–D.3、Theorems 1–2；Figures 4–5。

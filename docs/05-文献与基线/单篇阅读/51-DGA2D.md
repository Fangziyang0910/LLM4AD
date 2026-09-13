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
|完整方法跨 12 个 CO benchmark 有竞争力|Table 1、Figure 3、Appendix Table 8|直接支持（整法有效性）|匹配协议下的整法比较直接支持该方案在 12 个 CO benchmark 上优于基线；不能顺带推出每个组件分别有效（需由消融隔离）。|
|directed graph 表示有益|§4.3、Table 2|直接支持|同搜索设置逐步放松/替换结构表示，DGA²D 最好。|
|first-order credit 优于无上下文信用|§4.3、Table 3|直接支持|四任务上 first-order 一致优于 zero-order。|
|pipeline 内更长信用依赖一定更好|Table 3|反向或混合证据|在求解器内部组件执行拓扑中，second-order 与 full-path 较差（高方差噪声），支持一阶转移适度性；注意这属于 pipeline 内组件前后关系，不能外推为外层搜索祖先历史或 prompt 长度。|
|实现池容量的影响|§4.3.3、Table 4（K∈{4,8,15,20}）|部分支持|K=4 全面最差；FJSP 最优在 K=15、CVRP/MIS/3D-CLP 最优在 K=20——容量与任务相关，非越大越好。|
|first-order 的有限样本逻辑|Appendix D.2、Theorem 2（另有 Theorems 3–5：谱表达力、信用更新收敛与漂移跟踪界）|部分支持|在 first-order sufficiency 等假设下给出偏差—方差分析；论文在 D.2 自认不主张 first-order 普遍最优，真实任务未验证该假设。|

## 4. 机制的底层逻辑与概念层次界定

**极其关键的概念边界澄清**：
DGA²D 中的 first-order、second-order 与 full-path credit，讨论的是**一个算法求解器 Pipeline 内部不同功能组件（如构造算子 $\to$ 邻域搜索 $\to$ 局部扰动）在单次执行过程中的前后调用转移关系**。
- **它属于【求解器内部组件执行拓扑】**：zero-order 忽略前驱组件造成组合归因偏差，full-path 把具体组合路径切得太细造成样本稀疏高方差；first-order 是相邻组件配合度与统计样本复用率之间的折中；
- **它绝不是【外层搜索的祖先演化谱系（Ancestral Lineage）】**：它不涉及第 $g$ 代代码是由第 $g-1$ 代哪个父代代码变异而来的演化树；
- **它更不是【LLM 提示词的上下文长度（Prompt Length）】**：论文中“full-path 较差”仅仅说明复杂的长调用链需要过多评价样本来估计信用。**它可以启发“组件价值依赖运行上下文”的认识，但绝不能直接证明“TraceAAD 只应使用短历史”**。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- **组件价值依赖上下文环境**：同一段局部搜索代码在不同初始化算子之后的作用可能大相径庭，这启发我们：评估算法模块时不能脱离其运行上下文。
- **防止将管线拓扑与搜索谱系混为一谈**：TraceAAD 中的“轨迹”主要是外层代码生成与评估的演化历史（或求解器的解空间移动轨迹 PSTraj），与 DGA²D 的组件流水线图属于完全不同的系统层级；借鉴其信用分配思想时，必须准确对应到具体的决策变量上，绝不能把管线组件调用关系误读为演化谱系或 prompt 长度限制。

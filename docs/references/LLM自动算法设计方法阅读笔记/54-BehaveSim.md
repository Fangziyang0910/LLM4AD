# BehaveSim

- 论文：*Rethinking Code Similarity for Automated Algorithm Design with LLMs*；本地来源：`../../../../papers/Rethinking_Code_Similarity_for_Automated_Algorithm_Design/paper.pdf`；发表于 ICLR 2026；设计对象：用于 AAD 种群管理的算法行为相似度。

## 1. 核心问题与方法

BehaveSim 将候选在实例求解时产生的中间解序列定义为 problem-solving trajectory（PSTraj），用 DTW 对齐不同长度轨迹并计算相似度；它替代代码/embedding 相似度用于 FunSearch 多岛分配，也用于聚类解释算法族。

## 2. 论文宣称的机制贡献（逐项）

- 执行轨迹比静态代码更接近算法逻辑。
- 按行为构造岛可提高岛内一致性与岛间差异。
- 行为多样性可改善 FunSearch/EoH 的算法发现。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|BehaveSim 区分语法/输出误导案例|Tables 1–2、Appendix Tables 4–8|直接支持|四类构造案例逐项比较多种相似度。|
|集成后改善 AAD 表现|Figure 4、Table 3|部分支持|在三任务改善，但集成同时改变岛初始化和归属规则。|
|岛内更一致、岛间更分离|Appendix Figure 9|直接支持|与相同多岛骨架的 FunSearch 只改变 population management。|
|跨岛通信与 clustering 有益|Appendix Figure 10|直接支持|ASP 上改变 inter-island 概率并移除 clustering；任务范围单一。|
|DTW 是唯一正确轨迹度量|Appendix Figures 6–7、Table 9|反向或混合证据|DTW、ERP、mean-pairwise 高度相关，cosine 不同；支持行为轨迹而非 DTW 唯一性。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

代码不同可执行同一策略，代码相近也可沿完全不同搜索路径；PSTraj 直接观测算法与状态空间的交互。代价是相似度依赖 probe instances 与记录粒度，且轨迹采集增加 evaluator 成本。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 应把设计 lineage 与候选执行轨迹分开：前者解释思想来路，后者测行为新颖性。最小验证：用 held-out probes 建 BehaveSim，再看其是否预测互补路线贡献。

## 6. 证据边界

三项 AAD 任务、特定轨迹定义与 probe 分布；DTW 距离不等于因果机制差异。trajectory truncation/sampling 虽有敏感性分析，部署成本和跨规模稳定性仍需单独评估。

# Mutation Without Variation

- 论文：*Mutation Without Variation: Convergence Dynamics in LLM-Driven Program Evolution*；本地来源：`../../../../papers/Mutation_Without_Variation_Convergence_Dynamics_in_LLM_Driven_Program_Evolution/paper.pdf`；研究对象：无选择压力的 LLM 程序 mutation chain，而非一套新 AAD 方法。

## 1. 核心问题与方法

论文在有限强类型 DSL 中反复让 LLM 变异同一程序，刻意移除 fitness 与 selection，分别以完整程序和抽象 skeleton 统计唯一状态、重访、转移图、自环与周期，并改变 prompt、模型和随机重复；经典 GP subtree mutation 是算子对照。

## 2. 论文宣称的机制贡献（逐项）

- LLM mutation 自身会把链推向受限 attractor，而非持续产生新结构。
- 结构收敛比词法/终端变化更严重，短周期和自环主导。
- prompt 与模型改变收敛速度，但不能消除该现象；经典 GP 无同等坍缩。

## 3. 实验究竟支持了什么

|机制主张|论文证据|证据等级|判断|
|---|---|---|---|
|无选择时仍出现结构 attractor|§5.1–5.3、Tables 1–3、Figures 3–7|直接支持|隔离 mutation 后，多数链在 50–100 步内平台化；87% 链中超过 93% 变异重访 skeleton。|
|短周期/自环主导|§5.4、Table 4、Figure 1|直接支持|program 层多为 2-cycle，skeleton 层平均周期接近 1。|
|效应依模型和 prompt 但普遍存在|§5.3、Figures 5–7|部分支持|跨所测条件方向稳定，但模型比较仅四 prompts 且部分无重复。|
|这是 LLM 而非一般 mutation 的性质|经典 GP subtree mutation 对照，§5|部分支持|GP 未见可比收敛；但两种算子语义、步幅和有效性分布并不完全匹配。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

LLM rewrite 受高概率模板、指令服从和语义保持先验牵引，反复应用会进入少数吸引盆；表面变量替换还能继续发生，却不再扩展算法骨架。选择压力可能进一步加剧或抵消这一偏置，本文没有测量。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 必须同时统计代码、结构和行为新颖性，并检测 lineage 的自环/短周期。最小验证：在相同父代与 prompt 下运行无选择 mutation chains，比较模型与算子。

## 6. 证据边界

DSL 有界且只分析 genotype，没有执行行为与 fitness；结论不能直接写成 AAD 必然性能坍缩。模型级结论的重复量不均，作者也明确建议扩展到行为表示。

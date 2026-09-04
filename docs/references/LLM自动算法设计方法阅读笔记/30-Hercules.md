# Hercules

- 论文：*Efficient Heuristics Generation for Solving Combinatorial Optimization Problems Using Large Language Models*；本地来源：`../../../../papers/Efficient_Heuristics_Generation_for_Solving_Combinatorial_Optimization_Problems_Using_La/sample-sigconf.tex`；设计对象：GLS、构造式、ACO 启发信息与 NCO attention 重塑。

## 1. 核心问题与方法

Hercules 针对 LLM-EPS 的两类代价：搜索方向常空泛，且所有候选都要真评估。CAP（Core Abstraction Prompting）有基于 information gain 的理论刻画（IG = $-\sum_j p_j\log p_j \in (0,\log(k{+}1)]$，附录 A 证明）并做分段使用：前 λ=0.7 比例迭代用精英启发式的抽象组件，之后直接用父代核心组件以保多样性；配 rank-based 父代选择。Hercules-P 额外用 PPP：让 LLM 根据待评估代码与已评估代码的语义相似性预测性能及置信度；只真实评估部分候选，用预测分数参与筛选（ConS 分层复评兜底）。搜索仍是进化式产生、评估和保留候选；Hercules-P 相对 Hercules 搜索时间减 7%–59%，代价是上下文 token 约 1.5 倍。

## 2. 论文宣称的机制贡献（逐项）

- CAP 以精英程序的抽象知识提高搜索方向的具体性。
- PPP 用语义相似性和置信度减少昂贵 evaluator 调用。
- 同一框架适配多种 GAF，而非只设计单一优先级函数。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|在 TSP GLS 上兼顾质量与搜索成本|§`4.1`，Tables `tabgls`、`tabsearchreport`|间接支持|支持所列设置与基线的整法比较；成本口径应与 evaluator、模型调用和候选数一起读。|
|可迁移到构造、ACO、NCO 设计对象|§`4.2`--`4.4`，Tables `tabselect`、`tabaco`、`tabNCO`|间接支持|说明完整方法覆盖多个设计槽；没有将“对象扩展”自身与统一控制器独立随机化。|
|CAP、PPP 各自有效|Table `ablation`、§`4.5`；Fig. `accuracy`（EXEMPLAR 变体预测精度箱线图，图非表）|直接支持|组件消融提供直接证据（w/o rank-based selection 8.49 vs 完整 11.10），但范围限于所挑任务与预算。|
|预测性能可替代真实评估|PPP 描述、Table `tabsearchreport`|间接支持|论文自认 PPP 预测精度一般（预测值与真值 Pearson r=0.39、ANOVA p=0.6），EXEMPLAR 使预测精度中位数提升 26%/37%（p=0.048/0.004），可靠性靠 ConS 分层复评兜底；支持用于该筛选流程降低调用，不能证明预测分数可作可靠 fitness。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

CAP 将候选代码压缩为可复用的“设计理由”，减少 LLM 从冗长实现中提取信号的负担；PPP 是代理模型，只有在代码语义相似与真实性能相关时才节省预算。二者会耦合：CAP 使候选更同质时 PPP 可能更准，却也可能压低探索多样性。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把已验证程序归纳成短、可追溯的改进线索。前提：摘要要链接原程序与分数。风险：摘要丢失关键条件而诱导伪迁移。最小验证：同父代、同提示下比较原代码上下文与结构化摘要。
- 可学习点：以代理评估做预筛。前提：必须保留校准集与真实复核。风险：优化代理而非任务。最小验证：报告被筛掉候选中真实 top-k 的漏检率。

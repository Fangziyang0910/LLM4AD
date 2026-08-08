# MEoH

- 论文：MEoH；本地来源：[`papers/MEoH/MEoH.tex`](../../../../papers/MEoH/MEoH.tex)；设计对象：面向组合优化的启发式代码。

## 1. 核心问题与方法

MEoH 在 EoH 式 LLM—evaluator 演化框架上，把“多种演化策略/提示角色”作为并行产生候选的来源，再以适应度筛选。论文的重点是避免单一 prompt 反复诱导同类代码：不同策略承担不同的改良或发散功能，候选仍必须通过可执行 evaluator。

## 2. 论文宣称的机制贡献（逐项）

- 多策略 LLM 演化扩大启发式搜索的覆盖。
- 依据候选质量进行选择，使策略生成受任务反馈约束。
- 通过组合策略降低单模型提示的同质化。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|MEoH 在论文基准上取得主结果|§Experiments 的 `tab: BPP`、`tab: TSP_GLS_random`、`tab: TSP_GLS`|间接支持|这些是整法与其他方法比较，不能分配给多目标组件。|
|dominance-dissimilarity 优于常规 MOEA|§Comparison to Conventional MOEAs，`fig: BPP_ablation`、`fig: TSP_ablation`|部分支持|该受控方法家族对照支持其在 BPP/TSP 设置的相对表现；不隔离其中距离或支配关系。|
|策略带来更大多样性|`fig:dominance-dissimilarity` 与 `fig: TSP_GLS_best_EoH`|间接支持|前者是机制示意，后者展示 non-dominated heuristics/any-time 表现，未直接测量行为多样性。|
|某一策略是增益来源|无逐策略、等预算消融时|未验证|整体优于基线不能分配信用。|

## 4. 机制的底层逻辑

阅读分析：多个 prompt 相当于对生成分布施加不同条件，只有当这些条件产生可区分、互补的行为区域时，才会提升探索。统一 evaluator 负责选择但也可能让所有策略迅速收敛到同一局部模式；故“策略数量”不是充分机制，分配和去重才是隐含条件。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把生成路线显式标记并保留其结果。前提：路线可比较。风险：并行 prompt 只是加倍 token。最小验证：固定调用预算下报告每路线有效率、独特行为数和 best 改进。
- 可学习点：用轨迹证据判断互补而非凭角色名称。前提：有行为或思想标签。风险：文本角色与实际变异脱钩。最小验证：抽样审计路线间的 diff/评估差异。

## 6. 证据边界

MEoH 的证据由文内 BPP、随机 TSP 与 TSPLIB 配置限定；正文表格报告 in-/out-of-distribution 结果，但没有为每项机制给出独立种子方差或显著性检验。评价实例是否和搜索实例分离仍决定泛化解释。

## 7. 论文内定位

入口：[`MEoH.tex`](../../../../papers/MEoH/MEoH.tex)。使用 §Methodology（`alg:MEoH`、`fig:dominance-dissimilarity`）、§Experiments（`tab: BPP`、`tab: TSP_GLS_random`、`tab: TSP_GLS`、`fig: BPP_ablation`、`fig: TSP_ablation`）及 §Algorithm Details。

# BEAM

- 论文：*BEAM: Bi-level Memory-adaptive Algorithmic Evolution*；本地来源：`../../../../papers/BEAM_Bi-level_Memory-adaptive_Algorithmic_Evolution/bare_jrnl_new_sample4.tex`，含 `sections/beam_framework.tex`、`sections/experiments.tex`、`sections/appendix.tex`；设计对象：算法/启发式程序及其进化记忆。

## 1. 核心问题与方法

BEAM 的题名给出两条主轴：bi-level evolution 与 memory-adaptive。主 tex 将框架、benchmark、实验拆分输入；其意图是在候选级改进之外维护/更新可复用记忆，以影响后续生成和选择，而非仅保存当前最优代码。阅读时应把“记忆内容、检索规则、上层更新”与底层变异算子分开。

## 2. 论文宣称的机制贡献（逐项）

- 双层进化：候选算法层与记忆/策略层共同更新。
- 自适应记忆：按表现调整未来上下文或搜索偏好。
- 面向算法演化而非一次性代码生成。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|BEAM 在所列 benchmark 有端到端收益|Tables `tab:traditionalmain`、`tab:CAF`、`tab:CombinedTable`、`tab:BBOBTable`|间接支持|完整系统相对基线的比较不能归因于双层或自适应记忆。|
|自适应记忆有贡献|Table `tab:ablation` 的 BEAM 与 BE；TSP-500、CVRP-500、Ackley/Rastrigin|部分支持|该表的 “Adaptive Memory” 小节只比较 BEAM/BE，支持该命名版本差异；主文未逐项披露所有上下文/调用成本。|
|教育/搜索控制方式的影响|Table `tab:ablation` 的 One-Shot 与 MCTS|部分支持|表直接比较两个教育方式于 MIS、CVRP、CAF，但不是“二层机制”的单组件消融。|
|记忆积累解释性能|案例/轨迹分析|间接支持|案例证明“发生过”，不证明是平均收益原因。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

记忆若存储可复用的失败模式、策略与其适用条件，可使进化从反复采样转向有条件复用；双层更新又能避免一成不变的检索。但记忆会形成路径依赖：早期噪声被放大，且额外上下文与更大计算预算可伪造“记忆收益”。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：每条历史应区分程序、改动、证据和适用条件。前提：检索单位有 provenance。风险：把低质量摘要当知识。最小验证：固定提示长度，比较有 provenance 的轨迹摘要与随机历史。
- 可学习点：记忆更新应有淘汰条件。前提：能重新验证其价值。风险：上下文膨胀、确认偏差。最小验证：记录每条被检索记忆后的真实增益分布。

## 6. 证据边界

Table `tab:budget` 给出与 LHH 比较的 evolving-stage 预算；四张端到端比较表为 `tab:traditionalmain`、`tab:CAF`、`tab:CombinedTable`、`tab:BBOBTable`。`tab:ablation` 未报告随机种子置信区间或逐项 token/调用控制，因此不能把完整 BEAM 的优势拆成双层控制、记忆内容或代码复杂度的因果效应；Fig. `fig:codeLength` 还显示其生成程序更长、更复杂，是重要混杂。

## 7. 论文内定位

主入口 `bare_jrnl_new_sample4.tex`；`sections/beam_framework.tex`（机制）、`sections/benchmark.tex`（任务）、`sections/experiments.tex`（Tables `tab:traditionalmain`、`tab:CAF`、`tab:CombinedTable`、`tab:BBOBTable`、`tab:ablation`）、`sections/appendix.tex`（设置与补充）。

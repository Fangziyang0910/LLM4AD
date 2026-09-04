# BEAM

- 论文：*BEAM: Bi-level Memory-adaptive Algorithmic Evolution for LLM-Powered Heuristic Design*；本地来源：`../../../../papers/BEAM_Bi-level_Memory-adaptive_Algorithmic_Evolution/bare_jrnl_new_sample4.tex`，含 `sections/beam_framework.tex`、`sections/benchmark.tex`、`sections/experiments.tex`、`sections/appendix.tex`；设计对象：完整求解器（算法结构 + 函数实现）。

## 1. 核心问题与方法

题名的三条主轴对应三个组件。**bi-level** 指算法结构层与函数实现层的双层优化：依据质量分解 $Q(I)=Q_s(\mathcal S(I))+\sum_i Q_{f_i}(f_i\mid\mathcal S(I))$，外层（Exterior，结构变量）用 GA 进化，内层（Interior，函数实现变量 $w$）用 MCTS 搜索——"教育"即 Interior 层为实现函数，经 Fixing 与 CMA-ES Calibration 完成参数校准。**Memory-adaptive** 的 Adaptive Memory 是标注为 "an external optimization mechanism" 的第三个组件，传统单层 LHH 因结构限制无法支持；记忆条目评分 $S(f)=\alpha_1\tilde F_{fit}+\alpha_2\tilde F_{nov}+\alpha_3\tilde F_{use}-\alpha_4\tilde F_{age}$，长期效用 $U^*(f)=\lambda S(f)+(1-\lambda)\bar\Delta(f)$，容量 $C_{max}$、相似度阈值与替换裕度控制更新，每 `am_interval` 代触发。第三条贡献是 **KA-guided 评测管线**（HeuBase 知识库 + KnoBase 基准）：批评现有 LHH 只评单函数，主张评测"设计完整 solver"。

## 2. 论文宣称的机制贡献（逐项）

- 结构层（GA）×函数层（MCTS）的双层进化，使 LLM 搜索可分解为"选结构"与"实现函数"两级。
- 自适应外部记忆按适应度/新颖性/使用度/年龄维护可复用函数，跨代注入生成上下文。
- KA-guided 评测管线与"设计完整求解器"的评测主张。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|BEAM 在所列 benchmark 有端到端收益|Tables `tab:traditionalmain`、`tab:CAF`、`tab:CombinedTable`、`tab:BBOBTable`|间接支持|完整系统相对基线的比较不能归因于双层或自适应记忆。|
|自适应记忆有贡献|Table `tab:ablation` 的 BEAM 与 BE；TSP-500、CVRP-500、Ackley/Rastrigin；`tab:stability`（5 次运行 BEAM 3.46±0.01 vs BE 4.41±0.19）|部分支持|该表的 "Adaptive Memory" 小节只比较 BEAM/BE，支持该命名版本差异及其稳定性；主文未逐项披露所有上下文/调用成本。|
|教育/搜索控制方式的影响|Table `tab:ablation` 的 One-Shot 与 MCTS（MIS、CVRP、CAF）|部分支持|CAF 上 MCTS 教育输给 One-Shot（8.17 vs 5.12，论文归因于任务简单、结构比函数重要）——教育方式的收益有任务条件；且这不是双层机制的单组件消融。|
|KA-guided 评测管线有效|`sections/benchmark.tex` 的 BBOB 与 PMSP/KnoBase 实验|间接支持|管线主张以案例与端到端实验呈现，无管线 on/off 消融。|
|记忆积累解释性能|案例/轨迹分析|间接支持|案例证明“发生过”，不证明是平均收益原因。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

阅读分析：质量分解 $Q(I)=Q_s+\sum_i Q_{f_i}$ 是双层的前提假设——结构选对时函数实现的边际贡献才可分离；结构错误时函数层的精炼预算全部浪费（与 BaSE 的"好族稀有时广度优先"同一逻辑）。记忆若存储可复用的失败模式、策略与其适用条件，可使进化从反复采样转向有条件复用；但记忆会形成路径依赖，且额外上下文与更大计算预算可伪造"记忆收益"。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：每条历史应区分程序、改动、证据和适用条件。前提：检索单位有 provenance。风险：把低质量摘要当知识。最小验证：固定提示长度，比较有 provenance 的轨迹摘要与随机历史。
- 可学习点：记忆更新应有淘汰条件。前提：能重新验证其价值。风险：上下文膨胀、确认偏差。最小验证：记录每条被检索记忆后的真实增益分布。

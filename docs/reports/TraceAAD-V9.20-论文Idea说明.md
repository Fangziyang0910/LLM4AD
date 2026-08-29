# TraceAAD V9.20：论文 Idea 说明

## 一句话贡献

TraceAAD 将自动算法设计建模为：在有限真实评价预算下，为已经测量过的算法状态购买连续改写机会，并将“机会给谁”与“这次机会如何辅助决策”明确分离；执行轨迹只作为行为坐标参与覆盖和参考检索，形成轨迹则保存算法改进的来时路。

## What

现有 LLM-AAD 搜索器通常把候选生成、候选排序和上下文拼接写在一个循环里。V9.20 把最小决策对象显式化：

1. **Opportunity Allocation** 根据质量分位、直接改写的延续价值和行为邻域覆盖度，决定下一个被改写的算法状态；
2. **Assisted Decision** 在已选状态上选择 Develop、Explore 或 Crossover，并提供与 action 匹配的形成路径、失败摘要或参考算法。

这两个机制都围绕同一个 evaluator 状态工作。算法不是一次性生成的候选集合，而是一个被反复测量、反复改写的状态。

## Why

AAD 的瓶颈不是单次生成能否写出一段合理代码，而是在固定评价预算中，哪些状态值得再次请求模型、下一次请求应携带什么证据。质量排序会反复购买已经显眼的状态；没有直接 outcome 账本时，搜索器也无法区分“还没有尝试”和“已经尝试但没有延续价值”。

V9.20 用直接机会账本估计 continuation value，用少量显式覆盖质量分布保留未充分开发的行为邻域，再将上下文按 Develop、Explore、Crossover 分工。这样，预算策略负责购买机会，辅助决策负责提高机会兑现为有价值 `Idea + Code` 的概率。

## So what

这给 LLM-AAD 一个更清晰的抽象：搜索器不是普通的候选排序器，而是**改写机会控制器**。该抽象能够解释为什么相同的生成模型在不同任务上需要不同的机会购买节奏，也为过程证据提供了直接接口：每个主槽位都有父节点、action、上下文证据、真实评价结果和最终信用。

## 方法结构

设有效状态为 `A_t`。机会分配使用：

`C_t(a) = (1 + improvements(a)) / (2 + opportunities(a))`

`H_t(a) = 0.5 Q_t(a) + 0.5 C_t(a)`

并用目标 ESS 的 Boltzmann 分布得到质量主分布。BehaveSim 邻域内的机会数形成覆盖分布，最终：

`p_parent(a) = 0.80 p_quality(a) + 0.20 p_coverage(a)`

选定父节点后，从行为距离和质量分位共同检索参考节点，计算三个 action utility 并以温度 `0.35` 采样。Develop 使用形成路径和完整直接 ledger；Explore 使用压缩 ledger 以减少路径锚定；Crossover 同时提供两个算法的代码、行为标签、距离和形成路径。

## 可验证预测

正式实验应围绕整体搜索行为与 held-out 质量展开，并分别记录：

- 机会概率是否随直接改善率和邻域覆盖变化；
- 三种 action 是否产生不同的上下文和改写结果分布；
- Crossover 是否真的形成新的、可执行的混合，而不是重复父节点；
- 行为覆盖是否减少预算集中在少数已显眼状态；
- 搜索曲线和 held-out 结果是否在三重复中同向改善。

这些是预注册的可检验预测，不是当前已得到的性能结论。

## 与已有 AAD 搜索的关系

V9.20 仍然使用树状形成关系、真实 evaluator 和有限预算，与 population、树搜索及 trajectory-guided AAD 兼容。新意在于把“再改写谁”和“如何帮助改写”作为两个可审查的策略层，并把执行轨迹限制在它能够支持的行为坐标职责内。形成路径不是算法语义标签；BehaveSim 距离也不是语义相似度或因果有效性证明。

## 贡献边界

论文可以主张一个明确的机制设计、可恢复实现和完整过程审计接口。V9.20 目前没有正式实验，因此不能主张相对 V9.16、V9.19 或其他方法的性能优势。若后续只完成联合版本运行，结论仍然应归因于整体 V9.20；组件效果需要匹配的单因素对照。

## 论文建议结构

1. Introduction：有限预算下的改写机会问题。
2. Problem Formulation：测量状态、主评价预算、形成路径和直接机会账本。
3. Opportunity Allocation：continuation value、目标 ESS 和行为覆盖混合。
4. Assisted Decision：Develop、Explore、Crossover 的 action-matched context contract。
5. Implementation and Reliability：tracked evaluator、duplicate、bounded repair、checkpoint。
6. Experiments：搜索曲线、held-out 泛化和过程审计；质量与机制证据分开。
7. Limitations：任务依赖、行为距离边界、联合机制的归因限制。

当前不添加未经程序化核验的论文引用；相关工作应从仓库已有参考资料和正式检索结果建立。

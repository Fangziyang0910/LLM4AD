# AAD 搜索机制综合：种群、树、轨迹与图

本文从搜索状态、信用分配和生成上下文三个正交维度，综合分析 EoH、ReEvo、HSEvo、MEoH、MCTS-AHD、ShinkaEvolve 和 PathWise，并据此定位 TraceAAD 的机制假设。重点不是复述论文叙事，而是区分哪些机制有实验证据、哪些只是动机解释，以及不同方法究竟把什么当作可继续投资的搜索资产。

## 1. 统一视角

LLM-based automatic algorithm design 是一种带反馈的昂贵程序搜索。每种方法都必须回答以下问题：

1. **搜索状态是什么**：单个程序、种群、树节点、轨迹、图状态，还是 Pareto set。
2. **搜索经验如何压缩**：只保留 fitness，还是同时保留代码、思想、父子关系、修改动作、反思、路径和多目标指标。
3. **预算如何分配**：精英选择、rank sampling、diversity-aware selection、UCT/UCB、bandit，还是 LLM policy planning。
4. **LLM 看到什么上下文**：当前个体、多个 parent、tree path、trajectory history、reflection、public metrics 或 parent metadata。
5. **什么会被保留**：最好个体、非支配个体、多样性个体、未完成潜力分支、可复用轨迹，还是完整 ancestry。

因此，种群法和树法的根本差异不只是数据结构，而是它们把搜索经验压缩成了不同的可采样对象。

更稳健的分析框架是三个正交维度：

| 维度 | 典型选项 |
|---|---|
| memory structure | active population、derivation tree、archive、trajectory library、semantic graph |
| credit assignment | endpoint fitness、best descendant、backprop、path return、stepwise delta、pairwise rank |
| generation context | current program、parent set、lineage、trajectory history、reflection、critic feedback |

同一个方法可以在不同维度上混合多种机制。例如，ShinkaEvolve 同时使用 archive、islands 和 UCB bandit；PathWise 同时使用 outer population 和 inner graph。因此，“population method”“tree method”更适合表示主要搜索单位，而不是互斥的算法类别。

### 1.1 统一视角的适用边界

“带反馈的昂贵搜索”作为统一框架有解释力，但不能把 AAD 简化成普通黑箱优化：

1. **搜索步长由 LLM 先验主导**。一次 forward pass 会把巨大的程序空间压缩到很窄的候选分布。因此 prompt/context 构造不是搜索外部的辅助模块，而是搜索策略本身。
2. **反馈不一定干净**。fitness 可能带有随机性、实例偏差、复杂度目标和 deceptive local optimum。不同 credit 规则对这些噪声的敏感性不同。
3. **搜索同时在抽取知识**。reflection、critic 和 derivation rationale 的产物不仅是候选程序，也可能是下一轮生成的经验。
4. **最终目标通常是可迁移算法，而不是单个 benchmark 上的最高分程序**。如果只使用 in-distribution fitness，搜索可能走向过拟合。

因此，AAD 方法应同时被看作程序搜索器和经验抽取器。搜索单位、credit 和 context 三者必须一起分析。

## 2. 种群方法的底层逻辑

种群方法把搜索经验压缩成 bounded active set。每个个体通常是一个已评估的 heuristic，保存代码、自然语言思想和 fitness。搜索通过 parent selection、LLM mutation/crossover、evaluation 和 survival 形成闭环。

### 2.1 EoH：思想和代码共同进化

EoH 将 heuristic 表示为 natural language description、code 和 fitness，并维护固定大小的 population。每一代通过五类 prompt strategy 生成候选，再按 fitness 保留下一代 population。

它的主要贡献是把 algorithmic thought 作为显式搜索对象，使 LLM 可以在高层机制和低层实现之间迁移。局限是 survival pressure 很强：暂时较差但包含有用思想的中间个体，可能在获得多步 refinement 之前就被淘汰。

### 2.2 ReEvo：反思作为 in-context 信号

ReEvo 仍以种群为主，但加入 pairwise comparison、short-term reflection、crossover、long-term reflection 和 elitist mutation。reflection 将评价反馈语言化，再注入下一轮 prompt。

论文把 reflection 描述成“verbal gradient”并声称它改变了 fitness landscape，但这一理论解释的证据较弱。已有 landscape 实验规模有限，且 neighborhood 定义本身包含 prompt 和 reflection，因此无法清楚区分“reflection 改变搜索空间几何”和“reflection 提供更好的上下文”。

更稳妥的结论是：reflection 的可复用机制是把 evaluation feedback verbalize 成更有信息量的 in-context signal，而不是已经被证明的 landscape 几何变换。其局限是历史经验主要被压缩成文本反思，而不是结构化保留为可分叉路径。

### 2.3 HSEvo：把多样性变成可测量变量

HSEvo 关注 objective 与 population diversity 的冲突。它通过代码清洗、格式标准化、code embedding 和 diversity metrics 测量 population 结构，再用 harmony search 优化高分个体的参数。

它的启发是：多样性不应只是直觉上的“看起来不同”，而应成为可测量、可控制的搜索变量。但它仍在 program population 层管理个体，不能直接回答一条改进路径中哪一步导致了改进或退步。

### 2.4 MEoH：从单目标精英主义到 Pareto 保留

MEoH 将 heuristic search 建模为 multi-objective optimization，维护 non-dominated set，并同时考虑 objective-space dominance 和 search-space dissimilarity。

它重新定义了“值得保留”：一个程序即使不是单一 fitness 上的最优，也可能因为速度、复杂度、泛化性或结构差异而有价值。对 TraceAAD 的启发是，轨迹价值也不应只由 endpoint fitness 决定。

## 3. 树方法的底层逻辑

树方法把搜索经验压缩成 ancestry tree。节点是 heuristic，边是 LLM action，预算主要由 tree policy 分配。节点质量、访问次数、UCT/UCB bonus 和 progressive widening 共同决定哪个分支继续被开发。

### 3.1 MCTS-AHD：树组织上下文，UCT 分配预算

MCTS-AHD 不只是 EoH 的“树版本”。它改变了搜索状态的记忆方式和评估信用的路由方式：

- EoH 主要维护 bounded active population；
- MCTS-AHD 保存 derivation tree 中的节点、访问次数和父子关系；
- selection 使用 UCT；
- expansion 通过 mutation、crossover 和 path action 生成子节点；
- simulation 评估子节点；
- backprop 将后代质量回传给祖先；
- progressive widening 控制节点何时继续扩展。

论文的主要动机是 population survival 可能过早丢弃较弱的中间 heuristic。但这一动机本身没有被充分验证：论文主要提供叙述和个别节点案例，没有统计“被淘汰的弱节点经过后续 refinement 后有多少真正变好”。论文还承认每个 heuristic 的扩展次数有限并依赖 LLM 单次生成质量，这与“通过多步 refinement 挽救弱节点”的叙事存在张力。

从已有消融更能确认的机制是：

1. **UCT 与 progressive widening 提供了有结构的预算分配**。progressive widening 的消融带来约 1.47pp 的差异。
2. **tree-path prompt 提供了多步 lineage 上下文**。s1 action 的消融带来约 1.26pp 的差异。

因此，更准确的概括是：MCTS-AHD 主要换了一种预算分配方式和上下文组织方式，而不是已经证明“保留弱节点”本身有效。

### 3.2 MCTS-AHD 的信用分配风险

MCTS-AHD 的 backprop 使用 best-child quality 的 max 语义，而不是标准均值；这一规则没有被充分消融。它可能产生 over-credit：某个早期节点只是偶然拥有一个强后代，却被沿 ancestry 赋予过高价值。

这暴露出树法的关键问题：节点知道 visit/value，却不一定知道一段修改序列是稳定改进、先升后跌，还是已经饱和。若要把 trajectory 作为搜索单位，必须把 fitness delta 归因到具体 step，而不是再次把整条路径压成一个没有解释的回传标量。

## 4. ShinkaEvolve 与 PathWise

### 4.1 ShinkaEvolve：工程化资源调度

ShinkaEvolve 更像可扩展的 evolutionary discovery scheduler。它维护 fixed-size archive 和 islands，从 archive/islands 中采样 parent 与 inspiration programs，支持 diff edit、full rewrite、crossover、novelty rejection、LLM bandit 和 meta-scratchpad。

它的主要价值在资源调度：

1. parent sampling 同时考虑 performance 与 novelty；
2. islands 和 migration 保护不同 discovery substreams；
3. novelty rejection 减少重复评估；
4. LLM bandit 学习不同模型或生成器在当前任务上的收益；
5. meta-scratchpad 将成功经验压缩成后续上下文。

ShinkaEvolve 对 TraceAAD 的启发是：trajectory-level sampling、diversity、novelty、operator value 和 model feedback 都可以被看作需要调度的资源。

### 4.2 PathWise：语义图和状态感知规划

PathWise 用 entailment graph 保存 heuristic 节点、derivation rationale 和 parent metadata，并将 heuristic evolution 建模为 MDP。outer population 控制状态规模，inner graph 保存局部派生过程；policy、world model 和 critic 共同参与生成与反馈。

其重要机制包括：

1. **动作语义化**：policy 生成自然语言 derivation rationale，而不是只在固定 operator 中选择；
2. **图状态规划**：parent selection 可以使用 derivation history、diversity 和上下文；
3. **critic 反馈**：policy critic 和 world model critic 比较不同 rollout，产生 routed feedback；
4. **混合管理**：outer population 与 inner graph 共同限制搜索规模。

需要谨慎区分的是：PathWise 的论文消融主要支持 critic feedback 和 prompt-level diversity，而没有直接比较 entailment graph 与扁平 population，也没有比较语义化 rationale 与固定 operator。因此，graph/MDP 本身的增量价值仍未被独立证明。

PathWise 还存在预算口径问题：一次 evaluation 背后包含多次 rollout 和 critic 调用，输入 token 可能是 baseline 的数倍，但论文主要控制 evaluation budget，没有完全控制 token/call budget。跨方法比较时必须匹配 token、LLM call 和 evaluation budget，否则无法区分“机制更好”和“调用更多”。

## 5. 四类搜索单位的比较

| 范式 | 搜索单位 | 经验压缩方式 | 预算分配 | 主要优势 | 主要风险 |
|---|---|---|---|---|---|
| Population | program individual | bounded population、archive、reflection | rank、elite、diversity、weighted sampling | 简单高效，适合 recombination 和规模化管理 | 暂时较差的中间思想容易被淘汰 |
| Tree/MCTS | heuristic node / branch | derivation tree、Q/N、backprop | UCT、progressive widening | lineage 显式，能重访非精英分支 | credit 回传和树结构可能限制语义表达 |
| Path/Trajectory | bounded improvement path | trajectory history、endpoint、step outcome | path score、trajectory UCB、prefix branching | 把改进过程形状作为信用对象 | 轨迹管理不当时会退化成短链 population |
| Entailment Graph | graph state / parent-set action | semantic graph、parent metadata、reflection | LLM policy、critic feedback | 语义规划能力强，能动态发明 action | agent/context 成本高，早期稳定性敏感 |

这个划分不是互斥分类。真实方法往往是混合体；更有解释力的比较单位仍然是：

$$
\text{memory structure}
\times
\text{credit assignment}
\times
\text{generation context}.
$$

## 6. TraceAAD 的定位

TraceAAD 应被定位为一种 **trajectory-as-individual** 的自动算法设计方法，而不是轻量 MCTS 或带轨迹的 EoH。它把一段有界的改进路径作为搜索单位，通过 endpoint quality、path quality、step outcome 和 UCB 探索项分配预算，并允许从轨迹内部的有效前缀重新分叉。

其核心假设是：

> 有用的算法思路可能存在于一条部分改进路径上，而不只存在于当前最优程序或单个树节点中。

TraceAAD 从不同范式中吸收了不同成分：

1. 从 population 方法借鉴 active set、archive、diversity、crossover 和 operator portfolio，但管理对象从 program 改为 trajectory；
2. 从 MCTS 借鉴 ancestry、UCB 和非精英重访，但不把全部预算绑定到树的 UCT/backprop；
3. 从 PathWise 借鉴结构化反馈和上下文组织，但保持轻量的 action/code 两阶段生成，不直接引入多 agent graph planning；
4. 从多目标搜索借鉴 Pareto survival，同时保留质量、路径潜力和多样性；
5. 从经验蒸馏方法借鉴将历史机制统计写回后续 prompt。

TraceAAD 的真正技术增量不应只是“增加 trajectory 数据结构”，而应体现在三个可验证的机制差异：

- **path-level credit** 是否比 endpoint-only 更能预测后续改进；
- **prefix branching** 是否能回收 population survival 丢掉的中间思想；
- **trajectory-level diversity and survival** 是否比 program-level 多样性更能避免搜索坍缩。

这些都是待验证假设，不应直接写成已被证明的优势。

## 7. 可验证假设与消融

后续实验应优先验证以下假设：

1. **Path value 假设**：path-level value 能比 endpoint-only 更早识别稳定改进方向。
2. **Prefix branching 假设**：当 endpoint 退步或 plateau 时，从内部 base node 分叉能恢复有价值的中间思想。
3. **Trajectory diversity 假设**：active trajectory 的机制多样性比 active program 的多样性更能避免坍缩。
4. **Operator portfolio 假设**：endpoint refinement、backtrack、crossover、simplify、novelty jump 和 scale transfer 在不同阶段具有不同收益。
5. **Multi-objective trajectory 假设**：将 quality、potential、novelty、diversity 和可选泛化信号用于轨迹管理，比单一 endpoint fitness 更稳健。

建议的消融包括：

- endpoint-only sampling；
- 去掉 path value；
- 去掉 prefix branching；
- 随机轨迹采样；
- 去掉 top-k softmax 或 elite protection；
- 去掉 diversity/novelty gate；
- 去掉 Pareto survival；
- 去掉 operator adaptation；
- 对比 scalar-only 与显式 per-instance generalization evidence。

所有跨方法比较都应尽可能匹配 evaluation budget、LLM call budget、token budget 和并发配置，否则无法区分机制收益与调用规模差异。

## 8. 论文依据索引

- EoH：papers/EoH/3-method.tex 描述 thought+code 表示、五类 prompt strategies、每代 5N 生成和 best N survival。
- ReEvo：papers/ReEvo/sections/04_evolution.tex 描述 short-term/long-term reflection、crossover 和 elitist mutation；sections/06_ablation.tex 给出 reflection 与 landscape 相关实验。
- HSEvo：papers/HSEvo/aaai25.tex 描述 diversity metrics、diversity/objective trade-off、flash reflection 和 harmony search。
- MEoH：papers/MEoH/MEoH.tex 描述 multi-objective heuristic search、non-dominated set 和 dominance-dissimilarity。
- MCTS-AHD：papers/MCTS-AHD/icml2025.tex 描述 MCTS tree、UCT、progressive widening、tree-path action 和 backprop；其中“保留弱节点”的动机与多步 refinement 的因果贡献仍需要独立验证。
- ShinkaEvolve：papers/ShinkaEvolve/sections/03_method.tex 描述 archive/islands、weighted parent sampling、novelty rejection、LLM bandit 和 meta-scratchpad。
- PathWise：papers/PathWise/example_paper.tex 描述 entailment graph、MDP view、policy/world model/critics 和 outer population + inner graph；graph 本身的独立消融仍不足。

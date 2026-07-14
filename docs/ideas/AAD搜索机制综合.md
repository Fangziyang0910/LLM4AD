# AAD 搜索机制综合：种群、树、轨迹与图

2026-07-08

这份笔记基于当前 `papers/` 中的 EoH、ReEvo、HSEvo、MEoH、MCTS-AHD、ShinkaEvolve 和 PathWise，抽象它们在自动算法设计中的底层搜索机制。这里关注每类方法如何保存搜索经验、如何分配评估预算、如何给 LLM 构造上下文，以及如何避免搜索坍缩。

> 重要前置：这些论文的"机制解释"很多是叙事性的。逐篇核实原文后发现，部分核心论断缺乏因果验证，有的是结构性自我矛盾或不可证伪的同义反复（具体见第 2、3、5 节）。本笔记在引用时已就地标注证据强度，不再把这些叙事当成已被证明的机制事实。

## 1. 统一视角

LLM-based automatic algorithm design 本质上是在巨大程序空间中做带反馈的昂贵搜索。所有方法都要回答同一组系统问题：

1. **状态对象是什么**：单个程序、种群、树节点、轨迹、图状态，还是 Pareto set。
2. **搜索经验如何压缩**：只保留 fitness，还是保留代码、思想、父子关系、修改动作、反思、路径和多目标指标。
3. **预算如何分配**：精英选择、rank sampling、diversity-aware selection、UCT/UCB、bandit，还是 LLM policy planning。
4. **LLM 看到什么上下文**：当前个体、多个 parents、tree path、trajectory history、reflection、public metrics、parent metadata。
5. **什么会被保留**：最好个体、非支配个体、多样性个体、未完成潜力分支、可复用轨迹，还是完整 ancestry。

因此，种群法和树法的区别不只是数据结构不同，而是它们把搜索经验压缩成了不同的可采样对象。

### 统一视角的适用边界

“带反馈的昂贵搜索”作为统一框架是成立且有解释力的，但它隐含把 LLM-based AHD 当作黑箱搜索，有四个维度没有被覆盖，而它们对 TraceAAD 的设计有直接后果：

1. **搜索步长由 LLM 先验主导，不是黑箱搜索**。LLM 自身带巨大程序先验，一次 forward pass 就把天文数字的程序空间压成一个极窄的子流形。因此实际被探索的不是“巨大程序空间”，而是“先验之后的残差空间”。推论：prompt / context 构造不是辅助，而是一等搜索策略——这解释了为什么 ReEvo 的 reflection、PathWise 的 entailment graph、以及 trajectory history 都被反复证明有用，它们都在改这个先验的形状，而不只是决定“LLM 看到什么上下文”。
2. **反馈本身是核心难点，不只是“有反馈”**。反馈常常带噪（seed / benchmark 采样）、多目标（efficiency / complexity）、甚至 deceptive（局部 fitness 高但泛化差）。推论：TraceAAD 的 `Q_path` 该如何估计，取决于是否承认反馈不干净；若反馈有噪，endpoint-only 的 `Q` 需要多次评估或贝叶斯化；若反馈多目标，trajectory score 不应塌缩成单一标量（对应第 8 节的 multi-objective trajectory 假设）。
3. **这些方法同时在做知识抽取，这是搜索之外的成分**。reflection 和 entailment graph 的产出不是“更好的候选”，而是“下一步该往哪走”——它们从搜索轨迹中蒸馏可复用的算法知识（verbal gradient、derivation rationale）。推论：trajectory 在 TraceAAD 中同时承担“搜索单位”和“知识载体”两个角色，把这两者显式区分（“这条路径值多少钱” vs “这条路教会了什么”）可能是一个真实的创新方向。
4. **终极目标是泛化的算法，不是 benchmark 最优程序**。纯搜索若只盯 in-distribution fitness，会滑向 overfitting。推论：trajectory 的价值最终应包含某种泛化信号，而非仅仅是当前 benchmark 上的 fitness 序列。

因此，更精确但更啰嗦的表述是：LLM-based AHD 是在一个被 LLM 先验压缩过的、巨大但实际只被窄探索的程序空间上，做带噪、多目标反馈的昂贵搜索，并同步从搜索轨迹中蒸馏可复用的算法知识，以泛化为终极目标。作为统一视角的入口，原文那句话足够锋利，不必替换；以上四点是它的适用边界，也是后续设计 `Q_path`、credit assignment 和评估协议时应锚定的约束。

## 2. 种群方法的底层逻辑

种群方法把搜索经验压缩成一个 bounded active set。每个个体通常是一个已评估的 heuristic，保存代码、自然语言思想和 fitness。搜索通过 parent selection、LLM mutation/crossover、evaluation、survival 形成闭环。

### EoH：思想和代码共同进化

EoH 的关键是把 heuristic 表示成 natural language description + code + fitness，并维护大小为 `N` 的 population。每一代调用五类 prompt strategy 生成最多 `5N` 个新 heuristic，再选择 best `N` 进入下一代。它的 E1/E2 做 exploration，M1/M2/M3 做 modification、parameter tuning 和 simplification。论文中明确写到 EoH 同时维护 thought 和 code，并通过 crossover/mutation/selection 演化 population；每代保留 best `N`。

EoH 的优势是简单、灵活、采样吞吐高，且通过 thought+code 让 LLM 能在高层机制和低层实现之间来回迁移。它的局限是 survival pressure 很强：暂时低分但有潜力的中间个体容易被淘汰，后续多步 refinement 的机会不足。

### ReEvo：反思作为 in-context 信号（“语言梯度”是未证伪的比喻）

ReEvo 仍是种群搜索，但把 pairwise comparison 和 accumulated reflection 加入搜索闭环。每轮包含 selection、short-term reflection、crossover、long-term reflection、elitist mutation。short-term reflection 比较两个 parent 的相对表现，long-term reflection 累积经验，再用于改进当前 elite。

论文把 reflection 包装成“verbal gradient”，声称它平滑了 fitness landscape、提升了 correlation length。但核实原文后这个说法的证据很弱：唯一的 landscape 实验只在 TSP50/ACO 上跑了 3 runs × 40 步随机游走，只报了两个标量、没有 landscape 图，且只测了 short-term reflection。更结构性的是，论文把 neighborhood 定义为 `N(h)={h|LLM(h|h_c,x)>ξ}`，把 prompt（含 reflection 文本）直接嵌进了 landscape 的定义里，于是“reflection 改变了 landscape 几何”和“reflection 只是给 LLM 喂了更好的 in-context 示例”在它的框架里成了同一句话，不是两个可分离假设——这个比喻因此几乎不可证伪。ablation 也显示 reflection 的实际收益不大：short-term reflection 的 objective 提升基本在噪声内（white-box +0.06、black-box +0.09），只有 long-term reflection 在 black-box 下贡献相对明显（约 0.36）。

因此更准确的读法是：reflection 真正起作用的方式，很可能是把 evaluation feedback verbalize 成更有信息量的 in-context 信号，而不是真的改变了某个搜索空间的几何。它对 TraceAAD 可复用的启发是“把反馈语言化、喂回 context”这个具体操作，而不是“verbal gradient”这个理论框架。局限依旧：主搜索单位仍是当前 population 个体，历史被压缩成 reflection，而不是结构化保留为可分叉的路径。

### HSEvo：把多样性变成被测量和控制的变量

HSEvo 的出发点是：EoH 多样性高但 objective 不稳定，ReEvo objective 好但 diversity 不够。它提出用代码清洗、格式标准化、code embedding 和 Shannon-Wiener / cumulative diversity metrics 来测量 population diversity，并用 harmony search 优化 best individuals 的参数，缓解 diversity 和 exploitation 的冲突。

HSEvo 的优势是把“探索与利用”从直觉变成可观察变量。它说明仅仅增加 diversity 不够，diverse structures 还需要 local parameter optimization 才能转化为 objective gain。局限是它仍在 population 层管理个体，多样性主要基于代码/embedding 分布，路径中哪些动作导致改进或退步没有成为独立信用对象。

### MEoH：从单目标精英主义到 Pareto 保留

MEoH 指出已有 LLM-based heuristic search 多数只优化单一 performance objective，忽略 efficiency、complexity 等实践目标。它把 heuristic search 建模为 multi-objective optimization，维护 non-dominated set，并提出 dominance-dissimilarity mechanism：同时考虑 objective-space dominance 和 search-space code dissimilarity，用于 parent selection 和 population management。

MEoH 的优势是重新定义“好”的含义。一个 heuristic 不一定要在单一分数上最优，可能因为速度、复杂度、泛化性或结构差异而值得保留。它给 TraceAAD 的启发是：轨迹价值最终不应只看 endpoint fitness，也可以扩展为 endpoint quality、path quality、runtime、complexity、diversity 的组合。

## 3. 树方法的底层逻辑

树方法把搜索经验压缩成 ancestry tree。每个节点是 heuristic，每条边是一次 LLM action。预算不再主要由 population survival 决定，而由 tree policy 决定：节点的 quality、visit count、UCT/UCB bonus 和 progressive widening 共同控制哪个分支继续被开发。

### MCTS-AHD：用树组织上下文 + UCT 分配预算（“保留弱节点”是未验证的动机）

MCTS-AHD 名义上的动机是：population-based AHD 会直接丢弃 lower-performance heuristics，而这些 heuristic 可能经过多步 refinement 后变好。但核实原文后这个动机其实没有被验证：全篇只有 motivation 段落和附录一个挑出来的 `t=611` 节点例子，没有任何“被丢弃的弱启发式继续扩展后有多少变好”的统计；而且论文自己在解释 black-box 表现差时承认，MCTS-AHD 对每个启发式只做有限次扩展、高度依赖 LLM 单次生成质量——这和“靠多步 refinement 救回弱节点”的叙事是矛盾的。

机制本身：节点存储 `Q` 和 `N`，selection 使用 normalized quality + exploration term，expansion 通过 e2/m1/m2/s1 等 LLM action 生成 children，simulation 评估 child，backprop 用 best child quality 更新 ancestors，progressive widening 重新扩展非叶节点。它的真正有效成分，从能拿到的 ablation 看，更可能是下面两点而非“保留弱节点”：

1. **预算分配有统计结构**：UCT 用 value 和 visit count 平衡探索/利用，exploration decay 让早期更探索、后期更收敛。progressive widening 单独 ablate 贡献约 1.47pp（TSP50）。
2. **tree-path prompt 利用 lineage**：s1 action 可以分析 root-to-leaf path 中的多个 heuristic，提取有益设计。它单独 ablate 贡献约 1.26pp。两者量级已接近 MCTS-AHD 相对 EoH 的总优势（约 1.8pp），也就是说，“保留弱节点”几乎没有可归因的贡献，树更多是作为 prompt 的组织方式 + UCT 的预算容器在起作用。

局限也因此更具体：

1. **“非精英分支仍可被开发”这条名义优势未被证实**：没有任何实验证明被保留的弱中间节点真的带来了收益。
2. **backprop 用 max（best child）而非标准均值，且从未被 ablate**：这极可能在 over-credit 祖先——把“碰巧有强后代”的功劳算给了其实无贡献的早期节点。这是 MCTS credit misassignment 的经典病，也是 TraceAAD 设计 path-level credit 时必须直接处理的隐患。
3. **节点局部语义偏弱**：UCT 知道 visit/value，但不知道一段修改序列是稳定改进、先升后跌，还是已经饱和。

所以更准确的说法不是“树法解决了 population 的过早丢弃”，而是“树法换了一种预算分配和上下文组织方式”；它仍把核心搜索单位放在 node 和 branch 上，没有把一段改进过程本身作为可评分对象。

## 4. ShinkaEvolve：工程化资源调度视角

ShinkaEvolve 更像一个可扩展的 evolutionary discovery scheduler。它维护 fixed-size archive、islands、elite constraint，从 archive/islands 中采样 parent 和 inspiration programs；mutation 包含 diff edit、full rewrite、crossover；novelty rejection 用 code embedding similarity 和 LLM novelty judge 拦截近重复候选；execution feedback 又驱动 UCB1-based LLM ensemble 和 meta-scratchpad。

它的底层优势不是单一搜索公式，而是把真实系统中的资源调度做完整：

1. parent sampling 同时考虑 performance 和 novelty，避免 hill climbing 过早 plateau。
2. islands 和 migration 保护不同 discovery substreams。
3. novelty rejection 减少重复评估，提升评估预算效率。
4. LLM bandit 把“哪个模型更会在当前任务上产生改进”也变成可学习对象。
5. meta-scratchpad 把 recent successful solutions 压缩成全局提示经验。

ShinkaEvolve 给 TraceAAD 的启发是：当轨迹机制成立后，下一步不是盲目增加 prompt，而是把 trajectory-level sampling、diversity、novelty、operator value 和 model/value feedback 作为可调度资源。

## 5. PathWise：语义图和状态感知规划（graph 本身的价值未被实验证实）

PathWise 与 TraceAAD 关系最近。它批评 population methods 的 fixed rules 和 tree methods 的 UCT/statistical tree 缺少 semantic derivation memory，提出用 entailment graph 作为 compact stateful memory：每个节点保存 `(h, kappa, description, performance, parent metadata)`，边记录 parent set 在 derivation rationale `kappa` 下如何 entail child。它把 heuristic evolution 建模为 MDP：state 是 graph/frontier，action 是 `(parent set, derivation rationale)`，transition 是 world model 生成新 heuristic，reward 来自 performance 并转化为 critic feedback。

需要警惕的是：entailment graph / MDP 这套核心机制在论文里没有任何 ablation。所有消融（critic、prompt 级多样性、超参数）都建立在“图已经存在”的前提下，从未对比过“图 vs 扁平 population”，也没对比“语义化 κ vs 固定算子”。真正被实验支持的提升来源是 Policy Critic 反馈 + prompt 级多样性，而这两者在原则上根本不依赖图结构。更麻烦的是 budget 混淆：PathWise 的每次 evaluation 背后是多次 rollout + critic 调用，输入 token 是 baseline 的 2–5 倍，但论文只控制了 evaluation budget、没控制 token/call budget，因此无法排除“提升主要来自更多更精细的 LLM 调用”这个替代解释。

它的设计可以这样读：

1. **动作语义化**：policy agent 生成自然语言 derivation rationale，而非只在固定 operator 里选——但“语义化 vs 固定算子”未做对比。
2. **图状态规划**：parent selection 可以考虑 derivation history、diversity 和上下文——但图相对扁平 population 的增量价值未知。
3. **多智能体反馈**：policy critic 排 action，world model critic 对比 best/worst rollout，形成 routed reflections——这是被 ablation 支持的主要收益来源。
4. **混合图和种群**：outer population 控制状态规模，inner graph 保留派生过程。

成本也实在：多 agent/多 action/多 rollout 使 prompt/context 管理更重；早期 stochasticity 会影响 graph 质量和 critic 对比信号（论文 Limitations 自己承认）；输入 token 明显增加，过大的 population 或过多 action/rollout 会削弱 reasoning。

因此，说“PathWise 证明了 derivation memory 和 state-aware planning 的价值”是不准确的——它没有把图结构本身和“更多调用 + 更精细反馈”分开。对 TraceAAD 真正可复用的，是“routed critic 反馈 + 多 rollout 对比 + 多样性 prompt”这套不依赖图的机制；而“图/MDP 作为状态化记忆”是否值得，仍是一个开放问题。这也提醒：TraceAAD 的任何 baseline 对比都必须 token/call-budget-matched，否则无法区分“机制更好”和“调用更多”。

## 6. 三类核心搜索单位的本质区别

| 范式             | 搜索单位                        | 经验压缩方式                                              | 预算分配                                       | 最强优势                                  | 主要风险                                        |
| ---------------- | ------------------------------- | --------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------- | ----------------------------------------------- |
| Population       | program individual              | bounded active population + optional archive/reflection   | rank/elite/diversity/weighted sampling         | 简单高效，适合 recombination 和规模化管理 | 暂时差的中间思想容易被淘汰                      |
| Tree/MCTS        | heuristic node / branch         | full derivation tree + Q/N/backprop                       | UCT + progressive widening                     | 保留 lineage，能继续开发非精英节点        | 树路径和回传规则可能限制语义表达                |
| Path/Trajectory  | bounded improvement path        | trajectory history + endpoint + step outcomes             | path score + trajectory UCB + prefix branching | 把“改进过程形状”变成信用对象            | 需要设计好轨迹管理，否则会退化成短链 population |
| Entailment Graph | graph state / parent set action | semantic derivation graph + parent metadata + reflections | LLM policy + critic feedback                   | 语义规划能力强，能动态发明 action         | agent/context 成本高，早期稳定性敏感            |

从第一性原理看，它们的根本差异是“什么被认为是值得继续投资的资产”：

- 种群法认为资产是当前高质量或高多样性的 candidate。
- MCTS 认为资产是有访问统计和子孙潜力的 branch。
- PathWise 认为资产是可被 policy/critic 解释的 graph state。
- TraceAAD 应该认为资产是一段可复用的 improvement trajectory：它包含 endpoint，也包含哪些动作让算法变好、变坏、停滞，以及从哪里回退更合理。

不过这个范式划分是软的：真实方法都是混合体（ShinkaEvolve 已经把 archive/islands/UCB bandit 混在一起，PathWise 是 outer population + inner graph），把它们塞进四个格子会掩盖实际机制。更扎实的第一性原理框架其实是第 1 节那三个正交维度的组合空间——memory structure × credit assignment × generation context——而不是四个离散范式。另外，几乎所有论文的跨方法对比都只控制了 evaluation budget、没控制 token/call budget，因此“某范式优于另一范式”的结论普遍带有“调用更多”的混淆，TraceAAD 复现对比时必须配对预算。

## 7. 对 TraceAAD 的定位

TraceAAD 不应被定位为“轻量 MCTS”或“带轨迹的 EoH”。更准确的定位是：

> TraceAAD 是一种 trajectory-as-individual 的自动算法设计方法。它把一段有界改进路径作为搜索单位，通过 endpoint quality、path quality、step outcome 和 UCB 探索项来分配预算，并允许从轨迹内部的有效前缀重新分叉。

但有一个前提必须先讲清楚：TraceAAD 整条路线的地基——「有用的算法思路往往藏在部分改进路径上，而不在当前最优程序或单个节点里」——是一个整个领域都没有用统计验证过的假设。MCTS-AHD 本可以用一个极简单的实验验证它，却选择了讲故事（见第 3 节）。所以 TraceAAD 在“造方法”之前，应该先“做诊断”：这个假设若不成立，trajectory-as-individual 的价值就不存在。

它试图同时吸收三类方法的优点：

1. 从 population 方法借鉴 active set、archive、diversity、crossover、operator portfolio，但管理对象从 program 变成 trajectory。
2. 从 MCTS 借鉴 ancestry、UCB 和非精英重访，但不把预算分配完全绑定到整棵树的 UCT/backprop。
3. 从 PathWise 借鉴“把反馈 verbalize / 结构化保留”的思路，但保持轻量的两阶段 action/code generation，而不是直接引入多 agent graph planning（何况 graph 本身的价值未被证实）。

这样，TraceAAD 的潜在优势可以说清楚——但每一条都应被当成待验证假设，而不是既定优势：

- **是否真能回收 population 丢掉的中间思想**：核心待验证项。方法是用已有 MCTS 树做诊断（见下方），看被保留但从未进 elite 的弱中间节点，其后代有多少最终超过当时 elite；回收率低，则 prefix branching 的动机就要重写。
- **path-level credit 是否比 endpoint-only 更有预测力**：用 endpoint-only / path-mean / path-max / stepwise-attributed 四种 Q 打分，看哪种与“该子树最终产出好解”相关性最高。若 path 类 Q 没超过 endpoint Q，所谓“第三轴”就不存在。
- **是否避免了 MCTS 的 credit misassignment**：MCTS-AHD 的 max-backprop 从未被 ablate，可能 over-credit 祖先。TraceAAD 若做 path credit，必须用 stepwise attribution（把 fitness delta 归因到具体 step），而不是再发明一个 path 级 aggregate 标量——这才是 trajectory 相对 node 的真正信息增量，也是真正的技术难点。
- **缓解 tree 的结构约束**：用全局 derivation graph 保存 ancestry，但采样对象是 bounded trajectory，不必让所有信用都沿树回传。
- **比 PathWise 更简单可消融**：当前机制只依赖 node、edge、action、fitness change、base node 和 trajectory score，容易在 LLM4AD 平台上做 endpoint-only、no-path、no-branching、random-trajectory 等消融。

诊断实验几乎是零成本的：手上已有的 `LLM4AD/experiments/.../logs/mcts_state.jsonl`（完整树状态）和 `mcts_events.jsonl`（selection/expansion/backprop 事件）就足以回答上面前两个问题，不需要先实现完整 TraceAAD。

## 8. 可验证假设

后续实验应围绕这些机制假设，而不是只看最终 best score。其中假设 1、2 是 TraceAAD 的存亡判断，且是整个领域都没人验证过的，应该最先用第 7 节的诊断实验回答，再决定要不要造完整方法：

1. **Path value 假设**：`Q_path` 能比 endpoint-only 更早识别稳定改进方向。（领域未验证）
2. **Prefix branching 假设**：当 endpoint 退步或 plateau 时，从内部 base node 分叉能恢复有价值中间思想。（领域未验证，且 MCTS-AHD 的“保留弱节点”动机也未被证实）
3. **Trajectory diversity 假设**：active trajectories 的机制多样性比 active programs 的多样性更能避免搜索坍缩。
4. **Operator portfolio 假设**：endpoint mutation、backtrack branching、trajectory crossover、simplify/distill、novelty jump 在不同搜索阶段有不同收益。
5. **Multi-objective trajectory 假设**：把 runtime、complexity、novelty 加入 trajectory management 可以减少高分但低效或脆弱的 heuristic。

这些假设对应的消融应包括：endpoint-only sampling、去掉 `Q_path`、去掉 prefix branching、随机轨迹采样、去掉 top-k softmax、无 archive、无 diversity/novelty gate、无 operator adaptation。所有跨方法对比都必须在 token/call-budget-matched 下进行，否则无法区分“机制更好”与“调用更多”。

## 9. 论文依据索引

- EoH: `papers/EoH/3-method.tex` 描述 thought+code 表示、五类 prompt strategies、每代 `5N` 生成与 best `N` survival。
- ReEvo: `papers/ReEvo/sections/04_evolution.tex` 描述 short-term/long-term reflection、crossover、elitist mutation；`sections/06_ablation.tex` §6.1 给出 neighborhood 定义 `:11` 与 landscape 实验（仅 TSP50/ACO、3 runs×40 步、无图），§6.2 Table 2 给 reflection 的 ablation 收益（short-term 在噪声内、long-term black-box ~0.36）。
- HSEvo: `papers/HSEvo/aaai25.tex` 描述 diversity metrics、diversity/objective trade-off、flash reflection、harmony search。
- MEoH: `papers/MEoH/MEoH.tex` 描述 multi-objective heuristic search、non-dominated set、dominance-dissimilarity。
- MCTS-AHD: `papers/MCTS-AHD/icml2025.tex` 描述 population 丢弃低分 heuristic 的 motivation（`:161-164`、仅一个 `:1277` anecdote）、MCTS tree、UCT、progressive widening（ablate ~1.47pp）、tree-path s1 action（ablate ~1.26pp）、max-backprop（`:268-273`，从未 ablate）；§B `:1269` 与“多步 refinement”叙事自相矛盾。
- ShinkaEvolve: `papers/ShinkaEvolve/sections/03_method.tex` 描述 archive/islands、weighted parent sampling、novelty rejection、LLM bandit、meta-scratchpad。
- PathWise: `papers/PathWise/example_paper.tex` 描述 entailment graph、MDP view、policy/world model/critics、outer population + inner graph；ablation 仅覆盖 critic（`:836-886`）与 prompt 级多样性，graph 本身无 ablation；附录 Cost Analysis（`:2513-2634`）显示输入 token 为 baseline 2–5 倍，仅控 evaluation budget。

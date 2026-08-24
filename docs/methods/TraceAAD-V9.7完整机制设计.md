# TraceAAD V9.7：轨迹条件的自动算法设计进化

V9.7 的实现由两条相互作用的机制主线组成：**轨迹感知的计算分配**与**轨迹条件的单步生成**。

## 1. 研究对象：在问题上设计可复用算法

自动算法设计（Automatic Algorithm Design, AAD）研究的对象不是某一个实例上的解，而是：给定一个问题及其结构约束，构造一个能够反复求解该问题的可执行算法，并使它在目标评价下具有尽可能好的表现。

一个可执行的 AAD 任务由三部分组成：问题描述、算法模板与可编辑范围、以及已知的评价器：

$$
\mathcal T_{\mathrm{AD}}=(d_{\mathcal T},K,\mathcal E).
$$

其中 $d_{\mathcal T}$ 说明问题、输入输出和约束，$K$ 给出完整程序骨架及允许设计的部分，$\mathcal E$ 运行完整程序并返回可比较的质量。给定候选算法 $r$，模板实例化为 $P_r=K[r]$，评价器在搜索实例集 $S_{\mathrm{search}}$ 上给出：

$$
q(r)=\begin{cases}
\mathcal E(P_r;S_{\mathrm{search}}), & \mathrm{maximize},\\
-\mathcal E(P_r;S_{\mathrm{search}}), & \mathrm{minimize}.
\end{cases}
$$

传统算法设计依赖专家理解问题结构、提出算法思想、完成实现并反复实验，成本高且难以系统覆盖设计空间。NFL 定理说明，不存在对所有问题都普遍最优的算法；可用的算法必须利用目标问题的结构。因此 AAD 的目标是降低面向特定问题进行算法设计的成本，而不是寻找脱离问题的通用最优程序。

## 2. 从 AAD 到 evaluator 驱动的进化搜索

LLM4AD 将 LLM 放入一个 evaluator 驱动的算法进化过程。候选算法是进化个体，LLM 负责提出语义层面的变异或重组，评价器提供外部适应度事实，搜索控制器决定哪些候选继续获得生成机会：

$$
\text{select parent and context}
\rightarrow
\text{LLM proposes Idea + Code}
\rightarrow
\text{execute complete program}
\rightarrow
\text{evaluator scores candidate}
\rightarrow
\text{record formation facts and reallocate}.
$$

这里的“进化”是广义的程序进化。它继承了进化搜索的基本问题：在有限评价预算下，如何在不同候选路线之间分配机会，如何在已有方向内继续开发，同时保留产生新方向的可能。树、种群、谱系和预算分配属于这一搜索范式的控制结构。

AAD 中的个体是可复用的算法程序。LLM 根据问题说明、当前代码和上下文完成语义变异，评价器检查完整程序的有效性、约束和性能。

因此，V9.7 的机制来源可以分成三层：

| 层次 | 继承或解决的问题 | V9.7 中的对应机制 |
| --- | --- | --- |
| 进化搜索共性 | 个体、变异、适应度、选择、探索—利用、有限预算 | 程序候选、LLM 生成、evaluator、分配器、Refine/Explore |
| AAD 特有 | 设计可复用算法，而非只求单实例解；满足代码和运行约束 | 算法模板、完整程序评价、`Idea + Code`、有效性与 held-out |
| TraceAAD 的研究对象 | 改进轨迹是否能改善进化中的下一步分配与生成 | 轨迹感知分配、父代来时路、轨迹条件的 LLM 算子 |

## 3. TraceAAD 的科学问题与贡献

一次 LLM 生成通常只能完成一个局部设计动作。TraceAAD 的出发点是：算法设计不是彼此独立的代码采样，而是逐步引入思想、试错、修正和精炼的形成过程。当前算法的形成轨迹包含了代码本身之外的设计信息，例如某个机制从何处引入、实际改动了什么、结果是改善还是退步。

TraceAAD 将这一过程视为一种**近似马尔可夫的算法设计状态**：当前代码仍是必要状态，但要让下一步决策更有依据，还需要与当前锚点匹配的形成历史。该假设不是把 AAD 变成一个抽象的 MDP，也不要求每次评价都回答预先定义的问题；它服务于 AAD 的核心目标：在固定问题和有限评价预算下生成更好的可复用算法。

V9.7 的研究贡献可表述为三个相互连接的命题：

1. **将改进轨迹作为生成条件。** 轨迹不只保存候选的谱系关系，还被压缩为当前锚点的父代改进来时路，作为 LLM 下一步算法变异的条件。它回答的是：给定当前算法，模型能否利用算法怎样形成来提出更可靠的下一步修改。
2. **将有限评价预算分配到轨迹结构。** 搜索器先决定继续哪一个来源区域，再决定该区域内从哪个形成状态出发。它回答的是：在算法候选具有不同发展阶段和质量时，有限的 evaluator 调用应如何支持已有方向的继续进化，并保留结构性探索入口。
3. **把两者放入一个原子、即时反馈的进化协议。** 每次只选择一个锚点、抽取一个生成意图、生成一个完整候选、进行一次真实评价并立即更新。这样轨迹信息改变生成，评价事实改变后续分配，二者通过进化循环耦合。

V9.7 的 novelty 不在于单独提出树、UCB 形式或探索—利用这几个进化搜索常见概念，而在于把**算法改进轨迹的内容**放入 AAD 的 LLM 变异条件，并将**轨迹状态的投资**作为有限预算下的独立控制问题。EoH 已经区分探索与开发生成，ReEvo 已经使用反思和历史摘要，MCTS-AHD 已经使用树搜索分配评价，BaSE 已经研究跨轨迹的计算分配；V9.7 试图把“轨迹如何改变下一步生成”和“预算应投向哪条形成路线”放进同一个算法设计进化过程。

## 4. 两条核心机制

### 4.1 轨迹感知的计算分配：决定从哪里继续进化

给定当前搜索森林，分配机制 $\mu(a_t\mid\mathcal H_t)$ 决定下一次 LLM 响应从哪里启动。它不改变固定锚点和生成意图下的候选分布；它决定有限预算落在哪些来源、哪些算法状态上，从而影响搜索的整体进化走向。

V9.7 将分配分为路线层和锚点层。路线是初始来源对应的全部形成状态，锚点是某份程序在一条具体形成路径上的位置。先选路线，再在该路线内选锚点：

$$
S_t^{\mathrm{route}}(r)=q_t^*(r)+\frac{s}{\sqrt{N_t(r)+1}},
\qquad
S_t^{\mathrm{anchor}}(a)=q(a)+\frac{s}{\sqrt{n_t(a)+1}}.
$$

其中：

- $q_t^*(r)=\max_{a\in r}q(a)$ 是路线已经达到过的最好质量；
- $N_t(r)=\sum_{a\in r}n_t(a)$ 是路线累计获得的生成机会；
- $q(a)$ 是锚点当前程序的质量；
- $n_t(a)$ 是从该锚点直接发起的生成次数；
- $s$ 是初始化阶段一步变化尺度的固定估计。

路线层承担来源之间的深度—广度权衡，锚点层承担同一路线内部的状态回访与局部开发。两层使用相同的形式，但比较对象不同。路线保存的是 provenance，不自动等同于算法簇；同一路线内部仍可能发生重要的算法机制迁移。

#### 初始化与尺度

搜索开始时建立 $K=8$ 条代码互异的初始路线。每个根随后接受一次 Refine bootstrap。对 bootstrap 中成功形成新子节点的转换，取父子有向质量差的绝对值；改善、退步和持平都计入：

$$
s=\begin{cases}
\operatorname{median}(D_{\mathrm{init}}), & D_{\mathrm{init}}\neq\varnothing,\\
0, & D_{\mathrm{init}}=\varnothing.
\end{cases}
$$

$s$ 是任务内一步修改幅度的启发式尺度，不是路线潜力、置信区间或未来边际收益的估计。正式搜索开始后不再重估。

### 4.2 轨迹条件的单步生成：决定如何继续进化

选定锚点后，生成机制控制条件提议分布：

$$
P(x_{t+1}\mid x_t,h_t,o_t),
$$

其中 $x_t$ 是当前完整算法，$h_t$ 是从根到当前锚点的父代形成历史，$o_t$ 是本轮生成意图。该机制决定候选更可能在当前算法方向内发展，还是更可能提出结构上不同的设计。

#### 父代改进来时路

默认上下文包含当前完整算法和该锚点的父代形成路径。路径沿唯一父链回溯，保留最近至多 8 条真实形成事件。每条事件包括：

````text
[History i] Formation step
Idea: ...
Change: ...
Result: improve | regress | plateau
Fitness: parent -> child
````

`Idea` 是当时生成时声明的设计意图；`Change` 由父子实际代码推导；`Result` 和 `Fitness` 来自真实评价。形成历史说明当前算法怎样走到这里，不把事后结果倒灌为当时的信用。

直接子代尝试仍完整保存在搜索事实中，但不默认放入当前生成提示。它们描述的是从该锚点试过什么，而不是当前程序怎样形成。两类信息在作用上分开：来时路主要提供设计形成条件，直接尝试主要提供可审计的局部搜索记录。

#### Refine 与 Explore

V9.7 用两个固定生成意图区分族内开发与结构性探索：

- **Refine**：沿当前设计方向做一次聚焦修改，利用当前代码及其改进历史继续发展；
- **Explore**：寻找实质不同的改进方向，可以替换或重组当前设计的重要部分。

意图概率固定为：

$$
P(\mathrm{Refine})=0.7,\qquad P(\mathrm{Explore})=0.3.
$$

两种意图共享任务说明、当前代码、父代来时路和输出契约，只改变本轮修改目标。Refine 旨在提高当前算法族内的局部开发命中率，Explore 旨在保留跨方向提议的入口。固定比例是 V9.7 的控制条件，不是已经证明的最优调度策略。

#### 候选和反馈

每次分配只生成一个候选。一次 LLM 响应输出可选的短 Idea 和一份完整、可执行的程序。完整程序是有效性的硬条件；Idea 缺失不使候选无效。新候选由 evaluator 真实运行并按有向质量标记为 `improve`、`plateau`、`regress` 或 `invalid`。这些标签是形成事实，不形成额外奖励，也不沿祖先回传。

## 5. 搜索状态

V9.7 使用由 8 个根组成的森林，每个根下维护单亲形成树。搜索状态分为程序、锚点和生成尝试：

- **程序（program）**：一份实际评价过的完整代码及其真实质量。相同代码复用评价结果。
- **锚点（anchor）**：程序沿某条父链形成的位置，绑定父节点、形成事件和访问次数。同一代码沿不同路径到达时，可以形成不同锚点。
- **尝试（attempt）**：一次从锚点发起的 Idea + Code 生成及其评价结果。无效、空操作、重复和退步都保留。

程序、锚点和尝试分别记录代码、形成路径和已经发起的生成。搜索树由此构成形成事实的记录与条件化状态空间。

若候选代码与当前锚点相同，或回到祖先程序，则记录尝试但不创建新状态。若同一程序已在其他路线出现，复用评价结果，并可在当前历史上创建新锚点。因此程序数、锚点数和尝试数分别回答不同问题。

## 6. 完整运行协议

正式搜索使用每次运行 1000 次真实评价。新程序消耗一次评价；已评价过的相同程序复用结果。评价预算与 LLM 调用数、token 数和墙钟时间分开记录。

````text
Input: task, evaluator, LLM, real evaluator budget B = 1000

Generate K = 8 unique valid roots; create one root anchor each.
For each root, generate one bootstrap candidate with Refine.
Set s = median |q(child) - q(root)| over valid bootstrap transitions
    (s = 0 if none).

While evaluator budget remains:
    Score every route by q*(route) + s / sqrt(N(route) + 1).
    Select the highest-scoring route.
    Within that route, score every anchor by q(anchor) + s / sqrt(n(anchor) + 1).
    Select the highest-scoring anchor.

    Build the selected anchor's parent improvement path, at most 8 events.
    Draw Refine with probability 0.7, otherwise draw Explore.
    Generate one optional Idea and one complete program.
    Increment the selected anchor's access count.
    Evaluate a new program or reuse a cached result.
    Record the attempt and create a child anchor only for a valid new relation.
    Update facts and reselect.

Return the best unique program by the true objective.
````

同分时优先访问次数更少、创建更早的对象。停止条件是评价预算耗尽，不因连续无改善提前停止。最终程序只按真实质量排序；路线、访问次数和生成意图不参与最终排序。

## 7. 两条机制的分工

V9.7 中两条核心机制的职责可以明确写成：

| 机制 | 直接控制的对象 | 在 AAD 中承担的作用 |
| --- | --- | --- |
| 轨迹感知分配 | $\mu(a_t\mid\mathcal H_t)$ | 决定有限评价预算投向哪些来源和算法状态，影响整体进化路径 |
| 轨迹条件生成 | $P(x_{t+1}\mid x_t,h_t,o_t)$ | 决定给定算法状态如何产生下一步思想和程序，影响单步进化质量与可达方向 |
| evaluator | 候选质量与有效性 | 把算法运行结果转化为搜索可用的外部事实 |
| 森林、锚点与历史 | 状态表示与形成记录 | 保存轨迹，提供条件化上下文和可审计关系 |
| Refine/Explore 意图 | 变异任务与修改尺度 | 在族内开发与结构性探索之间设定提议分布 |

当前最准确的 V9.7 研究定位是：

> **在 evaluator 驱动的自动算法设计进化中，利用改进来时路提高 LLM 作为单步语义变异算子的可靠性，并用简单的路线—锚点分配把有限评价预算投入到可继续发展的算法状态。**

该版本已经把 AAD 的任务目标、进化搜索的基本控制和 TraceAAD 的轨迹假设放在同一条因果链上；它仍未解决跨算法区域的长期潜力估计，也未证明固定分配公式或 `0.7/0.3` 意图比例在所有任务上最优。

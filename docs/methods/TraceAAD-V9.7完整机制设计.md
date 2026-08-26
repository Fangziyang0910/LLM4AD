# TraceAAD V9.7：轨迹引导的自动算法设计

V9.7 围绕两个基础问题实现：**给哪一个节点一次决策机会**，以及**怎样帮它做出更好的决策**。改进轨迹参与评分与预算分配，也作为上下文辅助下一步改写。

## 1. 研究对象：优化求解一类问题的算法函数

自动算法设计（Automatic Algorithm Design, AAD）用大语言模型提出并改写算法函数，再与进化机制结合，优化该函数，使其能够求解一类问题。待优化对象是程序模板中的函数 $r$；单个实例上的解只是评价该函数的观测。

一个可执行的 AAD 任务由三部分组成：问题描述、算法模板与可编辑范围、以及已知的评价器：

$$
\mathcal T_{\mathrm{AD}}=(d_{\mathcal T},K,\mathcal E).
$$

其中 $d_{\mathcal T}$ 说明问题、输入输出和约束，$K$ 给出完整程序骨架及允许设计的部分，$\mathcal E$ 运行完整程序并返回可比较的质量。给定候选函数 $r$，模板实例化为 $P_r=K[r]$，评价器在训练集 $S_{\mathrm{train}}$ 上给出：

$$
q(r)=\begin{cases}
\mathcal E(P_r;S_{\mathrm{train}}), & \mathrm{maximize},\\
-\mathcal E(P_r;S_{\mathrm{train}}), & \mathrm{minimize}.
\end{cases}
$$

测试集为不同规模的新实例，不参与优化。

传统算法设计依赖专家理解问题结构、提出算法思想、完成实现并反复实验，成本高且难以系统覆盖设计空间。NFL 定理说明，不存在对所有问题都普遍最优的算法；可用的算法必须利用目标问题的结构。因此 AAD 的目标是降低面向特定问题类优化算法函数的成本。

## 2. 从 AAD 到 evaluator 驱动的进化搜索

LLM4AD 将 LLM 生成与进化机制结合：候选函数是进化个体，LLM 负责提出语义层面的变异或重组，评价器提供外部适应度事实，搜索控制器决定哪些候选继续获得生成机会：

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

这里的“进化”是广义的程序进化。树、种群和预算分配属于这一搜索范式的控制结构。

AAD 中的个体是待优化的算法函数。LLM 根据问题说明、当前代码和上下文完成语义变异，评价器检查完整程序的有效性、约束和性能。

因此，V9.7 的机制来源可以分成三层：

| 层次 | 继承或解决的问题 | V9.7 中的对应机制 |
| --- | --- | --- |
| 进化搜索共性 | 个体、变异、适应度、选择、探索—利用、有限预算 | 程序候选、LLM 生成、evaluator、分配器、Refine/Explore |
| AAD | 优化算法函数以求解一类问题；完整程序评价与 held-out 检验函数质量 | 算法模板、完整程序评价、`Idea + Code`、有效性与 held-out |
| TraceAAD 的研究对象 | 改进轨迹能否改善给哪个节点机会、以及怎样辅助该节点决策 | 轨迹参与评分与预算分配；父代来时路进入生成提示 |

## 3. TraceAAD 的科学问题与贡献

一次 LLM 生成给出算法函数的一个新版本。TraceAAD 的出发点是：函数优化是逐步引入思想、试错、修正和精炼的形成过程。当前函数的形成轨迹包含了代码本身之外的设计信息，例如某个机制从何处引入、实际改动了什么、结果是改善还是退步。

两个基础问题是：给哪一个节点一次决策机会，以及怎样帮它做出更好的决策。创新是把改进轨迹融入这两件事。

V9.7 的实现对应三个相连的环节：

1. **轨迹辅助下一步决策。** 把与当前锚点匹配的父代改进来时路压缩进提示，给 LLM 合适、有价值的上下文，避免冗余信息干扰。
2. **轨迹参与评分与预算分配。** 搜索器先决定继续哪一个来源区域，再决定该区域内从哪个形成状态出发，把下一次改写机会分给某一个节点。
3. **原子、即时反馈协议。** 每次只选择一个锚点、抽取一个生成意图、生成一个完整候选、进行一次真实评价并立即更新。

EoH 已经区分探索与开发生成，ReEvo 已经使用反思和历史摘要，MCTS-AHD 已经使用树搜索分配评价，BaSE 已经研究跨轨迹的计算分配。V9.7 把轨迹内容送入下一步改写，并把轨迹状态上的投资作为有限预算下的独立控制问题。

## 4. 两条核心机制

### 4.1 给哪一个节点一次决策机会

给定当前搜索森林，分配机制决定下一次 LLM 响应从哪个节点启动。它决定有限预算落在哪些来源、哪些算法状态上。

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

路线层承担来源之间的深度—广度权衡，锚点层承担同一路线内部的状态回访与局部开发。两层使用相同的形式，但比较对象不同。路线保存的是形成来源。目前还没有办法真正区分算法簇。

#### 初始化与尺度

搜索开始时建立 $K=8$ 条初始路线。每个根随后接受一次 Refine bootstrap。对 bootstrap 中成功形成新子节点的转换，取父子有向质量差的绝对值；改善、退步和持平都计入：

$$
s=\begin{cases}
\operatorname{median}(D_{\mathrm{init}}), & D_{\mathrm{init}}\neq\varnothing,\\
0, & D_{\mathrm{init}}=\varnothing.
\end{cases}
$$

$s$ 是任务内一步修改幅度的启发式尺度，不是路线潜力、置信区间或未来边际收益的估计。正式搜索开始后不再重估。

### 4.2 怎样帮节点做出更好的决策

选定锚点后，向 LLM 提供合适、有价值的上下文与提示，辅助这一次改写，避免冗余信息干扰。上下文包含当前完整算法和该锚点的父代形成历史；提示协议决定本轮修改任务。

#### 父代改进来时路

默认上下文包含当前完整算法和该锚点的父代形成路径。路径沿唯一父链回溯，保留最近至多 8 条真实形成事件。每条事件包括：

````text
[History i] Formation step
Idea: ...
Change: ...
Result: improve | regress | plateau
Fitness: parent -> child
````

`Idea` 是当时生成时声明的设计意图；`Change` 由父子实际代码推导；`Result` 和 `Fitness` 来自真实评价。形成历史说明当前算法怎样形成。信用使用当时可观测的信号。

直接子代尝试仍完整保存在搜索事实中，但不默认放入当前生成提示。来时路提供当前程序如何形成的事实；直接尝试记录从该锚点已经试过什么。

#### Refine 与 Explore

算子与探索、利用结合：一些算子应鼓励有效探索，一些算子应鼓励有效利用。V9.7 使用 Refine 与 Explore 两种提示协议：

- **Refine**：沿当前设计方向做一次聚焦修改，利用当前代码及其改进历史继续发展；
- **Explore**：寻找实质不同的改进方向，可以替换或重组当前设计的重要部分。

意图概率固定为：

$$
P(\mathrm{Refine})=0.7,\qquad P(\mathrm{Explore})=0.3.
$$

两种意图共享任务说明、当前代码、父代来时路和输出契约，只改变本轮修改目标。固定比例是 V9.7 的控制条件。

#### 候选和反馈

每次分配只生成一个候选。一次 LLM 响应输出可选的短 Idea 和一份完整、可执行的程序。完整程序是有效性的硬条件；Idea 缺失不使候选无效。新候选由 evaluator 真实运行并按有向质量标记为 `improve`、`plateau`、`regress` 或 `invalid`。这些标签是形成事实，不形成额外奖励。

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
| 预算分配 | 下一次被改写的节点 | 给哪一个节点一次决策机会 |
| 上下文与提示 | 当前节点看到的历史与本轮任务 | 帮该节点做出更好的决策 |
| evaluator | 候选质量与有效性 | 把算法运行结果转化为搜索可用的外部事实 |
| 森林、锚点与历史 | 状态表示与形成记录 | 保存轨迹，供评分、分配和生成上下文使用 |
| Refine/Explore | 本轮提示协议 | 现有版本中与探索、利用结合的两种提示 |

当前最准确的 V9.7 研究定位是：

> **在 evaluator 驱动的自动算法设计中，用改进来时路为当前节点提供改写上下文，并用简单的路线—锚点分配把有限评价预算交给可继续改写的算法状态。**

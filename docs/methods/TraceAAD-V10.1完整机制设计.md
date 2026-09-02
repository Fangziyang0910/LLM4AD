# TraceAAD V10.1 完整机制设计

V10.1 由质量偏置的父节点分配、多算子扩展和轨迹条件生成组成。**预算分配层**按当前真实质量概率性选择一个父节点。**辅助生成层**在同一冻结父节点上执行 Refine、Pivot 和 Fuse，各生成并评价一个子代。科学主张见[研究认识](../knowledge/研究认识.md)，预算分配的跨版本经验见[预算分配经验](../knowledge/预算分配经验.md)。树结构、有界执行修复与评价口径沿用既有 TraceAAD 平台。

三个核心对象：搜索树、轨迹、两层机制。

## 1. 两层框架

**预算分配层。** 利用全部有效节点的当前真实质量，概率性选择下一个获得扩展批次的父节点：

$$
H_t \rightarrow n_t.
$$

**辅助生成层。** 冻结选中节点 $n_t$ 及其轨迹，在同一快照上独立执行当前可用的三种算子：

$$
n_t
\rightarrow
\{\mathrm{Refine},\mathrm{Pivot},\mathrm{Fuse}\}
\rightarrow
\{c_R,c_P,c_F\}.
$$

预算只在节点层分配。三个算子不相互竞争预算；它们在每个完整扩展批次中各获得一个 primary evaluator slot。轨迹只作为生成证据。

## 2. 搜索树与轨迹

### 2.1 搜索树

搜索状态的核心是节点。父子关系由 `parent_id` 隐式定义整棵搜索树：

$$
n=
(\mathrm{id},\ \mathrm{code},\ \mathrm{idea},\ \mathrm{fitness\_raw},
\ \mathrm{parent\_id},\ \mathrm{origin\_operator},\ \mathrm{donor\_id},
\ \mathrm{created\_slot},\ \mathrm{created\_batch})
$$

- $\mathrm{id}$：节点唯一标识
- $\mathrm{code}$：该节点的代码
- $\mathrm{fitness\_raw}$：evaluator 返回的原始性能
- $\mathrm{idea}$：从父节点演化到该节点时采用的设计思想
- $\mathrm{parent\_id}$：直接父节点；根节点为空
- $\mathrm{origin\_operator}$：生成该节点的算子，取值为 Init、Refine、Pivot 或 Fuse
- $\mathrm{donor\_id}$：仅 Fuse 子节点记录的参考节点 ID，其余节点为空
- $\mathrm{created\_slot}$：生成评价该节点所消耗的 primary slot
- $\mathrm{created\_batch}$：该节点所属的扩展批次；根节点属于初始化批次

不另存 transition 对象。一次成功生成之后，节点保留模型写出的 idea、最终 code、evaluator 给出的原始 fitness，以及形成该节点所需的算子、父节点、批次和可选 donor 元数据。`origin_operator` 使同一父节点上 Refine、Pivot 和 Fuse 的配对结果可以直接恢复。

Fuse 不改变单父树结构。对于由 target $n$ 和 donor $r$ 形成的子节点 $c$，结构父节点始终是 target：

$$
\operatorname{parent}(c)=n,
\qquad
\operatorname{donor\_id}(c)=\operatorname{id}(r).
$$

$r$ 记录为生成参考元数据，谱系仅记录 target 单父指针。

节点身份由形成路径和可执行程序共同确定。Canonical code 定义为解析器提取出的完整可执行函数代码，统一换行符并去除首尾空白；V10.1 不做 AST 改写或语义等价归并。随后按其 SHA-256 hash 判断代码相同：

- child 与 parent 的 canonical code 相同时，不建立新节点
- 同一 `(parent_id, canonical_code)` 已存在时，不重复建立节点
- 相同 canonical code 从不同 parent 到达时，可以建立不同节点，因为它们具有不同轨迹和下一步生成上下文

所有生成与评价尝试另写入 append-only attempt ledger。失败、无效输出、执行修复和重复代码即使没有形成节点，也必须留下事实记录并计入实际消耗的预算。每条 attempt 至少记录：

```text
attempt_id
batch_id
parent_id
origin_operator
donor_id
prompt_hash
response
status
fitness_raw
repair_count
primary_slot
```

Attempt ledger 用于预算核算和实验分析，不形成第二种搜索节点，也不改变单父树拓扑。

### 2.2 轨迹

轨迹从当前节点沿父指针向上动态回溯生成，用于刻画节点的演化路径。

对当前节点，最多向上追溯 8 代祖先。轨迹规范：

- 仅包含父代与祖先节点
- 每一代展示该节点形成时的 Idea 和 Fitness 结果
- 提升/下降根据相邻代 fitness 动态计算
- 祖先节点仅提供 Idea 与 Fitness，代码上下文仅保留当前算法
- 谱系与轨迹仅沿 target 父代回溯

$$
\tau(n)=\operatorname{TraceBack}(n.\mathrm{parent},H)
=\bigl[n^{(-k)},\ldots,n^{(-2)},n^{(-1)}\bigr],
\qquad k\le 8
$$

$n^{(-1)}$ 是直接父代。需要某个节点的轨迹时才回溯，现场组织上下文。

$$
\text{Search Tree}
\rightarrow
\text{Dynamic Trace Construction}
\rightarrow
\text{Trajectory Context}
$$

### 2.3 轨迹上下文表示

Fitness 与 Outcome 合并成一行。这里展示 evaluator 返回的原始 fitness；优化方向由 Task Contract 说明。最多展示 8 个祖先节点；必要时可额外读取最老祖先的父节点，仅用于计算 trend，但不展示。

$$
\mathrm{trend}_i
=
\operatorname{Compare}(\mathrm{fitness}_i,\mathrm{fitness}_{i-1})
$$

$\mathrm{trend}_i$ 由 $\operatorname{Compare}$ 沿用 evaluator 本身的相等/容差判定动态计算，内部按优化方向统一为 normalized improvement，取值为 Improved、Degraded 或 Unchanged。根节点仅展示 Fitness 数值。

推荐表示：

```text
# Historical Design Trajectory

Generation -3
Idea: <idea>
Fitness: 0.731 (Improved)

Generation -2
Idea: <idea>
Fitness: 0.746 (Improved)

Generation -1
Idea: <idea>
Fitness: 0.739 (Degraded)
```

### 2.4 轨迹在两层中的定位

1. **预算分配层**：不使用轨迹，只根据节点当前真实质量构造父节点选择概率。
2. **辅助生成层**：把选中节点的轨迹作为历史证据交给三个算子。

## 3. 预算分配层

V10.1 使用 Quality-biased Probabilistic Node Allocation。

每一个扩展批次开始时，预算分配回答：

> 下一个 Refine / Pivot / Fuse 扩展批次应该给哪个父节点？

机制一次只选择父节点：

$$
n_t\sim p_t(n).
$$

- $p_t$：由全部有效节点的当前真实质量构造的概率分布
- $n_t$：本轮获得扩展批次的父节点

一个完整扩展批次随后覆盖 Refine、Pivot 和 Fuse。搜索的深挖、换向和跨分支组合由算子集合实现；节点分配只需选择一个有价值的当前算法作为共同 scaffold。

### 3.1 初始化与节点质量

搜索以 $N_0=8$ 个独立生成、有效且 canonical code 互异的根节点初始化。每个根候选只看到 Task Contract，不看到已经生成的其他根节点。初始化持续进行单候选生成与评价，直到获得 8 个有效唯一根节点：

```text
while valid_unique_roots < 8 and primary budget remains:
    generate one root from Task Contract
    evaluate the candidate
    consume one primary slot
    if valid and canonical code is new:
        add it as a root with origin_operator = Init
```

无效候选和重复代码不进入树，但保留在 attempt ledger 中并消耗其实际使用的 primary slot。若预算耗尽仍不足 8 个有效唯一根节点，运行以初始化未完成结束，不进入主搜索循环。

时刻 $t$ 的全部有效节点构成候选集：

$$
\mathcal A_t=\{n_1,\ldots,n_N\}.
$$

记 evaluator 返回的原始结果为 $f(n)$，内部用于分配的当前真实质量为 $q(n)$。对标量最大化和最小化任务：

$$
q(n)
=
\begin{cases}
f(n), & \mathrm{maximize},\\
-f(n), & \mathrm{minimize}.
\end{cases}
$$

$q(n)$ 只统一优化方向，保留原始 fitness gap，不改成 rank。预算分配只使用 $q(n)$；给 LLM 的 Current Algorithm 和轨迹始终展示原始 $f(n)$，并在 Task Contract 中明确 Objective 是 maximize 还是 minimize。

$q(n)$ 的语义是程序已经实现的质量。机制使用高质量程序作为后续设计实验的 scaffold，不额外估计 trajectory slope、novelty、momentum 或 node success rate。

### 3.2 ESS 校准的 Boltzmann 选择

父节点选择概率为：

$$
p_t(n)
=
\frac{\exp\left(\beta_t q(n)\right)}
{\sum_{m\in\mathcal A_t}\exp\left(\beta_t q(m)\right)}.
$$

实现时减去当前最大 $q$ 后再计算指数，不改变概率。分配从该分布抽样：

$$
n_t\sim p_t.
$$

逆温度 $\beta_t\ge 0$ 不使用跨任务固定值，而是由目标有效样本量校准。定义：

$$
\operatorname{ESS}(p_t)
=
\frac{1}{\sum_{n\in\mathcal A_t}p_t(n)^2},
$$

$$
E_t
=
\min\left(N,\max\left(\rho N,K_{\min}\right)\right),
\qquad
\rho=0.10,
\qquad
K_{\min}=2.
$$

并列最高质量节点决定非负逆温度下可达到的最低 ESS。定义：

$$
k_{\max}
=
\left|
\left\{n\in\mathcal A_t:q(n)=\max_{m\in\mathcal A_t}q(m)\right\}
\right|,
$$

$$
E_t^*=\max(E_t,k_{\max}).
$$

当 $E_t^*<N$ 时，在 $\beta_t\ge 0$ 上数值求解，使 $\operatorname{ESS}(p_t)$ 尽可能接近可达目标 $E_t^*$；当目标只能在 $\beta_t\to\infty$ 时达到时，使用数值容差内最接近的有限值。当 $E_t^*=N$ 时取 $\beta_t=0$，在候选节点上均匀抽样。每轮同时记录目标 ESS 和实际 ESS。

该分布使高质量节点获得更高概率，同时为其他节点保留非零概率。节点层的探索由这个概率支持提供；设计行为的多样性由三个算子提供。

### 3.3 固定算子扩展

V10.1 使用三个一级算子：

$$
\mathcal O
=
\{\mathrm{Refine},\ \mathrm{Pivot},\ \mathrm{Fuse}\}.
$$

三个算子在标准批次中各执行一次。$1{:}1{:}1$ 的语义是在尚无可靠条件算子价值估计时，对三类设计行为做完整覆盖的中性实验协议。

对一个已有节点 $n$，下一步设计的知识来源只有三类：继续开发当前思想、从当前节点重新建立核心思想、引入其他分支的思想。三个算子因此按本次设计与当前核心思想的关系，以及是否引入外部分支信息来划分。

算子定义的对象是**算法设计行为**。判定依据是核心设计假设之间的关系，与改动的代码行数、文本距离或结构规模无关。一行公式可以更换核心假设，而大范围重构也可以只是更好地实现原有假设。

| 算子 | 核心问题 | 与当前核心思想的关系 | 外部参考节点 |
| --- | --- | --- | --- |
| Refine | 这个方向还能怎样做好 | 保留并继续开发 | 不需要 |
| Pivot | 从这个节点还能打开什么新方向 | 放弃或重构 | 不需要 |
| Fuse | 其他分支有什么有价值的机制可以融合 | 保留有价值的主体并组合外部思想 | 需要 |

#### Refine：继续开发当前思想

Refine 回答：这个已经形成的算法思想还能怎样做得更好？

它接受当前算法的核心设计假设，并在这个方向上形成更好的实现。它可以改进已有机制、增加服务于核心思想的辅助机制、修改公式与参数、调整机制之间的互作、删除冗余逻辑、修正局部不合理设计，或更准确地实现当前 idea。修改可以很大，共同约束是：

$$
\operatorname{PreserveCurrentCoreIdea}(n')=\mathrm{true}.
$$

形式上，Refine 是同一核心假设的继续开发：

$$
\mathrm{Refine}
=
\mathrm{SameCoreHypothesis}
+
\mathrm{BetterRealization}.
$$

#### Pivot：从当前节点打开新方向

Pivot 回答：如果当前核心思路不再被接受，从这个算法基础还能建立什么新的设计原则？

当前节点仍是设计起点。它提供可运行的程序框架、当前代码、已有决策结构和匹配的来时路；其核心设计假设不再是需要继续维持的约束。Pivot 重新思考主要决策原则或核心机制，从当前基础形成一个实质不同的设计方向。

$$
\operatorname{StartFrom}(n')=n,
\qquad
\operatorname{PreserveCurrentCoreIdea}(n')=\mathrm{false}.
$$

Pivot 与 Refine 的分界是是否继续接受当前核心设计假设。Pivot 可能只改变一个关键公式，Refine 也可能重写大量代码。

形式上，Pivot 保留当前节点作为 scaffold，更换其 hypothesis：

$$
\mathrm{Pivot}
=
\mathrm{SameStartingNode}
+
\mathrm{DifferentCoreHypothesis}.
$$

#### Fuse：组合其他分支的设计机制

Fuse 回答：其他分支已经形成的有价值机制，能否与当前方向构成互补？

Fuse 是两个已评价节点之间的定向组合。当前节点 $n$ 是 target，参考节点 $r$ 是 donor。新算法应保留 target 中至少一个实质性机制，吸收 donor 中至少一个实质性且相容的机制，并让两者在新算法中承担互补的角色：

$$
(n,r)\rightarrow n'.
$$

Fuse 表达的是算法思想的语义组合与重组。机械复制 donor 代码、用 donor 替换 target、拼接两段独立逻辑，或者没有机制分工地平均两个公式，都不构成 Fuse。结果应是一个内部一致的算法，其中两个来源机制有明确且互补的作用。

参考节点 $r$ 的选择与呈现由 Fuse 内部机制确定。$r$ 是 Fuse 的执行条件，不参与父节点预算分配。

### 3.4 算子边界

三个算子覆盖从已有节点出发的三类一级设计行为：沿当前思想继续开发、从当前节点更换设计假设、引入外部分支进行组合。每次生成以其主要设计意图归入其中一类。

Restart 不属于固定扩展算子集。V10.1 在初始化时提供多个根节点。

思想与实现的局部不一致属于 Refine 的可执行内容，不另设 SemanticRepair。语法错误、运行时错误和接口错误由已有的有界执行修复处理，不记为搜索算子。

### 3.5 三种搜索生长方式

三个算子在搜索树上形成三种不同的生长方式：

$$
\begin{aligned}
\mathrm{Refine} &:\ \mathrm{Depth},\\
\mathrm{Pivot} &:\ \mathrm{Branching},\\
\mathrm{Fuse} &:\ \mathrm{Recombination}.
\end{aligned}
$$

Refine 沿当前 lineage 继续开发；Pivot 从当前节点建立新的设计方向；Fuse 使不同 lineage 之间发生知识交换。每个完整扩展批次同时覆盖深挖、换向和重组。

### 3.6 Fuse 的 donor 选择

Donor 由 Fuse 内部的固定策略选择：

$$
r=R(n,H_t).
$$

策略 $R$ 在批次冻结前为 Fuse 选择 donor。Donor 不参与节点抽样。

对 target $n$，先构造有效的外部分支候选：

$$
D(n)
=
\left\{
r\in H_t:
\begin{array}{l}
r\text{ has valid code, idea, and fitness},\\
r\ne n,\\
r\notin\operatorname{Ancestors}(n),\\
r\notin\operatorname{Descendants}(n),\\
\operatorname{CanonicalCode}(r)\ne\operatorname{CanonicalCode}(n)
\end{array}
\right\}.
$$

候选 donor 必须来自与 target 无祖先或后代关系的其他分支，并且具有不同的可执行程序。这保证 Refine 和 Pivot 负责单条 lineage 内的开发与换向，Fuse 负责分支之间的知识引入。根节点可以成为 donor；donor eligibility 由绝对 evaluator 质量支持，与相对父节点的改善无关。

在排序前按 canonical code 去重。同一代码对应多个形成状态时，保留 $q$ 最高者；若 $q$ 相同，按节点 ID 做确定性择一。对去重后的 $D(n)$ 按 $q$ 排序，保留前 $K=5$ 个节点：

$$
D_5(n)=\operatorname{TopK}_{5}
\bigl(\operatorname{DeduplicateCode}(D(n)),q\bigr).
$$

使用独立的 donor 随机流对 $D_5(n)$ 产生一个有种子的随机排列。按该顺序尝试 donor，并按第 4.5 节的上下文长度规则检查能否完整呈现 target code 和 donor code；第一个可用节点成为 donor。Top-5 提供高质量约束，随机排列避免所有 Fuse 始终吸向同一节点，同时保持可复现。

8 个有效唯一根节点使主搜索开始时每个 target 都有其他分支候选。只有 $D_5(n)=\varnothing$，或所有 Top-5 donor 都无法在输入限制内完整呈现时，本轮不执行 Fuse，并在 attempt ledger 中记录具体原因。第一版 donor selector 只使用树拓扑、canonical code、当前质量和独立的有种子随机流。

### 3.7 冻结扩展批次与预算

选中父节点 $n_t$ 后，先冻结本轮的父节点快照：

- Current Idea、Fitness 和 Code
- 沿 `parent_id` 构造的 Historical Design Trajectory
- Fuse 可用时，由 $R(n_t,H_t)$ 选定的 donor Idea、Fitness 和 Code

令 $\mathcal O_t(n_t)$ 为本轮可执行的算子集。Refine 和 Pivot 始终可用；存在可完整呈现的 donor 时 Fuse 可用：

$$
\mathcal O_t(n_t)
=
\{\mathrm{Refine},\mathrm{Pivot}\}
\cup
\begin{cases}
\{\mathrm{Fuse}\}, & \operatorname{DonorAvailable}(n_t),\\
\varnothing, & \operatorname{otherwise}.
\end{cases}
$$

包含三个算子的扩展批次是标准父节点分配单元。每次父节点选择预先分配三个相互独立的 sibling 实验，重新决策粒度为完整扩展批次，以批次内的配对算子覆盖交换逐评价重新选择父节点的 option value。

对每个 $o\in\mathcal O_t(n_t)$，基于同一冻结快照独立生成一个候选并进行一次正式评价：

$$
c_o
\sim
P(\cdot\mid n_t,\tau(n_t),o,r_o),
$$

其中 $r_o$ 仅在 $o=\mathrm{Fuse}$ 时非空。每个已调度算子各消耗一个 primary evaluator slot。三个 sibling 在生成和评价期间都看不到其他 sibling 的结果。整批结束后，所有有效子代以 `parent_id=n_t.id` 加入树，然后重新计算全树的 $p_{t+1}$ 并选择下一个父节点。

设批次开始时剩余 $b_t$ 个 primary slots。当 $b_t\ge |\mathcal O_t(n_t)|$ 时执行全部可用算子。当 $b_t<|\mathcal O_t(n_t)|$ 时，使用独立的 tail-operator 随机流从 $\mathcal O_t(n_t)$ 中无放回均匀抽取 $b_t$ 个算子，用完剩余预算。

这个批次自然记录同一父节点上 Refine、Pivot 和 Fuse 的配对结果。当前机制不用这些结果自适应分配算子预算。

### 3.8 随机流与评价可比性

父节点抽样、donor 抽样和尾部算子抽样使用相互独立的随机流：

```text
parent_rng
donor_rng
tail_operator_rng
```

每次 LLM 请求的采样种子由 `(run_seed, batch_id, origin_operator)` 确定。改变 donor 策略或尾部预算处理时，不推进父节点随机流。这使只改变一个机制的配对实验不会因共享 RNG 的消费顺序改变后续全部父节点选择。

Evaluator 的随机性必须只由运行种子、任务实例和 evaluator 规定，不依赖 sibling 的执行顺序或 origin operator。同一批次的 Refine、Pivot 和 Fuse 因而面对相同的任务实例与评价随机条件。

## 4. 辅助生成层

发生在预算分配已经选定父节点并冻结批次快照之后。这一层对批次中每个可用算子做条件化生成，不重选父节点或调整算子预算。LLM 获得的共同上下文固定为四部分：Task Contract、Current Algorithm、Historical Design Trajectory、Operator Instruction。Fuse 额外获得一个 Reference Algorithm。

$$
\mathrm{TaskContract}
+\mathrm{Current}
+\mathrm{TrajectoryEvidence}
+\mathrm{OperatorInstruction}
\rightarrow
\mathrm{Idea}+\mathrm{Code}.
$$

### 4.1 上下文组织

````text
# Task Contract
<问题描述与设计目标>
<待优化函数、输入输出与调用时机>
<函数参数真实提供的可用信息>
Objective: <maximize / minimize>
<有效性约束与不可使用的状态>


# Current Algorithm
Idea: <当前节点 idea>
Fitness: <当前节点 raw fitness>

```python
<当前节点 code>
```


# Historical Design Trajectory

Generation -k
Idea: <idea>
Fitness: <fitness> (<Improved / Degraded / Unchanged>)

...

Generation -1
Idea: <idea>
Fitness: <fitness> (<Improved / Degraded / Unchanged>)


# Improvement Operator
Operator: <operator>

Instruction:
<该算子的固定语义定义>
````

### 4.2 四部分的职责

**Task Contract** 是任务与接口层契约。它说明要解决什么问题、可编辑函数在什么时刻被调用、参数与返回值的语义、真实可用的信息、有效性约束和评价方向。信息边界在这里统一规定：

> Design the algorithm using only information available through the target function interface. Do not assume access to unavailable state or future information.

它的含义是：只基于目标函数接口中真实可获得的信息设计算法，不得假设存在未提供的 solver 状态、未来决策、未来输入、evaluator 信息或隐藏变量。这条约束对三个算子完全相同，不属于任何一个算子的语义。

**Current Algorithm** 是本次生成的当前事实，包含选中节点的 Idea、原始 Fitness 和完整代码。内部用于父节点分配的 $q(n)$ 不展示给模型；优化方向由 Task Contract 的 Objective 给出。

**Historical Design Trajectory** 是当前节点的历史证据。它说明这个设计如何形成、沿路引入了哪些思想以及相应的评价结果。轨迹帮助模型理解过去发生了什么，不规定这一步应当采取什么行为，也不因最近一步的改善或退化自行改变已选算子。

**Operator Instruction** 是本轮唯一的设计行为约束。它只规定这一次是 Refine、Pivot 还是 Fuse，以及该设计行为的语义边界。它不重复任务接口、可用信息或输出格式要求。

三个算子的信息来源固定为：

$$
\begin{aligned}
\mathrm{Refine} &: \mathrm{TaskContract}+\mathrm{Current}+\mathrm{TargetTrajectory},\\
\mathrm{Pivot} &: \mathrm{TaskContract}+\mathrm{Current}+\mathrm{TargetTrajectory},\\
\mathrm{Fuse} &: \mathrm{TaskContract}+\mathrm{Current}+\mathrm{TargetTrajectory}+\mathrm{Donor}.
\end{aligned}
$$

只有 Fuse 可以看到其他节点的 Idea、Fitness 和 Code。Refine 和 Pivot 只使用当前节点及其自身轨迹；Pivot 的新方向由当前节点内部重新思考得到。

这种分工把上下文中的内容分成四种性质：

- Task Contract：任务与接口事实
- Current Algorithm：当前节点事实
- Historical Design Trajectory：历史证据
- Operator Instruction：本轮行为约束

### 4.3 统一生成契约

生成契约在三个算子之外统一定义，对 Refine、Pivot 和 Fuse 完全一致：

1. **执行已调度算子。** 父节点和本批次的算子集已经固定。每次生成直接执行当前算子 $o$。
2. **有效使用形成证据。** 候选不应空泛重复轨迹中已有的 Idea；它根据轨迹理解当前设计，同时始终执行已选算子。
3. **让 Idea 成为语义压缩单元。** Idea 用一句话说明本次引入、修改或组合的实际算法机制。它不写空泛的“综合多种因素”、性能声称、实现琐事或推理过程。后续轨迹直接使用这个 Idea。
4. **保持程序契约。** Code 是一份完整可执行实现，保持 Task Contract 规定的函数签名、输入输出与调用约定。
5. **只输出 Idea + Code。** 模型不额外输出 reasoning、轨迹分析或算子标签。

### 4.4 算子的固定语义指令

#### Refine

```text
Continue developing the current algorithmic direction. Preserve the core design
principle of the current algorithm. Use the historical trajectory as evidence to
understand how this direction has developed and what has already been tried,
then make a coherent improvement that better realizes or strengthens the current
idea. The implementation may change substantially if needed, but do not replace
the core algorithmic principle with a different one.
```

#### Pivot

```text
Develop a materially different algorithmic direction from the current node.
Treat the current code as a usable starting scaffold, but do not assume that its
core design principle should be preserved. Use the historical trajectory as
evidence to understand which directions have already been explored, then
introduce a different primary algorithmic mechanism. The change must be
different at the mechanism level, not merely parameter tuning, coefficient
adjustment, or superficial restructuring.
```

#### Fuse

```text
Create a coherent algorithm by combining complementary mechanisms from the
current algorithm and the provided reference algorithm. Treat the current
algorithm as the target design and the reference algorithm as an external source
of design knowledge. Preserve a substantive mechanism from the current
algorithm and incorporate a compatible mechanism from the reference algorithm.
Integrate them according to their algorithmic roles rather than mechanically
copying code, averaging formulas, concatenating logic, or replacing the current
algorithm with the reference algorithm.
```

Idea + Code 的输出契约由统一生成契约规定，不在三个算子指令中重复。

### 4.5 Fuse 的参考上下文

Fuse 执行时在 target 轨迹之后增加一个 donor 参考，再给出 Fuse 指令：

````text
# Reference Algorithm
Idea: <donor idea>
Fitness: <donor fitness>

```python
<donor code>
```


# Improvement Operator
Operator: Fuse

Instruction:
<Fuse 的固定定义>
````

Fuse 默认不提供 donor 的轨迹。target 轨迹负责说明当前设计的来时路；donor 只提供可供组合的外部机制。因此 Fuse 的完整上下文为：

$$
\mathrm{TaskContract}
+\mathrm{Target}
+\mathrm{TargetTrajectory}
+\mathrm{DonorIdea}
+\mathrm{DonorFitness}
+\mathrm{DonorCode}
+\mathrm{FuseInstruction}.
$$

上下文超长时按以下优先级处理：

1. Task Contract、Current Code 和 Operator Instruction 始终完整保留。
2. Refine 或 Pivot 超长时，从最老一代开始减少 target trajectory，直到满足输入限制。
3. Fuse 超长时同样先从最老一代开始减少 target trajectory，target code 和 donor code 都不得截断。
4. 若不含 trajectory 的 Task Contract、Target、Donor 和 Fuse Instruction 仍然超长，则按 donor 随机排列尝试下一个 Top-5 donor。
5. 所有 Top-5 donor 都无法完整呈现时，本轮 Fuse 不可用，并记录 `context_overflow`。

代码截断会破坏算法机制的可理解性和可执行性，因此不作为上下文压缩方式。

### 4.6 这一层的研究问题

轨迹作为生成上下文，是否能够比仅使用当前代码和算子提示，更好地帮助 LLM 产生下一步算法设计。

## 5. 搜索循环

$$
H_t
\rightarrow p_t(n)
\rightarrow n_t
\rightarrow \text{Frozen Batch Context}
\rightarrow \{\mathrm{Refine},\mathrm{Pivot},\mathrm{Fuse}\}
\rightarrow \text{Generation and Evaluation}
\rightarrow \text{Batch Commit}
\rightarrow H_{t+1}
$$

Frozen Batch Context 由第 3.7 节的父节点快照和第 4 节的辅助生成上下文构成。子代整批提交后才进入下一轮父节点选择。

## 6. 固定参数与记录

| 项目 | 值 |
| --- | --- |
| 正式 primary evaluator 预算 $B$ | 1000 |
| 初始有效根节点 $N_0$ | 8 |
| 父节点分数 | 当前真实质量 $q(n)$ |
| ESS 比例 $\rho$ | 0.10 |
| 最小目标 ESS $K_{\min}$ | 2 |
| 标准扩展算子 | Refine、Pivot、Fuse |
| Fuse donor Top-$K$ | 5 |
| 轨迹长度上限 | 8 代祖先 |

每个批次记录父节点候选数 $N$、最高质量并列数 $k_{\max}$、$\beta_t$、目标与实际 ESS、选中父节点及其概率、冻结快照 ID、已调度算子、Fuse donor ID、每个 attempt 的状态与原始 fitness，以及批次前后剩余的 primary slots。

过程统计至少包括每个节点的父节点入选次数、同一父节点的累计扩展次数、各算子的有效率和重复率、unique improvement rate、global-best parent budget share，以及不同 origin operator 子节点的出生质量变化和后续重访。

## 7. 实验识别与过程诊断

实现完成后先识别三个组件是否产生预期行为，再进入正式 1000-eval 比较。以下实验不改变 V10.1 的在线分配规则。

### 7.1 Stage P：算子识别

固定一组父节点、轨迹、donor 和随机种子，分别执行 Refine、Pivot 和 Fuse，检查：

- valid rate 与 duplicate rate
- Refine 和 Pivot 是否形成不同的 proposal distribution
- Pivot 是否改变主要算法机制
- Fuse 是否同时保留 target 机制并吸收 donor 机制
- Fuse 的 donor copy rate
- 三个算子的即时 fitness 分布

只有三个算子形成可区分的设计行为，固定三算子扩展才具有机制含义。

### 7.2 Ablation A：轨迹上下文

固定父节点分配、算子、donor、随机种子和评价预算，比较：

$$
\mathrm{Current}+\mathrm{Operator}
$$

与：

$$
\mathrm{Current}+\mathrm{Trajectory}+\mathrm{Operator}.
$$

该对照识别轨迹作为生成证据的贡献。

### 7.3 Ablation B：完整批次

在总 primary evaluator 预算相同、父节点选择规则相同的条件下比较：

- V10.1：选中父节点后完整执行 Refine、Pivot 和 Fuse
- 单算子对照：选中父节点后从 Refine、Pivot 和 Fuse 中均匀抽取一个

该对照识别完整算子覆盖相对逐评价重新选择父节点的净作用。

### 7.4 预注册过程风险

以下风险通过日志独立诊断，在线父节点分数严格基于当前真实质量：

1. **新方向缺少成熟机会。** 记录 Pivot child 的 birth drop、再次成为父节点的比例、首次重访等待时间，以及重访后 Refine 是否形成突破。非零抽样概率不等于有限预算中的有效暴露。
2. **固定算子覆盖与任务不匹配。** 分任务报告三算子的有效率、即时改善率和后续贡献；$1{:}1{:}1$ 只表示中性覆盖。
3. **Elite 过度扩展。** 记录 parent selection count、global-best parent budget share、duplicate rate 和 unique improvement rate。
4. **轨迹锚定 Pivot。** 在 Stage P 中比较 Refine/Pivot 的机制差异与代码差异，确认完整 trajectory 没有把 Pivot 压缩成局部精炼。

这些诊断用于判断后续版本需要解决的瓶颈；诊断结果不在 V10.1 运行期间改变在线机制，父节点 score 始终只有 $q(n)$。

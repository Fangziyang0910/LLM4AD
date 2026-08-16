# TraceAAD V9.9：轨迹优先的锚点—算子联合决策

> 状态：机制已实现，局部协议已收口，尚未获得完整搜索结果。[V9.8](TraceAAD-v9.8完整机制设计.md) 仍是已完成 Stage P 的对照版本，已有正式结果仍以 [V9.7](TraceAAD-v9.7完整机制设计.md) 为准。实现入口：`llm4ad/method/traceaad_v9_9/` 与 `experiments.runners.traceaad.run --version v9_9`。
> 设计依据：[V9.7 搜索几何诊断](../analysis/TraceAAD-V9.7搜索几何诊断.md)、[V9.8 机制识别实验分析](../analysis/TraceAAD-V9.8机制识别实验分析.md)、[固定锚点单步生成实验](../experiments/TraceAAD-固定锚点单步生成识别实验.md)、[BaSE 阅读笔记](../references/LLM自动算法设计方法阅读笔记/28-Compute-Allocation-BaSE.md)与[研究认识](../knowledge/研究认识.md)。
> 版本边界：V9.9 保留“一次选择、一次生成、一次评价、一次更新”的原子循环。在线分配不再使用 root route 或 Explore-defined hypothesis 聚合，而是直接对锚点—算子组合建模。

## 1. 核心设计

每一份计算预算同时回答两个问题：

1. 从哪个历史锚点继续；
2. 在该锚点采用 Refine 还是 Explore。

V9.9 将二者写成联合决策：

$$
P(a,o\mid\mathcal H)
=
\mu(a\mid\mathcal H)\,
\pi(o\mid a,\mathcal H).
$$

其中 $\mathcal H$ 是当前完整搜索历史，$a$ 是锚点，$o\in\{R,E\}$ 是 Refine 或 Explore。运行顺序是先按全局状态选择锚点，再根据该锚点的状态选择算子。

完整原子循环为：

$$
a_t\sim\mu(a\mid\mathcal H_t),
\qquad
o_t\sim\pi(o\mid a_t,\mathcal H_t),
$$

$$
C_t=\operatorname{Ctx}(a_t,o_t),
\qquad
x_{t+1}\sim P(\cdot\mid a_t,C_t,o_t),
$$

$$
q_{t+1}=\operatorname{Evaluate}(x_{t+1}),
\qquad
\mathcal H_{t+1}=\operatorname{Update}(\mathcal H_t,x_{t+1}).
$$

每次模型响应后立即重新计算状态和概率，不承诺未来多步预算。

## 2. 在线状态

### 2.1 程序

程序 $x$ 是评价器执行过的一份唯一代码。记原始适应度为 $f(x)$，任务方向为 $d\in\{+1,-1\}$：

$$
q(x)=d f(x).
$$

最大化任务取 $d=+1$，最小化任务取 $d=-1$，搜索内部统一为 $q$ 越大越好。

### 2.2 锚点

锚点表示某份程序在一条具体形成路径中的位置：

$$
a=\langle x(a),p(a),e(a),n_R(a),n_E(a)\rangle.
$$

$p(a)$ 是结构父锚点，$e(a)$ 是形成该锚点的事件，$n_R(a)$ 与 $n_E(a)$ 分别是从该锚点发起的 Refine 和 Explore 响应数。

同一代码沿不同形成路径到达时，可以对应不同锚点。它们共享程序的真实适应度，但保留各自的来时路与算子计数。

### 2.3 来时路与形成事件

锚点 $a$ 的来时路 $\tau(a)$ 是从根到该锚点的唯一父链。每个形成事件记录：

- 当时采用的 Refine 或 Explore；
- 模型声明的 Idea；
- 父子实际代码的紧凑变化；
- `improve / plateau / regress` 结果；
- 父子真实适应度与发生顺序。

来时路保存真实形成过程，不被压缩为趋势、推进率、成熟度或单一轨迹奖励。

### 2.4 不进入在线分配的对象

V9.9 不维护 route 或 hypothesis 级在线信用。root provenance、静态算法簇、行为簇、代码新颖性和程序复杂度都不参与分配。

从当前锚点发起过哪些候选及其结果继续作为搜索事实保存，但不进入锚点评分、算子评分或生成上下文。尝试结果不向起始锚点或祖先回传信用。

## 3. 初始化

初始化独立生成并真实评价 8 个有效且代码互异的根。8 个根使用相同的从零设计算法指令和不同采样种子；生成某个根时不展示其他候选、已有搜索历史或全局总结。全部 8 个根进入搜索状态，由在线分配决定后续命运。初始化不做质量筛选，也不丢弃任何有效互异根。

初始化不预先规划多种策略，不使用在线聚类、机制标签、embedding 或额外 judge。代码互异只是最低条件；多次独立采样只增加起点多样性的机会，不保证算法簇覆盖。

正式根的状态为：

$$
\tau(a_{\mathrm{root}})=\varnothing,
\qquad
n_R(a_{\mathrm{root}})=n_E(a_{\mathrm{root}})=0,
\qquad
C_R(a_{\mathrm{root}})=0.
$$

V9.9 不执行 root bootstrap。来时路由正式搜索自然形成。

## 4. 锚点—算子优先级

### 4.1 当前质量

每次原子决策前，将当前搜索状态中全部唯一程序的 $q$ 转为中秩百分位。若程序数为 $M>1$，严格低于 $q(x)$ 的程序数为 $L(x)$，等于 $q(x)$ 的程序数为 $E(x)$：

$$
Q(x)=
\frac{L(x)+(E(x)-1)/2}{M-1}.
$$

只有一个程序时定义 $Q(x)=0.5$。锚点质量为：

$$
Q(a)=Q(x(a)).
$$

相同程序的锚点共享 $Q$。每次决策都重新计算当前排名；距离宽限中的祖先质量也使用同一时刻的排名。

### 4.2 算子尝试不足

两个算子分别维护尝试不足项：

$$
U_R(a)=\frac{1}{\sqrt{n_R(a)+1}},
\qquad
U_E(a)=\frac{1}{\sqrt{n_E(a)+1}}.
$$

该项只表示从当前锚点给对应算子投入了多少生成机会，不估计真实潜力。

有效候选、invalid、no-op、祖先返回、重复代码和缓存命中都增加起始锚点对应的算子计数。模型请求在获得响应前因传输错误失败时不计数。

### 4.3 距离衰减的 Refine 回撤宽限

记祖先锚点 $b$ 到当前锚点 $a$ 的形成步数为 $d(a,b)$，最近 8 个祖先为 $\mathrm{Anc}_8(a)$。来时路上的历史质量差按代数距离衰减，并且只读取与生成上下文相同的轨迹窗口：

$$
D_h(a)
=
\max_{b\in\mathrm{Anc}_8(a)}
2^{-d(a,b)/h}
[Q(b)-Q(a)]_+,
\qquad
h=4.
$$

路径短于 8 步时，$\mathrm{Anc}_8(a)$ 就是全部祖先。相距 1、4 和 8 代的祖先分别保留约 $84\%$、$50\%$ 和 $25\%$ 的影响；第 9 代及更远的祖先不进入 $D_h$。系统选择窗口内距离衰减后最大的质量差，因此最近一个稍好的祖先可以比窗口内更远的历史最好祖先更有作用。

Refine 宽限还随当前锚点的 Refine 响应数衰减：

$$
C_R(a)
=
\frac{D_h(a)}
{\sqrt{n_R(a)+1}}.
$$

若当前锚点已不低于最近 8 个祖先，或锚点为根，则 $C_R(a)=0$。该项只进入 Refine，避免暂时退步的状态在获得打磨机会前被立即淘汰，同时不鼓励从未站稳的状态继续 Explore。

距离使用形成路径上的代数，不使用全局迭代时间。锚点休眠时代码没有变化，算子尝试次数和父代关系都不随等待而遗忘。生成提示若因上下文超限再删更早事件，分配仍使用协议窗口 8，不跟运行时截断耦合。

### 4.4 两个优先级

V9.9 只用当前质量、来时路回撤和算子尝试次数构造优先级：

$$
S_R(a)=Q(a)+\lambda_U U_R(a)+C_R(a),
$$

$$
S_E(a)=Q(a)+\lambda_U U_E(a),
$$

$$
\lambda_U=0.25.
$$

当前已实现的改善由 $Q(a)$ 表达，不再增加独立的正向趋势、历史平均收益或推进率奖励。

## 5. 联合预算分配

### 5.1 条件权重与基础算子倾向

Refine 与 Explore 保留 `0.7/0.3` 的基础倾向，并由锚点状态动态修正：

$$
\bar W_R(a)
=
0.7\exp\left(\frac{S_R(a)}{T}\right),
$$

$$
\bar W_E(a)
=
0.3\exp\left(\frac{S_E(a)}{T}\right),
$$

$$
T=0.25.
$$

定义锚点总权重：

$$
A(a)=\bar W_R(a)+\bar W_E(a).
$$

基础比例规定两个优先级相同时的算子倾向。由于 $Q$ 在锚点内抵消，

$$
S_E(a)-S_R(a)=0.25\bigl(U_E(a)-U_R(a)\bigr)-C_R(a)\le 0.25.
$$

$T=0.25$ 时，Explore 成为该锚点首选算子需要

$$
S_E-S_R>0.25\log\frac{0.7}{0.3}\approx0.212.
$$

因此默认仍是 Refine 主导；当 Refine 已大量尝试、Explore 几乎未试、且没有回撤宽限时，Explore 可以略高于 $50\%$，理论上限约 $53.8\%$。$C_R>0$ 时 Explore 份额被进一步压低。实际运行比例由锚点状态决定，但不在线学习算子回报。

### 5.2 几何秩锚点分配

将全部锚点按 $A(a)$ 从高到低排列。第 $j$ 个位置的几何权重为：

$$
g_j=2^{-(j-1)/h_A},
\qquad
h_A=5.
$$

若多个锚点的 $A(a)$ 完全相同，它们共同占据连续位置。记这组位置为 $B(a)$，组内每个锚点取得该位置段总权重的平均值：

$$
\widetilde\mu(a)
=
\frac{1}{|B(a)|}
\sum_{j\in B(a)}g_j.
$$

锚点分配概率为：

$$
\mu(a\mid\mathcal H)
=
\frac{\widetilde\mu(a)}
{\sum_b\widetilde\mu(b)}.
$$

当锚点足够多且没有大规模平局时，排名前 10 的锚点合计获得约 $75\%$ 的概率，前 20 个获得约 $93.75\%$。每个锚点始终保留非零概率；低排名锚点继续增加时，几何长尾的总质量有界。

该规则不设置 Top-K 硬门槛，也不额外混入全局均匀概率。它同时避免硬截断造成永久淘汰，以及全量 softmax 中大量普通锚点凭数量稀释预算。

### 5.3 锚点内算子选择

选中锚点后：

$$
\pi(R\mid a,\mathcal H)
=
\frac{\bar W_R(a)}
{\bar W_R(a)+\bar W_E(a)},
$$

$$
\pi(E\mid a,\mathcal H)
=
\frac{\bar W_E(a)}
{\bar W_R(a)+\bar W_E(a)}.
$$

最终联合概率为：

$$
P(a,o\mid\mathcal H)
=
\mu(a\mid\mathcal H)\,
\pi(o\mid a,\mathcal H).
$$

$Q(a)$ 同时进入两个优先级，因此在同一锚点的算子概率中抵消。这是有意分工：当前质量决定是否值得从这里继续，来时路回撤与算子投入决定选中后怎样改。V9.9 不预设高质量锚点必然 Refine，也不预设低质量锚点必然 Explore。

## 6. 生成上下文

Refine 与 Explore 共享同一套轨迹上下文：

$$
\operatorname{Ctx}(a,o)
=
\{\operatorname{Task},
\operatorname{Intent}(o),
q(a),
\operatorname{Code}(a),
\operatorname{ParentPath}_8(a)\}.
$$

上下文包含任务定义与接口约束、当前真实适应度、当前完整代码，以及父链上最近 8 个形成事件。形成事件按真实顺序展示 operator、Idea、实际代码变化、结果和父子适应度。

从当前锚点发起的已有子代尝试不进入提示。上下文超限时从最早形成事件开始删除；任务约束、算子意图、当前适应度和当前完整代码始终保留。

提示不加入其他锚点代码、全局 Idea Bank、hypothesis 清单、静态簇标签或模型生成的全局总结。

## 7. 生成意图与输出

Refine 的语义是发展当前算法方向：

> Develop the current algorithmic direction. Preserve its central design principle and make one focused change that improves, completes, or repairs its implementation, using the recorded formation path as evidence.

Explore 的语义是提出替代算法方向：

> Propose one coherent alternative algorithmic direction. Change the central decision principle rather than tuning parameters or adding cosmetic complexity. Return one complete valid implementation that later steps could refine.

每次模型调用只输出一个简短 Idea 和一份完整程序：

````text
Idea: <one short statement of the implemented mechanism>
Code:
```python
<one complete executable implementation>
```
````

实际代码决定新程序、重复关系和后续轨迹事实；Idea 只作为当时声明保存。

## 8. 原子更新

每个响应按以下顺序更新：

1. 重新计算当前全部程序的 $Q$、全部锚点的 $S_R/S_E$、条件权重与几何秩概率；
2. 概率化选择一个锚点；
3. 根据该锚点的条件权重概率化选择 Refine 或 Explore；
4. 构造当前完整代码与匹配 parent path，调用模型一次；
5. 获得模型响应后，增加起始锚点对应的 $n_R$ 或 $n_E$；
6. 对全局新程序进行真实评价；若程序已见则复用既有评价；
7. 按下述重复规则形成 child 时，创建一个以所选锚点为父节点的新锚点，记录形成事件，新锚点的两个算子计数均从 0 开始；
8. 保存事实并返回第一步。

无论本次生成采用 Refine 还是 Explore，形成的 child 都只是一个新锚点。V9.9 不创建 hypothesis，不给任何 child 预先承诺后续预算。

### 8.1 重复程序

全局新且有效的程序在真实评价后创建一个 child 锚点。全局已见程序若不是当前程序或其祖先，且当前父锚点尚未连接到该程序，则复用既有评价并创建一个新的 child 锚点；它与其他同代码锚点共享适应度，但拥有自己的形成事件、来时路和算子计数。

Invalid、no-op、祖先返回以及当前父锚点已经连接过的重复程序只记录响应，不创建锚点。Refine 与 Explore 使用相同的重复规则，因为 V9.9 不再把 Explore child 解释为新的 hypothesis 边界。

候选解析、代码缓存和异常恢复属于实现卫生；除非改变上述搜索行为，否则不构成独立机制。

## 9. 预算、停止与最终程序

正式预算为 1000 次真实 evaluator 调用，其中包含初始化 8 次评价。模型响应若没有启动新的真实评价，不消耗 evaluator 预算，但仍增加起始锚点的算子计数。

搜索不因长期没有改善而提前停止。评价预算耗尽后，从全部已评价的唯一程序中按真实任务目标选择全局最好程序。锚点优先级、算子概率、访问次数和形成深度都不参与最终排序。

## 10. 完整算法

```text
Input: task, evaluator, LLM, evaluator budget B = 1000

Independently generate and evaluate 8 valid code-unique roots.
Keep all 8 as root anchors.
Initialize every root with an empty path and zero Refine/Explore counts.

While evaluator budget remains:
    Recompute current midrank quality Q for every unique program.

    For every anchor:
        Compute operator-specific under-exposure U_R and U_E.
        Compute distance-decayed Refine grace C_R from the recent 8 ancestors.
        Compute S_R and S_E.
        Compute conditional weights W_R and W_E and total weight A.

    Rank all anchors by A.
    Convert ranks to a geometric anchor distribution with rank half-life 5.
    Sample one anchor.
    Sample Refine or Explore from that anchor's conditional weights.

    Build task + intent + current fitness + current code + parent-path context.
    Generate one Idea and one complete program.
    Increment the selected anchor's corresponding operator count.
    Evaluate globally new code, or reuse a cached evaluation for seen code.
    Create a child only when allowed by the duplicate rule.
    Append the resulting response and formation facts.

Return the globally best evaluated unique program by the true objective.
```

## 11. 固定参数

- 独立根数：8，全部保留；
- 真实评价预算：1000；
- parent path 与回撤宽限窗口：最近 8 个形成事件 / 祖先；
- 算子尝试不足权重：$\lambda_U=0.25$；
- 来时路距离半衰期：$h=4$；
- Refine/Explore 基础倾向：`0.7/0.3`；
- 条件权重温度：$T=0.25$；
- 锚点排名半衰期：$h_A=5$。

## 12. 实现不变量

1. 一份在线决策恰好选择一个锚点、一个算子并启动一次模型响应。
2. 每个响应完成后重新计算全部在线质量、优先级和概率，不锁定未来预算。
3. 在线评分只读取当前质量、匹配来时路和当前锚点的算子响应次数。
4. 已有子代结果、后代成功、route、hypothesis、算法簇与程序复杂度不进入评分。
5. 所有锚点在几何秩分配中保持非零概率；不得用 Top-K 硬截断替代。
6. 来时路影响按形成代数衰减，并且只读取最近 8 个祖先；全局等待时间不改变父代关系或算子计数。
7. 当前质量同时进入两个算子优先级，不直接规定算子类型。
8. Refine 与 Explore 使用相同的 parent-path 上下文，只改变生成意图。
9. 根没有来时路，不执行强制 bootstrap。
10. 最终程序只按真实任务目标选择。

## 13. 与 V9.8 的最小差异

- 在线投资单位从 hypothesis 与锚点两层收缩为锚点；
- 联合决策从“先固定抽取 operator，再选 hypothesis/anchor”改为“先选锚点，再按锚点状态选 operator”；
- 删除 hypothesis boundary、hypothesis 前沿、历史平均发展收益 $M$ 与 hypothesis 计数；
- 跨边界宽限改为锚点来时路上按代数距离与 Refine 次数共同衰减的回撤宽限；
- 锚点质量从原始 $q$ 改为当前唯一程序的中秩百分位；
- 锚点分配从 argmax 改为全量几何秩概率，不设置 Top-K；
- 初始化从 8 roots 加逐根 bootstrap 改为 8 个独立根全部保留，不执行 bootstrap，也不做初始化质量筛选；
- Refine/Explore 基础倾向仍为 `0.7/0.3`，$T=0.25$ 使 Explore 仅在明显欠尝试且无回撤宽限时可以略高于 $50\%$；
- 回撤宽限只扫描与 parent path 相同的最近 8 个祖先，$h=4$ 保持不变；
- parent path 与单次 `Idea + Code` 生成协议保持不变；Refine 与 Explore 继续共享同一轨迹上下文。

## 14. 首轮过程记录，不进入在线决策

同代码经不同形成路径到达时，仍作为不同锚点竞争。几何秩 $h_A=5$ 保持不变。二者都不在首轮改公式。每次选择必须记录：

- 所选锚点的程序副本数 $m(x)=|\{a:x(a)=x\}|$，以及该程序的合计 $\mu$；
- top-5 / top-10 / top-20 的 $\mu$ 合计，以及 top-10 中的唯一程序数；
- 副本数大于 1 的程序所占据的总 $\mu$；
- 选择熵、所选锚点秩、$P(E\mid a)$ 与 $C_R(a)$。

若 top-10 长期只有很少的唯一程序，再考虑把 $\mu(a)$ 按 $m(x(a))$ 摊还；现在不加这条修正。$h_A$ 是否过激进，也只根据实际集中度与被救回路线的最低秩事后判断。

V9.9 是新的联合协议。其未来完整结果只能评价整体搜索行为；在没有单因素对照时，不能把性能变化归因于任一分数项、初始化或概率化规则。

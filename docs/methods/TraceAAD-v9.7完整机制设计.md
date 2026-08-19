# TraceAAD V9.7：改进轨迹引导的自动算法设计

本文定义 TraceAAD V9.7 的搜索表示、预算分配、父代来时路上下文与生成协议。过程问题按层次见 [research 索引](../research/README.md)。

> 版本边界：正统 V9.7 是仅父代来时路的完整协议，进入[实验总汇](../experiments/实验总汇.md)第 4.1–4.2 节；V9.7-batch 使用 V9.6 的 `formation + direct attempts` 历史上下文，结果在各任务页。

## 1. 设计立场

自动算法设计在有限真实评价预算下反复修改已有程序。系统选择一个已有算法状态，由语言模型提出修改，执行完整程序，再由评价器给出结果。最终程序只是这条过程的一个截面。单独看它，无法说明算法怎样形成、之前做过什么、每次尝试的效果如何。

TraceAAD 把这段改进轨迹作为下一步搜索决策与生成指导的主要信息来源。每消耗一份评价预算，方法回答三个问题：从哪里继续，当前算法怎样走到这里，以及这一次怎么改。分配决定从哪里生成，来时路提供已有改进信息，生成意图规定本轮修改方向与结构变化程度；三者共同影响下一候选的生成。树结构、路线、锚点和生成意图都服务于这条轨迹。

完整循环如下。先选一条初始来源对应的路线，再在该路线内选一个锚点。构造该锚点的父代改进来时路，按固定比例抽取 Refine 或 Explore，生成一个 Idea 与一份完整程序。真实评价后把结果写回轨迹与访问统计，然后重新选择。

## 2. 搜索表示

搜索状态是由多条初始路线组成的森林。森林只保存已经发生的程序、生成关系和评价结果。

**程序**是评价器实际执行过的一份代码，带有真实适应度。搜索内部把适应度统一为越大越好的有向质量 $q$：最大化任务取 $q(p)=\operatorname{fitness}(p)$，最小化任务取 $q(p)=-\operatorname{fitness}(p)$。任务原生目标仍用于最终报告。

**锚点**是程序在一条具体形成路径中的位置。它绑定当前程序、唯一结构父节点、到达该节点的那次生成，以及从该位置已经发起过的生成次数 $n(a)$。锚点质量就是所绑定程序的有向质量。同一份代码若沿不同路径到达，对应不同锚点：它们共享程序事实，不共享来时路与访问计数。下一步生成的条件因此是当前代码加上该锚点的形成历史。

**路线**是同一初始根所衍生的全部锚点，$r=\{a:\operatorname{root}(a)=r_0\}$。它是生成拓扑上的来源单位。不同根不必对应不同算法思想，同一路线内部也可以换掉核心机制。路线级分配平衡的是不同初始来源获得的生成机会。对路线 $r$，当前最好质量

$$
q^*(r)=\max_{a\in r}q(a)
$$

表示这条路线已经达到过的质量区域。累计生成次数

$$
N_t(r)=\sum_{a\in r}n_t(a)
$$

计量该来源已经获得多少次生成机会，不是节点数，也不是评价器调用次数。

一次生成记录起始锚点、本轮意图、声明的 Idea、实际代码变化、质量变化，以及 `improve`、`plateau`、`regress` 或 `invalid` 结局。只有完成后的事实才进入后续上下文。

## 3. 初始化

搜索开始时建立 $K=8$ 条独立初始路线。$K$ 是协议常数。每条路线先独立生成一份有效且代码互不相同的根程序，不读取已有搜索历史。代码互异是很弱的条件：两份根可以只是实现细节不同。路线分配默认这些根提供了一些值得分别投入的起点，在线规则并不额外要求思想差异。

每个根随后恰好接受一次 Refine 生成。这次 bootstrap 给该路线建立第一条可观察的形成事件，并为后面的乐观项提供任务内的一步变化尺度。

对 bootstrap 中成功形成新子节点的转换，取父子有向质量差的绝对值。改善、退步和持平都进入集合 $D_{\mathrm{init}}$；无效、未形成新节点的重复或空操作不进入。乐观尺度

$$
s=
\begin{cases}
\operatorname{median}(D_{\mathrm{init}}), & D_{\mathrm{init}}\neq\varnothing,\\
0, & D_{\mathrm{init}}=\varnothing.
\end{cases}
$$

正式搜索开始后 $s$ 固定，后续结果不再重估。它是从 Refine 一步变化估得的启发式尺度，同时用于路线层和锚点层，不是置信区间，也不是路线潜力的估计。

## 4. 预算分配

从哪里继续拆成两步：先决定下一份生成机会交给哪条路线，再决定从该路线的哪个锚点出发。路线层用于调节不同初始来源之间的长期预算集中；锚点层用于调节同一来源内部不同状态的访问。两层使用同形的质量加乐观项，质量对象和计数不同。

路线分数为

$$
S_t^{\mathrm{route}}(r)=q^*(r)+\frac{s}{\sqrt{N_t(r)+1}}.
$$

选择 $r_t^*=\arg\max_r S_t^{\mathrm{route}}(r)$。当前最好结果提供利用依据；该来源获得的生成机会越少，额外项越大。

在选中路线内，锚点分数为

$$
S_t^{\mathrm{anchor}}(a)=q(a)+\frac{s}{\sqrt{n_t(a)+1}}.
$$

选择 $a_t^*=\arg\max_{a\in r_t^*} S_t^{\mathrm{anchor}}(a)$。新子节点 $n_t(a)=0$，初始乐观项为 $s$，仍须与路线内其他锚点比较，没有强制扩展资格。

分数相同时，优先访问次数更少、创建更早的对象，使同分分配确定。

当轮只在选中路线内竞争。一个高质量锚点若所属路线本轮落选，当轮没有机会。路线分数把当前最好质量与欠投入补偿写在一起，不另行估计未来突破潜力。

## 5. 生成上下文

选定锚点后，模型看到当前完整算法，以及从该路线根到该锚点的父代改进来时路。来时路说明当前算法怎样一步步形成。从当前锚点出发已经做过的子代尝试仍作为搜索事实记录，不进入提示。

匹配单位是代码加具体锚点。同一份代码由不同状态产生时，不混合它们的形成历史。来时路沿父链回溯，保留最靠近当前锚点的形成步骤，至多 8 条；不足则全部保留。事件按真实发生顺序排列。该规则只决定哪些事件进入提示。上下文不足时从最早事件开始丢弃，当前代码与任务说明始终保留。

每条事件写出当时声明的 Idea、由父子实际代码得到的紧凑修改摘要（增删行数，以及两侧各至多两行实际改动）、`improve` / `regress` / `plateau` 结局，以及父子真实分数：

```text
[History i] Formation step
Idea: ...
Change: ...
Result: improve | regress | plateau
Fitness: parent -> child
```

Idea 是当时的声明。修改与结果共同构成可审计事实；方法不从多行耦合修改中识别某一行的单独因果贡献。历史提供已经发生的形成事实。生成意图规定本轮修改方向与结构变化程度，预算落点由分配承担。提示中也不包含其他路线的代码或模型生成的全局总结。

## 6. 生成意图与候选

锚点确定后，生成意图规定本轮修改方向与结构变化程度。两种意图共享任务说明、当前完整代码、同一条来时路和同一输出契约，差异只在指令。

Refine 要求沿当前设计方向继续发展：

> Continue improving the current algorithm within its existing design. Make one focused modification based on the current algorithm and its improvement history.

Explore 要求寻找实质不同的设计方向，可以替换或重组当前设计的重要部分：

> Seek a materially different way to improve the current algorithm. Do not merely tune parameters or make a small local modification. You may replace or substantially restructure an important part of the current design.

固定混合为 $P(\mathrm{Refine})=0.7$、$P(\mathrm{Explore})=0.3$。该比例是协议常数。意图不依据单步持平、成熟度或历史信用切换。抽取由搜索种子与迭代序号确定性映射，保证恢复一致。初始化阶段的 bootstrap 固定为 Refine。

每次分配只生成一个候选。一次模型调用输出可选的短 Idea 与一份完整可执行程序，处理后立即重新选择。完整程序是有效性的硬条件；Idea 缺失不使候选无效。系统从父子实际代码推导修改摘要。

有效候选按有向质量分为 `improve`、`plateau`、`regress`；无法形成可执行程序或评价无效记为 `invalid`。这些标签用于事实记录和历史展示，不形成额外奖励，也不沿祖先回传。

若候选代码与当前锚点相同，或回到祖先程序，则记录该事件但不创建新状态。若同一程序已在其他路线出现，复用其评价结果，并可以在当前历史上创建新锚点。因此程序数量与锚点数量不必相等。

## 7. 更新、停止与最终程序

正式比较使用每次运行 1000 次真实评价。新程序消耗一次评价；已评价过的相同程序复用结果。停止条件是评价预算耗尽，不因连续无改善提前结束。

锚点访问次数在完成一次模型响应后加一。无效、空操作和重复生成仍计为一次生成机会。路线累计次数为路线内各锚点访问次数之和。生成机会与评价次数分开计量：持续产生无效或重复的路线仍被视为已经投入。

一次候选完成后，森林保存该次生成的意图、Idea、修改与结局。一次生成结果通过两类状态更新影响后续搜索：新 child 扩展可用来时路；完成的生成响应更新锚点与路线访问计数。前者作用于生成上下文，后者作用于预算分配。

搜索结束后，在全部唯一程序中按有向质量取最好者；质量完全相同时偏好更短、更早发现的程序。路线分数、访问次数和生成意图不参与最终排序。

## 8. 算法

```text
Input: task, evaluator, LLM, real evaluator budget B = 1000

Generate K = 8 unique valid roots; create one root anchor each.
For each root, generate one bootstrap candidate with Refine.
Set s = median |q(child) - q(root)| over valid bootstrap transitions
    (s = 0 if none).

While evaluator budget remains:
    Score each route:  q*(r) + s / sqrt(N(r) + 1)
    Select the highest-scoring route r.
    Score each anchor in r:  q(a) + s / sqrt(n(a) + 1)
    Select the highest-scoring anchor a.

    Build the parent path from the route root to a
        (most recent formation steps, at most 8).
    Draw intent: Refine with probability 0.7, else Explore.
    Generate one optional Idea and one complete program.
    Increment n(a).
    Evaluate if the program is new; otherwise reuse the cached result.
    Record the attempt; create a child anchor only for a valid new relation.
    Update route statistics and reselect.

Return the best unique program by true objective.
```

## 9. 解释边界

当前 V9.7 是联合协议：两级预算分配、父代改进来时路、固定 Refine/Explore 混合、初始化与更新规则共同决定完整搜索行为。四任务三重复的正式 held-out 已齐并进入代表性同场。相对 V9.7-batch 的差为 TSP 0.08%/0.08%/0.56%、CVRP −2.0%/−2.8%/−2.9%、OP +0.03%/−0.65%/−2.6%、OBP 六档均在 1% 以内，不能识别父代来时路在完整搜索中的净贡献。V9.7-batch 只能评价旧历史上下文下的联合系统；固定锚点实验只识别历史上下文的单步作用。

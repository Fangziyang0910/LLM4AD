# TraceAAD V9.14：单根算法树上的轨迹条件进化

> V9.14 以 [V9.7](TraceAAD-v9.7完整机制设计.md) 为直接基线。两条机制主线——轨迹条件的单步生成与轨迹感知的两级分配——及其全部超参数原样保留：8 个初始来源、Refine 0.7 / Explore 0.3 意图、8 条父代来时路与一步尺度 $s$。变化集中在搜索状态的表示与评价预算的口径：森林的双层实体统一为一棵带虚拟根的单父纯算法树（无边实体），代码哈希去重与评价缓存移除，每个生成的候选都真实评价。V9.14 因此是 V9.7 机制主干的精简重写，定位为后续机制实验的干净基线；树形表示回到 [V9-Core](TraceAAD-v9-core完整机制设计.md) 的传统，分配与生成机制采用 V9.7 的版本。

## 1. 演变总览

| 对象 | V9.7 | V9.14 |
| --- | --- | --- |
| 搜索状态 | 森林：程序 + 锚点双层实体 | 单根树：纯算法节点单一实体 |
| 树的形状 | 每个根一棵形成树，根之间互不连接 | 虚拟根连接 8 个初始算法，全树单父 |
| 重复代码 | 哈希去重，跨路径复用评价 | 生成即评价，重复代码形成新节点 |
| 尝试与失败 | 7 类尝试边，失败尝试入库 | 树上只存有效节点，失败尝试只进运行日志 |
| 分配公式 | $q+s/\sqrt{n+1}$ 两级（路线 → 锚点） | 同一公式两级（分支 → 算法节点） |
| 生成机制 | 来时路 + Refine/Explore + Idea/Code 契约 | 不变 |

V9.7 用程序、锚点、尝试三层实体描述搜索状态，并以代码哈希支撑两项设计：同一代码沿不同路径到达时形成不同锚点，相同代码再次出现时复用评价结果。这两项设计都不是 V9.7 检验的研究对象，却带来持续的状态口径负担：程序数、锚点数与尝试数各自回答不同问题，预算消耗随缓存命中变得非线性，重复与缓存相关的尝试需要五类标签分别记录。

V9.14 的判断是：这部分复杂度属于表示层，移除后机制主干的因果链更短。每个节点是一次成功生成事件的直接产物，自身承载生成它的演化元数据；来时路成为节点的父链属性，每次成功响应恰好消耗一次真实评价。

## 2. 搜索状态：带虚拟根的单父纯算法树

搜索开始时建立虚拟根（ID=0），其下生成 $K=8$ 个初始算法，每个初始算法是一条 Level-1 分支的根；正式搜索不再新增分支。算法 $a$ 所属分支 $b(a)$ 由其祖先链上的第一个真实祖先给出，分支因此是子树而非额外维护的对象。

节点保存机制所需的全部事实：完整代码、真实 fitness、有向质量、父节点引用 `parent_id`、被选为父代的累计次数 $c(a)$，以及生成该节点时的演化事实：生成意图（Refine / Explore）、模型声明的 Idea、父子实际代码差异（diff 及增删行数）、质量变化量 $\mathrm{d}q$、定性结果（improve / plateau / regress）。自增 ID 记录创建次序，用于平局判定。有向质量沿用 V9.7 的标准化：maximize 任务取 $q(a)=\mathcal E(P_a)$，minimize 任务取 $q(a)=-\mathcal E(P_a)$。

每个节点由一次成功评价的生成事件创建：同一代码重复生成即为新节点，代码不再全局唯一。解析失败或评价失败的生成不创建节点，父节点访问计数 $c(a)$ 照常递增，失败事实直接记录至运行日志（`evaluations.csv` 与 `errors.jsonl`）。

节点的形成路径即其祖先链上的算法节点序列：

$$
\tau(a)=(a_1,a_2,\ldots,a_k=a).
$$

## 3. 生成即评价：线性的预算口径

每次 LLM 响应解析为一份完整可执行程序后，立即由 evaluator 真实运行并计入评价预算；不做任何形式的代码查重或结果复用。预算口径由此完全线性：

- 每次成功解析的候选恰好消耗一次真实评价，无论代码是否与已有节点重复；
- 重复代码成为树上独立的新节点，可在同一分支内或不同分支间多次出现；
- 初始化阶段重复的根代码同样成立新分支；V9.7 中重复根被拒绝并重新生成；
- 解析失败的响应不消耗评价预算；评价运行失败（返回非有限值）按一次真实评价计入，与 V9.7 一致。

正式预算为每次运行 $B=1000$ 次真实评价，评价预算与 LLM 调用数分开记录。

## 4. 两级分配：在单根树上的重述

分配机制 $\mu(a_t\mid\mathcal H_t)$ 决定下一次生成从哪个算法出发。先选分支，再在分支内选算法：

$$
S_t^{\mathrm{branch}}(b)=q_t^*(b)+\frac{s}{\sqrt{N_t(b)+1}},
\qquad
S_t^{\mathrm{algo}}(a)=q(a)+\frac{s}{\sqrt{c_t(a)+1}}.
$$

其中：

- $q_t^*(b)=\max_{a\in b}q(a)$ 是分支已经达到过的最好质量；
- $N_t(b)=\sum_{a\in b}c_t(a)$ 是分支累计获得的生成机会；
- $q(a)$ 是算法自身质量，$c_t(a)$ 是它被选为父代的次数；
- $s$ 是初始化阶段估计的一步变化尺度，正式搜索开始后不再重估。

公式、两级语义与 V9.7 逐字相同：第一级承担来源之间的深度—广度权衡，第二级承担分支内部的状态回访与局部开发；同分时优先访问更少、创建更早的对象。对应关系为路线 ↔ 分支、锚点 ↔ 算法节点。分支保存的是 provenance，不自动等同于算法簇；同一分支内部仍可能发生重要的算法机制迁移。

实质差别只在状态本体。锚点是"程序在某条路径上的位置"，同一程序可形成多个锚点并跨路线复用评价；算法节点是一次生成事件的产物，代码与位置在同一实体上。分配看到的回访计数因此直接就是"从这份具体代码重新出发"的次数。

初始化与尺度沿用 V9.7：8 个初始算法各接受一次 Refine bootstrap，对完成评价的 bootstrap 转换取父子有向质量差的绝对值，改善、退步与持平都计入：

$$
s=\operatorname{median}(D_{\mathrm{init}}),
$$

$D_{\mathrm{init}}$ 为空时取 $s=0$。$s$ 是任务内一步修改幅度的启发式尺度，解释与 V9.7 相同。

## 5. 轨迹条件的单步生成

生成机制完全沿用 V9.7，条件分布为

$$
P(x_{t+1}\mid x_t,h_t,o_t),
$$

其中 $x_t$ 是选中算法的完整代码，$h_t$ 是其父链上最近 8 个祖先节点的形成元数据，$o_t$ 是本轮意图。每条事件包含当时的 Idea、实际代码 Change、定性 Result 与 Fitness 变化，内容与渲染格式与 V9.7 相同。Refine 以 0.7 概率抽取，聚焦当前设计方向的局部修改；Explore 以 0.3 概率抽取，寻求结构性不同的改进方向。响应契约为可选短 Idea 与一份完整程序。机制的完整论证见 [V9.7 规范](TraceAAD-v9.7完整机制设计.md)第 4 节。

来时路在单根树上取得直接定义：选中节点的父链即来时路，形成路径不再经过额外的索引表。

## 6. 完整运行协议

````text
Input: task, evaluator, LLM, real evaluator budget B = 1000

Generate K = 8 valid initial algorithms under the virtual root;
    each becomes the root of one Level-1 branch.
For each initial algorithm, generate one bootstrap candidate with Refine.
Set s = median |q(child) - q(parent)| over evaluated bootstrap transitions
    (s = 0 if none).

While evaluator budget remains:
    Score every branch by q*(branch) + s / sqrt(N(branch) + 1).
    Select the highest-scoring branch.
    Within that branch, score every algorithm by
        q(algo) + s / sqrt(count(algo) + 1); select the highest.
    Build the selected algorithm's parent improvement path, at most 8 ancestor events.
    Draw Refine with probability 0.7, otherwise draw Explore.
    Generate one optional Idea and one complete program.
    Increment the selected algorithm's count.
    Evaluate the generated program for real; no dedup, no reuse.
    If evaluation succeeds:
        create and insert a child Algorithm node into the tree.
    Update facts and reselect.

Return the best algorithm by the true objective.
````

停止条件是评价预算耗尽，不因连续无改善提前停止。最终排序只看真实质量，平局取代码更短者、再取创建更早者；分支、访问次数与生成意图不参与最终排序。

## 7. 对照口径与研究定位

与 V9.7 对照时，两处口径差需要在读取数字前明确：

1. **预算语义。** V9.7 中重复代码复用缓存结果、不消耗评价预算；V9.14 每个候选真实评价。同样 1000 次预算下，V9.14 花在互异代码上的评价数随模型重复率下降，任务间重复率差异会直接进入两版本差距 [待验证]。
2. **状态计数。** V9.7 分别报告程序数、锚点数与尝试数；V9.14 只有算法节点数，成功评价数与新增节点数一一相等。

分配公式、意图比例、来时路内容与长度、初始化与 $s$ 估计、停止条件与最终排序完全沿用 V9.7。两版本的搜索行为差异因此集中来源于预算口径与状态表示，而不来自任何分配或生成机制的改动。

> **V9.14 的研究定位：在统一的单父纯算法树上，以改进来时路为条件的单步生成与两级 UCB 分配构成 V9.7 机制主干的线性预算重写；它检验该主干在去掉双层实体与评价缓存后是否保持同样的搜索质量，并作为后续机制实验的干净基线。**

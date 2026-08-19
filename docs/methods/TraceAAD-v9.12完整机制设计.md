# TraceAAD V9.12：进展条件的精炼与探索

> V9.12 以 [V9.11](TraceAAD-v9.11完整机制设计.md) 为直接基线。它保留 V9.11 的轨迹、路线—锚点选择、Refine / Explore 生成和一次探索后续改进，只把“何时使用 Explore”从固定停滞触发改为基于当前这段轨迹进展的概率选择。设计依据是 [研究认识](../knowledge/研究认识.md)、[V9.8 机制诊断](../analysis/TraceAAD-V9.8机制诊断.md) 和 [V9.11 机制诊断](../analysis/TraceAAD-V9.11机制诊断.md)。

## 1. 核心判断

LLM 在自动算法设计中通常更擅长沿着一个已经可行的方向做局部改进。结构性探索不是每隔固定次数都应该发生的动作：一个方向正在持续产生改进时，继续 Refine 更符合有限预算的目标；沿当前方向继续 Refine 已经多次不能改善时，才逐渐提高 Explore 的概率。

V9.12 因此只引入一个主要机制：

**根据当前这段来时路上的 Refine 进展，动态调整 Refine / Explore 的选择概率。**

搜索节律由证据自然形成：

```text
当前方向缺少失败积累，或最近 Refine 仍在改善
    -> 主要采用 Refine

最近 Refine 多次不能改善
    -> 提高 Explore 概率

Explore 形成有效新程序
    -> 紧邻再给一次 Refine，避免第一版改写立即失去机会

后续
    -> 回到普通概率选择
```

V9.12 不设“连续 H 次没有全局突破就必须 Explore”的硬规则。全局最好没有变化只是一个结果，不足以单独决定当前方向已经成熟。

## 2. 直觉与反直觉

直觉是：仍然有效的改进方向值得继续投入。

反直觉是：探索不一定发生在全局停滞之后，而应由当前方向的局部改进是否仍然有效决定；全局最好暂时没有变化时，只要当前方向仍能产生局部改善，就不应强行打断它。

V9.12 也不把前期、中期、后期写成固定阶段。八个独立根已经提供了初始方向覆盖，之后的集中和重新开放由实际形成过程决定。

## 3. 研究对象

### 3.1 轨迹条件生成

给定锚点 $a$ 的当前程序 $x(a)$、匹配的父代形成路径 $h(a)$ 和生成意图 $o$，模型生成一个完整候选：

$$
x_{t+1}
\sim
P(\cdot\mid x(a_t),h(a_t),o_t).
$$

Refine 负责延续和修正当前方向；Explore 负责改变核心决策原则或主要搜索结构。两种意图都继续读取当前锚点的父代来时路。V9.12 不采用“Explore 隐藏历史”的上下文改动，因为现有固定锚点结果没有证明父代来时路稳定伤害 Explore。

### 3.2 轨迹感知的计算分配

每份预算仍然回答两个问题：从哪个形成路径中的锚点继续，以及在该锚点采用哪一种生成意图。

路线仍只表示根来源，锚点仍表示程序与形成路径的联合状态。路线和锚点不是在线算法簇标签；算法族和簇结构只作为离线解释对象，不进入 V9.12 控制。

## 4. 初始化与普通选择

初始化和 V9.11 相同：

1. 独立生成 8 个代码互异且有效的根；
2. 每个根进行一次 Refine bootstrap；
3. 用有效 bootstrap 的一步有向质量变化绝对值中位数估计共享尺度 $s_0$；
4. 根和 bootstrap 的真实 evaluator 调用计入 1000 次预算。

没有待处理探索后续改进时，路线与锚点选择保持 V9.11：

$$
S_t^{route}(r)
=
q_t^*(r)
+
\frac{s_0}{\sqrt{N_t(r)+1}},
$$

$$
S_t^{anchor}(a)
=
q(x(a))
+
\frac{s_0}{\sqrt{n_t(a)+1}}.
$$

选择分数只表达当前质量和已获得的机会。V9.12 不在路线或锚点选择中加入趋势、算法簇、长期信用或后验。

这里也划清两个决策的职责：路线—锚点选择已经回答“哪个程序值得获得下一次预算”，算子选择只回答“对这个程序继续 Refine，还是尝试改变方向”。因此 V9.12 不再把当前程序的全局质量排名重复写入算子概率。否则，同一程序即使自身没有变化，也会因为后来产生了更多低质量程序而被动提高排名并改变 Explore 概率。

### 4.1 代表性工作的机制取舍

代表性方法提供了四种不同处理方式。EoH 用固定的探索与开发算子组合；HiFo-Prompt 根据全局停滞和文本多样性阈值在离散模式间切换；A2DEPT 为每个节点维护由父子结果更新的算子权重，并以 softmax 保留所有算子的非零概率；BaSE 在线读取不同轨迹的进展，把预算转向仍能改善的轨迹，但不改变轨迹内部的局部生成规则。

V9.12 只吸收其中共同而朴素的原则：算子不应固定分配，最近进展应改变下一步概率，同时任何算子都不应被永久关闭。它不采用全局阈值硬切换，不累计带幅度的长期权重，也不引入第二套 bandit。TraceAAD 已经有路线—锚点分配，当前缺口只是让生成意图随所选轨迹的真实进展变化。

## 5. 进展条件的算子概率

### 5.1 当前这段来时路

一次有效 Explore child 表示核心决策原则或主要结构发生了有意改变。Explore 之前的 Refine 成败描述旧方向，不能继续决定新方向的算子概率。因此，从当前锚点沿父链向上找到最近一次由 Explore 形成的锚点；该锚点至当前锚点构成当前这段来时路。若父链中没有 Explore，则从根开始。

这只是按真实生成意图划分进展统计的作用范围，不是在线算法簇识别。Explore 没有形成有效 child 时也不会开启新段。

### 5.2 Refine 是否仍然有效

父子形成边只保存成功形成的程序，不能完整表达 Refine 是否仍然有效。V9.12 因此读取完整响应事实：从当前这段来时路中的锚点发出的 Refine 响应中，按发生时间取最近 $L=8$ 次。探索后的固定一次 Refine 是新方向的第一条真实进展观测，同样进入统计；旧方向的响应已经由 Explore 边界自然隔开，不需要再作特殊排除。

对其中第 $i$ 次 Refine 响应定义：

$$
y_i=
\begin{cases}
1, & \text{the response forms a child with } q_{child}>q_{start},\\
0, & \text{otherwise}.
\end{cases}
$$

invalid、no-op、重复、祖先返回、评价失败、持平和退步都取 $y_i=0$；没有获得完整模型响应的传输失败不进入统计。

定义 Refine 失败证据：

$$
F_t(a)
=
\frac{1}{L}
\sum_{i=1}^{k(a)}(1-y_i),
\qquad
L=8,
\qquad
0\le k(a)\le L.
$$

未出现的观测不补作失败。因而一次失败只贡献 $1/8$，连续获得足够多的不改善响应后 $F_t(a)$ 才逐渐接近 1；最近 Refine 重新产生改善时，旧失败会随着滚动窗口移出。

这个信号只回答“沿当前这段来时路继续 Refine 最近是否有效”，不估计算法方向的价值、未来收益或长期潜力。窗口长度与默认展示的 8 条形成历史一致，不引入第二个时间尺度。

### 5.3 唯一的概率规则

V9.12 使用固定的稀疏探索下限和有限的探索上限：

$$
p_E(a)
=
p_{min}
+
(p_{max}-p_{min})F_t(a),
$$

$$
p_R(a)=1-p_E(a),
$$

其中协议取

$$
p_{min}=0.10,
\qquad
p_{max}=0.30.
$$

这条规则的行为含义是：

- 新方向没有 Refine 失败积累，Explore 保持 10% 的稀疏机会；
- 最近 Refine 仍在改善时，失败证据不增加，并随旧失败移出窗口而下降；
- 最近 8 次 Refine 都不能改善时，Explore 概率提高到 30%；
- Explore 永远不会变成确定性动作，Refine 也永远不会被完全删除。

每次选择锚点后，按照 $p_R(a)$ 和 $p_E(a)$ 独立抽取本轮生成意图。概率在每次响应完成后重新计算，不锁定未来的算子份额。

Explore 只有在至少还剩 2 次真实评价时才允许执行，因为一次有效 Explore child 必须保留兑现紧邻 Refine 的预算。预算末尾若抽到 Explore，则本轮改为 Refine；这只是保证原子循环完整，不改变正常搜索阶段的概率规则。

`0.10` 沿用 V9.11 持续停滞时约一成的稀疏 Explore 节律，`0.30` 不超过 V9.7 / V9.9 的基础 Explore 比例。二者是首轮协议边界，不是任务相关参数，也不是需要通过消融搜索的调参目标。即使局部 Refine 完全停止改善，Refine 仍然是普通决策中的多数算子。

## 6. 探索后的最小后续机会

如果 Explore 响应形成有效 child 锚点 $a_c$，下一次响应直接从 $a_c$ 采用 Refine，并读取包含这次实际变化的父代来时路：

$$
a_{t+1}=a_c,
\qquad
o_{t+1}=R.
$$

该后续机会只持续一次。完成后立即恢复普通路线—锚点选择和概率算子选择，无论后续程序改善、持平、退步、invalid、no-op 还是重复。

这不是固定周期中的“着陆阶段”，只是对一次结构性改写的最小形成容忍：第一版代码分数较低时，不立即把它当作没有价值；一次后续修正仍不能使它竞争时，也不继续保护。

若 Explore 没有形成有效 child，则不产生后续资格，下一轮直接进行普通概率选择。

## 7. 生成上下文与输出

Refine 与 Explore 都读取：

1. 任务定义和执行契约；
2. 当前程序的真实 fitness；
3. 当前完整代码；
4. 当前锚点最近 8 条父代形成事件；
5. 本轮生成意图和统一输出契约。

形成事件只记录真实发生的 `Intent + Idea + Compact Actual Change + Result`。从当前锚点发出的已有子代尝试、其他路线代码、全局总结和算法簇标签不进入默认提示。

Refine 要求模型延续当前核心决策原则，作一个聚焦的改进或修复。Explore 要求模型改变核心决策原则、搜索结构或主要信息利用方式，避免只作参数微调或装饰性改写。两者均只输出一个简短 Idea 和一份完整可执行 Code。

## 8. 完整原子循环

```text
Initialize 8 code-unique roots.
Refine each root once and estimate s0.
Set exploration_followup = null.

While real evaluator budget remains:
    If exploration_followup exists:
        Select exploration_followup.
        Set intent = Refine.
        Clear exploration_followup after this response.
    Else:
        Select one route by q_best + s0 / sqrt(N + 1).
        Select one anchor in that route by q + s0 / sqrt(n + 1).

        Find the current trajectory segment after the latest valid Explore child.
        Compute F(anchor) from up to 8 recent Refine responses
        started from anchors in this segment.
        Set p_explore = 0.10 + 0.20 * F(anchor).
        Sample Refine or Explore using p_explore.
        If fewer than 2 real evaluations remain, replace Explore with Refine.

    Build Task + Current Fitness + Current Code + Parent Improvement Path + intent.
    Generate one Idea + Code response.
    Parse, evaluate or reuse, and record all facts.

    If this was Explore and a valid child anchor was formed:
        Set exploration_followup to that child.

Return the globally best unique program by the true task objective.
```

传输失败没有完整响应时不增加机会计数；解析失败、invalid、no-op、重复和 evaluator 失败都作为本轮已发生的事实记录。只有有效 Explore child 才创建一次后续机会。

## 9. 机制边界

V9.12 明确不加入：

- 固定 `H=8` 的全局停滞触发；
- 固定的全局 Explore 比例；
- Thompson Sampling、Beta 后验、pending credit 或父链信用回传；
- 算法簇标签、代码聚类、embedding、全局 Idea Bank 或 judge；
- 任务特定的探索比例、质量阈值或后续长度；
- 多步强制 rollout；
- 对 Explore 使用另一套隐藏历史或失败摘要；
- 趋势、成熟度、路线推进率等额外在线分数。

路线—锚点选择仍是 V9.11 的实现骨架。V9.12 只研究一个问题：**当前这段轨迹的局部进展，能否比固定周期更自然地决定 Refine / Explore 的比例。**

## 10. 预期搜索节律

### 10.1 前期

8 个独立根提供初始方向覆盖。新根及其早期后代缺少 Refine 停滞证据，因此主要进行 Refine，同时保留少量 Explore 概率。

### 10.2 中期

某些方向如果持续改善，其父链上的 $F_t(a)$ 较低，系统会把更多响应用于 Refine。新方向必须逐步积累 Refine 失败，才会提高下一次 Explore 概率，不会因为全局暂时停滞而被强行打断。

### 10.3 后期

当前方向如果仍然产生 Refine 改进，继续发展；如果最近 Refine 大多不再改善，Explore 概率上升，系统重新尝试替代方向。一次有效 Explore 会开启新的进展统计，避免旧方向的失败迫使新方向连续探索。这个过程不依赖人工定义的前期、中期和后期边界。

## 11. 设计假设与不确定性

### 11.1 设计假设

1. 最近这段父链上的 Refine 成败可以粗粒度反映局部改进是否仍然有效。
2. 有效 Explore 改变了生成方向，因此应重新开始累计 Refine 进展。
3. 一次 Explore 后续 Refine 足以避免最明显的第一版代码早夭，同时不会形成长期预算承诺。

这些假设是机制设计依据，不是已被 V9.12 验证的结论。

### 11.2 可能的失败

- 当前方向仍有潜力，但连续随机失败使 $p_E$ 暂时升高；
- 任务评价噪声使短窗口内的 $F_t(a)$ 不稳定；
- 好方向需要多于一次后续 Refine，当前最小机会仍不足；
- 路线—锚点选择的集中使概率算子只能在少数来源上发挥作用；
- LLM 对某些任务的优势算法本来就很少提出，动态算子无法凭空创造这些方向。

这些边界说明 V9.12 试图修复的是“探索时机”，不声称同时解决提议分布、发展长度和路线表示的所有问题。

## 12. 首轮实验协议

V9.12 作为完整版本运行，不先展开概率边界、窗口长度、路线选择和后续机会的消融矩阵。开始正式批次前做小预算 smoke，确认：

1. 同一锚点在不同 Refine 进展状态下的 $p_E$ 能发生变化；
2. 新锚点没有历史时使用 $p_E=p_{min}$；
3. Explore 形成有效 child 后恰好获得一次后续 Refine；
4. Explore 前的 Refine 结果不进入新方向的进展统计，后续 Refine 则进入；
5. 后续响应完成后回到普通概率选择；
6. checkpoint 恢复不会改变父链进展统计或算子概率。

正式批次仍为四任务三重复、每次 1000 次真实 evaluator 调用。首轮过程分析只报告：

- $p_E$ 随局部 Refine 进展的分布；
- Refine / Explore 的实际份额及其任务差异；
- Explore child 与后续 Refine 的相对变化；
- 预算从仍能 Refine 的方向转向需要改变方向的过程证据。

最终性能仍以完整 held-out 结果为准。单次算子比例或某个局部统计不作为机制有效的充分证据。

## 13. 两句话方法说明

TraceAAD V9.12 不按固定周期切换精炼与探索，而是读取当前这段改进来时路上的 Refine 是否仍然有效：仍有收益的方向继续发展，连续不能改善时才逐渐提高替代探索概率。结构性 Explore 形成有效程序后开启新的进展统计并获得一次紧邻 Refine，随后重新回到普通竞争，从而在有限预算下连接局部精炼与必要的方向变化。
